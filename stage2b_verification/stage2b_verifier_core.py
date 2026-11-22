#!/usr/bin/env python3
"""Deterministic Stage 2B symbolic certificate verifier (SPEC v2.4 Section 12).

Zero-API-cost offline verifier that validates alignment, mechanism consistency,
source-to-concat equivalence claims, full-answer checks, and per-shard
Leave-One-Out mathematical witnesses using exact SymPy symbolic evaluation.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.logic.boolalg import BooleanFunction
from sympy.parsing.sympy_parser import (
    convert_xor,
    implicit_multiplication_application,
    parse_expr,
    standard_transformations,
)

SCRIPT_DIR = Path(__file__).resolve().parent
SHARDING_DIR = SCRIPT_DIR.parent / "colab_stage2_sharding"
if str(SHARDING_DIR) not in sys.path:
    sys.path.insert(0, str(SHARDING_DIR))

from audit_alignment import audit_shard_candidate  # noqa: E402

STAGE2B_RULE_VERSION = "stage2b-sympy-cert-v1.1.0"

FAIL_REASONS = {
    "FAIL_K_OUT_OF_RANGE",
    "FAIL_REDUNDANT_PREMISE",
    "FAIL_MECHANISM_MISMATCH",
    "FAIL_AMBIGUOUS_STEM",
    "FAIL_SOURCE_ANSWER_MISMATCH",
    "FAIL_MISSING_REFERENCE_ASSET",
    "FAIL_CERTIFICATE_UNAVAILABLE",
    "FAIL_SHARD_ALIGNMENT",
}

_TRANSFORMS = standard_transformations + (convert_xor,)
_TRANSFORMS_IMPLICIT = standard_transformations + (
    convert_xor,
    implicit_multiplication_application,
)


class IntervalMin(sp.Function):
    """Exact symbolic minimum over [lo, hi] (evaluates once only dummy var remains free)."""

    @classmethod
    def eval(cls, expr: Any, var: Any, lo: Any, hi: Any) -> Any:
        if isinstance(var, sp.Symbol) and expr.free_symbols <= {var} and not lo.free_symbols and not hi.free_symbols:
            from sympy.calculus.util import minimum

            try:
                return sp.simplify(minimum(expr, var, sp.Interval(lo, hi)))
            except Exception:
                return None
        return None


class IntervalMax(sp.Function):
    """Exact symbolic maximum over [lo, hi] (evaluates once only dummy var remains free)."""

    @classmethod
    def eval(cls, expr: Any, var: Any, lo: Any, hi: Any) -> Any:
        if isinstance(var, sp.Symbol) and expr.free_symbols <= {var} and not lo.free_symbols and not hi.free_symbols:
            from sympy.calculus.util import maximum

            try:
                return sp.simplify(maximum(expr, var, sp.Interval(lo, hi)))
            except Exception:
                return None
        return None


def _interval_open(lo: Any, hi: Any) -> sp.Interval:
    return sp.Interval.open(lo, hi)


def _solve_set_eval(cond: Any, var: sp.Symbol, domain: Any = sp.S.Reals) -> Any:
    try:
        return sp.solveset(cond, var, domain)
    except Exception:
        return sp.EmptySet


class SolveSet(sp.Function):
    """Deferred symbolic solveset wrapper evaluated once only dummy var remains free."""

    @classmethod
    def eval(cls, cond: Any, var: Any, domain: Any = sp.S.Reals) -> Any:
        if isinstance(var, sp.Symbol) and (cond.free_symbols <= {var}) and not getattr(domain, "free_symbols", set()):
            try:
                return sp.solveset(cond, var, domain)
            except Exception:
                return None
        return None


def _resolve_seq_lo(lo_val: Any) -> int | None:
    if getattr(lo_val, "free_symbols", set()):
        return None
    try:
        return int(lo_val)
    except Exception:
        try:
            return int(sp.Integer(lo_val))
        except Exception:
            return None


def _eval_universal_int_rel(rel: Any, var: sp.Symbol, lo_int: int) -> Any:
    """Evaluate a relational predicate universally over integer var >= lo_int.

    Returns sp.true only when symbolically proved for all integers var >= lo_int,
    sp.false when any concrete integer probe var in [lo_int, lo_int + 4] falsifies it,
    or None if undecided.
    """
    if rel is sp.true or rel is True:
        return sp.true
    if rel is sp.false or rel is False:
        return sp.false
    if isinstance(rel, sp.Eq):
        diff = rel.lhs - rel.rhs
        if not (diff.free_symbols <= {var}):
            return None
        t_nonneg = sp.Dummy("seq_t", integer=True, nonnegative=True)
        for candidate in (diff.subs(var, lo_int + t_nonneg), diff):
            try:
                cand_expr = candidate.doit() if hasattr(candidate, "doit") else candidate
                if sp.simplify(cand_expr) == 0:
                    return sp.true
            except Exception:
                pass
        for offset in range(5):
            n_val = sp.Integer(lo_int + offset)
            try:
                sub_val = diff.subs(var, n_val)
                if hasattr(sub_val, "doit"):
                    sub_val = sub_val.doit()
                sub_val = sp.simplify(sub_val)
                if sub_val != 0 and not sub_val.free_symbols:
                    if abs(complex(sub_val.evalf(30))) > 1e-12:
                        return sp.false
            except Exception:
                pass
        return None

    if isinstance(rel, (sp.Lt, sp.Le, sp.Gt, sp.Ge, sp.Ne)):
        if not (rel.free_symbols <= {var}):
            return None
        for offset in range(5):
            n_val = sp.Integer(lo_int + offset)
            try:
                sub_rel = rel.subs(var, n_val)
                if hasattr(sub_rel, "doit"):
                    sub_rel = sub_rel.doit()
                if _decide_sympy_bool(sub_rel) is False:
                    return sp.false
            except Exception:
                pass
        t_nonneg = sp.Dummy("seq_t", integer=True, nonnegative=True)
        try:
            diff_shifted = (rel.lhs - rel.rhs).subs(var, lo_int + t_nonneg)
            if hasattr(diff_shifted, "doit"):
                diff_shifted = diff_shifted.doit()
            diff_shifted = sp.simplify(diff_shifted)
            if isinstance(rel, sp.Gt) and diff_shifted.is_positive is True:
                return sp.true
            if isinstance(rel, sp.Ge) and diff_shifted.is_nonnegative is True:
                return sp.true
            if isinstance(rel, sp.Lt) and diff_shifted.is_negative is True:
                return sp.true
            if isinstance(rel, sp.Le) and diff_shifted.is_nonpositive is True:
                return sp.true
            if isinstance(rel, sp.Ne) and diff_shifted.is_zero is False:
                return sp.true
        except Exception:
            pass
    return None


class SeqEq(BooleanFunction):
    """Universal sequence equality over integer var >= lo (default lo=1)."""

    @classmethod
    def eval(cls, *args: Any) -> Any:
        if len(args) == 2 and isinstance(args[0], sp.Eq):
            lhs, rhs = args[0].lhs, args[0].rhs
            var, lo_val = args[1], 1
        elif len(args) == 3 and isinstance(args[0], sp.Eq):
            lhs, rhs = args[0].lhs, args[0].rhs
            var, lo_val = args[1], args[2]
        elif len(args) == 3:
            lhs, rhs, var = args
            lo_val = 1
        elif len(args) == 4:
            lhs, rhs, var, lo_val = args
        else:
            return None
        if not isinstance(var, sp.Symbol):
            return None
        lo_int = _resolve_seq_lo(lo_val)
        if lo_int is None:
            return None
        return _eval_universal_int_rel(sp.Eq(lhs, rhs), var, lo_int)


class ForAllInt(BooleanFunction):
    """Universal sequence relation over integer var >= lo (default lo=1)."""

    @classmethod
    def eval(cls, *args: Any) -> Any:
        if len(args) == 2:
            rel, var = args
            lo_val = 1
        elif len(args) == 3:
            rel, var, lo_val = args
        else:
            return None
        if not isinstance(var, sp.Symbol):
            return None
        lo_int = _resolve_seq_lo(lo_val)
        if lo_int is None:
            return None
        return _eval_universal_int_rel(rel, var, lo_int)


def canonical_json_bytes(obj: Any) -> bytes:
    """Serialize any JSON structure deterministically as compact UTF-8 bytes."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(obj: Any) -> str:
    return sha256_bytes(canonical_json_bytes(obj))


def _base_sympy_ns(extra_symbols: dict[str, Any] | None = None) -> dict[str, Any]:
    """Build restricted, safe SymPy namespace (no builtins / arbitrary Python)."""
    ns: dict[str, Any] = {
        "sp": sp,
        "Symbol": sp.Symbol,
        "Rational": sp.Rational,
        "Integer": sp.Integer,
        "Float": sp.Float,
        "pi": sp.pi,
        "E": sp.E,
        "e": sp.E,
        "I": sp.I,
        "oo": sp.oo,
        "inf": sp.oo,
        "nan": sp.nan,
        "true": sp.true,
        "false": sp.false,
        "True": sp.true,
        "False": sp.false,
        "Abs": sp.Abs,
        "abs": sp.Abs,
        "sqrt": sp.sqrt,
        "cbrt": sp.cbrt,
        "exp": sp.exp,
        "log": sp.log,
        "ln": sp.log,
        "sin": sp.sin,
        "cos": sp.cos,
        "tan": sp.tan,
        "cot": sp.cot,
        "sec": sp.sec,
        "csc": sp.csc,
        "asin": sp.asin,
        "acos": sp.acos,
        "atan": sp.atan,
        "sinh": sp.sinh,
        "cosh": sp.cosh,
        "tanh": sp.tanh,
        "floor": sp.floor,
        "ceiling": sp.ceiling,
        "ceil": sp.ceiling,
        "binomial": sp.binomial,
        "factorial": sp.factorial,
        "gcd": sp.gcd,
        "lcm": sp.lcm,
        "Mod": sp.Mod,
        "isprime": sp.isprime,
        "Max": sp.Max,
        "Min": sp.Min,
        "max": sp.Max,
        "min": sp.Min,
        "Sum": sp.Sum,
        "Product": sp.Product,
        "diff": sp.diff,
        "integrate": sp.integrate,
        "limit": sp.limit,
        "simplify": sp.simplify,
        "Eq": sp.Eq,
        "Ne": sp.Ne,
        "Lt": sp.Lt,
        "Le": sp.Le,
        "Gt": sp.Gt,
        "Ge": sp.Ge,
        "And": sp.And,
        "Or": sp.Or,
        "Not": sp.Not,
        "Implies": sp.Implies,
        "Interval": sp.Interval,
        "IntervalOpen": _interval_open,
        "IntervalMin": IntervalMin,
        "IntervalMax": IntervalMax,
        "SolveSet": SolveSet,
        "SeqEq": SeqEq,
        "ForAllInt": ForAllInt,
        "FiniteSet": sp.FiniteSet,
        "Union": sp.Union,
        "Contains": sp.Contains,
        "S": sp.S,
        "Reals": sp.S.Reals,
        "Integers": sp.S.Integers,
        "Naturals": sp.S.Naturals,
        "Naturals0": sp.S.Naturals0,
        "ConditionSet": sp.ConditionSet,
        "Tuple": sp.Tuple,
    }
    if extra_symbols:
        ns.update(extra_symbols)
    return ns


def _make_symbol(name: str, assumption: str | None = None) -> sp.Symbol:
    kw: dict[str, bool] = {}
    if assumption:
        low = assumption.lower()
        if "positive" in low:
            kw["positive"] = True
            kw["real"] = True
        elif "nonnegative" in low:
            kw["nonnegative"] = True
            kw["real"] = True
        elif "negative" in low:
            kw["negative"] = True
            kw["real"] = True
        if "integer" in low or "int" == low:
            kw["integer"] = True
        elif "real" in low or not kw:
            kw["real"] = True
    else:
        kw["real"] = True
    return sp.Symbol(name, **kw)


def _split_top_level_commas(text: str) -> list[str]:
    parts: list[str] = []
    depth = 0
    cur: list[str] = []
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            parts.append("".join(cur).strip())
            cur = []
        else:
            cur.append(ch)
    tail = "".join(cur).strip()
    if tail:
        parts.append(tail)
    return parts


def _normalize_relation_string(expr_str: str) -> str:
    """Convert chained or infix relations like '0 < x < 1' or 'lhs = rhs' safely."""
    s = expr_str.strip()
    if not s:
        return s
    # Replace unicode operators if any slipped in.
    s = (
        s.replace("≤", "<=")
        .replace("≥", ">=")
        .replace("≠", "!=")
        .replace("＝", "=")
        .replace("×", "*")
        .replace("−", "-")
    )
    return s


def _parse_single_relation_or_expr(s: str, ns: dict[str, Any]) -> Any:
    s = _normalize_relation_string(s)
    # Detect top-level relational operators outside parentheses/brackets.
    ops: list[tuple[int, int, str]] = []
    depth = 0
    i = 0
    while i < len(s):
        ch = s[i]
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        elif depth == 0:
            two = s[i : i + 2]
            if two in ("==", "!=", "<=", ">="):
                ops.append((i, i + 2, two))
                i += 2
                continue
            if ch in ("=", "<", ">"):
                ops.append((i, i + 1, ch))
                i += 1
                continue
        i += 1

    if ops:
        parts: list[str] = []
        rel_ops: list[str] = []
        prev = 0
        for start, end, op in ops:
            parts.append(s[prev:start].strip())
            rel_ops.append(op)
            prev = end
        parts.append(s[prev:].strip())
        if len(parts) >= 2 and all(parts):
            parsed_parts = [_parse_raw_sympy(p, ns) for p in parts]
            clauses = []
            for idx, op in enumerate(rel_ops):
                lhs, rhs = parsed_parts[idx], parsed_parts[idx + 1]
                if op in ("=", "=="):
                    clauses.append(sp.Eq(lhs, rhs))
                elif op == "!=":
                    clauses.append(sp.Ne(lhs, rhs))
                elif op == "<":
                    clauses.append(sp.Lt(lhs, rhs))
                elif op == "<=":
                    clauses.append(sp.Le(lhs, rhs))
                elif op == ">":
                    clauses.append(sp.Gt(lhs, rhs))
                elif op == ">=":
                    clauses.append(sp.Ge(lhs, rhs))
            return clauses[0] if len(clauses) == 1 else sp.And(*clauses)

    return _parse_raw_sympy(s, ns)


def _parse_raw_sympy(expr_str: str, ns: dict[str, Any]) -> Any:
    s = _normalize_relation_string(expr_str)
    # Strip wrapping $ if passed accidentally.
    if len(s) >= 2 and s[0] == "$" and s[-1] == "$":
        s = s[1:-1].strip()
    local_dict = dict(ns)
    try:
        return parse_expr(s, local_dict=local_dict, transformations=_TRANSFORMS, evaluate=True)
    except Exception:
        return parse_expr(s, local_dict=local_dict, transformations=_TRANSFORMS_IMPLICIT, evaluate=True)


def parse_symbolic(expr_str: str, symbols_map: dict[str, Any] | None = None, fn_map: dict[str, sp.Lambda] | None = None) -> Any:
    """Safely parse a SymPy string with optional declared symbols and concrete function substitutions."""
    ns = _base_sympy_ns(symbols_map)
    if fn_map:
        for fname, lam in fn_map.items():
            ns[fname] = lam
    # Reject obvious dangerous tokens cleanly.
    if re.search(r"__|import\b|exec\b|eval\b|open\b|os\.|sys\.", expr_str):
        raise ValueError(f"Disallowed token in symbolic expression: {expr_str!r}")
    return _parse_single_relation_or_expr(expr_str, ns)


def _evaluate_predicate(
    pred_str: str,
    symbols_map: dict[str, sp.Symbol],
    assignment: dict[str, Any],
    fn_defs: dict[str, tuple[str, str]] | None = None,
) -> tuple[bool | None, str]:
    """Evaluate a predicate string under concrete variable assignments & function defs.

    Returns (True/False, "") if decisively evaluated, or (None, reason) if symbolic evaluation
    remains undecided.
    """
    # Build concrete function lambdas (from problem/witness fn_defs).
    fn_map: dict[str, sp.Lambda] = {}
    if fn_defs:
        for fname, (arg_str, body_str) in fn_defs.items():
            args = [symbols_map.get(a.strip()) or _make_symbol(a.strip(), "real") for a in arg_str.split(",") if a.strip()]
            body_ns = dict(symbols_map)
            for a in args:
                body_ns[a.name] = a
            body_expr = parse_symbolic(body_str, body_ns, fn_map=None)
            fn_map[fname] = sp.Lambda(tuple(args), body_expr)

    # Allow witness assignment values to reference earlier assigned symbols or concrete numbers.
    parsed_assign: dict[sp.Symbol, Any] = {}
    for k, v in assignment.items():
        if isinstance(v, dict) or k.endswith("_fn") or "(" in k:
            continue
        sym = symbols_map.get(k) or _make_symbol(k, "real")
        symbols_map[k] = sym
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            val = sp.Integer(v) if isinstance(v, int) else sp.nsimplify(str(v))
        else:
            val = parse_symbolic(str(v), symbols_map, fn_map)
        if hasattr(val, "doit") and hasattr(val, "has") and val.has(sp.Sum, sp.Product) and not getattr(val, "free_symbols", set()):
            val = val.doit()
        parsed_assign[sym] = val

    # Resolve interdependent substitutions up to depth 3.
    for _ in range(3):
        changed = False
        for sym, val in list(parsed_assign.items()):
            new_val = val.subs(parsed_assign) if hasattr(val, "subs") else val
            if new_val != val:
                parsed_assign[sym] = sp.simplify(new_val)
                changed = True
        if not changed:
            break

    # Apply function-level derivative helpers e.g. f'(x) -> df/dx if string used derivative notation.
    pred = parse_symbolic(pred_str, symbols_map, fn_map)

    # Check if predicate is a universally quantified "ForAll(x, domain, cond)" helper or contains free dummy var.
    if hasattr(pred, "subs"):
        sub_pred = pred.subs(parsed_assign)
    else:
        sub_pred = pred
    if hasattr(sub_pred, "doit"):
        sub_pred = sub_pred.doit()

    sub_pred = _simplify_boolean_expr(sub_pred)
    if sub_pred is sp.true or sub_pred is True:
        return True, ""
    if sub_pred is sp.false or sub_pred is False:
        return False, ""

    # Try exact relational evaluation via difference simplification.
    res = _decide_sympy_bool(sub_pred)
    if res is not None:
        return res, ""
    return None, f"Undecided predicate {pred_str!r} -> {sub_pred}"


def _simplify_boolean_expr(expr: Any) -> Any:
    if isinstance(expr, sp.And):
        args = [_simplify_boolean_expr(a) for a in expr.args]
        if any(a is sp.false or a is False for a in args):
            return sp.false
        rem = [a for a in args if not (a is sp.true or a is True)]
        if not rem:
            return sp.true
        return sp.And(*rem)
    if isinstance(expr, sp.Or):
        args = [_simplify_boolean_expr(a) for a in expr.args]
        if any(a is sp.true or a is True for a in args):
            return sp.true
        rem = [a for a in args if not (a is sp.false or a is False)]
        if not rem:
            return sp.false
        return sp.Or(*rem)
    if isinstance(expr, sp.Not):
        inner = _simplify_boolean_expr(expr.args[0])
        if inner is sp.true or inner is True:
            return sp.false
        if inner is sp.false or inner is False:
            return sp.true
        return sp.Not(inner)
    if isinstance(expr, sp.Contains):
        elem, sset = expr.args
        try:
            contained = sset.contains(sp.simplify(elem))
            if contained is sp.true or contained is True:
                return sp.true
            if contained is sp.false or contained is False:
                return sp.false
        except Exception:
            pass
    decided = _decide_sympy_bool(expr)
    if decided is True:
        return sp.true
    if decided is False:
        return sp.false
    return expr


def _decide_sympy_bool(expr: Any) -> bool | None:
    if expr is sp.true or expr is True:
        return True
    if expr is sp.false or expr is False:
        return False
    if isinstance(expr, (sp.And, sp.Or, sp.Not)):
        simp = _simplify_boolean_expr(expr)
        if simp is sp.true or simp is True:
            return True
        if simp is sp.false or simp is False:
            return False
        return None
    if isinstance(expr, (sp.Eq, sp.Ne)):
        if isinstance(expr.lhs, sp.Set) or isinstance(expr.rhs, sp.Set):
            try:
                eq_set = expr.lhs == expr.rhs or sp.SymmetricDifference(expr.lhs, expr.rhs) is sp.EmptySet
                return eq_set if isinstance(expr, sp.Eq) else (not eq_set)
            except Exception:
                return None
        if isinstance(expr.lhs, sp.Tuple) or isinstance(expr.rhs, sp.Tuple):
            if not (isinstance(expr.lhs, sp.Tuple) and isinstance(expr.rhs, sp.Tuple)) or len(expr.lhs) != len(expr.rhs):
                return False if isinstance(expr, sp.Eq) else True
            elem_eqs = [_decide_sympy_bool(sp.Eq(a, b)) for a, b in zip(expr.lhs, expr.rhs)]
            if all(x is True for x in elem_eqs):
                return True if isinstance(expr, sp.Eq) else False
            if any(x is False for x in elem_eqs):
                return False if isinstance(expr, sp.Eq) else True
            return None
        diff_expr = sp.simplify((expr.lhs - expr.rhs).doit() if hasattr(expr.lhs - expr.rhs, "doit") else (expr.lhs - expr.rhs))
        if diff_expr == 0:
            return True if isinstance(expr, sp.Eq) else False
        if not diff_expr.free_symbols and isinstance(diff_expr, sp.Expr):
            try:
                val = complex(diff_expr.evalf(50))
                if abs(val) > 1e-20:
                    return False if isinstance(expr, sp.Eq) else True
            except Exception:
                pass
        # If symbolic expression still differs identically, test non-zero polynomial/analytic difference.
        if diff_expr.is_number is False and diff_expr.free_symbols:
            try:
                syms = sorted(diff_expr.free_symbols, key=lambda s: s.name)
                probe_offsets = (1, 2, 3, 4) if any(s.is_integer for s in syms) else (0,)
                for off in probe_offsets:
                    if not any(s.is_integer for s in syms):
                        probe = {s: sp.Rational(idx + 2, 7) for idx, s in enumerate(syms)}
                    else:
                        probe = {s: idx + off for idx, s in enumerate(syms)}
                    cand = diff_expr.subs(probe)
                    if hasattr(cand, "doit"):
                        cand = cand.doit()
                    pv = complex(sp.simplify(cand).evalf(30))
                    if abs(pv) > 1e-12:
                        return False if isinstance(expr, sp.Eq) else True
            except Exception:
                pass
        return None

    if isinstance(expr, (sp.Lt, sp.Le, sp.Gt, sp.Ge)):
        diff_expr = sp.simplify(expr.lhs - expr.rhs)
        if diff_expr.free_symbols:
            return None
        try:
            # First try exact symbolic truth check.
            simp_rel = sp.simplify(expr.func(diff_expr, 0))
            if simp_rel is sp.true or simp_rel is True:
                return True
            if simp_rel is sp.false or simp_rel is False:
                return False
            # Fallback only when free_symbols is empty: high-precision constant evaluation
            # strictly guarded away from zero (for transcendental comparisons like pi + e vs ...).
            val = complex(diff_expr.evalf(50))
            if abs(val.imag) > 1e-25:
                return None
            rv = val.real
            if abs(rv) <= 1e-25:
                return None
            if isinstance(expr, sp.Lt):
                return rv < 0
            if isinstance(expr, sp.Le):
                return rv <= 0
            if isinstance(expr, sp.Gt):
                return rv > 0
            if isinstance(expr, sp.Ge):
                return rv >= 0
        except Exception:
            return None
    return None


def _extract_fn_defs(witness: dict[str, Any], base_fns: dict[str, Any]) -> dict[str, tuple[str, str]]:
    merged: dict[str, tuple[str, str]] = {}
    for src in (base_fns, witness.get("functions") or {}):
        for fname, spec in src.items():
            if isinstance(spec, dict) and "args" in spec and "expr" in spec:
                merged[fname] = (str(spec["args"]), str(spec["expr"]))
            elif isinstance(spec, str) and "->" in spec:
                left, right = spec.split("->", 1)
                merged[fname] = (left.strip(), right.strip())
    return merged


def _eval_target(
    target_expr_str: str,
    symbols_map: dict[str, sp.Symbol],
    witness: dict[str, Any],
    base_fns: dict[str, Any],
) -> Any:
    assignment = witness.get("values") or {}
    fn_defs = _extract_fn_defs(witness, base_fns)
    fn_map: dict[str, sp.Lambda] = {}
    for fname, (arg_str, body_str) in fn_defs.items():
        args = [symbols_map.get(a.strip()) or _make_symbol(a.strip(), "real") for a in arg_str.split(",") if a.strip()]
        body_ns = dict(symbols_map)
        for a in args:
            body_ns[a.name] = a
        body_expr = parse_symbolic(body_str, body_ns, fn_map=None)
        fn_map[fname] = sp.Lambda(tuple(args), body_expr)

    parsed_assign: dict[sp.Symbol, Any] = {}
    for k, v in assignment.items():
        if isinstance(v, dict):
            continue
        sym = symbols_map.get(k) or _make_symbol(k, "real")
        symbols_map[k] = sym
        val = sp.Integer(v) if isinstance(v, int) else parse_symbolic(str(v), symbols_map, fn_map)
        if hasattr(val, "doit") and hasattr(val, "has") and val.has(sp.Sum, sp.Product) and not getattr(val, "free_symbols", set()):
            val = val.doit()
        parsed_assign[sym] = val
    for _ in range(3):
        for sym, val in list(parsed_assign.items()):
            if hasattr(val, "subs"):
                parsed_assign[sym] = sp.simplify(val.subs(parsed_assign))

    texpr = parse_symbolic(target_expr_str, symbols_map, fn_map)
    if hasattr(texpr, "subs"):
        texpr = texpr.subs(parsed_assign)
    if hasattr(texpr, "doit"):
        texpr = texpr.doit()
    return sp.simplify(texpr)


def _check_all_predicates(
    preds: list[str],
    symbols_map: dict[str, sp.Symbol],
    witness: dict[str, Any],
    base_fns: dict[str, Any],
) -> tuple[bool, list[str]]:
    assignment = witness.get("values") or {}
    fn_defs = _extract_fn_defs(witness, base_fns)
    failures: list[str] = []
    for p in preds:
        ok, msg = _evaluate_predicate(p, dict(symbols_map), assignment, fn_defs)
        if ok is not True:
            failures.append(msg or f"Predicate evaluated False: {p}")
    return (len(failures) == 0), failures


def _check_any_predicate_false(
    preds: list[str],
    symbols_map: dict[str, sp.Symbol],
    witness: dict[str, Any],
    base_fns: dict[str, Any],
) -> bool:
    """Verify at least one predicate of shard j is definitively FALSE under the counter-witness."""
    assignment = witness.get("values") or {}
    fn_defs = _extract_fn_defs(witness, base_fns)
    for p in preds:
        ok, _ = _evaluate_predicate(p, dict(symbols_map), assignment, fn_defs)
        if ok is False:
            return True
    return False


def _has_witness_payload(witness: Any) -> bool:
    return isinstance(witness, dict) and bool(witness.get("values") or witness.get("functions"))


def _expr_references_domain(expr_str: str, symbols_map: dict[str, sp.Symbol], base_fns: dict[str, Any], witness: dict[str, Any]) -> bool:
    ident_set = set(symbols_map.keys()) | set(base_fns.keys()) | set((witness.get("functions") or {}).keys())
    tokens = set(re.findall(r"\b[A-Za-z_]\w*\b", expr_str))
    return bool(tokens & ident_set)


def verify_certificate_record(
    candidate_rec: dict[str, Any],
    cert_payload: dict[str, Any] | None,
    stage1_manifest_sha256: str,
    verifier_config_sha256: str,
) -> dict[str, Any]:
    """Deterministically verify one Stage 2B certificate against Stage 2A + SPEC v2.4 rules.

    Returns both the compact Stage 2B result dict (SPEC 12.3) and full verifiable report dict.
    """
    item_id = candidate_rec["item_id"]
    provisional_mech = candidate_rec["provisional_mechanism"]
    question = candidate_rec["question"]
    parsed_output = candidate_rec.get("parsed_output")

    base_binding_obj = {
        "stage1_manifest_sha256": stage1_manifest_sha256,
        "item_id": item_id,
        "question_sha256": candidate_rec["question_sha256"],
        "macro_subfield": candidate_rec["macro_subfield"],
        "provisional_mechanism": provisional_mech,
        "stage2b_rule_version": STAGE2B_RULE_VERSION,
        "verifier_config_sha256": verifier_config_sha256,
        "parsed_output_sha256": sha256_json(parsed_output) if parsed_output is not None else None,
        "certificate_sha256": sha256_json(cert_payload) if cert_payload is not None else None,
    }
    binding_digest = sha256_json(base_binding_obj)

    def make_fail(reason: str, details: list[str], verified_mech: str | None = None, k_val: int | None = None) -> dict[str, Any]:
        report = {
            "item_id": item_id,
            "verify_status": "VERIFY_FAIL",
            "canonical_k": k_val,
            "verified_mechanism": verified_mech,
            "reject_reason": reason,
            "binding_digest": binding_digest,
            "certificate_sha256": base_binding_obj["certificate_sha256"],
            "stage2b_rule_version": STAGE2B_RULE_VERSION,
            "details": details,
        }
        report_sha = sha256_json(report)
        return {
            "result": {
                "item_id": item_id,
                "verify_status": "VERIFY_FAIL",
                "canonical_k": k_val,
                "verified_mechanism": verified_mech,
                "reject_reason": reason,
                "certificate_sha256": base_binding_obj["certificate_sha256"],
                "verification_report_sha256": report_sha,
                "binding_digest": binding_digest,
            },
            "report": report,
        }

    # 1. Alignment & shard count check (always re-computed from raw source text).
    if parsed_output is None or not isinstance(parsed_output, dict):
        return make_fail("FAIL_SHARD_ALIGNMENT", ["Missing parsed_output from Stage 2A."])
    if parsed_output.get("status") != "CANDIDATE":
        return make_fail("FAIL_SHARD_ALIGNMENT", [f"Stage 2A status is {parsed_output.get('status')!r}, not CANDIDATE."])

    audit = audit_shard_candidate(question, parsed_output)
    if not audit["alignment_passed"]:
        errs = [f"{e['code']}: {e['message']}" for e in audit.get("errors", [])]
        k_val = audit.get("total_shards")
        if any(e.get("code") == "INVALID_SHARD_COUNT" for e in audit.get("errors", [])):
            return make_fail("FAIL_K_OUT_OF_RANGE", errs, k_val=k_val)
        return make_fail("FAIL_SHARD_ALIGNMENT", errs, k_val=k_val)

    k = int(audit["total_shards"])
    if not (3 <= k <= 6):
        return make_fail("FAIL_K_OUT_OF_RANGE", [f"total_shards={k} not in [3,6]."], k_val=k)

    # 2. Certificate availability check.
    if not cert_payload or not isinstance(cert_payload, dict):
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", ["No Stage 2B formal certificate provided."], k_val=k)

    if cert_payload.get("status") == "UNVERIFIABLE":
        reason = cert_payload.get("reject_reason") or "FAIL_CERTIFICATE_UNAVAILABLE"
        if reason not in FAIL_REASONS:
            reason = "FAIL_CERTIFICATE_UNAVAILABLE"
        return make_fail(reason, [str(cert_payload.get("reason_detail") or "Marked UNVERIFIABLE.")], k_val=k)

    # 3. Mechanism consistency check.
    v_mech = cert_payload.get("verified_mechanism")
    if v_mech != provisional_mech:
        return make_fail(
            "FAIL_MECHANISM_MISMATCH",
            [f"verified_mechanism={v_mech!r} != provisional_mechanism={provisional_mech!r}."],
            verified_mech=v_mech,
            k_val=k,
        )

    # 4. Redundancy / Equivalence semantic flags.
    redundant = cert_payload.get("redundant_shards") or []
    if redundant:
        return make_fail(
            "FAIL_REDUNDANT_PREMISE",
            [f"Redundant shards identified: {redundant}; detail={cert_payload.get('redundancy_reason', '')}"],
            verified_mech=v_mech,
            k_val=k,
        )

    if cert_payload.get("concat_equivalent_to_original") is not True:
        return make_fail(
            "FAIL_SOURCE_ANSWER_MISMATCH",
            [f"concat_equivalent_to_original is not True: {cert_payload.get('equivalence_note', '')}"],
            verified_mech=v_mech,
            k_val=k,
        )

    # 5. Formal symbolic SymPy checks.
    var_specs = cert_payload.get("variables") or {}
    if not isinstance(var_specs, dict) or not var_specs:
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", ["Empty variables declaration in certificate."], verified_mech=v_mech, k_val=k)

    symbols_map: dict[str, sp.Symbol] = {}
    for vname, vtype in var_specs.items():
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", str(vname)):
            return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"Invalid variable name {vname!r}."], verified_mech=v_mech, k_val=k)
        symbols_map[vname] = _make_symbol(vname, str(vtype) if vtype else "real")

    shard_preds: dict[str, list[str]] = cert_payload.get("shard_constraints") or {}
    expected_keys = ["S1"] + [f"S{j}" for j in range(2, k + 1)]
    for sk in expected_keys[1:]:
        preds = shard_preds.get(sk)
        if not isinstance(preds, list) or not preds:
            return make_fail(
                "FAIL_CERTIFICATE_UNAVAILABLE",
                [f"Missing formal symbolic constraints for shard {sk}."],
                verified_mech=v_mech,
                k_val=k,
            )
    s1_preds = shard_preds.get("S1") or []
    if not isinstance(s1_preds, list):
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", ["Invalid shard_constraints.S1."], verified_mech=v_mech, k_val=k)

    target_expr = str(cert_payload.get("target_expr") or "").strip()
    if not target_expr:
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", ["Missing target_expr."], verified_mech=v_mech, k_val=k)

    base_fns = cert_payload.get("functions") or {}

    # 5a. Full-condition source-answer check.
    full_check = cert_payload.get("full_answer_check") or {}
    all_preds = list(s1_preds)
    for sk in expected_keys[1:]:
        all_preds.extend(shard_preds[sk])

    full_witness = full_check.get("witness")
    if not _has_witness_payload(full_witness):
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", ["Missing full_answer_check.witness."], verified_mech=v_mech, k_val=k)

    try:
        ok_full, errs_full = _check_all_predicates(all_preds, symbols_map, full_witness, base_fns)
    except Exception as exc:
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"SymPy error in full_answer_check: {exc}"], verified_mech=v_mech, k_val=k)
    if not ok_full:
        return make_fail("FAIL_SOURCE_ANSWER_MISMATCH", [f"Full witness violates full constraints: {errs_full}"], verified_mech=v_mech, k_val=k)

    try:
        full_val = _eval_target(target_expr, dict(symbols_map), full_witness, base_fns)
    except Exception as exc:
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"Failed evaluating target_expr on full_witness: {exc}"], verified_mech=v_mech, k_val=k)

    ans_kind = full_check.get("answer_kind", "scalar")
    try:
        if ans_kind in ("scalar", "max", "min", "expression"):
            decl_str = str(full_check.get("declared_value_sympy") or "").strip()
            if not decl_str:
                return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", ["Missing declared_value_sympy."], verified_mech=v_mech, k_val=k)
            decl_val = _eval_target(decl_str, dict(symbols_map), full_witness, base_fns)
            eq_res = _decide_sympy_bool(sp.Eq(full_val, decl_val))
            if eq_res is not True:
                return make_fail(
                    "FAIL_SOURCE_ANSWER_MISMATCH",
                    [f"Full witness target={full_val} does not equal declared_value={decl_val}."],
                    verified_mech=v_mech,
                    k_val=k,
                )
        elif ans_kind in ("interval", "set"):
            interval_str = str(full_check.get("declared_interval_sympy") or full_check.get("declared_value_sympy") or "").strip()
            if not interval_str:
                return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", ["Missing declared_interval_sympy."], verified_mech=v_mech, k_val=k)
            decl_interval = _eval_target(interval_str, dict(symbols_map), full_witness, base_fns)
            if isinstance(full_val, sp.Set):
                eq_set = _decide_sympy_bool(sp.Eq(full_val, decl_interval))
                if eq_set is not True:
                    return make_fail(
                        "FAIL_SOURCE_ANSWER_MISMATCH",
                        [f"Full witness target set={full_val} does not equal declared set={decl_interval}."],
                        verified_mech=v_mech,
                        k_val=k,
                    )
            else:
                in_set = decl_interval.contains(full_val)
                if _decide_sympy_bool(sp.Eq(in_set, sp.true)) is not True and in_set is not sp.true:
                    return make_fail(
                        "FAIL_SOURCE_ANSWER_MISMATCH",
                        [f"Full witness target={full_val} not contained in declared_interval={decl_interval}."],
                        verified_mech=v_mech,
                        k_val=k,
                    )
        else:
            return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"Unsupported answer_kind {ans_kind!r}."], verified_mech=v_mech, k_val=k)
    except Exception as exc:
        return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"SymPy error comparing full target vs declared answer: {exc}"], verified_mech=v_mech, k_val=k)

    # 5b. Leave-One-Out verification for every j = 2 .. k.
    loo_map = cert_payload.get("leave_one_out") or {}
    loo_summary: dict[str, Any] = {}
    for sk in expected_keys[1:]:
        entry = loo_map.get(sk)
        if not isinstance(entry, dict):
            return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"Missing leave_one_out[{sk}]."], verified_mech=v_mech, k_val=k)

        remaining_preds = list(s1_preds)
        for other_sk in expected_keys[1:]:
            if other_sk != sk:
                remaining_preds.extend(shard_preds[other_sk])

        if v_mech == "TypeA_UniqueValue":
            w1 = entry.get("witness_1") or full_witness
            w2 = entry.get("witness_2")
            ubw = entry.get("unbounded_witness")
            has_w2 = _has_witness_payload(w2)
            ubw_vexpr = str(ubw.get("violation_expr") or "").strip() if isinstance(ubw, dict) else ""
            has_ubw = _has_witness_payload(ubw) and bool(ubw_vexpr)
            if not _has_witness_payload(w1) or not (has_w2 or has_ubw):
                return make_fail(
                    "FAIL_CERTIFICATE_UNAVAILABLE",
                    [f"{sk}: Missing TypeA witness_1 and (witness_2 or unbounded_witness)."],
                    verified_mech=v_mech,
                    k_val=k,
                )
            try:
                ok1, errs1 = _check_all_predicates(remaining_preds, symbols_map, w1, base_fns)
                if has_w2:
                    ok2, errs2 = _check_all_predicates(remaining_preds, symbols_map, w2, base_fns)
                    if not ok1 or not ok2:
                        return make_fail(
                            "FAIL_CERTIFICATE_UNAVAILABLE",
                            [f"{sk}: Leave-One-Out witness failed remaining constraints: w1={errs1}, w2={errs2}"],
                            verified_mech=v_mech,
                            k_val=k,
                        )
                    violated_j = _check_any_predicate_false(shard_preds[sk], symbols_map, w2, base_fns)
                    if not violated_j:
                        return make_fail(
                            "FAIL_REDUNDANT_PREMISE",
                            [f"{sk}: counter-witness w2 did not falsify any predicate of {sk}; necessity unproven."],
                            verified_mech=v_mech,
                            k_val=k,
                        )
                    v1 = _eval_target(target_expr, dict(symbols_map), w1, base_fns)
                    v2 = _eval_target(target_expr, dict(symbols_map), w2, base_fns)
                    diff_res = _decide_sympy_bool(sp.Ne(v1, v2))
                    if diff_res is not True:
                        return make_fail(
                            "FAIL_CERTIFICATE_UNAVAILABLE",
                            [f"{sk}: TypeA witness targets not provably distinct: v1={v1}, v2={v2}"],
                            verified_mech=v_mech,
                            k_val=k,
                        )
                    loo_summary[sk] = {"v1": str(v1), "v2": str(v2), "verified": True}
                else:
                    ok_ub, errs_ub = _check_all_predicates(remaining_preds, symbols_map, ubw, base_fns)
                    if not ok1 or not ok_ub:
                        return make_fail(
                            "FAIL_CERTIFICATE_UNAVAILABLE",
                            [f"{sk}: Leave-One-Out unbounded_witness failed remaining constraints: w1={errs1}, ubw={errs_ub}"],
                            verified_mech=v_mech,
                            k_val=k,
                        )
                    violated_j = _check_any_predicate_false(shard_preds[sk], symbols_map, ubw, base_fns)
                    if not violated_j:
                        return make_fail(
                            "FAIL_REDUNDANT_PREMISE",
                            [f"{sk}: unbounded_witness did not falsify any predicate of {sk}; necessity unproven."],
                            verified_mech=v_mech,
                            k_val=k,
                        )
                    if not _expr_references_domain(ubw_vexpr, symbols_map, base_fns, ubw):
                        return make_fail(
                            "FAIL_CERTIFICATE_UNAVAILABLE",
                            [f"{sk}: unbounded_witness.violation_expr must reference problem symbols/functions."],
                            verified_mech=v_mech,
                            k_val=k,
                        )
                    ub_fns = _extract_fn_defs(ubw, base_fns)
                    ok_viol, err_viol = _evaluate_predicate(ubw_vexpr, dict(symbols_map), ubw.get("values") or {}, ub_fns)
                    if ok_viol is not True:
                        return make_fail(
                            "FAIL_CERTIFICATE_UNAVAILABLE",
                            [f"{sk}: unbounded_witness.violation_expr not satisfied: {err_viol or ubw_vexpr}"],
                            verified_mech=v_mech,
                            k_val=k,
                        )
                    loo_summary[sk] = {"unbounded_witness": ubw_vexpr, "verified": True}
            except Exception as exc:
                return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"{sk}: SymPy LOO error: {exc}"], verified_mech=v_mech, k_val=k)

        elif v_mech == "TypeB_Extremum":
            rw = entry.get("relaxed_witness")
            if not _has_witness_payload(rw):
                return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"{sk}: Missing TypeB relaxed_witness."], verified_mech=v_mech, k_val=k)
            try:
                okr, errsr = _check_all_predicates(remaining_preds, symbols_map, rw, base_fns)
                if not okr:
                    return make_fail(
                        "FAIL_CERTIFICATE_UNAVAILABLE",
                        [f"{sk}: relaxed_witness violated remaining constraints: {errsr}"],
                        verified_mech=v_mech,
                        k_val=k,
                    )
                violated_j = _check_any_predicate_false(shard_preds[sk], symbols_map, rw, base_fns)
                if not violated_j:
                    return make_fail(
                        "FAIL_REDUNDANT_PREMISE",
                        [f"{sk}: relaxed_witness did not falsify any predicate of {sk}; necessity unproven."],
                        verified_mech=v_mech,
                        k_val=k,
                    )
                rv = _eval_target(target_expr, dict(symbols_map), rw, base_fns)
                if ans_kind == "max":
                    decl_val = parse_symbolic(str(full_check["declared_value_sympy"]), symbols_map, None)
                    better = _decide_sympy_bool(sp.Gt(rv, decl_val))
                    msg_cmp = f"rv={rv} > max={decl_val}"
                elif ans_kind == "min":
                    decl_val = parse_symbolic(str(full_check["declared_value_sympy"]), symbols_map, None)
                    better = _decide_sympy_bool(sp.Lt(rv, decl_val))
                    msg_cmp = f"rv={rv} < min={decl_val}"
                elif ans_kind == "interval":
                    decl_interval = parse_symbolic(str(full_check["declared_interval_sympy"]), symbols_map, None)
                    in_set = decl_interval.contains(rv)
                    better = in_set is sp.false or in_set is False or (_decide_sympy_bool(sp.Eq(in_set, sp.false)) is True)
                    msg_cmp = f"rv={rv} outside {decl_interval}"
                else:
                    return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"{sk}: Unsupported TypeB answer_kind={ans_kind!r}."], verified_mech=v_mech, k_val=k)

                if better is not True:
                    return make_fail(
                        "FAIL_CERTIFICATE_UNAVAILABLE",
                        [f"{sk}: TypeB relaxed witness did not strictly beat/escape optimum ({msg_cmp})."],
                        verified_mech=v_mech,
                        k_val=k,
                    )
                loo_summary[sk] = {"relaxed_value": str(rv), "check": msg_cmp, "verified": True}
            except Exception as exc:
                return make_fail("FAIL_CERTIFICATE_UNAVAILABLE", [f"{sk}: SymPy LOO error: {exc}"], verified_mech=v_mech, k_val=k)

    # All checks passed -> construct acyclic report & certificate hash.
    report = {
        "item_id": item_id,
        "verify_status": "VERIFY_PASS",
        "canonical_k": k,
        "verified_mechanism": v_mech,
        "reject_reason": "",
        "binding_digest": binding_digest,
        "certificate_sha256": base_binding_obj["certificate_sha256"],
        "stage2b_rule_version": STAGE2B_RULE_VERSION,
        "full_answer_verified": str(full_val),
        "leave_one_out_verified": loo_summary,
    }
    report_sha = sha256_json(report)
    return {
        "result": {
            "item_id": item_id,
            "verify_status": "VERIFY_PASS",
            "canonical_k": k,
            "verified_mechanism": v_mech,
            "reject_reason": "",
            "certificate_sha256": base_binding_obj["certificate_sha256"],
            "verification_report_sha256": report_sha,
            "binding_digest": binding_digest,
        },
        "report": report,
    }


def _run_self_tests() -> None:
    manifest_h = "a" * 64
    cfg_h = "b" * 64

    # 1. Valid TypeA scalar
    cand_a = {
        "item_id": "t-a",
        "question": "已知正数a,b，a+b=5，a=2，求b。",
        "question_sha256": "c" * 64,
        "macro_subfield": "G2_Alg_Ineq",
        "provisional_mechanism": "TypeA_UniqueValue",
        "parsed_output": {
            "status": "CANDIDATE",
            "initial_segments": ["已知正数a,b", "求b。"],
            "initial_shard": "已知正数 $a,b$，求 $b$。",
            "shards": [
                {"shard_id": "S2", "segments": ["a+b=5"], "shard": "$a+b=5$。"},
                {"shard_id": "S3", "segments": ["a=2"], "shard": "$a=2$。"},
            ],
        },
    }
    cert_a = {
        "status": "VERIFIABLE",
        "verified_mechanism": "TypeA_UniqueValue",
        "concat_equivalent_to_original": True,
        "redundant_shards": [],
        "variables": {"a": "positive", "b": "positive"},
        "shard_constraints": {"S1": ["a > 0", "b > 0"], "S2": ["a + b = 5"], "S3": ["a = 2"]},
        "target_expr": "b",
        "full_answer_check": {"answer_kind": "scalar", "declared_value_sympy": "3", "witness": {"values": {"a": 2, "b": 3}}},
        "leave_one_out": {
            "S2": {"witness_2": {"values": {"a": 2, "b": 4}}},
            "S3": {"witness_2": {"values": {"a": 1, "b": 4}}},
        },
    }
    r_a = verify_certificate_record(cand_a, cert_a, manifest_h, cfg_h)
    assert r_a["result"]["verify_status"] == "VERIFY_PASS", r_a

    # 2. Redundant premise caught deterministically
    cert_red = json.loads(json.dumps(cert_a))
    cert_red["leave_one_out"]["S3"]["witness_2"]["values"] = {"a": 2, "b": 3}
    r_red = verify_certificate_record(cand_a, cert_red, manifest_h, cfg_h)
    assert r_red["result"]["verify_status"] == "VERIFY_FAIL", r_red
    assert r_red["result"]["reject_reason"] == "FAIL_REDUNDANT_PREMISE", r_red

    # 3. Valid TypeB Extremum with function substitution
    cand_b = {
        "item_id": "t-b",
        "question": "已知函数f(x)=a*x+1，若f(1)>=0，求a的取值范围。",
        "question_sha256": "d" * 64,
        "macro_subfield": "G1_Func_Calc_Trig",
        "provisional_mechanism": "TypeB_Extremum",
        "parsed_output": {
            "status": "CANDIDATE",
            "initial_segments": ["已知函数", "求a的取值范围。"],
            "initial_shard": "已知函数 $f(x)$，求 $a$ 的取值范围。",
            "shards": [
                {"shard_id": "S2", "segments": ["f(x)=a*x+1"], "shard": "$f(x)=a*x+1$。"},
                {"shard_id": "S3", "segments": ["若f(1)>=0"], "shard": "若 $f(1)>=0$。"},
            ],
        },
    }
    cert_b = {
        "status": "VERIFIABLE",
        "verified_mechanism": "TypeB_Extremum",
        "concat_equivalent_to_original": True,
        "redundant_shards": [],
        "variables": {"a": "real", "x": "real"},
        "functions": {"f": {"args": "x", "expr": "a*x + 1"}},
        "shard_constraints": {"S1": [], "S2": ["Eq(f(0), 1)", "Eq(f(1), a + 1)"], "S3": ["f(1) >= 0"]},
        "target_expr": "a",
        "full_answer_check": {
            "answer_kind": "interval",
            "declared_interval_sympy": "Interval(-1, oo)",
            "witness": {"values": {"a": -1}},
        },
        "leave_one_out": {
            "S2": {
                "relaxed_witness": {
                    "values": {"a": -5},
                    "functions": {"f": {"args": "x", "expr": "1"}},
                }
            },
            "S3": {"relaxed_witness": {"values": {"a": -2}}},
        },
    }
    r_b = verify_certificate_record(cand_b, cert_b, manifest_h, cfg_h)
    assert r_b["result"]["verify_status"] == "VERIFY_PASS", r_b

    # 4. Symbolic expression answer_kind ("expression" e.g. curve equation)
    cand_e = {
        "item_id": "t-e",
        "question": "已知抛物线y^2=2px，p>0，准线x=-1，求方程。",
        "question_sha256": "e" * 64,
        "macro_subfield": "G4_Geom_Analytic",
        "provisional_mechanism": "TypeA_UniqueValue",
        "parsed_output": {
            "status": "CANDIDATE",
            "initial_segments": ["已知抛物线", "求方程。"],
            "initial_shard": "已知抛物线，求方程。",
            "shards": [
                {"shard_id": "S2", "segments": ["y^2=2px"], "shard": "$y^2=2px$。"},
                {"shard_id": "S3", "segments": ["p>0"], "shard": "$p>0$。"},
                {"shard_id": "S4", "segments": ["准线x=-1"], "shard": "准线 $x=-1$。"},
            ],
        },
    }
    cert_e = {
        "status": "VERIFIABLE",
        "verified_mechanism": "TypeA_UniqueValue",
        "concat_equivalent_to_original": True,
        "redundant_shards": [],
        "variables": {"p": "real", "k": "real", "x": "real", "y": "real"},
        "shard_constraints": {"S1": [], "S2": ["Eq(k, 2*p)"], "S3": ["p > 0"], "S4": ["Eq(Abs(p)/2, 1)"]},
        "target_expr": "y**2 - k*x",
        "full_answer_check": {
            "answer_kind": "expression",
            "declared_value_sympy": "y**2 - 4*x",
            "witness": {"values": {"p": 2, "k": 4}},
        },
        "leave_one_out": {
            "S2": {"witness_2": {"values": {"p": 2, "k": 6}}},
            "S3": {"witness_2": {"values": {"p": -2, "k": -4}}},
            "S4": {"witness_2": {"values": {"p": 1, "k": 2}}},
        },
    }
    r_e = verify_certificate_record(cand_e, cert_e, manifest_h, cfg_h)
    assert r_e["result"]["verify_status"] == "VERIFY_PASS", r_e

    # 5. Symbolic IntervalMin test (e.g. f(x)=exp(x-1)+a*x >= 0 for all real x)
    cand_f = {
        "item_id": "t-f",
        "question": "已知函数f(x)=e^{x-1}+ax。若f(x)>=0恒成立，求a的取值范围。",
        "question_sha256": "f" * 64,
        "macro_subfield": "G1_Func_Calc_Trig",
        "provisional_mechanism": "TypeB_Extremum",
        "parsed_output": {
            "status": "CANDIDATE",
            "initial_segments": ["已知函数", "求a的取值范围。"],
            "initial_shard": "已知函数 $f(x)$，求 $a$ 的取值范围。",
            "shards": [
                {"shard_id": "S2", "segments": ["f(x)=e^{x-1}+ax."], "shard": "$f(x)=e^{x-1}+ax.$"},
                {"shard_id": "S3", "segments": ["若f(x)>=0恒成立"], "shard": "若 $f(x)>=0$ 恒成立。"},
            ],
        },
    }
    cert_f = {
        "status": "VERIFIABLE",
        "verified_mechanism": "TypeB_Extremum",
        "concat_equivalent_to_original": True,
        "redundant_shards": [],
        "variables": {"a": "real", "x": "real"},
        "functions": {"f": {"args": "x", "expr": "exp(x-1) + a*x"}},
        "shard_constraints": {
            "S1": [],
            "S2": ["Eq(f(0), exp(-1))", "Eq(f(1), 1 + a)"],
            "S3": ["IntervalMin(f(x), x, -oo, oo) >= 0"],
        },
        "target_expr": "a",
        "full_answer_check": {
            "answer_kind": "interval",
            "declared_interval_sympy": "Interval(-1, 0)",
            "witness": {"values": {"a": -1}},
        },
        "leave_one_out": {
            "S2": {
                "relaxed_witness": {
                    "values": {"a": 5},
                    "functions": {"f": {"args": "x", "expr": "1"}},
                }
            },
            "S3": {"relaxed_witness": {"values": {"a": -2}}},
        },
    }
    r_f = verify_certificate_record(cand_f, cert_f, manifest_h, cfg_h)
    assert r_f["result"]["verify_status"] == "VERIFY_PASS", r_f

    # 6. Symbolic SeqEq + Sum test with function-swap LOO witnesses (empty values={})
    cand_g = {
        "item_id": "t-g",
        "question": "已知数列a_n满足a_1=1，a_{n+1}=2a_n，b_n=a_n+2，求b_n的前n项和。",
        "question_sha256": "1" * 64,
        "macro_subfield": "G3_Seq_Alg_Ineq",
        "provisional_mechanism": "TypeA_UniqueValue",
        "parsed_output": {
            "status": "CANDIDATE",
            "initial_segments": ["已知数列a_n满足", "求b_n的前n项和。"],
            "initial_shard": "已知数列 $\\{a_n\\}$ 满足，求 $\\{b_n\\}$ 的前 $n$ 项和。",
            "shards": [
                {"shard_id": "S2", "segments": ["a_1=1，"], "shard": "$a_1=1$。"},
                {"shard_id": "S3", "segments": ["a_{n+1}=2a_n，"], "shard": "$a_{n+1}=2a_n$。"},
                {"shard_id": "S4", "segments": ["b_n=a_n+2，"], "shard": "$b_n=a_n+2$。"},
            ],
        },
    }
    cert_g = {
        "status": "VERIFIABLE",
        "verified_mechanism": "TypeA_UniqueValue",
        "concat_equivalent_to_original": True,
        "redundant_shards": [],
        "variables": {"n": "positive_integer", "k": "positive_integer"},
        "functions": {
            "a": {"args": "n", "expr": "2**(n-1)"},
            "b": {"args": "n", "expr": "2**(n-1) + 2"},
        },
        "shard_constraints": {
            "S1": [],
            "S2": ["Eq(a(1), 1)"],
            "S3": ["And(SeqEq(a(n+1), 2*a(n), n), ForAllInt(a(n) >= 1, n))"],
            "S4": ["SeqEq(b(n), a(n) + 2, n)"],
        },
        "target_expr": "Sum(b(k), (k, 1, n))",
        "full_answer_check": {
            "answer_kind": "expression",
            "declared_value_sympy": "2**n + 2*n - 1",
            "witness": {"values": {}, "functions": {"a": {"args": "n", "expr": "2**(n-1)"}, "b": {"args": "n", "expr": "2**(n-1) + 2"}}},
        },
        "leave_one_out": {
            "S2": {"witness_2": {"values": {}, "functions": {"a": {"args": "n", "expr": "2**n"}, "b": {"args": "n", "expr": "2**n + 2"}}}},
            "S3": {"witness_2": {"values": {}, "functions": {"a": {"args": "n", "expr": "1"}, "b": {"args": "n", "expr": "3"}}}},
            "S4": {"witness_2": {"values": {}, "functions": {"b": {"args": "n", "expr": "2**(n-1)"}}}},
        },
    }
    r_g = verify_certificate_record(cand_g, cert_g, manifest_h, cfg_h)
    assert r_g["result"]["verify_status"] == "VERIFY_PASS", r_g

    # 7. TypeA unbounded_witness test (removing domain shard S4 breaks global extremum existence)
    cand_h = {
        "item_id": "t-h",
        "question": "已知y=x，z=1，x<=1，求x+z的最大值。",
        "question_sha256": "2" * 64,
        "macro_subfield": "G1_Func_Calc_Trig",
        "provisional_mechanism": "TypeA_UniqueValue",
        "parsed_output": {
            "status": "CANDIDATE",
            "initial_segments": ["求x+z的最大值。"],
            "initial_shard": "求 $x+z$ 的最大值。",
            "shards": [
                {"shard_id": "S2", "segments": ["已知y=x，"], "shard": "已知 $y=x$。"},
                {"shard_id": "S3", "segments": ["z=1，"], "shard": "$z=1$。"},
                {"shard_id": "S4", "segments": ["x<=1，"], "shard": "$x<=1$。"},
            ],
        },
    }
    cert_h = {
        "status": "VERIFIABLE",
        "verified_mechanism": "TypeA_UniqueValue",
        "concat_equivalent_to_original": True,
        "redundant_shards": [],
        "variables": {"x": "real", "y": "real", "z": "real"},
        "functions": {"f": {"args": "y,z", "expr": "y + z"}},
        "shard_constraints": {
            "S1": [],
            "S2": ["Eq(y, x)"],
            "S3": ["Eq(z, 1)"],
            "S4": ["Le(x, 1)"],
        },
        "target_expr": "f(y, z)",
        "full_answer_check": {
            "answer_kind": "max",
            "declared_value_sympy": "2",
            "witness": {"values": {"x": 1, "y": 1, "z": 1}},
        },
        "leave_one_out": {
            "S2": {"witness_2": {"values": {"x": 1, "y": 2, "z": 1}}, "witness_1": {"values": {"x": 1, "y": 1, "z": 1}}},
            "S3": {"witness_2": {"values": {"x": 1, "y": 1, "z": 5}}},
            "S4": {"unbounded_witness": {"values": {"x": 5, "y": 5, "z": 1}, "violation_expr": "f(y, z) > 2"}},
        },
    }
    r_h = verify_certificate_record(cand_h, cert_h, manifest_h, cfg_h)
    assert r_h["result"]["verify_status"] == "VERIFY_PASS", r_h
    print("All Stage 2B verifier core self-tests PASSED.")


if __name__ == "__main__":
    if "--self-test" in sys.argv:
        _run_self_tests()
