#!/usr/bin/env python3
"""Deterministic SymPy final-answer equivalence grader for Experiments 2–4.

Combines fast regex / \\boxed{} extraction with Stage 2B verified reference SymPy targets
(declared_value_sympy / declared_interval_sympy) and official `final_answer` normalization.
Escalates only genuinely ambiguous surface forms (`AMBIGUOUS_NEED_LLM`) to Dual-Judge.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

import sympy as sp
from sympy.core.relational import Relational
from sympy.logic.boolalg import Boolean

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2B_DIR = SCRIPT_DIR.parent / "stage2b_verification"
for p in (SCRIPT_DIR, STAGE2B_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from protocol_utils import extract_boxed_answers  # noqa: E402
from stage2b_verifier_core import _decide_sympy_bool, _make_symbol, parse_symbolic  # noqa: E402


ABSTAIN_PATTERNS = (
    r"无法确定",
    r"条件不足",
    r"缺少.{0,12}条件",
    r"信息不足",
    r"不能唯一确定",
    r"不能确定",
    r"未能完成思考",
    r"未生成可见回答",
)


def _strip_outer_math(text: str) -> str:
    s = text.strip()
    if s.startswith("$$") and s.endswith("$$") and len(s) >= 4:
        s = s[2:-2].strip()
    elif s.startswith("$") and s.endswith("$") and len(s) >= 2:
        s = s[1:-1].strip()
    s = s.rstrip("。.;；,， ")
    return s


def _extract_brace_group(s: str, open_idx: int) -> tuple[str, int] | None:
    if open_idx >= len(s) or s[open_idx] != "{":
        return None
    depth = 0
    for cur in range(open_idx, len(s)):
        if s[cur] == "{":
            depth += 1
        elif s[cur] == "}":
            depth -= 1
            if depth == 0:
                return s[open_idx + 1 : cur], cur + 1
    return None


def _replace_balanced_latex(s: str) -> str:
    """Recursively expand balanced \\frac/\\dfrac/\\tfrac, \\sqrt, _{...}, and ^{...}."""
    for _ in range(12):
        changed = False
        # 1. \frac{a}{b}, \dfrac{a}{b}, \tfrac{a}{b}
        m_frac = re.search(r"\\[dt]?frac\s*\{", s)
        if m_frac:
            b1 = _extract_brace_group(s, m_frac.end() - 1)
            if b1 is not None:
                num_s, pos2 = b1
                while pos2 < len(s) and s[pos2].isspace():
                    pos2 += 1
                b2 = _extract_brace_group(s, pos2)
                if b2 is not None:
                    den_s, end_pos = b2
                    s = f"{s[:m_frac.start()]}(({num_s})/({den_s})){s[end_pos:]}"
                    changed = True
                    continue
        # 2. \sqrt[k]{x} or \sqrt{x}
        m_sqrt = re.search(r"\\sqrt\s*(?:\[([^\[\]]+)\]\s*)?\{", s)
        if m_sqrt:
            b1 = _extract_brace_group(s, m_sqrt.end() - 1)
            if b1 is not None:
                rad_s, end_pos = b1
                deg = m_sqrt.group(1)
                repl = f"(({rad_s})**(1/({deg})))" if deg else f"sqrt({rad_s})"
                s = f"{s[:m_sqrt.start()]}{repl}{s[end_pos:]}"
                changed = True
                continue
        # 3. Subscripts a_{n+1} / a_{n} -> a_n when simple identifier
        m_sub = re.search(r"([A-Za-z]+)_\{([A-Za-z0-9]+)\}", s)
        if m_sub:
            s = f"{s[:m_sub.start()]}{m_sub.group(1)}_{m_sub.group(2)}{s[m_sub.end():]}"
            changed = True
            continue
        # 4. Exponents ^{...} -> **(...)
        m_exp = re.search(r"\^\s*\{", s)
        if m_exp:
            b1 = _extract_brace_group(s, m_exp.end() - 1)
            if b1 is not None:
                exp_s, end_pos = b1
                s = f"{s[:m_exp.start()]}**({exp_s}){s[end_pos:]}"
                changed = True
                continue
        if not changed:
            break
    # Simple single-token exponent x^2 or 2^n -> **2 / **n
    s = re.sub(r"\^\s*([A-Za-z0-9]+)", r"**(\1)", s)
    # Unwrap redundant braces around single symbols like {a_n} -> a_n
    s = re.sub(r"\{\s*([A-Za-z_][A-Za-z0-9_]*)\s*\}", r"\1", s)
    return s


def latex_to_sympy_candidate(raw: str) -> str:
    """Convert common Chinese math / LaTeX surface strings cleanly into SymPy syntax."""
    s = _strip_outer_math(raw)
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "").replace("\\ ", " ")
    s = s.replace("，", ",").replace("＋", "+").replace("－", "-").replace("＝", "=").replace("（", "(").replace("）", ")")
    s = s.replace("\\cdot", "*").replace("\\times", "*").replace("×", "*").replace("÷", "/")
    s = s.replace("π", "pi").replace("\\pi", "pi")
    s = s.replace("+\\infty", "oo").replace("-\\infty", "-oo").replace("\\infty", "oo")
    s = s.replace("+∞", "oo").replace("-∞", "-oo").replace("∞", "oo")
    s = _replace_balanced_latex(s)
    fn_map = {
        "ln": "log",
        "arcsin": "asin",
        "arccos": "acos",
        "arctan": "atan",
    }
    s = re.sub(
        r"\\(sin|cos|tan|cot|sec|csc|ln|log|exp|arcsin|arccos|arctan)\b",
        lambda m: " " + fn_map.get(m.group(1), m.group(1)),
        s,
    )
    # Parenthesize bare function arguments like "log 2" or "sin A" -> "log(2)" / "sin(A)"
    s = re.sub(
        r"\b(sin|cos|tan|cot|sec|csc|log|exp|asin|acos|atan)\s+([0-9]+(?:\.[0-9]+)?|[A-Za-z_][A-Za-z0-9_]*)\b",
        r"\1(\2)",
        s,
    )
    s = s.replace("\\cup", "|").replace("∪", "|")
    # Strip benign LHS label e.g. "x = 5", "a_n = 2*n-1", "S(2) = 32" while preserving curve equations like "y**(2) = x".
    if "=" in s and not any(op in s for op in ("==", "<=", ">=", "!=")):
        parts = s.split("=")
        if len(parts) == 2:
            lhs = parts[0].strip()
            if re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*(?:\s*\(\s*[A-Za-z0-9_,\s]+\s*\))?", lhs):
                s = parts[1].strip()
    return s.strip()


def _parse_interval_or_union(raw: str, sym_map: dict[str, sp.Symbol]) -> sp.Set | None:
    s = _strip_outer_math(raw)
    s = s.replace("，", ",").replace("＋", "+").replace("－", "-").replace("（", "(").replace("）", ")")
    s = s.replace("\\left", "").replace("\\right", "").replace("\\,", "")
    s = s.replace("+\\infty", "oo").replace("-\\infty", "-oo").replace("\\infty", "oo")
    s = s.replace("+∞", "oo").replace("-∞", "-oo").replace("∞", "oo")
    s = re.sub(r"\\(?:mathbf|mathbb)\{R\}|R(?=\s*$)", "Interval(-oo, oo)", s)
    parts = re.split(r"\\cup|∪|\band\b|或", s)
    sets: list[sp.Set] = []
    for p in parts:
        p = p.strip()
        if not p:
            continue
        if p.startswith("{") and p.endswith("}"):
            elems = [parse_symbolic(latex_to_sympy_candidate(x), sym_map) for x in p[1:-1].split(",") if x.strip()]
            sets.append(sp.FiniteSet(*elems))
            continue
        m = re.fullmatch(r"([\[(])\s*([^,]+)\s*,\s*([^\]\)]+)\s*([\])])", p)
        if m:
            lb, a_str, b_str, rb = m.groups()
            lo = parse_symbolic(latex_to_sympy_candidate(a_str), sym_map)
            hi = parse_symbolic(latex_to_sympy_candidate(b_str), sym_map)
            sets.append(sp.Interval(lo, hi, left_open=(lb == "("), right_open=(rb == ")")))
            continue
        return None
    if not sets:
        return None
    return sp.Union(*sets) if len(sets) > 1 else sets[0]


# Unconditional tentative/unverified-scope/retraction cues anywhere in the reply force LLM arbitration.
STRONG_NONFINAL_OR_TENTATIVE_RE = re.compile(
    r"(?:未证实|尚未证实|仅在|只考虑|暂且|暂取|暂定|姑且|先假设|先设|假设|假定|撤回|推翻前面|更正先前|修正先前|此前.{0,8}(?:有误|不对|错误)|"
    r"\bunverified\b|\btentative(?:ly)?\b|\bprovisional(?:ly)?\b|\bassum(?:e|ing)\b|\bsuppos(?:e|ing)\b|"
    r"\bunder (?:the )?assumption\b|\bonly in (?:the )?case\b|\bonly when\b|\bretract(?:ing|ed)?\b|\bwithdraw(?:ing|n)?\b)",
    re.IGNORECASE,
)

# Conditional / case-restriction cues checked over the local multi-sentence context around the answer.
CONDITIONAL_MARKER_RE = re.compile(
    r"(?:若|如果|假如|假设|当[^，。；;\n]{1,30}时|在[^，。；;\n]{0,20}(?:前提|假设|条件|情形|情况)下|"
    r"\bif\b|\bwhen\b|\bsuppos(?:e|ing)\b|\bassum(?:e|ing)\b|\bprovided(?: that)?\b|\bunless\b|\bgiven that\b|\bfor (?:the )?case\b|\bin case\b)",
    re.IGNORECASE,
)


def _split_sentence_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    start = 0
    for m in re.finditer(r"[。；;\n]+|(?<!\d)\.(?!\d)", text):
        end = m.start()
        if end > start:
            spans.append((start, end, text[start:end].strip()))
        start = m.end()
    if start < len(text):
        spans.append((start, len(text), text[start:].strip()))
    return [sp_item for sp_item in spans if sp_item[2]]


def _answer_context_window(text: str, pos: int, prev_sentences: int = 2) -> str:
    """Return up to `prev_sentences` preceding sentences plus the answer sentence and trailing context."""
    spans = _split_sentence_spans(text)
    if not spans:
        return text
    hit_idx = len(spans) - 1
    for i, (s_start, s_end, _) in enumerate(spans):
        if s_start <= pos <= s_end:
            hit_idx = i
            break
        if pos < s_start:
            hit_idx = max(0, i - 1)
            break
    lo_idx = max(0, hit_idx - prev_sentences)
    ctx_start = spans[lo_idx][0]
    return text[ctx_start:].strip()


def _extract_candidate_answer_text(visible_reply: str) -> tuple[str | None, str]:
    """Extract unambiguous final candidate answer; escalate cross-sentence conditionals, branching, or retraction to LLM."""
    text = (visible_reply or "").strip()
    if not text:
        return None, "EMPTY_REPLY"
    if STRONG_NONFINAL_OR_TENTATIVE_RE.search(text):
        return None, "CONDITIONAL_OR_RETRACTED_CONTEXT"
    boxes = extract_boxed_answers(text)
    if boxes:
        last_box = boxes[-1]
        last_pos = text.rfind(r"\boxed{" + last_box)
        if last_pos == -1:
            last_pos = text.rfind(r"\boxed{")
        # 1. Trailing abstention/contradiction after the last box -> ambiguous
        trailing = text[max(0, last_pos) :]
        if any(re.search(pat, trailing) for pat in ABSTAIN_PATTERNS) and not any(re.search(pat, last_box) for pat in ABSTAIN_PATTERNS):
            return None, "CONTRADICTORY_TRAILING_ABSTAIN"
        # 2. Multiple boxes with distinct surface strings -> escalate (branching/retraction/multi-answer)
        distinct_norm = {latex_to_sympy_candidate(b) for b in boxes}
        if len(distinct_norm) > 1:
            return None, "MULTIPLE_DISTINCT_BOXES"
        # 3. Cross-sentence context window around last box contains conditional qualification -> escalate
        ctx_window = _answer_context_window(text, max(0, last_pos), prev_sentences=2)
        if CONDITIONAL_MARKER_RE.search(ctx_window):
            return None, "CONDITIONAL_BOX_CONTEXT"
        return last_box, "BOXED"
    # Check explicit abstention when no box is present.
    tail = text[-300:]
    if any(re.search(pat, tail) for pat in ABSTAIN_PATTERNS):
        return None, "ABSTENTION_TEXT"
    # Fallback: final line marker e.g. "最终答案：..." or "答案为 ..."
    matches = list(re.finditer(r"(?:最终)?答案(?:为|是|：|:)\s*([^\n。；;]+)", text))
    if len(matches) > 1:
        return None, "MULTIPLE_ANSWER_PHRASES"
    if len(matches) == 1:
        ctx_window = _answer_context_window(text, matches[0].start(), prev_sentences=2)
        if CONDITIONAL_MARKER_RE.search(ctx_window):
            return None, "CONDITIONAL_PHRASE_CONTEXT"
        return matches[0].group(1).strip(), "ANSWER_PHRASE"
    return None, "NO_BOX_OR_PHRASE"


def grade_final_answer_sympy(
    visible_reply: str,
    final_answer: str,
    cert_payload: dict[str, Any] | None = None,
    empty_truncated: bool = False,
) -> dict[str, Any]:
    """Deterministically grade a final reply against the certified SymPy reference answer.

    Returns:
      status in {"CORRECT", "INCORRECT_ABSTAIN", "INCORRECT_EXACT", "AMBIGUOUS_NEED_LLM"}
      with y_L, y_U bounds in {0, 1}.
    """
    if empty_truncated or not (visible_reply or "").strip():
        return {"status": "INCORRECT_ABSTAIN", "y_L": 0, "y_U": 0, "extracted": None, "reason": "empty_or_truncated"}

    cand_raw, extract_kind = _extract_candidate_answer_text(visible_reply)
    if cand_raw is None:
        if extract_kind == "ABSTENTION_TEXT":
            return {"status": "INCORRECT_ABSTAIN", "y_L": 0, "y_U": 0, "extracted": None, "reason": "abstention"}
        return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": None, "reason": extract_kind}

    # If the extracted box itself explicitly says "无法确定"/"无解"/"条件不足", treat as abstention.
    if any(re.search(pat, cand_raw) for pat in ABSTAIN_PATTERNS):
        return {"status": "INCORRECT_ABSTAIN", "y_L": 0, "y_U": 0, "extracted": cand_raw, "reason": "boxed_abstention"}

    full_check = (cert_payload or {}).get("full_answer_check") or {}
    ans_kind = full_check.get("answer_kind", "scalar")
    var_specs = (cert_payload or {}).get("variables") or {}
    sym_map: dict[str, sp.Symbol] = {vn: _make_symbol(vn, vt) for vn, vt in var_specs.items()}

    # Detect if official final_answer is itself an implicit/Cartesian curve equation (e.g. x^2/4 + y^2 = 1).
    ref_eq_obj: Any = None
    if final_answer:
        try:
            cand_ref = parse_symbolic(latex_to_sympy_candidate(final_answer), sym_map)
            if isinstance(cand_ref, sp.Eq):
                ref_eq_obj = cand_ref
        except Exception:
            ref_eq_obj = None

    if ref_eq_obj is not None:
        try:
            pred_eq = parse_symbolic(latex_to_sympy_candidate(cand_raw), sym_map)
            if isinstance(pred_eq, sp.Eq):
                f_ref = sp.expand(ref_eq_obj.lhs - ref_eq_obj.rhs)
                f_pred = sp.expand(pred_eq.lhs - pred_eq.rhs)
                if f_ref != 0 and f_pred != 0:
                    ratio = sp.simplify(sp.cancel(f_pred / f_ref))
                    if not ratio.free_symbols and ratio != 0 and ratio.is_finite is not False:
                        return {"status": "CORRECT", "y_L": 1, "y_U": 1, "extracted": cand_raw, "reason": f"sympy_curve_eq({extract_kind})"}
                    if f_pred.free_symbols and (f_pred.free_symbols <= f_ref.free_symbols):
                        return {"status": "INCORRECT_EXACT", "y_L": 0, "y_U": 0, "extracted": cand_raw, "reason": f"sympy_curve_ne:{pred_eq}!={ref_eq_obj}"}
            return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": cand_raw, "reason": "curve_eq_need_llm"}
        except Exception as exc:
            return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": cand_raw, "reason": f"curve_parse_err:{exc}"}

    # Build reference SymPy object (prefer certified expression/interval, fallback to raw final_answer).
    ref_obj: Any = None
    try:
        if ans_kind in ("interval", "set"):
            ref_str = str(full_check.get("declared_interval_sympy") or full_check.get("declared_value_sympy") or "").strip()
            if ref_str:
                ref_obj = parse_symbolic(ref_str, sym_map)
            else:
                ref_obj = _parse_interval_or_union(final_answer, sym_map)
        else:
            ref_str = str(full_check.get("declared_value_sympy") or "").strip()
            ref_obj = parse_symbolic(ref_str, sym_map) if ref_str else parse_symbolic(latex_to_sympy_candidate(final_answer), sym_map)
    except Exception:
        ref_obj = None

    if ref_obj is None:
        return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": cand_raw, "reason": "ref_parse_failed"}

    # Try interval/set comparison first when ref_obj is a SymPy Set.
    try:
        if isinstance(ref_obj, sp.Set):
            pred_set = _parse_interval_or_union(cand_raw, sym_map)
            if pred_set is None:
                pred_set = parse_symbolic(latex_to_sympy_candidate(cand_raw), sym_map)
            if isinstance(pred_set, sp.Set):
                eq = _decide_sympy_bool(sp.Eq(pred_set, ref_obj))
                if eq is True:
                    return {"status": "CORRECT", "y_L": 1, "y_U": 1, "extracted": cand_raw, "reason": f"sympy_set_eq({extract_kind})"}
                if eq is False:
                    return {"status": "INCORRECT_EXACT", "y_L": 0, "y_U": 0, "extracted": cand_raw, "reason": f"sympy_set_ne:{pred_set}!={ref_obj}"}
            return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": cand_raw, "reason": "set_not_decided"}

        # Scalar / expression / extremum comparison.
        pred_expr = parse_symbolic(latex_to_sympy_candidate(cand_raw), sym_map)
        if isinstance(pred_expr, (Relational, Boolean)) and not isinstance(ref_obj, (Relational, Boolean)):
            return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": cand_raw, "reason": "relational_vs_expr_need_llm"}
        eq = _decide_sympy_bool(sp.Eq(pred_expr, ref_obj))
        if eq is True:
            return {"status": "CORRECT", "y_L": 1, "y_U": 1, "extracted": cand_raw, "reason": f"sympy_expr_eq({extract_kind})"}
        if eq is False:
            return {"status": "INCORRECT_EXACT", "y_L": 0, "y_U": 0, "extracted": cand_raw, "reason": f"sympy_expr_ne:{pred_expr}!={ref_obj}"}
    except Exception as exc:
        return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": cand_raw, "reason": f"sympy_parse_err:{exc}"}

    return {"status": "AMBIGUOUS_NEED_LLM", "y_L": 0, "y_U": 1, "extracted": cand_raw, "reason": "sympy_undecided"}


def _run_self_tests() -> None:
    cert_scalar = {
        "variables": {},
        "full_answer_check": {"answer_kind": "scalar", "declared_value_sympy": "sqrt(5)/5"},
    }
    assert grade_final_answer_sympy(r"由正弦定理得 $\boxed{\frac{1}{\sqrt{5}}}$。", r"$\frac{\sqrt{5}}{5}$", cert_scalar)["status"] == "CORRECT"
    assert grade_final_answer_sympy(r"答案是 $\boxed{\frac{2}{5}}$。", r"$\frac{\sqrt{5}}{5}$", cert_scalar)["status"] == "INCORRECT_EXACT"

    cert_int = {
        "variables": {"a": "real"},
        "full_answer_check": {"answer_kind": "interval", "declared_interval_sympy": "Union(IntervalOpen(-oo, 0), Interval(1, oo))"},
    }
    assert grade_final_answer_sympy(r"综上，$a$ 的取值范围为 $\boxed{(-\infty, 0) \cup [1, +\infty)}$。", "", cert_int)["status"] == "CORRECT"
    assert grade_final_answer_sympy(r"综上，取值范围为 $\boxed{(0, 1)}$。", "", cert_int)["status"] == "INCORRECT_EXACT"

    # Truncated empty reply or explicit abstention -> INCORRECT_ABSTAIN (never CORRECT, never Wait-confused)
    assert grade_final_answer_sympy("", "5", cert_scalar, empty_truncated=True)["status"] == "INCORRECT_ABSTAIN"
    assert grade_final_answer_sympy("即使给全了条件依然条件不足，无法确定答案。", "5", cert_scalar)["status"] == "INCORRECT_ABSTAIN"

    # Unboxed narrative reply with unclear phrase -> escalates cleanly to AMBIGUOUS_NEED_LLM
    amb = grade_final_answer_sympy("综上所述这道题可以通过几何法求得最终点坐标在第一象限。", "5", cert_scalar)
    assert amb["status"] == "AMBIGUOUS_NEED_LLM"

    # Multi-box branching / retraction / conditional final box / trailing abstention -> MUST escalate to LLM
    multi_b = grade_final_answer_sympy(r"若 $a>0$ 得 $\boxed{1}$，若 $a<0$ 得 $\boxed{\frac{\sqrt{5}}{5}}$。", r"$\frac{\sqrt{5}}{5}$", cert_scalar)
    assert multi_b["status"] == "AMBIGUOUS_NEED_LLM" and multi_b["reason"] == "MULTIPLE_DISTINCT_BOXES"
    cond_b = grade_final_answer_sympy(r"当 $x>1$ 时，最终答案为 $\boxed{\frac{\sqrt{5}}{5}}$。", r"$\frac{\sqrt{5}}{5}$", cert_scalar)
    assert cond_b["status"] == "AMBIGUOUS_NEED_LLM" and cond_b["reason"] == "CONDITIONAL_BOX_CONTEXT"
    trail_abs = grade_final_answer_sympy(r"初步可得 $\boxed{\frac{\sqrt{5}}{5}}$，但实际上缺少约束仍无法确定。", r"$\frac{\sqrt{5}}{5}$", cert_scalar)
    assert trail_abs["status"] == "AMBIGUOUS_NEED_LLM" and trail_abs["reason"] == "CONTRADICTORY_TRAILING_ABSTAIN"

    # Cross-sentence Chinese/English conditional framing & retraction -> MUST escalate to LLM (never false CORRECT)
    cross_sent = grade_final_answer_sympy("仅在未证实的假设 $a=0$ 下讨论。\n答案为 $\\boxed{5}$。", "5", {"variables": {}, "full_answer_check": {"answer_kind": "scalar", "declared_value_sympy": "5"}})
    assert cross_sent["status"] == "AMBIGUOUS_NEED_LLM"
    eng_cond = grade_final_answer_sympy("Assuming $a=0$ holds.\nTherefore the answer is $\\boxed{5}$.", "5", {"variables": {}, "full_answer_check": {"answer_kind": "scalar", "declared_value_sympy": "5"}})
    assert eng_cond["status"] == "AMBIGUOUS_NEED_LLM"
    eng_retract = grade_final_answer_sympy("Retracting my prior claim; answer is $\\boxed{5}$.", "5", {"variables": {}, "full_answer_check": {"answer_kind": "scalar", "declared_value_sympy": "5"}})
    assert eng_retract["status"] == "AMBIGUOUS_NEED_LLM"

    # Curve equations, LHS assignments, nested \frac{9^{n+1}...}, n\ln 2, and full-width comma intervals
    cert_curve = {"variables": {"x": "real", "y": "real"}, "full_answer_check": {"answer_kind": "scalar", "declared_value_sympy": "x**2/4 + y**2"}}
    assert grade_final_answer_sympy(r"轨迹方程为 $\boxed{\frac{x^2}{4}+y^2=1}$。", r"$\frac{x^2}{4}+y^2=1$", cert_curve)["status"] == "CORRECT"
    assert grade_final_answer_sympy(r"化简得 $\boxed{x^2+4y^2-4=0}$。", r"$\frac{x^2}{4}+y^2=1$", cert_curve)["status"] == "CORRECT"
    assert grade_final_answer_sympy(r"方程为 $\boxed{\frac{x^2}{4}-y^2=1}$。", r"$\frac{x^2}{4}+y^2=1$", cert_curve)["status"] == "INCORRECT_EXACT"

    cert_seq = {"variables": {"n": "integer"}, "full_answer_check": {"answer_kind": "expression", "declared_value_sympy": "n*log(2)"}}
    assert grade_final_answer_sympy(r"通项公式为 $\boxed{a_n=n\ln 2}$。", r"a_n=n\ln 2", cert_seq)["status"] == "CORRECT"
    cert_s2 = {"variables": {}, "full_answer_check": {"answer_kind": "scalar", "declared_value_sympy": "32"}}
    assert grade_final_answer_sympy(r"解得 $\boxed{S(2)=32}$。", r"$S (2)=32$", cert_s2)["status"] == "CORRECT"
    cert_nested = {"variables": {"n": "integer"}, "full_answer_check": {"answer_kind": "expression", "declared_value_sympy": "(9**(n+1) - 8*n**2 - 9)/8"}}
    assert grade_final_answer_sympy(r"前n项和为 $\boxed{\frac{9^{n+1}-8n^2-9}{8}}$。", r"$\frac{9^{n+1}-8n^2-9}{8}$", cert_nested)["status"] == "CORRECT"
    cert_fw_int = {"variables": {"a": "real"}, "full_answer_check": {"answer_kind": "interval", "declared_interval_sympy": "Interval(-1, oo)"}}
    assert grade_final_answer_sympy(r"取值范围是 $\boxed{[-1，+\infty )}$。", r"$[-1，+\infty )$", cert_fw_int)["status"] == "CORRECT"

    print("sympy_answer_grader.py self-tests PASSED.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        _run_self_tests()
