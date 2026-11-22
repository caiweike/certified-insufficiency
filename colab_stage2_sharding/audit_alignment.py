#!/usr/bin/env python3
"""Deterministic post-processing alignment auditor for Math-LiC Stage 2A outputs."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


VALID_REASON_CODES = {
    "TOO_FEW_SHARDS",
    "TOO_MANY_SHARDS",
    "COUPLED_DEFINITION_CONSTRAINT",
    "ATOMIC_FORMULA_UNSPLITTABLY_BOUND",
    "AMBIGUOUS_TARGET_SEPARATION",
    "OTHER",
}

SAFE_GLUE_WORDS = {
    "已知",
    "设",
    "若",
    "当",
    "时",
    "则",
    "且",
    "其中",
    "满足",
    "有",
    "得",
    "在",
    "中",
    "并",
    "又",
    "而",
    "即",
    "的",
    "与",
    "和",
}

# Trailing punctuation that may legitimately turn into sentence-ending punctuation inside a shard.
TRAILING_PUNCT_RE = re.compile(r"[.,;:!?，。；：！？、．\s]+$")
SAFE_UNCOVERED_CHAR_RE = re.compile(r"^[\s\$\.,;:，。；：、．]*$")
UNESCAPED_DOLLAR_RE = re.compile(r"(?<!\\)\$")
CJK_CLASS = r"\u4e00-\u9fff\u3000-\u303f\uff00-\uffef"
CJK_LEFT_SPACE_RE = re.compile(rf"(?<=[{CJK_CLASS}])\s+")
CJK_RIGHT_SPACE_RE = re.compile(rf"\s+(?=[{CJK_CLASS}])")
CJK_LEFT_COMMA_RE = re.compile(rf"(?<=[{CJK_CLASS}]),")
CJK_RIGHT_COMMA_RE = re.compile(rf",(?=[{CJK_CLASS}])")
CJK_LEFT_SEMI_RE = re.compile(rf"(?<=[{CJK_CLASS}]);")
CJK_RIGHT_SEMI_RE = re.compile(rf";(?=[{CJK_CLASS}])")
SPACE_BEFORE_LATEX_CMD_RE = re.compile(r"(?<=[A-Za-z0-9}])\s+(?=\\[A-Za-z])")
SAFE_HTML_TAG_RE = re.compile(r"</?(?:sub|sup|i|b|em|strong|br)\s*/?>", re.IGNORECASE)
RELATIONAL_LEAK_RE = re.compile(r"(=|<|>|\\le\b|\\ge\b|\\leq\b|\\geq\b|\\ne\b|\\neq\b|\\in\b|≤|≥|≠|∈)")


def strip_trailing_punct(text: str) -> str:
    return TRAILING_PUNCT_RE.sub("", text.strip())


def normalize_for_formula_check(text: str) -> str:
    """Normalize only non-semantic math delimiters ($), fullwidth '=', newlines, CJK-adjacent spacing,
    CJK-adjacent commas/semicolons, LaTeX brace escapes (\\{ -> {), and space before \\cmd.

    Crucially preserves ALL whitespace and punctuation inside ASCII/LaTeX expressions
    (e.g., '2 x+1' != '2x+1', 'sin A' != 'sinA', 'A,B,C' stays 'A,B,C').
    """
    out = text.replace("\r\n", " ").replace("\n", " ")
    out = out.replace("＝", "=")
    out = UNESCAPED_DOLLAR_RE.sub("", out)
    out = out.replace(r"\{", "{").replace(r"\}", "}")
    out = SPACE_BEFORE_LATEX_CMD_RE.sub("", out)
    out = CJK_LEFT_SPACE_RE.sub("", out)
    out = CJK_RIGHT_SPACE_RE.sub("", out)
    out = CJK_LEFT_COMMA_RE.sub("，", out)
    out = CJK_RIGHT_COMMA_RE.sub("，", out)
    out = CJK_LEFT_SEMI_RE.sub("；", out)
    out = CJK_RIGHT_SEMI_RE.sub("；", out)
    return out.strip()


HANZI_RUN_RE = re.compile(r"[\u4e00-\u9fff]+")
POINT_COORDS_RE = re.compile(r"^([A-Z])(\([^()]+\))$")
IGNORABLE_HANZI_GLUE = {"已知", "设", "且", "的", "为", "是", "有", "得"}
LEADING_HANZI_GLUE_RE = re.compile(r"^(?:已知|设|若|且|其中|又|则)+")


def check_segment_preserved_in_shard(seg_core: str, norm_shard: str, search_pos: int) -> tuple[bool, int, str]:
    """Check that seg_core's math expressions/numbers and non-glue Hanzi content appear verbatim in norm_shard."""
    norm_core = normalize_for_formula_check(seg_core)
    if not norm_core:
        return True, search_pos, norm_core

    found_at = norm_shard.find(norm_core, search_pos)
    if found_at != -1:
        return True, found_at + len(norm_core), norm_core

    # Fallback: allow inserting clarifying nouns/scaffolding (e.g., "抛物线 C: y^2=4x" -> "抛物线 C 的方程为 y^2=4x",
    # "点 A(-2,0)" -> "点 A 的坐标为 (-2,0)", "在 G 的焦点" -> "位于抛物线G的焦点"),
    # while STILL requiring EVERY math expression, number, and non-glue Hanzi stem in order.
    tokens = re.split(r"([\u4e00-\u9fff]+)", norm_core)
    pos = search_pos
    for idx, tok in enumerate(tokens):
        if not tok:
            continue
        if HANZI_RUN_RE.fullmatch(tok):
            stripped_tok = LEADING_HANZI_GLUE_RE.sub("", tok) if idx <= 1 else tok
            sub_stems = [s for s in re.split(r"[的]", stripped_tok) if s and s not in IGNORABLE_HANZI_GLUE]
            for stem in sub_stems:
                if stem == "在":
                    p1 = norm_shard.find("在", pos)
                    p2 = norm_shard.find("位于", pos)
                    candidates = [p for p in (p1, p2) if p != -1]
                    if not candidates:
                        return False, search_pos, norm_core
                    best = min(candidates)
                    pos = best + (2 if best == p2 and p1 != p2 else 1)
                else:
                    found = norm_shard.find(stem, pos)
                    if found == -1:
                        return False, search_pos, norm_core
                    pos = found + len(stem)
        else:
            cleaned = tok.strip(" \t\r\n.,;，。；：、:")
            if not cleaned:
                continue
            found = norm_shard.find(cleaned, pos)
            if found != -1:
                pos = found + len(cleaned)
                continue
            # Allow separating point name and coordinate tuple ONLY when preceded by "点" (e.g., 点 A 的坐标为 (-2,0))
            m_pt = POINT_COORDS_RE.fullmatch(cleaned)
            prev_hanzi = tokens[idx - 1] if idx > 0 else ""
            if m_pt and "点" in prev_hanzi:
                p_name, p_coords = m_pt.group(1), m_pt.group(2)
                f1 = norm_shard.find(p_name, pos)
                if f1 != -1:
                    f2 = norm_shard.find(p_coords, f1 + len(p_name))
                    if f2 != -1:
                        pos = f2 + len(p_coords)
                        continue
            # Allow separating named object from its equation across ": " or "：" (e.g., 抛物线 C: y^2=4x -> 抛物线 C 的方程为 y^2=4x)
            m_colon = re.fullmatch(r"([A-Za-z]+)(?::\s+|：)(.+)", cleaned)
            if m_colon:
                obj_name, eq_body = m_colon.group(1), m_colon.group(2)
                f1 = norm_shard.find(obj_name, pos)
                if f1 != -1:
                    f2 = norm_shard.find(eq_body, f1 + len(obj_name))
                    if f2 != -1:
                        pos = f2 + len(eq_body)
                        continue
            return False, search_pos, norm_core

    return True, pos, norm_core


def find_canonical_spans(question: str, seg: str) -> list[tuple[int, int, str]]:
    """Locate `seg` in `question`, allowing safe delimiter/boundary snapping (only outer '$' or trailing punct/space)."""
    exact = find_all_occurrences(question, seg)
    out: list[tuple[int, int, str]] = [(s, s + len(seg), seg) for s in exact]
    if exact and not (seg.strip().startswith("$") or seg.strip().endswith("$")):
        return out

    # Build strictly non-semantic trimmed candidates (strip accidental outer '$' or trailing punct).
    candidates_str: list[str] = []
    stripped_dollar = seg.strip()
    variants = [
        stripped_dollar.lstrip("$"),
        stripped_dollar.rstrip("$"),
        stripped_dollar.strip("$"),
        strip_trailing_punct(stripped_dollar),
        strip_trailing_punct(stripped_dollar.lstrip("$")),
        strip_trailing_punct(stripped_dollar.strip("$")),
    ]
    for v in variants:
        v = v.strip()
        if v and len(v) >= 2 and v != seg and v not in candidates_str:
            candidates_str.append(v)

    for v in candidates_str:
        occs = find_all_occurrences(question, v)
        for s in occs:
            e = s + len(v)
            # Skip strictly contained subspans of an already collected match (prefer wider outer '$' match).
            if any(os <= s and e <= oe for os, oe, _ in out):
                continue
            out.append((s, e, question[s:e]))
        if out and not exact:
            return out
    if out:
        return out

    # Last fallback: regex allowing optional spacing ONLY at CJK/math boundaries and CJK commas/semis.
    base = strip_trailing_punct(stripped_dollar.strip("$")).strip()
    if len(base) >= 3:
        parts: list[str] = []
        for idx, ch in enumerate(base):
            prev_c = base[idx - 1] if idx > 0 else ""
            next_c = base[idx + 1] if idx + 1 < len(base) else ""
            cjk_neighbor = ("\u4e00" <= prev_c <= "\u9fff") or ("\u4e00" <= next_c <= "\u9fff")
            if ch in (",", "，") and cjk_neighbor:
                parts.append("[,，]")
            elif ch in (";", "；") and cjk_neighbor:
                parts.append("[;；]")
            elif ch == " " and (cjk_neighbor or prev_c == "$" or next_c == "$"):
                parts.append(r"\s*")
            elif "\u4e00" <= ch <= "\u9fff":
                # Allow optional $ or space adjacent to CJK boundaries
                parts.append(re.escape(ch) + r"(?:[\s\$]*)" if idx + 1 < len(base) and base[idx + 1] != " " and not ("\u4e00" <= base[idx + 1] <= "\u9fff") else re.escape(ch))
            else:
                parts.append(re.escape(ch))
        pattern = "".join(parts)
        try:
            matches = list(re.finditer(pattern, question))
        except re.error:
            matches = []
        if matches:
            return [(m.start(), m.end(), m.group(0)) for m in matches]

    return []


def count_unescaped_dollars(text: str) -> int:
    return len(UNESCAPED_DOLLAR_RE.findall(text))


def find_all_occurrences(haystack: str, needle: str) -> list[int]:
    if not needle:
        return []
    starts: list[int] = []
    pos = 0
    while True:
        idx = haystack.find(needle, pos)
        if idx == -1:
            break
        starts.append(idx)
        pos = idx + 1
    return starts


def resolve_non_overlapping_spans(
    question: str,
    shard_segments: list[tuple[str, list[str]]],
    max_steps: int = 20000,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, str]]]:
    """Find non-overlapping exact substring spans in `question` respecting per-shard left-to-right order."""
    errors: list[dict[str, str]] = []
    entries: list[dict[str, Any]] = []

    for shard_idx, (shard_id, segs) in enumerate(shard_segments):
        if not isinstance(segs, list) or len(segs) == 0:
            errors.append(
                {
                    "code": "EMPTY_SEGMENTS_LIST",
                    "message": f"{shard_id} segments list must be a non-empty list of strings.",
                }
            )
            continue
        for seg_idx, seg in enumerate(segs):
            if not isinstance(seg, str) or not seg:
                errors.append(
                    {
                        "code": "EMPTY_SEGMENT_STRING",
                        "message": f"{shard_id}.segments[{seg_idx}] is empty or not a string.",
                    }
                )
                continue
            occ_spans = find_canonical_spans(question, seg)
            if not occ_spans:
                errors.append(
                    {
                        "code": "SEGMENT_NOT_FOUND",
                        "message": f"{shard_id}.segments[{seg_idx}] is not an exact substring of question: {seg!r}",
                    }
                )
                continue
            entries.append(
                {
                    "shard_idx": shard_idx,
                    "shard_id": shard_id,
                    "seg_idx": seg_idx,
                    "text": seg,
                    "canonical_texts": {(s, e): txt for s, e, txt in occ_spans},
                    "length": len(seg),
                    "candidates": [(s, e) for s, e, _ in occ_spans],
                }
            )

    if errors:
        return None, errors

    # Solve most-constrained (fewest occurrences, longest segments) first to keep backtracking tiny.
    order = sorted(
        range(len(entries)),
        key=lambda i: (len(entries[i]["candidates"]), -entries[i]["length"], i),
    )

    assigned: list[tuple[int, int] | None] = [None] * len(entries)
    steps = 0

    def overlaps(span: tuple[int, int], used_idx: int) -> bool:
        other = assigned[used_idx]
        if other is None:
            return False
        return not (span[1] <= other[0] or span[0] >= other[1])

    def respects_shard_order(entry_idx: int, span: tuple[int, int]) -> bool:
        e = entries[entry_idx]
        for j, other in enumerate(assigned):
            if other is None:
                continue
            ej = entries[j]
            if ej["shard_idx"] == e["shard_idx"]:
                if ej["seg_idx"] < e["seg_idx"] and not (other[1] <= span[0]):
                    return False
                if ej["seg_idx"] > e["seg_idx"] and not (span[1] <= other[0]):
                    return False
        return True

    def backtrack(k: int) -> bool:
        nonlocal steps
        if k == len(order):
            return True
        steps += 1
        if steps > max_steps:
            return False
        idx = order[k]
        for span in entries[idx]["candidates"]:
            conflict = False
            for j in range(len(entries)):
                if overlaps(span, j):
                    conflict = True
                    break
            if conflict or not respects_shard_order(idx, span):
                continue
            assigned[idx] = span
            if backtrack(k + 1):
                return True
            assigned[idx] = None
        return False

    if not backtrack(0):
        # Differentiate overlap vs multi-segment internal order vs timeout.
        return None, [
            {
                "code": "OVERLAPPING_SEGMENT_SPANS",
                "message": (
                    "Could not assign pairwise non-overlapping, shard-ordered intervals to all segments. "
                    "Two or more segments overlap or violate within-shard document order."
                ),
            }
        ]

    resolved: list[dict[str, Any]] = []
    for i, e in enumerate(entries):
        span = assigned[i]
        assert span is not None
        resolved.append(
            {
                "shard_id": e["shard_id"],
                "segment_index": e["seg_idx"],
                "text": e["canonical_texts"][span],
                "reported_text": e["text"],
                "start": span[0],
                "end": span[1],
            }
        )
    resolved.sort(key=lambda x: (x["start"], x["end"]))
    return resolved, []


def classify_uncovered_text(text: str) -> tuple[bool, str, str]:
    """Return (is_safe, category, residual_bad_text)."""
    if SAFE_UNCOVERED_CHAR_RE.match(text):
        return True, "delimiter_or_punct", ""

    residual = re.sub(r"[\s\$\.,;:，。；：、．]+", "", text)
    # Iteratively strip whitelisted Chinese connective words from residual.
    changed = True
    while residual and changed:
        changed = False
        for w in sorted(SAFE_GLUE_WORDS, key=len, reverse=True):
            if residual.startswith(w):
                residual = residual[len(w) :]
                changed = True
                break
    if not residual:
        return True, "glue_word", ""
    return False, "unexpected", residual


def audit_shard_candidate(question: str, raw_output: dict[str, Any]) -> dict[str, Any]:
    """Deterministically verify Stage 2A sharding JSON against Math-LiC v2.6 alignment rules."""
    errors: list[dict[str, str]] = []
    warnings: list[str] = []

    if not isinstance(raw_output, dict):
        return {
            "status": "INVALID",
            "alignment_passed": False,
            "errors": [{"code": "INVALID_JSON_TYPE", "message": "Output must be a JSON object."}],
            "warnings": [],
        }

    status = raw_output.get("status")
    if status == "UNSHARDABLE":
        reason_code = raw_output.get("reason_code", "")
        reason = raw_output.get("reason", "")
        if reason_code not in VALID_REASON_CODES:
            errors.append(
                {
                    "code": "INVALID_REASON_CODE",
                    "message": f"Invalid reason_code {reason_code!r}; expected one of {sorted(VALID_REASON_CODES)}.",
                }
            )
        if not isinstance(reason, str) or not reason.strip():
            errors.append(
                {
                    "code": "MISSING_UNSHARDABLE_REASON",
                    "message": "UNSHARDABLE output must provide a non-empty reason string.",
                }
            )
        return {
            "status": "UNSHARDABLE",
            "alignment_passed": len(errors) == 0,
            "reason_code": reason_code,
            "reason": reason,
            "errors": errors,
            "warnings": warnings,
        }

    if status != "CANDIDATE":
        return {
            "status": str(status),
            "alignment_passed": False,
            "errors": [
                {
                    "code": "INVALID_STATUS",
                    "message": f"status must be 'CANDIDATE' or 'UNSHARDABLE', got {status!r}.",
                }
            ],
            "warnings": [],
        }

    initial_segments = raw_output.get("initial_segments")
    initial_shard = raw_output.get("initial_shard")
    shards = raw_output.get("shards")

    if not isinstance(initial_shard, str) or not initial_shard.strip():
        errors.append({"code": "EMPTY_INITIAL_SHARD", "message": "initial_shard must be a non-empty string."})
    if not isinstance(shards, list):
        errors.append({"code": "INVALID_SHARDS_LIST", "message": "shards must be a JSON array."})
        return {"status": "CANDIDATE", "alignment_passed": False, "errors": errors, "warnings": warnings}

    total_shards = 1 + len(shards)
    if not (3 <= total_shards <= 6):
        errors.append(
            {
                "code": "INVALID_SHARD_COUNT",
                "message": f"Total shard count must be 3..6 (S1 + 2..5 follow-ups), got {total_shards}.",
            }
        )

    expected_ids = [f"S{i}" for i in range(2, len(shards) + 2)]
    actual_ids = [s.get("shard_id") if isinstance(s, dict) else None for s in shards]
    if actual_ids != expected_ids:
        errors.append(
            {
                "code": "INVALID_SHARD_IDS",
                "message": f"shard_id values must be sequentially {expected_ids}, got {actual_ids}.",
            }
        )

    shard_segments_input: list[tuple[str, list[str]]] = [("S1", initial_segments)]
    shard_texts: list[tuple[str, str, list[str]]] = [
        ("S1", initial_shard if isinstance(initial_shard, str) else "", initial_segments or [])
    ]
    for idx, s in enumerate(shards):
        sid = s.get("shard_id") if isinstance(s, dict) else f"S{idx+2}"
        if not isinstance(s, dict):
            errors.append({"code": "INVALID_SHARD_OBJECT", "message": f"shards[{idx}] is not a JSON object."})
            continue
        stext = s.get("shard")
        ssegs = s.get("segments")
        if not isinstance(stext, str) or not stext.strip():
            errors.append({"code": "EMPTY_SHARD_TEXT", "message": f"{sid}.shard must be a non-empty string."})
            stext = ""
        shard_segments_input.append((sid, ssegs))
        shard_texts.append((sid, stext, ssegs if isinstance(ssegs, list) else []))

    if errors:
        return {"status": "CANDIDATE", "alignment_passed": False, "errors": errors, "warnings": warnings}

    # 1. Exact substring span alignment & non-overlap check
    resolved_spans, span_errors = resolve_non_overlapping_spans(question, shard_segments_input)
    errors.extend(span_errors)

    uncovered_spans: list[dict[str, Any]] = []
    if resolved_spans is not None:
        cursor = 0
        for sp in resolved_spans:
            if sp["start"] > cursor:
                gap = question[cursor : sp["start"]]
                ok, cat, bad = classify_uncovered_text(gap)
                uncovered_spans.append(
                    {"start": cursor, "end": sp["start"], "text": gap, "category": cat}
                )
                if not ok:
                    errors.append(
                        {
                            "code": "UNCOVERED_NON_TRIVIAL_TEXT",
                            "message": f"Uncovered substring at [{cursor}:{sp['start']}] contains non-glue content {bad!r} (full gap: {gap!r}).",
                        }
                    )
            cursor = sp["end"]
        if cursor < len(question):
            gap = question[cursor:]
            ok, cat, bad = classify_uncovered_text(gap)
            uncovered_spans.append(
                {"start": cursor, "end": len(question), "text": gap, "category": cat}
            )
            if not ok:
                errors.append(
                    {
                        "code": "UNCOVERED_NON_TRIVIAL_TEXT",
                        "message": f"Trailing uncovered substring at [{cursor}:{len(question)}] contains non-glue content {bad!r} (full gap: {gap!r}).",
                    }
                )

    # Map (shard_id, seg_idx) -> canonical matched span text from question when resolved
    canonical_seg_map: dict[tuple[str, int], str] = {}
    if resolved_spans is not None:
        for sp in resolved_spans:
            canonical_seg_map[(sp["shard_id"], sp["segment_index"])] = sp["text"]

    # 2. Per-shard formula fidelity, balanced math delimiters, relation-split guard, and S1 symbol-reuse guard
    added_delimiters_count = 0
    for sid, stext, ssegs in shard_texts:
        if not isinstance(ssegs, list):
            ssegs = []
        if sid != "S1":
            order_ref = re.search(r"(?:所得图形|上述条件|前述|由上式|如前所述)", stext)
            if order_ref:
                errors.append(
                    {
                        "code": "ORDER_DEPENDENT_REFERENCE",
                        "message": f"{sid} contains order-dependent phrase {order_ref.group(0)!r} in {stext!r}; shards S2..Sk must be self-contained and order-independent.",
                    }
                )
            for seg_idx, seg in enumerate(ssegs):
                if not isinstance(seg, str):
                    continue
                canonical_seg = canonical_seg_map.get((sid, seg_idx), seg).lstrip(" \t\r\n$")
                if re.match(r"^(?:=|＝|<|>|\\le\b|\\ge\b|\\leq\b|\\geq\b|\\neq\b|≤|≥|≠)", canonical_seg):
                    errors.append(
                        {
                            "code": "SPLIT_AT_RELATION_OPERATOR",
                            "message": f"{sid}.segments[{seg_idx}] starts with a relational operator ({canonical_seg[:16]!r}); splitting a single equation/inequality at '=' or inequality is forbidden.",
                        }
                    )
        dcount = count_unescaped_dollars(stext)
        if dcount % 2 != 0:
            errors.append(
                {
                    "code": "UNBALANCED_MATH_DELIMITER",
                    "message": f"{sid} has an odd number of unescaped '$' characters ({dcount}): {stext!r}",
                }
            )

        raw_dcount = sum(count_unescaped_dollars(seg) for seg in ssegs if isinstance(seg, str))
        if dcount > raw_dcount:
            added_delimiters_count += dcount - raw_dcount

        norm_shard = normalize_for_formula_check(stext)
        search_pos = 0
        for seg_idx, seg in enumerate(ssegs):
            if not isinstance(seg, str):
                continue
            canonical_seg = canonical_seg_map.get((sid, seg_idx), seg)
            core = strip_trailing_punct(canonical_seg)
            if not core:
                continue
            ok_seg, new_pos, norm_core = check_segment_preserved_in_shard(core, norm_shard, search_pos)
            if not ok_seg:
                errors.append(
                    {
                        "code": "FORMULA_OR_SEGMENT_ALTERED",
                        "message": (
                            f"{sid}.segments[{seg_idx}] core expression was altered or missing in {sid} text. "
                            f"Expected verbatim core {norm_core!r} inside {norm_shard!r}."
                        ),
                    }
                )
            else:
                search_pos = new_pos

        # Check S1 added scaffolding AND pre-target text do not smuggle relational constraints or positivity conditions
        if sid == "S1":
            if re.search(r"(?:^|[，,。\.；;\s])当时(?=[，,。\.；;\s]|$)", norm_shard):
                errors.append(
                    {
                        "code": "DANGLING_TEMPLATE_TEXT",
                        "message": f"S1 contains dangling '当时' left after stripping '当 <cond> 时': {stext!r}. Move '当...时' completely into a follow-up shard.",
                    }
                )
            residual_s1 = norm_shard
            for seg_idx, seg in enumerate(ssegs):
                if not isinstance(seg, str):
                    continue
                canonical_seg = canonical_seg_map.get((sid, seg_idx), seg)
                core = strip_trailing_punct(canonical_seg)
                if core:
                    residual_s1 = residual_s1.replace(normalize_for_formula_check(core), " ", 1)
            leak = RELATIONAL_LEAK_RE.search(SAFE_HTML_TAG_RE.sub("", residual_s1))
            if leak:
                errors.append(
                    {
                        "code": "S1_LEAKAGE_IN_REUSE",
                        "message": (
                            f"S1 non-segment scaffolding contains relational operator/constraint {leak.group(0)!r} "
                            f"in residual text {residual_s1!r}."
                        ),
                    }
                )
            # Also verify S1 pre-target setup (text before final goal verb like 求/试求/计算/问)
            # does not smuggle equations, inequalities, or positivity/non-zero restrictions.
            target_matches = list(re.finditer(r"(?:试求|求|计算|问)", norm_shard))
            if target_matches:
                pre_target = SAFE_HTML_TAG_RE.sub("", norm_shard[: target_matches[-1].start()])
                pre_rel = RELATIONAL_LEAK_RE.search(pre_target)
                pre_qual = re.search(r"(?:各项均为正数|公差不为[0零]|公比不为|恒成立)", pre_target)
                if pre_rel:
                    errors.append(
                        {
                            "code": "S1_LEAKAGE_IN_PRE_TARGET",
                            "message": (
                                f"S1 pre-target setup contains relational constraint {pre_rel.group(0)!r} in {pre_target!r}; "
                                "move equations/inequalities/parameter bounds into S2..Sk."
                            ),
                        }
                    )
                if pre_qual:
                    errors.append(
                        {
                            "code": "S1_LEAKAGE_IN_PRE_TARGET",
                            "message": (
                                f"S1 pre-target setup contains qualitative/sign constraint {pre_qual.group(0)!r} in {pre_target!r}; "
                                "move it to a follow-up shard S2..Sk."
                            ),
                        }
                    )

    concat_prompt = "\n".join([initial_shard] + [s["shard"] for s in shards]) if not errors else None
    canonical_order = [s["shard_id"] for s in shards]

    return {
        "status": "CANDIDATE",
        "alignment_passed": len(errors) == 0,
        "total_shards": total_shards,
        "errors": errors,
        "warnings": warnings,
        "audit_record": {
            "segment_spans": resolved_spans,
            "uncovered_spans": uncovered_spans,
            "added_delimiters_count": added_delimiters_count,
            "canonical_disclosure_order": canonical_order,
            "concat_prompt": concat_prompt,
        },
    }


def run_self_test() -> None:
    print("Running audit_alignment.py self-tests...")

    # Test 1: Synthetic one-shot example from prompt Section 9
    q_syn = "已知函数 $g(x)=|x-1|+|x-b|(b>0)$，若 $g(1)=2$，求 $b$ 的值。"
    out_syn = {
        "status": "CANDIDATE",
        "initial_segments": ["已知函数", "求 $b$ 的值。"],
        "initial_shard": "已知函数 $g(x)$，求 $b$ 的值。",
        "shards": [
            {"shard_id": "S2", "segments": ["g(x)=|x-1|+|x-b|"], "shard": "$g(x)=|x-1|+|x-b|$。"},
            {"shard_id": "S3", "segments": ["(b>0)"], "shard": "参数满足 $(b>0)$。"},
            {"shard_id": "S4", "segments": ["若 $g(1)=2$"], "shard": "若 $g(1)=2$。"},
        ],
    }
    res_syn = audit_shard_candidate(q_syn, out_syn)
    assert res_syn["alignment_passed"], f"Synthetic test failed: {res_syn['errors']}"

    # Test 2: Valid sharding of ob-comp-376
    q_376 = "已知函数 $f(x)=|2 x+1|-|3 x-a|(a>0)$\n当 $a=3$ 时, 求不等式 $f(x)>2$ 的解集."
    out_376 = {
        "status": "CANDIDATE",
        "initial_segments": ["已知函数", "求不等式 $f(x)>2$ 的解集."],
        "initial_shard": "已知函数 $f(x)$，求不等式 $f(x)>2$ 的解集。",
        "shards": [
            {"shard_id": "S2", "segments": ["f(x)=|2 x+1|-|3 x-a|"], "shard": "$f(x)=|2 x+1|-|3 x-a|$。"},
            {"shard_id": "S3", "segments": ["(a>0)"], "shard": "参数满足 $(a>0)$。"},
            {"shard_id": "S4", "segments": ["当 $a=3$ 时"], "shard": "当 $a=3$ 时。"},
        ],
    }
    res_376 = audit_shard_candidate(q_376, out_376)
    assert res_376["alignment_passed"], f"ob-comp-376 valid test failed: {res_376['errors']}"

    # Test 3: Valid sharding of ob-cee-4027 (wrapped line-level $...$)
    q_4027 = "$在\\triangle ABC中,角A,B,C所对的边分别为a,b,c.已知4a=\\sqrt{5}c,cos C=\\frac{3}{5}.$\n$求sin A的值;$"
    out_4027 = {
        "status": "CANDIDATE",
        "initial_segments": ["在\\triangle ABC中,角A,B,C所对的边分别为a,b,c.", "求sin A的值;"],
        "initial_shard": "在 $\\triangle ABC$ 中，角 $A,B,C$ 所对的边分别为 $a,b,c$，求 $sin A$ 的值。",
        "shards": [
            {"shard_id": "S2", "segments": ["已知4a=\\sqrt{5}c"], "shard": "已知 $4a=\\sqrt{5}c$。"},
            {"shard_id": "S3", "segments": ["cos C=\\frac{3}{5}."], "shard": "$cos C=\\frac{3}{5}$。"},
        ],
    }
    res_4027 = audit_shard_candidate(q_4027, out_4027)
    assert res_4027["alignment_passed"], f"ob-cee-4027 valid test failed: {res_4027['errors']}"

    # Test 4: Negative - altered space inside math expression (2x+1 instead of 2 x+1)
    bad_space = json.loads(json.dumps(out_376))
    bad_space["shards"][0]["shard"] = "$f(x)=|2x+1|-|3 x-a|$。"
    res_bad_space = audit_shard_candidate(q_376, bad_space)
    assert not res_bad_space["alignment_passed"]
    assert any(e["code"] == "FORMULA_OR_SEGMENT_ALTERED" for e in res_bad_space["errors"])

    # Test 5: Negative - overlapping spans between S1 and S2
    bad_overlap = json.loads(json.dumps(out_376))
    bad_overlap["initial_segments"] = ["已知函数 $f(x)", "求不等式 $f(x)>2$ 的解集."]
    bad_overlap["shards"][0]["segments"] = ["$f(x)=|2 x+1|-|3 x-a|"]
    res_bad_overlap = audit_shard_candidate(q_376, bad_overlap)
    assert not res_bad_overlap["alignment_passed"]
    assert any(e["code"] == "OVERLAPPING_SEGMENT_SPANS" for e in res_bad_overlap["errors"])

    # Test 6: Negative - dropped condition left uncovered in question
    bad_drop = json.loads(json.dumps(out_376))
    bad_drop["shards"].pop()  # drop 当 $a=3$ 时
    res_bad_drop = audit_shard_candidate(q_376, bad_drop)
    assert not res_bad_drop["alignment_passed"]
    assert any(e["code"] == "UNCOVERED_NON_TRIVIAL_TEXT" for e in res_bad_drop["errors"])

    # Test 7: Negative - wrong shard_id starting at S1
    bad_ids = json.loads(json.dumps(out_4027))
    bad_ids["shards"][0]["shard_id"] = "S1"
    bad_ids["shards"][1]["shard_id"] = "S2"
    res_bad_ids = audit_shard_candidate(q_4027, bad_ids)
    assert not res_bad_ids["alignment_passed"]
    assert any(e["code"] == "INVALID_SHARD_IDS" for e in res_bad_ids["errors"])

    # Test 8: Negative - S1 symbol reuse smuggling relational constraint
    bad_leak = json.loads(json.dumps(out_376))
    bad_leak["initial_shard"] = "已知函数 $f(x)=|2 x+1|-|3 x-a|$，求不等式 $f(x)>2$ 的解集。"
    res_bad_leak = audit_shard_candidate(q_376, bad_leak)
    assert not res_bad_leak["alignment_passed"]
    assert any(e["code"] == "S1_LEAKAGE_IN_REUSE" for e in res_bad_leak["errors"])

    # Test 9: Valid UNSHARDABLE declaration
    res_unsh = audit_shard_candidate(
        q_376,
        {
            "status": "UNSHARDABLE",
            "reason_code": "TOO_MANY_SHARDS",
            "reason": "Example test reason",
        },
    )
    assert res_unsh["alignment_passed"]

    # Test 10: CEE brace escaping ({a_n} -> $\{a_n\}$) and separating 各项均为正数 into S2
    q_4569 = "$已知各项均为正数的等差数列{a_n}，a_2=5，2a_1，a_3，a_5+2成等比数列.$\n$求a_n的通项公式；$"
    out_4569 = {
        "status": "CANDIDATE",
        "initial_segments": ["等差数列{a_n}，", "求a_n的通项公式；"],
        "initial_shard": "已知等差数列 $\\{a_n\\}$，求 $a_n$ 的通项公式。",
        "shards": [
            {"shard_id": "S2", "segments": ["已知各项均为正数的"], "shard": "数列 $\\{a_n\\}$ 各项均为正数。"},
            {"shard_id": "S3", "segments": ["a_2=5"], "shard": "$a_2=5$。"},
            {"shard_id": "S4", "segments": ["2a_1，a_3，a_5+2成等比数列"], "shard": "$2a_1$，$a_3$，$a_5+2$ 成等比数列。"},
        ],
    }
    res_4569 = audit_shard_candidate(q_4569, out_4569)
    assert res_4569["alignment_passed"], f"ob-cee-4569 brace test failed: {res_4569['errors']}"

    # Test 11: Non-semantic delimiter snapping (outer $ in segments on ob-comp-376) and ob-comp-711 leading $
    out_376_dollar = json.loads(json.dumps(out_376))
    out_376_dollar["shards"][0]["segments"] = ["$f(x)=|2 x+1|-|3 x-a|$"]
    out_376_dollar["shards"][1]["segments"] = ["$(a>0)$"]
    res_376_dollar = audit_shard_candidate(q_376, out_376_dollar)
    assert res_376_dollar["alignment_passed"], f"Delimiter snap test failed: {res_376_dollar['errors']}"

    # Test 12: Geometry noun clarification ("抛物线 $C$ 的方程为 $y^{2}=4 x$。" and "点 $A$ 的坐标为 $(-2,0)$。")
    q_250 = "已知 $O$ 为坐标原点, 抛物线 $C: y^{2}=4 x$, 点 $A(-2,0)$, 设直线 $l$ 与 $C$ 交于不同的两点 $P 、 Q$.\n若直线 $l \\perp x$ 轴, 求直线 $P A$ 的斜率的取值范围."
    out_250 = {
        "status": "CANDIDATE",
        "initial_segments": ["已知 $O$ 为坐标原点", "求直线 $P A$ 的斜率的取值范围."],
        "initial_shard": "已知 $O$ 为坐标原点，存在抛物线 $C$、点 $A$ 与直线 $l$（交 $C$ 于 $P, Q$），求直线 $P A$ 的斜率的取值范围。",
        "shards": [
            {"shard_id": "S2", "segments": ["抛物线 $C: y^{2}=4 x$"], "shard": "抛物线 $C$ 的方程为 $y^{2}=4 x$。"},
            {"shard_id": "S3", "segments": ["点 $A(-2,0)$"], "shard": "点 $A$ 的坐标为 $(-2,0)$。"},
            {"shard_id": "S4", "segments": ["设直线 $l$ 与 $C$ 交于不同的两点 $P 、 Q$."], "shard": "设直线 $l$ 与 $C$ 交于不同的两点 $P 、 Q$。"},
            {"shard_id": "S5", "segments": ["若直线 $l \\perp x$ 轴"], "shard": "若直线 $l \\perp x$ 轴。"},
        ],
    }
    res_250 = audit_shard_candidate(q_250, out_250)
    assert res_250["alignment_passed"], f"Geometry clarification test failed: {res_250['errors']}"

    # Test 13: \left does NOT false-trigger \le in S1_LEAKAGE_IN_REUSE
    q_seq = "已知数列 $\\left\\{b_{n}\\right\\}$ 满足 $b_{n}=n+1$，若 $b_{1}=2$，求数列 $\\left\\{b_{n}\\right\\}$ 的前 $n$ 项和。"
    out_seq = {
        "status": "CANDIDATE",
        "initial_segments": ["已知数列 $\\left\\{b_{n}\\right\\}$", "求数列 $\\left\\{b_{n}\\right\\}$ 的前 $n$ 项和。"],
        "initial_shard": "已知数列 $\\left\\{b_{n}\\right\\}$，求数列 $\\left\\{b_{n}\\right\\}$ 的前 $n$ 项和。",
        "shards": [
            {"shard_id": "S2", "segments": ["满足 $b_{n}=n+1$"], "shard": "数列 $\\left\\{b_{n}\\right\\}$ 满足 $b_{n}=n+1$。"},
            {"shard_id": "S3", "segments": ["若 $b_{1}=2$"], "shard": "若 $b_{1}=2$。"},
        ],
    }
    res_seq = audit_shard_candidate(q_seq, out_seq)
    assert res_seq["alignment_passed"], f"\\left boundary test failed: {res_seq['errors']}"

    # Test 14: Reject splitting a single equation at '='
    q_eq_split = "已知函数 $f(x)$ 满足 $x+1=2x-3$，且 $x>0$，求 $x$。"
    out_eq_split = {
        "status": "CANDIDATE",
        "initial_segments": ["已知函数 $f(x)$", "求 $x$。"],
        "initial_shard": "已知函数 $f(x)$，求 $x$。",
        "shards": [
            {"shard_id": "S2", "segments": ["x+1"], "shard": "$x+1$。"},
            {"shard_id": "S3", "segments": ["=2x-3"], "shard": "$=2x-3$。"},
            {"shard_id": "S4", "segments": ["$x>0$"], "shard": "$x>0$。"},
        ],
    }
    res_eq_split = audit_shard_candidate(q_eq_split, out_eq_split)
    assert not res_eq_split["alignment_passed"]
    assert any(e["code"] == "SPLIT_AT_RELATION_OPERATOR" for e in res_eq_split["errors"])

    # Test 15: Leading Hanzi glue attached to noun (已知抛物线G：y^2=2px -> 抛物线 G 的方程为 y^2=2px)
    q_4859_clean = "$已知抛物线G：y^2＝2px，其中p>0.点M(2，0)在G的焦点F的右侧.$\n$求抛物线的方程；$"
    out_4859_clean = {
        "status": "CANDIDATE",
        "initial_segments": ["求抛物线的方程；"],
        "initial_shard": "已知抛物线 $G$，求抛物线的方程。",
        "shards": [
            {"shard_id": "S2", "segments": ["已知抛物线G：y^2＝2px，其中p>0."], "shard": "抛物线 $G$ 的方程为 $y^2＝2px$，其中 $p>0$。"},
            {"shard_id": "S3", "segments": ["点M(2，0)在G的焦点F的右侧."], "shard": "点 $M(2，0)$ 位于抛物线 $G$ 的焦点 $F$ 的右侧。"},
        ],
    }
    res_4859_clean = audit_shard_candidate(q_4859_clean, out_4859_clean)
    assert res_4859_clean["alignment_passed"], f"Glue prefix test failed: {res_4859_clean['errors']}"

    # Test 16: Reject equation or positivity smuggling directly inside S1 initial_segments (e.g., ob-cee-4471 / 4569)
    bad_s1_smuggle = json.loads(json.dumps(out_4569))
    bad_s1_smuggle["initial_segments"] = ["已知各项均为正数的等差数列{a_n}，", "求a_n的通项公式；"]
    bad_s1_smuggle["initial_shard"] = "已知各项均为正数的等差数列 $\\{a_n\\}$，求 $a_n$ 的通项公式。"
    bad_s1_smuggle["shards"] = [
        {"shard_id": "S2", "segments": ["a_2=5"], "shard": "$a_2=5$。"},
        {"shard_id": "S3", "segments": ["2a_1，a_3，a_5+2成等比数列"], "shard": "$2a_1$，$a_3$，$a_5+2$ 成等比数列。"},
    ]
    res_bad_smuggle = audit_shard_candidate(q_4569, bad_s1_smuggle)
    assert not res_bad_smuggle["alignment_passed"]
    assert any(e["code"] == "S1_LEAKAGE_IN_PRE_TARGET" for e in res_bad_smuggle["errors"])

    # Also verify ob-comp-711 "最大值为 f(x_0)" in S1 still passes pre-target check
    q_711 = "设函数\n\n$f(x)=\\frac{x(1-x)}{x^{3}-x+1}(0<x<1)$.\n\n若 $f(x)$ 的最大值为 $f\\left(x_{0}\\right)$, 求 $x_{0}$的大小。"
    out_711 = {
        "status": "CANDIDATE",
        "initial_segments": ["设函数", "若 $f(x)$ 的最大值为 $f\\left(x_{0}\\right)$, 求 $x_{0}$的大小。"],
        "initial_shard": "设函数 $f(x)$，若 $f(x)$ 的最大值为 $f\\left(x_{0}\\right)$，求 $x_{0}$ 的大小。",
        "shards": [
            {"shard_id": "S2", "segments": ["f(x)=\\frac{x(1-x)}{x^{3}-x+1}"], "shard": "$f(x)=\\frac{x(1-x)}{x^{3}-x+1}$。"},
            {"shard_id": "S3", "segments": ["(0<x<1)."], "shard": "定义域满足 $(0<x<1)$。"},
        ],
    }
    res_711 = audit_shard_candidate(q_711, out_711)
    assert res_711["alignment_passed"], f"ob-comp-711 pre-target regression failed: {res_711['errors']}"

    # Test 17: Reject dangling '当时,' inside S1
    bad_dang = json.loads(json.dumps(out_376))
    bad_dang["initial_shard"] = "已知函数 $f(x)$，当时, 求不等式 $f(x)>2$ 的解集."
    res_bad_dang = audit_shard_candidate(q_376, bad_dang)
    assert not res_bad_dang["alignment_passed"]
    assert any(e["code"] == "DANGLING_TEMPLATE_TEXT" for e in res_bad_dang["errors"])

    # Test 18: Reject order-dependent reference '所得图形' inside S3
    q_tri = "给定多边形 $P$. 用对角线分割 $P$. 所得图形称为 $T$. 当条件成立时, 求数量."
    bad_order = {
        "status": "CANDIDATE",
        "initial_segments": ["给定多边形 $P$.", "求数量."],
        "initial_shard": "给定多边形 $P$，求数量。",
        "shards": [
            {"shard_id": "S2", "segments": ["用对角线分割 $P$."], "shard": "用对角线分割 $P$。"},
            {"shard_id": "S3", "segments": ["所得图形称为 $T$."], "shard": "所得图形称为 $T$。"},
            {"shard_id": "S4", "segments": ["当条件成立时"], "shard": "当条件成立时。"},
        ],
    }
    res_bad_order = audit_shard_candidate(q_tri, bad_order)
    assert not res_bad_order["alignment_passed"]
    assert any(e["code"] == "ORDER_DEPENDENT_REFERENCE" for e in res_bad_order["errors"])

    # Test 19: Safe HTML subscript/superscript tags (F<sub>1</sub>) in S1 do not trigger false '<' S1 leakage
    q_html = "已知椭圆C，F<sub>1</sub>，F<sub>2</sub>为其焦点，C的方程为x^2/4+y^2=1，点P在C上，求|PF<sub>1</sub>|+|PF<sub>2</sub>|。"
    out_html = {
        "status": "CANDIDATE",
        "initial_segments": ["已知椭圆C，", "求|PF<sub>1</sub>|+|PF<sub>2</sub>|。"],
        "initial_shard": "已知椭圆 $C$ 的焦点为 $F<sub>1</sub>, F<sub>2</sub>$，求 $|PF<sub>1</sub>|+|PF<sub>2</sub>|$。",
        "shards": [
            {"shard_id": "S2", "segments": ["F<sub>1</sub>，F<sub>2</sub>为其焦点，"], "shard": "$F<sub>1</sub>$，$F<sub>2</sub>$ 为其焦点。"},
            {"shard_id": "S3", "segments": ["C的方程为x^2/4+y^2=1，"], "shard": "$C$ 的方程为 $x^2/4+y^2=1$。"},
            {"shard_id": "S4", "segments": ["点P在C上，"], "shard": "点 $P$ 在 $C$ 上。"},
        ],
    }
    res_html = audit_shard_candidate(q_html, out_html)
    assert res_html["alignment_passed"], f"HTML sub tag test failed: {res_html['errors']}"

    print("All 19 self-tests PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit Stage 2A sharding outputs.")
    parser.add_argument("--self-test", action="store_true", help="Run built-in unit checks.")
    args = parser.parse_args()
    if args.self_test:
        run_self_test()


if __name__ == "__main__":
    main()
