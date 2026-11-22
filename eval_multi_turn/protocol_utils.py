#!/usr/bin/env python3
"""Deterministic multi-turn protocol definitions, paired disclosure plans, and <think> splitters.

Strictly implements eval_multi_turn/EXPERIMENT_CONTRACT.md (v2.0):
- Shared Chinese instructions placed in the first user turn (NO system role, native HF templates).
- Paired order pi_{i,r} shared across CONCAT, DYN-FREE, DYN-REQUEST, and DYN-REQUEST-GUARD.
- Robust last-</think> splitting preventing private reasoning leakage or empty-wait mislabeling.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from pathlib import Path
from typing import Any

PROTOCOLS_EXP2 = ("FULL", "CONCAT", "CONCAT_MATCHED", "DYN_FREE", "DYN_REQUEST")
PROTOCOLS_ALL = ("FULL", "CONCAT", "CONCAT_MATCHED", "DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD")

NONFINAL_SCOPES = ("Wait", "Conditional", "Partial/Other", "Commit", "Unresolved")

PRIMARY_MODELS = (
    "Qwen/Qwen3-8B",
    "Qwen/Qwen3-14B",
    "zai-org/GLM-Z1-9B-0414",
)

MODEL_PRESETS: dict[str, dict[str, Any]] = {
    "Qwen/Qwen3-8B": {
        "short_name": "Qwen3-8B",
        "family": "Qwen3",
        "scale_tier": "8B",
        "dtype": "bfloat16",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 16384,
        "max_model_len": 32768,
        "max_num_seqs_40gb": 24,
        "default_revision": "b968826d9c46dd6066d109eabc6255188de91218",
        "enable_thinking": True,
        "tokenizer_mode": "auto",
        "system_prompt_file": None,
        "pure_text_mm_zero": False,
    },
    "Qwen/Qwen3-14B": {
        "short_name": "Qwen3-14B",
        "family": "Qwen3",
        "scale_tier": "14B",
        "dtype": "bfloat16",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": 20,
        "min_p": 0.0,
        "max_tokens": 16384,
        "max_model_len": 32768,
        "max_num_seqs_40gb": 12,
        "default_revision": "40c069824f4251a91eefaf281ebe4c544efd3e18",
        "enable_thinking": True,
        "tokenizer_mode": "auto",
        "system_prompt_file": None,
        "pure_text_mm_zero": False,
    },
    "zai-org/GLM-Z1-9B-0414": {
        "short_name": "GLM-Z1-9B",
        "family": "GLM",
        "scale_tier": "9B",
        "dtype": "bfloat16",
        "temperature": 0.6,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 16384,
        "max_model_len": 32768,
        "max_num_seqs_40gb": 24,
        "default_revision": "b221b06fefb23ca320922cf6e68ab5f2fb82de81",
        "enable_thinking": True,
        "tokenizer_mode": "auto",
        "system_prompt_file": None,
        "pure_text_mm_zero": False,
    },
    "mistralai/Ministral-3-8B-Reasoning-2512": {
        "short_name": "Ministral-3-8B-Reasoning",
        "family": "Mistral3",
        "scale_tier": "8B",
        "dtype": "bfloat16",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 8192,
        "max_model_len": 16384,
        "enable_thinking": True,
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "system_prompt_file": "SYSTEM_PROMPT.txt",
        "pure_text_mm_zero": True,
    },
    "mistralai/Ministral-3-14B-Reasoning-2512": {
        "short_name": "Ministral-3-14B-Reasoning",
        "family": "Mistral3",
        "scale_tier": "14B",
        "dtype": "bfloat16",
        "temperature": 0.7,
        "top_p": 0.95,
        "top_k": -1,
        "min_p": 0.0,
        "max_tokens": 8192,
        "max_model_len": 16384,
        "enable_thinking": True,
        "tokenizer_mode": "mistral",
        "config_format": "mistral",
        "load_format": "mistral",
        "system_prompt_file": "SYSTEM_PROMPT.txt",
        "pure_text_mm_zero": True,
    },
}

TASK_PREAMBLE = (
    "【任务规则】\n"
    "请你严格仅基于目前已经给出的信息作答，并清晰区分已知条件、显式假设与推导结论。\n"
    "1. 如果当前信息尚不足以唯一确定原题所求结果（例如关键参数、方程或约束缺失），允许等待补充条件、说明缺少什么信息或仅在明确写出的假设前提下讨论；\n"
    "2. 严禁自行补充、猜测或编造未给出的前提条件；\n"
    "3. 后续是否还会提供更多条件未知，请勿假设一定存在或一定不存在后续补充。"
)

SCOPE_GUARD_PREAMBLE = (
    TASK_PREAMBLE
    + "\n"
    + "【充分性与作用域自检】\n"
    + "在给出任何目标答案前，你必须先逐项核对题目所求量所需的全部约束是否已在当前对话中完全确定：\n"
    + "- 若仍有自由度、未定参数或缺失约束导致目标结果可随补充条件改变，即使被要求给出当前答案，也绝对不要输出无条件限定的确定答案或无限定 \\boxed{...}；\n"
    + "- 此时要么直接指出所缺条件并保持等待，要么必须在最终结论句中显式绑定具体假设或当前放宽范围（例如“仅在假设……且无其他约束的前提下，当前结果为……”）。"
)

NONFINAL_REQUEST_SUFFIX = "\n\n请基于目前已提供的信息给出当前答案，并简要说明依据。"

FINAL_SUFFIX = "\n\n全部条件已给出，请给出最终解答与最终答案（使用 \\boxed{...} 标出），若仍无法确定请明确说明原因。"


def prompt_manifest() -> dict[str, Any]:
    texts = {
        "TASK_PREAMBLE": TASK_PREAMBLE,
        "SCOPE_GUARD_PREAMBLE": SCOPE_GUARD_PREAMBLE,
        "NONFINAL_REQUEST_SUFFIX": NONFINAL_REQUEST_SUFFIX,
        "FINAL_SUFFIX": FINAL_SUFFIX,
    }
    return {
        "prompts": texts,
        "sha256": {k: hashlib.sha256(v.encode("utf-8")).hexdigest() for k, v in texts.items()},
    }


def _seed_int(*parts: Any) -> int:
    key = ":".join(str(p) for p in parts).encode("utf-8")
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big") & 0x7FFFFFFF


def _non_canonical_perm(tail_ids: list[str], seed_int: int, forbid: list[str] | None = None) -> list[str]:
    """Return a deterministic non-canonical permutation of tail_ids (avoiding canonical order)."""
    if len(tail_ids) <= 1:
        return list(tail_ids)
    if len(tail_ids) == 2:
        return [tail_ids[1], tail_ids[0]]
    rng = random.Random(seed_int)
    perm = list(tail_ids)
    for _ in range(32):
        rng.shuffle(perm)
        if perm != tail_ids and (forbid is None or perm != forbid):
            return list(perm)
    # Deterministic cyclic fallback guaranteed != tail_ids
    shifted = tail_ids[1:] + tail_ids[:1]
    if forbid is not None and shifted == forbid and len(tail_ids) >= 3:
        shifted = tail_ids[2:] + tail_ids[:2]
    return shifted


def build_disclosure_plan(
    items: list[dict[str, Any]],
    seed_base: int = 20261024,
    reps: int = 3,
    protocols: tuple[str, ...] = PROTOCOLS_ALL,
) -> list[dict[str, Any]]:
    """Build a frozen disclosure plan & independent per-request seed manifest.

    For each (item_id, rep):
      - CONCAT always uses canonical order [S1, S2, ..., Sk] across all reps (LiC standard).
      - rep == 0: canonical order [S1, S2, ..., Sk] for CONCAT_MATCHED & dynamic protocols
        (with strictly independent request_seeds for CONCAT vs CONCAT_MATCHED).
      - rep >= 1: forced non-canonical permutation of [S2, ..., Sk] shared across
        CONCAT_MATCHED, DYN_FREE, DYN_REQUEST, and DYN_REQUEST_GUARD.
    """
    plan: list[dict[str, Any]] = []
    for item in items:
        iid = item["item_id"]
        shards = item["shards"]
        s1 = shards[0]["shard_id"]
        assert s1 == "S1", f"{iid}: first shard must be S1, got {s1}"
        tail_ids = [s["shard_id"] for s in shards[1:]]
        canonical_order = [s1] + list(tail_ids)
        prev_perm: list[str] | None = None
        for rep in range(reps):
            if rep == 0:
                order = list(canonical_order)
                order_kind = "canonical"
            else:
                perm = _non_canonical_perm(
                    tail_ids,
                    _seed_int(seed_base, iid, "order", rep),
                    forbid=prev_perm if len(tail_ids) >= 3 else None,
                )
                prev_perm = perm
                order = [s1] + perm
                order_kind = "paired_permuted"

            req_seeds = {
                proto: [
                    _seed_int(seed_base, iid, proto, rep, turn_idx)
                    for turn_idx in range(1, (2 if proto in ("FULL", "CONCAT", "CONCAT_MATCHED") else len(order) + 1))
                ]
                for proto in protocols
            }
            plan.append(
                {
                    "item_id": iid,
                    "rep": rep,
                    "canonical_k": len(order),
                    "canonical_order": canonical_order,
                    "order_kind": order_kind,
                    "disclosure_order": order,
                    "request_seeds": req_seeds,
                }
            )
    return plan


def build_turn_user_text(
    item: dict[str, Any],
    protocol: str,
    disclosure_order: list[str],
    turn_idx: int,
) -> str:
    """Construct the exact user message for a given protocol and 1-based turn_idx."""
    shard_map = {s["shard_id"]: s["shard"] for s in item["shards"]}
    is_guard = protocol == "DYN_REQUEST_GUARD"
    preamble = SCOPE_GUARD_PREAMBLE if is_guard else TASK_PREAMBLE

    if protocol == "FULL":
        assert turn_idx == 1
        return f"{preamble}\n\n【题目】\n{item['question']}{FINAL_SUFFIX}"

    if protocol == "CONCAT":
        assert turn_idx == 1
        canonical_concat = "\n".join(s["shard"] for s in item["shards"])
        return f"{preamble}\n\n【题目】\n{canonical_concat}{FINAL_SUFFIX}"

    if protocol == "CONCAT_MATCHED":
        assert turn_idx == 1
        paired_concat = "\n".join(shard_map[sid] for sid in disclosure_order)
        return f"{preamble}\n\n【题目】\n{paired_concat}{FINAL_SUFFIX}"

    if protocol in ("DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"):
        k = len(disclosure_order)
        assert 1 <= turn_idx <= k
        shard_text = shard_map[disclosure_order[turn_idx - 1]]
        prefix = f"{preamble}\n\n" if turn_idx == 1 else ""
        if turn_idx == k:
            return f"{prefix}{shard_text}{FINAL_SUFFIX}"
        if protocol in ("DYN_REQUEST", "DYN_REQUEST_GUARD"):
            return f"{prefix}{shard_text}{NONFINAL_REQUEST_SUFFIX}"
        return f"{prefix}{shard_text}"

    raise ValueError(f"Unknown protocol: {protocol!r}")


def split_thinking_response(
    raw_text: str,
    prompt_text: str = "",
    finish_reason: str | None = None,
) -> dict[str, Any]:
    """Safely separate chain-of-thought reasoning from user-visible reply.

    Handles both `<think>...</think>` (Qwen3 / DeepSeek-R1) and `[THINK]...[/THINK]`
    (Ministral-3-Reasoning), including cases where the prompt ends with an opening tag
    or generation truncates before the closing tag.
    Never maps unclosed/truncated reasoning into `visible_reply`.
    """
    text = raw_text or ""
    close_matches = list(re.finditer(r"(?:</think>|\[/THINK\])", text))
    if close_matches:
        last_close = close_matches[-1]
        thought_raw = text[: last_close.start()]
        thought_clean = re.sub(r"^\s*(?:<think>|\[THINK\])\s*", "", thought_raw, count=1).strip()
        visible = text[last_close.end() :].strip()
        return {
            "reasoning_trace": thought_clean,
            "visible_reply": visible,
            "thinking_truncated": False,
            "visible_truncated": finish_reason == "length" and not visible,
            "empty_truncated": not visible and finish_reason == "length",
        }

    prompt_opened_think = bool(re.search(r"(?:<think>|\[THINK\])\s*$", prompt_text or ""))
    raw_opened_think = ("<think>" in text) or ("[THINK]" in text)
    if raw_opened_think or (prompt_opened_think and finish_reason == "length"):
        thought_clean = re.sub(r"^\s*(?:<think>|\[THINK\])\s*", "", text, count=1).strip()
        return {
            "reasoning_trace": thought_clean,
            "visible_reply": "",
            "thinking_truncated": True,
            "visible_truncated": True,
            "empty_truncated": True,
        }

    visible = text.strip()
    return {
        "reasoning_trace": "",
        "visible_reply": visible,
        "thinking_truncated": False,
        "visible_truncated": finish_reason == "length",
        "empty_truncated": not visible,
    }


def extract_boxed_answers(text: str) -> list[str]:
    """Extract all balanced \\boxed{...} payloads cleanly, handling arbitrary brace nesting."""
    results: list[str] = []
    idx = 0
    marker = r"\boxed{"
    while idx < len(text):
        pos = text.find(marker, idx)
        if pos == -1:
            break
        brace_start = pos + len(marker) - 1
        depth = 0
        end = -1
        for cur in range(brace_start, len(text)):
            ch = text[cur]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = cur
                    break
        if end != -1:
            results.append(text[brace_start + 1 : end].strip())
            idx = end + 1
        else:
            break
    return results


def safe_history_assistant_text(split_res: dict[str, Any], empty_history_mode: str = "empty") -> str:
    """Return assistant content for next-turn history.

    Defaults to genuine empty string (`""`) so truncation does not inject synthetic text.
    If a template rejects empty assistant content, pass `--empty-history-mode placeholder`
    and disclose it explicitly as an input intervention.
    """
    vis = split_res["visible_reply"].strip()
    if vis:
        return vis
    if empty_history_mode == "placeholder":
        return "[本轮思考达到输出上限，未生成可见回答]"
    return ""


def _run_self_tests() -> None:
    # 1. Paired order consistency across CONCAT and dynamic protocols
    mock_item = {
        "item_id": "mock-1",
        "question": "FULL_Q",
        "shards": [
            {"shard_id": "S1", "shard": "设求x。"},
            {"shard_id": "S2", "shard": "条件A。"},
            {"shard_id": "S3", "shard": "条件B。"},
            {"shard_id": "S4", "shard": "条件C。"},
        ],
    }
    plan = build_disclosure_plan([mock_item], seed_base=20261024, reps=3)
    assert len(plan) == 3
    assert plan[0]["order_kind"] == "canonical"
    assert plan[0]["disclosure_order"] == ["S1", "S2", "S3", "S4"]
    # rep=1 and rep=2 must be strictly non-canonical AND distinct for k=4
    assert plan[1]["order_kind"] == "paired_permuted" and plan[1]["disclosure_order"] != ["S1", "S2", "S3", "S4"]
    assert plan[2]["order_kind"] == "paired_permuted" and plan[2]["disclosure_order"] != ["S1", "S2", "S3", "S4"]
    assert plan[1]["disclosure_order"] != plan[2]["disclosure_order"]
    # CONCAT vs CONCAT_MATCHED have strictly distinct request seeds even at rep=0
    assert plan[0]["request_seeds"]["CONCAT"] != plan[0]["request_seeds"]["CONCAT_MATCHED"]

    canonical_concat_body = "\n".join(s["shard"] for s in mock_item["shards"])
    for p in plan:
        order = p["disclosure_order"]
        assert order[0] == "S1" and set(order) == {"S1", "S2", "S3", "S4"}
        concat_txt = build_turn_user_text(mock_item, "CONCAT", order, 1)
        assert canonical_concat_body in concat_txt  # CONCAT ALWAYS canonical
        matched_txt = build_turn_user_text(mock_item, "CONCAT_MATCHED", order, 1)
        expected_matched_body = "\n".join(dict((s["shard_id"], s["shard"]) for s in mock_item["shards"])[sid] for sid in order)
        assert expected_matched_body in matched_txt
        assert concat_txt.endswith(FINAL_SUFFIX) and matched_txt.endswith(FINAL_SUFFIX)
        # Check same FINAL_SUFFIX across all protocols
        full_txt = build_turn_user_text(mock_item, "FULL", order, 1)
        dyn_f_last = build_turn_user_text(mock_item, "DYN_FREE", order, 4)
        dyn_r_last = build_turn_user_text(mock_item, "DYN_REQUEST", order, 4)
        assert all(t.endswith(FINAL_SUFFIX) for t in (full_txt, concat_txt, matched_txt, dyn_f_last, dyn_r_last))
        # Turn 1 must start with TASK_PREAMBLE (no system role needed), SCOPE_GUARD uses SCOPE_GUARD_PREAMBLE
        assert build_turn_user_text(mock_item, "DYN_REQUEST", order, 1).startswith(TASK_PREAMBLE)
        assert build_turn_user_text(mock_item, "DYN_REQUEST_GUARD", order, 1).startswith(SCOPE_GUARD_PREAMBLE)
        # Nonfinal request suffix parity
        assert build_turn_user_text(mock_item, "DYN_REQUEST", order, 2).endswith(NONFINAL_REQUEST_SUFFIX)
        assert not build_turn_user_text(mock_item, "DYN_FREE", order, 2).endswith(NONFINAL_REQUEST_SUFFIX)

    # 2. Determinism of disclosure plan & seeds
    plan2 = build_disclosure_plan([mock_item], seed_base=20261024, reps=3)
    assert plan == plan2

    # 3. Robust <think> splitter checks
    # Case A: prompt injected <think>, output starts directly with thoughts + </think>
    s_a = split_thinking_response("一步步分析...\n</think>\n当前缺少边界条件，请补充。", prompt_text="user...\n<think>\n", finish_reason="stop")
    assert s_a["reasoning_trace"] == "一步步分析..."
    assert s_a["visible_reply"] == "当前缺少边界条件，请补充。"
    assert not s_a["thinking_truncated"] and not s_a["empty_truncated"]

    # Case B: truncated inside prompt-injected <think> with finish_reason="length"
    s_b = split_thinking_response("还在推导中但没写完...", prompt_text="user...\n<think>\n", finish_reason="length")
    assert s_b["visible_reply"] == ""
    assert s_b["thinking_truncated"] and s_b["empty_truncated"]
    assert safe_history_assistant_text(s_b, "empty") == ""
    assert safe_history_assistant_text(s_b, "placeholder") != ""

    # Case B2: Ministral-3-14B-Reasoning [THINK]...[/THINK] tags
    s_m1 = split_thinking_response("[THINK]先分析自由度...[/THINK]目前条件不足，需补充方程。", finish_reason="stop")
    assert s_m1["reasoning_trace"] == "先分析自由度..."
    assert s_m1["visible_reply"] == "目前条件不足，需补充方程。"
    assert not s_m1["thinking_truncated"]

    s_m2 = split_thinking_response("[THINK]思考未完成", finish_reason="length")
    assert s_m2["visible_reply"] == "" and s_m2["thinking_truncated"] and s_m2["empty_truncated"]

    # Case C: nested braces inside \boxed{...}
    boxes = extract_boxed_answers(r"综上可得答案为 $\boxed{\frac{\sqrt{5}}{5}}$。")
    assert boxes == [r"\frac{\sqrt{5}}{5}"], boxes

    print("protocol_utils.py self-tests PASSED.")


def generate_disclosure_plan_file(input_path: Path, output_path: Path, seed_base: int = 20261024, reps: int = 3) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    with input_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                items.append(json.loads(s))
    plan = build_disclosure_plan(items, seed_base=seed_base, reps=reps)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("wb") as fh:
        for row in plan:
            fh.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8"))
    return plan


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--build-plan", action="store_true")
    parser.add_argument("--input", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--seed-base", type=int, default=20261024)
    parser.add_argument("--reps", type=int, default=3)
    args = parser.parse_args()
    if args.self_test:
        _run_self_tests()
    elif args.build_plan:
        assert args.input is not None and args.output is not None, "--input and --output required with --build-plan"
        rows = generate_disclosure_plan_file(args.input, args.output, seed_base=args.seed_base, reps=args.reps)
        sha = hashlib.sha256(args.output.read_bytes()).hexdigest()
        print(f"Wrote {len(rows)} plan entries to {args.output} (sha256={sha})")
