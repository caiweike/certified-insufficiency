#!/usr/bin/env python3
"""Batch Stage 2A Math-LiC sharding & deterministic alignment audit via vLLM (Colab A100)."""

from __future__ import annotations

import argparse
import inspect
import json
import re
import time
from collections import Counter
from pathlib import Path
from typing import Any

from audit_alignment import VALID_REASON_CODES, audit_shard_candidate


SHARD_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": ["CANDIDATE", "UNSHARDABLE"]},
        "initial_segments": {"type": "array", "items": {"type": "string"}},
        "initial_shard": {"type": "string"},
        "shards": {
            "type": "array",
            "items": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "shard_id": {"type": "string"},
                    "segments": {"type": "array", "items": {"type": "string"}},
                    "shard": {"type": "string"},
                },
                "required": ["shard_id", "segments", "shard"],
            },
        },
        "reason_code": {"type": "string", "enum": sorted(VALID_REASON_CODES)},
        "reason": {"type": "string"},
    },
    "required": ["status"],
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def _restore_latex_control_chars(obj: Any) -> Any:
    """Restore accidental JSON-parsed control chars (\\b, \\f, \\r, \\t, and LaTeX \\n*) back to backslashes."""
    if isinstance(obj, str):
        out = (
            obj.replace("\x08", "\\b")
            .replace("\x0c", "\\f")
            .replace("\x0d", "\\r")
            .replace("\t", "\\t")
        )
        out = re.sub(r"\n(?=(?:eq|abla|ot\b|u\b|atural))", r"\\n", out)
        return out
    if isinstance(obj, list):
        return [_restore_latex_control_chars(x) for x in obj]
    if isinstance(obj, dict):
        return {k: _restore_latex_control_chars(v) for k, v in obj.items()}
    return obj


def extract_json_object(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_text.strip()
    # Strip <think>...</think> if a thinking tag leaked despite enable_thinking=False.
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()
    fence_match = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    if fence_match:
        text = fence_match.group(1).strip()
    else:
        first = text.find("{")
        last = text.rfind("}")
        if first != -1 and last != -1 and last > first:
            text = text[first : last + 1]
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        # Repair unescaped LaTeX backslashes (e.g. \sin, \geq, \left, \pi) inside JSON strings.
        repaired = re.sub(r'(?<!\\)((?:\\\\)*)\\(?![\\"/bfnrtu])', r"\1\\\\", text)
        try:
            parsed = json.loads(repaired)
        except json.JSONDecodeError:
            return None, f"JSONDecodeError: {exc}"
    parsed = _restore_latex_control_chars(parsed)
    if not isinstance(parsed, dict):
        return None, f"Expected JSON object, got {type(parsed).__name__}"
    return parsed, None


def render_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    model_name: str,
    enable_thinking: bool = False,
) -> str:
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_name.lower():
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


def make_sampling_params(
    sampling_params_cls: Any,
    vllm_module: Any,
    max_tokens: int,
    use_guided_json: bool,
) -> Any:
    kwargs: dict[str, Any] = {
        "temperature": 0.0,
        "top_p": 1.0,
        "max_tokens": max_tokens,
    }
    if not use_guided_json:
        return sampling_params_cls(**kwargs)

    sig_params = inspect.signature(sampling_params_cls.__init__).parameters
    # Support newer vLLM StructuredOutputsParams and older GuidedDecodingParams.
    structured_cls = getattr(vllm_module.sampling_params, "StructuredOutputsParams", None)
    guided_cls = getattr(vllm_module.sampling_params, "GuidedDecodingParams", None)

    if "structured_outputs" in sig_params and structured_cls is not None:
        kwargs["structured_outputs"] = structured_cls(json=SHARD_JSON_SCHEMA)
    elif "guided_decoding" in sig_params and guided_cls is not None:
        kwargs["guided_decoding"] = guided_cls(json=SHARD_JSON_SCHEMA)
    return sampling_params_cls(**kwargs)


def build_pass1_messages(system_prompt: str, question: str) -> list[dict[str, str]]:
    user_msg = (
        "请严格按系统提示词规则，对下面这道数学题进行分片并只输出合法 JSON"
        "（注意 JSON 字符串中的所有 LaTeX 反斜杠必须双写转义，例如 \\\\frac、\\\\sin、\\\\geq）：\n\n"
        f"instruction:\n{question}"
    )
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
    ]


def build_repair_messages(
    system_prompt: str,
    question: str,
    prev_raw: str,
    audit_errors: list[dict[str, str]],
) -> list[dict[str, str]]:
    err_lines = "\n".join(f"- [{e['code']}] {e['message']}" for e in audit_errors)
    user_msg = (
        f"请严格按系统提示词规则，对下面这道数学题进行分片并只输出合规 JSON：\n\ninstruction:\n{question}"
    )
    repair_msg = (
        "你上一次输出的 JSON 未通过确定性对齐审计，错误如下：\n"
        f"{err_lines}\n\n"
        "请逐条修复以上错误并重新输出完整 JSON（特别注意：\n"
        "1. initial_segments 与 segments 必须是原文 instruction 的逐字连续子串：绝对不要把原文的英文逗号 ','、句号 '.' 改成中文 '，' '。'，也不要往 segments 里多加原文该处没有的 '$' 符号；\n"
        "2. initial_segments 必须完整包含 S1 求解目标中的数学表达式（如不等式 f(x)<0）；\n"
        "3. 每个分片文本（initial_shard / shard）必须逐字包含其对应 segments 的文字与公式，不要改写（例如不要把“当 $a=0$ 时”改写成“考虑 $a=0$ 的情况”，也不要删改公式空格或把 sin A 改成 \\sin A）；\n"
        "4. 严禁在等号或不等号处把单个方程拆成两片，也不要把题设方程、不等式、圆/椭圆方程或“各项均为正数/公差不为零”塞进 S1，也不要在 S1 留下残缺空壳“当时，”；\n"
        "5. shards 中的 shard_id 必须从 S2 开始依次编号；\n"
        "6. 输出合法 JSON 时，所有 LaTeX 反斜杠必须双写转义（如 \\\\frac、\\\\sin、\\\\geq）；\n"
        "7. 重要计数提醒：总片数 = 1(S1) + 后续片数(S2..Sk)。只要后续至少有 2 个条件片（S2 和 S3），总片数即为 1+2=3 片，已合格！严禁把原本可拆成 S1+S2+S3 的题目错算成总片数为 2 而逃避改判为 UNSHARDABLE，也不要在修复时把 S2..Sk 的方程或正数限制塞进 S1。）"
    )
    clean_prev = re.sub(r"<think>.*?</think>", "", prev_raw, flags=re.DOTALL).strip()
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": clean_prev},
        {"role": "user", "content": repair_msg},
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2A Math-LiC sharding on vLLM.")
    parser.add_argument("--input", type=Path, default=Path("stage2_input_candidates.jsonl"))
    parser.add_argument("--prompt", type=Path, default=Path("prompt_math_lic_v2_6.txt"))
    parser.add_argument("--output", type=Path, default=Path("stage2a_sharded_candidates.jsonl"))
    parser.add_argument("--summary", type=Path, default=Path("stage2a_audit_summary.json"))
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-32B-AWQ")
    parser.add_argument("--dtype", type=str, default="auto")
    parser.add_argument("--quantization", type=str, default="auto")
    parser.add_argument("--max-model-len", type=int, default=8192)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    parser.add_argument("--max-tokens", type=int, default=1536)
    parser.add_argument(
        "--status-filter",
        type=str,
        default="ALL",
        choices=["ALL", "SELECTED_DEV", "SELECTED_TEST", "ELIGIBLE_RESERVE"],
        help="Optionally shard only a subset (e.g. SELECTED_DEV for a fast pilot).",
    )
    parser.add_argument("--limit", type=int, default=0, help="Optional cap on items for quick smoke tests.")
    parser.add_argument("--no-guided-json", action="store_true")
    parser.add_argument("--enforce-eager", action="store_true", help="Disable CUDA graphs to save VRAM on 24GB GPUs like L4.")
    parser.add_argument("--enable-thinking", action="store_true", help="Enable Qwen3 thinking mode (recommended with --no-guided-json).")
    parser.add_argument("--max-retries", type=int, default=1)
    args = parser.parse_args()

    import torch  # noqa: PLC0415
    import vllm  # noqa: PLC0415
    from vllm import LLM, SamplingParams  # noqa: PLC0415

    resolved_dtype = args.dtype
    resolved_quant = args.quantization
    if torch.cuda.is_available():
        major, minor = torch.cuda.get_device_capability(0)
        if major < 8:
            if resolved_dtype == "auto":
                resolved_dtype = "float16"
                print(f"[GPU SM {major}.{minor}] T4/Turing does not support bfloat16; using dtype='float16'.")
            if resolved_quant.lower() == "auto" and "awq" in args.model.lower():
                resolved_quant = "awq"
                print(f"[GPU SM {major}.{minor}] Using quantization='awq' (instead of awq_marlin).")

    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    items = read_jsonl(args.input)
    if args.status_filter != "ALL":
        items = [x for x in items if x["status_code"] == args.status_filter]
    if args.limit > 0:
        items = items[: args.limit]

    print(f"Loaded {len(items)} items (filter={args.status_filter}). Initializing vLLM model {args.model}...")
    t0 = time.time()
    llm_kwargs: dict[str, Any] = {
        "model": args.model,
        "dtype": resolved_dtype,
        "max_model_len": args.max_model_len,
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "trust_remote_code": True,
    }
    if resolved_quant and resolved_quant.lower() not in ("none", "auto"):
        llm_kwargs["quantization"] = resolved_quant
    if args.enforce_eager:
        llm_kwargs["enforce_eager"] = True
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    sampling_params = make_sampling_params(
        SamplingParams,
        vllm,
        max_tokens=args.max_tokens,
        use_guided_json=not args.no_guided_json,
    )
    print(f"Model ready in {time.time() - t0:.1f}s. Running Pass 1 on {len(items)} items...")

    # Pass 1
    t_gen = time.time()
    pass1_prompts = [
        render_chat_prompt(
            tokenizer,
            build_pass1_messages(system_prompt, x["question"]),
            args.model,
            enable_thinking=args.enable_thinking,
        )
        for x in items
    ]
    pass1_outputs = llm.generate(pass1_prompts, sampling_params)
    print(f"Pass 1 generation finished in {time.time() - t_gen:.1f}s.")

    results: list[dict[str, Any]] = []
    repair_indices: list[int] = []

    for idx, (item, out) in enumerate(zip(items, pass1_outputs)):
        raw_text = out.outputs[0].text
        parsed, parse_err = extract_json_object(raw_text)
        if parsed is None:
            audit = {
                "status": "PARSE_ERROR",
                "alignment_passed": False,
                "errors": [{"code": "JSON_PARSE_ERROR", "message": parse_err or "Failed to parse JSON"}],
                "warnings": [],
            }
        else:
            audit = audit_shard_candidate(item["question"], parsed)

        record = {
            "item_id": item["item_id"],
            "status_code": item["status_code"],
            "macro_subfield": item["macro_subfield"],
            "provisional_mechanism": item["provisional_mechanism"],
            "slot_index": item.get("slot_index"),
            "reserve_rank": item.get("reserve_rank"),
            "source_dataset": item.get("source_dataset"),
            "subfield": item.get("subfield"),
            "answer_type": item.get("answer_type"),
            "question": item["question"],
            "question_sha256": item["question_sha256"],
            "final_answer": item.get("final_answer"),
            "solution": item.get("solution"),
            "sharding_model": args.model,
            "attempts": 1,
            "raw_output": raw_text,
            "parsed_output": parsed,
            "audit": audit,
        }
        results.append(record)
        if not audit["alignment_passed"]:
            repair_indices.append(idx)

    # Pass 2 repair (optional)
    if args.max_retries > 0 and repair_indices:
        print(f"Running Pass 2 alignment repair on {len(repair_indices)} failed items...")
        t_rep = time.time()
        repair_prompts = [
            render_chat_prompt(
                tokenizer,
                build_repair_messages(
                    system_prompt,
                    results[idx]["question"],
                    results[idx]["raw_output"],
                    results[idx]["audit"]["errors"],
                ),
                args.model,
                enable_thinking=args.enable_thinking,
            )
            for idx in repair_indices
        ]
        repair_outputs = llm.generate(repair_prompts, sampling_params)
        print(f"Pass 2 finished in {time.time() - t_rep:.1f}s.")

        for idx, out in zip(repair_indices, repair_outputs):
            raw_text = out.outputs[0].text
            parsed, parse_err = extract_json_object(raw_text)
            if parsed is None:
                audit = {
                    "status": "PARSE_ERROR",
                    "alignment_passed": False,
                    "errors": [{"code": "JSON_PARSE_ERROR", "message": parse_err or "Failed to parse JSON"}],
                    "warnings": [],
                }
            else:
                audit = audit_shard_candidate(results[idx]["question"], parsed)
            results[idx]["pass1_failed_errors"] = results[idx]["audit"]["errors"]
            results[idx]["attempts"] = 2
            results[idx]["raw_output"] = raw_text
            results[idx]["parsed_output"] = parsed
            results[idx]["audit"] = audit

    # Save JSONL and summary
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        for rec in results:
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            fh.write(line.encode("utf-8"))

    outcome_counts = Counter()
    bucket_candidate_passed = Counter()
    shard_count_dist = Counter()
    error_code_counts = Counter()
    unshardable_reasons = Counter()

    for rec in results:
        audit = rec["audit"]
        st = audit["status"]
        passed = audit["alignment_passed"]
        outcome_counts[f"{st}__passed={passed}"] += 1
        if st == "CANDIDATE" and passed:
            key = f"{rec['macro_subfield']}__{rec['provisional_mechanism']}__{rec['status_code']}"
            bucket_candidate_passed[key] += 1
            shard_count_dist[str(audit["total_shards"])] += 1
        elif st == "UNSHARDABLE" and passed:
            unshardable_reasons[audit.get("reason_code", "UNKNOWN")] += 1
        if not passed:
            for e in audit.get("errors", []):
                error_code_counts[e["code"]] += 1

    summary = {
        "model": args.model,
        "total_items": len(results),
        "pass1_failures_sent_to_repair": len(repair_indices),
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "passed_candidates_by_bucket_and_split": dict(sorted(bucket_candidate_passed.items())),
        "passed_shard_count_distribution": dict(sorted(shard_count_dist.items())),
        "unshardable_reason_codes": dict(sorted(unshardable_reasons.items())),
        "remaining_alignment_error_codes": dict(sorted(error_code_counts.items())),
        "elapsed_total_sec": round(time.time() - t0, 2),
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_bytes(
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print("\n=== Stage 2A Sharding & Audit Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
