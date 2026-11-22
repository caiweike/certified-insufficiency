#!/usr/bin/env python3
"""Local OpenAI-compatible API runner for Math-LiC Stage 2A sharding & deterministic alignment audit.

Supports DeepSeek V4.1 (model='deepseek-flash'), DeepSeek-V4-Pro ('deepseek-v4-pro'),
OpenAI, DashScope (Qwen-Max), and any OpenAI-compatible endpoint.
"""

from __future__ import annotations

import argparse
import json
import os
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

from audit_alignment import audit_shard_candidate
from run_colab_a100_sharding import (
    build_pass1_messages,
    build_repair_messages,
    extract_json_object,
    read_jsonl,
)


SCRIPT_DIR = Path(__file__).resolve().parent


class CapabilityState:
    """Thread-safe tracker for endpoint optional kwargs (reasoning_effort, extra_thinking, json_object)."""

    def __init__(
        self,
        use_reasoning_effort: str | None,
        use_json_object: bool,
        disable_thinking_extra_body: bool = False,
    ) -> None:
        self._lock = threading.Lock()
        self.reasoning_effort = use_reasoning_effort
        self.use_json_object = use_json_object
        self.disable_thinking_extra_body = disable_thinking_extra_body

    def disable_reasoning_effort(self) -> None:
        with self._lock:
            self.reasoning_effort = None

    def disable_thinking_extra(self) -> None:
        with self._lock:
            self.disable_thinking_extra_body = False

    def disable_json_object(self) -> None:
        with self._lock:
            self.use_json_object = False


def resolve_reasoning_effort(model: str, setting: str) -> tuple[str | None, bool]:
    if setting in ("off", "auto"):
        # Omit optional reasoning_effort/thinking fields for clean non-thinking default
        # (some OpenAI-compatible gateways treat literal 'none' as enabling reasoning).
        return None, False
    if setting in ("none", "low", "medium", "high", "max"):
        return setting, (setting == "none" and "deepseek" in model.lower())
    return None, False


def call_chat_completion(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    max_tokens: int,
    temperature: float,
    caps: CapabilityState,
    max_api_retries: int = 3,
) -> tuple[str, dict[str, int]]:
    last_exc: Exception | None = None
    for attempt in range(max_api_retries):
        kwargs: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": max_tokens,
            "temperature": temperature,
        }
        if caps.use_json_object:
            kwargs["response_format"] = {"type": "json_object"}
        if caps.reasoning_effort is not None:
            kwargs["reasoning_effort"] = caps.reasoning_effort
        if caps.disable_thinking_extra_body:
            kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

        try:
            resp = client.chat.completions.create(**kwargs)
            msg = resp.choices[0].message
            content = msg.content or ""
            usage_obj = getattr(resp, "usage", None)
            usage = {
                "prompt_tokens": int(getattr(usage_obj, "prompt_tokens", 0) or 0),
                "completion_tokens": int(getattr(usage_obj, "completion_tokens", 0) or 0),
                "reasoning_tokens": int(
                    getattr(getattr(usage_obj, "completion_tokens_details", None), "reasoning_tokens", 0) or 0
                ),
                "prompt_cache_hit_tokens": int(
                    getattr(usage_obj, "prompt_cache_hit_tokens", 0)
                    or getattr(getattr(usage_obj, "prompt_tokens_details", None), "cached_tokens", 0)
                    or 0
                ),
            }
            return content, usage
        except Exception as exc:
            last_exc = exc
            err_lower = str(exc).lower()
            status_code = getattr(exc, "status_code", None)
            # Graceful capability downgrade if endpoint rejects optional fields.
            if "thinking" in err_lower and caps.disable_thinking_extra_body:
                caps.disable_thinking_extra()
                continue
            if "reasoning_effort" in err_lower and caps.reasoning_effort is not None:
                caps.disable_reasoning_effort()
                continue
            if "response_format" in err_lower and caps.use_json_object:
                caps.disable_json_object()
                continue
            if status_code == 400:
                if caps.disable_thinking_extra_body:
                    caps.disable_thinking_extra()
                    continue
                if caps.reasoning_effort is not None:
                    caps.disable_reasoning_effort()
                    continue
                if caps.use_json_object:
                    caps.disable_json_object()
                    continue
            # Do not retry authentication or bad-model errors.
            if any(k in err_lower for k in ("401", "invalid api key", "authentication", "model_not_found")):
                raise
            if attempt + 1 < max_api_retries:
                time.sleep(min(2 ** attempt, 8))
    assert last_exc is not None
    raise last_exc


def evaluate_raw_response(question: str, raw_text: str) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    parsed, parse_err = extract_json_object(raw_text)
    if parsed is None:
        return None, {
            "status": "PARSE_ERROR",
            "alignment_passed": False,
            "errors": [{"code": "JSON_PARSE_ERROR", "message": parse_err or "Failed to parse JSON"}],
            "warnings": [],
        }
    return parsed, audit_shard_candidate(question, parsed)


def process_stage2a_with_client(
    client: Any,
    items: list[dict[str, Any]],
    system_prompt: str,
    model: str,
    max_tokens: int,
    temperature: float,
    workers: int,
    max_repair_retries: int,
    caps: CapabilityState,
    existing_passed: dict[str, dict[str, Any]] | None = None,
    repair_model: str | None = None,
    repair_caps: CapabilityState | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int], int]:
    existing_passed = existing_passed or {}
    results: list[dict[str, Any] | None] = [None] * len(items)
    usage_totals = Counter()
    pending_indices: list[int] = []

    for idx, item in enumerate(items):
        iid = item["item_id"]
        if iid in existing_passed:
            results[idx] = existing_passed[iid]
        else:
            pending_indices.append(idx)

    def run_pass1(idx: int) -> tuple[int, dict[str, Any], dict[str, int]]:
        item = items[idx]
        msgs = build_pass1_messages(system_prompt, item["question"])
        raw_text, usage = call_chat_completion(
            client, model, msgs, max_tokens=max_tokens, temperature=temperature, caps=caps
        )
        parsed, audit = evaluate_raw_response(item["question"], raw_text)
        rec = {
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
            "sharding_model": model,
            "attempts": 1,
            "raw_output": raw_text,
            "parsed_output": parsed,
            "audit": audit,
        }
        return idx, rec, usage

    # Warm up prefix cache & validate endpoint on item 0 synchronously before fan-out.
    if pending_indices:
        first_idx = pending_indices[0]
        t_first = time.time()
        print(f"[Pass 1] Warming up item 1/{len(pending_indices)} ({items[first_idx]['item_id']})...", flush=True)
        _, first_rec, first_usage = run_pass1(first_idx)
        results[first_idx] = first_rec
        usage_totals.update(first_usage)
        print(
            f"[Pass 1 warmup] 1/{len(pending_indices)} {first_rec['item_id']} -> "
            f"{first_rec['audit']['status']} (passed={first_rec['audit']['alignment_passed']}) "
            f"in {time.time() - t_first:.1f}s",
            flush=True,
        )

        rest = pending_indices[1:]
        if rest:
            done_count = 1
            t_batch = time.time()
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futs = {pool.submit(run_pass1, i): i for i in rest}
                for fut in as_completed(futs):
                    i = futs[fut]
                    try:
                        _, rec, usage = fut.result()
                        results[i] = rec
                        usage_totals.update(usage)
                        status_msg = f"{rec['audit']['status']} passed={rec['audit']['alignment_passed']}"
                    except Exception as exc:
                        item = items[i]
                        results[i] = {
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
                            "sharding_model": model,
                            "attempts": 1,
                            "raw_output": "",
                            "parsed_output": None,
                            "audit": {
                                "status": "API_ERROR",
                                "alignment_passed": False,
                                "errors": [{"code": "API_ERROR", "message": str(exc)}],
                                "warnings": [],
                            },
                        }
                        status_msg = f"API_ERROR: {exc}"
                    done_count += 1
                    print(
                        f"[Pass 1] {done_count}/{len(pending_indices)} {items[i]['item_id']} -> "
                        f"{status_msg} ({time.time() - t_batch:.1f}s)",
                        flush=True,
                    )

    repair_indices = [
        i
        for i, rec in enumerate(results)
        if rec is not None
        and rec["item_id"] not in existing_passed
        and not rec["audit"]["alignment_passed"]
        and rec["audit"]["status"] != "API_ERROR"
    ]

    if max_repair_retries > 0 and repair_indices:
        eff_repair_model = repair_model or model
        eff_repair_caps = repair_caps or caps
        print(
            f"Running Pass 2 alignment repair on {len(repair_indices)} items "
            f"(model={eff_repair_model}, reasoning_effort={eff_repair_caps.reasoning_effort})...",
            flush=True,
        )

        def run_pass2(idx: int) -> tuple[int, str, dict[str, Any] | None, dict[str, Any], dict[str, int]]:
            rec = results[idx]
            assert rec is not None
            msgs = build_repair_messages(
                system_prompt,
                rec["question"],
                rec["raw_output"],
                rec["audit"]["errors"],
            )
            raw_text, usage = call_chat_completion(
                client,
                eff_repair_model,
                msgs,
                max_tokens=max_tokens,
                temperature=temperature,
                caps=eff_repair_caps,
            )
            parsed, audit = evaluate_raw_response(rec["question"], raw_text)
            return idx, raw_text, parsed, audit, usage

        done_rep = 0
        t_rep = time.time()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(run_pass2, i): i for i in repair_indices}
            for fut in as_completed(futs):
                i = futs[fut]
                rec = results[i]
                assert rec is not None
                done_rep += 1
                try:
                    _, raw_text, parsed, audit, usage = fut.result()
                    usage_totals.update(usage)
                    rec["pass1_failed_errors"] = rec["audit"]["errors"]
                    rec["attempts"] = 2
                    rec["raw_output"] = raw_text
                    rec["parsed_output"] = parsed
                    rec["audit"] = audit
                    print(
                        f"[Pass 2] {done_rep}/{len(repair_indices)} {rec['item_id']} -> "
                        f"{audit['status']} passed={audit['alignment_passed']} ({time.time() - t_rep:.1f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    rec["pass2_api_error"] = str(exc)
                    print(
                        f"[Pass 2] {done_rep}/{len(repair_indices)} {rec['item_id']} -> API_ERROR: {exc}",
                        flush=True,
                    )

    final_results = [r for r in results if r is not None]
    return final_results, dict(usage_totals), len(repair_indices)


def build_summary(
    results: list[dict[str, Any]],
    model: str,
    repair_count: int,
    usage_totals: dict[str, int],
    elapsed_sec: float,
) -> dict[str, Any]:
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

    return {
        "model": model,
        "total_items": len(results),
        "pass1_failures_sent_to_repair": repair_count,
        "outcome_counts": dict(sorted(outcome_counts.items())),
        "passed_candidates_by_bucket_and_split": dict(sorted(bucket_candidate_passed.items())),
        "passed_shard_count_distribution": dict(sorted(shard_count_dist.items())),
        "unshardable_reason_codes": dict(sorted(unshardable_reasons.items())),
        "remaining_alignment_error_codes": dict(sorted(error_code_counts.items())),
        "token_usage": usage_totals,
        "elapsed_total_sec": round(elapsed_sec, 2),
    }


def run_mock_test() -> None:
    """Verify full Pass 1 + Pass 2 + summary flow offline using a mock OpenAI client."""
    import tempfile

    class _FakeMsg:
        def __init__(self, content: str) -> None:
            self.content = content

    class _FakeChoice:
        def __init__(self, content: str) -> None:
            self.message = _FakeMsg(content)

    class _FakeUsage:
        def __init__(self) -> None:
            self.prompt_tokens = 1000
            self.completion_tokens = 200
            self.prompt_cache_hit_tokens = 800

    class _FakeResp:
        def __init__(self, content: str) -> None:
            self.choices = [_FakeChoice(content)]
            self.usage = _FakeUsage()

    class _FakeCompletions:
        def create(self, **kwargs: Any) -> _FakeResp:
            msgs = kwargs["messages"]
            if len(msgs) == 2:
                # Pass 1: fail intentionally with altered formula so Pass 2 repair runs.
                return _FakeResp(
                    json.dumps(
                        {
                            "status": "CANDIDATE",
                            "initial_segments": ["已知函数", "求 $b$ 的值。"],
                            "initial_shard": "已知函数 $g(x)$，求 $b$ 的值。",
                            "shards": [
                                {"shard_id": "S2", "segments": ["g(x)=|x-1|+|x-b|"], "shard": "$g(x)=|x-2|+|x-b|$。"},
                                {"shard_id": "S3", "segments": ["(b>0)"], "shard": "参数满足 $(b>0)$。"},
                                {"shard_id": "S4", "segments": ["若 $g(1)=2$"], "shard": "若 $g(1)=2$。"},
                            ],
                        },
                        ensure_ascii=False,
                    )
                )
            return _FakeResp(
                json.dumps(
                    {
                        "status": "CANDIDATE",
                        "initial_segments": ["已知函数", "求 $b$ 的值。"],
                        "initial_shard": "已知函数 $g(x)$，求 $b$ 的值。",
                        "shards": [
                            {"shard_id": "S2", "segments": ["g(x)=|x-1|+|x-b|"], "shard": "$g(x)=|x-1|+|x-b|$。"},
                            {"shard_id": "S3", "segments": ["(b>0)"], "shard": "参数满足 $(b>0)$。"},
                            {"shard_id": "S4", "segments": ["若 $g(1)=2$"], "shard": "若 $g(1)=2$。"},
                        ],
                    },
                    ensure_ascii=False,
                )
            )

    class _FakeChat:
        def __init__(self) -> None:
            self.completions = _FakeCompletions()

    class _FakeClient:
        def __init__(self) -> None:
            self.chat = _FakeChat()

    fake_items = [
        {
            "item_id": "mock-1",
            "status_code": "SELECTED_DEV",
            "macro_subfield": "G1_Func_Calc_Trig",
            "provisional_mechanism": "TypeA_UniqueValue",
            "slot_index": 1,
            "question": "已知函数 $g(x)=|x-1|+|x-b|(b>0)$，若 $g(1)=2$，求 $b$ 的值。",
            "question_sha256": "0" * 64,
        }
    ]
    sys_prompt = (SCRIPT_DIR / "prompt_math_lic_v2_6.txt").read_text(encoding="utf-8").strip()
    caps = CapabilityState("low", True)
    results, usage, repair_cnt = process_stage2a_with_client(
        _FakeClient(),
        fake_items,
        sys_prompt,
        "deepseek-flash",
        max_tokens=4096,
        temperature=0.0,
        workers=2,
        max_repair_retries=1,
        caps=caps,
    )
    assert len(results) == 1
    assert repair_cnt == 1
    assert results[0]["attempts"] == 2
    assert results[0]["audit"]["alignment_passed"] is True
    assert usage["prompt_cache_hit_tokens"] == 1600
    with tempfile.TemporaryDirectory() as tmp:
        out_p = Path(tmp) / "mock_out.jsonl"
        sum_p = Path(tmp) / "mock_sum.json"
        out_p.write_bytes((json.dumps(results[0], ensure_ascii=False) + "\n").encode("utf-8"))
        summary = build_summary(results, "deepseek-flash", repair_cnt, usage, 0.1)
        sum_p.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2) + "\n").encode("utf-8"))
    print("run_api_sharding.py mock end-to-end test PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Stage 2A Math-LiC sharding via OpenAI-compatible API.")
    parser.add_argument("--input", type=Path, default=SCRIPT_DIR / "stage2_input_candidates.jsonl")
    parser.add_argument("--prompt", type=Path, default=SCRIPT_DIR / "prompt_math_lic_v2_6.txt")
    parser.add_argument("--output", type=Path, default=SCRIPT_DIR / "stage2a_api_sharded.jsonl")
    parser.add_argument("--summary", type=Path, default=SCRIPT_DIR / "stage2a_api_summary.json")
    parser.add_argument("--model", type=str, default="deepseek-flash", help="e.g. deepseek-flash, deepseek-v4-pro, gpt-6.1-sol, qwen3.8-max")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com",
    )
    parser.add_argument("--api-key", type=str, default=None, help="Or set DEEPSEEK_API_KEY / OPENAI_API_KEY env var.")
    parser.add_argument(
        "--reasoning-effort",
        type=str,
        default="auto",
        choices=["auto", "off", "none", "low", "medium", "high", "max"],
        help="'auto' disables thinking ('none') on Pass 1 for fast ~1-2s sharding.",
    )
    parser.add_argument(
        "--repair-reasoning-effort",
        type=str,
        default="auto",
        choices=["auto", "off", "none", "low", "medium", "high", "max"],
        help="'auto' enables brief thinking ('low') on Pass 2 repair for DeepSeek V4/V4.1.",
    )
    parser.add_argument("--repair-model", type=str, default=None, help="Optional stronger model for Pass 2 repair only.")
    parser.add_argument("--max-tokens", type=int, default=4096)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=45.0)
    parser.add_argument("--workers", type=int, default=20)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--status-filter",
        type=str,
        default="SELECTED_DEV",
        choices=["ALL", "SELECTED_DEV", "SELECTED_TEST", "ELIGIBLE_RESERVE"],
        help="Default is SELECTED_DEV for fast safety verification before running ALL.",
    )
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Reuse alignment-passed records already in --output.")
    parser.add_argument("--no-json-object", action="store_true")
    parser.add_argument("--mock-test", action="store_true", help="Run offline mock verification without API calls.")
    args = parser.parse_args()

    if args.mock_test:
        run_mock_test()
        return

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "Missing API key. Set $env:DEEPSEEK_API_KEY='your_key' (or $env:OPENAI_API_KEY) before running."
        )

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    items = read_jsonl(args.input)
    if args.status_filter != "ALL":
        items = [x for x in items if x["status_code"] == args.status_filter]
    if args.limit > 0:
        items = items[: args.limit]

    existing_passed: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        for rec in read_jsonl(args.output):
            if rec.get("audit", {}).get("alignment_passed"):
                existing_passed[rec["item_id"]] = rec
        print(f"Resuming: reusing {len(existing_passed)} passed items from {args.output}")

    effort, extra_disable = resolve_reasoning_effort(args.model, args.reasoning_effort)
    caps = CapabilityState(
        use_reasoning_effort=effort,
        use_json_object=not args.no_json_object,
        disable_thinking_extra_body=extra_disable,
    )
    eff_repair_model = args.repair_model or args.model
    if args.repair_reasoning_effort == "auto":
        m_rep = eff_repair_model.lower()
        rep_effort = "low" if ("deepseek-flash" in m_rep or "deepseek-v4" in m_rep) else effort
        rep_disable = False
    else:
        rep_effort, rep_disable = resolve_reasoning_effort(eff_repair_model, args.repair_reasoning_effort)
    repair_caps = CapabilityState(
        use_reasoning_effort=rep_effort,
        use_json_object=not args.no_json_object,
        disable_thinking_extra_body=rep_disable,
    )
    print(
        f"Running Stage 2A API sharding: items={len(items)} filter={args.status_filter} "
        f"model={args.model} base_url={args.base_url} pass1_effort={effort} "
        f"pass2_model={eff_repair_model} pass2_effort={rep_effort} workers={args.workers} timeout={args.timeout}s",
        flush=True,
    )

    t0 = time.time()
    results, usage_totals, repair_cnt = process_stage2a_with_client(
        client=client,
        items=items,
        system_prompt=system_prompt,
        model=args.model,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workers=args.workers,
        max_repair_retries=args.max_retries,
        caps=caps,
        existing_passed=existing_passed,
        repair_model=eff_repair_model,
        repair_caps=repair_caps,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        for rec in results:
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            fh.write(line.encode("utf-8"))

    summary = build_summary(results, args.model, repair_cnt, usage_totals, time.time() - t0)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_bytes(
        (json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8")
    )
    print("\n=== Stage 2A API Sharding & Audit Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
