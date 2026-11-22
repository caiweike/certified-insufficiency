#!/usr/bin/env python3
"""Generate and deterministically verify Stage 2B SymPy certificates (SPEC v2.4).

Per SPEC Section 12.1, defaults to `--status-filter SELECTED_DEV` so verification
rules and certificates are debugged strictly on Dev before promoting/freezing Test.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
SHARDING_DIR = SCRIPT_DIR.parent / "colab_stage2_sharding"
BASELINE_MANIFEST = (
    SCRIPT_DIR.parent / "benchmark_spec_v2_4" / "reference_baseline" / "selection_manifest.json"
)
DEFAULT_INPUT = SHARDING_DIR / "stage2a_api_sharded.jsonl"
DEFAULT_PROMPT = SCRIPT_DIR / "prompt_stage2b_cert_v1.txt"
DEFAULT_OUTPUT = SCRIPT_DIR / "stage2b_verified_records.jsonl"
DEFAULT_SUMMARY = SCRIPT_DIR / "stage2b_summary.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(SHARDING_DIR) not in sys.path:
    sys.path.insert(0, str(SHARDING_DIR))

from run_api_sharding import (  # noqa: E402
    CapabilityState,
    call_chat_completion,
    read_jsonl,
    resolve_reasoning_effort,
)
from stage2b_verifier_core import (  # noqa: E402
    STAGE2B_RULE_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    verify_certificate_record,
)


def _append_report_detail(v_out: dict[str, Any], detail: str) -> None:
    """Append diagnostic detail and keep verification_report_sha256 synchronized."""
    v_out["report"]["details"].append(detail)
    v_out["result"]["verification_report_sha256"] = sha256_json(v_out["report"])


def parse_json_object(raw_text: str) -> tuple[dict[str, Any] | None, str | None]:
    text = raw_text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        obj = json.loads(text)
        return (obj if isinstance(obj, dict) else None), (None if isinstance(obj, dict) else "Root must be JSON object.")
    except json.JSONDecodeError as exc:
        # Attempt minimal single-backslash repair if raw text contained LaTeX escapes.
        fixed = re.sub(r'(?<!\\)\\(?![\\"/bfnrtu])', r"\\\\", text)
        try:
            obj = json.loads(fixed)
            return (obj if isinstance(obj, dict) else None), None
        except json.JSONDecodeError:
            return None, f"JSONDecodeError: {exc}"


def format_item_prompt(rec: dict[str, Any]) -> str:
    po = rec.get("parsed_output") or {}
    shards_block = [f"S1: {po.get('initial_shard', '')}"]
    for s in po.get("shards") or []:
        sid = s.get("shard_id")
        stext = s.get("shard") or s.get("shard_text") or s.get("text") or ""
        shards_block.append(f"{sid}: {stext}")

    sol_text = ""
    sol = rec.get("solution")
    if isinstance(sol, list):
        sol_text = "\n".join(str(x) for x in sol)
    elif sol is not None:
        sol_text = str(sol)

    return (
        f"item_id: {rec['item_id']}\n"
        f"macro_subfield: {rec['macro_subfield']}\n"
        f"provisional_mechanism: {rec['provisional_mechanism']}\n"
        f"official_final_answer: {rec.get('final_answer')}\n\n"
        f"original_question:\n{rec['question']}\n\n"
        f"official_solution:\n{sol_text}\n\n"
        f"candidate_shards:\n" + "\n".join(shards_block)
    )


def build_pass1_messages(system_prompt: str, rec: dict[str, Any]) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_item_prompt(rec)},
    ]


def build_repair_messages(
    system_prompt: str,
    rec: dict[str, Any],
    prev_raw: str,
    fail_reason: str,
    details: list[str],
) -> list[dict[str, str]]:
    err_txt = f"{fail_reason}: " + "; ".join(details)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": format_item_prompt(rec)},
        {"role": "assistant", "content": prev_raw},
        {
            "role": "user",
            "content": (
                "The deterministic SymPy verifier rejected your previous JSON certificate with:\n"
                f"{err_txt}\n\n"
                "Fix the SymPy syntax or witness values cleanly so every remaining shard predicate holds "
                "and target shard Sj evaluates False on the counter-witness. "
                "If a shard Sj really IS mathematically redundant or cannot be honestly formalized without guessing, "
                "report redundant_shards or mark status='UNVERIFIABLE' with the exact reject_reason. "
                "Return ONLY the corrected raw JSON object."
            ),
        },
    ]


def compute_config_sha(prompt_path: Path, model: str, max_tokens: int, reasoning_effort: str | None) -> str:
    cfg = {
        "stage2b_rule_version": STAGE2B_RULE_VERSION,
        "prompt_sha256": hashlib.sha256(prompt_path.read_bytes()).hexdigest(),
        "model": model,
        "max_tokens": max_tokens,
        "reasoning_effort": reasoning_effort,
    }
    return sha256_bytes(canonical_json_bytes(cfg))


def _needs_repair(verify_out: dict[str, Any], cert_obj: dict[str, Any] | None = None) -> bool:
    """Retry certificate-construction / SymPy syntax / witness failures (not intentional rejections)."""
    res = verify_out["result"]
    if res["verify_status"] == "VERIFY_PASS":
        return False
    if res["reject_reason"] in {
        "FAIL_CERTIFICATE_UNAVAILABLE",
        "FAIL_SOURCE_ANSWER_MISMATCH",
    }:
        return True
    if res["reject_reason"] == "FAIL_REDUNDANT_PREMISE":
        # Retry only if the LLM declared NO redundant shards (meaning its witness failed to falsify Sj).
        return bool(cert_obj is not None and not cert_obj.get("redundant_shards"))
    return False


def verify_stage2b_batch(
    client: Any,
    records: list[dict[str, Any]],
    system_prompt: str,
    model: str,
    manifest_sha: str,
    cfg_sha: str,
    max_tokens: int,
    temperature: float,
    workers: int,
    max_repair_retries: int,
    caps: CapabilityState,
    existing_records: dict[str, dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, int]]:
    existing_records = existing_records or {}
    outputs: list[dict[str, Any] | None] = [None] * len(records)
    usage_totals: Counter[str] = Counter()
    pending_indices: list[int] = []

    for idx, rec in enumerate(records):
        iid = rec["item_id"]
        if iid in existing_records:
            outputs[idx] = existing_records[iid]
        elif not (
            rec.get("audit", {}).get("alignment_passed")
            and rec.get("audit", {}).get("status") == "CANDIDATE"
        ):
            # Deterministically fail non-passing Stage 2A records without API calls.
            v_out = verify_certificate_record(rec, None, manifest_sha, cfg_sha)
            outputs[idx] = _assemble_record(rec, None, "", v_out, attempts=0)
        else:
            pending_indices.append(idx)

    def run_one(idx: int) -> tuple[int, dict[str, Any], dict[str, int]]:
        rec = records[idx]
        msgs = build_pass1_messages(system_prompt, rec)
        raw, usage = call_chat_completion(
            client, model, msgs, max_tokens=max_tokens, temperature=temperature, caps=caps
        )
        cert_obj, parse_err = parse_json_object(raw)
        if cert_obj is None:
            v_out = verify_certificate_record(rec, None, manifest_sha, cfg_sha)
            _append_report_detail(v_out, f"LLM_JSON_PARSE_ERROR: {parse_err}")
        else:
            v_out = verify_certificate_record(rec, cert_obj, manifest_sha, cfg_sha)
        return idx, _assemble_record(rec, cert_obj, raw, v_out, attempts=1), usage

    if pending_indices:
        # Item 0 warmup before fan-out
        first = pending_indices[0]
        t0 = time.time()
        print(f"[Stage2B Pass 1] Warming up 1/{len(pending_indices)} ({records[first]['item_id']})...", flush=True)
        _, first_rec, first_usage = run_one(first)
        outputs[first] = first_rec
        usage_totals.update(first_usage)
        res0 = first_rec["stage2b_result"]
        print(
            f"[Stage2B Pass 1 warmup] 1/{len(pending_indices)} {first_rec['item_id']} -> "
            f"{res0['verify_status']} ({res0['reject_reason'] or 'OK'}) in {time.time()-t0:.1f}s",
            flush=True,
        )

        rest = pending_indices[1:]
        if rest:
            done = 1
            t_b = time.time()
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futs = {pool.submit(run_one, i): i for i in rest}
                for fut in as_completed(futs):
                    i = futs[fut]
                    done += 1
                    try:
                        _, assembled, usage = fut.result()
                        outputs[i] = assembled
                        usage_totals.update(usage)
                        r = assembled["stage2b_result"]
                        print(
                            f"[Stage2B Pass 1] {done}/{len(pending_indices)} {assembled['item_id']} -> "
                            f"{r['verify_status']} ({r['reject_reason'] or 'OK'}) ({time.time()-t_b:.1f}s)",
                            flush=True,
                        )
                    except Exception as exc:
                        rec = records[i]
                        v_out = verify_certificate_record(rec, None, manifest_sha, cfg_sha)
                        _append_report_detail(v_out, f"API_ERROR: {exc}")
                        outputs[i] = _assemble_record(rec, None, "", v_out, attempts=1, api_error=str(exc))
                        print(
                            f"[Stage2B Pass 1] {done}/{len(pending_indices)} {rec['item_id']} -> API_ERROR: {exc}",
                            flush=True,
                        )

    repair_indices = [
        i
        for i in pending_indices
        if outputs[i] is not None
        and _needs_repair(
            {
                "result": outputs[i]["stage2b_result"],
                "report": outputs[i]["stage2b_report"],
            },
            outputs[i].get("certificate"),
        )
        and outputs[i].get("api_error") is None
        and outputs[i].get("certificate") is not None
        and outputs[i]["certificate"].get("status") != "UNVERIFIABLE"
    ]

    if max_repair_retries > 0 and repair_indices:
        print(f"Running Stage 2B Pass 2 repair on {len(repair_indices)} items...", flush=True)

        def run_repair(idx: int) -> tuple[int, dict[str, Any], dict[str, int]]:
            assembled = outputs[idx]
            assert assembled is not None
            rec = records[idx]
            r = assembled["stage2b_result"]
            rep = assembled["stage2b_report"]
            msgs = build_repair_messages(
                system_prompt,
                rec,
                assembled["raw_cert_output"],
                r["reject_reason"],
                rep.get("details") or [],
            )
            raw, usage = call_chat_completion(
                client, model, msgs, max_tokens=max_tokens, temperature=temperature, caps=caps
            )
            cert_obj, parse_err = parse_json_object(raw)
            if cert_obj is None:
                v_out = verify_certificate_record(rec, None, manifest_sha, cfg_sha)
                _append_report_detail(v_out, f"REPAIR_JSON_PARSE_ERROR: {parse_err}")
            else:
                v_out = verify_certificate_record(rec, cert_obj, manifest_sha, cfg_sha)
            return idx, _assemble_record(rec, cert_obj, raw, v_out, attempts=2), usage

        done_r = 0
        t_r = time.time()
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = {pool.submit(run_repair, i): i for i in repair_indices}
            for fut in as_completed(futs):
                i = futs[fut]
                done_r += 1
                try:
                    _, assembled, usage = fut.result()
                    outputs[i] = assembled
                    usage_totals.update(usage)
                    r = assembled["stage2b_result"]
                    print(
                        f"[Stage2B Pass 2] {done_r}/{len(repair_indices)} {assembled['item_id']} -> "
                        f"{r['verify_status']} ({r['reject_reason'] or 'OK'}) ({time.time()-t_r:.1f}s)",
                        flush=True,
                    )
                except Exception as exc:
                    print(
                        f"[Stage2B Pass 2] {done_r}/{len(repair_indices)} {records[i]['item_id']} -> API_ERROR: {exc}",
                        flush=True,
                    )

    final_rows = [r for r in outputs if r is not None]
    return final_rows, dict(usage_totals)


def _assemble_record(
    stage2a_rec: dict[str, Any],
    cert_obj: dict[str, Any] | None,
    raw_output: str,
    verify_out: dict[str, Any],
    attempts: int,
    api_error: str | None = None,
) -> dict[str, Any]:
    out = {
        "item_id": stage2a_rec["item_id"],
        "status_code": stage2a_rec["status_code"],
        "macro_subfield": stage2a_rec["macro_subfield"],
        "provisional_mechanism": stage2a_rec["provisional_mechanism"],
        "slot_index": stage2a_rec.get("slot_index"),
        "reserve_rank": stage2a_rec.get("reserve_rank"),
        "source_dataset": stage2a_rec.get("source_dataset"),
        "subfield": stage2a_rec.get("subfield"),
        "answer_type": stage2a_rec.get("answer_type"),
        "question": stage2a_rec["question"],
        "question_sha256": stage2a_rec["question_sha256"],
        "final_answer": stage2a_rec.get("final_answer"),
        "solution": stage2a_rec.get("solution"),
        "parsed_output": stage2a_rec.get("parsed_output"),
        "attempts": attempts,
        "raw_cert_output": raw_output,
        "certificate": cert_obj,
        "stage2b_result": verify_out["result"],
        "stage2b_report": verify_out["report"],
    }
    if api_error:
        out["api_error"] = api_error
    return out


def build_summary(rows: list[dict[str, Any]], model: str, usage: dict[str, int], elapsed: float) -> dict[str, Any]:
    status_counts = Counter()
    reason_counts = Counter()
    by_bucket_split = Counter()
    for r in rows:
        res = r["stage2b_result"]
        st = res["verify_status"]
        status_counts[st] += 1
        if res["reject_reason"]:
            reason_counts[res["reject_reason"]] += 1
        if st == "VERIFY_PASS":
            by_bucket_split[f"{r['macro_subfield']}__{r['provisional_mechanism']}__{r['status_code']}"] += 1
    return {
        "model": model,
        "rule_version": STAGE2B_RULE_VERSION,
        "total_items": len(rows),
        "verify_status_counts": dict(sorted(status_counts.items())),
        "reject_reason_counts": dict(sorted(reason_counts.items())),
        "passed_by_bucket_and_split": dict(sorted(by_bucket_split.items())),
        "token_usage": usage,
        "elapsed_total_sec": round(elapsed, 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate & verify Stage 2B certificates.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--prompt", type=Path, default=DEFAULT_PROMPT)
    parser.add_argument("--manifest", type=Path, default=BASELINE_MANIFEST)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--summary", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--model", type=str, default="auto/deepseek-v4.1-flash")
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default="low", choices=["auto", "off", "none", "low", "medium", "high", "max"])
    parser.add_argument("--max-tokens", type=int, default=32768)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--max-retries", type=int, default=1)
    parser.add_argument(
        "--status-filter",
        type=str,
        default="SELECTED_DEV",
        choices=["ALL", "SELECTED_DEV", "SELECTED_TEST", "ELIGIBLE_RESERVE"],
        help="Default SELECTED_DEV per SPEC 12.1.",
    )
    parser.add_argument("--resume", action="store_true", help="Reuse VERIFY_PASS records already in --output.")
    args = parser.parse_args()

    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Set --api-key or $env:DEEPSEEK_API_KEY / $env:OPENAI_API_KEY.")

    from openai import OpenAI  # noqa: PLC0415

    client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=args.timeout)
    system_prompt = args.prompt.read_text(encoding="utf-8").strip()
    manifest_sha = sha256_bytes(args.manifest.read_bytes())

    effort, extra_disable = resolve_reasoning_effort(args.model, args.reasoning_effort)
    caps = CapabilityState(
        use_reasoning_effort=effort,
        use_json_object=True,
        disable_thinking_extra_body=extra_disable,
    )
    cfg_sha = compute_config_sha(args.prompt, args.model, args.max_tokens, effort)

    all_records = read_jsonl(args.input)
    if args.status_filter != "ALL":
        records = [r for r in all_records if r["status_code"] == args.status_filter]
    else:
        records = all_records

    existing: dict[str, dict[str, Any]] = {}
    if args.resume and args.output.exists():
        for rec in read_jsonl(args.output):
            if rec.get("stage2b_result", {}).get("verify_status") == "VERIFY_PASS":
                existing[rec["item_id"]] = rec
        print(f"Resuming Stage 2B: reusing {len(existing)} VERIFY_PASS records from {args.output}")

    t0 = time.time()
    rows, usage = verify_stage2b_batch(
        client=client,
        records=records,
        system_prompt=system_prompt,
        model=args.model,
        manifest_sha=manifest_sha,
        cfg_sha=cfg_sha,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        workers=args.workers,
        max_repair_retries=args.max_retries,
        caps=caps,
        existing_records=existing,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        for row in rows:
            fh.write((json.dumps(row, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8"))

    summary = build_summary(rows, args.model, usage, time.time() - t0)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_bytes((json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n").encode("utf-8"))
    print("\n=== Stage 2B Certificate Verification Summary ===")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
