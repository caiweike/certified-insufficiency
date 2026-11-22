#!/usr/bin/env python3
"""Unified, reproducible Experiment 1 runner (Stage 1 -> Stage 2A -> Stage 2B -> Freeze & Offline Verify).

Two modes:
  - `--mode verify-offline` (default): Zero-API verification of Stage 1 baseline + frozen Stage 2B release.
  - `--mode build-from-scratch`: Clean single-pass Stage 2A + Stage 2B execution over a uniform
    reserve prefix (`--max-reserve-per-bucket 30`), followed by deterministic state-machine freeze
    and offline SymPy + SHA-256 certificate verification.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent
SPEC_DIR = ROOT_DIR / "benchmark_spec_v2_4"
STAGE1_BASELINE = SPEC_DIR / "reference_baseline"
STAGE2A_DIR = ROOT_DIR / "colab_stage2_sharding"
STAGE2B_DIR = ROOT_DIR / "stage2b_verification"


def run_step(cmd: list[str], cwd: Path | None = None) -> None:
    print(f"\n>>> Running: {' '.join(cmd)}", flush=True)
    proc = subprocess.run(cmd, cwd=str(cwd or ROOT_DIR))
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def verify_stage1_baseline(comp_parquet: Path | None = None, cee_parquet: Path | None = None) -> None:
    m_bytes = (STAGE1_BASELINE / "selection_manifest.json").read_bytes()
    actual_sha = hashlib.sha256(m_bytes).hexdigest()
    expected_sha = (STAGE1_BASELINE / "selection_manifest.json.sha256").read_text(encoding="utf-8").strip().split()[0]
    assert actual_sha == expected_sha, f"Stage 1 manifest SHA mismatch: {actual_sha} != {expected_sha}"
    report = json.loads((STAGE1_BASELINE / "baseline_report.json").read_text(encoding="utf-8"))
    for fname, f_sha in report["file_sha256"].items():
        got = hashlib.sha256((STAGE1_BASELINE / fname).read_bytes()).hexdigest()
        assert got == f_sha, f"Stage 1 baseline file {fname} SHA mismatch: {got} != {f_sha}"
    print(f"Stage 1 baseline SHA-256 verified OK (manifest={actual_sha}).", flush=True)
    if comp_parquet and cee_parquet:
        run_step(
            [
                sys.executable,
                str(SPEC_DIR / "verify_reference.py"),
                "--comp-parquet",
                str(comp_parquet),
                "--cee-parquet",
                str(cee_parquet),
                "--audit-json",
                str(SPEC_DIR / "audit_overrides.v2.3.json"),
                "--report",
                str(SPEC_DIR / "baseline_report.json"),
            ]
        )


def verify_offline(release_dir: Path, comp_parquet: Path | None = None, cee_parquet: Path | None = None) -> None:
    verify_stage1_baseline(comp_parquet, cee_parquet)
    run_step(
        [
            sys.executable,
            str(STAGE2B_DIR / "verify_certificates.py"),
            "--release-dir",
            str(release_dir),
            "--baseline-dir",
            str(STAGE1_BASELINE),
        ]
    )
    print(f"\nExperiment 1 offline reproducibility check PASSED for {release_dir}")


def build_from_scratch(args: argparse.Namespace) -> None:
    api_key = args.api_key or os.environ.get("DEEPSEEK_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("Missing API key. Pass --api-key or set $env:DEEPSEEK_API_KEY / $env:OPENAI_API_KEY.")

    work_dir = args.out_dir.resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    release_dir = work_dir / "frozen_release"

    inputs_jsonl = work_dir / "stage2_inputs.jsonl"
    stage2a_jsonl = work_dir / "stage2a_sharded.jsonl"
    stage2a_sum = work_dir / "stage2a_summary.json"
    stage2b_jsonl = work_dir / "stage2b_verified.jsonl"
    stage2b_sum = work_dir / "stage2b_summary.json"

    # 1. Verify Stage 1 deterministic reference baseline.
    verify_stage1_baseline(args.comp_parquet, args.cee_parquet)

    # 2. Extract uniform Stage 2 candidate slice (TEST 100 + DEV 20 + top K reserve per bucket).
    run_step(
        [
            sys.executable,
            str(STAGE2A_DIR / "prepare_stage2_inputs.py"),
            "--baseline-dir",
            str(STAGE1_BASELINE),
            "--output",
            str(inputs_jsonl),
            "--max-reserve-per-bucket",
            str(args.max_reserve_per_bucket),
        ]
    )

    # 3. Single-pass clean Stage 2A sharding & deterministic alignment audit (NO --resume patching).
    run_step(
        [
            sys.executable,
            str(STAGE2A_DIR / "run_api_sharding.py"),
            "--input",
            str(inputs_jsonl),
            "--prompt",
            str(STAGE2A_DIR / "prompt_math_lic_v2_6.txt"),
            "--output",
            str(stage2a_jsonl),
            "--summary",
            str(stage2a_sum),
            "--base-url",
            args.base_url,
            "--api-key",
            api_key,
            "--model",
            args.model,
            "--reasoning-effort",
            args.stage2a_reasoning_effort,
            "--repair-reasoning-effort",
            args.stage2a_repair_reasoning_effort,
            "--max-tokens",
            str(args.stage2a_max_tokens),
            "--timeout",
            str(args.timeout),
            "--workers",
            str(args.workers),
            "--status-filter",
            "ALL",
        ]
    )

    # 4. Single-pass clean Stage 2B symbolic certificate synthesis & SymPy verification.
    run_step(
        [
            sys.executable,
            str(STAGE2B_DIR / "generate_stage2b_certificates.py"),
            "--input",
            str(stage2a_jsonl),
            "--prompt",
            str(STAGE2B_DIR / "prompt_stage2b_cert_v1.txt"),
            "--manifest",
            str(STAGE1_BASELINE / "selection_manifest.json"),
            "--output",
            str(stage2b_jsonl),
            "--summary",
            str(stage2b_sum),
            "--base-url",
            args.base_url,
            "--api-key",
            api_key,
            "--model",
            args.model,
            "--reasoning-effort",
            "low",
            "--max-tokens",
            str(args.stage2b_max_tokens),
            "--timeout",
            str(args.timeout),
            "--workers",
            str(args.workers),
            "--max-retries",
            str(args.stage2b_max_retries),
            "--status-filter",
            "ALL",
        ]
    )

    # 5. Deterministic reserve-promotion state machine & freeze.
    run_step(
        [
            sys.executable,
            str(STAGE2B_DIR / "run_stage2b_freeze.py"),
            "--baseline-dir",
            str(STAGE1_BASELINE),
            "--stage2b",
            str(stage2b_jsonl),
            "--out-dir",
            str(release_dir),
            "--quota-profile",
            args.quota_profile,
        ]
    )

    # 6. Independent offline SymPy + SHA-256 certificate re-verification.
    run_step(
        [
            sys.executable,
            str(STAGE2B_DIR / "verify_certificates.py"),
            "--release-dir",
            str(release_dir),
            "--baseline-dir",
            str(STAGE1_BASELINE),
        ]
    )

    build_meta = {
        "model": args.model,
        "max_reserve_per_bucket": args.max_reserve_per_bucket,
        "quota_profile": args.quota_profile,
        "stage2a_reasoning_effort": args.stage2a_reasoning_effort,
        "stage2a_repair_reasoning_effort": args.stage2a_repair_reasoning_effort,
        "stage2a_max_tokens": args.stage2a_max_tokens,
        "stage2b_reasoning_effort": "low",
        "stage2b_max_tokens": args.stage2b_max_tokens,
        "stage2b_max_retries": args.stage2b_max_retries,
        "release_dir": str(release_dir),
    }
    (work_dir / "exp1_build_config.json").write_text(
        json.dumps(build_meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"\nExperiment 1 clean single-pass build & offline verification PASSED -> {release_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproducible Experiment 1 pipeline runner.")
    parser.add_argument(
        "--mode",
        type=str,
        default="verify-offline",
        choices=["verify-offline", "build-from-scratch"],
    )
    parser.add_argument("--out-dir", type=Path, default=ROOT_DIR / "exp1_clean_release_v1")
    parser.add_argument(
        "--release-dir",
        type=Path,
        default=ROOT_DIR / "exp1_codex_gpt61_v1" / "frozen_release_verified_120",
    )
    parser.add_argument("--comp-parquet", type=Path, default=None)
    parser.add_argument("--cee-parquet", type=Path, default=None)
    parser.add_argument(
        "--base-url",
        type=str,
        default=os.environ.get("DEEPSEEK_BASE_URL") or os.environ.get("OPENAI_BASE_URL") or "https://api.deepseek.com",
    )
    parser.add_argument("--api-key", type=str, default=None)
    parser.add_argument("--model", type=str, default="deepseek-chat")
    parser.add_argument("--max-reserve-per-bucket", type=int, default=30)
    parser.add_argument(
        "--quota-profile",
        type=str,
        default="verified_120_v1",
        choices=["v2_4", "g3_rebalanced_120", "verified_120_v1"],
    )
    parser.add_argument("--stage2a-reasoning-effort", type=str, default="off", choices=["off", "low", "medium", "high"])
    parser.add_argument("--stage2a-repair-reasoning-effort", type=str, default="low", choices=["off", "low", "medium", "high"])
    parser.add_argument("--stage2a-max-tokens", type=int, default=16384)
    parser.add_argument("--stage2b-max-tokens", type=int, default=32768)
    parser.add_argument("--stage2b-max-retries", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=150.0)
    parser.add_argument("--workers", type=int, default=12)
    args = parser.parse_args()

    if args.mode == "verify-offline":
        verify_offline(args.release_dir, args.comp_parquet, args.cee_parquet)
    else:
        build_from_scratch(args)


if __name__ == "__main__":
    main()
