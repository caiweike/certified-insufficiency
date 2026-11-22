#!/usr/bin/env python3
"""Prepare Stage 2A sharding input candidates from the verified v2.4 baseline."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


DEFAULT_BASELINE_DIR = (
    Path(__file__).resolve().parent.parent / "benchmark_spec_v2_4" / "reference_baseline"
)
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parent / "stage2_input_candidates.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    records: list[dict] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            text = line.strip()
            if text:
                records.append(json.loads(text))
    return records


def build_stage2_inputs(baseline_dir: Path, max_reserve_per_bucket: int = 15) -> list[dict]:
    dev_items = read_jsonl(baseline_dir / "provisional_dev_20.jsonl")
    test_items = read_jsonl(baseline_dir / "provisional_test_100.jsonl")
    reserve_items = read_jsonl(baseline_dir / "eligible_reserve_pool.jsonl")

    for item in dev_items:
        assert item["status_code"] == "SELECTED_DEV"
    for item in test_items:
        assert item["status_code"] == "SELECTED_TEST"

    # Sort deterministically by bucket, then reserve_rank, then item_id.
    eligible_reserves = [
        item
        for item in reserve_items
        if item["status_code"] == "ELIGIBLE_RESERVE"
        and item["reserve_rank"] is not None
        and int(item["reserve_rank"]) <= max_reserve_per_bucket
    ]
    eligible_reserves.sort(
        key=lambda x: (
            x["macro_subfield"],
            x["provisional_mechanism"],
            int(x["reserve_rank"]),
            x["item_id"],
        )
    )

    combined = dev_items + test_items + eligible_reserves
    seen_ids: set[str] = set()
    for item in combined:
        if item["item_id"] in seen_ids:
            raise ValueError(f"Duplicate item_id detected: {item['item_id']}")
        seen_ids.add(item["item_id"])
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build stage2_input_candidates.jsonl from v2.4 reference baseline."
    )
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--max-reserve-per-bucket", type=int, default=15)
    args = parser.parse_args()

    records = build_stage2_inputs(args.baseline_dir, args.max_reserve_per_bucket)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        for rec in records:
            line = json.dumps(rec, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n"
            fh.write(line.encode("utf-8"))

    status_counts = Counter(r["status_code"] for r in records)
    reserve_bucket_counts = Counter(
        (r["macro_subfield"], r["provisional_mechanism"])
        for r in records
        if r["status_code"] == "ELIGIBLE_RESERVE"
    )
    print(f"Wrote {len(records)} items to {args.output}")
    print("Status counts:", dict(status_counts))
    print("Reserve bucket counts:", {f"{k[0]}__{k[1]}": v for k, v in sorted(reserve_bucket_counts.items())})


if __name__ == "__main__":
    main()
