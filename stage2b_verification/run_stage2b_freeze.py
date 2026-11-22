#!/usr/bin/env python3
"""Deterministic Stage 2B reserve-promotion state machine & benchmark freezer (SPEC v2.4 Section 12.4-12.5).

Consumes Stage 2B verification results and Stage 1 v2.4 baseline files in fixed order:
  split in ["TEST", "DEV"] -> sorted 8 buckets -> ascending slot_index -> reserve_rank = 1, 2, ...

Zero LLM/API calls. If any bucket exhausts its reserve pool, exits with code 5 and does NOT
publish frozen_manifest.json.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
BASELINE_DIR = SCRIPT_DIR.parent / "benchmark_spec_v2_4" / "reference_baseline"
DEFAULT_STAGE2B = SCRIPT_DIR.parent / "exp1_codex_gpt61_v1" / "stage2b_verified_all469.jsonl"
DEFAULT_OUT_DIR = SCRIPT_DIR.parent / "exp1_codex_gpt61_v1" / "frozen_release_verified_120"

QUOTA_PROFILES: dict[str, dict[str, dict[str, int]]] = {
    "v2_4": {
        "TEST": {
            "G1_Func_Calc_Trig__TypeA_UniqueValue": 16,
            "G1_Func_Calc_Trig__TypeB_Extremum": 9,
            "G2_Alg_Ineq__TypeA_UniqueValue": 16,
            "G2_Alg_Ineq__TypeB_Extremum": 9,
            "G3_Seq_Comb_NT__TypeA_UniqueValue": 16,
            "G3_Seq_Comb_NT__TypeB_Extremum": 9,
            "G4_Geom_Analytic__TypeA_UniqueValue": 16,
            "G4_Geom_Analytic__TypeB_Extremum": 9,
        },
        "DEV": {
            "G1_Func_Calc_Trig__TypeA_UniqueValue": 3,
            "G1_Func_Calc_Trig__TypeB_Extremum": 2,
            "G2_Alg_Ineq__TypeA_UniqueValue": 3,
            "G2_Alg_Ineq__TypeB_Extremum": 2,
            "G3_Seq_Comb_NT__TypeA_UniqueValue": 3,
            "G3_Seq_Comb_NT__TypeB_Extremum": 2,
            "G4_Geom_Analytic__TypeA_UniqueValue": 3,
            "G4_Geom_Analytic__TypeB_Extremum": 2,
        },
    },
    # Intra-G3 rebalance (-2 G3_TypeB TEST -> +2 G3_TypeA TEST) preserving 100 TEST / 20 DEV
    # when pure-SymPy formal verification rejects unformalizable G3_TypeB combinatorics.
    "g3_rebalanced_120": {
        "TEST": {
            "G1_Func_Calc_Trig__TypeA_UniqueValue": 16,
            "G1_Func_Calc_Trig__TypeB_Extremum": 9,
            "G2_Alg_Ineq__TypeA_UniqueValue": 16,
            "G2_Alg_Ineq__TypeB_Extremum": 9,
            "G3_Seq_Comb_NT__TypeA_UniqueValue": 18,
            "G3_Seq_Comb_NT__TypeB_Extremum": 7,
            "G4_Geom_Analytic__TypeA_UniqueValue": 16,
            "G4_Geom_Analytic__TypeB_Extremum": 9,
        },
        "DEV": {
            "G1_Func_Calc_Trig__TypeA_UniqueValue": 3,
            "G1_Func_Calc_Trig__TypeB_Extremum": 2,
            "G2_Alg_Ineq__TypeA_UniqueValue": 3,
            "G2_Alg_Ineq__TypeB_Extremum": 2,
            "G3_Seq_Comb_NT__TypeA_UniqueValue": 3,
            "G3_Seq_Comb_NT__TypeB_Extremum": 2,
            "G4_Geom_Analytic__TypeA_UniqueValue": 3,
            "G4_Geom_Analytic__TypeB_Extremum": 2,
        },
    },
    # Yield-feasible rebalance preserving 100 TEST / 20 DEV and G1-G4 balance (35/27/29/29)
    # given exhaustive Stage 1 formalizable capacities in G2_TypeA (16/45) and G3_TypeB (5/28).
    "verified_120_v1": {
        "TEST": {
            "G1_Func_Calc_Trig__TypeA_UniqueValue": 19,
            "G1_Func_Calc_Trig__TypeB_Extremum": 10,
            "G2_Alg_Ineq__TypeA_UniqueValue": 13,
            "G2_Alg_Ineq__TypeB_Extremum": 10,
            "G3_Seq_Comb_NT__TypeA_UniqueValue": 21,
            "G3_Seq_Comb_NT__TypeB_Extremum": 3,
            "G4_Geom_Analytic__TypeA_UniqueValue": 15,
            "G4_Geom_Analytic__TypeB_Extremum": 9,
        },
        "DEV": {
            "G1_Func_Calc_Trig__TypeA_UniqueValue": 4,
            "G1_Func_Calc_Trig__TypeB_Extremum": 2,
            "G2_Alg_Ineq__TypeA_UniqueValue": 2,
            "G2_Alg_Ineq__TypeB_Extremum": 2,
            "G3_Seq_Comb_NT__TypeA_UniqueValue": 4,
            "G3_Seq_Comb_NT__TypeB_Extremum": 1,
            "G4_Geom_Analytic__TypeA_UniqueValue": 3,
            "G4_Geom_Analytic__TypeB_Extremum": 2,
        },
    },
}

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from stage2b_verifier_core import (  # noqa: E402
    STAGE2B_RULE_VERSION,
    canonical_json_bytes,
    sha256_bytes,
    sha256_json,
    verify_certificate_record,
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as fh:
        for row in rows:
            fh.write(canonical_json_bytes(row) + b"\n")


def run_freeze_state_machine(
    baseline_dir: Path,
    stage2b_records_path: Path,
    out_dir: Path,
    dry_run_partial: bool = False,
    quota_profile: str = "v2_4",
) -> int:
    quotas = QUOTA_PROFILES[quota_profile]
    manifest_path = baseline_dir / "selection_manifest.json"
    manifest_sha_path = baseline_dir / "selection_manifest.json.sha256"
    manifest_bytes = manifest_path.read_bytes()
    manifest_sha = sha256_bytes(manifest_bytes)
    if manifest_sha_path.exists():
        expected_sha = manifest_sha_path.read_text(encoding="utf-8").strip().split()[0]
        if expected_sha != manifest_sha:
            raise ValueError(f"Stage 1 manifest SHA mismatch: {manifest_sha} != {expected_sha}")
    stage1_manifest = json.loads(manifest_bytes.decode("utf-8"))

    test_items = read_jsonl(baseline_dir / "provisional_test_100.jsonl")
    dev_items = read_jsonl(baseline_dir / "provisional_dev_20.jsonl")
    reserve_items = read_jsonl(baseline_dir / "eligible_reserve_pool.jsonl")

    stage2b_by_id: dict[str, dict[str, Any]] = {
        r["item_id"]: r for r in read_jsonl(stage2b_records_path)
    }

    # Index slots by (split, bucket) -> sorted by slot_index.
    slots: dict[tuple[str, tuple[str, str]], list[dict[str, Any]]] = defaultdict(list)
    for item in test_items:
        bk = (item["macro_subfield"], item["provisional_mechanism"])
        slots[("TEST", bk)].append(item)
    for item in dev_items:
        bk = (item["macro_subfield"], item["provisional_mechanism"])
        slots[("DEV", bk)].append(item)
    for key in slots:
        slots[key].sort(key=lambda x: int(x["slot_index"]))

    reserves_by_bucket: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for item in reserve_items:
        bk = (item["macro_subfield"], item["provisional_mechanism"])
        reserves_by_bucket[bk].append(item)
    for bk in reserves_by_bucket:
        reserves_by_bucket[bk].sort(key=lambda x: (int(x["reserve_rank"]), x["item_id"]))

    all_buckets = sorted({(x["macro_subfield"], x["provisional_mechanism"]) for x in test_items + dev_items})
    reserve_cursor: dict[tuple[str, str], int] = {bk: 0 for bk in all_buckets}

    events: list[dict[str, Any]] = []
    ledger_rows: list[dict[str, Any]] = []
    frozen_test: list[dict[str, Any]] = []
    frozen_dev: list[dict[str, Any]] = []
    touched_certs: dict[str, dict[str, Any]] = {}
    touched_reports: dict[str, dict[str, Any]] = {}
    event_seq = 0
    exhausted_buckets: list[str] = []

    def evaluate_item(stage1_item: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any] | None]:
        iid = stage1_item["item_id"]
        rec2b = stage2b_by_id.get(iid)
        if rec2b is None:
            cand_stub = {
                "item_id": iid,
                "question": stage1_item["question"],
                "question_sha256": stage1_item["question_sha256"],
                "macro_subfield": stage1_item["macro_subfield"],
                "provisional_mechanism": stage1_item["provisional_mechanism"],
                "parsed_output": None,
            }
            v_out = verify_certificate_record(cand_stub, None, manifest_sha, "unevaluated")
            return v_out["result"], v_out["report"], None
        res = rec2b["stage2b_result"]
        rep = rec2b["stage2b_report"]
        # Re-verify acyclic report hash integrity (SPEC 12.3).
        if sha256_json(rep) != res["verification_report_sha256"]:
            raise ValueError(f"Corrupted Stage 2B report hash for {iid}")
        return res, rep, rec2b

    for split in ("TEST", "DEV"):
        for bk in all_buckets:
            bk_name = f"{bk[0]}__{bk[1]}"
            target_quota = quotas[split][bk_name]
            bucket_slots = slots[(split, bk)]
            accepted_count = 0
            vacant_slots: list[int] = []
            for slot_item in bucket_slots:
                slot_idx = int(slot_item["slot_index"])
                res, rep, rec2b = evaluate_item(slot_item)
                touched_reports[slot_item["item_id"]] = rep
                if rec2b and rec2b.get("certificate") is not None:
                    touched_certs[slot_item["item_id"]] = {
                        "item_id": slot_item["item_id"],
                        "parsed_output": rec2b.get("parsed_output"),
                        "certificate": rec2b.get("certificate"),
                    }

                if res["verify_status"] == "VERIFY_PASS" and accepted_count < target_quota:
                    accepted_count += 1
                    event_seq += 1
                    ev = {
                        "seq": event_seq,
                        "split": split,
                        "bucket": bk_name,
                        "slot_index": slot_idx,
                        "item_id": slot_item["item_id"],
                        "reserve_rank": None,
                        "event_type": f"INITIAL_ACCEPTED_{split}",
                        "verify_status": "VERIFY_PASS",
                        "reject_reason": "",
                        "binding_digest": res["binding_digest"],
                        "verification_report_sha256": res["verification_report_sha256"],
                    }
                    events.append(ev)
                    ledger_rows.append(ev)
                    frozen_entry = _build_frozen_entry(slot_item, rec2b, res, split, slot_idx, promoted_from_rank=None)
                    (frozen_test if split == "TEST" else frozen_dev).append(frozen_entry)
                else:
                    event_seq += 1
                    ev_rej = {
                        "seq": event_seq,
                        "split": split,
                        "bucket": bk_name,
                        "slot_index": slot_idx,
                        "item_id": slot_item["item_id"],
                        "reserve_rank": None,
                        "event_type": "INITIAL_REJECTED_IN_STAGE2" if res["verify_status"] != "VERIFY_PASS" else "INITIAL_CARVED_OUT_BY_QUOTA",
                        "verify_status": res["verify_status"],
                        "reject_reason": res["reject_reason"],
                        "binding_digest": res["binding_digest"],
                        "verification_report_sha256": res["verification_report_sha256"],
                    }
                    events.append(ev_rej)
                    ledger_rows.append(ev_rej)
                    vacant_slots.append(slot_idx)

            for extra_idx in range(len(bucket_slots) + 1, target_quota + 1):
                vacant_slots.append(extra_idx)

            need_from_reserve = max(0, target_quota - accepted_count)
            for slot_idx in vacant_slots[:need_from_reserve]:
                filled = False
                pool = reserves_by_bucket[bk]
                while reserve_cursor[bk] < len(pool):
                    cand_res = pool[reserve_cursor[bk]]
                    cand_iid = cand_res["item_id"]
                    cand_rank = int(cand_res["reserve_rank"])
                    if dry_run_partial and cand_iid not in stage2b_by_id:
                        break
                    reserve_cursor[bk] += 1
                    r_res, r_rep, r_rec2b = evaluate_item(cand_res)
                    touched_reports[cand_iid] = r_rep
                    if r_rec2b and r_rec2b.get("certificate") is not None:
                        touched_certs[cand_iid] = {
                            "item_id": cand_iid,
                            "parsed_output": r_rec2b.get("parsed_output"),
                            "certificate": r_rec2b.get("certificate"),
                        }

                    event_seq += 1
                    if r_res["verify_status"] == "VERIFY_PASS":
                        ev_prom = {
                            "seq": event_seq,
                            "split": split,
                            "bucket": bk_name,
                            "slot_index": slot_idx,
                            "item_id": cand_iid,
                            "reserve_rank": cand_rank,
                            "event_type": f"PROMOTED_TO_{split}",
                            "verify_status": "VERIFY_PASS",
                            "reject_reason": "",
                            "binding_digest": r_res["binding_digest"],
                            "verification_report_sha256": r_res["verification_report_sha256"],
                        }
                        events.append(ev_prom)
                        ledger_rows.append(ev_prom)
                        frozen_entry = _build_frozen_entry(
                            cand_res, r_rec2b, r_res, split, slot_idx, promoted_from_rank=cand_rank
                        )
                        (frozen_test if split == "TEST" else frozen_dev).append(frozen_entry)
                        filled = True
                        break
                    else:
                        ev_rrej = {
                            "seq": event_seq,
                            "split": split,
                            "bucket": bk_name,
                            "slot_index": slot_idx,
                            "item_id": cand_iid,
                            "reserve_rank": cand_rank,
                            "event_type": "RESERVE_REJECTED_IN_STAGE2",
                            "verify_status": "VERIFY_FAIL",
                            "reject_reason": r_res["reject_reason"],
                            "binding_digest": r_res["binding_digest"],
                            "verification_report_sha256": r_res["verification_report_sha256"],
                        }
                        events.append(ev_rrej)
                        ledger_rows.append(ev_rrej)

                if not filled:
                    exhausted_buckets.append(f"{split}:{bk_name}:slot={slot_idx}")

    # Always persist diagnostics/events/reports.
    out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_dir / "stage2_events.jsonl", events)
    write_jsonl(out_dir / "frozen_reports.jsonl", list(touched_reports.values()))
    write_jsonl(out_dir / "frozen_certificates.jsonl", list(touched_certs.values()))
    _write_ledger_csv(out_dir / "stage2_verification_ledger.csv", ledger_rows)

    if exhausted_buckets:
        print(f"ERROR: Reserve pool exhausted for slots: {exhausted_buckets}", file=sys.stderr)
        print("Frozen manifest NOT published (exit 5).", file=sys.stderr)
        return 5

    if len(frozen_test) != 100 or len(frozen_dev) != 20:
        print(f"ERROR: Unexpected frozen sizes TEST={len(frozen_test)}, DEV={len(frozen_dev)}", file=sys.stderr)
        return 5

    # Uniqueness invariants (item_id & cluster_id across 120 frozen items).
    all_120 = frozen_test + frozen_dev
    if len({x["item_id"] for x in all_120}) != 120:
        raise ValueError("Duplicate item_id in frozen 120!")
    cids = [x.get("cluster_id") for x in all_120 if x.get("cluster_id")]
    if len(cids) == 120 and len(set(cids)) != 120:
        raise ValueError("Duplicate cluster_id in frozen 120!")

    write_jsonl(out_dir / "frozen_test_100.jsonl", frozen_test)
    write_jsonl(out_dir / "frozen_dev_20.jsonl", frozen_dev)

    manifest = {
        "quota_profile": quota_profile,
        "test_quotas": quotas["TEST"],
        "dev_quotas": quotas["DEV"],
        "stage1_manifest_sha256": manifest_sha,
        "stage1_rule_version": stage1_manifest.get("rule_version"),
        "stage2b_rule_version": STAGE2B_RULE_VERSION,
        "frozen_test_count": len(frozen_test),
        "frozen_dev_count": len(frozen_dev),
        "bucket_reserve_cursors": {f"{k[0]}__{k[1]}": v for k, v in sorted(reserve_cursor.items())},
        "event_counts": dict(sorted(Counter(e["event_type"] for e in events).items())),
        "frozen_test_100_sha256": sha256_bytes((out_dir / "frozen_test_100.jsonl").read_bytes()),
        "frozen_dev_20_sha256": sha256_bytes((out_dir / "frozen_dev_20.jsonl").read_bytes()),
        "stage2_events_sha256": sha256_bytes((out_dir / "stage2_events.jsonl").read_bytes()),
    }
    manifest_out_bytes = json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8") + b"\n"
    (out_dir / "frozen_manifest.json").write_bytes(manifest_out_bytes)
    (out_dir / "frozen_manifest.json.sha256").write_text(
        f"{sha256_bytes(manifest_out_bytes)}  frozen_manifest.json\n", encoding="utf-8"
    )
    print(f"Published frozen release (Test=100, Dev=20) to {out_dir}")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    return 0


def _build_frozen_entry(
    stage1_item: dict[str, Any],
    rec2b: dict[str, Any] | None,
    res2b: dict[str, Any],
    split: str,
    slot_index: int,
    promoted_from_rank: int | None,
) -> dict[str, Any]:
    assert rec2b is not None
    po = rec2b["parsed_output"]
    shards = [{"shard_id": "S1", "shard": po["initial_shard"]}] + [
        {"shard_id": s["shard_id"], "shard": s.get("shard") or s.get("shard_text") or s.get("text")}
        for s in po["shards"]
    ]
    return {
        "item_id": stage1_item["item_id"],
        "split": split,
        "bucket": f"{stage1_item['macro_subfield']}__{stage1_item['provisional_mechanism']}",
        "macro_subfield": stage1_item["macro_subfield"],
        "verified_mechanism": res2b["verified_mechanism"],
        "slot_index": slot_index,
        "promoted_from_reserve_rank": promoted_from_rank,
        "cluster_id": stage1_item.get("cluster_id"),
        "question": stage1_item["question"],
        "question_sha256": stage1_item["question_sha256"],
        "final_answer": stage1_item.get("final_answer"),
        "canonical_k": res2b["canonical_k"],
        "shards": shards,
        "concat_question": "\n".join(s["shard"] for s in shards),
        "binding_digest": res2b["binding_digest"],
        "certificate_sha256": res2b["certificate_sha256"],
        "verification_report_sha256": res2b["verification_report_sha256"],
    }


def _write_ledger_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = [
        "seq",
        "split",
        "bucket",
        "slot_index",
        "item_id",
        "reserve_rank",
        "event_type",
        "verify_status",
        "reject_reason",
        "binding_digest",
        "verification_report_sha256",
    ]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for r in rows:
            row = dict(r)
            if row["reserve_rank"] is None:
                row["reserve_rank"] = ""
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run deterministic Stage 2B freeze state machine.")
    parser.add_argument("--baseline-dir", type=Path, default=BASELINE_DIR)
    parser.add_argument("--stage2b", type=Path, default=DEFAULT_STAGE2B)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--quota-profile",
        type=str,
        default="v2_4",
        choices=sorted(QUOTA_PROFILES),
    )
    args = parser.parse_args()
    raise SystemExit(
        run_freeze_state_machine(args.baseline_dir, args.stage2b, args.out_dir, quota_profile=args.quota_profile)
    )


if __name__ == "__main__":
    main()
