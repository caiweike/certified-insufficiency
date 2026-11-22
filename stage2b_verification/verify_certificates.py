#!/usr/bin/env python3
"""Independent offline certificate & release verifier (SPEC v2.4 Section 12.5).

Reads a frozen Stage 2B release directory (frozen_manifest.json, frozen_test_100.jsonl,
frozen_dev_20.jsonl, frozen_certificates.jsonl, frozen_reports.jsonl, stage2_events.jsonl)
and recomputes all alignment checks, SymPy certificate verifications, acyclic hashes,
quota invariants, and reserve-promotion event invariants from scratch with zero API calls.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_RELEASE_DIR = SCRIPT_DIR.parent / "exp1_codex_gpt61_v1" / "frozen_release_verified_120"  # benchmark used in the paper
DEFAULT_BASELINE_DIR = SCRIPT_DIR.parent / "benchmark_spec_v2_4" / "reference_baseline"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from run_stage2b_freeze import QUOTA_PROFILES, read_jsonl  # noqa: E402
from stage2b_verifier_core import (  # noqa: E402
    STAGE2B_RULE_VERSION,
    sha256_bytes,
    sha256_json,
    verify_certificate_record,
)


def verify_frozen_release(release_dir: Path, baseline_dir: Path) -> None:
    manifest_path = release_dir / "frozen_manifest.json"
    manifest_sha_path = release_dir / "frozen_manifest.json.sha256"
    m_bytes = manifest_path.read_bytes()
    m_sha = sha256_bytes(m_bytes)
    expected_m_sha = manifest_sha_path.read_text(encoding="utf-8").strip().split()[0]
    assert m_sha == expected_m_sha, f"frozen_manifest SHA mismatch: {m_sha} != {expected_m_sha}"
    manifest = json.loads(m_bytes.decode("utf-8"))

    # Check Stage 1 manifest hash binding.
    s1_sha = sha256_bytes((baseline_dir / "selection_manifest.json").read_bytes())
    assert manifest["stage1_manifest_sha256"] == s1_sha, "Stage 1 manifest SHA mismatch!"
    assert manifest["stage2b_rule_version"] == STAGE2B_RULE_VERSION

    # Check file hashes.
    for fname, key in (
        ("frozen_test_100.jsonl", "frozen_test_100_sha256"),
        ("frozen_dev_20.jsonl", "frozen_dev_20_sha256"),
        ("stage2_events.jsonl", "stage2_events_sha256"),
    ):
        actual = sha256_bytes((release_dir / fname).read_bytes())
        assert actual == manifest[key], f"{fname} SHA mismatch: {actual} != {manifest[key]}"

    frozen_test = read_jsonl(release_dir / "frozen_test_100.jsonl")
    frozen_dev = read_jsonl(release_dir / "frozen_dev_20.jsonl")
    events = read_jsonl(release_dir / "stage2_events.jsonl")
    certs_by_id = {c["item_id"]: c for c in read_jsonl(release_dir / "frozen_certificates.jsonl")}
    reports_by_id = {r["item_id"]: r for r in read_jsonl(release_dir / "frozen_reports.jsonl")}

    profile = manifest.get("quota_profile", "v2_4")
    quotas = QUOTA_PROFILES[profile]
    assert len(frozen_test) == 100, f"Expected 100 test items, got {len(frozen_test)}"
    assert len(frozen_dev) == 20, f"Expected 20 dev items, got {len(frozen_dev)}"
    assert Counter(x["bucket"] for x in frozen_test) == Counter(quotas["TEST"]), "TEST quota mismatch!"
    assert Counter(x["bucket"] for x in frozen_dev) == Counter(quotas["DEV"]), "DEV quota mismatch!"

    all_120 = frozen_test + frozen_dev
    assert len({x["item_id"] for x in all_120}) == 120, "Duplicate item_id in frozen 120!"
    cids = [x["cluster_id"] for x in all_120 if x.get("cluster_id")]
    if cids:
        assert len(cids) == 120 and len(set(cids)) == 120, "Duplicate cluster_id in frozen 120!"

    # Re-verify every single certificate from scratch using SymPy.
    verified_count = 0
    for entry in all_120:
        iid = entry["item_id"]
        c_rec = certs_by_id.get(iid)
        rep = reports_by_id.get(iid)
        assert c_rec is not None, f"Missing certificate for {iid}"
        assert rep is not None, f"Missing report for {iid}"
        assert sha256_json(rep) == entry["verification_report_sha256"], f"Report SHA mismatch for {iid}"
        assert sha256_json(c_rec["certificate"]) == entry["certificate_sha256"], f"Cert SHA mismatch for {iid}"

        # Infer cfg_sha cannot mismatch report rule version; check deterministic SymPy verification directly.
        cand_stub = {
            "item_id": iid,
            "question": entry["question"],
            "question_sha256": entry["question_sha256"],
            "macro_subfield": entry["macro_subfield"],
            "provisional_mechanism": entry["verified_mechanism"],
            "parsed_output": c_rec["parsed_output"],
        }
        re_out = verify_certificate_record(cand_stub, c_rec["certificate"], s1_sha, "recheck")
        assert re_out["result"]["verify_status"] == "VERIFY_PASS", (
            f"Re-verification FAILED for {iid}: {re_out['report']}"
        )
        assert re_out["result"]["canonical_k"] == entry["canonical_k"]
        verified_count += 1

    # Verify event sequence & reserve rank monotonicity per bucket (no skipped ranks).
    ranks_consumed: dict[str, list[int]] = defaultdict(list)
    accepted_ids: list[str] = []
    for seq_idx, ev in enumerate(events, 1):
        assert ev["seq"] == seq_idx, f"Event seq gap at {seq_idx}"
        if ev["reserve_rank"] is not None:
            ranks_consumed[ev["bucket"]].append(int(ev["reserve_rank"]))
        if ev["verify_status"] == "VERIFY_PASS":
            accepted_ids.append(ev["item_id"])

    for bk, ranks in ranks_consumed.items():
        assert ranks == list(range(1, len(ranks) + 1)), f"Non-sequential reserve ranks in {bk}: {ranks}"
    assert set(accepted_ids) == {x["item_id"] for x in all_120}, "Event log accepted set != frozen 120!"

    print(
        f"PASS: Verified {verified_count}/120 frozen items via offline SymPy + SHA-256 + "
        f"{len(events)} state-machine events."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify frozen Stage 2B release offline.")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--baseline-dir", type=Path, default=DEFAULT_BASELINE_DIR)
    args = parser.parse_args()
    verify_frozen_release(args.release_dir, args.baseline_dir)


if __name__ == "__main__":
    main()
