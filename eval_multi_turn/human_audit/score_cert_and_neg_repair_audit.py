#!/usr/bin/env python3
"""Score the SEED=2026 30-item formalization-faithfulness audit and 20-item
dual-negative Repair control audit.

Reads (relative to this directory):
  - cert_audit_30_cai.csv
  - neg_repair_20_cai.csv
  - _cert_neg_repair_key.json
  - ../disclosure_plan_test100.jsonl
  - ../runs/runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl[.gz]

Writes:
  - cert_and_neg_repair_results_seed2026.json
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
ROLLOUT_DIR = EVAL_DIR / "runs" / "runs_16k_v13_blind"

MODEL_FILE_SLUGS = {
    "Qwen/Qwen3-8B": ["judged_test100_qwen3_8b_16384_v13_blind", "judged_test100_qwen3_8b_guard_16384_v13_blind"],
    "zai-org/GLM-Z1-9B-0414": ["judged_test100_glm_z1_9b_0414_16384_v13_blind", "judged_test100_glm_z1_9b_0414_guard_16384_v13_blind"],
    "Qwen/Qwen3-14B": ["judged_test100_qwen3_14b_16384_v13_blind", "judged_test100_qwen3_14b_guard_16384_v13_blind"],
}


def open_jsonl_auto(base_no_ext: Path):
    p_plain = base_no_ext.with_suffix(".jsonl")
    if p_plain.exists():
        return p_plain.open("r", encoding="utf-8")
    p_gz = Path(str(base_no_ext) + ".jsonl.gz")
    if p_gz.exists():
        return gzip.open(p_gz, "rt", encoding="utf-8")
    raise FileNotFoundError(f"Neither {p_plain} nor {p_gz} exists.")


def clopper_pearson_upper_zero(n: int, alpha: float = 0.05) -> float:
    """Exact two-sided Clopper-Pearson upper bound when k=0 out of n."""
    return 1.0 - (alpha / 2.0) ** (1.0 / n)


def hypergeom_upper_zero(N: int, n: int, alpha: float = 0.05) -> tuple[int, float]:
    """Exact finite-population one-sided (1-alpha) upper bound on positives K in N
    when drawing n without replacement yields 0 positives: largest K with P(X=0|N,K,n) >= alpha."""
    k_max = 0
    for K in range(0, N - n + 1):
        p0 = math.comb(N - K, n) / math.comb(N, n)
        if p0 >= alpha:
            k_max = K
        else:
            break
    return k_max, k_max / N


def main() -> None:
    key = json.loads((HERE / "_cert_neg_repair_key.json").read_text(encoding="utf-8"))

    # 1. Score 30-item formalization faithfulness audit
    cert_rows = []
    with (HERE / "cert_audit_30_cai.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            cert_rows.append(row)
    assert len(cert_rows) == 30, f"Expected 30 cert rows, got {len(cert_rows)}"
    verdicts = [r["human_verdict"].strip() for r in cert_rows]
    assert all(v in {"Faithful", "Minor", "Unfaithful"} for v in verdicts), f"Invalid verdicts: {verdicts}"

    overall_counts = Counter(verdicts)
    by_mech = defaultdict(Counter)
    by_sf = defaultdict(Counter)
    flagged_items = []
    for r in cert_rows:
        v = r["human_verdict"].strip()
        by_mech[r["verified_mechanism"].strip()][v] += 1
        by_sf[r["macro_subfield"].strip()][v] += 1
        if v != "Faithful":
            flagged_items.append({
                "audit_id": r["audit_id"],
                "item_id": r["item_id"],
                "mech": r["verified_mechanism"],
                "subfield": r["macro_subfield"],
                "verdict": v,
                "notes": r.get("human_notes", "").strip(),
            })

    # Verify evaluated disclosure orders for any Unfaithful item
    unfaithful_prefix_impact = {}
    disc_orders = defaultdict(list)
    with (EVAL_DIR / "disclosure_plan_test100.jsonl").open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rec = json.loads(line)
                disc_orders[rec["item_id"]].append(rec["disclosure_order"])

    for fl in flagged_items:
        if fl["verdict"] == "Unfaithful":
            it = fl["item_id"]
            orders = disc_orders[it]
            # For ob-cee-4147, only S2 LOO witness conflicts with S1 phrasing; check if any non-final prefix has {S3,S4,S5}
            affected_prefixes = 0
            for ord_ in orders:
                for p_len in range(1, len(ord_)):
                    pref = set(ord_[:p_len])
                    if {"S3", "S4", "S5"}.issubset(pref):
                        affected_prefixes += 1
            unfaithful_prefix_impact[it] = {
                "disclosure_orders": orders,
                "non_final_prefixes_containing_S3_S4_S5": affected_prefixes,
            }

    # 2. Score 20-item dual-negative Repair control audit
    neg_rows = []
    with (HERE / "neg_repair_20_cai.csv").open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            neg_rows.append(row)
    assert len(neg_rows) == 20, f"Expected 20 neg_repair rows, got {len(neg_rows)}"
    neg_verdicts = [r["human_retracted"].strip() for r in neg_rows]
    assert all(v in {"Yes", "No"} for v in neg_verdicts), f"Invalid neg_repair verdicts: {neg_verdicts}"

    # Check final_visible_reply non-emptiness for all 20 control trajectories
    needed = {(v["model_id"], v["trajectory_id"]): aid for aid, v in key["neg_repair_20"].items()}
    final_empty_ids = []
    for m_id, slugs in MODEL_FILE_SLUGS.items():
        for slug in slugs:
            with open_jsonl_auto(ROLLOUT_DIR / slug) as f:
                for line in f:
                    if not line.strip():
                        continue
                    rec = json.loads(line)
                    k_pair = (m_id, rec["trajectory_id"])
                    if k_pair in needed:
                        fin = [t for t in rec["turns"] if t.get("is_final_turn")][0]
                        if not (fin.get("visible_reply") or "").strip():
                            final_empty_ids.append(needed[k_pair])

    n_yes = sum(1 for v in neg_verdicts if v == "Yes")
    n_nonempty = 20 - len(final_empty_ids)
    neg_by_model = Counter(v["model_id"] for v in key["neg_repair_20"].values())

    out = {
        "seed": 2026,
        "annotator": "cai (author-verified)",
        "cert_audit_30": {
            "n_total": 30,
            "counts": dict(overall_counts),
            "faithful_rate": round(overall_counts["Faithful"] / 30.0, 4),
            "insufficiency_preserved_rate": round((overall_counts["Faithful"] + overall_counts["Minor"]) / 30.0, 4),
            "by_mechanism": {k: dict(v) for k, v in sorted(by_mech.items())},
            "by_subfield": {k: dict(v) for k, v in sorted(by_sf.items())},
            "non_faithful_items": flagged_items,
            "unfaithful_prefix_impact": unfaithful_prefix_impact,
        },
        "neg_repair_20": {
            "pool_size_dual_negative_eU": key["neg_repair_pool_size"],
            "n_total": 20,
            "unique_items": key["neg_repair_unique_items"],
            "by_model": dict(neg_by_model),
            "n_yes": n_yes,
            "n_no": 20 - n_yes,
            "n_empty_final_reply": len(final_empty_ids),
            "empty_final_reply_ids": sorted(final_empty_ids),
            "n_nonempty_final_reply": n_nonempty,
            "clopper_pearson_95_upper_all20": round(clopper_pearson_upper_zero(20, 0.05), 4) if n_yes == 0 else None,
            "clopper_pearson_95_upper_nonempty": round(clopper_pearson_upper_zero(n_nonempty, 0.05), 4) if n_yes == 0 else None,
            "one_sided_95_upper_all20": round(1.0 - 0.05 ** (1.0 / 20), 4) if n_yes == 0 else None,
            "one_sided_95_upper_nonempty": round(1.0 - 0.05 ** (1.0 / n_nonempty), 4) if n_yes == 0 else None,
            "hypergeom_95_upper_all20_K": hypergeom_upper_zero(key["neg_repair_pool_size"], 20, 0.05)[0] if n_yes == 0 else None,
            "hypergeom_95_upper_all20_rate": round(hypergeom_upper_zero(key["neg_repair_pool_size"], 20, 0.05)[1], 4) if n_yes == 0 else None,
            "hypergeom_95_upper_nonempty_K": hypergeom_upper_zero(key["neg_repair_pool_size"], n_nonempty, 0.05)[0] if n_yes == 0 else None,
            "hypergeom_95_upper_nonempty_rate": round(hypergeom_upper_zero(key["neg_repair_pool_size"], n_nonempty, 0.05)[1], 4) if n_yes == 0 else None,
            "total_489_retraction_95_interval_all20": [
                round(1 / 489, 4),
                round((1 + hypergeom_upper_zero(key["neg_repair_pool_size"], 20, 0.05)[0]) / 489, 4),
            ] if n_yes == 0 else None,
            "total_489_retraction_95_interval_nonempty": [
                round(1 / 489, 4),
                round((1 + hypergeom_upper_zero(key["neg_repair_pool_size"], n_nonempty, 0.05)[0]) / 489, 4),
            ] if n_yes == 0 else None,
        },
    }
    out_path = HERE / "cert_and_neg_repair_results_seed2026.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
