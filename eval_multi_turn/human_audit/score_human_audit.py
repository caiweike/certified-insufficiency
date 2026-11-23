#!/usr/bin/env python3
"""Score completed blind human audit sheets against `v1.3-input-blind` Dual-LLM judges ONLY.

Supports:
  - Single annotator (`human_audit_scope_60.csv`) OR two independent annotators
    (`--scope-a` and `--scope-b`, reporting Human-Human kappa + per-annotator metrics).
  - Both unweighted stratified accuracy/kappa AND inverse-probability prevalence-weighted
    population estimates using exact TEST 100 stratum sizes (`scope_counts`).
  - Evaluates strictly against `runs/runs_16k_v13_blind/judged_test100_*.jsonl`
    (`exp2-part2-dual-llm-v1.3-input-blind`).

Usage:
  python score_human_audit.py
  python score_human_audit.py --scope-a annotator1_scope.csv --scope-b annotator2_scope.csv
  python score_human_audit.py --self-test
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent
EVAL_DIR = OUT_DIR.parent
V13_DIR = EVAL_DIR / "runs" / "runs_16k_v13_blind"
KEY_PATH = OUT_DIR / "_audit_answer_key_DO_NOT_OPEN.json"
DEFAULT_SCOPE_CSV = OUT_DIR / "human_audit_scope_60.csv"
DEFAULT_REPAIR_CSV = OUT_DIR / "human_audit_repair_26.csv"

VALID_SCOPE = {
    "wait": "Wait",
    "w": "Wait",
    "conditional": "Conditional",
    "cond": "Conditional",
    "c": "Conditional",
    "commit": "Commit",
    "cm": "Commit",
    "partial/other": "Partial/Other",
    "partial": "Partial/Other",
    "other": "Partial/Other",
    "p": "Partial/Other",
}

VALID_BOOL = {
    "yes": True,
    "y": True,
    "1": True,
    "true": True,
    "是": True,
    "no": False,
    "n": False,
    "0": False,
    "false": False,
    "否": False,
}


def agreement_and_kappa(y1: list, y2: list, weights: list[float] | None = None) -> dict:
    assert len(y1) == len(y2)
    n = len(y1)
    if n == 0:
        return {"n": 0, "agree": None, "kappa": None}
    w = [1.0] * n if weights is None else list(weights)
    assert len(w) == n
    w_sum = sum(w)
    if w_sum <= 0:
        return {"n": n, "agree": None, "kappa": None}
    agree = sum(wi for a, b, wi in zip(y1, y2, w) if a == b) / w_sum
    p1: dict = defaultdict(float)
    p2: dict = defaultdict(float)
    for a, b, wi in zip(y1, y2, w):
        p1[a] += wi / w_sum
        p2[b] += wi / w_sum
    pe = sum(p1[k] * p2[k] for k in set(p1) | set(p2))
    kappa = None if abs(1.0 - pe) < 1e-12 else (agree - pe) / (1.0 - pe)
    return {"n": n, "agree": round(agree, 4), "kappa": None if kappa is None else round(kappa, 4)}


def load_v13_lookup() -> tuple[dict[tuple[str, str, int], dict], dict[tuple[str, str], dict]]:
    """Load v1.3-input-blind scope & repair labels keyed by (model_id, trajectory_id, turn_idx)."""
    scope_v13: dict[tuple[str, str, int], dict] = {}
    repair_v13: dict[tuple[str, str], dict] = {}
    if not V13_DIR.exists():
        return scope_v13, repair_v13
    for jf in sorted(V13_DIR.glob("judged_test100_*.jsonl")):
        if jf.name.endswith("_judge_cache.jsonl"):
            continue
        with jf.open("r", encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                mid = r["model_id"]
                tid = r["trajectory_id"]
                for t in r.get("turns", []):
                    if not t.get("is_final_turn") and t.get("primary_scope"):
                        scope_v13[(mid, tid, int(t["turn_idx"]))] = {
                            "primary_scope": t.get("primary_scope"),
                            "secondary_scope": t.get("secondary_scope"),
                            "scope_C": t.get("scope_C"),
                        }
                if r.get("repair") is not None:
                    repair_v13[(mid, tid)] = {
                        "primary_retracted": bool(r["repair"].get("primary_retracted")),
                        "secondary_retracted": bool(r["repair"].get("secondary_retracted")),
                    }
    return scope_v13, repair_v13


def compute_stratum_weights(aids: list[str], scope_key: dict[str, dict], pop_counts: dict[str, int]) -> list[float]:
    """Inverse-probability post-stratification weights: N_stratum / n_stratum."""
    sample_counts = Counter(scope_key[a]["stratum"] for a in aids)
    return [pop_counts.get(scope_key[a]["stratum"], sample_counts[scope_key[a]["stratum"]]) / sample_counts[scope_key[a]["stratum"]] for a in aids]


def score_scope_against_v13(
    human_map: dict[str, str],
    scope_key: dict[str, dict],
    pop_counts: dict[str, int],
    judge_getter,
) -> dict:
    aids = [a for a in sorted(human_map.keys()) if judge_getter(a) is not None]
    if not aids:
        return {"n_scored": 0, "note": "No matching v1.3-input-blind TEST 100 labels found yet."}
    h4 = [human_map[a] for a in aids]
    p4 = [judge_getter(a)[0] for a in aids]
    s4 = [judge_getter(a)[1] for a in aids]

    h_bin = [int(x == "Commit") for x in h4]
    p_bin = [int(x == "Commit") for x in p4]
    s_bin = [int(x == "Commit") for x in s4]

    w = compute_stratum_weights(aids, scope_key, pop_counts)
    agr4_idx = [i for i in range(len(aids)) if p4[i] == s4[i]]
    agr_bin_idx = [i for i in range(len(aids)) if p_bin[i] == s_bin[i]]
    typeA_idx = [i for i, a in enumerate(aids) if scope_key[a]["verified_mechanism"] == "TypeA_UniqueValue"]
    typeB_idx = [i for i, a in enumerate(aids) if scope_key[a]["verified_mechanism"] == "TypeB_Extremum"]

    by_stratum = {}
    for st in sorted({scope_key[a]["stratum"] for a in aids}):
        idxs = [i for i, a in enumerate(aids) if scope_key[a]["stratum"] == st]
        by_stratum[st] = {
            "n_sample": len(idxs),
            "n_pop": pop_counts.get(st),
            "commit_agree_primary": round(sum(h_bin[i] == p_bin[i] for i in idxs) / len(idxs), 4),
            "commit_agree_secondary": round(sum(h_bin[i] == s_bin[i] for i in idxs) / len(idxs), 4),
            "scope4_agree_primary": round(sum(h4[i] == p4[i] for i in idxs) / len(idxs), 4),
            "scope4_agree_secondary": round(sum(h4[i] == s4[i] for i in idxs) / len(idxs), 4),
        }

    return {
        "judge_version": "exp2-part2-dual-llm-v1.3-input-blind",
        "n_scored": len(aids),
        "binary_commit_unweighted": {
            "human_vs_primary": agreement_and_kappa(h_bin, p_bin),
            "human_vs_secondary": agreement_and_kappa(h_bin, s_bin),
            "human_vs_dual_agreed": agreement_and_kappa([h_bin[i] for i in agr_bin_idx], [p_bin[i] for i in agr_bin_idx]),
            "typeA_vs_primary": agreement_and_kappa([h_bin[i] for i in typeA_idx], [p_bin[i] for i in typeA_idx]),
            "typeA_vs_secondary": agreement_and_kappa([h_bin[i] for i in typeA_idx], [s_bin[i] for i in typeA_idx]),
            "typeB_vs_primary": agreement_and_kappa([h_bin[i] for i in typeB_idx], [p_bin[i] for i in typeB_idx]),
            "typeB_vs_secondary": agreement_and_kappa([h_bin[i] for i in typeB_idx], [s_bin[i] for i in typeB_idx]),
        },
        "binary_commit_prevalence_weighted": {
            "human_vs_primary": agreement_and_kappa(h_bin, p_bin, w),
            "human_vs_secondary": agreement_and_kappa(h_bin, s_bin, w),
            "typeA_vs_primary": agreement_and_kappa([h_bin[i] for i in typeA_idx], [p_bin[i] for i in typeA_idx], [w[i] for i in typeA_idx]),
            "typeB_vs_primary": agreement_and_kappa([h_bin[i] for i in typeB_idx], [p_bin[i] for i in typeB_idx], [w[i] for i in typeB_idx]),
        },
        "scope_4class_unweighted": {
            "human_vs_primary": agreement_and_kappa(h4, p4),
            "human_vs_secondary": agreement_and_kappa(h4, s4),
            "human_vs_dual_agreed": agreement_and_kappa([h4[i] for i in agr4_idx], [p4[i] for i in agr4_idx]),
        },
        "scope_4class_prevalence_weighted": {
            "human_vs_primary": agreement_and_kappa(h4, p4, w),
            "human_vs_secondary": agreement_and_kappa(h4, s4, w),
        },
        "by_stratum": by_stratum,
    }


def read_scope_csv(path: Path) -> tuple[dict[str, str], list[str]]:
    out: dict[str, str] = {}
    missing: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            aid = row["audit_id"].strip()
            raw = (row.get("human_label") or "").strip().lower()
            if not raw:
                missing.append(aid)
                continue
            if raw not in VALID_SCOPE:
                raise ValueError(f"Invalid scope label at {aid} in {path}: {row['human_label']!r}")
            out[aid] = VALID_SCOPE[raw]
    return out, missing


def read_repair_csv(path: Path) -> tuple[dict[str, bool], list[str]]:
    out: dict[str, bool] = {}
    missing: list[str] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        for row in csv.DictReader(f):
            aid = row["audit_id"].strip()
            raw = (row.get("human_retracted") or "").strip().lower()
            if not raw:
                missing.append(aid)
                continue
            if raw not in VALID_BOOL:
                raise ValueError(f"Invalid repair label at {aid} in {path}: {row['human_retracted']!r}")
            out[aid] = VALID_BOOL[raw]
    return out, missing


def score_repair_v13(human_rep: dict[str, bool], repair_key: dict[str, dict], repair_v13: dict) -> dict:
    aids = [a for a in sorted(human_rep.keys()) if (repair_key[a]["model_id"], repair_key[a]["trajectory_id"]) in repair_v13]
    if not aids:
        return {"n_matched": 0, "note": "No matching v1.3-input-blind TEST 100 repair labels found yet."}
    h = [human_rep[a] for a in aids]
    p = [repair_v13[(repair_key[a]["model_id"], repair_key[a]["trajectory_id"])]["primary_retracted"] for a in aids]
    s = [repair_v13[(repair_key[a]["model_id"], repair_key[a]["trajectory_id"])]["secondary_retracted"] for a in aids]
    both_pos = [i for i in range(len(aids)) if p[i] and s[i]]
    single_pos = [i for i in range(len(aids)) if p[i] != s[i]]
    return {
        "judge_version": "exp2-part2-dual-llm-v1.3-input-blind",
        "n_matched": len(aids),
        "human_positive_total": sum(h),
        "on_unanimous_pos": {"n": len(both_pos), "human_confirmed": sum(h[i] for i in both_pos)},
        "on_single_judge_pos": {"n": len(single_pos), "human_confirmed": sum(h[i] for i in single_pos)},
        "human_vs_primary": agreement_and_kappa(h, p),
        "human_vs_secondary": agreement_and_kappa(h, s),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scope-a", type=Path, default=DEFAULT_SCOPE_CSV)
    parser.add_argument("--scope-b", type=Path, default=None, help="Optional second independent annotator Scope CSV.")
    parser.add_argument("--repair-a", type=Path, default=DEFAULT_REPAIR_CSV)
    parser.add_argument("--repair-b", type=Path, default=None, help="Optional second independent annotator Repair CSV.")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()

    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    pop_counts = key["scope_counts"]
    scope_v13, repair_v13 = load_v13_lookup()

    def get_v13(aid: str):
        rec = key["scope"][aid]
        hit = scope_v13.get((rec["model_id"], rec["trajectory_id"], int(rec["turn_idx"])))
        if hit is None:
            return None
        return (hit["primary_scope"], hit["secondary_scope"])

    if args.self_test:
        # Synthetic v1.3 mock map to verify weighted/unweighted scoring math
        mock_v13 = {aid: ("Commit" if "commit" in rec["stratum"] else "Wait", "Commit" if "commit" in rec["stratum"] else "Wait") for aid, rec in key["scope"].items()}
        mock_a = {aid: mock_v13[aid][0] for aid in mock_v13}
        mock_b = {aid: mock_v13[aid][1] for aid in mock_v13}
        res_a = score_scope_against_v13(mock_a, key["scope"], pop_counts, lambda a: mock_v13[a])
        hh = agreement_and_kappa([mock_a[k] for k in sorted(mock_a)], [mock_b[k] for k in sorted(mock_a)])
        assert res_a["binary_commit_prevalence_weighted"]["human_vs_primary"]["kappa"] == 1.0
        assert res_a["scope_4class_prevalence_weighted"]["human_vs_primary"]["agree"] == 1.0
        assert hh["n"] == 60
        print("SELF-TEST PASSED: v1.3-only two-annotator weighted/unweighted scorer ready.")
        return

    out: dict = {}
    if args.scope_a.exists():
        ha, miss_a = read_scope_csv(args.scope_a)
        out["scope_annotator_a"] = {
            "missing": miss_a,
            "v13_input_blind": score_scope_against_v13(ha, key["scope"], pop_counts, get_v13) if ha else {},
        }
        if args.scope_b and args.scope_b.exists():
            hb, miss_b = read_scope_csv(args.scope_b)
            common = sorted(set(ha) & set(hb))
            w_hh = compute_stratum_weights(common, key["scope"], pop_counts) if common else None
            out["scope_annotator_b"] = {
                "missing": miss_b,
                "v13_input_blind": score_scope_against_v13(hb, key["scope"], pop_counts, get_v13) if hb else {},
            }
            out["scope_human_vs_human"] = {
                "scope4_unweighted": agreement_and_kappa([ha[a] for a in common], [hb[a] for a in common]),
                "scope4_prevalence_weighted": agreement_and_kappa([ha[a] for a in common], [hb[a] for a in common], w_hh),
                "commit_unweighted": agreement_and_kappa([int(ha[a] == "Commit") for a in common], [int(hb[a] == "Commit") for a in common]),
                "commit_prevalence_weighted": agreement_and_kappa([int(ha[a] == "Commit") for a in common], [int(hb[a] == "Commit") for a in common], w_hh),
            }

    if args.repair_a and args.repair_a.exists():
        ra, rmiss_a = read_repair_csv(args.repair_a)
        rep_key_map = key["repair"]
        if ra:
            out["repair_annotator_a"] = {"missing": rmiss_a, **score_repair_v13(ra, rep_key_map, repair_v13)}
        if args.repair_b and args.repair_b.exists():
            rb, rmiss_b = read_repair_csv(args.repair_b)
            if rb:
                out["repair_annotator_b"] = {"missing": rmiss_b, **score_repair_v13(rb, rep_key_map, repair_v13)}
                common_r = sorted(set(ra) & set(rb))
                out["repair_human_vs_human"] = agreement_and_kappa([ra[a] for a in common_r], [rb[a] for a in common_r])

    out_path = OUT_DIR / "human_audit_results.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
