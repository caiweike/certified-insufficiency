#!/usr/bin/env python3
"""Score the full 68-item TypeA dual-author independent certificate audit (SEED=2026)
using no-adjudication dual-rater [L, U] bounds and exact evaluated-prefix impact.

Reads (relative to this directory):
  - typeA68_audit_cai.csv
  - typeA68_audit_tang.csv
  - _typeA68_key.json
  - ../revision_analysis_v13/traj.csv

Writes:
  - typeA68_audit_results_seed2026.json
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
TRAJ_CSV = EVAL_DIR / "revision_analysis_v13" / "traj.csv"
DYN_PROTOS = ["DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"]


def agreement_metrics(a: list, b: list, cats: list) -> dict:
    n = len(a)
    n_ag = sum(x == y for x, y in zip(a, b))
    po = n_ag / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[c] / n) * (cb[c] / n) for c in cats)
    kappa = 0.0 if abs(1.0 - pe) < 1e-12 else (po - pe) / (1.0 - pe)
    k = len(cats)
    pi = {c: (ca[c] + cb[c]) / (2.0 * n) for c in cats}
    pe_ac1 = sum(pi[c] * (1.0 - pi[c]) for c in cats) / (k - 1)
    ac1 = 1.0 if abs(1.0 - pe_ac1) < 1e-12 else (po - pe_ac1) / (1.0 - pe_ac1)
    pabak = (k * po - 1.0) / (k - 1)
    return {
        "n": n,
        "n_agree": n_ag,
        "agree_rate": round(po, 4),
        "cohen_kappa": round(kappa, 4),
        "gwet_ac1": round(ac1, 4),
        "pabak": round(pabak, 4),
    }


def count_flagged_prefixes(aff_str: str) -> int:
    """Count distinct RepX-TY tokens specified in affected_prefixes."""
    toks = re.findall(r"Rep\s*\d+\s*-\s*T\d+", aff_str or "", flags=re.IGNORECASE)
    return len({re.sub(r"\s+", "", t).lower() for t in toks})


def main() -> None:
    key = json.loads((HERE / "_typeA68_key.json").read_text(encoding="utf-8"))
    cai = {r["audit_id"]: r for r in csv.DictReader((HERE / "typeA68_audit_cai.csv").open(encoding="utf-8-sig"))}
    tang = {r["audit_id"]: r for r in csv.DictReader((HERE / "typeA68_audit_tang.csv").open(encoding="utf-8-sig"))}
    ids = sorted(key["items"].keys())
    assert sorted(cai.keys()) == ids and sorted(tang.keys()) == ids and len(ids) == 68

    ca_v = [cai[i]["human_verdict"].strip() for i in ids]
    ta_v = [tang[i]["human_verdict"].strip() for i in ids]
    ca_f = [cai[i]["prefix_flip"].strip() for i in ids]
    ta_f = [tang[i]["prefix_flip"].strip() for i in ids]
    assert all(v in {"Faithful", "Minor", "Unfaithful"} for v in ca_v + ta_v)
    assert all(f in {"Yes", "No"} for f in ca_f + ta_f)

    # Dual-rater [L, U] bounds (no post-hoc self-adjudication)
    both_faithful = sum(x == "Faithful" and y == "Faithful" for x, y in zip(ca_v, ta_v))
    either_faithful = sum(x == "Faithful" or y == "Faithful" for x, y in zip(ca_v, ta_v))
    both_nec = sum(x != "Unfaithful" and y != "Unfaithful" for x, y in zip(ca_v, ta_v))
    either_nec = sum(x != "Unfaithful" or y != "Unfaithful" for x, y in zip(ca_v, ta_v))
    both_flip_items = sum(x == "Yes" and y == "Yes" for x, y in zip(ca_f, ta_f))
    either_flip_items = sum(x == "Yes" or y == "Yes" for x, y in zip(ca_f, ta_f))

    # Count affected evaluated prefixes under [L, U]
    total_eval_prefixes = int(key["total_evaluated_nonfinal_prefixes"])
    prefixes_L = 0
    prefixes_U = 0
    flagged_flip_items = []
    disagreements = []

    T = pd.read_csv(TRAJ_CSV)
    T_dyn = T[T["protocol"].isin(DYN_PROTOS)]

    for i in ids:
        iid = key["items"][i]["item_id"]
        cv, tv = cai[i]["human_verdict"].strip(), tang[i]["human_verdict"].strip()
        cf, tf = cai[i]["prefix_flip"].strip(), tang[i]["prefix_flip"].strip()
        ca_aff, ta_aff = cai[i].get("affected_prefixes", "").strip(), tang[i].get("affected_prefixes", "").strip()
        if cv != tv or cf != tf:
            disagreements.append({
                "audit_id": i,
                "item_id": iid,
                "cai_verdict": cv,
                "cai_flip": cf,
                "cai_affected": ca_aff,
                "tang_verdict": tv,
                "tang_flip": tf,
                "tang_affected": ta_aff,
            })
        if cf == "Yes" or tf == "Yes":
            n_c = count_flagged_prefixes(ca_aff) if cf == "Yes" else 0
            n_t = count_flagged_prefixes(ta_aff) if tf == "Yes" else 0
            if cf == "Yes" and tf == "Yes":
                prefixes_L += min(n_c, n_t)
            prefixes_U += max(n_c, n_t)
            sub = T_dyn[T_dyn["item"] == iid]
            flagged_flip_items.append({
                "audit_id": i,
                "item_id": iid,
                "cai_affected": ca_aff,
                "tang_affected": ta_aff,
                "n_prefixes_U": max(n_c, n_t),
                "n_dyn_trajs": int(len(sub)),
                "sum_e_L": int(sub["e_L"].sum()),
                "sum_e_U": int(sub["e_U"].sum()),
            })

    total_affected_eL = sum(x["sum_e_L"] for x in flagged_flip_items)
    total_affected_eU = sum(x["sum_e_U"] for x in flagged_flip_items)

    cm = Counter(zip(ca_v, ta_v))
    out = {
        "seed": 2026,
        "protocol": "full 68/68 TypeA independent dual-author census (no post-hoc adjudication; dual-rater [L, U] bounds)",
        "n_typeA_items": 68,
        "total_evaluated_nonfinal_prefixes": total_eval_prefixes,
        "total_unique_evaluated_prefixes": int(key["total_unique_evaluated_prefixes"]),
        "rater_counts": {
            "cai": {"verdict": dict(Counter(ca_v)), "prefix_flip": dict(Counter(ca_f))},
            "tang": {"verdict": dict(Counter(ta_v)), "prefix_flip": dict(Counter(ta_f))},
        },
        "confusion_cai_vs_tang": {f"{k[0]}__{k[1]}": v for k, v in sorted(cm.items())},
        "inter_rater_agreement": {
            "three_class_FMU": agreement_metrics(ca_v, ta_v, ["Faithful", "Minor", "Unfaithful"]),
            "binary_strict_faithful": agreement_metrics([x == "Faithful" for x in ca_v], [x == "Faithful" for x in ta_v], [True, False]),
            "binary_necessity_preserved": agreement_metrics([x != "Unfaithful" for x in ca_v], [x != "Unfaithful" for x in ta_v], [True, False]),
            "binary_prefix_flip": agreement_metrics(ca_f, ta_f, ["No", "Yes"]),
        },
        "dual_rater_bounds": {
            "strict_faithful_items": [both_faithful, either_faithful],
            "strict_faithful_pct": [round(both_faithful / 68 * 100, 2), round(either_faithful / 68 * 100, 2)],
            "necessity_preserved_items": [both_nec, either_nec],
            "necessity_preserved_pct": [round(both_nec / 68 * 100, 2), round(either_nec / 68 * 100, 2)],
            "prefix_flip_items": [both_flip_items, either_flip_items],
            "prefix_flip_eval_prefixes": [prefixes_L, prefixes_U],
            "prefix_flip_eval_prefixes_pct": [round(prefixes_L / total_eval_prefixes * 100, 3), round(prefixes_U / total_eval_prefixes * 100, 3)],
            "affected_premature_commits_eL": total_affected_eL,
            "affected_premature_commits_eU": total_affected_eU,
            "max_metric_bias_pp": 0.0 if total_affected_eU == 0 else None,
        },
        "flagged_flip_items": flagged_flip_items,
        "disagreements": disagreements,
    }
    out_path = HERE / "typeA68_audit_results_seed2026.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
