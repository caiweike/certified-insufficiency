#!/usr/bin/env python3
"""Score the SEED=2027 Follow-Up Blind Human Audit (Traj-60 & Repair-50-TypeA).

Supports 1 independent external rater (minimum) or multiple raters (automatically
computing per-rater estimates + multi-rater [L, U] and majority bounds).

Usage:
  python score_followup_audit_seed2027.py --traj traj_60_yu.csv --denom repair50_denom_yu.csv
  python score_followup_audit_seed2027.py --traj traj_60_yu.csv --denom repair50_denom_yu.csv repair50_denom_cai.csv repair50_denom_tang.csv
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
KEY_PATH = HERE / "_followup_audit_key_DO_NOT_OPEN.json"
VALID_SCOPES = {"Wait", "Conditional", "Commit", "Partial/Other"}
STRATA_ORDER = ("agree_Commit", "disagree_Commit", "agree_NonCommit")


def read_csv_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def rater_name_from_path(path: Path, prefix: str) -> str:
    stem = path.stem
    if stem.startswith(prefix):
        return stem[len(prefix):]
    return stem


def stratified_srswor_stats(
    stratum_values: dict[str, list[int]],
    pop_counts: dict[str, int],
    n_boot: int = 10000,
    seed: int = 2027,
) -> dict:
    """Exact finite-population stratified SRSWOR mean, unbiased SE, normal & bootstrap 95% CIs."""
    N_total = sum(pop_counts[s] for s in STRATA_ORDER)
    est = 0.0
    var_hat = 0.0
    per_stratum = {}
    for s in STRATA_ORDER:
        vals = stratum_values[s]
        n_h = len(vals)
        N_h = pop_counts[s]
        W_h = N_h / N_total
        p_h = sum(vals) / n_h
        s2_h = (n_h / (n_h - 1)) * p_h * (1.0 - p_h) if n_h > 1 else 0.0
        fpc = 1.0 - (n_h / N_h)
        est += W_h * p_h
        var_hat += (W_h ** 2) * fpc * (s2_h / n_h)
        per_stratum[s] = {
            "N_h": N_h,
            "n_h": n_h,
            "W_h": round(W_h, 6),
            "hits": int(sum(vals)),
            "rate_pct": round(100.0 * p_h, 2),
        }

    se = math.sqrt(max(0.0, var_hat))
    ci_norm = [max(0.0, est - 1.96 * se), min(1.0, est + 1.96 * se)]

    # Stratified finite-population bootstrap (Bickel-Freedman / Gross style within-stratum resampling)
    rng = random.Random(seed)
    boot_ests = []
    # Build pseudo-populations per stratum preserving exact N_h and empirical proportion
    for _ in range(n_boot):
        b_est = 0.0
        for s in STRATA_ORDER:
            vals = stratum_values[s]
            n_h = len(vals)
            W_h = pop_counts[s] / N_total
            # Resample n_h with replacement, then apply FPC scaling around stratum mean
            draw_mean = sum(vals[rng.randrange(n_h)] for _ in range(n_h)) / n_h
            p_h = sum(vals) / n_h
            fpc_sqrt = math.sqrt(1.0 - n_h / pop_counts[s])
            adj = p_h + fpc_sqrt * (draw_mean - p_h)
            b_est += W_h * adj
        boot_ests.append(min(1.0, max(0.0, b_est)))
    boot_ests.sort()
    ci_boot = [boot_ests[int(0.025 * n_boot)], boot_ests[int(0.975 * n_boot)]]

    return {
        "estimate_pct": round(100.0 * est, 2),
        "se_pp": round(100.0 * se, 2),
        "ci95_normal_pct": [round(100.0 * ci_norm[0], 2), round(100.0 * ci_norm[1], 2)],
        "ci95_strat_boot_pct": [round(100.0 * ci_boot[0], 2), round(100.0 * ci_boot[1], 2)],
        "per_stratum": per_stratum,
    }


def score_traj_single(rows: list[dict], key_traj: dict) -> dict:
    trajs = key_traj["trajectories"]
    pop_counts = key_traj["stratum_population_counts"]

    # Index turn ratings
    by_tid: dict[str, dict[int, str]] = defaultdict(dict)
    for row in rows:
        tid = row["audit_id"].strip()
        t_idx = int(row["turn_idx"])
        scope = row["human_scope"].strip()
        if scope not in VALID_SCOPES:
            raise ValueError(f"Invalid or blank human_scope at {tid} T{t_idx}: {scope!r}")
        if t_idx in by_tid[tid]:
            raise ValueError(f"Duplicate turn rating: {tid} T{t_idx}")
        by_tid[tid][t_idx] = scope

    if set(by_tid.keys()) != set(trajs.keys()):
        missing = sorted(set(trajs.keys()) - set(by_tid.keys()))
        extra = sorted(set(by_tid.keys()) - set(trajs.keys()))
        raise ValueError(f"Trajectory ID mismatch: missing={missing[:5]}, extra={extra[:5]}")

    e_by_stratum: dict[str, list[int]] = {s: [] for s in STRATA_ORDER}
    yj_L_by_stratum: dict[str, list[int]] = {s: [] for s in STRATA_ORDER}
    yj_U_by_stratum: dict[str, list[int]] = {s: [] for s in STRATA_ORDER}
    # Also compute Commit+Conditional trajectory rate for diagnostic comparison
    cc_by_stratum: dict[str, list[int]] = {s: [] for s in STRATA_ORDER}
    turn_confusion = Counter()
    traj_confusion = Counter()

    for tid, info in sorted(trajs.items()):
        expected_turns = [int(t["turn_idx"]) for t in info["turns"]]
        if sorted(by_tid[tid].keys()) != sorted(expected_turns):
            raise ValueError(f"Turn mismatch on {tid}: expected {expected_turns}, got {sorted(by_tid[tid].keys())}")
        s_name = info["stratum"]
        scopes = [by_tid[tid][t_idx] for t_idx in expected_turns]
        h_E = int("Commit" in scopes)
        h_CC = int(any(sc in ("Commit", "Conditional") for sc in scopes))
        e_by_stratum[s_name].append(h_E)
        cc_by_stratum[s_name].append(h_CC)
        yj_L_by_stratum[s_name].append(int(info["y_L"]) * h_E)
        yj_U_by_stratum[s_name].append(int(info["y_U"]) * h_E)
        traj_confusion[(f"eL={info['e_L']},eU={info['e_U']}", f"hE={h_E}")] += 1
        for tinfo in info["turns"]:
            t_idx = int(tinfo["turn_idx"])
            turn_confusion[(f"eL={tinfo['e_L']},eU={tinfo['e_U']}", by_tid[tid][t_idx])] += 1

    return {
        "n_trajectories": len(trajs),
        "n_nonfinal_turns": sum(len(v) for v in by_tid.values()),
        "E_weighted": stratified_srswor_stats(e_by_stratum, pop_counts),
        "Y_minus_J_L_weighted": stratified_srswor_stats(yj_L_by_stratum, pop_counts),
        "Y_minus_J_U_weighted": stratified_srswor_stats(yj_U_by_stratum, pop_counts),
        "Commit_or_Cond_weighted": stratified_srswor_stats(cc_by_stratum, pop_counts),
        "traj_confusion_unweighted": {f"{k[0]} | {k[1]}": v for k, v in sorted(traj_confusion.items())},
        "turn_confusion_unweighted": {f"{k[0]} | {k[1]}": v for k, v in sorted(turn_confusion.items())},
    }


def parse_denom_single(rows: list[dict], key_denom: dict) -> dict[str, dict]:
    items = key_denom["items"]
    out = {}
    for row in rows:
        did = row["audit_id"].strip()
        if did not in items:
            raise ValueError(f"Unknown audit_id: {did}")
        q1 = row["q1_unconditional_claim"].strip()
        q2 = row["q2_conflicts_final"].strip()
        q3 = row["q3_pending_at_final"].strip()
        if q1 not in ("Yes", "No"):
            raise ValueError(f"Invalid q1 on {did}: {q1!r}")
        if q1 == "No":
            q2 = "NA"
            q3 = "NA"
        else:
            if q2 not in ("Yes", "No"):
                raise ValueError(f"Invalid q2 on {did} (q1=Yes): {q2!r}")
            if q2 == "No":
                q3 = "NA"
            else:
                if not items[did]["has_intervening"]:
                    if q3 not in ("Yes", ""):
                        raise ValueError(f"Impossible q3={q3!r} on {did} with no intervening turns")
                    q3 = "Yes"
                elif q3 not in ("Yes", "No"):
                    raise ValueError(f"Invalid q3 on {did} (q2=Yes, has intervening): {q3!r}")
        out[did] = {"q1": q1, "q2": q2, "q3": q3}

    if set(out.keys()) != set(items.keys()):
        missing = sorted(set(items.keys()) - set(out.keys()))
        raise ValueError(f"Missing Repair-50-TypeA IDs: {missing[:5]}")
    return out


def summarize_denom_map(dmap: dict[str, dict], key_denom: dict) -> dict:
    items = key_denom["items"]
    n1_ids = sorted(d for d, r in dmap.items() if r["q1"] == "Yes")
    n2_ids = sorted(d for d, r in dmap.items() if r["q1"] == "Yes" and r["q2"] == "Yes")
    n3_ids = sorted(d for d, r in dmap.items() if r["q1"] == "Yes" and r["q2"] == "Yes" and r["q3"] == "Yes")
    n3_no_interm = sum(1 for d in n3_ids if not items[d]["has_intervening"])
    n3_with_interm = sum(1 for d in n3_ids if items[d]["has_intervening"])
    n3_yL1 = sum(1 for d in n3_ids if items[d]["y_L"] == 1)
    return {
        "n_audited_TypeA": len(items),
        "N1_confirmed_unconditional_claim": len(n1_ids),
        "N2_conflicts_with_final": len(n2_ids),
        "N3_still_pending_at_final": len(n3_ids),
        "N3_breakdown": {
            "no_intervening_turns_t0_eq_k_minus_1": n3_no_interm,
            "with_intervening_turns_unretracted": n3_with_interm,
            "final_answer_correct_yL1": n3_yL1,
        },
        "retracted_at_final_out_of_N3": f"0/{len(n3_ids)}",
        "N1_orig_ids": [items[d]["orig_repair_audit_id"] for d in n1_ids],
        "N2_orig_ids": [items[d]["orig_repair_audit_id"] for d in n2_ids],
        "N3_orig_ids": [items[d]["orig_repair_audit_id"] for d in n3_ids],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traj", nargs="*", type=Path, default=[], help="Traj-60 CSV file(s)")
    parser.add_argument("--denom", nargs="*", type=Path, default=[], help="Repair-50-TypeA CSV file(s)")
    parser.add_argument("--out", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    key = json.loads(KEY_PATH.read_text(encoding="utf-8"))
    results: dict = {"seed": key["seed"], "protocol_version": key["protocol_version"]}

    if args.traj:
        traj_res = {}
        for p in args.traj:
            name = rater_name_from_path(p, "traj_60_")
            traj_res[name] = score_traj_single(read_csv_rows(p), key["traj_60"])
        results["traj_60"] = traj_res

    if args.denom:
        denom_per_rater = {}
        parsed_maps = {}
        for p in args.denom:
            name = rater_name_from_path(p, "repair50_denom_")
            pmap = parse_denom_single(read_csv_rows(p), key["repair_50_typea"])
            parsed_maps[name] = pmap
            denom_per_rater[name] = summarize_denom_map(pmap, key["repair_50_typea"])

        denom_section: dict = {"per_rater": denom_per_rater}
        if len(parsed_maps) >= 2:
            dids = sorted(key["repair_50_typea"]["items"].keys())
            raters = list(parsed_maps.keys())
            m = len(raters)
            agg = {}
            for label, fn in [
                ("unanimous_L", lambda flags: all(flags)),
                ("any_U", lambda flags: any(flags)),
                ("majority", lambda flags: sum(flags) > m / 2.0),
            ]:
                n1 = sum(fn([parsed_maps[r][d]["q1"] == "Yes" for r in raters]) for d in dids)
                n2 = sum(fn([parsed_maps[r][d]["q1"] == "Yes" and parsed_maps[r][d]["q2"] == "Yes" for r in raters]) for d in dids)
                n3 = sum(fn([parsed_maps[r][d]["q1"] == "Yes" and parsed_maps[r][d]["q2"] == "Yes" and parsed_maps[r][d]["q3"] == "Yes" for r in raters]) for d in dids)
                agg[label] = {
                    "N1_confirmed_unconditional_claim": n1,
                    "N2_conflicts_with_final": n2,
                    "N3_still_pending_at_final": n3,
                    "retracted_at_final_out_of_N3": f"0/{n3}",
                }
            denom_section["multi_rater_aggregate"] = agg
        results["repair_50_typea"] = denom_section

    text = json.dumps(results, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        args.out.write_text(text, encoding="utf-8", newline="\n")
    print(text)


if __name__ == "__main__":
    main()
