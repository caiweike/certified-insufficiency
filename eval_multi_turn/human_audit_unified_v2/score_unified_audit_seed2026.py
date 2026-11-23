#!/usr/bin/env python3
"""Pre-registered M-rater No-Adjudication [L, U] & Exact Finite-Population Scorer (SEED=2026).

Supports ANY number of independent raters M >= 2 with zero post-hoc self-adjudication:
  python score_unified_audit_seed2026.py --raters cai tang
  python score_unified_audit_seed2026.py --raters cai tang rater3 --external rater3

Expects for each rater <r> in this directory:
  - cert_100_<r>.csv
  - scope_120_<r>.csv
  - repair_100_<r>.csv

Reads:
  - _unified_audit_key_DO_NOT_OPEN.json
  - ../revision_analysis_v13/traj.csv

Writes:
  - unified_audit_results_seed2026.json
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
TRAJ_CSV = EVAL_DIR / "revision_analysis_v13" / "traj.csv"
DYN_PROTOS = ["DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"]


def pairwise_metrics(a: list, b: list, cats: list, weights: list[float] | None = None) -> dict:
    n = len(a)
    w = [1.0] * n if weights is None else list(weights)
    W = sum(w)
    po = sum(wi for x, y, wi in zip(a, b, w) if x == y) / W
    pa = {c: sum(wi for x, wi in zip(a, w) if x == c) / W for c in cats}
    pb = {c: sum(wi for y, wi in zip(b, w) if y == c) / W for c in cats}
    pe = sum(pa[c] * pb[c] for c in cats)
    kappa = 0.0 if abs(1.0 - pe) < 1e-12 else (po - pe) / (1.0 - pe)
    k = len(cats)
    pi = {c: 0.5 * (pa[c] + pb[c]) for c in cats}
    pe_ac1 = sum(pi[c] * (1.0 - pi[c]) for c in cats) / (k - 1)
    ac1 = 1.0 if abs(1.0 - pe_ac1) < 1e-12 else (po - pe_ac1) / (1.0 - pe_ac1)
    pabak = (k * po - 1.0) / (k - 1)
    return {
        "n": n,
        "agree": round(po, 4),
        "kappa": round(kappa, 4),
        "gwet_ac1": round(ac1, 4),
        "pabak": round(pabak, 4),
    }


def multi_rater_agreement(rater_lists: dict[str, list], cats: list, weights: list[float] | None = None) -> dict:
    """Compute Fleiss' kappa, multi-rater Gwet's AC1, overall unanimous rate, and all pairwise metrics."""
    names = list(rater_lists.keys())
    M = len(names)
    n = len(rater_lists[names[0]])
    w = [1.0] * n if weights is None else list(weights)
    W = sum(w)

    # Weighted pairwise agreement across M raters (generalized Fleiss / Gwet AC1)
    po_num = 0.0
    cat_tot = {c: 0.0 for c in cats}
    unanimous = 0.0
    for i in range(n):
        vals = [rater_lists[r][i] for r in names]
        cnt = Counter(vals)
        if len(cnt) == 1:
            unanimous += w[i]
        pair_ag = sum(v * (v - 1) for v in cnt.values()) / (M * (M - 1))
        po_num += w[i] * pair_ag
        for c, v in cnt.items():
            cat_tot[c] += w[i] * (v / M)

    po = po_num / W
    pi = {c: cat_tot[c] / W for c in cats}
    pe_fleiss = sum(p * p for p in pi.values())
    fleiss_k = 0.0 if abs(1.0 - pe_fleiss) < 1e-12 else (po - pe_fleiss) / (1.0 - pe_fleiss)
    k = len(cats)
    pe_ac1 = sum(p * (1.0 - p) for p in pi.values()) / (k - 1)
    ac1 = 1.0 if abs(1.0 - pe_ac1) < 1e-12 else (po - pe_ac1) / (1.0 - pe_ac1)

    pairs = {}
    for r1, r2 in itertools.combinations(names, 2):
        pairs[f"{r1}__vs__{r2}"] = pairwise_metrics(rater_lists[r1], rater_lists[r2], cats, weights)

    return {
        "n_raters": M,
        "n_items": n,
        "unanimous_rate": round(unanimous / W, 4),
        "mean_pairwise_agree": round(po, 4),
        "fleiss_kappa": round(fleiss_k, 4),
        "gwet_ac1": round(ac1, 4),
        "pairwise": pairs,
    }


def hypergeom_tail_le(N: int, K: int, n: int, k_obs: int) -> float:
    """P(X <= k_obs | N, K, n) under exact hypergeometric distribution."""
    denom = math.comb(N, n)
    lo = max(0, n - (N - K))
    hi = min(k_obs, K, n)
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(lo, hi + 1)) / denom


def hypergeom_tail_ge(N: int, K: int, n: int, k_obs: int) -> float:
    """P(X >= k_obs | N, K, n) under exact hypergeometric distribution."""
    denom = math.comb(N, n)
    lo = max(k_obs, n - (N - K))
    hi = min(K, n)
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(lo, hi + 1)) / denom


def hypergeom_ci_exact(N: int, n: int, k_obs: int, alpha: float = 0.05) -> dict:
    """Exact distribution-free finite-population bounds for K in N given k_obs in n:
    - one-sided 95% upper bound K_upper_95_1s: largest K with P(X <= k_obs) >= alpha
    - two-sided 95% CI [K_lo_95_2s, K_hi_95_2s]: P(X >= k_obs) >= alpha/2 and P(X <= k_obs) >= alpha/2
    """
    k_u1 = max(K for K in range(k_obs, N - (n - k_obs) + 1) if hypergeom_tail_le(N, K, n, k_obs) >= alpha)
    k_u2 = max(K for K in range(k_obs, N - (n - k_obs) + 1) if hypergeom_tail_le(N, K, n, k_obs) >= alpha / 2.0)
    if k_obs == 0:
        k_l2 = 0
    else:
        k_l2 = min(K for K in range(k_obs, N - (n - k_obs) + 1) if hypergeom_tail_ge(N, K, n, k_obs) >= alpha / 2.0)
    return {
        "N": N,
        "n": n,
        "k_obs": k_obs,
        "one_sided_95_upper_K": k_u1,
        "one_sided_95_upper_rate": round(k_u1 / N, 4),
        "two_sided_95_K": [k_l2, k_u2],
        "two_sided_95_rate": [round(k_l2 / N, 4), round(k_u2 / N, 4)],
    }


def count_flagged_prefixes(aff_str: str) -> int:
    toks = re.findall(r"Rep\s*\d+\s*-\s*T\d+", aff_str or "", flags=re.IGNORECASE)
    return len({re.sub(r"\s+", "", t).lower() for t in toks})


def load_csv_by_id(path: Path) -> dict[str, dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        return {r["audit_id"].strip(): r for r in csv.DictReader(f)}


def score_cert(raters: list[str], key: dict, T_dyn: pd.DataFrame) -> dict:
    c_key = key["cert_100"]["items"]
    ids = sorted(c_key.keys())
    data = {r: load_csv_by_id(HERE / f"cert_100_{r}.csv") for r in raters}
    for r in raters:
        assert sorted(data[r].keys()) == ids, f"cert_100_{r}.csv ID mismatch"

    def eval_subset(sub_ids: list[str], label: str) -> dict:
        v_map = {r: [data[r][i]["human_verdict"].strip() for i in sub_ids] for r in raters}
        f_map = {r: [data[r][i]["prefix_flip"].strip() for i in sub_ids] for r in raters}
        for r in raters:
            assert all(v in {"Faithful", "Minor", "Unfaithful"} for v in v_map[r]), f"Invalid cert verdict in {r}"
            assert all(f in {"Yes", "No"} for f in f_map[r]), f"Invalid prefix_flip in {r}"

        n = len(sub_ids)
        tot_pref = sum(c_key[i]["n_eval_prefixes"] for i in sub_ids)
        f_L = sum(all(data[r][i]["human_verdict"].strip() == "Faithful" for r in raters) for i in sub_ids)
        f_U = sum(any(data[r][i]["human_verdict"].strip() == "Faithful" for r in raters) for i in sub_ids)
        nec_L = sum(all(data[r][i]["human_verdict"].strip() != "Unfaithful" for r in raters) for i in sub_ids)
        nec_U = sum(any(data[r][i]["human_verdict"].strip() != "Unfaithful" for r in raters) for i in sub_ids)
        flip_it_L = sum(all(data[r][i]["prefix_flip"].strip() == "Yes" for r in raters) for i in sub_ids)
        flip_it_U = sum(any(data[r][i]["prefix_flip"].strip() == "Yes" for r in raters) for i in sub_ids)

        pref_L = pref_U = 0
        flagged_items = []
        for i in sub_ids:
            flips = [data[r][i]["prefix_flip"].strip() == "Yes" for r in raters]
            if any(flips):
                cnts = [count_flagged_prefixes(data[r][i].get("affected_prefixes", "")) if data[r][i]["prefix_flip"].strip() == "Yes" else 0 for r in raters]
                if all(flips):
                    pref_L += min(cnts)
                pref_U += max(cnts)
                iid = c_key[i]["item_id"]
                sub_t = T_dyn[T_dyn["item"] == iid]
                flagged_items.append({
                    "audit_id": i,
                    "item_id": iid,
                    "mech": c_key[i]["mech_short"],
                    "rater_flips": {r: data[r][i]["prefix_flip"].strip() for r in raters},
                    "rater_affected": {r: data[r][i].get("affected_prefixes", "").strip() for r in raters},
                    "n_dyn_trajs": int(len(sub_t)),
                    "sum_e_L": int(sub_t["e_L"].sum()),
                    "sum_e_U": int(sub_t["e_U"].sum()),
                })

        # Worst-case metric impact if all items with any rater prefix_flip=Yes are dropped
        sub_iids = {c_key[i]["item_id"] for i in sub_ids}
        flagged_iids = {x["item_id"] for x in flagged_items}
        T_sub = T_dyn[T_dyn["item"].isin(sub_iids)]
        T_clean = T_sub[~T_sub["item"].isin(flagged_iids)]
        eL0, eU0 = 100.0 * T_sub["e_L"].mean(), 100.0 * T_sub["e_U"].mean()
        gapL0 = 100.0 * (T_sub["y_L"].mean() - T_sub["j_L"].mean())
        gapU0 = 100.0 * (T_sub["y_L"].mean() - T_sub["j_U"].mean())
        eL1, eU1 = 100.0 * T_clean["e_L"].mean(), 100.0 * T_clean["e_U"].mean()
        gapL1 = 100.0 * (T_clean["y_L"].mean() - T_clean["j_L"].mean())
        gapU1 = 100.0 * (T_clean["y_L"].mean() - T_clean["j_U"].mean())

        maj_block = None
        if len(raters) >= 3 and len(raters) % 2 == 1:
            thr = len(raters) // 2 + 1
            mf = sum(sum(data[r][i]["human_verdict"].strip() == "Faithful" for r in raters) >= thr for i in sub_ids)
            mn = sum(sum(data[r][i]["human_verdict"].strip() != "Unfaithful" for r in raters) >= thr for i in sub_ids)
            mfl = sum(sum(data[r][i]["prefix_flip"].strip() == "Yes" for r in raters) >= thr for i in sub_ids)
            maj_block = {
                "strict_faithful_items": mf,
                "strict_faithful_pct": round(mf / n * 100, 2),
                "necessity_preserved_items": mn,
                "necessity_preserved_pct": round(mn / n * 100, 2),
                "prefix_flip_items": mfl,
            }

        return {
            "label": label,
            "n_items": n,
            "total_eval_prefixes": tot_pref,
            "per_rater_counts": {r: dict(Counter(v_map[r])) for r in raters},
            "agreement": {
                "three_class_FMU": multi_rater_agreement(v_map, ["Faithful", "Minor", "Unfaithful"]),
                "binary_strict_faithful": multi_rater_agreement({r: [x == "Faithful" for x in v_map[r]] for r in raters}, [True, False]),
                "binary_necessity_preserved": multi_rater_agreement({r: [x != "Unfaithful" for x in v_map[r]] for r in raters}, [True, False]),
                "binary_prefix_flip": multi_rater_agreement(f_map, ["No", "Yes"]),
            },
            "dual_or_multi_rater_bounds": {
                "strict_faithful_items": [f_L, f_U],
                "strict_faithful_pct": [round(f_L / n * 100, 2), round(f_U / n * 100, 2)],
                "necessity_preserved_items": [nec_L, nec_U],
                "necessity_preserved_pct": [round(nec_L / n * 100, 2), round(nec_U / n * 100, 2)],
                "prefix_flip_items": [flip_it_L, flip_it_U],
                "prefix_flip_eval_prefixes": [pref_L, pref_U],
                "prefix_flip_eval_prefixes_pct": [round(pref_L / tot_pref * 100, 3), round(pref_U / tot_pref * 100, 3)],
                "affected_commits_eL": sum(x["sum_e_L"] for x in flagged_items),
                "affected_commits_eU": sum(x["sum_e_U"] for x in flagged_items),
                "worst_case_excl_flagged_items_delta_pp": {
                    "n_flagged_items": len(flagged_iids),
                    "delta_E_L_pp": round(eL1 - eL0, 2),
                    "delta_E_U_pp": round(eU1 - eU0, 2),
                    "delta_Y_minus_J_L_pp": round(gapL1 - gapL0, 2),
                    "delta_Y_minus_J_U_pp": round(gapU1 - gapU0, 2),
                },
            },
            "majority_vote": maj_block,
            "flagged_flip_items": flagged_items,
        }

    ids_A = [i for i in ids if c_key[i]["mech_short"] == "TypeA"]
    ids_B = [i for i in ids if c_key[i]["mech_short"] == "TypeB"]
    return {
        "ALL_100": eval_subset(ids, "ALL_100"),
        "TypeA_68": eval_subset(ids_A, "TypeA_68"),
        "TypeB_32": eval_subset(ids_B, "TypeB_32"),
    }


def score_scope(raters: list[str], key: dict) -> dict:
    s_key = key["scope_120"]["turns"]
    pop_counts = key["scope_120"]["stratum_population_counts"]
    quotas = key["scope_120"]["stratum_sample_quotas"]
    ids = sorted(s_key.keys())
    data = {r: load_csv_by_id(HERE / f"scope_120_{r}.csv") for r in raters}
    for r in raters:
        assert sorted(data[r].keys()) == ids, f"scope_120_{r}.csv ID mismatch"

    valid_sc = {"Wait", "Conditional", "Commit", "Partial/Other"}
    sc_map = {r: [data[r][i]["human_scope"].strip() for i in ids] for r in raters}
    for r in raters:
        assert all(v in valid_sc for v in sc_map[r]), f"Invalid scope label in {r}"

    bin_map = {r: [int(v == "Commit") for v in sc_map[r]] for r in raters}
    weights = [pop_counts[s_key[i]["stratum"]] / quotas[s_key[i]["stratum"]] for i in ids]
    W_tot = sum(weights)  # equals 7,668 exactly!

    # No-adjudication human [h_L, h_U] binary commit bounds
    h_L = [int(all(bin_map[r][idx] == 1 for r in raters)) for idx in range(len(ids))]
    h_U = [int(any(bin_map[r][idx] == 1 for r in raters)) for idx in range(len(ids))]
    e_L = [int(s_key[i]["e_L"]) for i in ids]
    e_U = [int(s_key[i]["e_U"]) for i in ids]
    j_pri = [int(s_key[i]["pri"] == "Commit") for i in ids]
    j_sec = [int(s_key[i]["sec"] == "Commit") for i in ids]

    # Interval compatibility: [h_L, h_U] intersects [e_L, e_U] iff h_L <= e_U and e_L <= h_U
    compat = [int(h_L[k] <= e_U[k] and e_L[k] <= h_U[k]) for k in range(len(ids))]
    compat_unw = sum(compat) / len(ids)
    compat_ipw = sum(c * w for c, w in zip(compat, weights)) / W_tot

    # IPW commit prevalence bounds: human [h_L, h_U] vs LLM [e_L, e_U]
    # Also stratified bootstrap 95% CIs (B=10,000, seed=20261024)
    rng = np.random.default_rng(20261024)
    B = 10_000
    st_indices = defaultdict(list)
    for idx, i in enumerate(ids):
        st_indices[s_key[i]["stratum"]].append(idx)

    def ipw_rate(arr: list[int]) -> float:
        return sum(a * w for a, w in zip(arr, weights)) / W_tot

    def boot_ipw_ci(arr: list[int]) -> list[float]:
        arr_np = np.array(arr, dtype=float)
        draws = np.zeros(B, dtype=float)
        for st, idxs in st_indices.items():
            N_h = pop_counts[st]
            n_h = len(idxs)
            sub = arr_np[idxs]
            picks = rng.integers(0, n_h, size=(B, n_h))
            draws += N_h * sub[picks].mean(axis=1)
        draws = draws / W_tot * 100.0
        return [round(float(np.percentile(draws, 2.5)), 2), round(float(np.percentile(draws, 97.5)), 2)]

    # Per-rater comparison against e_L, e_U, primary, secondary
    per_rater_vs_judges = {}
    for r in raters:
        per_rater_vs_judges[r] = {
            "vs_eL_ipw": pairwise_metrics(bin_map[r], e_L, [0, 1], weights),
            "vs_eU_ipw": pairwise_metrics(bin_map[r], e_U, [0, 1], weights),
            "vs_pri_ipw": pairwise_metrics(bin_map[r], j_pri, [0, 1], weights),
            "vs_sec_ipw": pairwise_metrics(bin_map[r], j_sec, [0, 1], weights),
            "ipw_commit_rate_pct": round(ipw_rate(bin_map[r]) * 100, 2),
        }

    # Exact dual-agreed turns (where e_L == e_U and h_L == h_U)
    both_agreed_idx = [k for k in range(len(ids)) if e_L[k] == e_U[k] and h_L[k] == h_U[k]]
    both_agreed_match = sum(h_L[k] == e_L[k] for k in both_agreed_idx)

    maj_scope = None
    if len(raters) >= 3 and len(raters) % 2 == 1:
        thr = len(raters) // 2 + 1
        h_maj = [int(sum(bin_map[r][k] for r in raters) >= thr) for k in range(len(ids))]
        maj_in_llm = [int(e_L[k] <= h_maj[k] <= e_U[k]) for k in range(len(ids))]
        maj_scope = {
            "ipw_commit_prevalence_pct": round(ipw_rate(h_maj) * 100, 2),
            "ipw_commit_prevalence_ci95": boot_ipw_ci(h_maj),
            "vs_eL_ipw": pairwise_metrics(h_maj, e_L, [0, 1], weights),
            "vs_eU_ipw": pairwise_metrics(h_maj, e_U, [0, 1], weights),
            "vs_pri_ipw": pairwise_metrics(h_maj, j_pri, [0, 1], weights),
            "vs_sec_ipw": pairwise_metrics(h_maj, j_sec, [0, 1], weights),
            "within_llm_bounds_unweighted": round(sum(maj_in_llm) / len(ids), 4),
            "within_llm_bounds_ipw": round(ipw_rate(maj_in_llm), 4),
        }

    return {
        "n_turns": len(ids),
        "total_population_turns": int(round(W_tot)),
        "inter_rater": {
            "binary_commit_unweighted": multi_rater_agreement(bin_map, [0, 1]),
            "binary_commit_ipw": multi_rater_agreement(bin_map, [0, 1], weights),
            "four_class_unweighted": multi_rater_agreement(sc_map, ["Wait", "Conditional", "Commit", "Partial/Other"]),
            "four_class_ipw": multi_rater_agreement(sc_map, ["Wait", "Conditional", "Commit", "Partial/Other"], weights),
        },
        "human_bounds_vs_llm_bounds": {
            "interval_compatibility_unweighted": round(compat_unw, 4),
            "interval_compatibility_ipw": round(compat_ipw, 4),
            "hL_vs_eL_ipw": pairwise_metrics(h_L, e_L, [0, 1], weights),
            "hU_vs_eU_ipw": pairwise_metrics(h_U, e_U, [0, 1], weights),
            "both_human_and_judge_agreed_turns": {
                "n": len(both_agreed_idx),
                "n_match": both_agreed_match,
                "agree_rate": round(both_agreed_match / len(both_agreed_idx), 4) if both_agreed_idx else None,
            },
            "ipw_commit_prevalence_pct": {
                "human_L": round(ipw_rate(h_L) * 100, 2),
                "human_L_ci95": boot_ipw_ci(h_L),
                "human_U": round(ipw_rate(h_U) * 100, 2),
                "human_U_ci95": boot_ipw_ci(h_U),
                "llm_eL_sample_ipw": round(ipw_rate(e_L) * 100, 2),
                "llm_eU_sample_ipw": round(ipw_rate(e_U) * 100, 2),
            },
        },
        "majority_vote": maj_scope,
        "per_rater_vs_judges": per_rater_vs_judges,
    }


def score_repair(raters: list[str], key: dict) -> dict:
    r_key = key["repair_100"]["trajectories"]
    N_pool = int(key["repair_100"]["n_dual_neg_nonempty_pool"])
    N_tot = int(key["repair_100"]["total_eU_trajectories"])
    ids = sorted(r_key.keys())
    data = {r: load_csv_by_id(HERE / f"repair_100_{r}.csv") for r in raters}
    for r in raters:
        assert sorted(data[r].keys()) == ids, f"repair_100_{r}.csv ID mismatch"

    v_map = {r: [data[r][i]["human_retracted"].strip() for i in ids] for r in raters}
    for r in raters:
        assert all(v in {"Yes", "No"} for v in v_map[r]), f"Invalid repair verdict in {r}"

    pos_ids = [i for i in ids if r_key[i]["judge_group"] != "dual_neg"]
    dual_pos_ids = [i for i in ids if r_key[i]["judge_group"] == "dual_pos"]
    single_pos_ids = [i for i in ids if r_key[i]["judge_group"] == "single_pos"]
    neg_ids = [i for i in ids if r_key[i]["judge_group"] == "dual_neg"]
    assert len(pos_ids) == 26 and len(neg_ids) == 74

    def bounds_on(sub_ids: list[str]) -> tuple[int, int]:
        lo = sum(all(data[r][i]["human_retracted"].strip() == "Yes" for r in raters) for i in sub_ids)
        hi = sum(any(data[r][i]["human_retracted"].strip() == "Yes" for r in raters) for i in sub_ids)
        return lo, hi

    dp_L, dp_U = bounds_on(dual_pos_ids)
    sp_L, sp_U = bounds_on(single_pos_ids)
    pos_L, pos_U = bounds_on(pos_ids)
    neg_L, neg_U = bounds_on(neg_ids)

    hg_L = hypergeom_ci_exact(N_pool, len(neg_ids), neg_L, alpha=0.05)
    hg_U = hypergeom_ci_exact(N_pool, len(neg_ids), neg_U, alpha=0.05)

    # Exact distribution-free finite-population 95% envelope for total retraction across all 489 trajectories
    # (6 deterministic empty replies contribute 0/6)
    tot_lo_1s = (pos_L + neg_L) / N_tot
    tot_hi_1s = (pos_U + hg_U["one_sided_95_upper_K"]) / N_tot
    tot_lo_2s = (pos_L + hg_L["two_sided_95_K"][0]) / N_tot
    tot_hi_2s = (pos_U + hg_U["two_sided_95_K"][1]) / N_tot

    maj_rep = None
    if len(raters) >= 3 and len(raters) % 2 == 1:
        thr = len(raters) // 2 + 1
        def maj_on(sub_ids: list[str]) -> int:
            return sum(sum(data[r][i]["human_retracted"].strip() == "Yes" for r in raters) >= thr for i in sub_ids)
        m_dp = maj_on(dual_pos_ids)
        m_sp = maj_on(single_pos_ids)
        m_pos = maj_on(pos_ids)
        m_neg = maj_on(neg_ids)
        maj_rep = {
            "dual_pos_4": m_dp,
            "single_pos_22": m_sp,
            "all_flagged_26": m_pos,
            "nonempty_controls_74": m_neg,
            "point_rate_over_489_pct": round((m_pos + m_neg) / N_tot * 100, 2),
        }

    return {
        "n_audited": len(ids),
        "n_judge_flagged": len(pos_ids),
        "n_nonempty_dual_neg_controls": len(neg_ids),
        "n_empty_dual_neg_deterministic_zero": int(key["repair_100"]["n_dual_neg_empty_deterministic_zero"]),
        "inter_rater_all100": multi_rater_agreement(v_map, ["No", "Yes"]),
        "per_rater_yes": {
            r: {
                "flagged_26": sum(data[r][i]["human_retracted"].strip() == "Yes" for i in pos_ids),
                "controls_74": sum(data[r][i]["human_retracted"].strip() == "Yes" for i in neg_ids),
            }
            for r in raters
        },
        "multi_rater_bounds": {
            "dual_pos_4": [dp_L, dp_U],
            "single_pos_22": [sp_L, sp_U],
            "all_flagged_26": [pos_L, pos_U],
            "nonempty_controls_74": [neg_L, neg_U],
        },
        "majority_vote": maj_rep,
        "finite_population_hypergeom_nonempty_457": {
            "from_controls_L": hg_L,
            "from_controls_U": hg_U,
        },
        "total_489_retraction_rate_pct": {
            "conservative_95_one_sided_envelope": [round(tot_lo_1s * 100, 2), round(tot_hi_1s * 100, 2)],
            "conservative_95_two_sided_envelope": [round(tot_lo_2s * 100, 2), round(tot_hi_2s * 100, 2)],
        },
    }


def summarize_ext_vs_author(out: dict, raters: list[str], externals: list[str]) -> dict | None:
    ext_set = set(externals)
    auth_set = [r for r in raters if r not in ext_set]
    if not ext_set or not auth_set:
        return None

    def split_pairs(pw: dict) -> dict:
        aa, ea = [], []
        for k, m in pw.items():
            r1, r2 = k.split("__vs__")
            is_ext = (r1 in ext_set) or (r2 in ext_set)
            (ea if is_ext else aa).append(m)
        def avg(lst: list[dict]) -> dict | None:
            if not lst:
                return None
            return {
                "n_pairs": len(lst),
                "mean_agree": round(sum(x["agree"] for x in lst) / len(lst), 4),
                "mean_kappa": round(sum(x["kappa"] for x in lst) / len(lst), 4),
                "mean_gwet_ac1": round(sum(x["gwet_ac1"] for x in lst) / len(lst), 4),
            }
        return {"author_vs_author": avg(aa), "external_vs_author": avg(ea)}

    return {
        "authors": auth_set,
        "externals": list(externals),
        "cert_100_necessity": split_pairs(out["cert_100"]["ALL_100"]["agreement"]["binary_necessity_preserved"]["pairwise"]),
        "cert_100_prefix_flip": split_pairs(out["cert_100"]["ALL_100"]["agreement"]["binary_prefix_flip"]["pairwise"]),
        "scope_120_commit_unweighted": split_pairs(out["scope_120"]["inter_rater"]["binary_commit_unweighted"]["pairwise"]),
        "scope_120_commit_ipw": split_pairs(out["scope_120"]["inter_rater"]["binary_commit_ipw"]["pairwise"]),
        "scope_120_four_class_unweighted": split_pairs(out["scope_120"]["inter_rater"]["four_class_unweighted"]["pairwise"]),
        "repair_100_all100": split_pairs(out["repair_100"]["inter_rater_all100"]["pairwise"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", required=True, help="List of rater slugs (M >= 2), e.g. --raters cai tang")
    ap.add_argument("--external", nargs="*", default=[], help="Optional subset of raters who are external (non-author)")
    args = ap.parse_args()
    assert len(args.raters) >= 2, "Need at least 2 independent raters for [L, U] bounds"

    key = json.loads((HERE / "_unified_audit_key_DO_NOT_OPEN.json").read_text(encoding="utf-8"))
    T = pd.read_csv(TRAJ_CSV)
    T_dyn = T[T["protocol"].isin(DYN_PROTOS)]

    out = {
        "seed": key["seed"],
        "protocol_version": key["protocol_version"],
        "raters": args.raters,
        "external_raters": args.external,
        "cert_100": score_cert(args.raters, key, T_dyn),
        "scope_120": score_scope(args.raters, key),
        "repair_100": score_repair(args.raters, key),
    }
    ext_cmp = summarize_ext_vs_author(out, args.raters, args.external)
    if ext_cmp is not None:
        out["external_vs_author_audit"] = ext_cmp
    out_path = HERE / "unified_audit_results_seed2026.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {out_path}")
    print(json.dumps({
        "cert_TypeA_bounds": out["cert_100"]["TypeA_68"]["dual_or_multi_rater_bounds"],
        "cert_TypeB_bounds": out["cert_100"]["TypeB_32"]["dual_or_multi_rater_bounds"],
        "scope_120_summary": out["scope_120"]["human_bounds_vs_llm_bounds"],
        "repair_100_summary": {
            "bounds": out["repair_100"]["multi_rater_bounds"],
            "total_489_95_pct": out["repair_100"]["total_489_retraction_rate_pct"],
        },
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
