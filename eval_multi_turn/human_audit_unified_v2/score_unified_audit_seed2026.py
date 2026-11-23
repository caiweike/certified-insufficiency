#!/usr/bin/env python3
"""Design-based M-rater No-Adjudication [L, U] Audit Scorer (SEED=2026).

Supports ANY number of independent raters M >= 2 with zero post-hoc self-adjudication:
  python score_unified_audit_seed2026.py --raters cai tang yu --external yu

Design-based inference correcting for balanced_pick() unequal inclusion probabilities:
  - Scope-120 : Estimates exact unit inclusion probabilities pi_i via M=50,000 Monte Carlo
                replays of balanced_pick() (exact lazy Fisher-Yates equivalent), then
                computes stratum-calibrated Hajek estimates with conservative linearized
                Hansen-Hurwitz unequal-probability 95% CIs (also reporting raw HT and
                naive N_h/n_h for sensitivity comparison).
  - Repair-100: Computes exact unit inclusion probabilities pi_j across all 457 non-empty
                dual-negative controls via M=50,000 replays of balanced_pick() per model,
                and computes a Monte Carlo design-calibrated adversarial 95% upper bound
                (max_{|S|=K} P_design(S cap A = empty) >= alpha via greedy + 1-swap search
                and item-cluster packing). Naive SRSWOR hypergeometric is retained ONLY as
                a documented diagnostic contrast showing why SRSWOR understates miss risk.

Expects for each rater <r> in this directory:
  - cert_100_<r>.csv
  - scope_120_<r>.csv
  - repair_100_<r>.csv

Reads:
  - _unified_audit_key_DO_NOT_OPEN.json
  - build_unified_audit_seed2026.py (population definitions & balanced_pick constants)
  - ../runs/runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl[.gz]
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
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
TRAJ_CSV = EVAL_DIR / "revision_analysis_v13" / "traj.csv"
DYN_PROTOS = ["DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"]

sys.path.insert(0, str(HERE))
import build_unified_audit_seed2026 as bmod  # noqa: E402


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
    denom = math.comb(N, n)
    lo = max(0, n - (N - K))
    hi = min(k_obs, K, n)
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(lo, hi + 1)) / denom


def hypergeom_tail_ge(N: int, K: int, n: int, k_obs: int) -> float:
    denom = math.comb(N, n)
    lo = max(k_obs, n - (N - K))
    hi = min(K, n)
    return sum(math.comb(K, x) * math.comb(N - K, n - x) for x in range(lo, hi + 1)) / denom


def hypergeom_ci_srswor_diagnostic(N: int, n: int, k_obs: int, alpha: float = 0.05) -> dict:
    """Naive SRSWOR hypergeometric CI — retained ONLY as a diagnostic contrast to balanced_pick."""
    k_u1 = max(K for K in range(k_obs, N - (n - k_obs) + 1) if hypergeom_tail_le(N, K, n, k_obs) >= alpha)
    k_u2 = max(K for K in range(k_obs, N - (n - k_obs) + 1) if hypergeom_tail_le(N, K, n, k_obs) >= alpha / 2.0)
    k_l2 = 0 if k_obs == 0 else min(
        K for K in range(k_obs, N - (n - k_obs) + 1) if hypergeom_tail_ge(N, K, n, k_obs) >= alpha / 2.0
    )
    return {
        "N": N,
        "n": n,
        "k_obs": k_obs,
        "one_sided_95_upper_K": k_u1,
        "one_sided_95_upper_rate": round(k_u1 / N, 4),
        "two_sided_95_K": [k_l2, k_u2],
        "two_sided_95_rate": [round(k_l2 / N, 4), round(k_u2 / N, 4)],
    }


def load_populations() -> tuple[dict[str, list[dict]], dict[str, list[dict]]]:
    """Reconstruct exact Scope-120 and Repair-100 population pools identically to build_unified_audit_seed2026.py."""
    scope_pools: dict[str, list[dict]] = {k: [] for k in bmod.SCOPE_QUOTAS}
    repair_neg_by_model: dict[str, list[dict]] = defaultdict(list)

    for slug in bmod.JUDGED_SLUGS:
        for rec in bmod.read_jsonl_auto(bmod.V13_DIR / f"{slug}.jsonl"):
            proto = rec.get("protocol")
            if proto not in bmod.DYN_PROTOS:
                continue
            mid = rec["model_id"]
            tid = rec["trajectory_id"]
            iid = rec["item_id"]
            mech_short = "TypeA" if rec.get("verified_mechanism") == "TypeA_UniqueValue" else "TypeB"
            turns = rec["turns"]
            for td in turns:
                if td.get("is_final_turn"):
                    continue
                t_idx = int(td["turn_idx"])
                st = bmod.scope_stratum(
                    str(td.get("primary_scope") or ""),
                    str(td.get("secondary_scope") or ""),
                    mech_short,
                )
                scope_pools[st].append({
                    "unique_key": (mid, tid, t_idx),
                    "model_id": mid,
                    "protocol": proto,
                    "item_id": iid,
                    "stratum": st,
                })
            if int(rec.get("e_U") or 0) == 1:
                rep_obj = rec.get("repair") or {}
                if not rep_obj.get("primary_retracted") and not rep_obj.get("secondary_retracted"):
                    fin_td = [td for td in turns if td.get("is_final_turn")][0]
                    if (fin_td.get("visible_reply") or "").strip():
                        repair_neg_by_model[mid].append({
                            "unique_key": (mid, tid),
                            "model_id": mid,
                            "protocol": proto,
                            "item_id": iid,
                        })
    return scope_pools, repair_neg_by_model


def draw_balanced_pick_fast(
    item_ids: list[str],
    bucket_members: list[list[int]],
    n: int,
    rng: random.Random,
) -> list[int]:
    """Distributionally exact, fast O(n + k_buckets) simulator of build_unified_audit_seed2026.balanced_pick().

    Proof of distributional equivalence:
      Shuffling the whole pool uniformly induces independent uniform permutations within
      each group bucket. Popping from the end of each bucket is therefore identically
      distributed to lazy Fisher-Yates sampling without replacement inside each bucket.
    """
    nb = len(bucket_members)
    b_order = list(range(nb))
    rng.shuffle(b_order)

    # Maintain lazy Fisher-Yates state only for buckets actually touched
    b_copies = [list(m) for m in bucket_members]
    b_rem = [len(m) for m in bucket_members]

    picked: list[int] = []
    used_items: set[str] = set()
    used_idx: set[int] = set()

    # Pass 1: unique items across balanced buckets
    progressed = True
    while len(picked) < n and progressed:
        progressed = False
        for bi in b_order:
            if len(picked) >= n:
                break
            rem = b_rem[bi]
            arr = b_copies[bi]
            while rem > 0:
                j = rng.randrange(rem)
                cand = arr[j]
                rem -= 1
                arr[j] = arr[rem]
                arr[rem] = cand
                if item_ids[cand] not in used_items:
                    picked.append(cand)
                    used_items.add(item_ids[cand])
                    used_idx.add(cand)
                    progressed = True
                    break
            b_rem[bi] = rem

    # Pass 2: remaining unused units balanced across buckets (only reached if unique items < n)
    if len(picked) < n:
        rem_buckets: list[list[int]] = []
        for m in bucket_members:
            left = [x for x in m if x not in used_idx]
            if left:
                rem_buckets.append(left)
        r_order = list(range(len(rem_buckets)))
        rng.shuffle(r_order)
        r_rem = [len(m) for m in rem_buckets]
        progressed = True
        while len(picked) < n and progressed:
            progressed = False
            for ri in r_order:
                if len(picked) >= n:
                    break
                rem = r_rem[ri]
                if rem > 0:
                    arr = rem_buckets[ri]
                    j = rng.randrange(rem)
                    cand = arr[j]
                    rem -= 1
                    arr[j] = arr[rem]
                    picked.append(cand)
                    used_idx.add(cand)
                    r_rem[ri] = rem
                    progressed = True
    return picked


def simulate_scope_inclusion_probs(scope_pools: dict[str, list[dict]], mc_draws: int, seed: int = 20260501) -> tuple[dict, dict]:
    """Estimate exact design inclusion probabilities pi_i for all 7,668 non-final turns under balanced_pick."""
    rng = random.Random(seed)
    pi_lookup: dict[tuple, float] = {}
    stratum_diag: dict[str, dict] = {}
    for st, q in bmod.SCOPE_QUOTAS.items():
        pool_s = sorted(scope_pools[st], key=lambda x: x["unique_key"])
        N_h = len(pool_s)
        item_ids = [x["item_id"] for x in pool_s]
        bmap: dict[tuple, list[int]] = defaultdict(list)
        for i, x in enumerate(pool_s):
            bmap[(x["model_id"], x["protocol"])].append(i)
        bucket_members = [bmap[k] for k in sorted(bmap.keys())]

        hits = np.zeros(N_h, dtype=np.int64)
        for _ in range(mc_draws):
            for idx in draw_balanced_pick_fast(item_ids, bucket_members, q, rng):
                hits[idx] += 1
        pis = hits / float(mc_draws)
        for i, x in enumerate(pool_s):
            pi_lookup[x["unique_key"]] = float(pis[i])
        stratum_diag[st] = {
            "N": N_h,
            "n": q,
            "naive_pi": round(q / N_h, 5),
            "design_pi_min": round(float(pis.min()), 5),
            "design_pi_median": round(float(np.median(pis)), 5),
            "design_pi_max": round(float(pis.max()), 5),
            "max_to_min_ratio": round(float(pis.max() / max(pis.min(), 1e-12)), 2),
        }
    return pi_lookup, stratum_diag


def simulate_repair_design_bound(
    repair_neg_by_model: dict[str, list[dict]],
    mc_draws: int,
    seed: int = 20260505,
) -> dict:
    """Monte Carlo design-calibrated adversarial 95% upper bound under balanced_pick for Repair-100 controls (N=457, n=74)."""
    rng = random.Random(seed)
    mids = list(bmod.REPAIR_NEG_MODEL_QUOTAS.keys())
    model_diag: dict[str, dict] = {}
    worst_p: dict[str, list[float]] = {}
    max_k_per_model = 45

    for mid in mids:
        q = bmod.REPAIR_NEG_MODEL_QUOTAS[mid]
        pool_s = sorted(repair_neg_by_model[mid], key=lambda x: x["unique_key"])
        N_m = len(pool_s)
        item_ids = [x["item_id"] for x in pool_s]
        bmap: dict[str, list[int]] = defaultdict(list)
        for i, x in enumerate(pool_s):
            bmap[x["protocol"]].append(i)
        bucket_members = [bmap[k] for k in sorted(bmap.keys())]

        not_picked = np.ones((N_m, mc_draws), dtype=bool)
        for d in range(mc_draws):
            for idx in draw_balanced_pick_fast(item_ids, bucket_members, q, rng):
                not_picked[idx, d] = False

        pis = 1.0 - not_picked.mean(axis=1)
        model_diag[mid] = {
            "N": N_m,
            "n": q,
            "naive_pi": round(q / N_m, 4),
            "design_pi_min": round(float(pis.min()), 4),
            "design_pi_median": round(float(np.median(pis)), 4),
            "design_pi_max": round(float(pis.max()), 4),
        }

        # Adversarial subset maximization of P_design(S cap A = empty | |S|=k):
        # Greedy forward selection + 1-swap local refinement, plus whole-item cluster packing check
        cur = np.ones(mc_draws, dtype=bool)
        chosen: list[int] = []
        p_best = [1.0]
        for k in range(1, max_k_per_model + 1):
            best_i, best_c = -1, -1
            for i in range(N_m):
                if i in chosen:
                    continue
                c = int(np.count_nonzero(cur & not_picked[i]))
                if c > best_c:
                    best_c = c
                    best_i = i
            chosen.append(best_i)
            cur = cur & not_picked[best_i]

            s_set = set(chosen)
            improved = True
            while improved:
                improved = False
                for out_i in list(s_set):
                    rem = s_set - {out_i}
                    base = np.ones(mc_draws, dtype=bool)
                    for j in rem:
                        base &= not_picked[j]
                    best_in, best_val = out_i, best_c
                    for in_i in range(N_m):
                        if in_i in rem:
                            continue
                        val = int(np.count_nonzero(base & not_picked[in_i]))
                        if val > best_val:
                            best_val = val
                            best_in = in_i
                    if best_in != out_i:
                        s_set.remove(out_i)
                        s_set.add(best_in)
                        best_c = best_val
                        improved = True
            p_best.append(best_c / float(mc_draws))

        icnt = Counter(item_ids)
        items_sorted = sorted(
            icnt.keys(),
            key=lambda iid: (-sum(1 for x in pool_s if x["item_id"] == iid and x["protocol"] == "DYN_FREE"), -icnt[iid]),
        )
        ord_whole = [i for iid in items_sorted for i, x in enumerate(pool_s) if x["item_id"] == iid]
        cm = np.ones(mc_draws, dtype=bool)
        for k in range(1, max_k_per_model + 1):
            cm = cm & not_picked[ord_whole[k - 1]]
            v = int(np.count_nonzero(cm)) / float(mc_draws)
            if v > p_best[k]:
                p_best[k] = v

        worst_p[mid] = p_best

    # Combine across the 3 independent models: max_{k1+k2+k3 = K} prod_m P_m(k_m)
    p_comb: dict[int, float] = {}
    best_split: dict[int, tuple[int, int, int]] = {}
    for K in range(0, 60):
        best = 0.0
        sp = (0, 0, 0)
        for k1 in range(min(K, max_k_per_model) + 1):
            for k2 in range(min(K - k1, max_k_per_model) + 1):
                k3 = K - k1 - k2
                if k3 > max_k_per_model:
                    continue
                val = worst_p[mids[0]][k1] * worst_p[mids[1]][k2] * worst_p[mids[2]][k3]
                if val > best:
                    best = val
                    sp = (k1, k2, k3)
        p_comb[K] = best
        best_split[K] = sp

    k95_1s = max(K for K, v in p_comb.items() if v >= 0.05)
    k95_2s = max(K for K, v in p_comb.items() if v >= 0.025)
    return {
        "mc_draws_per_model": mc_draws,
        "mc_seed": seed,
        "model_inclusion_probs": model_diag,
        "worst_case_P_zero_obs_at_K16_srswor_cutoff": round(p_comb[16], 4),
        "one_sided_95_upper_K": k95_1s,
        "one_sided_95_upper_rate": round(k95_1s / 457.0, 4),
        "one_sided_95_P_at_K": round(p_comb[k95_1s], 4),
        "one_sided_95_split": list(best_split[k95_1s]),
        "two_sided_95_upper_K": k95_2s,
        "two_sided_95_upper_rate": round(k95_2s / 457.0, 4),
        "two_sided_95_P_at_K": round(p_comb[k95_2s], 4),
        "two_sided_95_split": list(best_split[k95_2s]),
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
                cnts = [
                    count_flagged_prefixes(data[r][i].get("affected_prefixes", ""))
                    if data[r][i]["prefix_flip"].strip() == "Yes"
                    else 0
                    for r in raters
                ]
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


def score_scope(
    raters: list[str],
    key: dict,
    pi_lookup: dict[tuple, float],
    stratum_diag: dict[str, dict],
    mc_draws: int,
) -> dict:
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

    # Design weights:
    # 1. Raw Horvitz-Thompson w_ht = 1 / pi_i
    # 2. Stratum-calibrated Hajek w_hajek = N_h * (1 / pi_i) / sum_{j in s_h} (1 / pi_j)  [PRIMARY]
    # 3. Naive N_h / n_h [sensitivity comparison only]
    strata_list = [s_key[i]["stratum"] for i in ids]
    raw_ht = [1.0 / pi_lookup[(s_key[i]["model_id"], s_key[i]["trajectory_id"], int(s_key[i]["turn_idx"]))] for i in ids]
    naive_w = [pop_counts[s_key[i]["stratum"]] / quotas[s_key[i]["stratum"]] for i in ids]

    st_indices: dict[str, list[int]] = defaultdict(list)
    for idx, st in enumerate(strata_list):
        st_indices[st].append(idx)

    hajek_w = [0.0] * len(ids)
    for st, idxs in st_indices.items():
        s_ht = sum(raw_ht[k] for k in idxs)
        N_h = float(pop_counts[st])
        for k in idxs:
            hajek_w[k] = N_h * (raw_ht[k] / s_ht)

    N_tot = float(sum(pop_counts.values()))  # 7,668 exactly

    def w_mean(arr: list[int], w: list[float]) -> float:
        return sum(a * wi for a, wi in zip(arr, w)) / sum(w)

    def hajek_linearized_ci(arr: list[int]) -> dict:
        """Stratum-calibrated Hajek ratio estimator with conservative linearized Hansen-Hurwitz 95% CI."""
        est = 0.0
        var = 0.0
        for st, idxs in st_indices.items():
            N_h = float(pop_counts[st])
            n_h = len(idxs)
            d = np.array([raw_ht[k] for k in idxs], dtype=float)
            y = np.array([arr[k] for k in idxs], dtype=float)
            ybar_h = float(np.sum(d * y) / np.sum(d))
            est += (N_h / N_tot) * ybar_h
            u = n_h * (d / np.sum(d)) * (y - ybar_h)
            var_h = float(np.sum(u * u) / (n_h * (n_h - 1)))
            var += ((N_h / N_tot) ** 2) * var_h
        se = math.sqrt(var)
        lo = max(0.0, est - 1.96 * se)
        hi = min(1.0, est + 1.96 * se)
        return {
            "pct": round(est * 100.0, 2),
            "se_pct": round(se * 100.0, 2),
            "ci95": [round(lo * 100.0, 2), round(hi * 100.0, 2)],
        }

    h_L = [int(all(bin_map[r][idx] == 1 for r in raters)) for idx in range(len(ids))]
    h_U = [int(any(bin_map[r][idx] == 1 for r in raters)) for idx in range(len(ids))]
    e_L = [int(s_key[i]["e_L"]) for i in ids]
    e_U = [int(s_key[i]["e_U"]) for i in ids]
    j_pri = [int(s_key[i]["pri"] == "Commit") for i in ids]
    j_sec = [int(s_key[i]["sec"] == "Commit") for i in ids]

    compat = [int(h_L[k] <= e_U[k] and e_L[k] <= h_U[k]) for k in range(len(ids))]
    compat_unw = sum(compat) / len(ids)
    compat_hajek = hajek_linearized_ci(compat)

    per_rater_vs_judges = {}
    for r in raters:
        per_rater_vs_judges[r] = {
            "vs_eL_design_hajek": pairwise_metrics(bin_map[r], e_L, [0, 1], hajek_w),
            "vs_eU_design_hajek": pairwise_metrics(bin_map[r], e_U, [0, 1], hajek_w),
            "vs_pri_design_hajek": pairwise_metrics(bin_map[r], j_pri, [0, 1], hajek_w),
            "vs_sec_design_hajek": pairwise_metrics(bin_map[r], j_sec, [0, 1], hajek_w),
            "design_hajek_commit_rate": hajek_linearized_ci(bin_map[r]),
        }

    both_agreed_idx = [k for k in range(len(ids)) if e_L[k] == e_U[k] and h_L[k] == h_U[k]]
    both_agreed_match = sum(h_L[k] == e_L[k] for k in both_agreed_idx)

    maj_scope = None
    h_maj = None
    if len(raters) >= 3 and len(raters) % 2 == 1:
        thr = len(raters) // 2 + 1
        h_maj = [int(sum(bin_map[r][k] for r in raters) >= thr) for k in range(len(ids))]
        maj_in_llm = [int(e_L[k] <= h_maj[k] <= e_U[k]) for k in range(len(ids))]
        maj_prev = hajek_linearized_ci(h_maj)
        maj_scope = {
            "design_hajek_commit_prevalence_pct": maj_prev["pct"],
            "design_hajek_commit_prevalence_se_pct": maj_prev["se_pct"],
            "design_hajek_commit_prevalence_ci95": maj_prev["ci95"],
            "vs_eL_design_hajek": pairwise_metrics(h_maj, e_L, [0, 1], hajek_w),
            "vs_eU_design_hajek": pairwise_metrics(h_maj, e_U, [0, 1], hajek_w),
            "vs_pri_design_hajek": pairwise_metrics(h_maj, j_pri, [0, 1], hajek_w),
            "vs_sec_design_hajek": pairwise_metrics(h_maj, j_sec, [0, 1], hajek_w),
            "within_llm_bounds_unweighted": round(sum(maj_in_llm) / len(ids), 4),
            "within_llm_bounds_design_hajek": round(w_mean(maj_in_llm, hajek_w), 4),
            "within_llm_bounds_design_hajek_ci95": hajek_linearized_ci(maj_in_llm)["ci95"],
        }

    hL_prev = hajek_linearized_ci(h_L)
    hU_prev = hajek_linearized_ci(h_U)

    # Sensitivity table comparing calibrated Hajek vs raw HT vs naive N_h/n_h
    weight_sensitivity = {}
    for lbl, wvec in [("design_calibrated_hajek", hajek_w), ("raw_horvitz_thompson", raw_ht), ("naive_Nh_over_nh", naive_w)]:
        entry = {
            "weight_sum": round(sum(wvec), 1),
            "interval_compatibility_pct": round(w_mean(compat, wvec) * 100.0, 2),
            "hL_vs_eL_agree_pct": round(pairwise_metrics(h_L, e_L, [0, 1], wvec)["agree"] * 100.0, 2),
            "human_L_pct": round(w_mean(h_L, wvec) * 100.0, 2),
            "human_U_pct": round(w_mean(h_U, wvec) * 100.0, 2),
            "llm_eL_pct": round(w_mean(e_L, wvec) * 100.0, 2),
            "llm_eU_pct": round(w_mean(e_U, wvec) * 100.0, 2),
        }
        if h_maj is not None:
            entry["human_majority_pct"] = round(w_mean(h_maj, wvec) * 100.0, 2)
        weight_sensitivity[lbl] = entry

    return {
        "n_turns": len(ids),
        "total_population_turns": int(round(N_tot)),
        "weighting_method": f"stratum_calibrated_hajek_from_balanced_pick_mc (M={mc_draws})",
        "design_inclusion_prob_diagnostics": stratum_diag,
        "inter_rater": {
            "binary_commit_unweighted": multi_rater_agreement(bin_map, [0, 1]),
            "binary_commit_design_hajek": multi_rater_agreement(bin_map, [0, 1], hajek_w),
            "four_class_unweighted": multi_rater_agreement(sc_map, ["Wait", "Conditional", "Commit", "Partial/Other"]),
            "four_class_design_hajek": multi_rater_agreement(sc_map, ["Wait", "Conditional", "Commit", "Partial/Other"], hajek_w),
        },
        "human_bounds_vs_llm_bounds": {
            "interval_compatibility_unweighted": round(compat_unw, 4),
            "interval_compatibility_design_hajek": round(compat_hajek["pct"] / 100.0, 4),
            "interval_compatibility_design_hajek_ci95_pct": compat_hajek["ci95"],
            "hL_vs_eL_design_hajek": pairwise_metrics(h_L, e_L, [0, 1], hajek_w),
            "hU_vs_eU_design_hajek": pairwise_metrics(h_U, e_U, [0, 1], hajek_w),
            "both_human_and_judge_agreed_turns": {
                "n": len(both_agreed_idx),
                "n_match": both_agreed_match,
                "agree_rate": round(both_agreed_match / len(both_agreed_idx), 4) if both_agreed_idx else None,
            },
            "design_hajek_commit_prevalence_pct": {
                "human_L": hL_prev["pct"],
                "human_L_se": hL_prev["se_pct"],
                "human_L_ci95": hL_prev["ci95"],
                "human_U": hU_prev["pct"],
                "human_U_se": hU_prev["se_pct"],
                "human_U_ci95": hU_prev["ci95"],
                "llm_eL_exact_population": round(w_mean(e_L, hajek_w) * 100.0, 2),
                "llm_eU_exact_population": round(w_mean(e_U, hajek_w) * 100.0, 2),
            },
        },
        "majority_vote": maj_scope,
        "weight_sensitivity_comparison": weight_sensitivity,
        "per_rater_vs_judges": per_rater_vs_judges,
    }


def score_repair(raters: list[str], key: dict, design_bound: dict) -> dict:
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

    # Primary: Monte Carlo design-calibrated adversarial 95% bound under balanced_pick (valid when neg_U == 0)
    assert neg_U == 0, "Design-calibrated 0-observed bound requires neg_U == 0; extend simulator for neg_U > 0."
    tot_lo_1s = (pos_L + neg_L) / N_tot
    tot_hi_1s = (pos_U + design_bound["one_sided_95_upper_K"]) / N_tot
    tot_lo_2s = (pos_L + neg_L) / N_tot
    tot_hi_2s = (pos_U + design_bound["two_sided_95_upper_K"]) / N_tot

    # Diagnostic only: naive SRSWOR hypergeometric (NOT valid under balanced_pick)
    hg_diag = hypergeom_ci_srswor_diagnostic(N_pool, len(neg_ids), neg_U, alpha=0.05)

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
        "design_calibrated_mc_adversarial_nonempty_457": design_bound,
        "total_489_retraction_rate_pct": {
            "design_calibrated_95_one_sided_envelope": [round(tot_lo_1s * 100, 2), round(tot_hi_1s * 100, 2)],
            "design_calibrated_95_two_sided_envelope": [round(tot_lo_2s * 100, 2), round(tot_hi_2s * 100, 2)],
        },
        "srswor_reference_invalid_for_balanced_pick": {
            "note": "Documented diagnostic only: assumes SRSWOR rather than balanced_pick; actual P_design(0 obs | K=16) is ~0.226, not 0.05.",
            "nonempty_457_hypergeom": hg_diag,
            "total_489_one_sided_if_srswor_pct": [round(pos_L / N_tot * 100, 2), round((pos_U + hg_diag["one_sided_95_upper_K"]) / N_tot * 100, 2)],
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
        "scope_120_commit_design_hajek": split_pairs(out["scope_120"]["inter_rater"]["binary_commit_design_hajek"]["pairwise"]),
        "scope_120_four_class_unweighted": split_pairs(out["scope_120"]["inter_rater"]["four_class_unweighted"]["pairwise"]),
        "repair_100_all100": split_pairs(out["repair_100"]["inter_rater_all100"]["pairwise"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raters", nargs="+", required=True, help="List of rater slugs (M >= 2), e.g. --raters cai tang yu")
    ap.add_argument("--external", nargs="*", default=[], help="Optional subset of raters who are external (non-author)")
    ap.add_argument("--mc-draws", type=int, default=50000, help="Monte Carlo replays of balanced_pick for design-based inference")
    args = ap.parse_args()
    assert len(args.raters) >= 2, "Need at least 2 independent raters for [L, U] bounds"

    key = json.loads((HERE / "_unified_audit_key_DO_NOT_OPEN.json").read_text(encoding="utf-8"))
    T = pd.read_csv(TRAJ_CSV)
    T_dyn = T[T["protocol"].isin(DYN_PROTOS)]

    scope_pools, repair_neg_by_model = load_populations()
    pi_scope, scope_diag = simulate_scope_inclusion_probs(scope_pools, mc_draws=args.mc_draws, seed=20260501)
    repair_design_bound = simulate_repair_design_bound(repair_neg_by_model, mc_draws=args.mc_draws, seed=20260505)

    out = {
        "seed": key["seed"],
        "protocol_version": key["protocol_version"] + "_design_based_inference",
        "raters": args.raters,
        "external_raters": args.external,
        "mc_design_draws": args.mc_draws,
        "cert_100": score_cert(args.raters, key, T_dyn),
        "scope_120": score_scope(args.raters, key, pi_scope, scope_diag, args.mc_draws),
        "repair_100": score_repair(args.raters, key, repair_design_bound),
    }
    ext_cmp = summarize_ext_vs_author(out, args.raters, args.external)
    if ext_cmp is not None:
        out["external_vs_author_audit"] = ext_cmp

    out_path = HERE / "unified_audit_results_seed2026.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")
    print(f"Saved {out_path}")
    print(json.dumps({
        "scope_120_design_hajek": out["scope_120"]["human_bounds_vs_llm_bounds"],
        "scope_120_majority": out["scope_120"]["majority_vote"],
        "scope_120_inter_rater_design_hajek": {
            "binary_agree": out["scope_120"]["inter_rater"]["binary_commit_design_hajek"]["mean_pairwise_agree"],
            "binary_ac1": out["scope_120"]["inter_rater"]["binary_commit_design_hajek"]["gwet_ac1"],
            "four_class_agree": out["scope_120"]["inter_rater"]["four_class_design_hajek"]["mean_pairwise_agree"],
            "four_class_kappa": out["scope_120"]["inter_rater"]["four_class_design_hajek"]["fleiss_kappa"],
        },
        "scope_120_weight_sensitivity": out["scope_120"]["weight_sensitivity_comparison"],
        "repair_100_design_bound": out["repair_100"]["design_calibrated_mc_adversarial_nonempty_457"],
        "repair_100_total_489_pct": out["repair_100"]["total_489_retraction_rate_pct"],
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
