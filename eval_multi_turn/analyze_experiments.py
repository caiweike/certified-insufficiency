#!/usr/bin/env python3
"""Statistical analyzer & table generator for Experiments 2, 3, and 4.

Implements strict contract math:
- Problem-level aggregation across order+sampling runs (NEVER treating reps as independent items).
- Structural zeros E=0, J=Y for FULL and CONCAT.
- Exact bound propagation: E_L=max e_{t,L}, E_U=max e_{t,U}, J_L=Y_L(1-E_U), J_U=Y_U(1-E_L).
- RQ1 decomposition Y - J = YE + P(E=1 | Y=1).
- Contrast intervals [A_L - B_U, A_U - B_L] + 10,000 paired problem-bootstrap 95% CIs (seed=20261024),
  strictly separating annotation-uncertainty envelopes from sampling CIs.
- Experiment 3: 5-class reply distributions, explicit Repair Rate, TypeA/B & k=3..6 breakdowns,
  and per-turn hazard rate h(t) for first Commit before turn t.
- Experiment 4A/4B: Dual-Judge Cohen's kappa + DYN_REQUEST_GUARD full tradeoff vector
  (E, J, Y, t=k abstention rate, EMPTY_TRUNCATED rate, token usage).
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from protocol_utils import NONFINAL_SCOPES, PROTOCOLS_ALL  # noqa: E402


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def paired_problem_bootstrap(
    vals_a_l: list[float],
    vals_a_u: list[float],
    vals_b_l: list[float],
    vals_b_u: list[float],
    n_boot: int = 10000,
    seed: int = 20261024,
) -> dict[str, Any]:
    """Compute paired problem-bootstrap 95% envelope on A - B with opposite-endpoint bounds."""
    n = len(vals_a_l)
    if n == 0:
        return {"point_L": 0.0, "point_U": 0.0, "env_lo": 0.0, "env_hi": 0.0}
    diff_l = [al - bu for al, bu in zip(vals_a_l, vals_b_u)]
    diff_u = [au - bl for au, bl in zip(vals_a_u, vals_b_l)]
    rng = random.Random(seed)
    boot_l: list[float] = []
    boot_u: list[float] = []
    for _ in range(n_boot):
        idxs = [rng.randrange(n) for _ in range(n)]
        boot_l.append(sum(diff_l[i] for i in idxs) / n)
        boot_u.append(sum(diff_u[i] for i in idxs) / n)
    boot_l.sort()
    boot_u.sort()
    lo_idx = int(0.025 * n_boot)
    hi_idx = min(n_boot - 1, int(0.975 * n_boot))
    return {
        "point_L": round(sum(diff_l) / n, 4),
        "point_U": round(sum(diff_u) / n, 4),
        "env_lo": round(boot_l[lo_idx], 4),
        "env_hi": round(boot_u[hi_idx], 4),
        "n_problems": n,
    }


def cohen_kappa(pairs: list[tuple[str, str]]) -> float | None:
    if not pairs:
        return None
    n = len(pairs)
    labels = sorted({a for a, _ in pairs} | {b for _, b in pairs})
    p0 = sum(1 for a, b in pairs if a == b) / n
    ca = Counter(a for a, _ in pairs)
    cb = Counter(b for _, b in pairs)
    pe = sum((ca[l] / n) * (cb[l] / n) for l in labels)
    if abs(1.0 - pe) < 1e-12:
        return None
    return round((p0 - pe) / (1.0 - pe), 4)


def _trajectory_ye_bounds(tr: dict[str, Any]) -> tuple[float, float]:
    fg = tr["final_grading"]
    yl, yu = fg["y_L"], fg["y_U"]
    el, eu = tr["e_L"], tr["e_U"]
    return float(yl * el), float(yu * eu)


def _minimax_binary_ratio(options: list[tuple[bool, bool, bool]]) -> list[float | None]:
    """Compute exact [min, max] of sum(num)/sum(den) over binary item states (c00, c01, c11), requiring sum(den) > 0."""
    if not any(c01 or c11 for _, c01, c11 in options):
        return [None, None]
    min_num = sum(1 for c00, c01, c11 in options if not c00 and not c01 and c11)
    min_den = min_num + sum(1 for _, c01, _ in options if c01)
    min_ratio = 1.0 if min_den == 0 else min_num / min_den

    max_c11 = sum(1 for _, _, c11 in options if c11)
    max_c01_forced = sum(1 for c00, c01, c11 in options if not c00 and c01 and not c11)
    max_den = max_c11 + max_c01_forced
    max_ratio = 0.0 if max_den == 0 else max_c11 / max_den
    return [round(min_ratio, 4), round(max_ratio, 4)]


def _repair_bounds(rec: dict[str, Any]) -> tuple[int, int]:
    rep = rec.get("repair") or {}
    if "retracted_L" in rep and "retracted_U" in rep:
        return int(rep["retracted_L"]), int(rep["retracted_U"])
    if isinstance(rep.get("retracted"), bool):
        val = 1 if rep["retracted"] else 0
        return val, val
    return (0, 1) if rec.get("e_U", 0) == 1 else (0, 0)


def _build_confusion_matrix(pairs: list[tuple[str, str]], labels: tuple[str, ...]) -> dict[str, dict[str, int]]:
    cm: dict[str, dict[str, int]] = {r: {c: 0 for c in labels} for r in labels}
    for a, b in pairs:
        if a in cm and b in cm[a]:
            cm[a][b] += 1
    return cm


def _final_abstain_bounds(tr: dict[str, Any]) -> tuple[int, int]:
    fg = tr.get("final_grading") or {}
    if "explicit_abstain_L" in fg and "explicit_abstain_U" in fg:
        return int(fg["explicit_abstain_L"]), int(fg["explicit_abstain_U"])
    if (fg.get("sympy") or {}).get("status") == "INCORRECT_ABSTAIN":
        return 1, 1
    return 0, 0


def _turn_trunc_flags(t: dict[str, Any]) -> tuple[bool, bool, bool]:
    if "empty_trunc" in t or "vis_trunc" in t or "non_trunc_empty" in t:
        return bool(t.get("empty_trunc")), bool(t.get("vis_trunc")), bool(t.get("non_trunc_empty"))
    return bool(t.get("empty_truncated")), False, False


def _traj_has_any_trunc(tr: dict[str, Any]) -> bool:
    for t in tr.get("turns", []):
        et, vt, _ = _turn_trunc_flags(t)
        if et or vt:
            return True
    fg = tr.get("final_grading") or {}
    if "empty_trunc" in fg or "vis_trunc" in fg:
        return bool(fg.get("empty_trunc") or fg.get("vis_trunc"))
    return bool(tr.get("any_empty_truncated"))


def _compute_protocol_stats(
    by_proto: dict[str, list[dict[str, Any]]],
    include_full_details: bool = True,
) -> tuple[dict[str, Any], dict[str, dict[str, dict[str, float]]]]:
    proto_summary: dict[str, Any] = {}
    prob_metrics: dict[str, dict[str, dict[str, float]]] = {}

    for proto, plist in sorted(by_proto.items()):
        by_item: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for tr in plist:
            by_item[tr["item_id"]].append(tr)
        prob_metrics[proto] = {}
        item_ids = sorted(by_item)
        for iid in item_ids:
            runs = by_item[iid]
            m = len(runs)
            y_l = sum(r["final_grading"]["y_L"] for r in runs) / m
            y_u = sum(r["final_grading"]["y_U"] for r in runs) / m
            e_l = sum(r["e_L"] for r in runs) / m
            e_u = sum(r["e_U"] for r in runs) / m
            j_l = sum(r["j_L"] for r in runs) / m
            j_u = sum(r["j_U"] for r in runs) / m
            ye_l = sum(_trajectory_ye_bounds(r)[0] for r in runs) / m
            ye_u = sum(_trajectory_ye_bounds(r)[1] for r in runs) / m
            trunc_any = sum(1 for r in runs if _traj_has_any_trunc(r)) / m
            prob_metrics[proto][iid] = {
                "y_L": y_l, "y_U": y_u, "e_L": e_l, "e_U": e_u,
                "j_L": j_l, "j_U": j_u, "ye_L": ye_l, "ye_U": ye_u,
                "trunc_any": trunc_any,
            }

        def mean_key(k: str) -> float:
            return round(sum(prob_metrics[proto][i][k] for i in item_ids) / len(item_ids), 4) if item_ids else 0.0

        rec_out: dict[str, Any] = {
            "n_problems": len(item_ids),
            "n_trajectories": len(plist),
            "structural_zero_E": all(r.get("structural_zero_E") for r in plist),
            "Y": [mean_key("y_L"), mean_key("y_U")],
            "E": [mean_key("e_L"), mean_key("e_U")],
            "J": [mean_key("j_L"), mean_key("j_U")],
        }
        if include_full_details:
            p_e1_given_y1 = _minimax_binary_ratio(
                [
                    (
                        r["final_grading"]["y_L"] == 0,
                        r["final_grading"]["y_U"] == 1 and r["e_L"] == 0,
                        r["final_grading"]["y_U"] == 1 and r["e_U"] == 1,
                    )
                    for r in plist
                ]
            )
            n_traj = len(plist)
            f_empty_trunc = sum(
                1 for r in plist
                if r["final_grading"].get("empty_trunc", bool((r["turns"][-1] if r["turns"] else {}).get("empty_truncated")))
            )
            f_vis_trunc = sum(1 for r in plist if r["final_grading"].get("vis_trunc"))
            f_non_trunc_empty = sum(1 for r in plist if r["final_grading"].get("non_trunc_empty"))
            f_abs_l = sum(_final_abstain_bounds(r)[0] for r in plist)
            f_abs_u = sum(_final_abstain_bounds(r)[1] for r in plist)

            all_turns = [t for r in plist for t in r["turns"]]
            n_turns = len(all_turns)
            t_empty_trunc = sum(1 for t in all_turns if _turn_trunc_flags(t)[0])
            t_vis_trunc = sum(1 for t in all_turns if _turn_trunc_flags(t)[1])
            t_non_trunc_empty = sum(1 for t in all_turns if _turn_trunc_flags(t)[2])
            tot_toks = sum(r.get("total_prompt_tokens", 0) + r.get("total_completion_tokens", 0) for r in plist)

            rec_out.update(
                {
                    "Y_minus_J_eq_YE": [mean_key("ye_L"), mean_key("ye_U")],
                    "P_E1_given_Y1": p_e1_given_y1,
                    "final_empty_trunc_rate": round(f_empty_trunc / n_traj, 4) if n_traj else None,
                    "final_vis_trunc_rate": round(f_vis_trunc / n_traj, 4) if n_traj else None,
                    "final_non_trunc_empty_rate": round(f_non_trunc_empty / n_traj, 4) if n_traj else None,
                    "final_explicit_abstain_rate": [round(f_abs_l / n_traj, 4), round(f_abs_u / n_traj, 4)] if n_traj else None,
                    "empty_truncated_turn_rate": round(t_empty_trunc / n_turns, 4) if n_turns else None,
                    "vis_truncated_turn_rate": round(t_vis_trunc / n_turns, 4) if n_turns else None,
                    "non_trunc_empty_turn_rate": round(t_non_trunc_empty / n_turns, 4) if n_turns else None,
                    "mean_tokens_per_trajectory": round(tot_toks / n_traj, 1) if n_traj else None,
                }
            )
        proto_summary[proto] = rec_out

    return proto_summary, prob_metrics


def analyze_model(rows: list[dict[str, Any]], n_boot: int = 10000, seed: int = 20261024) -> dict[str, Any]:
    by_proto: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        by_proto[r["protocol"]].append(r)

    # 1. Experiment 2: per-protocol summary + truncation-excluded sensitivity
    proto_summary, prob_metrics = _compute_protocol_stats(by_proto, include_full_details=True)

    clean_by_proto: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        if not _traj_has_any_trunc(r):
            clean_by_proto[r["protocol"]].append(r)
    no_trunc_summary, _ = _compute_protocol_stats(clean_by_proto, include_full_details=False)

    # Paired contrasts (pre-specified contrast pairs; does not assume additivity across factors)
    contrast_specs = [
        ("FULL_minus_CONCAT_Y", "FULL", "CONCAT", "y", True),
        ("CONCAT_minus_CONCAT_MATCHED_Y", "CONCAT", "CONCAT_MATCHED", "y", True),
        ("CONCAT_MATCHED_minus_DYN_FREE_Y", "CONCAT_MATCHED", "DYN_FREE", "y", True),
        ("CONCAT_minus_DYN_FREE_Y", "CONCAT", "DYN_FREE", "y", False),
        ("DYN_REQUEST_minus_DYN_FREE_Y", "DYN_REQUEST", "DYN_FREE", "y", False),
        ("DYN_REQUEST_minus_DYN_FREE_E", "DYN_REQUEST", "DYN_FREE", "e", True),
        ("DYN_REQUEST_GUARD_minus_DYN_REQUEST_E", "DYN_REQUEST_GUARD", "DYN_REQUEST", "e", True),
        ("DYN_REQUEST_GUARD_minus_DYN_REQUEST_J", "DYN_REQUEST_GUARD", "DYN_REQUEST", "j", True),
        ("DYN_REQUEST_GUARD_minus_DYN_REQUEST_Y", "DYN_REQUEST_GUARD", "DYN_REQUEST", "y", False),
    ]
    contrasts: dict[str, Any] = {}
    for cname, pa, pb, mkey, is_primary in contrast_specs:
        if pa not in prob_metrics or pb not in prob_metrics:
            continue
        common = sorted(set(prob_metrics[pa]) & set(prob_metrics[pb]))
        if not common:
            continue
        res = paired_problem_bootstrap(
            [prob_metrics[pa][i][f"{mkey}_L"] for i in common],
            [prob_metrics[pa][i][f"{mkey}_U"] for i in common],
            [prob_metrics[pb][i][f"{mkey}_L"] for i in common],
            [prob_metrics[pb][i][f"{mkey}_U"] for i in common],
            n_boot=n_boot,
            seed=seed,
        )
        res["is_primary"] = is_primary
        contrasts[cname] = res

    # Stratified rep>=1 paired comparison: canonical CONCAT vs paired non-canonical CONCAT_MATCHED
    perm_c = defaultdict(list)
    perm_cm = defaultdict(list)
    for r in rows:
        if r.get("rep", 0) >= 1 or r.get("order_kind") == "paired_permuted":
            if r["protocol"] == "CONCAT":
                perm_c[r["item_id"]].append(r)
            elif r["protocol"] == "CONCAT_MATCHED":
                perm_cm[r["item_id"]].append(r)
    common_perm = sorted(set(perm_c) & set(perm_cm))
    if common_perm:
        res_perm = paired_problem_bootstrap(
            [sum(x["final_grading"]["y_L"] for x in perm_c[i]) / len(perm_c[i]) for i in common_perm],
            [sum(x["final_grading"]["y_U"] for x in perm_c[i]) / len(perm_c[i]) for i in common_perm],
            [sum(x["final_grading"]["y_L"] for x in perm_cm[i]) / len(perm_cm[i]) for i in common_perm],
            [sum(x["final_grading"]["y_U"] for x in perm_cm[i]) / len(perm_cm[i]) for i in common_perm],
            n_boot=n_boot,
            seed=seed,
        )
        res_perm["is_primary"] = True
        contrasts["CONCAT_minus_CONCAT_MATCHED_Y_permuted_only"] = res_perm

    # Order breakdown (canonical rep=0 vs paired_permuted rep>=1)
    order_breakdown: dict[str, Any] = {}
    for okind in ("canonical", "paired_permuted"):
        sub = [r for r in rows if r.get("order_kind") == okind]
        if not sub:
            continue
        order_breakdown[okind] = {}
        for proto in sorted({r["protocol"] for r in sub}):
            psub = [r for r in sub if r["protocol"] == proto]
            m = len(psub)
            order_breakdown[okind][proto] = {
                "n_trajectories": m,
                "Y": [round(sum(r["final_grading"]["y_L"] for r in psub) / m, 4), round(sum(r["final_grading"]["y_U"] for r in psub) / m, 4)],
                "E": [round(sum(r["e_L"] for r in psub) / m, 4), round(sum(r["e_U"] for r in psub) / m, 4)],
                "J": [round(sum(r["j_L"] for r in psub) / m, 4), round(sum(r["j_U"] for r in psub) / m, 4)],
            }

    # 2. Experiment 3: nonfinal reply distributions, explicit Repair, TypeA/B & k=3..6 + hazard rate h(t)
    scope_dists: dict[str, Any] = {}
    repair_summary: dict[str, Any] = {}
    hazard_by_proto: dict[str, Any] = {}
    mech_k_breakdown: dict[str, Any] = defaultdict(dict)

    for proto in ("DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"):
        plist = by_proto.get(proto, [])
        if not plist:
            continue
        nf_counts = Counter()
        nf_total = 0
        for r in plist:
            for t in r["turns"]:
                if not t["is_final_turn"]:
                    nf_total += 1
                    nf_counts[t.get("scope_C", "Unresolved")] += 1
        scope_dists[proto] = {
            "nonfinal_turns": nf_total,
            "rates": {
                k: round(nf_counts[k] / nf_total, 4) if nf_total else None
                for k in (*NONFINAL_SCOPES, "NonCommit_Disagreed", "EMPTY_TRUNCATED", "NON_TRUNC_EMPTY")
            },
            "counts": dict(nf_counts),
        }

        e_pos_u = [r for r in plist if r["e_U"] == 1]
        e_pos_l = [r for r in plist if r["e_L"] == 1]
        if plist and all(r.get("repair_skipped") for r in plist):
            repair_summary[proto] = {
                "skipped": True,
                "event_positive_L": len(e_pos_l),
                "event_positive_U": len(e_pos_u),
            }
        else:
            rep_counts = Counter((r.get("repair") or {}).get("repair_type", "none") for r in e_pos_u)
            rep_bounds_by_id = {id(r): _repair_bounds(r) for r in plist}
            explicit_repair_rate = _minimax_binary_ratio(
                [
                    (
                        r["e_L"] == 0,
                        r["e_U"] == 1 and rep_bounds_by_id[id(r)][0] == 0,
                        r["e_U"] == 1 and rep_bounds_by_id[id(r)][1] == 1,
                    )
                    for r in plist
                ]
            )
            confirmed_repair_rate = _minimax_binary_ratio(
                [
                    (
                        False,
                        rep_bounds_by_id[id(r)][0] == 0,
                        rep_bounds_by_id[id(r)][1] == 1,
                    )
                    for r in e_pos_l
                ]
            )
            repair_given_y1_rate = _minimax_binary_ratio(
                [
                    (
                        r["e_L"] == 0 or r["final_grading"]["y_L"] == 0,
                        r["e_U"] == 1 and r["final_grading"]["y_U"] == 1 and rep_bounds_by_id[id(r)][0] == 0,
                        r["e_U"] == 1 and r["final_grading"]["y_U"] == 1 and rep_bounds_by_id[id(r)][1] == 1,
                    )
                    for r in plist
                ]
            )
            repair_summary[proto] = {
                "skipped": False,
                "event_positive_L": len(e_pos_l),
                "event_positive_U": len(e_pos_u),
                "explicit_retracted_range": [
                    sum(rep_bounds_by_id[id(r)][0] for r in e_pos_l),
                    sum(rep_bounds_by_id[id(r)][1] for r in e_pos_u),
                ],
                "confirmed_repair_rate": confirmed_repair_rate,
                "explicit_repair_rate": explicit_repair_rate,
                "repair_given_Y1_rate": repair_given_y1_rate,
                "repair_types": dict(rep_counts),
            }

        # First-commit turn & discrete conditional hazard h(t) = P(first Commit at t | no Commit before t)
        max_nf = max((r["canonical_k"] - 1 for r in plist), default=0)
        hazards: dict[str, Any] = {}
        for t_idx in range(1, max_nf + 1):
            at_risk_l = 0
            at_risk_u = 0
            first_l = 0
            first_u = 0
            h_opts: list[tuple[bool, bool, bool]] = []
            for r in plist:
                nf_turns = [t for t in r["turns"] if not t["is_final_turn"]]
                if len(nf_turns) < t_idx:
                    continue
                prior_el = 1 if any(t.get("e_L", 0) == 1 for t in nf_turns[: t_idx - 1]) else 0
                prior_eu = 1 if any(t.get("e_U", 0) == 1 for t in nf_turns[: t_idx - 1]) else 0
                cur_el = int(nf_turns[t_idx - 1].get("e_L", 0))
                cur_eu = int(nf_turns[t_idx - 1].get("e_U", 0))
                if prior_eu == 0:
                    at_risk_l += 1
                    first_l += cur_el
                if prior_el == 0:
                    at_risk_u += 1
                    first_u += cur_eu
                h_opts.append(
                    (
                        prior_eu == 1,
                        prior_el == 0 and cur_el == 0,
                        prior_el == 0 and cur_eu == 1,
                    )
                )
            hazards[f"turn_{t_idx}"] = {
                "at_risk_range": [at_risk_l, at_risk_u],
                "first_commit_range": [first_l, first_u],
                "hazard_rate": _minimax_binary_ratio(h_opts),
            }
        hazard_by_proto[proto] = hazards

        # Mechanism & depth breakdown (note k=6 n=6 is descriptive exploratory)
        for mech in ("TypeA_UniqueValue", "TypeB_Extremum"):
            sub_m = [r for r in plist if r["verified_mechanism"] == mech]
            if sub_m:
                mech_k_breakdown[proto][mech] = {
                    "n_trajectories": len(sub_m),
                    "E": [round(sum(r["e_L"] for r in sub_m) / len(sub_m), 4), round(sum(r["e_U"] for r in sub_m) / len(sub_m), 4)],
                    "J": [round(sum(r["j_L"] for r in sub_m) / len(sub_m), 4), round(sum(r["j_U"] for r in sub_m) / len(sub_m), 4)],
                }
        for k_val in (3, 4, 5, 6):
            sub_k = [r for r in plist if r["canonical_k"] == k_val]
            if sub_k:
                mech_k_breakdown[proto][f"k={k_val}"] = {
                    "n_trajectories": len(sub_k),
                    "exploratory_only": k_val == 6,
                    "E": [round(sum(r["e_L"] for r in sub_k) / len(sub_k), 4), round(sum(r["e_U"] for r in sub_k) / len(sub_k), 4)],
                    "J": [round(sum(r["j_L"] for r in sub_k) / len(sub_k), 4), round(sum(r["j_U"] for r in sub_k) / len(sub_k), 4)],
                }

    # 3. Experiment 4A: Pure Dual-LLM confusion matrices, agreement/kappa, & failure counts
    versions = sorted({str(r.get("grading_rule_version") or r["final_grading"].get("grading_rule_version") or "unknown") for r in rows})
    rule_version: Any = versions[0] if len(versions) == 1 else versions
    final_sources = Counter(r["final_grading"].get("source", "unknown") for r in rows)

    scope_pairs = [
        (t["primary_scope"], t["secondary_scope"])
        for r in rows
        for t in r["turns"]
        if not t["is_final_turn"]
        and not t.get("primary_failed")
        and not t.get("secondary_failed")
        and t.get("primary_scope") in NONFINAL_SCOPES
        and t.get("secondary_scope") in NONFINAL_SCOPES
    ]
    commit_pairs = [
        ("Commit" if a == "Commit" else "NonCommit", "Commit" if b == "Commit" else "NonCommit")
        for a, b in scope_pairs
        if a != "Unresolved" and b != "Unresolved"
    ]
    final_pairs: list[tuple[str, str]] = []
    for r in rows:
        fg = r["final_grading"]
        if fg.get("primary_failed") or fg.get("secondary_failed"):
            continue
        po = fg.get("primary_llm")
        so = fg.get("secondary_llm")
        if (
            isinstance(po, dict)
            and isinstance(so, dict)
            and isinstance(po.get("correct"), bool)
            and isinstance(so.get("correct"), bool)
            and not po.get("uncertain")
            and not so.get("uncertain")
        ):
            final_pairs.append(("True" if po["correct"] else "False", "True" if so["correct"] else "False"))

    repair_pairs: list[tuple[str, str]] = []
    repair_type_pairs: list[tuple[str, str]] = []
    repair_types_labels = ("none", "retracted_wrong_commit", "qualified_earlier_certainty")
    for r in rows:
        rep = r.get("repair")
        if not isinstance(rep, dict):
            continue
        if rep.get("primary_failed") or rep.get("secondary_failed"):
            continue
        pr = rep.get("primary_retracted")
        sr = rep.get("secondary_retracted")
        if isinstance(pr, bool) and isinstance(sr, bool):
            repair_pairs.append(("True" if pr else "False", "True" if sr else "False"))
        pt = rep.get("primary_repair_type")
        st = rep.get("secondary_repair_type")
        if pt in repair_types_labels and st in repair_types_labels:
            repair_type_pairs.append((str(pt), str(st)))

    judge_failures = {
        "final_primary_fail": sum(1 for r in rows if r["final_grading"].get("primary_failed")),
        "final_secondary_fail": sum(1 for r in rows if r["final_grading"].get("secondary_failed")),
        "scope_primary_fail": sum(1 for r in rows for t in r["turns"] if not t["is_final_turn"] and t.get("primary_failed")),
        "scope_secondary_fail": sum(1 for r in rows for t in r["turns"] if not t["is_final_turn"] and t.get("secondary_failed")),
        "repair_primary_fail": sum(
            1 for r in rows
            if isinstance(r.get("repair"), dict)
            and (r["repair"].get("primary_failed") or (r["repair"].get("repair_status") == "PARSE_FAIL" and not r["repair"].get("secondary_failed")))
        ),
        "repair_secondary_fail": sum(
            1 for r in rows
            if isinstance(r.get("repair"), dict)
            and r["repair"].get("secondary_failed")
        ),
    }

    exp4a_audit = {
        "grading_rule_version": rule_version,
        "final_source_counts": dict(final_sources),
        "judge_failures": judge_failures,
        "dual_scope_pairs": len(scope_pairs),
        "dual_scope_agreement": round(sum(1 for a, b in scope_pairs if a == b) / len(scope_pairs), 4) if scope_pairs else None,
        "dual_scope_category_kappa": cohen_kappa(scope_pairs),
        "dual_scope_kappa": cohen_kappa(scope_pairs),
        "scope_confusion_matrix": _build_confusion_matrix(scope_pairs, NONFINAL_SCOPES),
        "dual_commit_pairs": len(commit_pairs),
        "dual_commit_agreement": round(sum(1 for a, b in commit_pairs if a == b) / len(commit_pairs), 4) if commit_pairs else None,
        "dual_commit_kappa": cohen_kappa(commit_pairs),
        "commit_confusion_matrix": _build_confusion_matrix(commit_pairs, ("Commit", "NonCommit")),
        "dual_final_pairs": len(final_pairs),
        "dual_final_agreement": round(sum(1 for a, b in final_pairs if a == b) / len(final_pairs), 4) if final_pairs else None,
        "dual_final_kappa": cohen_kappa(final_pairs),
        "final_confusion_matrix": _build_confusion_matrix(final_pairs, ("True", "False")),
        "dual_repair_pairs": len(repair_pairs),
        "dual_repair_agreement": round(sum(1 for a, b in repair_pairs if a == b) / len(repair_pairs), 4) if repair_pairs else None,
        "dual_repair_kappa": cohen_kappa(repair_pairs),
        "repair_confusion_matrix": _build_confusion_matrix(repair_pairs, ("True", "False")),
        "dual_repair_type_agreement": round(sum(1 for a, b in repair_type_pairs if a == b) / len(repair_type_pairs), 4) if repair_type_pairs else None,
        "dual_repair_type_kappa": cohen_kappa(repair_type_pairs),
        "repair_type_confusion_matrix": _build_confusion_matrix(repair_type_pairs, repair_types_labels),
    }

    return {
        "grading_rule_version": rule_version,
        "exp2_protocol_summary": proto_summary,
        "exp2_no_trunc_sensitivity": no_trunc_summary,
        "exp2_contrasts": contrasts,
        "exp2_order_breakdown": order_breakdown,
        "exp3_nonfinal_scope": scope_dists,
        "exp3_explicit_repair": repair_summary,
        "exp3_commit_hazard": hazard_by_proto,
        "exp3_mechanism_and_depth": dict(mech_k_breakdown),
        "exp4a_judge_calibration": exp4a_audit,
    }


def _run_mock_tests() -> None:
    # Build synthetic 10-problem x 5-protocol x 2-rep judged dataset verifying invariants end-to-end.
    rows: list[dict[str, Any]] = []
    for i in range(10):
        iid = f"it-{i}"
        k = 3 if i < 6 else 4
        mech = "TypeA_UniqueValue" if i < 7 else "TypeB_Extremum"
        for rep in range(2):
            for proto in PROTOCOLS_ALL:
                is_single = proto in ("FULL", "CONCAT", "CONCAT_MATCHED")
                y_val = 1 if (proto != "DYN_FREE" or i < 8) else 0
                if is_single:
                    e_l, e_u = 0, 0
                    turns = [{"turn_idx": 1, "is_final_turn": True, "empty_truncated": False}]
                elif proto == "DYN_REQUEST":
                    c1 = "Commit" if i < 3 else ("Unresolved" if i == 9 else "Conditional")
                    el1, eu1 = (1, 1) if c1 == "Commit" else ((0, 1) if c1 == "Unresolved" else (0, 0))
                    turns = [
                        {"turn_idx": 1, "is_final_turn": False, "scope_C": c1, "primary_scope": c1, "secondary_scope": "Commit" if c1 == "Unresolved" else c1, "e_L": el1, "e_U": eu1, "empty_truncated": False},
                        {"turn_idx": 2, "is_final_turn": False, "scope_C": "Partial/Other", "primary_scope": "Partial/Other", "secondary_scope": "Partial/Other", "e_L": 0, "e_U": 0, "empty_truncated": False},
                        {"turn_idx": 3, "is_final_turn": True, "empty_truncated": False},
                    ]
                    e_l, e_u = el1, eu1
                elif proto == "DYN_REQUEST_GUARD":
                    turns = [
                        {"turn_idx": 1, "is_final_turn": False, "scope_C": "Conditional", "primary_scope": "Conditional", "secondary_scope": "Conditional", "e_L": 0, "e_U": 0, "empty_truncated": False},
                        {"turn_idx": 2, "is_final_turn": False, "scope_C": "NonCommit_Disagreed", "primary_scope": "Wait", "secondary_scope": "Partial/Other", "e_L": 0, "e_U": 0, "empty_truncated": False},
                        {"turn_idx": 3, "is_final_turn": True, "empty_truncated": False},
                    ]
                    e_l, e_u = 0, 0
                else:
                    turns = [
                        {"turn_idx": 1, "is_final_turn": False, "scope_C": "Wait", "primary_scope": "Wait", "secondary_scope": "Wait", "e_L": 0, "e_U": 0, "empty_truncated": False},
                        {"turn_idx": 2, "is_final_turn": False, "scope_C": "Wait", "primary_scope": "Wait", "secondary_scope": "Wait", "e_L": 0, "e_U": 0, "empty_truncated": False},
                        {"turn_idx": 3, "is_final_turn": True, "empty_truncated": False},
                    ]
                    e_l, e_u = 0, 0

                rows.append(
                    {
                        "trajectory_id": f"{iid}___{proto}___r{rep}",
                        "item_id": iid,
                        "verified_mechanism": mech,
                        "canonical_k": k,
                        "protocol": proto,
                        "rep": rep,
                        "order_kind": "canonical" if rep == 0 else "paired_permuted",
                        "structural_zero_E": is_single,
                        "e_L": e_l,
                        "e_U": e_u,
                        "j_L": y_val * (1 - e_u),
                        "j_U": y_val * (1 - e_l),
                        "final_grading": {
                            "sympy": {"status": "CORRECT" if y_val == 1 else "INCORRECT_EXACT"},
                            "y_L": y_val,
                            "y_U": y_val,
                        },
                        "turns": turns,
                        "repair": {"retracted": i == 0, "repair_type": "retracted_wrong_commit" if i == 0 else "none"} if e_u == 1 else None,
                        "total_prompt_tokens": 100,
                        "total_completion_tokens": 60,
                        "any_empty_truncated": False,
                    }
                )

    report = analyze_model(rows, n_boot=500, seed=20261024)
    dr = report["exp2_protocol_summary"]["DYN_REQUEST"]
    # Check RQ1 identity: Y - J == Y * E on bounds
    assert dr["Y"] == [1.0, 1.0]
    assert dr["E"] == [0.3, 0.4]
    assert dr["J"] == [0.6, 0.7]
    assert dr["Y_minus_J_eq_YE"] == [0.3, 0.4]
    assert report["exp2_protocol_summary"]["FULL"]["structural_zero_E"] is True
    assert report["exp2_protocol_summary"]["CONCAT_MATCHED"]["structural_zero_E"] is True
    assert "CONCAT_minus_CONCAT_MATCHED_Y_permuted_only" in report["exp2_contrasts"]
    assert "CONCAT_MATCHED_minus_DYN_FREE_Y" in report["exp2_contrasts"]
    assert report["exp2_protocol_summary"]["FULL"]["E"] == [0.0, 0.0]
    # Exact minimax bounds: 2 retracted out of 6 confirmed E_L=1 and up to 8 E_U=1 trajectories -> [0.25, 0.3333]
    assert report["exp3_explicit_repair"]["DYN_REQUEST"]["explicit_repair_rate"] == [0.25, 0.3333]
    assert report["exp3_explicit_repair"]["DYN_REQUEST"]["confirmed_repair_rate"] == [0.3333, 0.3333]
    # Zero denominator repair returns [None, None], not 0.0
    assert report["exp3_explicit_repair"]["DYN_FREE"]["explicit_repair_rate"] == [None, None]
    assert report["exp4a_judge_calibration"]["dual_scope_category_kappa"] is not None
    assert report["exp4a_judge_calibration"]["dual_commit_kappa"] == 1.0
    assert report["exp4a_judge_calibration"]["dual_scope_category_kappa"] < 1.0

    # Brute-force verify _minimax_binary_ratio on 200 random small configurations
    import itertools
    rng_check = random.Random(42)
    assert _minimax_binary_ratio([]) == [None, None]
    assert _minimax_binary_ratio([(True, False, False), (True, False, False)]) == [None, None]
    assert cohen_kappa([("Wait", "Wait")] * 5) is None
    subsets = [
        (True, False, False),
        (False, True, False),
        (False, False, True),
        (True, True, False),
        (True, False, True),
        (False, True, True),
        (True, True, True),
    ]
    for _ in range(200):
        vec = [rng_check.choice(subsets) for _ in range(rng_check.randint(1, 6))]
        item_choices = []
        for c00, c01, c11 in vec:
            ch = []
            if c00:
                ch.append((0, 0))
            if c01:
                ch.append((0, 1))
            if c11:
                ch.append((1, 1))
            item_choices.append(ch)
        ratios = []
        for combo in itertools.product(*item_choices):
            s_num = sum(n for n, _ in combo)
            s_den = sum(d for _, d in combo)
            if s_den > 0:
                ratios.append(s_num / s_den)
        expected = [round(min(ratios), 4), round(max(ratios), 4)] if ratios else [None, None]
        assert _minimax_binary_ratio(vec) == expected, f"Mismatch for {vec}: {_minimax_binary_ratio(vec)} vs {expected}"

    # Verify zero Y_U gives P(E=1|Y=1)==[None, None] and uncertain prior-turn hazard bounds
    edge_rows = [
        {
            "trajectory_id": "e1___DYN_FREE___r0",
            "item_id": "e1",
            "verified_mechanism": "TypeA_UniqueValue",
            "canonical_k": 3,
            "protocol": "DYN_FREE",
            "rep": 0,
            "order_kind": "canonical",
            "structural_zero_E": False,
            "e_L": 0,
            "e_U": 1,
            "j_L": 0,
            "j_U": 0,
            "final_grading": {"sympy": {"status": "INCORRECT_ABSTAIN"}, "y_L": 0, "y_U": 0},
            "turns": [
                {"turn_idx": 1, "is_final_turn": False, "scope_C": "Unresolved", "e_L": 0, "e_U": 1, "empty_truncated": False},
                {"turn_idx": 2, "is_final_turn": False, "scope_C": "Commit", "e_L": 1, "e_U": 1, "empty_truncated": False},
                {"turn_idx": 3, "is_final_turn": True, "empty_truncated": False},
            ],
            "repair": None,
        },
        {
            "trajectory_id": "e2___DYN_FREE___r0",
            "item_id": "e2",
            "verified_mechanism": "TypeA_UniqueValue",
            "canonical_k": 3,
            "protocol": "DYN_FREE",
            "rep": 0,
            "order_kind": "canonical",
            "structural_zero_E": False,
            "e_L": 0,
            "e_U": 0,
            "j_L": 0,
            "j_U": 0,
            "final_grading": {"sympy": {"status": "INCORRECT_EXACT"}, "y_L": 0, "y_U": 0},
            "turns": [
                {"turn_idx": 1, "is_final_turn": False, "scope_C": "Wait", "e_L": 0, "e_U": 0, "empty_truncated": False},
                {"turn_idx": 2, "is_final_turn": False, "scope_C": "Wait", "e_L": 0, "e_U": 0, "empty_truncated": False},
                {"turn_idx": 3, "is_final_turn": True, "empty_truncated": False},
            ],
            "repair": None,
        },
    ]
    edge_rep = analyze_model(edge_rows, n_boot=50, seed=1)
    assert edge_rep["exp2_protocol_summary"]["DYN_FREE"]["P_E1_given_Y1"] == [None, None]
    t2_haz = edge_rep["exp3_commit_hazard"]["DYN_FREE"]["turn_2"]
    assert t2_haz["at_risk_range"] == [1, 2]
    assert t2_haz["hazard_rate"] == [0.0, 0.5]

    # Pure Dual-LLM v1.0 mock dataset (zero SymPy, repair_skipped=True, 4-way trunc/abstain, 3 CMs, failure counts, no_trunc_sensitivity)
    v1_rows = [
        {
            "trajectory_id": "p1___FULL___r0",
            "item_id": "p1",
            "verified_mechanism": "TypeA_UniqueValue",
            "canonical_k": 3,
            "protocol": "FULL",
            "rep": 0,
            "order_kind": "canonical",
            "structural_zero_E": True,
            "e_L": 0,
            "e_U": 0,
            "j_L": 1,
            "j_U": 1,
            "grading_rule_version": "exp2-part2-dual-llm-v1.0",
            "repair_skipped": True,
            "final_grading": {
                "grading_rule_version": "exp2-part2-dual-llm-v1.0",
                "primary_llm": {"correct": True, "explicit_abstain": False, "uncertain": False},
                "secondary_llm": {"correct": True, "explicit_abstain": False, "uncertain": False},
                "y_L": 1,
                "y_U": 1,
                "explicit_abstain_L": 0,
                "explicit_abstain_U": 0,
                "empty_trunc": False,
                "vis_trunc": False,
                "non_trunc_empty": False,
                "primary_failed": False,
                "secondary_failed": False,
                "source": "llm_dual_agree:1",
            },
            "turns": [{"turn_idx": 1, "is_final_turn": True, "empty_trunc": False, "vis_trunc": False, "non_trunc_empty": False}],
            "repair": None,
            "total_prompt_tokens": 50,
            "total_completion_tokens": 20,
            "any_empty_truncated": False,
        },
        {
            "trajectory_id": "p2___FULL___r0",
            "item_id": "p2",
            "verified_mechanism": "TypeA_UniqueValue",
            "canonical_k": 3,
            "protocol": "FULL",
            "rep": 0,
            "order_kind": "canonical",
            "structural_zero_E": True,
            "e_L": 0,
            "e_U": 0,
            "j_L": 0,
            "j_U": 0,
            "grading_rule_version": "exp2-part2-dual-llm-v1.0",
            "repair_skipped": True,
            "final_grading": {
                "grading_rule_version": "exp2-part2-dual-llm-v1.0",
                "primary_llm": {"correct": False, "explicit_abstain": True, "uncertain": False},
                "secondary_llm": {"correct": False, "explicit_abstain": True, "uncertain": False},
                "y_L": 0,
                "y_U": 0,
                "explicit_abstain_L": 1,
                "explicit_abstain_U": 1,
                "empty_trunc": False,
                "vis_trunc": True,
                "non_trunc_empty": False,
                "primary_failed": False,
                "secondary_failed": False,
                "source": "llm_dual_agree:0",
            },
            "turns": [{"turn_idx": 1, "is_final_turn": True, "empty_trunc": False, "vis_trunc": True, "non_trunc_empty": False}],
            "repair": None,
            "total_prompt_tokens": 50,
            "total_completion_tokens": 16384,
            "any_empty_truncated": False,
        },
        {
            "trajectory_id": "p1___DYN_FREE___r0",
            "item_id": "p1",
            "verified_mechanism": "TypeA_UniqueValue",
            "canonical_k": 3,
            "protocol": "DYN_FREE",
            "rep": 0,
            "order_kind": "canonical",
            "structural_zero_E": False,
            "e_L": 1,
            "e_U": 1,
            "j_L": 0,
            "j_U": 0,
            "grading_rule_version": "exp2-part2-dual-llm-v1.0",
            "repair_skipped": True,
            "final_grading": {
                "grading_rule_version": "exp2-part2-dual-llm-v1.0",
                "primary_llm": {"correct": True, "explicit_abstain": False, "uncertain": False},
                "secondary_llm": None,
                "y_L": 0,
                "y_U": 1,
                "explicit_abstain_L": 0,
                "explicit_abstain_U": 1,
                "empty_trunc": False,
                "vis_trunc": False,
                "non_trunc_empty": False,
                "primary_failed": False,
                "secondary_failed": True,
                "source": "llm_secondary_failed",
            },
            "turns": [
                {"turn_idx": 1, "is_final_turn": False, "scope_C": "Wait", "primary_scope": "Wait", "secondary_scope": "Wait", "primary_failed": False, "secondary_failed": False, "e_L": 0, "e_U": 0, "empty_trunc": False, "vis_trunc": False, "non_trunc_empty": False},
                {"turn_idx": 2, "is_final_turn": False, "scope_C": "Commit", "primary_scope": "Commit", "secondary_scope": "Commit", "primary_failed": False, "secondary_failed": False, "e_L": 1, "e_U": 1, "empty_trunc": False, "vis_trunc": False, "non_trunc_empty": False},
                {"turn_idx": 3, "is_final_turn": True, "empty_trunc": False, "vis_trunc": False, "non_trunc_empty": False},
            ],
            "repair": None,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 60,
            "any_empty_truncated": False,
        },
        {
            "trajectory_id": "p2___DYN_FREE___r0",
            "item_id": "p2",
            "verified_mechanism": "TypeA_UniqueValue",
            "canonical_k": 3,
            "protocol": "DYN_FREE",
            "rep": 0,
            "order_kind": "canonical",
            "structural_zero_E": False,
            "e_L": 0,
            "e_U": 0,
            "j_L": 0,
            "j_U": 0,
            "grading_rule_version": "exp2-part2-dual-llm-v1.0",
            "repair_skipped": True,
            "final_grading": {
                "grading_rule_version": "exp2-part2-dual-llm-v1.0",
                "primary_llm": None,
                "secondary_llm": None,
                "y_L": 0,
                "y_U": 0,
                "explicit_abstain_L": 0,
                "explicit_abstain_U": 0,
                "empty_trunc": True,
                "vis_trunc": False,
                "non_trunc_empty": False,
                "primary_failed": False,
                "secondary_failed": False,
                "source": "empty_truncated",
            },
            "turns": [
                {"turn_idx": 1, "is_final_turn": False, "scope_C": "NON_TRUNC_EMPTY", "primary_failed": False, "secondary_failed": False, "e_L": 0, "e_U": 0, "empty_trunc": False, "vis_trunc": False, "non_trunc_empty": True},
                {"turn_idx": 2, "is_final_turn": False, "scope_C": "Unresolved", "primary_scope": "Unresolved", "secondary_scope": "Wait", "primary_failed": True, "secondary_failed": False, "e_L": 0, "e_U": 1, "empty_trunc": False, "vis_trunc": False, "non_trunc_empty": False},
                {"turn_idx": 3, "is_final_turn": True, "empty_trunc": True, "vis_trunc": False, "non_trunc_empty": False},
            ],
            "repair": None,
            "total_prompt_tokens": 100,
            "total_completion_tokens": 16384,
            "any_empty_truncated": True,
        },
    ]
    v1_rep = analyze_model(v1_rows, n_boot=50, seed=1)
    assert v1_rep["grading_rule_version"] == "exp2-part2-dual-llm-v1.0"
    assert v1_rep["exp3_explicit_repair"]["DYN_FREE"] == {"skipped": True, "event_positive_L": 1, "event_positive_U": 1}
    assert v1_rep["exp2_protocol_summary"]["FULL"]["final_vis_trunc_rate"] == 0.5
    assert v1_rep["exp2_protocol_summary"]["FULL"]["final_explicit_abstain_rate"] == [0.5, 0.5]
    assert v1_rep["exp2_no_trunc_sensitivity"]["FULL"]["n_trajectories"] == 1
    assert v1_rep["exp2_no_trunc_sensitivity"]["FULL"]["Y"] == [1.0, 1.0]
    assert v1_rep["exp3_nonfinal_scope"]["DYN_FREE"]["counts"]["NON_TRUNC_EMPTY"] == 1
    cal = v1_rep["exp4a_judge_calibration"]
    assert cal["judge_failures"] == {
        "final_primary_fail": 0,
        "final_secondary_fail": 1,
        "scope_primary_fail": 1,
        "scope_secondary_fail": 0,
        "repair_primary_fail": 0,
        "repair_secondary_fail": 0,
    }
    assert cal["final_confusion_matrix"] == {"True": {"True": 1, "False": 0}, "False": {"True": 0, "False": 1}}
    assert cal["commit_confusion_matrix"] == {"Commit": {"Commit": 1, "NonCommit": 0}, "NonCommit": {"Commit": 0, "NonCommit": 1}}
    assert cal["scope_confusion_matrix"]["Wait"]["Wait"] == 1 and cal["scope_confusion_matrix"]["Commit"]["Commit"] == 1
    print("analyze_experiments.py mock tests PASSED.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze judged trajectories for Experiments 2–4.")
    parser.add_argument("--inputs", nargs="+", type=Path, default=None, help="Judged JSONL files (one per model).")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bootstrap-samples", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=20261024)
    parser.add_argument("--mock-test", action="store_true")
    args = parser.parse_args()

    if args.mock_test:
        _run_mock_tests()
        return

    assert args.inputs and args.output, "--inputs and --output required"
    rows_by_model: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_tids_by_model: dict[str, set[str]] = defaultdict(set)
    for path in args.inputs:
        rows = read_jsonl(path)
        if not rows:
            continue
        for r in rows:
            mid = str(r.get("model_id", path.stem))
            tid = str(r.get("trajectory_id", ""))
            if tid:
                assert tid not in seen_tids_by_model[mid], f"Duplicate trajectory_id {tid} for model {mid}"
                seen_tids_by_model[mid].add(tid)
            rows_by_model[mid].append(r)

    models_out: dict[str, Any] = {}
    rule_versions: set[str] = set()
    for model_id, rows in rows_by_model.items():
        mrep = analyze_model(rows, n_boot=args.bootstrap_samples, seed=args.seed)
        rv = mrep.get("grading_rule_version")
        if isinstance(rv, str):
            rule_versions.add(rv)
        elif isinstance(rv, list):
            rule_versions.update(str(x) for x in rv)
        models_out[model_id] = mrep
    out_report: dict[str, Any] = {
        "grading_rule_version": sorted(rule_versions)[0] if len(rule_versions) == 1 else sorted(rule_versions),
        "models": models_out,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(out_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote analysis report -> {args.output}")


if __name__ == "__main__":
    main()
