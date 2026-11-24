"""Recompute the Traj-60 criterion sensitivity (both E and Y-J) and TypeA commit concentration.

Usage (from the repo root, the bundle root, or with a bundle ZIP):
  python recompute_traj60_sensitivity.py [PATH]
Writes traj60_sensitivity.json next to this script.
"""
import csv, io, json, math, sys
from collections import Counter
from pathlib import Path
from zipfile import ZipFile

TRAJ_KEY = "human_audit_followup_seed2027/_followup_audit_key_DO_NOT_OPEN.json"
TRAJ_LAB = "human_audit_followup_seed2027/traj_60_yu.csv"
TRAJ_CSV = "revision_analysis_v13/traj.csv"
STRATA = ("agree_Commit", "disagree_Commit", "agree_NonCommit")


class Src:
    def __init__(self, p):
        p = Path(p)
        self.zip = ZipFile(p) if p.is_file() else None
        self.root = p
        self.names = self.zip.namelist() if self.zip else [
            x.relative_to(p).as_posix() for x in p.rglob("*") if x.is_file()]

    def read(self, suffix):
        hit = [n for n in self.names if n.endswith(suffix)]
        if not hit:
            raise FileNotFoundError(suffix)
        n = sorted(hit, key=len)[0]
        return (self.zip.read(n) if self.zip else (self.root / n).read_bytes()).decode("utf-8-sig")


def stratified_binary(N, vals_by_stratum):
    """Exact finite-population stratified SRSWOR mean, SE, and 95% normal CI."""
    tot = sum(N[s] for s in STRATA)
    est = 0.0
    var_hat = 0.0
    hits = {}
    for s in STRATA:
        vals = vals_by_stratum[s]
        n_h = len(vals)
        N_h = N[s]
        W_h = N_h / tot
        p_h = sum(vals) / n_h
        s2_h = (n_h / (n_h - 1)) * p_h * (1.0 - p_h) if n_h > 1 else 0.0
        fpc = 1.0 - n_h / N_h
        est += W_h * p_h
        var_hat += (W_h ** 2) * fpc * (s2_h / n_h)
        hits[s] = int(sum(vals))
    se = math.sqrt(max(0.0, var_hat))
    ci = [max(0.0, est - 1.96 * se), min(1.0, est + 1.96 * se)]
    return {
        "estimate_pct": round(100.0 * est, 1),
        "estimate_exact_pct": round(100.0 * est, 2),
        "se_pp": round(100.0 * se, 2),
        "ci95_normal_pct": [round(100.0 * ci[0], 1), round(100.0 * ci[1], 1)],
        "stratum_hits": hits,
    }


def eval_scheme(N, trajs, flag_fn):
    e_vals = {s: [] for s in STRATA}
    yjL_vals = {s: [] for s in STRATA}
    yjU_vals = {s: [] for s in STRATA}
    for aid, t in sorted(trajs.items()):
        s = t["stratum"]
        e = int(flag_fn(aid, t))
        e_vals[s].append(e)
        yjL_vals[s].append(e * int(t["y_L"]))
        yjU_vals[s].append(e * int(t["y_U"]))
    return {
        "E": stratified_binary(N, e_vals),
        "Y_minus_J_L": stratified_binary(N, yjL_vals),
        "Y_minus_J_U": stratified_binary(N, yjU_vals),
    }


def main():
    here = Path(__file__).resolve().parent
    src = Src(sys.argv[1] if len(sys.argv) > 1 else here.parents[1])
    key = json.loads(src.read(TRAJ_KEY))["traj_60"]
    N, n, trajs = key["stratum_population_counts"], key["stratum_sample_sizes"], key["trajectories"]
    lab = {(r["audit_id"], int(r["turn_idx"])): r["human_scope"]
           for r in csv.DictReader(io.StringIO(src.read(TRAJ_LAB)))}

    noncommit_flags = []
    dual_cond_ids = set()
    all_nc_flag_ids = set()
    for aid, t in sorted(trajs.items()):
        flagged = [u for u in t["turns"] if lab.get((aid, u["turn_idx"])) == "Commit"]
        if flagged and t["stratum"] == "agree_NonCommit":
            pairs = sorted({(u["primary_scope"], u["secondary_scope"]) for u in flagged})
            is_dual_cond = all(p == s == "Conditional" for p, s in pairs)
            all_nc_flag_ids.add(aid)
            if is_dual_cond:
                dual_cond_ids.add(aid)
            noncommit_flags.append({
                "audit_id": aid,
                "item_id": t["item_id"],
                "y_L": int(t["y_L"]),
                "y_U": int(t["y_U"]),
                "flagged_turns": [u["turn_idx"] for u in flagged],
                "judge_scopes": pairs,
                "dual_conditional": is_dual_cond,
            })

    y_splits = [aid for aid, t in sorted(trajs.items()) if int(t["y_L"]) != int(t["y_U"])]

    orig_fn = lambda aid, t: any(lab.get((aid, u["turn_idx"])) == "Commit" for u in t["turns"])
    keep_mixed_fn = lambda aid, t: orig_fn(aid, t) and (aid not in dual_cond_ids)
    rubric_fn = lambda aid, t: orig_fn(aid, t) and (aid not in all_nc_flag_ids)

    orig_res = eval_scheme(N, trajs, orig_fn)
    mixed_res = eval_scheme(N, trajs, keep_mixed_fn)
    rubric_res = eval_scheme(N, trajs, rubric_fn)

    tot_N = sum(N[s] for s in STRATA)
    nc_e_contrib = (N["agree_NonCommit"] * orig_res["E"]["stratum_hits"]["agree_NonCommit"] / n["agree_NonCommit"]) / tot_N
    share_nc = nc_e_contrib / (orig_res["E"]["estimate_exact_pct"] / 100.0)

    nc = [(aid, t) for aid, t in trajs.items() if t["stratum"] == "agree_NonCommit"]
    all_wait = [aid for aid, t in nc
                if all(u["primary_scope"] == u["secondary_scope"] == "Wait" for u in t["turns"])]
    all_wait_flagged = [aid for aid in all_wait if orig_fn(aid, trajs[aid])]
    n_nc = n["agree_NonCommit"]
    miss_ub = 1 - 0.05 ** (1 / n_nc)

    rows = list(csv.DictReader(io.StringIO(src.read(TRAJ_CSV))))
    ta = [r for r in rows if r["mech"] == "TypeA" and r["protocol"] in ("DYN_FREE", "DYN_REQUEST")]
    per_item = Counter()
    for r in ta:
        per_item[r["item"]] += int(r["e_L"])
    total = sum(per_item.values())
    top5 = [i for i, _ in per_item.most_common(5)]
    cells = {}
    for r in ta:
        if r["item"] in top5:
            continue
        c = cells.setdefault((r["model"], r["protocol"]), [0, 0, 0])
        c[0] += int(r["e_L"]); c[1] += int(r["e_U"]); c[2] += 1
    rng = [100 * v / c[2] for c in cells.values() for v in c[:2]]

    out = {
        "traj60": {
            "n_trajectories": len(trajs),
            "y_L_vs_y_U_disagreements_in_sample": y_splits,
            "stratum_hits_external_rater": orig_res["E"]["stratum_hits"],
            "E_reported_pct": orig_res["E"]["estimate_pct"],
            "Y_minus_J_reported_pct": orig_res["Y_minus_J_L"]["estimate_pct"],
            "share_of_estimate_from_agree_NonCommit": round(share_nc, 3),
            "agree_NonCommit_flags": noncommit_flags,
            "E_rescored_dual_conditional_as_noncommit_pct": mixed_res["E"]["estimate_pct"],
            "Y_minus_J_rescored_dual_conditional_as_noncommit_pct": mixed_res["Y_minus_J_L"]["estimate_pct"],
            "E_rescored_all_conditional_as_noncommit_pct": rubric_res["E"]["estimate_pct"],
            "Y_minus_J_rescored_all_conditional_as_noncommit_pct": rubric_res["Y_minus_J_L"]["estimate_pct"],
            "schemes_detailed": {
                "external_original": orig_res,
                "rescored_dual_conditional_as_noncommit": mixed_res,
                "rescored_all_scoped_as_noncommit": rubric_res,
            },
            "agree_NonCommit_all_dual_Wait": {"n": len(all_wait), "flagged": len(all_wait_flagged)},
            "agree_NonCommit_miss_rate_upper95_if_flags_scoped_pct": round(100 * miss_ub, 1),
            "implied_E_upper_pp": round(100 * miss_ub * N["agree_NonCommit"] / tot_N, 1),
        },
        "typeA_free_req_concentration": {
            "confirmed_commits_eL": total,
            "items_with_any_commit": sum(v > 0 for v in per_item.values()),
            "n_items": len(per_item),
            "top5_items": top5,
            "top5_share": round(sum(per_item[i] for i in top5) / total, 3),
            "E_range_excluding_top5_pct": [round(min(rng), 1), round(max(rng), 1)],
        },
    }
    (here / "traj60_sensitivity.json").write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
