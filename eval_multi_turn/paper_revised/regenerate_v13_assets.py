#!/usr/bin/env python3
"""Regenerate all 4 tables and 2 figures in `paper_revised/` strictly from `v1.3-input-blind`.

Reads:
  - runs/runs_16k_v13_blind/test100_exp2_4_full_summary_16384_v13_blind.json
  - runs/runs_16k_v13_blind/judged_dev20_*_16384_v13_blind.jsonl
  - revision_analysis_v13/traj.csv, turns.csv

Writes:
  - paper_revised/table1_main_results_rev.tex
  - paper_revised/table2_paired_contrasts_rev.tex
  - paper_revised/table3_exp3_mechanism_repair_rev.tex
  - paper_revised/table4_exp4a_judge_audit_rev.tex
  - paper_revised/fig1_yj_gap.{pdf,png}
  - paper_revised/fig2_hazard_guard_scope.{pdf,png}
  - revision_analysis_v13/v13_paper_numbers.json
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd

REV_DIR = Path(__file__).resolve().parent
EVAL_DIR = REV_DIR.parent
V13_DIR = EVAL_DIR / "runs" / "runs_16k_v13_blind"
RA_DIR = EVAL_DIR / "revision_analysis_v13"

MODELS_FULL = ["Qwen/Qwen3-8B", "zai-org/GLM-Z1-9B-0414", "Qwen/Qwen3-14B"]
MODELS_SHORT = ["Qwen3-8B", "GLM-Z1-9B-0414", "Qwen3-14B"]
MODELS_DISP = ["Qwen3-8B", "GLM-Z1-9B", "Qwen3-14B"]
PROTO_MACRO = {
    "FULL": r"\pFULL",
    "CONCAT": r"\pCON",
    "CONCAT_MATCHED": r"\pCONM",
    "DYN_FREE": r"\pFREE",
    "DYN_REQUEST": r"\pREQ",
    "DYN_REQUEST_GUARD": r"\pGRD",
}
DYN_PROTOS = ["DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"]

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "font.size": 8,
    "axes.titlesize": 8.5,
    "axes.labelsize": 8,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.0,
    "axes.linewidth": 0.7,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})


def fmt_lu(lo: float, hi: float, dec: int = 1, signed: bool = False) -> str:
    s_lo = f"{lo:+.{dec}f}" if signed else f"{lo:.{dec}f}"
    s_hi = f"{hi:+.{dec}f}" if signed else f"{hi:.{dec}f}"
    # Normalize -0.0 -> +0.0 if signed else 0.0
    if signed:
        s_lo = "+0.0" if s_lo == "-0.0" else s_lo
        s_hi = "+0.0" if s_hi == "-0.0" else s_hi
        s_lo = s_lo.replace("-", "$-$")
        s_hi = s_hi.replace("-", "$-$")
        s_lo = s_lo.replace("+", "$+$")
        s_hi = s_hi.replace("+", "$+$")
    if s_lo == s_hi:
        return s_lo
    return f"[{s_lo}, {s_hi}]"


def fmt_signed_num(v: float, dec: int = 1) -> str:
    s = f"{v:+.{dec}f}"
    if s in ("-0.0", "-0.00"):
        s = "+0.0" if dec == 1 else "+0.00"
    return s.replace("-", "$-$").replace("+", "$+$")


def fmt_env_ci(lo_pt: float, hi_pt: float, ci_lo: float, ci_hi: float, star_on_excl0: bool = True) -> str:
    pt = fmt_lu(lo_pt, hi_pt, dec=1, signed=True)
    ci = f"({fmt_signed_num(ci_lo)}, {fmt_signed_num(ci_hi)})"
    excl0 = (ci_lo > 0) or (ci_hi < 0)
    star = "$^*$" if (star_on_excl0 and excl0) else ""
    return f"{pt} {ci}{star}"


def cohen_kappa(a: list, b: list) -> tuple[float, float]:
    n = len(a)
    ag = sum(x == y for x, y in zip(a, b)) / n
    ca, cb = Counter(a), Counter(b)
    pe = sum((ca[k] / n) * (cb[k] / n) for k in sorted(set(ca) | set(cb)))
    return ag, (0.0 if abs(1 - pe) < 1e-12 else (ag - pe) / (1 - pe))


def main() -> None:
    summ = json.loads((V13_DIR / "test100_exp2_4_full_summary_16384_v13_blind.json").read_text(encoding="utf-8"))
    T = pd.read_csv(RA_DIR / "traj.csv")
    U = pd.read_csv(RA_DIR / "turns.csv")
    keys = []
    for _, r in T.iterrows():
        keys += [r.model] * (int(r.k) - 1 if r.protocol in DYN_PROTOS else 0)
    assert len(keys) == len(U)
    U["model"] = keys
    U["c_pri"] = (U.pri == "Commit").astype(int)
    U["c_sec"] = (U.sec == "Commit").astype(int)
    T = T.merge(U.groupby(["model", "traj"])[["c_pri", "c_sec"]].max().reset_index(),
                on=["model", "traj"], how="left").fillna({"c_pri": 0, "c_sec": 0})
    U2 = U.merge(T[["model", "traj", "protocol", "item", "mech", "k"]], on=["model", "traj"])

    # ---------------- TABLE I ----------------
    trunc_exact = {}
    for m_slug, m_disp in [("qwen3_8b", "Qwen3-8B"), ("glm_z1_9b_0414", "GLM-Z1-9B"), ("qwen3_14b", "Qwen3-14B")]:
        by_p = Counter()
        tot_p = Counter()
        for suf in ["16384_v13_blind.jsonl", "guard_16384_v13_blind.jsonl"]:
            for line in (V13_DIR / f"judged_test100_{m_slug}_{suf}").open(encoding="utf-8"):
                r = json.loads(line)
                tot_p[r["protocol"]] += 1
                if r.get("any_empty_truncated") or r.get("any_vis_truncated"):
                    by_p[r["protocol"]] += 1
        for pr, n in tot_p.items():
            trunc_exact[(m_disp, pr)] = by_p[pr] / n * 100.0

    t1_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Main results on \textsc{Test}~100 (100 problems $\times$ 3 reps = 300 trajectories per cell). $[L,U]$: dual-judge bounds (single value when $L{=}U$). $Y$: final accuracy; $E$: premature commitment on insufficient turns; $J{=}Y(1{-}E)$; $Y{-}J$: finally correct trajectories that committed prematurely.}",
        r"\label{tab:main_results}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{0.94}",
        r"\begin{tabular}{@{}llcccccccc@{}}",
        r"\toprule",
        r"Model & Protocol & $Y$ (\%) & $E$ (\%) & $J$ (\%) & $Y-J$ (\%) & $P(E{=}1\mid Y{=}1)$ & Abstain (\%) & Trunc. (\%) & Tok./Traj \\",
    ]
    for m_full, m_short, m_disp in zip(MODELS_FULL, MODELS_SHORT, MODELS_DISP):
        t1_lines.append(r"\midrule")
        m_data = summ["models"][m_full]["exp2_protocol_summary"]
        for i, pr in enumerate(["FULL", "CONCAT", "CONCAT_MATCHED", "DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"]):
            pd_ = m_data[pr]
            yL, yU = pd_["Y"][0] * 100, pd_["Y"][1] * 100
            eL, eU = pd_["E"][0] * 100, pd_["E"][1] * 100
            jL, jU = pd_["J"][0] * 100, pd_["J"][1] * 100
            yjL, yjU = pd_["Y_minus_J_eq_YE"][0] * 100, pd_["Y_minus_J_eq_YE"][1] * 100
            p_eyL, p_eyU = pd_["P_E1_given_Y1"][0] * 100, pd_["P_E1_given_Y1"][1] * 100
            abL, abU = pd_["final_explicit_abstain_rate"][0] * 100, pd_["final_explicit_abstain_rate"][1] * 100
            trunc = trunc_exact[(m_disp, pr)]
            tok = round(pd_["mean_tokens_per_trajectory"])
            prefix = rf"\multirow{{6}}{{*}}{{{m_disp}}}" if i == 0 else ""
            e_str = "0.0$^*$" if pr in ("FULL", "CONCAT", "CONCAT_MATCHED") else fmt_lu(eL, eU)
            yj_str = "0.0" if pr in ("FULL", "CONCAT", "CONCAT_MATCHED") else fmt_lu(yjL, yjU)
            pey_str = "0.0" if pr in ("FULL", "CONCAT", "CONCAT_MATCHED") else fmt_lu(p_eyL, p_eyU)
            t1_lines.append(
                f"{prefix} & {PROTO_MACRO[pr]} & {fmt_lu(yL, yU)} & {e_str} & {fmt_lu(jL, jU)} & {yj_str} & {pey_str} & {fmt_lu(abL, abU)} & {trunc:.1f} & {tok:,} \\\\"
            )
    t1_lines.extend([
        r"\bottomrule",
        r"\multicolumn{10}{@{}l}{\scriptsize $^*$Single-turn controls have no insufficient intermediate turns ($E\equiv 0$ structurally, hence $J\equiv Y$).}",
        r"\end{tabular}",
        r"\end{table*}",
    ])
    (REV_DIR / "table1_main_results_rev.tex").write_text("\n".join(t1_lines) + "\n", encoding="utf-8")

    # ---------------- TABLE II (paired bootstrap B=10,000, seed 20261024) ----------------
    rng = np.random.default_rng(20261024)
    B = 10_000
    items_all = sorted(T.item.unique())
    items_A = sorted(T[T.mech == "TypeA"].item.unique())
    items_B = sorted(T[T.mech == "TypeB"].item.unique())
    idx_all = rng.integers(0, len(items_all), (B, len(items_all)))
    idx_A = rng.integers(0, len(items_A), (B, len(items_A)))
    idx_B = rng.integers(0, len(items_B), (B, len(items_B)))

    def boot_env(m_short: str, p1: str, p2: str, col_L: str, col_U: str, item_list, idx_mat):
        sub = T[T.model == m_short]
        pivL = sub.pivot_table(index="item", columns="protocol", values=col_L, aggfunc="mean").loc[item_list]
        pivU = sub.pivot_table(index="item", columns="protocol", values=col_U, aggfunc="mean").loc[item_list]
        d_lo = (pivL[p1] - pivU[p2]).values * 100.0
        d_hi = (pivU[p1] - pivL[p2]).values * 100.0
        ci_lo = float(np.percentile(d_lo[idx_mat].mean(1), 2.5))
        ci_hi = float(np.percentile(d_hi[idx_mat].mean(1), 97.5))
        return float(d_lo.mean()), float(d_hi.mean()), ci_lo, ci_hi

    def boot_did(models: list[str], col: str):
        sub = T[T.model.isin(models) & T.protocol.isin(["DYN_REQUEST", "DYN_REQUEST_GUARD"])]
        piv = sub.pivot_table(index="item", columns="protocol", values=col, aggfunc="mean")
        d = (piv["DYN_REQUEST_GUARD"] - piv["DYN_REQUEST"]) * 100.0
        dA = d.loc[items_A].values
        dB = d.loc[items_B].values
        bA = dA[idx_A].mean(1)
        bB = dB[idx_B].mean(1)
        bD = bA - bB
        return {
            "A": (float(dA.mean()), float(np.percentile(bA, 2.5)), float(np.percentile(bA, 97.5))),
            "B": (float(dB.mean()), float(np.percentile(bB, 2.5)), float(np.percentile(bB, 97.5))),
            "DiD": (float(dA.mean() - dB.mean()), float(np.percentile(bD, 2.5)), float(np.percentile(bD, 97.5))),
        }

    def boot_scope_row(m_short: str, p1: str, p2: str, label: str):
        sub = U2[(U2.model == m_short) & (U2.protocol.isin([p1, p2]))].copy()
        sub["pri_hit"] = (sub.pri == label).astype(float)
        sub["sec_hit"] = (sub.sec == label).astype(float)
        sub["lo_hit"] = ((sub.pri == label) & (sub.sec == label)).astype(float)
        sub["hi_hit"] = ((sub.pri == label) | (sub.sec == label)).astype(float)
        # Item-level turn counts & hits (ratio estimator over items, matching analyze_experiments)
        grp = sub.groupby(["item", "protocol"]).agg(
            n=("t", "count"),
            pri=("pri_hit", "sum"), sec=("sec_hit", "sum"),
            lo=("lo_hit", "sum"), hi=("hi_hit", "sum"),
        ).unstack()
        n1 = grp["n"][p1].loc[items_all].values
        n2 = grp["n"][p2].loc[items_all].values
        res = {}
        for nm in ["pri", "sec"]:
            s1 = grp[nm][p1].loc[items_all].values
            s2 = grp[nm][p2].loc[items_all].values
            pt = (s1.sum() / n1.sum() - s2.sum() / n2.sum()) * 100.0
            bs = (s1[idx_all].sum(1) / n1[idx_all].sum(1) - s2[idx_all].sum(1) / n2[idx_all].sum(1)) * 100.0
            lo_ci, hi_ci = float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))
            res[nm] = (float(pt), lo_ci, hi_ci, (lo_ci > 0) or (hi_ci < 0))
        s1_lo = grp["lo"][p1].loc[items_all].values
        s2_hi = grp["hi"][p2].loc[items_all].values
        s1_hi = grp["hi"][p1].loc[items_all].values
        s2_lo = grp["lo"][p2].loc[items_all].values
        env_lo_bs = (s1_lo[idx_all].sum(1) / n1[idx_all].sum(1) - s2_hi[idx_all].sum(1) / n2[idx_all].sum(1)) * 100.0
        env_hi_bs = (s1_hi[idx_all].sum(1) / n1[idx_all].sum(1) - s2_lo[idx_all].sum(1) / n2[idx_all].sum(1)) * 100.0
        env_ci_lo = float(np.percentile(env_lo_bs, 2.5))
        env_ci_hi = float(np.percentile(env_hi_bs, 97.5))
        res["env_ci"] = (env_ci_lo, env_ci_hi)
        return res

    did_store = {}
    for col in ["e_L", "e_U", "c_pri", "c_sec", "j_L", "j_U"]:
        did_store[col] = {}
        for ms, nm in [([m], m) for m in MODELS_SHORT] + [(MODELS_SHORT, "Pooled")]:
            did_store[col][nm] = boot_did(ms, col)

    t2_rows = []
    specs = [
        (r"\pFULL{} $-$ \pCON{} ($\Delta Y$)", "FULL", "CONCAT", "y_L", "y_U", items_all, idx_all),
        (r"\pCON{} $-$ \pCONM{} ($\Delta Y$)", "CONCAT", "CONCAT_MATCHED", "y_L", "y_U", items_all, idx_all),
        (r"\pCONM{} $-$ \pFREE{} ($\Delta Y$)", "CONCAT_MATCHED", "DYN_FREE", "y_L", "y_U", items_all, idx_all),
        (r"\pREQ{} $-$ \pFREE{} ($\Delta Y$)", "DYN_REQUEST", "DYN_FREE", "y_L", "y_U", items_all, idx_all),
        (r"\pREQ{} $-$ \pFREE{} ($\Delta E$)", "DYN_REQUEST", "DYN_FREE", "e_L", "e_U", items_all, idx_all),
        ("MIDRULE", None, None, None, None, None, None),
        (r"\pGRD{} $-$ \pREQ{} ($\Delta E$)", "DYN_REQUEST_GUARD", "DYN_REQUEST", "e_L", "e_U", items_all, idx_all),
        (r"\pGRD{} $-$ \pREQ{} ($\Delta J$)", "DYN_REQUEST_GUARD", "DYN_REQUEST", "j_L", "j_U", items_all, idx_all),
        (r"\pGRD{} $-$ \pREQ{} ($\Delta Y$)", "DYN_REQUEST_GUARD", "DYN_REQUEST", "y_L", "y_U", items_all, idx_all),
        (r"\quad $\hookrightarrow$ \TA{} ($N{=}68$, $\Delta E$)", "DYN_REQUEST_GUARD", "DYN_REQUEST", "e_L", "e_U", items_A, idx_A),
        (r"\quad $\hookrightarrow$ \TB{} ($N{=}32$, $\Delta E$)", "DYN_REQUEST_GUARD", "DYN_REQUEST", "e_L", "e_U", items_B, idx_B),
    ]
    for lbl, p1, p2, cL, cU, ilist, imat in specs:
        if lbl == "MIDRULE":
            t2_rows.append(r"\midrule")
            continue
        cells = [fmt_env_ci(*boot_env(m, p1, p2, cL, cU, ilist, imat)) for m in MODELS_SHORT]
        t2_rows.append(f"{lbl} & " + " & ".join(cells) + r" \\")

    for col, lbl in [("e_L", r"\quad Interaction \TA$-$\TB, $E_L$"), ("e_U", r"\quad Interaction \TA$-$\TB, $E_U$")]:
        cells = []
        for m in MODELS_SHORT:
            pt, lo, hi = did_store[col][m]["DiD"]
            star = "$^*$" if (lo > 0 or hi < 0) else ""
            cells.append(f"{fmt_signed_num(pt)} ({fmt_signed_num(lo)}, {fmt_signed_num(hi)}){star}")
        t2_rows.append(f"{lbl} & " + " & ".join(cells) + r" \\")

    t2_rows.append(r"\midrule")
    scope_store = {}
    for lbl, p1, p2, sc_lbl in [
        (r"\pREQ{} $-$ \pFREE{} (\texttt{Wait})", "DYN_REQUEST", "DYN_FREE", "Wait"),
        (r"\pREQ{} $-$ \pFREE{} (\texttt{Cond.})", "DYN_REQUEST", "DYN_FREE", "Conditional"),
        (r"\pGRD{} $-$ \pREQ{} (\texttt{Wait})", "DYN_REQUEST_GUARD", "DYN_REQUEST", "Wait"),
        (r"\pGRD{} $-$ \pREQ{} (\texttt{Cond.})", "DYN_REQUEST_GUARD", "DYN_REQUEST", "Conditional"),
    ]:
        cells = []
        scope_store[(p1, p2, sc_lbl)] = {}
        for m in MODELS_SHORT:
            sr = boot_scope_row(m, p1, p2, sc_lbl)
            scope_store[(p1, p2, sc_lbl)][m] = sr
            elo, ehi = sr["env_ci"]
            pp, _, _, psig = sr["pri"]
            sp_, _, _, ssig = sr["sec"]
            dag = r"$^\dagger$"
            cells.append(
                f"[{fmt_signed_num(elo)}, {fmt_signed_num(ehi)}] / "
                f"{fmt_signed_num(pp)}{dag if psig else ''} / "
                f"{fmt_signed_num(sp_)}{dag if ssig else ''}"
            )
        t2_rows.append(f"{lbl} & " + " & ".join(cells) + r" \\")

    t2_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        r"\caption{Paired problem-bootstrap contrasts on \textsc{Test}~100 (pp, $B{=}10{,}000$). Outcome rows: point range $[L,U]$ and 95\% envelope CI (lo, hi); $^*$CI excludes 0. Interaction rows: $\Delta E_{\mathrm{TypeA}}{-}\Delta E_{\mathrm{TypeB}}$ for \pGRD$-$\pREQ{} (stratified bootstrap), by $E_L$/$E_U$; $^*$CI excludes 0. Scope rows: 95\% envelope CI / primary / secondary; $^\dagger$judge CI excludes 0 (no envelope does). \TA/\TB{} rows give envelopes; per-bound $E_L$/$E_U$ CIs appear in Fig.~\ref{fig:scope_hazard}b and Sec.~\ref{sec:exp4b}.}",
        r"\label{tab:paired_contrasts}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{4.0pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{@{}lccc@{}}",
        r"\toprule",
        r"Paired contrast (pp) & Qwen3-8B & GLM-Z1-9B & Qwen3-14B \\",
        r"\midrule",
        *t2_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (REV_DIR / "table2_paired_contrasts_rev.tex").write_text("\n".join(t2_lines) + "\n", encoding="utf-8")

    # ---------------- TABLE III ----------------
    d_rep_all = T[T.e_U == 1].copy()
    d_rep_all["rp"] = pd.to_numeric(d_rep_all.rep_pri, errors="coerce").fillna(0).astype(int)
    d_rep_all["rs"] = pd.to_numeric(d_rep_all.rep_sec, errors="coerce").fillna(0).astype(int)
    tot_both = int(((d_rep_all.rp == 1) & (d_rep_all.rs == 1)).sum())
    tot_any = int(((d_rep_all.rp == 1) | (d_rep_all.rs == 1)).sum())
    tot_n = len(d_rep_all)

    t3_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{Non-final scope distribution (852 insufficient turns per cell; Dis./Oth.\ = non-commit judge disagreements, Partial/Other, Unresolved, and $<0.3\%$ empty truncations), regime-stratified $E$ and $Y{{-}}J$ ($[L,U]$, \%), and unprompted explicit retraction among $E_U{{=}}1$ trajectories (confirmed by both judges / flagged by either / $n$). Pooled retraction: ${tot_both}/{tot_any}/{tot_n}$ (${tot_both/tot_n*100:.1f}\%$--${tot_any/tot_n*100:.1f}\%$).}}",
        r"\label{tab:exp3_mechanism}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.6pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{@{}llcccccccc@{}}",
        r"\toprule",
        r"Model & Protocol & Wait & Cond. & Commit & Dis./Oth. & \TA{} $E$ & \TA{} $Y{-}J$ & \TB{} $E$ & Retraction (both/any/$n$) \\",
    ]
    t3_stats = {}
    for m_short, m_disp in zip(MODELS_SHORT, MODELS_DISP):
        t3_lines.append(r"\midrule")
        for i, pr in enumerate(DYN_PROTOS):
            ug = U2[(U2.model == m_short) & (U2.protocol == pr)]
            vc = ug.scope.value_counts(normalize=True) * 100.0
            w_s, c_s, cm_s = vc.get("Wait", 0.0), vc.get("Conditional", 0.0), vc.get("Commit", 0.0)
            do_s = 100.0 - w_s - c_s - cm_s
            ta = T[(T.model == m_short) & (T.protocol == pr) & (T.mech == "TypeA")]
            tb = T[(T.model == m_short) & (T.protocol == pr) & (T.mech == "TypeB")]
            ta_eL, ta_eU = ta.e_L.mean() * 100, ta.e_U.mean() * 100
            ta_yjL, ta_yjU = (ta.y_L * ta.e_L).mean() * 100, (ta.y_U * ta.e_U).mean() * 100
            tb_eL, tb_eU = tb.e_L.mean() * 100, tb.e_U.mean() * 100
            rg = d_rep_all[(d_rep_all.model == m_short) & (d_rep_all.protocol == pr)]
            rboth = int(((rg.rp == 1) & (rg.rs == 1)).sum())
            rany = int(((rg.rp == 1) | (rg.rs == 1)).sum())
            rn = len(rg)
            t3_stats[(m_short, pr)] = {
                "ta_e": (ta_eL, ta_eU), "ta_yj": (ta_yjL, ta_yjU), "tb_e": (tb_eL, tb_eU),
                "ta_y": (ta.y_L.mean() * 100, ta.y_U.mean() * 100),
            }
            prefix = rf"\multirow{{3}}{{*}}{{{m_disp}}}" if i == 0 else ""
            t3_lines.append(
                f"{prefix} & {PROTO_MACRO[pr]} & {w_s:.1f} & {c_s:.1f} & {cm_s:.1f} & {do_s:.1f} & "
                f"{fmt_lu(ta_eL, ta_eU)} & {fmt_lu(ta_yjL, ta_yjU)} & {fmt_lu(tb_eL, tb_eU)} & {rboth} / {rany} / {rn} \\\\"
            )
    t3_lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table*}"])
    (REV_DIR / "table3_exp3_mechanism_repair_rev.tex").write_text("\n".join(t3_lines) + "\n", encoding="utf-8")

    # ---------------- TABLE IV (DEV 20 v1.3 + TEST 100 v1.3) ----------------
    dev_kappas = {"y": [], "e_traj": [], "e_turn": [], "scope": []}
    for m_slug in ["qwen3_8b", "glm_z1_9b_0414", "qwen3_14b"]:
        drows = [json.loads(l) for l in (V13_DIR / f"judged_dev20_{m_slug}_16384_v13_blind.jsonl").open(encoding="utf-8")]
        # Final Y (non-empty)
        yp, ys = [], []
        for r in drows:
            fg = r["final_grading"]
            if fg.get("primary_llm") and fg.get("secondary_llm"):
                yp.append(bool(fg["primary_llm"]["correct"]))
                ys.append(bool(fg["secondary_llm"]["correct"]))
        dev_kappas["y"].append(cohen_kappa(yp, ys)[1])
        # Dynamic scope & traj commit
        dyn_r = [r for r in drows if r["protocol"].startswith("DYN_")]
        tp = [int(any(t.get("primary_scope") == "Commit" for t in r["turns"] if not t["is_final_turn"])) for r in dyn_r]
        ts = [int(any(t.get("secondary_scope") == "Commit" for t in r["turns"] if not t["is_final_turn"])) for r in dyn_r]
        dev_kappas["e_traj"].append(cohen_kappa(tp, ts)[1])
        sp, ss = [], []
        for r in dyn_r:
            for t in r["turns"]:
                if not t["is_final_turn"] and t.get("primary_scope") and t.get("secondary_scope"):
                    sp.append(t["primary_scope"])
                    ss.append(t["secondary_scope"])
        dev_kappas["scope"].append(cohen_kappa(sp, ss)[1])
        dev_kappas["e_turn"].append(cohen_kappa([int(x == "Commit") for x in sp], [int(x == "Commit") for x in ss])[1])
    # Pooled DEV 20 Commit kappa
    dev_tp_all, dev_ts_all, dev_sp_all, dev_ss_all = [], [], [], []
    for m_slug in ["qwen3_8b", "glm_z1_9b_0414", "qwen3_14b"]:
        drows = [json.loads(l) for l in (V13_DIR / f"judged_dev20_{m_slug}_16384_v13_blind.jsonl").open(encoding="utf-8")]
        for r in drows:
            if not r["protocol"].startswith("DYN_"):
                continue
            dev_tp_all.append(int(any(t.get("primary_scope") == "Commit" for t in r["turns"] if not t["is_final_turn"])))
            dev_ts_all.append(int(any(t.get("secondary_scope") == "Commit" for t in r["turns"] if not t["is_final_turn"])))
            for t in r["turns"]:
                if not t["is_final_turn"] and t.get("primary_scope") and t.get("secondary_scope"):
                    dev_sp_all.append(t["primary_scope"])
                    dev_ss_all.append(t["secondary_scope"])
    dev_e_traj_pooled = cohen_kappa(dev_tp_all, dev_ts_all)[1]
    dev_e_turn_pooled = cohen_kappa([int(x == "Commit") for x in dev_sp_all], [int(x == "Commit") for x in dev_ss_all])[1]

    t4_rows = []
    t4_stats = {}
    for m_full, m_short, m_disp in zip(MODELS_FULL, MODELS_SHORT, MODELS_DISP):
        # Final Y from raw v1.3 JSONLs (non-empty)
        m_slug = {"Qwen3-8B": "qwen3_8b", "GLM-Z1-9B-0414": "glm_z1_9b_0414", "Qwen3-14B": "qwen3_14b"}[m_short]
        yp, ys = [], []
        for suf in ["16384_v13_blind.jsonl", "guard_16384_v13_blind.jsonl"]:
            for line in (V13_DIR / f"judged_test100_{m_slug}_{suf}").open(encoding="utf-8"):
                fg = json.loads(line)["final_grading"]
                if fg.get("primary_llm") and fg.get("secondary_llm"):
                    yp.append(bool(fg["primary_llm"]["correct"]))
                    ys.append(bool(fg["secondary_llm"]["correct"]))
        y_ag, y_kp = cohen_kappa(yp, ys)

        # Trajectory Commit E (over all 900 dynamic trajectories)
        subT = T[(T.model == m_short) & (T.protocol.isin(DYN_PROTOS))]
        e_ag, e_kp = cohen_kappa(subT.c_pri.astype(int).tolist(), subT.c_sec.astype(int).tolist())

        # Turn-level 4-class scope C (over non-empty turns)
        subU = U2[(U2.model == m_short) & (U2.pri.notna()) & (U2.sec.notna()) & (U2.pri != "") & (U2.sec != "")]
        c_ag, c_kp = cohen_kappa(subU.pri.tolist(), subU.sec.tolist())
        te_ag, te_kp = cohen_kappa((subU.pri == "Commit").astype(int).tolist(), (subU.sec == "Commit").astype(int).tolist())

        # Explicit repair
        rg = d_rep_all[d_rep_all.model == m_short]
        pp = int(((rg.rp == 1) & (rg.rs == 1)).sum())
        pm = int(((rg.rp == 1) & (rg.rs == 0)).sum())
        mp = int(((rg.rp == 0) & (rg.rs == 1)).sum())
        mm = int(((rg.rp == 0) & (rg.rs == 0)).sum())
        r_ag, r_kp = cohen_kappa(rg.rp.tolist(), rg.rs.tolist())
        pa = (2 * pp / (2 * pp + pm + mp)) if (2 * pp + pm + mp) > 0 else 0.0
        t4_stats[m_short] = {
            "y_ag": y_ag, "y_kp": y_kp, "e_ag": e_ag, "e_kp": e_kp,
            "te_ag": te_ag, "te_kp": te_kp, "c_ag": c_ag, "c_kp": c_kp,
            "r_ag": r_ag, "r_kp": r_kp, "pa": pa, "cells": (pp, pm, mp, mm),
        }
        kp_str = f"$-${abs(r_kp):.3f}" if r_kp < 0 else f"{r_kp:.3f}"
        t4_rows.append(
            f"{m_disp} & {y_ag*100:.2f}\\% & {y_kp:.3f} & {e_ag*100:.2f}\\% & {e_kp:.3f} & "
            f"{c_ag*100:.1f}\\% & {c_kp:.3f} & {r_ag*100:.1f}\\% & {kp_str} & {pa:.2f} & {pp}/{pm}/{mp}/{mm} \\\\"
        )

    t4_lines = [
        r"\begin{table*}[!t]",
        r"\centering",
        rf"\caption{{Dual-LLM judge agreement (primary \texttt{{gpt-6.1-sol}} vs.\ secondary \texttt{{deepseek-v4.1-flash}}; 0 parse failures on all 5,400 \textsc{{Test}} trajectories; on \textsc{{Dev}}~20, $\kappa_Y{{=}}1.000$, pooled $\kappa_E{{=}}{dev_e_traj_pooled:.3f}$ trajectory / ${dev_e_turn_pooled:.3f}$ turn, scope $\kappa{{=}}{min(dev_kappas['scope']):.3f}$--${max(dev_kappas['scope']):.3f}$). Commit $E$ is evaluated per dynamic trajectory ($n{{=}}900$/model; turn-level $\kappa_E{{=}}{min(v['te_kp'] for v in t4_stats.values()):.3f}$--${max(v['te_kp'] for v in t4_stats.values()):.3f}$). For explicit repair, raw agreement is inflated by the ${{\le}}5\%$ positive base rate; positive agreement $\mathrm{{PA}}{{=}}2n_{{++}}/(2n_{{++}}{{+}}n_{{+-}}{{+}}n_{{-+}})$ shows that the judges rarely agree on which trajectories retract.}}",
        r"\label{tab:exp4a_calibration}",
        r"\footnotesize",
        r"\setlength{\tabcolsep}{3.8pt}",
        r"\renewcommand{\arraystretch}{0.96}",
        r"\begin{tabular}{@{}lcccccccccc@{}}",
        r"\toprule",
        r" & \multicolumn{2}{c}{Final $Y$} & \multicolumn{2}{c}{Commit $E$} & \multicolumn{2}{c}{4-class scope $C$} & \multicolumn{4}{c}{Explicit repair} \\",
        r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-11}",
        r"Model & Agree & $\kappa$ & Agree & $\kappa$ & Agree & $\kappa$ & Agree & $\kappa$ & PA & $++/+-/-+/--$ \\",
        r"\midrule",
        *t4_rows,
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}",
    ]
    (REV_DIR / "table4_exp4a_judge_audit_rev.tex").write_text("\n".join(t4_lines) + "\n", encoding="utf-8")

    # ---------------- FIGURE 1: Single-column Y-J gap (All items vs TypeA core) ----------------
    fig, axes = plt.subplots(1, 2, figsize=(3.48, 1.85), sharey=True)
    colors = {"DYN_FREE": "#3b6fb6", "DYN_REQUEST": "#e07a3f", "DYN_REQUEST_GUARD": "#4ca066"}
    labels_p = {"DYN_FREE": "FREE", "DYN_REQUEST": "REQ", "DYN_REQUEST_GUARD": "GUARD"}
    x = np.arange(3)
    w = 0.25
    for ax, (title, filt) in zip(axes, [("(a) All items ($N{=}100$)", None), ("(b) TypeA core ($N{=}68$)", "TypeA")]):
        for j, pr in enumerate(DYN_PROTOS):
            mids, errs = [], []
            for m_short in MODELS_SHORT:
                sub = T[(T.model == m_short) & (T.protocol == pr)]
                if filt:
                    sub = sub[sub.mech == filt]
                lo = (sub.y_L * sub.e_L).mean() * 100
                hi = (sub.y_U * sub.e_U).mean() * 100
                mids.append((lo + hi) / 2.0)
                errs.append((hi - lo) / 2.0)
            ax.bar(
                x + (j - 1) * w, mids, width=w * 0.92, color=colors[pr],
                label=labels_p[pr], yerr=errs, capsize=1.8,
                error_kw={"elinewidth": 0.75, "ecolor": "#222222", "capthick": 0.75},
                zorder=3,
            )
        ax.set_title(title, pad=4, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(["Qwen3\n8B", "GLM-Z1\n9B", "Qwen3\n14B"], fontsize=7)
        ax.set_ylim(0, 20.5)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6, zorder=0)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"$Y - J$ (pp)", fontsize=8)
    axes[0].legend(frameon=False, ncol=3, loc="upper left", bbox_to_anchor=(0.0, 1.02),
                   handlelength=0.9, columnspacing=0.7, handletextpad=0.35, fontsize=6.6)
    fig.tight_layout(pad=0.4, w_pad=0.6)
    fig.savefig(REV_DIR / "fig1_yj_gap.pdf", bbox_inches="tight")
    fig.savefig(REV_DIR / "fig1_yj_gap.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # ---------------- FIGURE 2: 3-panel (a) h(t,k) heatmap, (b) Guard forest, (c) Scope stacked bar ----------------
    fig, axes = plt.subplots(1, 3, figsize=(7.16, 1.80), gridspec_kw={"width_ratios": [1.05, 1.25, 1.15]})

    # Panel (a): h(t, k) heatmap
    ax = axes[0]
    U2s = U2.sort_values(["model", "traj", "t"]).copy()
    U2s["prev"] = U2s.groupby(["model", "traj"])["e_L"].transform(lambda s: s.shift(fill_value=0).cummax())
    R = U2s[U2s.prev == 0]
    ks = [3, 4, 5, 6]
    ts = [1, 2, 3, 4, 5]
    mat = np.full((len(ks), len(ts)), np.nan)
    cnts = np.zeros((len(ks), len(ts)), dtype=int)
    for r_i, kv in enumerate(ks):
        for c_i, tv in enumerate(ts):
            if tv < kv:
                g = R[(R.k == kv) & (R.t == tv)]
                mat[r_i, c_i] = g.e_L.mean() * 100.0
                cnts[r_i, c_i] = len(g)
    cmap = plt.cm.Reds.copy()
    cmap.set_bad(color="white")
    ax.imshow(mat, cmap=cmap, vmin=0, vmax=24, aspect="auto")
    for r_i, kv in enumerate(ks):
        for c_i, tv in enumerate(ts):
            if tv < kv:
                val = mat[r_i, c_i]
                txt_col = "white" if val >= 14 else "#111111"
                ax.text(c_i, r_i - 0.12, f"{val:.1f}", ha="center", va="center", color=txt_col, fontsize=7.5, fontweight="bold" if tv == kv - 1 else "normal")
                ax.text(c_i, r_i + 0.24, f"n={cnts[r_i, c_i]}", ha="center", va="center", color=txt_col, fontsize=5.6, alpha=0.9)
                if tv == kv - 1:
                    rect = mpatches.Rectangle((c_i - 0.48, r_i - 0.48), 0.96, 0.96, fill=False, edgecolor="#111111", linewidth=1.2)
                    ax.add_patch(rect)
    ax.set_xticks(range(len(ts)))
    ax.set_xticklabels(ts)
    ax.set_yticks(range(len(ks)))
    ax.set_yticklabels(ks)
    ax.set_xlabel(r"Turn $t$  (boxed: last insufficient turn, $r=1$)", fontsize=7.2)
    ax.set_ylabel(r"Task depth $k$", fontsize=7.5)
    ax.set_title(r"(a) First-commit hazard (%), $E_L$", fontsize=8.2, pad=4)

    # Panel (b): Guard effect by regime forest plot
    ax = axes[1]
    y_pos = [3.0, 2.0, 1.0, 0.0]
    row_names = ["Qwen3-8B", "GLM-Z1-9B-0414", "Qwen3-14B", "Pooled"]
    row_disp = ["Qwen3-8B", "GLM-Z1-9B", "Qwen3-14B", "Pooled"]
    colA, colB = "#2b5c8f", "#b83333"
    ax.axvline(0, color="#333333", linewidth=0.8, zorder=1)
    ax.axhline(0.5, color="#bbbbbb", linestyle=":", linewidth=0.7, zorder=1)
    for y0, m_key in zip(y_pos, row_names):
        for off, col_key, reg, col_hex, solid in [
            (+0.24, "e_U", "A", colA, False),
            (+0.09, "e_L", "A", colA, True),
            (-0.09, "e_L", "B", colB, True),
            (-0.24, "e_U", "B", colB, False),
        ]:
            pt, lo, hi = did_store[col_key][m_key][reg]
            ls = "-" if solid else "--"
            mfc = col_hex if solid else "white"
            mk = "o" if solid else "s"
            ax.plot([lo, hi], [y0 + off, y0 + off], color=col_hex, linestyle=ls, linewidth=1.05, zorder=3)
            ax.plot(pt, y0 + off, marker=mk, markersize=4.0, color=col_hex, markerfacecolor=mfc, markeredgewidth=0.9, zorder=4)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(row_disp, fontsize=7.5)
    ax.set_xlim(-11.5, 13.5)
    ax.set_xlabel(r"$\Delta E = \mathrm{GUARD} - \mathrm{REQ}$ (pp, 95% CI)", fontsize=7.5)
    ax.set_title("(b) Guard effect by regime", fontsize=8.2, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.5)
    h_leg = [
        plt.Line2D([0], [0], color=colA, marker="o", markersize=3.8, lw=1.1, label="TypeA"),
        plt.Line2D([0], [0], color=colB, marker="o", markersize=3.8, lw=1.1, label="TypeB"),
        plt.Line2D([0], [0], color="#222222", marker="o", markersize=3.8, lw=1.1, label=r"$E_L$"),
        plt.Line2D([0], [0], color="#222222", marker="s", mfc="white", markersize=3.8, lw=1.0, ls="--", label=r"$E_U$"),
    ]
    ax.legend(handles=h_leg, frameon=False, ncol=4, loc="upper center", bbox_to_anchor=(0.5, -0.46),
              handlelength=1.1, columnspacing=0.65, handletextpad=0.3, fontsize=6.6)

    # Panel (c): Non-final scope stacked horizontal bar (3 models pooled)
    ax = axes[2]
    ys = np.arange(3)[::-1]
    p_names = ["FREE", "REQ", "GUARD"]
    seg_cols = ["#7d96c4", "#f1c232", "#c84646", "#bdbdbd"]
    seg_lbls = ["Wait", "Conditional", "Commit", "Other/disagree"]
    for idx_p, (pr, yv) in enumerate(zip(DYN_PROTOS, ys)):
        g = U2[U2.protocol == pr]
        vc = g.scope.value_counts(normalize=True) * 100.0
        vals = [vc.get("Wait", 0.0), vc.get("Conditional", 0.0), vc.get("Commit", 0.0)]
        vals.append(100.0 - sum(vals))
        left = 0.0
        for s_i, val in enumerate(vals):
            ax.barh(yv, val, left=left, height=0.54, color=seg_cols[s_i], edgecolor="white", linewidth=0.5,
                    label=seg_lbls[s_i] if idx_p == 0 else None)
            if val >= 8.0:
                ax.text(left + val / 2.0, yv, f"{val:.1f}", ha="center", va="center", fontsize=6.8,
                        color="#111111" if s_i != 2 else "white")
            left += val
    ax.set_yticks(ys)
    ax.set_yticklabels(p_names, fontsize=7.5)
    ax.set_xlim(0, 100)
    ax.set_xlabel("Share of insufficient turns (%)", fontsize=7.5)
    ax.set_title("(c) Non-final scope, 3 models pooled", fontsize=8.2, pad=4)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, ncol=2, loc="upper center", bbox_to_anchor=(0.5, -0.44),
              handlelength=0.9, columnspacing=0.8, handletextpad=0.35, fontsize=6.6)

    fig.tight_layout(pad=0.45, w_pad=0.8)
    fig.savefig(REV_DIR / "fig2_hazard_guard_scope.pdf", bbox_inches="tight")
    fig.savefig(REV_DIR / "fig2_hazard_guard_scope.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Save exact numbers JSON for prose verification
    out_summary = {
        "did": did_store,
        "t3": {f"{k[0]}__{k[1]}": v for k, v in t3_stats.items()},
        "t4": t4_stats,
        "dev20_v13": {
            "y_kappas": dev_kappas["y"],
            "e_traj_pooled": dev_e_traj_pooled,
            "e_turn_pooled": dev_e_turn_pooled,
            "scope_kappas": dev_kappas["scope"],
        },
        "retraction_pooled": {"both": tot_both, "any": tot_any, "n": tot_n},
    }
    (RA_DIR / "v13_paper_numbers.json").write_text(json.dumps(out_summary, indent=2), encoding="utf-8")

    # Print exact pooled e_U TypeA CI to 4 decimal places so prose never misstates exclusion of 0
    print("Pooled TypeA GUARD-REQ exact (4 dec):")
    for col in ["e_L", "e_U", "c_pri", "c_sec", "j_L", "j_U"]:
        print(f"  {col:5s}: A={did_store[col]['Pooled']['A']}  B={did_store[col]['Pooled']['B']}  DiD={did_store[col]['Pooled']['DiD']}")
    print("Table IV summary:", json.dumps(t4_stats, indent=2))
    print("Regenerated all 4 tables and 2 figures in paper_revised/ successfully.")


if __name__ == "__main__":
    main()
