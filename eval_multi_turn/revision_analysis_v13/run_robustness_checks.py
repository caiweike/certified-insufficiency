#!/usr/bin/env python3
"""Offline robustness checks for the v1.3-input-blind evaluation (B=10,000).

Reads (relative to repository root):
  - eval_multi_turn/revision_analysis_v13/traj.csv
  - eval_multi_turn/revision_analysis_v13/turns.csv
  - eval_multi_turn/runs/runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl[.gz]

Writes:
  - eval_multi_turn/revision_analysis_v13/robustness_checks_v13.json

Checks performed:
  1. Memorization / lexical-match sensitivity: exclude dynamic trajectories whose
     confirmed premature commit (\boxed{...}) lexically matches the final \boxed{...}.
  2. Discrete-time first-commit hazard model robustness (item-clustered bootstrap,
     B=10,000, seed=20261024): baseline, +regime/subfield covariates, excluding
     sparse k=6, excluding lexically matched trajectories, and TypeA-only excluding
     lexically matched trajectories. Explicitly separates fixed-k (beta_t) from
     fixed-remaining-clauses (beta_t + beta_k) and remaining-clause (beta_k) effects.
  3. Token-consumption decomposition explaining why FREE uses more tokens than REQ.
  4. Exploratory CoT (reasoning_trace) audit on TypeA trajectories whose confirmed
     boxed premature commit differs from the final boxed answer.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

RA_DIR = Path(__file__).resolve().parent
EVAL_DIR = RA_DIR.parent
ROLLOUT_DIR = EVAL_DIR / "runs" / "runs_16k_v13_blind"

MODELS_SHORT = ["Qwen3-8B", "GLM-Z1-9B-0414", "Qwen3-14B"]
MODEL_FILE_SLUGS = {
    "Qwen3-8B": ["judged_test100_qwen3_8b_16384_v13_blind", "judged_test100_qwen3_8b_guard_16384_v13_blind"],
    "GLM-Z1-9B-0414": ["judged_test100_glm_z1_9b_0414_16384_v13_blind", "judged_test100_glm_z1_9b_0414_guard_16384_v13_blind"],
    "Qwen3-14B": ["judged_test100_qwen3_14b_16384_v13_blind", "judged_test100_qwen3_14b_guard_16384_v13_blind"],
}
DYN_PROTOS = ["DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"]

# Transparent regex for explicit reference to the assistant's own prior derivation/answer in final-turn CoT
# (explicitly excludes "用户之前/前面" so references to prior user clauses are not counted)
PRIOR_REF_RE = re.compile(
    r"(?<!用户)(?:(?:之前|前面|刚才|上次|此前|先前|上一轮|早先)(?:我)?(?:已经|曾经|也)?(?:算|得|推|写|答|认为|以为|误|说|求|假设|得出|算出|推出|回答)|"
    r"(?:之前|前面|刚才|上次|此前|先前|上一轮|早先)的(?:答案|结论|结果|回答|推导|表达式|公式|值|假设|计算)|"
    r"(?:回顾|检查|修正|更正|推翻|撤回|修改)(?:之前|前面|刚才|上次|此前|先前|我)?的?(?:推导|答案|结论|结果|回答|计算))",
    re.IGNORECASE,
)


def open_jsonl_auto(base_no_ext: Path):
    """Open <base>.jsonl if present, otherwise <base>.jsonl.gz."""
    p_plain = base_no_ext.with_suffix(".jsonl")
    if p_plain.exists():
        return p_plain.open("r", encoding="utf-8")
    p_gz = Path(str(base_no_ext) + ".jsonl.gz")
    if p_gz.exists():
        return gzip.open(p_gz, "rt", encoding="utf-8")
    raise FileNotFoundError(f"Neither {p_plain} nor {p_gz} exists.")


def extract_last_boxed(text: str | None) -> str:
    """Balanced-brace extraction of the last \\boxed{...} expression."""
    if not text:
        return ""
    idx = text.rfind(r"\boxed{")
    if idx < 0:
        return ""
    start = idx + len(r"\boxed{")
    depth = 1
    i = start
    n = len(text)
    while i < n and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return ""
    return text[start : i - 1].strip()


def norm_box(s: str) -> str:
    """Compact lexical normalization for strict boxed-answer matching."""
    if not s:
        return ""
    s = re.sub(r"\s+", "", s)
    s = s.replace(r"\left", "").replace(r"\right", "").replace(r"\dfrac", r"\frac").replace(r"\tfrac", r"\frac")
    return s


def load_rollouts() -> dict[tuple[str, str], dict]:
    """Load all 5,400 TEST 100 rollouts keyed by (model_short, trajectory_id)."""
    out: dict[tuple[str, str], dict] = {}
    for m_short, slugs in MODEL_FILE_SLUGS.items():
        for slug in slugs:
            with open_jsonl_auto(ROLLOUT_DIR / slug) as f:
                for line in f:
                    if not line.strip():
                        continue
                    r = json.loads(line)
                    out[(m_short, r["trajectory_id"])] = r
    return out


def fit_logit_irls(X: np.ndarray, n_trials: np.ndarray, n_events: np.ndarray, max_iter: int = 30, tol: float = 1e-10) -> np.ndarray:
    """Exact weighted binomial logistic regression MLE via damped Newton-Raphson (IRLS)."""
    p = X.shape[1]
    beta = np.zeros(p, dtype=np.float64)
    # Initialize intercept at log-odds of pooled rate
    tot_n = float(n_trials.sum())
    tot_y = float(n_events.sum())
    rate = np.clip(tot_y / max(tot_n, 1.0), 1e-6, 1.0 - 1e-6)
    beta[0] = np.log(rate / (1.0 - rate))
    eps_ridge = 1e-9 * np.eye(p)

    for _ in range(max_iter):
        eta = np.clip(X @ beta, -30.0, 30.0)
        mu = 1.0 / (1.0 + np.exp(-eta))
        w = n_trials * mu * (1.0 - mu)
        g = X.T @ (n_events - n_trials * mu)
        H = (X.T * w) @ X + eps_ridge
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, g, rcond=None)[0]
        # Backtracking safeguard for rare extreme bootstrap draws
        max_step = float(np.max(np.abs(step)))
        if max_step > 5.0:
            step = step * (5.0 / max_step)
        beta_new = beta + step
        if float(np.max(np.abs(step))) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def run_hazard_spec(
    df_risk: pd.DataFrame,
    outcome_col: str,
    include_mech: bool,
    include_subfield: bool,
    B: int,
    rng: np.random.Generator,
) -> dict:
    """Fit discrete-time hazard logit (matching Sec. IV-C: t + k + protocol + model
    plus optional regime/subfield covariates) with item-clustered bootstrap (B draws)."""
    df = df_risk.copy()
    items = sorted(df["item"].unique())
    n_items = len(items)
    item_to_idx = {it: i for i, it in enumerate(items)}
    df["item_idx"] = df["item"].map(item_to_idx)

    # Build design columns: [1, t, k, protocol dummies, model dummies, (mech==TypeB), (subfield dummies)]
    cols = ["const", "t", "k"]
    df["const"] = 1.0
    for pr in DYN_PROTOS[1:]:
        cname = f"prot_{pr}"
        df[cname] = (df["protocol"] == pr).astype(float)
        cols.append(cname)
    for m in MODELS_SHORT[1:]:
        cname = f"mod_{m}"
        df[cname] = (df["model"] == m).astype(float)
        cols.append(cname)
    if include_mech:
        df["is_typeB"] = (df["mech"] == "TypeB").astype(float)
        cols.append("is_typeB")
    if include_subfield:
        subfields = sorted(df["g"].unique())
        for sf in subfields[1:]:
            cname = f"sf_{sf}"
            df[cname] = (df["g"] == sf).astype(float)
            cols.append(cname)

    # Aggregate into per-(item_idx, design-row) sufficient statistics
    grp_cols = ["item_idx"] + [c for c in cols if c != "const"]
    agg = (
        df.groupby(grp_cols, as_index=False)
        .agg(n_trials=(outcome_col, "count"), n_events=(outcome_col, "sum"))
    )
    agg["const"] = 1.0

    # Full-sample MLE
    X_full = agg[cols].to_numpy(dtype=np.float64)
    n_full = agg["n_trials"].to_numpy(dtype=np.float64)
    y_full = agg["n_events"].to_numpy(dtype=np.float64)
    beta_pt = fit_logit_irls(X_full, n_full, y_full)

    # Group row indices by item_idx for fast item-clustered bootstrap
    # Aggregate across items by multiplying per-item sufficient statistics by bootstrap item counts!
    # Wait: rows sharing the same feature vector across items can be collapsed into an (n_patterns x n_items) matrix!
    feat_cols = [c for c in cols if c != "const"]
    patterns = agg[feat_cols].drop_duplicates().reset_index(drop=True)
    patterns["pat_id"] = np.arange(len(patterns))
    agg_m = agg.merge(patterns, on=feat_cols, how="left")
    n_pat = len(patterns)

    N_mat = np.zeros((n_pat, n_items), dtype=np.float64)
    Y_mat = np.zeros((n_pat, n_items), dtype=np.float64)
    for row in agg_m.itertuples(index=False):
        N_mat[row.pat_id, row.item_idx] += row.n_trials
        Y_mat[row.pat_id, row.item_idx] += row.n_events

    X_pat = np.column_stack([np.ones(n_pat, dtype=np.float64), patterns[feat_cols].to_numpy(dtype=np.float64)])

    # Sanity check: pattern-collapsed MLE equals full-sample MLE
    beta_pat = fit_logit_irls(X_pat, N_mat.sum(axis=1), Y_mat.sum(axis=1))
    assert np.allclose(beta_pt, beta_pat, atol=1e-8), f"MLE mismatch: {beta_pt} vs {beta_pat}"

    # B item-clustered bootstrap draws
    idx_mat = rng.integers(0, n_items, size=(B, n_items))
    boot_betas = np.empty((B, len(cols)), dtype=np.float64)
    for b in range(B):
        counts = np.bincount(idx_mat[b], minlength=n_items).astype(np.float64)
        nb = N_mat @ counts
        yb = Y_mat @ counts
        boot_betas[b] = fit_logit_irls(X_pat, nb, yb)

    def summarize(pt_val: float, samples: np.ndarray) -> dict:
        lo, hi = float(np.percentile(samples, 2.5)), float(np.percentile(samples, 97.5))
        return {
            "pt": float(pt_val),
            "ci95": [lo, hi],
            "or_pt": float(np.exp(pt_val)),
            "or_ci95": [float(np.exp(lo)), float(np.exp(hi))],
            "excludes_0": bool((lo > 0.0) or (hi < 0.0)),
        }

    bt_idx = cols.index("t")
    bk_idx = cols.index("k")
    bt_plus_bk_pt = float(beta_pt[bt_idx] + beta_pt[bk_idx])
    bt_plus_bk_bs = boot_betas[:, bt_idx] + boot_betas[:, bk_idx]

    out = {
        "n_turns": int(len(df)),
        "n_events": int(df[outcome_col].sum()),
        "n_items": int(n_items),
        "n_patterns": int(n_pat),
        "B": int(B),
        "columns": cols,
        "beta_t_fixed_k": summarize(beta_pt[bt_idx], boot_betas[:, bt_idx]),
        "beta_t_plus_k_fixed_r": summarize(bt_plus_bk_pt, bt_plus_bk_bs),
        "beta_k_fixed_t": summarize(beta_pt[bk_idx], boot_betas[:, bk_idx]),
    }
    if include_mech:
        bm_idx = cols.index("is_typeB")
        out["beta_typeB"] = summarize(beta_pt[bm_idx], boot_betas[:, bm_idx])
    return out


def main() -> None:
    T = pd.read_csv(RA_DIR / "traj.csv")
    U = pd.read_csv(RA_DIR / "turns.csv")
    T_dyn = T[T["protocol"].isin(DYN_PROTOS)].copy()
    rollouts = load_rollouts()
    assert len(rollouts) == 5400, f"Expected 5,400 rollouts, got {len(rollouts)}"

    # Map (model, traj, t) -> completion_tokens, boxed, reasoning_trace, visible_reply
    turn_meta = {}
    final_meta = {}
    for (m_short, tid), rec in rollouts.items():
        if rec["protocol"] not in DYN_PROTOS:
            continue
        for td in rec["turns"]:
            t_idx = int(td["turn_idx"])
            vis = td.get("visible_reply") or ""
            cot = td.get("reasoning_trace") or ""
            box_raw = extract_last_boxed(vis)
            info = {
                "completion_tokens": int(td.get("completion_tokens") or 0),
                "visible_reply": vis,
                "reasoning_trace": cot,
                "box_raw": box_raw,
                "box_norm": norm_box(box_raw),
                "is_final": bool(td.get("is_final_turn")),
            }
            if info["is_final"]:
                final_meta[(m_short, tid)] = info
            else:
                turn_meta[(m_short, tid, t_idx)] = info

    # Attach model column to turns.csv (exact alignment matching regenerate_v13_assets.py)
    keys = []
    for r in T.itertuples(index=False):
        if r.protocol in DYN_PROTOS:
            keys.extend([r.model] * (int(r.k) - 1))
    assert len(keys) == len(U), f"Expected {len(U)} turn keys, got {len(keys)}"
    U["model"] = keys

    # Join turns with trajectory metadata
    U2 = U.merge(
        T_dyn[["model", "traj", "protocol", "item", "mech", "k", "g", "y_L", "y_U", "j_L", "j_U"]],
        on=["model", "traj"],
        how="left",
        suffixes=("", "_traj"),
    )
    assert len(U2) == 7668, f"Expected 7,668 non-final turns, got {len(U2)}"

    # ---------- CHECK 1: LEXICAL-MATCH (MEMORIZATION) SENSITIVITY ----------
    lex_matched_trajs: set[tuple[str, str]] = set()
    typeA_commit_stats = {
        "n_eL_trajs": 0,
        "n_with_both_boxes": 0,
        "n_lex_matched": 0,
        "n_boxed_changed": 0,
        "n_unboxed_commit_or_final": 0,
    }
    all_commit_stats = {
        "n_eL_trajs": 0,
        "n_with_both_boxes": 0,
        "n_lex_matched": 0,
        "n_boxed_changed": 0,
        "n_unboxed_commit_or_final": 0,
    }
    changed_typeA_cot_cases = []

    # Group confirmed commit turns (e_L == 1) per trajectory
    commit_turns = U2[U2["e_L"] == 1].sort_values(["model", "traj", "t"])
    by_traj = commit_turns.groupby(["model", "traj"])

    for (m_short, tid), grp in by_traj:
        mech = str(grp["mech"].iloc[0])
        item = str(grp["item"].iloc[0])
        proto = str(grp["protocol"].iloc[0])
        finfo = final_meta[(m_short, tid)]
        fbox = finfo["box_norm"]

        c_boxes = []
        first_commit_t = int(grp["t"].iloc[0])
        first_commit_box = ""
        for t_val in grp["t"].tolist():
            b_norm = turn_meta[(m_short, tid, int(t_val))]["box_norm"]
            if b_norm:
                c_boxes.append((int(t_val), b_norm))
                if not first_commit_box:
                    first_commit_box = b_norm

        all_commit_stats["n_eL_trajs"] += 1
        if mech == "TypeA":
            typeA_commit_stats["n_eL_trajs"] += 1

        matched = bool(fbox and any(b == fbox for _, b in c_boxes))
        if matched:
            lex_matched_trajs.add((m_short, tid))
            all_commit_stats["n_lex_matched"] += 1
            all_commit_stats["n_with_both_boxes"] += 1
            if mech == "TypeA":
                typeA_commit_stats["n_lex_matched"] += 1
                typeA_commit_stats["n_with_both_boxes"] += 1
        elif fbox and c_boxes:
            all_commit_stats["n_with_both_boxes"] += 1
            all_commit_stats["n_boxed_changed"] += 1
            if mech == "TypeA":
                typeA_commit_stats["n_with_both_boxes"] += 1
                typeA_commit_stats["n_boxed_changed"] += 1
                cot_txt = finfo["reasoning_trace"]
                matches = PRIOR_REF_RE.findall(cot_txt)
                has_ref = len(matches) > 0
                changed_typeA_cot_cases.append({
                    "model": m_short,
                    "protocol": proto,
                    "item": item,
                    "traj": tid,
                    "first_commit_t": first_commit_t,
                    "first_commit_box": first_commit_box,
                    "final_box": fbox,
                    "cot_len": len(cot_txt),
                    "cot_refs_prior": has_ref,
                    "matched_terms": sorted(set(m.lower() for m in matches))[:8],
                })
        else:
            all_commit_stats["n_unboxed_commit_or_final"] += 1
            if mech == "TypeA":
                typeA_commit_stats["n_unboxed_commit_or_final"] += 1

    T_dyn["lex_matched"] = [
        (r.model, r.traj) in lex_matched_trajs for r in T_dyn.itertuples(index=False)
    ]
    U2["lex_matched"] = [
        (r.model, r.traj) in lex_matched_trajs for r in U2.itertuples(index=False)
    ]

    def summarize_excl_lex(df_sub: pd.DataFrame) -> dict:
        res = {}
        for m in MODELS_SHORT:
            res[m] = {}
            for pr in DYN_PROTOS:
                g = df_sub[(df_sub["model"] == m) & (df_sub["protocol"] == pr)]
                res[m][pr] = {
                    "n": int(len(g)),
                    "e_L": round(float(g["e_L"].mean() * 100.0), 2),
                    "e_U": round(float(g["e_U"].mean() * 100.0), 2),
                    "y_L": round(float(g["y_L"].mean() * 100.0), 2),
                    "j_L": round(float(g["j_L"].mean() * 100.0), 2),
                    "yj_L": round(float((g["y_L"] - g["j_U"]).mean() * 100.0), 2),
                    "yj_U": round(float((g["y_U"] - g["j_L"]).mean() * 100.0), 2),
                }
        return res

    T_excl = T_dyn[~T_dyn["lex_matched"]]
    lex_check = {
        "n_lex_matched_dynamic_trajs": len(lex_matched_trajs),
        "all_commit_stats": all_commit_stats,
        "typeA_commit_stats": typeA_commit_stats,
        "excl_lex_ALL": summarize_excl_lex(T_excl),
        "excl_lex_TypeA": summarize_excl_lex(T_excl[T_excl["mech"] == "TypeA"]),
    }

    # ---------- CHECK 2: HAZARD MODEL ROBUSTNESS (B=10,000) ----------
    U2s = U2.sort_values(["model", "traj", "t"]).copy()
    U2s["prev_eL"] = U2s.groupby(["model", "traj"])["e_L"].transform(lambda s: s.shift(fill_value=0).cummax())
    U2s["prev_eU"] = U2s.groupby(["model", "traj"])["e_U"].transform(lambda s: s.shift(fill_value=0).cummax())
    R_L = U2s[U2s["prev_eL"] == 0].copy()
    R_U = U2s[U2s["prev_eU"] == 0].copy()

    rng = np.random.default_rng(20261024)
    B = 10_000
    hazard_checks = {
        "baseline_eL": run_hazard_spec(R_L, "e_L", include_mech=False, include_subfield=False, B=B, rng=rng),
        "baseline_eU": run_hazard_spec(R_U, "e_U", include_mech=False, include_subfield=False, B=B, rng=rng),
        "plus_covs_eL": run_hazard_spec(R_L, "e_L", include_mech=True, include_subfield=True, B=B, rng=rng),
        "excl_k6_plus_covs_eL": run_hazard_spec(R_L[R_L["k"] <= 5], "e_L", include_mech=True, include_subfield=True, B=B, rng=rng),
        "excl_lex_plus_covs_eL": run_hazard_spec(R_L[~R_L["lex_matched"]], "e_L", include_mech=True, include_subfield=True, B=B, rng=rng),
        "typeA_excl_lex_plus_sf_eL": run_hazard_spec(
            R_L[(R_L["mech"] == "TypeA") & (~R_L["lex_matched"])], "e_L", include_mech=False, include_subfield=True, B=B, rng=rng
        ),
    }

    # ---------- CHECK 3: FREE vs REQ vs GUARD TOKEN MECHANISM ----------
    U2["comp_tokens"] = [
        turn_meta[(r.model, r.traj, int(r.t))]["completion_tokens"] for r in U2.itertuples(index=False)
    ]
    T_dyn["final_comp_tokens"] = [
        final_meta[(r.model, r.traj)]["completion_tokens"] for r in T_dyn.itertuples(index=False)
    ]
    T_dyn["nonfinal_comp_tokens"] = T_dyn["tot_comp"] - T_dyn["final_comp_tokens"]

    token_check = {"by_protocol": {}, "by_protocol_and_scope": {}, "by_model_protocol": {}}
    for pr in DYN_PROTOS:
        ug = U2[U2["protocol"] == pr]
        tg = T_dyn[T_dyn["protocol"] == pr]
        token_check["by_protocol"][pr] = {
            "mean_nonfinal_turn_comp": round(float(ug["comp_tokens"].mean()), 1),
            "mean_traj_nonfinal_comp": round(float(tg["nonfinal_comp_tokens"].mean()), 1),
            "mean_final_turn_comp": round(float(tg["final_comp_tokens"].mean()), 1),
            "mean_traj_tot_comp": round(float(tg["tot_comp"].mean()), 1),
            "mean_traj_tot_tokens": round(float((tg["tot_prompt"] + tg["tot_comp"]).mean()), 1),
        }
        token_check["by_protocol_and_scope"][pr] = {}
        for sc in ["Wait", "Conditional", "Commit"]:
            sg = ug[ug["scope"] == sc]
            token_check["by_protocol_and_scope"][pr][sc] = {
                "n_turns": int(len(sg)),
                "share_pct": round(float(len(sg) / len(ug) * 100.0), 2),
                "mean_comp_tokens": round(float(sg["comp_tokens"].mean()), 1) if len(sg) else 0.0,
            }
    for m in MODELS_SHORT:
        token_check["by_model_protocol"][m] = {}
        for pr in DYN_PROTOS:
            ug = U2[(U2["model"] == m) & (U2["protocol"] == pr)]
            tg = T_dyn[(T_dyn["model"] == m) & (T_dyn["protocol"] == pr)]
            token_check["by_model_protocol"][m][pr] = {
                "mean_nonfinal_turn_comp": round(float(ug["comp_tokens"].mean()), 1),
                "mean_final_turn_comp": round(float(tg["final_comp_tokens"].mean()), 1),
            }

    # ---------- CHECK 4: EXPLORATORY COT AUDIT ON CHANGED TYPEA COMMITS ----------
    n_changed = len(changed_typeA_cot_cases)
    n_cot_ref = sum(1 for c in changed_typeA_cot_cases if c["cot_refs_prior"])
    by_model_cot = {}
    for m in MODELS_SHORT:
        sub_c = [c for c in changed_typeA_cot_cases if c["model"] == m]
        by_model_cot[m] = {
            "n_changed": len(sub_c),
            "n_cot_refs_prior": sum(1 for c in sub_c if c["cot_refs_prior"]),
            "pct": round(100.0 * sum(1 for c in sub_c if c["cot_refs_prior"]) / len(sub_c), 1) if sub_c else 0.0,
        }
    cot_check = {
        "n_typeA_boxed_changed": n_changed,
        "n_final_cot_refs_prior": n_cot_ref,
        "pct_final_cot_refs_prior": round(100.0 * n_cot_ref / n_changed, 1) if n_changed else 0.0,
        "by_model": by_model_cot,
        "sample_cases": changed_typeA_cot_cases[:15],
    }

    out = {
        "seed": 20261024,
        "bootstrap_B": B,
        "lexical_match_sensitivity": lex_check,
        "hazard_robustness": hazard_checks,
        "token_mechanism": token_check,
        "exploratory_cot_typeA_changed": cot_check,
    }
    out_path = RA_DIR / "robustness_checks_v13.json"
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    # Compact console summary
    print(f"Saved {out_path}")
    print("\n[1] Lexical-match exclusion:")
    print(f"  Total dynamic lex-matched trajs: {lex_check['n_lex_matched_dynamic_trajs']}")
    print(f"  TypeA e_L stats: {typeA_commit_stats}")
    for lbl, blk in [("TypeA excl-lex", lex_check["excl_lex_TypeA"]), ("ALL excl-lex", lex_check["excl_lex_ALL"])]:
        eL_fr = [blk[m][p]["e_L"] for m in MODELS_SHORT for p in ["DYN_FREE", "DYN_REQUEST"]]
        yj_lo = [blk[m][p]["yj_L"] for m in MODELS_SHORT for p in ["DYN_FREE", "DYN_REQUEST"]]
        yj_hi = [blk[m][p]["yj_U"] for m in MODELS_SHORT for p in ["DYN_FREE", "DYN_REQUEST"]]
        print(f"  {lbl:16s}: FREE/REQ e_L range = [{min(eL_fr):.1f}%, {max(eL_fr):.1f}%], Y-J range = [{min(yj_lo):.1f}%, {max(yj_hi):.1f}%]")

    print("\n[2] Hazard robustness (B=10,000):")
    for k_name, h in hazard_checks.items():
        bt = h["beta_t_fixed_k"]
        btr = h["beta_t_plus_k_fixed_r"]
        bk = h["beta_k_fixed_t"]
        print(
            f"  {k_name:26s} (N={h['n_turns']:4d}, items={h['n_items']:3d}): "
            f"bt={bt['pt']:+.3f} [{bt['ci95'][0]:+.2f},{bt['ci95'][1]:+.2f}] (OR={bt['or_pt']:.2f}) | "
            f"bt+bk={btr['pt']:+.3f} [{btr['ci95'][0]:+.2f},{btr['ci95'][1]:+.2f}] (OR={btr['or_pt']:.2f}) | "
            f"bk={bk['pt']:+.3f} [{bk['ci95'][0]:+.2f},{bk['ci95'][1]:+.2f}]"
        )

    print("\n[3] Token mechanism (pooled):")
    for pr, d in token_check["by_protocol"].items():
        print(f"  {pr:18s}: nonfinal/turn={d['mean_nonfinal_turn_comp']:6.1f}, final/turn={d['mean_final_turn_comp']:6.1f}, tot_comp={d['mean_traj_tot_comp']:7.1f}")
    for pr, scd in token_check["by_protocol_and_scope"].items():
        s_str = ", ".join(f"{sc}={v['mean_comp_tokens']:.0f} (n={v['n_turns']})" for sc, v in scd.items())
        print(f"  {pr:18s} by scope: {s_str}")

    print(f"\n[4] Exploratory CoT on changed TypeA commits: {n_cot_ref}/{n_changed} ({cot_check['pct_final_cot_refs_prior']}%) reference prior answer/derivation in final CoT")
    print("  by model:", by_model_cot)


if __name__ == "__main__":
    main()
