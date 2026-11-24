#!/usr/bin/env python3
"""Recompute post-v6 analyses (#9 item-clustered CIs, #10 hazard GLMM & top-5 drop,
#11 SymPy conflicting vs. equivalent early commits in TypeA FREE+REQ Y-J).

Usage (from repo root, bundle root, or with a bundle ZIP):
  python recompute_v7_analyses.py [PATH]
Writes v7_analyses.json next to this script.
"""

from __future__ import annotations

import gzip
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import sympy as sp
from numpy.polynomial.hermite import hermgauss
from scipy.optimize import minimize
from scipy.special import expit

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
if str(EVAL_DIR) not in sys.path:
    sys.path.insert(0, str(EVAL_DIR))
import sympy_answer_grader as sag  # noqa: E402


class BundleSrc:
    def __init__(self, p: Path):
        self.path = p
        self.zf = ZipFile(p) if p.is_file() and p.suffix.lower() == ".zip" else None
        if self.zf is None:
            self.names = [x.relative_to(p).as_posix() for x in sorted(p.rglob("*")) if x.is_file() and ".git" not in x.parts]
        else:
            self.names = self.zf.namelist()

    def close(self):
        if self.zf is not None:
            self.zf.close()

    def read_one(self, suffix: str) -> bytes:
        hits = [n for n in self.names if n.endswith(suffix) and not n.startswith("release/")]
        if not hits:
            hits = [n for n in self.names if n.endswith(suffix)]
        if not hits:
            raise FileNotFoundError(suffix)
        name = sorted(hits, key=len)[0]
        return self.zf.read(name) if self.zf else (self.path / name).read_bytes()

    def load_tracks(self) -> list[dict]:
        gz = [n for n in self.names if re.search(r"(^|/)judged_test100_.*_v13_blind\.jsonl\.gz$", n) and not n.startswith("release/")]
        raw = [n for n in self.names if re.search(r"(^|/)judged_test100_.*_v13_blind\.jsonl$", n) and not n.startswith("release/")]
        chosen = gz if len(gz) == 6 else raw
        if len(chosen) != 6:
            chosen = [n for n in self.names if re.search(r"(^|/)judged_test100_.*_v13_blind\.jsonl(\.gz)?$", n)]
        tracks = []
        for n in sorted(chosen):
            data = self.zf.read(n) if self.zf else (self.path / n).read_bytes()
            if n.endswith(".gz"):
                data = gzip.decompress(data)
            for line in data.splitlines():
                if line.strip():
                    tracks.append(json.loads(line))
        assert len(tracks) == 5400, f"Expected 5400 tracks, got {len(tracks)}"
        return tracks


def analysis_9_item_clustered_cis(tracks: list[dict], B: int = 10000, seed: int = 2026) -> dict:
    ta_fr = [t for t in tracks if t["verified_mechanism"] == "TypeA_UniqueValue" and t["protocol"] in ("DYN_FREE", "DYN_REQUEST")]
    assert len(ta_fr) == 1224
    ta_items = sorted({t["item_id"] for t in ta_fr})
    assert len(ta_items) == 68
    item_idx = {iid: i for i, iid in enumerate(ta_items)}
    cells = sorted({(t["model_id"], t["protocol"]) for t in ta_fr})
    cell_idx = {c: j for j, c in enumerate(cells)}

    eL = np.zeros((68, 6)); eU = np.zeros((68, 6))
    yjL = np.zeros((68, 6)); yjU = np.zeros((68, 6))
    for t in ta_fr:
        i = item_idx[t["item_id"]]; j = cell_idx[(t["model_id"], t["protocol"])]
        eL[i, j] += t["e_L"]; eU[i, j] += t["e_U"]
        yjL[i, j] += t["e_L"] * t["final_grading"]["y_L"]
        yjU[i, j] += t["e_U"] * t["final_grading"]["y_U"]

    rng = np.random.default_rng(seed)
    draws = rng.integers(0, 68, size=(B, 68))

    def summarize(mL, mU):
        pL = mL.sum(axis=1) / 18.0
        pU = mU.sum(axis=1) / 18.0
        bL = pL[draws].mean(axis=1) * 100.0
        bU = pU[draws].mean(axis=1) * 100.0
        loL, hiL = np.percentile(bL, [2.5, 97.5])
        loU, hiU = np.percentile(bU, [2.5, 97.5])
        per_cell = {}
        min_lo, max_hi = 100.0, 0.0
        for j, (m, pr) in enumerate(cells):
            cL = mL[:, j] / 3.0; cU = mU[:, j] / 3.0
            bcL = cL[draws].mean(axis=1) * 100.0
            bcU = cU[draws].mean(axis=1) * 100.0
            c_loL, c_hiL = np.percentile(bcL, [2.5, 97.5])
            c_loU, c_hiU = np.percentile(bcU, [2.5, 97.5])
            min_lo = min(min_lo, float(c_loL)); max_hi = max(max_hi, float(c_hiU))
            per_cell[f"{m.split('/')[-1]}:{pr}"] = {
                "L_pct": round(float(cL.mean() * 100.0), 1),
                "U_pct": round(float(cU.mean() * 100.0), 1),
                "ci95_L_pct": [round(float(c_loL), 1), round(float(c_hiL), 1)],
                "ci95_U_pct": [round(float(c_loU), 1), round(float(c_hiU), 1)],
            }
        return {
            "pooled_L_pct": round(float(pL.mean() * 100.0), 1),
            "pooled_U_pct": round(float(pU.mean() * 100.0), 1),
            "pooled_ci95_L_pct": [round(float(loL), 1), round(float(hiL), 1)],
            "pooled_ci95_U_pct": [round(float(loU), 1), round(float(hiU), 1)],
            "pooled_envelope_ci95_pct": [round(float(loL), 1), round(float(hiU), 1)],
            "cell_envelope_ci95_span_pct": [round(min_lo, 1), round(max_hi, 1)],
            "cells": per_cell,
        }

    return {"B": B, "seed": seed, "n_items": 68, "n_trajectories": 1224, "E": summarize(eL, eU), "Y_minus_J": summarize(yjL, yjU)}


def _build_X_y(rows):
    n = len(rows)
    X = np.zeros((n, 7)); y = np.zeros(n); iids = []
    for idx, (iid, mech, t, k, pi, mi, ev) in enumerate(rows):
        X[idx, 0] = 1.0; X[idx, 1] = t; X[idx, 2] = k
        if pi == 1: X[idx, 3] = 1.0
        elif pi == 2: X[idx, 4] = 1.0
        if mi == 1: X[idx, 5] = 1.0
        elif mi == 2: X[idx, 6] = 1.0
        y[idx] = ev; iids.append(iid)
    return X, y, iids


def _fit_logit_nr(X, y):
    b = np.zeros(X.shape[1])
    for _ in range(15):
        p = expit(X @ b)
        w = np.clip(p * (1.0 - p), 1e-9, None)
        H = (X.T * w) @ X + 1e-8 * np.eye(X.shape[1])
        step = np.linalg.solve(H, X.T @ (p - y))
        b -= step
        if np.max(np.abs(step)) < 1e-8:
            break
    return b


def analysis_10_hazard_robustness(tracks: list[dict], B: int = 2000, seed: int = 2026) -> dict:
    models_ord = ["Qwen/Qwen3-8B", "zai-org/GLM-Z1-9B-0414", "Qwen/Qwen3-14B"]
    protos_ord = ["DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"]
    risk_rows = []
    all_eL_by_item = Counter()
    for tr in tracks:
        if not tr["protocol"].startswith("DYN_"):
            continue
        if tr["e_L"] == 1:
            all_eL_by_item[tr["item_id"]] += 1
        for t in sorted([x for x in tr["turns"] if not x["is_final_turn"]], key=lambda x: x["turn_idx"]):
            ev = int(t["e_L"])
            risk_rows.append((tr["item_id"], tr["verified_mechanism"], int(t["turn_idx"]), int(tr["canonical_k"]),
                              protos_ord.index(tr["protocol"]), models_ord.index(tr["model_id"]), ev))
            if ev == 1:
                break

    top5_ta = {"ob-cee-4736", "ob-comp-370", "ob-comp-779", "ob-cee-4352", "ob-cee-4147"}
    top5_all = {iid for iid, _ in all_eL_by_item.most_common(5)}

    def boot_logit(rows):
        X, y, iids = _build_X_y(rows)
        b0 = _fit_logit_nr(X, y)
        by_it = defaultdict(list)
        for idx, iid in enumerate(iids):
            by_it[iid].append(idx)
        keys = sorted(by_it.keys())
        mats = [(X[by_it[k]], y[by_it[k]]) for k in keys]
        M = len(keys)
        rng = np.random.default_rng(seed)
        bt_a = np.empty(B); btk_a = np.empty(B)
        for bi in range(B):
            sel = rng.integers(0, M, size=M)
            bb = _fit_logit_nr(np.vstack([mats[s][0] for s in sel]), np.concatenate([mats[s][1] for s in sel]))
            bt_a[bi] = bb[1]; btk_a[bi] = bb[1] + bb[2]
        return {
            "n_risk_turns": len(rows), "n_items": M,
            "beta_t": round(float(b0[1]), 3),
            "OR_t": round(math.exp(float(b0[1])), 2),
            "OR_t_ci95": [round(math.exp(float(np.percentile(bt_a, 2.5))), 2), round(math.exp(float(np.percentile(bt_a, 97.5))), 2)],
            "beta_t_plus_k": round(float(b0[1] + b0[2]), 3),
            "OR_t_plus_k": round(math.exp(float(b0[1] + b0[2])), 2),
            "OR_t_plus_k_ci95": [round(math.exp(float(np.percentile(btk_a, 2.5))), 2), round(math.exp(float(np.percentile(btk_a, 97.5))), 2)],
        }

    def fit_glmm(rows):
        X, y, iids = _build_X_y(rows)
        by_it = defaultdict(list)
        for idx, iid in enumerate(iids):
            by_it[iid].append(idx)
        groups = [(X[idxs], y[idxs]) for idxs in by_it.values()]
        gh_x, gh_w = hermgauss(21)
        gh_z = math.sqrt(2.0) * gh_x
        gh_logw = np.log(gh_w / math.sqrt(math.pi))

        def nll_grad(params):
            beta = params[:-1]; sig = math.exp(params[-1])
            nll = 0.0; gb = np.zeros_like(beta); gls = 0.0
            for Xg, yg in groups:
                z = (Xg @ beta)[:, None] + sig * gh_z[None, :]
                ll_q = np.sum(yg[:, None] * z - np.logaddexp(0.0, z), axis=0) + gh_logw
                m = np.max(ll_q); w_q = np.exp(ll_q - m); Zg = np.sum(w_q); post = w_q / Zg
                nll -= (m + math.log(Zg))
                resid = yg[:, None] - expit(z)
                gb -= Xg.T @ (resid @ post)
                gls -= sig * np.sum((resid.sum(axis=0) * gh_z) * post)
            return nll, np.append(gb, gls)

        b0 = _fit_logit_nr(X, y)
        res = minimize(nll_grad, np.append(b0, math.log(1.5)), jac=True, method="BFGS")
        bg = res.x[:-1]; sig = math.exp(res.x[-1])
        icc = (sig ** 2) / (sig ** 2 + math.pi ** 2 / 3.0)
        eps = 1e-5; P = len(res.x); H = np.zeros((P, P))
        for k in range(P):
            e = np.zeros(P); e[k] = eps
            _, gp = nll_grad(res.x + e); _, gm = nll_grad(res.x - e)
            H[:, k] = (gp - gm) / (2 * eps)
        cov = np.linalg.inv(0.5 * (H + H.T))
        se_bt = math.sqrt(cov[1, 1])
        se_btk = math.sqrt(cov[1, 1] + cov[2, 2] + 2 * cov[1, 2])
        return {
            "n_risk_turns": len(rows), "n_items": len(groups), "converged": bool(res.success),
            "beta_t": round(float(bg[1]), 3), "se_beta_t": round(se_bt, 3),
            "OR_t": round(math.exp(float(bg[1])), 2),
            "OR_t_ci95": [round(math.exp(float(bg[1] - 1.96 * se_bt)), 2), round(math.exp(float(bg[1] + 1.96 * se_bt)), 2)],
            "beta_t_plus_k": round(float(bg[1] + bg[2]), 3), "se_beta_t_plus_k": round(se_btk, 3),
            "OR_t_plus_k": round(math.exp(float(bg[1] + bg[2])), 2),
            "OR_t_plus_k_ci95": [round(math.exp(float(bg[1] + bg[2] - 1.96 * se_btk)), 2), round(math.exp(float(bg[1] + bg[2] + 1.96 * se_btk)), 2)],
            "sigma_u": round(sig, 3), "ICC": round(icc, 3),
        }

    return {
        "drop_top5_TypeA_all100": boot_logit([r for r in risk_rows if r[0] not in top5_ta]),
        "drop_top5_all_dynamic": boot_logit([r for r in risk_rows if r[0] not in top5_all]),
        "TypeA_drop_top5_TypeA": boot_logit([r for r in risk_rows if r[1] == "TypeA_UniqueValue" and r[0] not in top5_ta]),
        "glmm_random_intercept_all100": fit_glmm(risk_rows),
        "glmm_random_intercept_TypeA68": fit_glmm([r for r in risk_rows if r[1] == "TypeA_UniqueValue"]),
    }


def analysis_11_sympy_conflicting_yj(tracks: list[dict], items: dict, certs: dict) -> dict:
    ta_fr = [t for t in tracks if t["verified_mechanism"] == "TypeA_UniqueValue" and t["protocol"] in ("DYN_FREE", "DYN_REQUEST")]

    def classify_box(raw_box: str, item_id: str) -> str:
        it = items[item_id]; cert = certs[item_id]
        s = re.split(r"\\quad|\\text\{或\}|或\s*\\approx|（|≈", raw_box.strip())[0].strip()
        if "无解" in s:
            return "CONFLICT_CONCRETE"
        g = sag.grade_final_answer_sympy(f"$\\boxed{{{s}}}$", it["final_answer"], cert)
        if g["status"] == "CORRECT":
            return "SYMPY_EQUIV"
        sym_map = {k: sp.Symbol(k) for k in cert.get("variables", {})}
        for ch in ["n", "x", "y", "a", "b", "c", "d", "k", "m", "p", "q", "r", "t", "C", "S", "alpha", "beta", "theta", "a_1", "a_n", "S_n", "S_1"]:
            sym_map.setdefault(ch, sp.Symbol(ch))
        fc = cert.get("full_answer_check", {})
        ref_str = str(fc.get("declared_value_sympy") or "").strip()
        try:
            ref_obj = sag.parse_symbolic(ref_str, sym_map) if ref_str else sag.parse_symbolic(sag.latex_to_sympy_candidate(it["final_answer"]), sym_map)
            ref_syms = {str(x) for x in ref_obj.free_symbols} if hasattr(ref_obj, "free_symbols") else set()
        except Exception:
            ref_obj, ref_syms = None, set()
        cand_s = s.split("=", 1)[1].strip() if ("=" in s and not re.search(r"[xy]\^?2", s)) else s
        try:
            pred_obj = sag.parse_symbolic(sag.latex_to_sympy_candidate(cand_s), sym_map)
            if pred_obj is not None and ref_obj is not None:
                if sp.simplify(pred_obj - ref_obj) == 0:
                    return "SYMPY_EQUIV"
                p_syms = {str(x) for x in pred_obj.free_symbols} if hasattr(pred_obj, "free_symbols") else set()
                return "PARAMETRIC" if (p_syms - ref_syms) else "CONFLICT_CONCRETE"
        except Exception:
            pass
        if g["status"] == "INCORRECT_EXACT":
            return "CONFLICT_CONCRETE"
        if re.search(r"\\alpha|\\beta|\\theta|a_1|a_n|S_n|S_1|f\(|\\log_2\s*r|\\sum|\\begin\{cases\}|\|k\|", s):
            return "PARAMETRIC"
        return "UNPARSED_BOX"

    out = {"n_typeA_free_req": len(ta_fr)}
    for side in ("L", "U"):
        yj = [t for t in ta_fr if t[f"e_{side}"] == 1 and t["final_grading"][f"y_{side}"] == 1]
        c = Counter(); c_unret = Counter()
        unparsed_examples = []
        for tr in yj:
            early = next(t for t in tr["turns"] if not t["is_final_turn"] and t[f"e_{side}"] == 1)
            boxes = sag.extract_boxed_answers(early.get("visible_reply") or "")
            rep = tr.get("repair", {})
            retracted = bool(rep.get("primary_retracted") or rep.get("secondary_retracted"))
            if not boxes:
                cat = "UNBOXED"
            else:
                cat = classify_box(boxes[-1], tr["item_id"])
                if cat == "UNPARSED_BOX":
                    unparsed_examples.append((tr["item_id"], boxes[-1], items[tr["item_id"]]["final_answer"]))
            c[cat] += 1
            if not retracted:
                c_unret[cat] += 1
        n_boxed = len(yj) - c["UNBOXED"]
        out[f"side_{side}"] = {
            "total_YJ": len(yj),
            "total_YJ_pct_of_1224": round(100.0 * len(yj) / len(ta_fr), 2),
            "boxed_total": n_boxed,
            "unboxed_total": c["UNBOXED"],
            "boxed_sympy_equiv": c["SYMPY_EQUIV"],
            "boxed_sympy_equiv_pct_of_boxed": round(100.0 * c["SYMPY_EQUIV"] / n_boxed, 1),
            "boxed_sympy_equiv_pct_of_YJ": round(100.0 * c["SYMPY_EQUIV"] / len(yj), 1),
            "boxed_sympy_equiv_pct_of_1224": round(100.0 * c["SYMPY_EQUIV"] / len(ta_fr), 2),
            "boxed_conflict_concrete": c["CONFLICT_CONCRETE"],
            "boxed_conflict_concrete_unretracted_screen": c_unret["CONFLICT_CONCRETE"],
            "boxed_conflict_concrete_pct_of_boxed": round(100.0 * c["CONFLICT_CONCRETE"] / n_boxed, 1),
            "boxed_conflict_concrete_pct_of_YJ": round(100.0 * c["CONFLICT_CONCRETE"] / len(yj), 1),
            "boxed_conflict_concrete_pct_of_1224": round(100.0 * c["CONFLICT_CONCRETE"] / len(ta_fr), 2),
            "boxed_conflict_concrete_unretracted_pct_of_1224": round(100.0 * c_unret["CONFLICT_CONCRETE"] / len(ta_fr), 2),
            "boxed_parametric": c["PARAMETRIC"],
            "boxed_parametric_pct_of_boxed": round(100.0 * c["PARAMETRIC"] / n_boxed, 1),
            "boxed_unparsed": c["UNPARSED_BOX"],
            "boxed_unparsed_examples": unparsed_examples,
        }
    return out


def main() -> None:
    src_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE.parents[1]
    src = BundleSrc(src_path)
    try:
        tracks = src.load_tracks()
        items = {json.loads(l)["item_id"]: json.loads(l) for l in src.read_one("frozen_test_100.jsonl").decode("utf-8").splitlines() if l.strip()}
        certs = {json.loads(l)["item_id"]: json.loads(l) for l in src.read_one("frozen_certificates.jsonl").decode("utf-8").splitlines() if l.strip()}
    finally:
        src.close()

    result = {
        "analysis_9_item_clustered_cis": analysis_9_item_clustered_cis(tracks),
        "analysis_10_hazard_robustness": analysis_10_hazard_robustness(tracks),
        "analysis_11_sympy_conflicting_yj": analysis_11_sympy_conflicting_yj(tracks, items, certs),
    }
    (HERE / "v7_analyses.json").write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
