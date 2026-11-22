#!/usr/bin/env python3
"""Deterministic SymPy audit of Final Y across all 5,400 TEST 100 trajectories.

Cross-checks every final visible reply against Stage-2B SymPy reference certificates
(`frozen_certificates.jsonl` + `frozen_test_100.jsonl`) using `sympy_answer_grader.py`.
Writes `sympy_final_y_audit.json` alongside this script.
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
EVAL_DIR = SCRIPT_DIR.parent
FROZEN_DIR = EVAL_DIR.parent / "exp1_codex_gpt61_v1" / "frozen_release_verified_120"
RUNS_DIR = EVAL_DIR / "runs" / "runs_16k_v13_blind"

sys.path.insert(0, str(EVAL_DIR))
from sympy_answer_grader import grade_final_answer_sympy  # noqa: E402


def main() -> None:
    items = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in (FROZEN_DIR / "frozen_test_100.jsonl").open(encoding="utf-8")
    }
    certs = {
        json.loads(line)["item_id"]: json.loads(line)
        for line in (FROZEN_DIR / "frozen_certificates.jsonl").open(encoding="utf-8")
    }

    files = sorted(
        f for f in RUNS_DIR.glob("judged_test100_*_v13_blind.jsonl") if not f.name.endswith("_judge_cache.jsonl")
    )

    overall_status = Counter()
    by_model: dict[str, Counter] = defaultdict(Counter)
    disagreements = []
    llm_uncertain_cases = []
    max_total_turn_ctx = 0
    max_prompt_tokens = 0

    for jf in files:
        with jf.open(encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                mid = r["model_id"]
                iid = r["item_id"]
                fg = r["final_grading"]
                y_l, y_u = int(fg["y_L"]), int(fg["y_U"])
                emp = bool(fg.get("empty_trunc"))
                vis = r.get("final_visible_reply") or ""

                for t in r["turns"]:
                    pt = int(t.get("prompt_tokens") or 0)
                    ct = int(t.get("completion_tokens") or 0)
                    if pt > max_prompt_tokens:
                        max_prompt_tokens = pt
                    if pt + ct > max_total_turn_ctx:
                        max_total_turn_ctx = pt + ct

                res = grade_final_answer_sympy(vis, items[iid]["final_answer"], certs[iid], empty_truncated=emp)
                st = res["status"]
                overall_status[st] += 1
                by_model[mid]["total"] += 1
                by_model[mid][f"status_{st}"] += 1

                sy = 1 if st == "CORRECT" else (0 if st in ("INCORRECT_EXACT", "INCORRECT_ABSTAIN") else None)
                if sy is not None:
                    by_model[mid]["sympy_decided"] += 1
                    if y_l == y_u:
                        by_model[mid]["decided_both_llm_agreed"] += 1
                        if sy == y_l:
                            by_model[mid]["exact_match"] += 1
                        else:
                            by_model[mid]["mismatch"] += 1
                            disagreements.append(
                                {
                                    "model_id": mid,
                                    "trajectory_id": r["trajectory_id"],
                                    "item_id": iid,
                                    "protocol": r["protocol"],
                                    "sympy_status": st,
                                    "sympy_y": sy,
                                    "llm_y": y_l,
                                    "extracted": res.get("extracted"),
                                    "gold": items[iid]["final_answer"],
                                    "reason": res.get("reason"),
                                    "primary_rationale": fg["primary_llm"].get("rationale", ""),
                                    "secondary_rationale": fg["secondary_llm"].get("rationale", ""),
                                }
                            )
                    else:
                        by_model[mid]["decided_llm_disagreed"] += 1
                        llm_uncertain_cases.append(
                            {
                                "model_id": mid,
                                "trajectory_id": r["trajectory_id"],
                                "sympy_status": st,
                                "sympy_y": sy,
                                "primary_y": int(bool(fg["primary_llm"].get("correct"))),
                                "secondary_y": int(bool(fg["secondary_llm"].get("correct"))),
                                "extracted": res.get("extracted"),
                                "gold": items[iid]["final_answer"],
                            }
                        )

    total_n = sum(c["total"] for c in by_model.values())
    total_decided = sum(c["sympy_decided"] for c in by_model.values())
    total_agreed_den = sum(c["decided_both_llm_agreed"] for c in by_model.values())
    total_match = sum(c["exact_match"] for c in by_model.values())
    total_mismatch = sum(c["mismatch"] for c in by_model.values())
    total_llm_dis = sum(c["decided_llm_disagreed"] for c in by_model.values())

    summary = {
        "total_trajectories": total_n,
        "max_prompt_tokens_single_turn": max_prompt_tokens,
        "max_prompt_plus_completion_single_turn": max_total_turn_ctx,
        "sympy_status_counts": dict(overall_status),
        "sympy_decided_total": total_decided,
        "sympy_decided_share": round(total_decided / total_n, 4),
        "on_dual_llm_consensus": {
            "n": total_agreed_den,
            "exact_match": total_match,
            "mismatch": total_mismatch,
            "agreement_rate": round(total_match / total_agreed_den, 6),
        },
        "on_dual_llm_disagreed": {
            "n": total_llm_dis,
            "matches_primary": sum(1 for x in llm_uncertain_cases if x["sympy_y"] == x["primary_y"]),
            "matches_secondary": sum(1 for x in llm_uncertain_cases if x["sympy_y"] == x["secondary_y"]),
            "cases": llm_uncertain_cases,
        },
        "by_model": {m: dict(c) for m, c in sorted(by_model.items())},
        "mismatches": disagreements,
    }

    out_path = SCRIPT_DIR / "sympy_final_y_audit.json"
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Total: {total_n} | Decided: {total_decided} ({total_decided/total_n:.2%})")
    print(f"On Dual-LLM consensus: {total_match}/{total_agreed_den} ({total_match/total_agreed_den:.4%}), mismatches={total_mismatch}")
    print(f"On Dual-LLM disagreed ({total_llm_dis}): matches primary={summary['on_dual_llm_disagreed']['matches_primary']}, secondary={summary['on_dual_llm_disagreed']['matches_secondary']}")
    print(f"Max prompt_tokens={max_prompt_tokens}, max(prompt+completion)={max_total_turn_ctx}")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
