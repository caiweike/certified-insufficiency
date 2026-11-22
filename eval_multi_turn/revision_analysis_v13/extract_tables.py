#!/usr/bin/env python3
"""Build the flat analysis tables used by paper_revised/regenerate_v13_assets.py.

Reads  runs/runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl  (run unpack_data.py first)
Writes revision_analysis_v13/traj.csv   one row per trajectory (5,400)
       revision_analysis_v13/turns.csv  one row per non-final turn of the multi-turn protocols (7,668)
Rows of turns.csv follow the trajectory order of traj.csv (the fixed file order below, then line order); downstream code
relies on this order to attach the model, because trajectory_id is not unique across models.
"""
import csv, json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RUNS = HERE.parent / "runs" / "runs_16k_v13_blind"


def main() -> None:
    order = ["qwen3_8b", "qwen3_8b_guard", "glm_z1_9b_0414", "glm_z1_9b_0414_guard", "qwen3_14b", "qwen3_14b_guard"]
    files = [RUNS / f"judged_test100_{m}_16384_v13_blind.jsonl" for m in order]
    missing = [f.name for f in files if not f.exists()]
    if missing:
        raise SystemExit(f"Missing {missing} in {RUNS}; run unpack_data.py first.")
    traj_rows, turn_rows = [], []
    for f in files:
        with f.open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                fg = r.get("final_grading") or {}
                rp = r.get("repair") or {}
                pl = fg.get("primary_llm") or {}
                sl = fg.get("secondary_llm") or {}
                turns = r["turns"]
                traj_rows.append(dict(
                    traj=r["trajectory_id"], model=r["model_id"].split("/")[-1], protocol=r["protocol"],
                    item=r["item_id"], rep=r["rep"], mech=r["verified_mechanism"][:5], k=r["canonical_k"],
                    g=r["macro_subfield"], y_L=fg.get("y_L"), y_U=fg.get("y_U"),
                    y_pri=int(bool(pl.get("correct"))), y_sec=int(bool(sl.get("correct"))),
                    ab_L=fg.get("explicit_abstain_L"), e_L=r.get("e_L"), e_U=r.get("e_U"),
                    j_L=r.get("j_L"), j_U=r.get("j_U"),
                    rep_pri=int(bool(rp.get("primary_retracted"))) if rp else "",
                    rep_sec=int(bool(rp.get("secondary_retracted"))) if rp else "",
                    rep_type_pri=rp.get("primary_repair_type", "") if rp else "",
                    rep_type_sec=rp.get("secondary_repair_type", "") if rp else "",
                    tot_prompt=r.get("total_prompt_tokens"), tot_comp=r.get("total_completion_tokens"),
                    max_turn_prompt=max((t.get("prompt_tokens") or 0) for t in turns),
                    max_turn_ctx=max((t.get("prompt_tokens") or 0) + (t.get("completion_tokens") or 0) for t in turns),
                    hist_has_think=int(any("<think>" in (t.get("history_assistant_text") or "") for t in turns)),
                ))
                for t in turns:
                    if t.get("is_final_turn"):
                        continue
                    turn_rows.append(dict(traj=r["trajectory_id"], t=t["turn_idx"], scope=t.get("scope_C"),
                                          pri=t.get("primary_scope"), sec=t.get("secondary_scope"),
                                          e_L=t.get("e_L"), e_U=t.get("e_U")))
    for name, rows in [("traj.csv", traj_rows), ("turns.csv", turn_rows)]:
        with (HERE / name).open("w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {name}: {len(rows)} rows")


if __name__ == "__main__":
    main()
