#!/usr/bin/env python3
"""Build 100% protocol-blind & judge-blind human audit sheets for TEST 100.

Generates:
  1. human_audit_scope_60.csv & human_audit_scope_60.md
     60 stratified non-final insufficient turns (20 agreed Commit, 16 agreed Wait,
     12 agreed Conditional, 6 Commit-disagreed boundary, 6 Wait-vs-Conditional boundary),
     strictly hiding protocol, model, mechanism, and both AI judge verdicts.
  2. human_audit_repair_26.csv & human_audit_repair_26.md
     ALL trajectories where at least one judge marked Explicit Repair positive
     (4 unanimously positive + 22 single-judge positive), strictly hiding model/protocol/verdicts.
  3. _audit_answer_key_DO_NOT_OPEN.json
     Locked key read only by score_human_audit.py after human annotation completes.
"""

from __future__ import annotations

import csv
import json
import random
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
RUNS_DIR = ROOT / "eval_multi_turn" / "runs" / "runs_16k_v13_blind"
FROZEN_TEST = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_test_100.jsonl"
OUT_DIR = Path(__file__).resolve().parent

JUDGED_FILES = [
    RUNS_DIR / "judged_test100_qwen3_8b_16384_v13_blind.jsonl",
    RUNS_DIR / "judged_test100_qwen3_8b_guard_16384_v13_blind.jsonl",
    RUNS_DIR / "judged_test100_glm_z1_9b_0414_16384_v13_blind.jsonl",
    RUNS_DIR / "judged_test100_glm_z1_9b_0414_guard_16384_v13_blind.jsonl",
    RUNS_DIR / "judged_test100_qwen3_14b_16384_v13_blind.jsonl",
    RUNS_DIR / "judged_test100_qwen3_14b_guard_16384_v13_blind.jsonl",
]

SEED = 2026


def load_shard_texts() -> dict[str, dict[str, str]]:
    """Map item_id -> {shard_id: raw_clause_text} without protocol instructions."""
    out: dict[str, dict[str, str]] = {}
    with FROZEN_TEST.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            iid = row["item_id"]
            shards = {}
            for s in row.get("shards", []):
                shards[s["shard_id"]] = (s.get("shard") or s.get("text") or "").strip()
            out[iid] = shards
    return out


def format_disclosed_clauses(item_shards: dict[str, str], disclosure_order: list[str], up_to_turn_idx: int) -> str:
    """Reconstruct exact clauses disclosed up to turn_idx (0-based), zero protocol leakage."""
    parts = []
    for k in range(up_to_turn_idx + 1):
        sid = disclosure_order[k]
        txt = item_shards.get(sid, f"[{sid}]")
        parts.append(f"[第{k + 1}轮披露/{sid}] {txt}")
    return "\n".join(parts)


def stratified_pick(pool: list[dict], n: int, rng: random.Random) -> list[dict]:
    """Sample n entries prioritizing balanced (model_id, protocol) coverage & unique item_ids."""
    shuffled = list(pool)
    rng.shuffle(shuffled)
    picked: list[dict] = []
    used_items: set[str] = set()
    used_keys: set[tuple] = set()

    # Pass 1: round-robin by (model_id, protocol) with unique items
    buckets: dict[tuple[str, str], list[dict]] = {}
    for x in shuffled:
        buckets.setdefault((x["model_id"], x["protocol"]), []).append(x)
    bkeys = sorted(buckets.keys())
    rng.shuffle(bkeys)

    progressed = True
    while len(picked) < n and progressed:
        progressed = False
        for bk in bkeys:
            if len(picked) >= n:
                break
            b = buckets[bk]
            while b:
                cand = b.pop()
                if cand["item_id"] not in used_items:
                    picked.append(cand)
                    used_items.add(cand["item_id"])
                    used_keys.add(cand["unique_key"])
                    progressed = True
                    break

    # Pass 2: fill remainder if unique items exhausted
    if len(picked) < n:
        for cand in shuffled:
            if len(picked) >= n:
                break
            if cand["unique_key"] not in used_keys:
                picked.append(cand)
                used_keys.add(cand["unique_key"])
    return picked


def main() -> None:
    rng = random.Random(SEED)
    shard_map = load_shard_texts()

    pools: dict[str, list[dict]] = {
        "commit_A": [],
        "commit_B": [],
        "wait_A": [],
        "wait_B": [],
        "cond_A": [],
        "cond_B": [],
        "disagree_commit": [],
        "disagree_wait_cond": [],
    }
    repair_cases: list[dict] = []

    for jf in JUDGED_FILES:
        with jf.open("r", encoding="utf-8") as f:
            for line in f:
                traj = json.loads(line)
                proto = traj["protocol"]
                if not proto.startswith("DYN_"):
                    continue
                iid = traj["item_id"]
                mech = traj["verified_mechanism"]
                order = traj["disclosure_order"]
                ishards = shard_map[iid]

                # Collect non-final scope turns
                first_commit_turn = None
                for t in traj["turns"]:
                    if t.get("is_final_turn"):
                        continue
                    if t.get("empty_trunc") or t.get("non_trunc_empty"):
                        continue
                    if t.get("e_U") == 1 and first_commit_turn is None:
                        first_commit_turn = t
                    p_sc = t.get("primary_scope")
                    s_sc = t.get("secondary_scope")
                    if not p_sc or not s_sc:
                        continue
                    t_idx = int(t["turn_idx"])
                    rec = {
                        "unique_key": (traj["model_id"], traj["trajectory_id"], t_idx),
                        "trajectory_id": traj["trajectory_id"],
                        "item_id": iid,
                        "model_id": traj["model_id"],
                        "protocol": proto,
                        "rep": traj["rep"],
                        "verified_mechanism": mech,
                        "canonical_k": traj["canonical_k"],
                        "turn_idx": t_idx,
                        "turn_number": t_idx,  # turn_idx is 1-based
                        "disclosed_clauses": format_disclosed_clauses(ishards, order, t_idx - 1),  # up_to index is 0-based
                        "visible_reply": (t.get("visible_reply") or "").strip(),
                        "primary_scope": p_sc,
                        "secondary_scope": s_sc,
                        "consensus_scope": t.get("scope_C"),
                        "e_L": int(t.get("e_L", 0)),
                        "e_U": int(t.get("e_U", 0)),
                        "primary_rationale": t.get("scope_rationale", ""),
                        "secondary_rationale": t.get("secondary_scope_rationale", ""),
                    }
                    is_A = mech == "TypeA_UniqueValue"
                    if p_sc == "Commit" and s_sc == "Commit":
                        pools["commit_A" if is_A else "commit_B"].append(rec)
                    elif p_sc == "Wait" and s_sc == "Wait":
                        pools["wait_A" if is_A else "wait_B"].append(rec)
                    elif p_sc == "Conditional" and s_sc == "Conditional":
                        pools["cond_A" if is_A else "cond_B"].append(rec)
                    elif (p_sc == "Commit") != (s_sc == "Commit"):
                        pools["disagree_commit"].append(rec)
                    elif {p_sc, s_sc} == {"Wait", "Conditional"}:
                        pools["disagree_wait_cond"].append(rec)

                # Collect ALL repair-positive/disagreed trajectories (v1.3 labels)
                rep_obj = traj.get("repair")
                if rep_obj and (rep_obj.get("primary_retracted") or rep_obj.get("secondary_retracted")):
                    ft = traj["turns"][-1]
                    all_clauses = format_disclosed_clauses(ishards, order, len(order) - 1)
                    commit_turn_num = first_commit_turn["turn_idx"] if first_commit_turn else -1
                    commit_reply = (first_commit_turn.get("visible_reply") or "").strip() if first_commit_turn else ""
                    repair_cases.append(
                        {
                            "trajectory_id": traj["trajectory_id"],
                            "item_id": iid,
                            "model_id": traj["model_id"],
                            "protocol": proto,
                            "rep": traj["rep"],
                            "verified_mechanism": mech,
                            "canonical_k": traj["canonical_k"],
                            "all_clauses": all_clauses,
                            "earlier_commit_turn": commit_turn_num,
                            "earlier_commit_reply": commit_reply,
                            "final_visible_reply": (ft.get("visible_reply") or "").strip(),
                            "primary_retracted": bool(rep_obj["primary_retracted"]),
                            "secondary_retracted": bool(rep_obj["secondary_retracted"]),
                            "primary_repair_type": rep_obj.get("primary_repair_type", ""),
                            "secondary_repair_type": rep_obj.get("secondary_repair_type", ""),
                            "primary_rationale": rep_obj.get("rationale", ""),
                            "secondary_rationale": rep_obj.get("secondary_rationale", ""),
                        }
                    )

    targets = [
        ("commit_A", 10),
        ("commit_B", 10),
        ("wait_A", 10),
        ("wait_B", 6),
        ("cond_A", 8),
        ("cond_B", 4),
        ("disagree_commit", 6),
        ("disagree_wait_cond", 6),
    ]
    selected_scope: list[dict] = []
    for stratum, cnt in targets:
        picked = stratified_pick(pools[stratum], cnt, rng)
        for p in picked:
            p["stratum"] = stratum
        selected_scope.extend(picked)

    rng.shuffle(selected_scope)
    rng.shuffle(repair_cases)

    scope_key: dict[str, dict] = {}
    scope_csv = OUT_DIR / "human_audit_scope_60.csv"
    scope_md = OUT_DIR / "human_audit_scope_60.md"

    with scope_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "audit_id",
                "turn_number",
                "disclosed_clauses_so_far",
                "model_visible_reply",
                "human_label",
                "human_notes",
            ]
        )
        for idx, rec in enumerate(selected_scope, 1):
            aid = f"S{idx:02d}"
            w.writerow([aid, rec["turn_number"], rec["disclosed_clauses"], rec["visible_reply"], "", ""])
            scope_key[aid] = {k: v for k, v in rec.items() if k not in ("unique_key", "disclosed_clauses", "visible_reply")}

    md_lines = [
        "# TEST 100 双盲人工复核手册 — 非末轮作用域（60 轮，`S01`–`S60`）",
        "",
        "> **严格盲审规则**：本手册完全隐藏了协议名称、模型名称、题目类型（TypeA/TypeB）及两位 AI 裁判的打分结果。",
        "> 请根据**“截至本轮已披露的条件”**与**“模型可见回复”**，在 `human_audit_scope_60.csv` 的 `human_label` 列填入以下 4 个标签之一：",
        "> - `Wait`：明确指出条件不足 / 需要补充条件，未给出确定答案，也未把具体答案绑在假设上。",
        "> - `Conditional`：给出候选答案或分情况讨论，但**明确将其限定在未确认假设或“仅在没有更多附加条件的前提下”**，未当作完整题的最终定论。",
        "> - `Commit`：把某个具体答案/范围/结论当作最终确定答案给出（例如直接 `\\boxed{...}` 定论、暗自补条件或未声明作用域限定）。",
        "> - `Partial/Other`：仅做部分化简或中间推导，既未明确要求补条件也未给出任何答案或候选结论。",
        "",
        "---",
        "",
    ]
    for idx, rec in enumerate(selected_scope, 1):
        aid = f"S{idx:02d}"
        md_lines.extend(
            [
                f"## {aid}（第 {rec['turn_number']} 轮，当前信息尚不完整）",
                "",
                "**截至本轮已披露条件：**",
                "```text",
                rec["disclosed_clauses"],
                "```",
                "",
                "**模型本轮可见回复：**",
                "",
                rec["visible_reply"],
                "",
                "**人工标注（填进 CSV）：** `Wait` / `Conditional` / `Commit` / `Partial/Other`",
                "",
                "---",
                "",
            ]
        )
    scope_md.write_text("\n".join(md_lines), encoding="utf-8")

    # Repair sheet (all repair-flagged trajectories)
    repair_key: dict[str, dict] = {}
    repair_csv = OUT_DIR / f"human_audit_repair_{len(repair_cases)}.csv"
    repair_md = OUT_DIR / f"human_audit_repair_{len(repair_cases)}.md"

    with repair_csv.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(
            [
                "audit_id",
                "earlier_commit_turn",
                "earlier_commit_reply",
                "final_visible_reply",
                "human_retracted",
                "human_notes",
            ]
        )
        for idx, rec in enumerate(repair_cases, 1):
            aid = f"R{idx:02d}"
            w.writerow([aid, rec["earlier_commit_turn"], rec["earlier_commit_reply"], rec["final_visible_reply"], "", ""])
            repair_key[aid] = {
                k: v for k, v in rec.items() if k not in ("all_clauses", "earlier_commit_reply", "final_visible_reply")
            }

    rmd = [
        f"# TEST 100 双盲人工复核手册 — 末轮显式撤回（全量 {len(repair_cases)} 例，`R01`–`R{len(repair_cases):02d}`）",
        "",
        f"> **覆盖范围**：包含全部 5,400 条轨迹中至少有一位 AI 裁判判为“主动撤回”的**全部 {len(repair_cases)} 例**（完全隐藏模型、协议与 AI 结论）。",
        f"> 请在 `human_audit_repair_{len(repair_cases)}.csv` 的 `human_retracted` 列填入：",
        "> - `Yes`：末轮回复**明确提及并撤回/修正**了早先轮次给出的错误定论或过度自信结论。",
        "> - `No`：末轮只是直接基于完整条件重新算出答案，**没有明确指出或撤回**早先的抢答结论。",
        "",
        "---",
        "",
    ]
    for idx, rec in enumerate(repair_cases, 1):
        aid = f"R{idx:02d}"
        rmd.extend(
            [
                f"## {aid}（早先抢答发生于第 {rec['earlier_commit_turn']} 轮）",
                "",
                "**完整题目条件：**",
                "```text",
                rec["all_clauses"],
                "```",
                "",
                f"**早先第 {rec['earlier_commit_turn']} 轮模型可见回复：**",
                "",
                rec["earlier_commit_reply"],
                "",
                "**末轮模型可见回复：**",
                "",
                rec["final_visible_reply"],
                "",
                "**人工标注（填进 CSV）：** `Yes` / `No`",
                "",
                "---",
                "",
            ]
        )
    repair_md.write_text("\n".join(rmd), encoding="utf-8")

    key_path = OUT_DIR / "_audit_answer_key_DO_NOT_OPEN.json"
    key_payload = {
        "seed": SEED,
        "scope_counts": {k: len(v) for k, v in pools.items()},
        "scope_selected_n": len(scope_key),
        "repair_selected_n": len(repair_key),
        "scope": scope_key,
        "repair": repair_key,
    }
    key_path.write_text(json.dumps(key_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Generated Scope blind sheet:  {scope_csv} ({len(scope_key)} turns)")
    print(f"Generated Scope blind guide:  {scope_md}")
    print(f"Generated Repair blind sheet: {repair_csv} ({len(repair_key)} cases)")
    print(f"Generated Repair blind guide: {repair_md}")
    print(f"Locked answer key saved to:   {key_path}")


if __name__ == "__main__":
    main()
