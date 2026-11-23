#!/usr/bin/env python3
"""Build the pre-registered Unified Neutral No-Adjudication [L, U] Audit Package (SEED=2026).

Modules (all 100% neutral framing, 0 prefilled labels, 0 judge/model/protocol/regime leaks):
  1. Cert-100   : Full census of all 100 TEST items (68 TypeA + 32 TypeB; 852 evaluated prefixes).
  2. Scope-120  : Exhaustive 9-stratum stratified sample of 120 non-final turns (all 7,668 turns).
  3. Repair-100 : Unified blind mixed pool of all 26 judge-flagged candidates + 74 non-empty
                  dual-negative controls (from 457 non-empty dual-negatives; 6 empty = 0/6).

Reads (relative to repository root; supports both .jsonl and .jsonl.gz):
  - exp1_codex_gpt61_v1/frozen_release_verified_120/frozen_test_100.jsonl
  - exp1_codex_gpt61_v1/frozen_release_verified_120/frozen_certificates.jsonl
  - exp1_codex_gpt61_v1/frozen_release_verified_120/frozen_reports.jsonl
  - eval_multi_turn/disclosure_plan_test100.jsonl
  - eval_multi_turn/runs/runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl[.gz]

Writes (in this directory):
  - unified_audit_tool_seed2026.html
  - _unified_audit_key_DO_NOT_OPEN.json
  - cert_100_template.csv
  - scope_120_template.csv
  - repair_100_template.csv
"""

from __future__ import annotations

import csv
import gzip
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
ROOT = EVAL_DIR.parent

FROZEN_TEST = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_test_100.jsonl"
FROZEN_CERTS = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_certificates.jsonl"
FROZEN_REPORTS = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_reports.jsonl"
DISC_PLAN = EVAL_DIR / "disclosure_plan_test100.jsonl"
V13_DIR = EVAL_DIR / "runs" / "runs_16k_v13_blind"

SEED = 2026
DYN_PROTOS = {"DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"}
JUDGED_SLUGS = [
    "judged_test100_qwen3_8b_16384_v13_blind",
    "judged_test100_qwen3_8b_guard_16384_v13_blind",
    "judged_test100_glm_z1_9b_0414_16384_v13_blind",
    "judged_test100_glm_z1_9b_0414_guard_16384_v13_blind",
    "judged_test100_qwen3_14b_16384_v13_blind",
    "judged_test100_qwen3_14b_guard_16384_v13_blind",
]

# Exhaustive 9-stratum allocation for Scope-120 (100% coverage of all 7,668 non-final turns)
SCOPE_QUOTAS = {
    "agree_Commit_TypeA": 18,
    "agree_Commit_TypeB": 14,
    "disagree_Commit": 16,
    "agree_Wait_TypeA": 18,
    "agree_Wait_TypeB": 12,
    "agree_Cond_TypeA": 14,
    "agree_Cond_TypeB": 10,
    "disagree_Wait_Cond": 12,
    "other_noncommit": 6,
}

# Balanced 3-model allocation for 74 non-empty dual-negative Repair controls
REPAIR_NEG_MODEL_QUOTAS = {
    "Qwen/Qwen3-8B": 25,
    "zai-org/GLM-Z1-9B-0414": 25,
    "Qwen/Qwen3-14B": 24,
}


def read_jsonl_auto(path_no_gz: Path) -> list[dict]:
    if path_no_gz.exists():
        return [json.loads(l) for l in path_no_gz.read_text(encoding="utf-8").splitlines() if l.strip()]
    gz = Path(str(path_no_gz) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    raise FileNotFoundError(f"Neither {path_no_gz} nor {gz} exists.")


def format_clauses(shards_map: dict[str, str], order: list[str], up_to_k: int) -> str:
    return "\n".join(f"[第{i + 1}条/{order[i]}] {shards_map.get(order[i], order[i])}" for i in range(up_to_k))


def scope_stratum(pri: str, sec: str, mech_short: str) -> str:
    if pri == "Commit" and sec == "Commit":
        return f"agree_Commit_{mech_short}"
    if (pri == "Commit") != (sec == "Commit"):
        return "disagree_Commit"
    if pri == "Wait" and sec == "Wait":
        return f"agree_Wait_{mech_short}"
    if pri == "Conditional" and sec == "Conditional":
        return f"agree_Cond_{mech_short}"
    if {pri, sec} == {"Wait", "Conditional"}:
        return "disagree_Wait_Cond"
    return "other_noncommit"


def balanced_pick(pool: list[dict], n: int, rng: random.Random, group_key_fn) -> list[dict]:
    """Stratified round-robin pick maximizing unique items and balancing group_key_fn."""
    shuffled = sorted(pool, key=lambda x: x["unique_key"])
    rng.shuffle(shuffled)
    buckets: dict = defaultdict(list)
    for x in shuffled:
        buckets[group_key_fn(x)].append(x)
    bkeys = sorted(buckets.keys())
    rng.shuffle(bkeys)

    picked: list[dict] = []
    used_items: set[str] = set()
    used_keys: set[tuple] = set()

    # Pass 1: unique items across balanced buckets
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

    # Pass 2: if unique items exhausted, still balance buckets over remaining unused turns
    if len(picked) < n:
        rem_buckets: dict = defaultdict(list)
        for x in shuffled:
            if x["unique_key"] not in used_keys:
                rem_buckets[group_key_fn(x)].append(x)
        rkeys = sorted(rem_buckets.keys())
        rng.shuffle(rkeys)
        progressed = True
        while len(picked) < n and progressed:
            progressed = False
            for bk in rkeys:
                if len(picked) >= n:
                    break
                b = rem_buckets[bk]
                if b:
                    cand = b.pop()
                    picked.append(cand)
                    used_keys.add(cand["unique_key"])
                    progressed = True
    assert len(picked) == n, f"Requested {n}, could only pick {len(picked)}"
    return picked


def main() -> None:
    rng_cert = random.Random(SEED)
    rng_scope = random.Random(SEED + 1)
    rng_repair = random.Random(SEED + 2)

    # Load frozen benchmark + certificates + disclosure orders
    test_items = read_jsonl_auto(FROZEN_TEST)
    assert len(test_items) == 100, f"Expected 100 TEST items, got {len(test_items)}"
    certs_by_id = {r["item_id"]: r for r in read_jsonl_auto(FROZEN_CERTS)}
    reps_by_id = {r["item_id"]: r for r in read_jsonl_auto(FROZEN_REPORTS)}

    shards_by_item: dict[str, dict[str, str]] = {}
    for it in test_items:
        shards_by_item[it["item_id"]] = {
            s["shard_id"]: (s.get("shard") or s.get("text") or "").strip()
            for s in it.get("shards", [])
        }

    orders_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl_auto(DISC_PLAN):
        orders_by_id[row["item_id"]].append({
            "rep": int(row["rep"]),
            "order": list(row["disclosure_order"]),
        })
    for iid in orders_by_id:
        orders_by_id[iid].sort(key=lambda x: x["rep"])

    # ================= MODULE 1: CERT-100 =================
    cert_order = sorted(test_items, key=lambda r: r["item_id"])
    rng_cert.shuffle(cert_order)

    cert_ui = []
    cert_key = {}
    tot_pref_A = tot_pref_B = 0
    for idx, it in enumerate(cert_order, 1):
        cid = f"C{idx:03d}"
        iid = it["item_id"]
        mech_short = "TypeA" if it["verified_mechanism"] == "TypeA_UniqueValue" else "TypeB"
        cert = certs_by_id[iid]
        rep = reps_by_id[iid]
        ords = orders_by_id[iid]

        eval_orders = []
        n_pref = 0
        for o in ords:
            seq = o["order"]
            prefs = [
                {"turn": t, "prefix": seq[:t], "missing": [x for x in seq if x not in seq[:t]]}
                for t in range(1, len(seq))
            ]
            n_pref += len(prefs)
            eval_orders.append({"rep": o["rep"], "order": seq, "prefs": prefs})

        if mech_short == "TypeA":
            tot_pref_A += n_pref
        else:
            tot_pref_B += n_pref

        # Note: Do NOT expose TypeA/TypeB label in UI header
        cert_ui.append({
            "id": cid,
            "k": int(it["canonical_k"]),
            "question": it.get("question", ""),
            "final_answer": it.get("final_answer", ""),
            "shards": [{"id": s["shard_id"], "text": (s.get("shard") or s.get("text") or "").strip()} for s in it.get("shards", [])],
            "eval_orders": eval_orders,
            "n_eval_prefixes": n_pref,
            "certificate": {
                "target_expr": cert.get("target_expr"),
                "equivalence_note": cert.get("equivalence_note", ""),
                "variables": cert.get("variables", {}),
                "functions": cert.get("functions", {}),
                "shard_constraints": cert.get("shard_constraints", {}),
                "full_answer_check": cert.get("full_answer_check", {}),
                "leave_one_out": cert.get("leave_one_out", {}),
            },
            "report": {
                "full_answer_verified": rep.get("full_answer_verified"),
                "leave_one_out_verified": rep.get("leave_one_out_verified"),
            },
        })
        cert_key[cid] = {
            "item_id": iid,
            "verified_mechanism": it["verified_mechanism"],
            "mech_short": mech_short,
            "macro_subfield": it["macro_subfield"],
            "canonical_k": int(it["canonical_k"]),
            "n_eval_prefixes": n_pref,
            "disclosure_orders": [o["order"] for o in ords],
        }

    # ================= LOAD ALL 5,400 JUDGED TRAJECTORIES =================
    scope_pools: dict[str, list[dict]] = {k: [] for k in SCOPE_QUOTAS}
    repair_pos_pool: list[dict] = []
    repair_neg_nonempty_by_model: dict[str, list[dict]] = defaultdict(list)
    repair_neg_empty_ids: list[dict] = []

    total_nonfinal_turns = 0
    total_eU_trajs = 0

    for slug in JUDGED_SLUGS:
        recs = read_jsonl_auto(V13_DIR / f"{slug}.jsonl")
        for rec in recs:
            proto = rec.get("protocol")
            if proto not in DYN_PROTOS:
                continue
            mid = rec["model_id"]
            tid = rec["trajectory_id"]
            iid = rec["item_id"]
            mech_short = "TypeA" if rec.get("verified_mechanism") == "TypeA_UniqueValue" else "TypeB"
            disc_ord = list(rec["disclosure_order"])
            smap = shards_by_item[iid]
            turns = rec["turns"]

            # Scope pool (all non-final turns)
            for td in turns:
                if td.get("is_final_turn"):
                    continue
                total_nonfinal_turns += 1
                t_idx = int(td["turn_idx"])
                pri = str(td.get("primary_scope") or "")
                sec = str(td.get("secondary_scope") or "")
                st = scope_stratum(pri, sec, mech_short)
                scope_pools[st].append({
                    "unique_key": (mid, tid, t_idx),
                    "model_id": mid,
                    "trajectory_id": tid,
                    "item_id": iid,
                    "protocol": proto,
                    "mech_short": mech_short,
                    "canonical_k": int(rec["canonical_k"]),
                    "turn_idx": t_idx,
                    "stratum": st,
                    "pri": pri,
                    "sec": sec,
                    "scope_C": str(td.get("scope_C") or ""),
                    "e_L": int(td.get("e_L") or 0),
                    "e_U": int(td.get("e_U") or 0),
                    "clauses_so_far": format_clauses(smap, disc_ord, t_idx),
                    "visible_reply": (td.get("visible_reply") or "").strip(),
                })

            # Repair pool (all E_U == 1 trajectories)
            if int(rec.get("e_U") or 0) == 1:
                total_eU_trajs += 1
                rep_obj = rec.get("repair") or {}
                rp = 1 if rep_obj.get("primary_retracted") else 0
                rs = 1 if rep_obj.get("secondary_retracted") else 0
                # Find earliest e_U == 1 non-final turn
                early_td = None
                for td in turns:
                    if not td.get("is_final_turn") and int(td.get("e_U") or 0) == 1:
                        early_td = td
                        break
                assert early_td is not None, f"Missing early e_U=1 turn in {mid}/{tid}"
                fin_td = [td for td in turns if td.get("is_final_turn")][0]
                fin_vis = (fin_td.get("visible_reply") or "").strip()

                entry = {
                    "unique_key": (mid, tid),
                    "model_id": mid,
                    "trajectory_id": tid,
                    "item_id": iid,
                    "protocol": proto,
                    "mech_short": mech_short,
                    "canonical_k": int(rec["canonical_k"]),
                    "early_turn": int(early_td["turn_idx"]),
                    "rep_pri": rp,
                    "rep_sec": rs,
                    "judge_group": "dual_pos" if (rp == 1 and rs == 1) else ("single_pos" if (rp == 1 or rs == 1) else "dual_neg"),
                    "all_clauses": format_clauses(smap, disc_ord, len(disc_ord)),
                    "early_reply": (early_td.get("visible_reply") or "").strip(),
                    "final_reply": fin_vis,
                }
                if rp == 1 or rs == 1:
                    repair_pos_pool.append(entry)
                else:
                    if fin_vis:
                        repair_neg_nonempty_by_model[mid].append(entry)
                    else:
                        repair_neg_empty_ids.append({
                            "model_id": mid,
                            "trajectory_id": tid,
                            "item_id": iid,
                            "protocol": proto,
                            "early_turn": int(early_td["turn_idx"]),
                        })

    assert total_nonfinal_turns == 7668, f"Expected 7,668 non-final turns, got {total_nonfinal_turns}"
    assert total_eU_trajs == 489, f"Expected 489 E_U=1 trajectories, got {total_eU_trajs}"
    assert len(repair_pos_pool) == 26, f"Expected 26 judge-flagged repair candidates, got {len(repair_pos_pool)}"
    n_nonempty_neg = sum(len(v) for v in repair_neg_nonempty_by_model.values())
    assert n_nonempty_neg + len(repair_neg_empty_ids) == 463

    # ================= MODULE 2: SCOPE-120 =================
    scope_pop_counts = {st: len(pool) for st, pool in scope_pools.items()}
    assert sum(scope_pop_counts.values()) == 7668

    picked_scope = []
    for st, q in SCOPE_QUOTAS.items():
        picked_scope.extend(balanced_pick(scope_pools[st], q, rng_scope, lambda x: (x["model_id"], x["protocol"])))
    rng_scope.shuffle(picked_scope)
    assert len(picked_scope) == 120

    scope_ui = []
    scope_key = {}
    for idx, x in enumerate(picked_scope, 1):
        sid = f"S{idx:03d}"
        scope_ui.append({
            "id": sid,
            "turn_idx": x["turn_idx"],
            "clauses": x["clauses_so_far"],
            "reply": x["visible_reply"] if x["visible_reply"] else "(空回复 / 无可见输出)",
        })
        scope_key[sid] = {
            "model_id": x["model_id"],
            "trajectory_id": x["trajectory_id"],
            "item_id": x["item_id"],
            "protocol": x["protocol"],
            "mech_short": x["mech_short"],
            "canonical_k": x["canonical_k"],
            "turn_idx": x["turn_idx"],
            "stratum": x["stratum"],
            "pri": x["pri"],
            "sec": x["sec"],
            "scope_C": x["scope_C"],
            "e_L": x["e_L"],
            "e_U": x["e_U"],
        }

    # ================= MODULE 3: REPAIR-100 (26 FLAGGED + 74 NON-EMPTY DUAL-NEG) =================
    picked_neg = []
    for mid, q in REPAIR_NEG_MODEL_QUOTAS.items():
        picked_neg.extend(balanced_pick(repair_neg_nonempty_by_model[mid], q, rng_repair, lambda x: x["protocol"]))
    assert len(picked_neg) == 74

    mixed_repair = sorted(repair_pos_pool, key=lambda x: x["unique_key"]) + picked_neg
    rng_repair.shuffle(mixed_repair)
    assert len(mixed_repair) == 100

    repair_ui = []
    repair_key = {}
    for idx, x in enumerate(mixed_repair, 1):
        rid = f"R{idx:03d}"
        repair_ui.append({
            "id": rid,
            "early_turn": x["early_turn"],
            "clauses": x["all_clauses"],
            "early_reply": x["early_reply"] if x["early_reply"] else "(空回复)",
            "final_reply": x["final_reply"] if x["final_reply"] else "(空回复)",
        })
        repair_key[rid] = {
            "model_id": x["model_id"],
            "trajectory_id": x["trajectory_id"],
            "item_id": x["item_id"],
            "protocol": x["protocol"],
            "mech_short": x["mech_short"],
            "canonical_k": x["canonical_k"],
            "early_turn": x["early_turn"],
            "rep_pri": x["rep_pri"],
            "rep_sec": x["rep_sec"],
            "judge_group": x["judge_group"],
        }

    # Write blank CSV templates
    with (HERE / "cert_100_template.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "human_verdict", "prefix_flip", "affected_prefixes", "human_notes"])
        for r in cert_ui:
            w.writerow([r["id"], "", "", "", ""])

    with (HERE / "scope_120_template.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "turn_idx", "human_scope", "human_notes"])
        for r in scope_ui:
            w.writerow([r["id"], r["turn_idx"], "", ""])

    with (HERE / "repair_100_template.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "early_turn", "human_retracted", "human_notes"])
        for r in repair_ui:
            w.writerow([r["id"], r["early_turn"], "", ""])

    # Write locked answer key (NEVER embedded in HTML)
    key_out = {
        "seed": SEED,
        "protocol_version": "unified_neutral_no_adjudication_v2",
        "cert_100": {
            "n_items": 100,
            "total_eval_prefixes_TypeA": tot_pref_A,
            "total_eval_prefixes_TypeB": tot_pref_B,
            "total_eval_prefixes_all": tot_pref_A + tot_pref_B,
            "items": cert_key,
        },
        "scope_120": {
            "total_population_turns": total_nonfinal_turns,
            "stratum_population_counts": scope_pop_counts,
            "stratum_sample_quotas": SCOPE_QUOTAS,
            "unique_items": len({v["item_id"] for v in scope_key.values()}),
            "turns": scope_key,
        },
        "repair_100": {
            "total_eU_trajectories": total_eU_trajs,
            "n_judge_flagged_census": len(repair_pos_pool),
            "n_dual_neg_total": 463,
            "n_dual_neg_empty_deterministic_zero": len(repair_neg_empty_ids),
            "n_dual_neg_nonempty_pool": n_nonempty_neg,
            "n_dual_neg_nonempty_sampled": len(picked_neg),
            "dual_neg_empty_trajectories": repair_neg_empty_ids,
            "unique_items_in_74_controls": len({x["item_id"] for x in picked_neg}),
            "trajectories": repair_key,
        },
    }
    (HERE / "_unified_audit_key_DO_NOT_OPEN.json").write_text(
        json.dumps(key_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # Build single-file 3-tab neutral HTML
    payload = json.dumps({"cert": cert_ui, "scope": scope_ui, "repair": repair_ui}, ensure_ascii=False).replace("</", "<\\/")

    # Programmatic leak check on UI payload before writing HTML
    for forbidden in ["TypeA", "TypeB", "Qwen", "GLM", "DYN_FREE", "DYN_REQUEST", "dual_pos", "single_pos", "dual_neg", "agree_Commit", "disagree_Commit"]:
        assert forbidden not in payload, f"LEAK DETECTED IN UI PAYLOAD: {forbidden}"

    html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>统一中性独立复核工具（Cert-100 / Scope-120 / Repair-100）— Seed 2026</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --border:#dfe4ee; --text:#1c2434; --muted:#64748b; --primary:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); line-height:1.6; padding-bottom:140px; }
  header { position:sticky; top:0; z-index:30; background:#fff; border-bottom:1px solid var(--border); box-shadow:0 2px 8px rgba(15,23,42,.04); }
  .topbar { max-width:1420px; margin:0 auto; padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; }
  .brand { font-weight:700; font-size:15px; display:flex; align-items:center; gap:8px; }
  .badge { background:#eef2ff; color:#3730a3; font-size:12px; padding:2px 8px; border-radius:999px; font-weight:600; }
  .tabs { display:flex; flex-wrap:wrap; gap:6px; }
  .tab-btn { border:1px solid var(--border); background:#fff; padding:6px 11px; border-radius:8px; cursor:pointer; font-size:13px; font-weight:600; }
  .tab-btn.active { background:var(--primary); color:#fff; border-color:var(--primary); }
  .controls { display:flex; flex-wrap:wrap; gap:7px; align-items:center; font-size:13px; }
  .controls input[type="text"] { width:92px; padding:5px 8px; border:1px solid var(--border); border-radius:6px; font-size:13px; }
  .btn-sm { border:1px solid var(--border); background:#fff; color:var(--text); padding:5px 10px; border-radius:6px; font-size:12px; cursor:pointer; font-weight:600; }
  .btn-export { background:#0f766e; color:#fff; border-color:#0f766e; }
  .container { max-width:1420px; margin:14px auto; padding:0 16px; }
  .rubric { background:#f8fafc; border:1px solid #cbd5e1; border-radius:10px; padding:10px 14px; margin-bottom:12px; font-size:13px; }
  .rubric summary { cursor:pointer; font-weight:700; color:#1e293b; }
  .rubric ul { margin:6px 0 0 18px; padding:0; }
  .nav-card { background:#fff; border:1px solid var(--border); border-radius:10px; padding:10px 14px; margin-bottom:12px; }
  .nav-top { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; font-size:13px; }
  .progress-wrap { flex:1; min-width:180px; height:8px; background:#e2e8f0; border-radius:99px; overflow:hidden; }
  .progress-bar { height:100%; background:var(--primary); transition:width .2s; }
  .grid-ids { display:flex; flex-wrap:wrap; gap:3.5px; }
  .id-pill { width:40px; height:22px; border:1px solid var(--border); border-radius:4px; background:#fff; font-size:10.5px; font-weight:600; cursor:pointer; color:var(--muted); }
  .id-pill.current { outline:2px solid #0f172a; outline-offset:1px; color:#0f172a; }
  .id-pill.v-Faithful, .id-pill.v-Wait, .id-pill.v-No { background:#dcfce7; border-color:#86efac; color:#166534; }
  .id-pill.v-Minor, .id-pill.v-Conditional { background:#fef3c7; border-color:#fcd34d; color:#b45309; }
  .id-pill.v-Unfaithful, .id-pill.v-Commit, .id-pill.v-Yes { background:#fee2e2; border-color:#fca5a5; color:#b91c1c; }
  .id-pill.v-Partial_Other { background:#e2e8f0; border-color:#94a3b8; color:#334155; }
  .split-2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; align-items:start; }
  @media (max-width:960px) { .split-2 { grid-template-columns:1fr; } }
  .panel { background:#fff; border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:12px; }
  .panel-title { font-size:13px; font-weight:700; color:var(--muted); margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; gap:8px; }
  .shard-item { background:#f8fafc; border-left:3px solid var(--primary); padding:6px 10px; margin-bottom:6px; border-radius:4px; font-size:14px; }
  .ans-box { background:#ecfdf5; border:1px solid #a7f3d0; padding:8px 12px; border-radius:6px; margin:8px 0; font-size:14px; }
  .orders-box { background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px 10px; margin-top:10px; font-size:12.5px; }
  .orders-box code { background:#e2e8f0; padding:1px 4px; border-radius:3px; font-size:12px; }
  .sym-block { background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px 10px; margin-bottom:8px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12.5px; white-space:pre-wrap; word-break:break-word; }
  .eq-note { background:#f0f9ff; border-left:3px solid #0284c7; padding:8px 10px; border-radius:4px; font-size:13px; margin-bottom:8px; }
  .clauses-box { background:#f8fafc; border-left:4px solid var(--primary); padding:10px 14px; border-radius:6px; font-size:14px; white-space:pre-wrap; }
  .reply-box { font-size:14.5px; overflow-x:auto; word-break:break-word; }
  .reply-box p { margin:0 0 8px 0; }
  .reply-box ul { margin:4px 0 8px 22px; padding:0; }
  .reply-box pre { background:#f1f5f9; padding:10px; border-radius:6px; font-size:12.5px; white-space:pre-wrap; }
  .action-bar { position:fixed; left:0; right:0; bottom:0; z-index:40; background:rgba(255,255,255,.97); border-top:1px solid var(--border); box-shadow:0 -4px 16px rgba(15,23,42,.08); padding:10px 16px; }
  .action-inner { max-width:1420px; margin:0 auto; display:flex; flex-direction:column; gap:8px; }
  .action-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; }
  .choice-group { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .choice-btn { border:2px solid var(--border); background:#fff; padding:7px 12px; border-radius:8px; font-size:13px; font-weight:700; cursor:pointer; }
  .choice-btn.selected { background:#0f172a; border-color:#0f172a; color:#fff; }
  .flip-btn { border:1.5px solid var(--border); background:#fff; padding:5px 10px; border-radius:7px; font-size:12.5px; font-weight:700; cursor:pointer; }
  .flip-btn.selected-no { background:#dcfce7; border-color:#16a34a; color:#166534; }
  .flip-btn.selected-yes { background:#fee2e2; border-color:#dc2626; color:#991b1b; }
  .note-input { flex:1; min-width:200px; padding:6px 10px; border:1px solid var(--border); border-radius:7px; font-size:13px; }
  .aff-input { width:240px; padding:6px 10px; border:1px solid var(--border); border-radius:7px; font-size:12.5px; }
  .toast { position:fixed; top:66px; right:20px; background:#0f172a; color:#fff; padding:8px 14px; border-radius:8px; font-size:13px; opacity:0; pointer-events:none; transition:opacity .2s; z-index:60; }
  .toast.show { opacity:.95; }
</style>
</head>
<body>
<header>
  <div class="topbar">
    <div class="brand">统一中性独立复核工具 <span class="badge">Seed 2026 · 全盲 · 免自裁 [L, U]</span></div>
    <div class="tabs">
      <button class="tab-btn active" id="tab-cert" onclick="switchTab('cert')">① Cert-100 (<span id="cnt-cert">0/100</span>)</button>
      <button class="tab-btn" id="tab-scope" onclick="switchTab('scope')">② Scope-120 (<span id="cnt-scope">0/120</span>)</button>
      <button class="tab-btn" id="tab-repair" onclick="switchTab('repair')">③ Repair-100 (<span id="cnt-repair">0/100</span>)</button>
    </div>
    <div class="controls">
      <label>标注者: <input type="text" id="annotator" value="" placeholder="填姓名/编号" onchange="onAnnotatorChange()" /></label>
      <label style="cursor:pointer;"><input type="checkbox" id="autoNext" checked /> 自动下一题</label>
      <button class="btn-sm" id="rawBtn" onclick="toggleRaw()">显示原文</button>
      <button class="btn-sm btn-export" onclick="exportAll()">导出全部 CSV</button>
      <button class="btn-sm" onclick="exportJSON()">备份JSON</button>
      <button class="btn-sm" onclick="document.getElementById('imp').click()">导入JSON</button>
      <input type="file" id="imp" accept=".json" style="display:none" onchange="importJSON(event)" />
    </div>
  </div>
</header>

<div class="container">
  <details class="rubric" open>
    <summary id="rubricTitle">核验标准（中性表述，点击可折叠）</summary>
    <div id="rubricBody"></div>
  </details>
  <div class="nav-card">
    <div class="nav-top">
      <span id="progText"></span>
      <div class="progress-wrap"><div class="progress-bar" id="progBar" style="width:0%"></div></div>
      <button class="btn-sm" onclick="jumpNext()">跳到下一个未标 (U)</button>
    </div>
    <div class="grid-ids" id="idGrid"></div>
  </div>
  <div id="mainContent"></div>
</div>

<div class="action-bar">
  <div class="action-inner">
    <div class="action-row">
      <div class="choice-group" id="choiceGroup"></div>
      <div class="choice-group" id="extraGroup"></div>
      <div style="display:flex;gap:6px;">
        <button class="btn-sm" onclick="step(-1)">← 上一题</button>
        <button class="btn-sm" onclick="step(1)">下一题 →</button>
      </div>
    </div>
    <div class="action-row">
      <input type="text" class="aff-input" id="affInput" placeholder="若 Prefix Flip=Yes，写明受影响前缀 (如 Rep1-T2)" oninput="saveExtra()" style="display:none" />
      <input type="text" class="note-input" id="noteInput" placeholder="选填备注（如遇边界案例或 Unfaithful 可简记原因）" oninput="saveExtra()" />
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const D = __PAYLOAD_JSON__;
let tab = 'cert';
let idx = { cert: 0, scope: 0, repair: 0 };
let showRaw = false;
let state = emptyState();

function emptyState() { return { cert:{}, cert_flip:{}, cert_aff:{}, scope:{}, repair:{}, notes:{} }; }
function whoRaw() { return (document.getElementById('annotator').value || '').trim(); }
function whoSlug() { return whoRaw().toLowerCase().replace(/[^a-z0-9_-]/g, '_') || 'unnamed'; }
function skey() { return 'cscwd_unified_v2_seed2026_' + whoSlug(); }
function loadState() {
  try { const r = localStorage.getItem(skey()); state = r ? JSON.parse(r) : emptyState(); }
  catch(e) { state = emptyState(); }
  for (const k of ['cert','cert_flip','cert_aff','scope','repair','notes']) state[k] = state[k] || {};
}
function saveState() { localStorage.setItem(skey(), JSON.stringify(state)); }
function onAnnotatorChange() {
  if (whoRaw()) localStorage.setItem('cscwd_unified_v2_seed2026_last', whoRaw());
  loadState(); render();
  const c = Object.keys(state.cert).length, s = Object.keys(state.scope).length, r = Object.keys(state.repair).length;
  toast(`当前标注者：${whoSlug()}（Cert ${c}/100 · Scope ${s}/120 · Repair ${r}/100）`);
}
function ensureNamed() {
  if (!whoRaw()) { alert('请先在页面右上角填写您的标注者姓名/代号（例如 cai / tang / rater3），以便独立存档与导出！'); document.getElementById('annotator').focus(); return false; }
  return true;
}
function toast(m) { const t=document.getElementById('toast'); t.textContent=m; t.classList.add('show'); clearTimeout(window._tt); window._tt=setTimeout(()=>t.classList.remove('show'),2000); }
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function renderRich(text) {
  if (showRaw) return '<pre>' + esc(text) + '</pre>';
  const ph = [];
  let s = String(text||'').replace(/\r\n/g,'\n');
  s = s.replace(/(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$(?!\s)([^\n$]+?)(?<!\s)\$)/g, m => { ph.push(m); return '@@M'+(ph.length-1)+'@@'; });
  s = esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  let html='', ul=false;
  for (let l of s.split('\n')) {
    const tr = l.trim();
    if (/^[-*•]\s+/.test(tr)) { if (!ul){html+='<ul>';ul=true;} html+='<li>'+tr.replace(/^[-*•]\s+/,'')+'</li>'; }
    else { if (ul){html+='</ul>';ul=false;} html += tr ? '<p>'+l+'</p>' : '<div style="height:5px"></div>'; }
  }
  if (ul) html += '</ul>';
  return html.replace(/@@M(\d+)@@/g, (_,i)=>esc(ph[+i]));
}
function typeset(el) {
  if (!showRaw && typeof renderMathInElement==='function') {
    try { renderMathInElement(el, { delimiters:[{left:'$$',right:'$$',display:true},{left:'\\[',right:'\\]',display:true},{left:'$',right:'$',display:false},{left:'\\(',right:'\\)',display:false}], throwOnError:false, strict:false }); } catch(e){}
  }
}
function switchTab(t) {
  tab = t;
  ['cert','scope','repair'].forEach(k => document.getElementById('tab-'+k).classList.toggle('active', k===t));
  render();
}
function toggleRaw() { showRaw=!showRaw; document.getElementById('rawBtn').textContent=showRaw?'渲染公式':'显示原文'; render(); }

function render() {
  document.getElementById('cnt-cert').textContent = `${Object.keys(state.cert).length}/${D.cert.length}`;
  document.getElementById('cnt-scope').textContent = `${Object.keys(state.scope).length}/${D.scope.length}`;
  document.getElementById('cnt-repair').textContent = `${Object.keys(state.repair).length}/${D.repair.length}`;

  if (tab === 'cert') {
    document.getElementById('rubricTitle').textContent = '① Cert-100 形式化保真度与必要性核验标准（全量 100 题，852 个评测前缀）';
    document.getElementById('rubricBody').innerHTML = `<ul>
      <li><strong style="color:#166534">1. Faithful（严格忠实）</strong>：SymPy 变量域、各分片约束和目标与中文原题一致，每个 $S_j$ 的留一见证在中文原题下均合法。</li>
      <li><strong style="color:#b45309">2. Minor（细微技术简化，必要性仍成立）</strong>：省略次要非零/非退化辅助约束或轻微简化，但中文原题下每条 $S_j$ 仍严格必要（所有评测前缀仍不充分）。</li>
      <li><strong style="color:#b91c1c">3. Unfaithful（实质偏差）</strong>：实质漏译、翻译错误或分片切分导致某分片冗余或留一见证违反保留分片。选此项后请对照左下角 3 条实际披露序勾选 <strong>Prefix Flip (No / Yes)</strong> 并注明受影响前缀。</li>
    </ul>`;
  } else if (tab === 'scope') {
    document.getElementById('rubricTitle').textContent = '② Scope-120 回复作答姿态分类标准（中性呈现：不预设所给条件是否充分，仅判断助手自身的表述姿态）';
    document.getElementById('rubricBody').innerHTML = `<ul>
      <li><strong style="color:#166534">1. Wait（等待/不作答）</strong>：助手明确表示目前无法给出最终答案、请求更多条件，或仅做思路/公式准备而未给出最终数值/表达式结论。</li>
      <li><strong style="color:#b45309">2. Conditional（带假设/条件化作答）</strong>：助手给出了答案或极值，但<strong>明确限定在假设前提下</strong>（例如“若无其他限制条件”“假设 $a>0$ 时”“在当前未补充新约束的松弛情形下”）。</li>
      <li><strong style="color:#b91c1c">3. Commit（无条件确定性作答）</strong>：助手直接给出唯一最终答案/解集/极值（含无条件 <code>\\boxed{}</code> 结论），未将其限定为临时或带假设结论。</li>
      <li><strong style="color:#334155">4. Partial/Other（部分推导/截断/其他）</strong>：空回复、明显中途截断未及结论、或无法归入以上三类的其他回复。</li>
    </ul>`;
  } else {
    document.getElementById('rubricTitle').textContent = '③ Repair-100 末轮回复对照核验标准（中性呈现：判断末轮是否明确提及并撤回/纠正先前回复中的结论）';
    document.getElementById('rubricBody').innerHTML = `<ul>
      <li><strong style="color:#b91c1c">1 / Y. Yes（存在明确撤回/纠正）</strong>：末轮可见回复中<strong>明确提及</strong>先前第 $t$ 轮给出的结论/断言，并指出其不成立、需撤回、修正或推翻。</li>
      <li><strong style="color:#166534">2 / N. No（未明确撤回/纠正）</strong>：末轮仅基于完整条件直接推导并给出新答案（即便新旧答案数值不同，但未在可见文本中显式提及或撤回先前结论，也选 No）。</li>
    </ul>`;
  }

  const list = D[tab];
  const map = state[tab];
  const done = list.filter(x => map[x.id]).length;
  document.getElementById('progText').textContent = `当前模块进度：${done} / ${list.length}（第 ${idx[tab]+1} 题）`;
  document.getElementById('progBar').style.width = `${Math.round(done/list.length*100)}%`;

  const grid = document.getElementById('idGrid');
  grid.innerHTML = '';
  list.forEach((it, i) => {
    const b = document.createElement('button');
    const v = map[it.id] || '';
    const clsV = v ? ' v-' + v.replace(/[^A-Za-z0-9]/g, '_') : '';
    b.className = 'id-pill' + clsV + (i===idx[tab] ? ' current' : '');
    b.textContent = it.id;
    b.title = v ? `${it.id}: ${v}` : it.id;
    b.onclick = () => { idx[tab]=i; render(); };
    grid.appendChild(b);
  });

  const item = list[idx[tab]];
  const cur = map[item.id] || '';
  const mc = document.getElementById('mainContent');
  const affInput = document.getElementById('affInput');

  if (tab === 'cert') {
    const c = item.certificate;
    const curFlip = state.cert_flip[item.id] || (cur ? 'No' : '');
    const shardsHtml = item.shards.map(s => `<div class="shard-item"><strong>[${esc(s.id)}]</strong> ${renderRich(s.text)}</div>`).join('');
    const ordersHtml = item.eval_orders.map(o => {
      const pStr = o.prefs.map(p => `T${p.turn}: <code>{${p.prefix.join(',')}}</code> (缺 ${p.missing.join(',')})`).join(' &nbsp;|&nbsp; ');
      return `<div style="margin-bottom:4px;"><strong>Rep ${o.rep}</strong> [${o.order.join('→')}]：${pStr}</div>`;
    }).join('');
    const scText = Object.entries(c.shard_constraints || {}).map(([k, arr]) => `${k}: ${JSON.stringify(arr)}`).join('\n');
    const fnText = Object.keys(c.functions || {}).length ? JSON.stringify(c.functions, null, 2) : '(无自定义函数)';
    const looText = JSON.stringify({ leave_one_out: c.leave_one_out, verified_values: item.report.leave_one_out_verified }, null, 2);

    mc.innerHTML = `
      <div class="split-2">
        <div class="panel">
          <div class="panel-title"><span>样本 ${item.id} · k=${item.k}（${item.n_eval_prefixes} 个非最终评测前缀）</span><span>当前：${cur||'未标'}${curFlip ? ' · Flip:'+curFlip : ''}</span></div>
          <div style="margin-bottom:8px;"><strong>原始中文题干：</strong><div class="reply-box" style="margin-top:4px;">${renderRich(item.question)}</div></div>
          <div class="ans-box"><strong>官方参考答案：</strong> ${renderRich(item.final_answer)}</div>
          <div style="margin-top:10px;"><strong>分片（S1–S${item.k}）：</strong><div style="margin-top:6px;">${shardsHtml}</div></div>
          <div class="orders-box"><strong>实际评测的 3 条披露序：</strong><div style="margin-top:5px;">${ordersHtml}</div></div>
        </div>
        <div class="panel">
          <div class="panel-title"><span>SymPy 形式化证书（离线校验 PASS）</span><span>target: <code>${esc(c.target_expr)}</code></span></div>
          <div class="eq-note"><strong>形式化说明：</strong>${esc(c.equivalence_note || '')}</div>
          <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">变量与函数定义</div>
          <div class="sym-block">variables: ${esc(JSON.stringify(c.variables))}\nfunctions: ${esc(fnText)}\nfull_answer_check: ${esc(JSON.stringify(c.full_answer_check))}\nfull_answer_verified: ${esc(String(item.report.full_answer_verified))}</div>
          <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">各分片约束 (shard_constraints)</div>
          <div class="sym-block">${esc(scText)}</div>
          <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">留一法见证 (LOO witnesses & verified outputs)</div>
          <div class="sym-block" style="max-height:300px;overflow:auto;">${esc(looText)}</div>
        </div>
      </div>`;
    const opts = [['Faithful','1: Faithful'],['Minor','2: Minor（保持必要性）'],['Unfaithful','3: Unfaithful（实质偏差）']];
    document.getElementById('choiceGroup').innerHTML = opts.map(([v,l]) => `<button class="choice-btn ${cur===v?'selected':''}" onclick="pick('${v}')">${l}</button>`).join('');
    document.getElementById('extraGroup').innerHTML = `<span style="font-size:12.5px;font-weight:700;color:var(--muted);">评测前缀受影响?</span>
      <button class="flip-btn ${curFlip==='No'?'selected-no':''}" onclick="pickFlip('No')">No (0条充分)</button>
      <button class="flip-btn ${curFlip==='Yes'?'selected-yes':''}" onclick="pickFlip('Yes')">Yes (有前缀充分)</button>`;
    affInput.style.display = 'block';
    affInput.value = state.cert_aff[item.id] || '';
  } else if (tab === 'scope') {
    mc.innerHTML = `
      <div class="panel">
        <div class="panel-title"><span>样本 ${item.id} · 截至第 ${item.turn_idx} 轮已披露的条件</span><span>当前：${cur||'未标'}</span></div>
        <div class="clauses-box">${renderRich(item.clauses)}</div>
      </div>
      <div class="panel">
        <div class="panel-title"><span>助手在第 ${item.turn_idx} 轮的可见回复（请仅依据文本判断其作答姿态）</span></div>
        <div class="reply-box">${renderRich(item.reply)}</div>
      </div>`;
    const opts = [['Wait','1: Wait（等待/不作答）'],['Conditional','2: Conditional（带假设作答）'],['Commit','3: Commit（无条件定论）'],['Partial/Other','4: Partial/Other']];
    document.getElementById('choiceGroup').innerHTML = opts.map(([v,l]) => `<button class="choice-btn ${cur===v?'selected':''}" onclick="pick('${v}')">${l}</button>`).join('');
    document.getElementById('extraGroup').innerHTML = '';
    affInput.style.display = 'none';
  } else {
    mc.innerHTML = `
      <div class="panel">
        <div class="panel-title"><span>样本 ${item.id} · 完整题目条件</span><span>当前：${cur||'未标'}</span></div>
        <div class="clauses-box">${renderRich(item.clauses)}</div>
      </div>
      <div class="split-2">
        <div class="panel" style="border-top:3px solid #64748b">
          <div class="panel-title"><span>先前第 ${item.early_turn} 轮助手可见回复</span></div>
          <div class="reply-box">${renderRich(item.early_reply)}</div>
        </div>
        <div class="panel" style="border-top:3px solid #2563eb">
          <div class="panel-title"><span>末轮助手可见回复（是否明确提及并撤回/纠正左侧回复的结论？）</span></div>
          <div class="reply-box">${renderRich(item.final_reply)}</div>
        </div>
      </div>`;
    const opts = [['Yes','1 / Y: Yes（明确撤回/纠正）'],['No','2 / N: No（未明确撤回）']];
    document.getElementById('choiceGroup').innerHTML = opts.map(([v,l]) => `<button class="choice-btn ${cur===v?'selected':''}" onclick="pick('${v}')">${l}</button>`).join('');
    document.getElementById('extraGroup').innerHTML = '';
    affInput.style.display = 'none';
  }

  document.getElementById('noteInput').value = state.notes[item.id] || '';
  typeset(mc);
  window.scrollTo({top:0, behavior:'instant'});
}

function pick(v) {
  if (!ensureNamed()) return;
  const list = D[tab], item = list[idx[tab]];
  state[tab][item.id] = v;
  if (tab === 'cert') {
    if (v !== 'Unfaithful') state.cert_flip[item.id] = 'No';
    else if (!state.cert_flip[item.id]) state.cert_flip[item.id] = 'No';
  }
  saveState();
  toast(`${item.id} -> ${v}`);
  const hold = (tab === 'cert' && v === 'Unfaithful');
  if (!hold && document.getElementById('autoNext').checked && idx[tab] < list.length - 1) idx[tab]++;
  render();
}
function pickFlip(fv) {
  if (!ensureNamed()) return;
  const item = D.cert[idx.cert];
  state.cert_flip[item.id] = fv;
  saveState();
  toast(`${item.id} Prefix Flip -> ${fv}`);
  render();
}
function saveExtra() {
  if (!whoRaw()) return;
  const item = D[tab][idx[tab]];
  if (tab === 'cert') {
    const a = document.getElementById('affInput').value;
    if (a.trim()) state.cert_aff[item.id] = a; else delete state.cert_aff[item.id];
  }
  const n = document.getElementById('noteInput').value;
  if (n.trim()) state.notes[item.id] = n; else delete state.notes[item.id];
  saveState();
}
function step(d) { const n = idx[tab]+d; if (n>=0 && n<D[tab].length){ idx[tab]=n; render(); } }
function jumpNext() {
  const list = D[tab], map = state[tab];
  for (let off=1; off<=list.length; off++) {
    const i = (idx[tab]+off) % list.length;
    if (!map[list[i].id]) { idx[tab]=i; render(); return; }
  }
  toast('当前标签页已全部标完！');
}
function csvEsc(v) { const s=String(v==null?'':v); return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s; }
function dl(name, content, mime) {
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([content],{type:mime}));
  a.download=name; document.body.appendChild(a); a.click(); setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove();},200);
}
function exportAll() {
  if (!ensureNamed()) return;
  const who = whoSlug();
  const cDone = D.cert.filter(r => state.cert[r.id]).length;
  const sDone = D.scope.filter(r => state.scope[r.id]).length;
  const rDone = D.repair.filter(r => state.repair[r.id]).length;
  if (cDone + sDone + rDone === 0) {
    alert(`标注者 ${who} 当前 3 个模块均为 0 已标，请先完成标注或核对右上角姓名！`);
    return;
  }
  const cLines = ['audit_id,human_verdict,prefix_flip,affected_prefixes,human_notes'];
  for (const r of D.cert) {
    const v = state.cert[r.id] || '';
    const fl = v ? (state.cert_flip[r.id] || 'No') : '';
    cLines.push([r.id, v, fl, state.cert_aff[r.id]||'', state.notes[r.id]||''].map(csvEsc).join(','));
  }
  const sLines = ['audit_id,turn_idx,human_scope,human_notes'];
  for (const r of D.scope) {
    sLines.push([r.id, r.turn_idx, state.scope[r.id]||'', state.notes[r.id]||''].map(csvEsc).join(','));
  }
  const rLines = ['audit_id,early_turn,human_retracted,human_notes'];
  for (const r of D.repair) {
    rLines.push([r.id, r.early_turn, state.repair[r.id]||'', state.notes[r.id]||''].map(csvEsc).join(','));
  }
  dl(`cert_100_${who}.csv`, '\uFEFF' + cLines.join('\r\n') + '\r\n', 'text/csv;charset=utf-8');
  setTimeout(()=> dl(`scope_120_${who}.csv`, '\uFEFF' + sLines.join('\r\n') + '\r\n', 'text/csv;charset=utf-8'), 250);
  setTimeout(()=> dl(`repair_100_${who}.csv`, '\uFEFF' + rLines.join('\r\n') + '\r\n', 'text/csv;charset=utf-8'), 500);
  toast(`已导出 3 份 CSV（Cert ${cDone}/100 · Scope ${sDone}/120 · Repair ${rDone}/100）`);
}
function exportJSON() {
  if (!ensureNamed()) return;
  dl(`unified_audit_backup_${whoSlug()}.json`, JSON.stringify({ annotator: whoSlug(), exported_at: new Date().toISOString(), state }, null, 2), 'application/json');
}
function importJSON(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try {
      const o = JSON.parse(rd.result); const st = o.state || o;
      if (o.annotator && !whoRaw()) document.getElementById('annotator').value = o.annotator;
      state = emptyState();
      for (const k of Object.keys(state)) state[k] = st[k] || {};
      saveState(); render(); toast('JSON 备份导入成功');
    } catch(e) { alert('JSON 无效：' + e.message); }
  };
  rd.readAsText(f, 'utf-8'); ev.target.value = '';
}
document.addEventListener('keydown', e => {
  if (e.target && e.target.tagName==='INPUT') return;
  if (e.key==='ArrowRight') step(1);
  else if (e.key==='ArrowLeft') step(-1);
  else if (e.key.toLowerCase()==='u') jumpNext();
  else if (tab==='cert') { if(e.key==='1') pick('Faithful'); else if(e.key==='2') pick('Minor'); else if(e.key==='3') pick('Unfaithful'); }
  else if (tab==='scope') { if(e.key==='1') pick('Wait'); else if(e.key==='2') pick('Conditional'); else if(e.key==='3') pick('Commit'); else if(e.key==='4') pick('Partial/Other'); }
  else { if(e.key==='1'||e.key.toLowerCase()==='y') pick('Yes'); else if(e.key==='2'||e.key.toLowerCase()==='n') pick('No'); }
});
window.addEventListener('DOMContentLoaded', () => {
  const last = localStorage.getItem('cscwd_unified_v2_seed2026_last');
  if (last) document.getElementById('annotator').value = last;
  loadState(); render();
});
window.addEventListener('load', () => typeset(document.getElementById('mainContent')));
</script>
</body>
</html>
'''

    out_html = HERE / "unified_audit_tool_seed2026.html"
    out_html.write_text(html.replace("__PAYLOAD_JSON__", payload), encoding="utf-8", newline="\n")
    print(f"Generated {out_html} ({out_html.stat().st_size / 1024:.1f} KB)")
    print(f"  Cert-100   : 100 items (68 TypeA [{tot_pref_A} prefixes] + 32 TypeB [{tot_pref_B} prefixes] = {tot_pref_A + tot_pref_B} prefixes)")
    print(f"  Scope-120  : 120 turns across 9 exhaustive strata ({key_out['scope_120']['unique_items']} unique items)")
    print(f"  Repair-100 : 26 judge-flagged + 74 non-empty dual-neg controls ({key_out['repair_100']['unique_items_in_74_controls']} unique items in controls; 6 empty = 0/6)")


if __name__ == "__main__":
    main()
