#!/usr/bin/env python3
"""Build the SEED=2027 Follow-Up Blind Human Audit Package.

Modules (100% blind, neutral framing, 0 judge/model/protocol/regime leaks):
  A. Traj-60        : Pure within-stratum SRSWOR sample of 60 TypeA FREE+REQ trajectories
                      (12 agree_Commit / 8 disagree_Commit / 40 agree_NonCommit out of 1,224).
                      Rater labels ALL non-final turns of each trajectory (final turn hidden).
  B. Repair-50-TypeA: Complete 50/50 census of the existing TypeA Repair-100 trajectories,
                      re-shuffled blindly (D01..D50) to measure 3 nested denominators:
                        Q1: confirmed unconditional concrete claim at anchor t0
                        Q2: anchor claim substantively conflicts with final reply k
                        Q3: conflicting claim still uncorrected before turn k

Reads (relative to repo root; supports .jsonl and .jsonl.gz):
  - exp1_codex_gpt61_v1/frozen_release_verified_120/frozen_test_100.jsonl
  - eval_multi_turn/runs/runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl[.gz]
  - eval_multi_turn/human_audit_unified_v2/_unified_audit_key_DO_NOT_OPEN.json

Writes (in this directory):
  - followup_audit_tool_seed2027.html
  - _followup_audit_key_DO_NOT_OPEN.json
  - traj_60_template.csv
  - repair50_denom_template.csv
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
EVAL_DIR = HERE.parent
ROOT = EVAL_DIR.parent

FROZEN_TEST = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_test_100.jsonl"
V13_DIR = EVAL_DIR / "runs" / "runs_16k_v13_blind"
UNIFIED_KEY = EVAL_DIR / "human_audit_unified_v2" / "_unified_audit_key_DO_NOT_OPEN.json"

SEED = 2027
TARGET_PROTOS = {"DYN_FREE", "DYN_REQUEST"}
JUDGED_SLUGS = [
    "judged_test100_qwen3_8b_16384_v13_blind",
    "judged_test100_qwen3_8b_guard_16384_v13_blind",
    "judged_test100_glm_z1_9b_0414_16384_v13_blind",
    "judged_test100_glm_z1_9b_0414_guard_16384_v13_blind",
    "judged_test100_qwen3_14b_16384_v13_blind",
    "judged_test100_qwen3_14b_guard_16384_v13_blind",
]

# Pure SRSWOR Neyman-aware allocation over all 1,224 TypeA FREE+REQ trajectories
TRAJ_STRATUM_ALLOC = {
    "agree_Commit": 12,
    "disagree_Commit": 8,
    "agree_NonCommit": 40,
}


def read_jsonl_auto(path_no_gz: Path) -> list[dict]:
    if path_no_gz.exists():
        return [json.loads(line) for line in path_no_gz.read_text(encoding="utf-8").splitlines() if line.strip()]
    gz = Path(str(path_no_gz) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return [json.loads(line) for line in f if line.strip()]
    raise FileNotFoundError(f"Neither {path_no_gz} nor {gz} exists.")


def sha256_obj(obj) -> str:
    encoded = json.dumps(obj, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def format_clause_line(shards_map: dict[str, str], sid: str, index_1based: int) -> str:
    return f"[第{index_1based}条/{sid}] {shards_map.get(sid, sid)}"


def traj_stratum(tr: dict) -> str:
    if int(tr["e_L"]) == 1:
        return "agree_Commit"
    if int(tr["e_U"]) == 1:
        return "disagree_Commit"
    return "agree_NonCommit"


def main() -> None:
    rng_traj = random.Random(SEED)
    rng_denom = random.Random(SEED + 1)

    test_items = read_jsonl_auto(FROZEN_TEST)
    assert len(test_items) == 100
    shards_by_item: dict[str, dict[str, str]] = {
        it["item_id"]: {
            s["shard_id"]: (s.get("shard") or s.get("text") or "").strip()
            for s in it.get("shards", [])
        }
        for it in test_items
    }

    # Load all 5,400 judged trajectories
    tracks: dict[tuple[str, str], dict] = {}
    for slug in JUDGED_SLUGS:
        for row in read_jsonl_auto(V13_DIR / f"{slug}.jsonl"):
            k = (row["model_id"], row["trajectory_id"])
            assert k not in tracks, f"Duplicate track: {k}"
            tracks[k] = row
    assert len(tracks) == 5400, f"Expected 5400 tracks, got {len(tracks)}"

    # ================= MODULE A: TRAJ-60 (Pure SRSWOR within 3 TypeA FREE+REQ strata) =================
    pools: dict[str, list[tuple[tuple[str, str], dict]]] = {
        "agree_Commit": [],
        "disagree_Commit": [],
        "agree_NonCommit": [],
    }
    for key_pair, tr in sorted(tracks.items()):
        if tr["verified_mechanism"] == "TypeA_UniqueValue" and tr["protocol"] in TARGET_PROTOS:
            pools[traj_stratum(tr)].append((key_pair, tr))

    pop_counts = {s: len(v) for s, v in pools.items()}
    assert sum(pop_counts.values()) == 1224, f"Unexpected TypeA FREE+REQ total: {pop_counts}"
    assert pop_counts == {"agree_Commit": 152, "disagree_Commit": 45, "agree_NonCommit": 1027}, pop_counts

    sampled_entries: list[tuple[str, tuple[str, str], dict]] = []
    for s_name in ("agree_Commit", "disagree_Commit", "agree_NonCommit"):
        n_s = TRAJ_STRATUM_ALLOC[s_name]
        chosen = rng_traj.sample(pools[s_name], n_s)
        for key_pair, tr in chosen:
            sampled_entries.append((s_name, key_pair, tr))
    assert len(sampled_entries) == 60
    rng_traj.shuffle(sampled_entries)

    traj_ui = []
    traj_key = {}
    total_traj_turns = 0
    for idx, (s_name, (model_id, traj_id), tr) in enumerate(sampled_entries, 1):
        tid = f"T{idx:02d}"
        iid = tr["item_id"]
        order = list(tr["disclosure_order"])
        smap = shards_by_item[iid]
        nf_turns = [t for t in tr["turns"] if not t["is_final_turn"]]
        nf_turns.sort(key=lambda t: int(t["turn_idx"]))
        total_traj_turns += len(nf_turns)

        turns_ui = []
        turns_key = []
        for t in nf_turns:
            t_idx = int(t["turn_idx"])
            new_sid = order[t_idx - 1]
            reply = (t.get("visible_reply") or "").strip() or "(空回复)"
            turns_ui.append({
                "turn_idx": t_idx,
                "new_clause": format_clause_line(smap, new_sid, t_idx),
                "cum_clauses": "\n".join(format_clause_line(smap, order[j], j + 1) for j in range(t_idx)),
                "reply": reply,
            })
            turns_key.append({
                "turn_idx": t_idx,
                "e_L": int(t.get("e_L", 0)),
                "e_U": int(t.get("e_U", 0)),
                "primary_scope": t.get("primary_scope"),
                "secondary_scope": t.get("secondary_scope"),
            })

        ui_obj = {
            "id": tid,
            "n_turns": len(turns_ui),
            "turns": turns_ui,
        }
        traj_ui.append(ui_obj)
        traj_key[tid] = {
            "stratum": s_name,
            "N_h": pop_counts[s_name],
            "n_h": TRAJ_STRATUM_ALLOC[s_name],
            "model_id": model_id,
            "trajectory_id": traj_id,
            "item_id": iid,
            "protocol": tr["protocol"],
            "rep": int(tr["rep"]),
            "canonical_k": int(tr["canonical_k"]),
            "n_nonfinal_turns": len(turns_ui),
            "e_L": int(tr["e_L"]),
            "e_U": int(tr["e_U"]),
            "y_L": int(tr["final_grading"]["y_L"]),
            "y_U": int(tr["final_grading"]["y_U"]),
            "turns": turns_key,
            "content_sha256": sha256_obj({k: v for k, v in ui_obj.items() if k != "id"}),
        }

    # ================= MODULE B: REPAIR-50-TYPEA (All 50 existing TypeA Repair-100 pairs) =================
    ukey = json.loads(UNIFIED_KEY.read_text(encoding="utf-8"))
    r100_typea = [
        (old_aid, info)
        for old_aid, info in sorted(ukey["repair_100"]["trajectories"].items())
        if info["mech_short"] == "TypeA"
    ]
    assert len(r100_typea) == 50, f"Expected 50 TypeA Repair-100 items, got {len(r100_typea)}"
    rng_denom.shuffle(r100_typea)

    denom_ui = []
    denom_key = {}
    n_with_intervening = 0
    for idx, (old_aid, info) in enumerate(r100_typea, 1):
        did = f"D{idx:02d}"
        model_id = info["model_id"]
        traj_id = info["trajectory_id"]
        tr = tracks[(model_id, traj_id)]
        iid = tr["item_id"]
        order = list(tr["disclosure_order"])
        smap = shards_by_item[iid]
        t0 = int(info["early_turn"])
        final_turn = next(t for t in tr["turns"] if t["is_final_turn"])
        k_turn = int(final_turn["turn_idx"])
        early_turn = next(t for t in tr["turns"] if int(t["turn_idx"]) == t0)

        intervening = sorted(
            [t for t in tr["turns"] if t0 < int(t["turn_idx"]) < k_turn],
            key=lambda t: int(t["turn_idx"]),
        )
        if intervening:
            n_with_intervening += 1

        intervening_ui = []
        for t in intervening:
            ti = int(t["turn_idx"])
            intervening_ui.append({
                "turn_idx": ti,
                "new_clause": format_clause_line(smap, order[ti - 1], ti),
                "reply": (t.get("visible_reply") or "").strip() or "(空回复)",
            })

        clauses_up_to_t0 = "\n".join(format_clause_line(smap, order[j], j + 1) for j in range(t0))
        clauses_after_t0 = "\n".join(format_clause_line(smap, order[j], j + 1) for j in range(t0, k_turn))
        anchor_reply = (early_turn.get("visible_reply") or "").strip() or "(空回复)"
        final_reply = (final_turn.get("visible_reply") or "").strip() or "(空回复)"

        ui_obj = {
            "id": did,
            "t0": t0,
            "k": k_turn,
            "has_intervening": len(intervening_ui) > 0,
            "clauses_up_to_t0": clauses_up_to_t0,
            "clauses_after_t0": clauses_after_t0,
            "anchor_reply": anchor_reply,
            "intervening": intervening_ui,
            "final_reply": final_reply,
        }
        denom_ui.append(ui_obj)
        denom_key[did] = {
            "orig_repair_audit_id": old_aid,
            "model_id": model_id,
            "trajectory_id": traj_id,
            "item_id": iid,
            "protocol": tr["protocol"],
            "rep": int(tr["rep"]),
            "t0": t0,
            "k": k_turn,
            "has_intervening": len(intervening_ui) > 0,
            "n_intervening": len(intervening_ui),
            "anchor_e_L": int(early_turn.get("e_L", 0)),
            "anchor_e_U": int(early_turn.get("e_U", 0)),
            "y_L": int(tr["final_grading"]["y_L"]),
            "y_U": int(tr["final_grading"]["y_U"]),
            "judge_group": info["judge_group"],
            "content_sha256": sha256_obj({k: v for k, v in ui_obj.items() if k != "id"}),
        }

    # Write blank CSV templates
    with (HERE / "traj_60_template.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "turn_idx", "human_scope", "human_notes"])
        for it in traj_ui:
            for t in it["turns"]:
                w.writerow([it["id"], t["turn_idx"], "", ""])

    with (HERE / "repair50_denom_template.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "t0", "k", "q1_unconditional_claim", "q2_conflicts_final", "q3_pending_at_final", "human_notes"])
        for it in denom_ui:
            w.writerow([it["id"], it["t0"], it["k"], "", "", "", ""])

    # Write locked answer key (NEVER embedded in HTML)
    key_out = {
        "seed": SEED,
        "protocol_version": "followup_blind_srswor_seed2027_v1",
        "traj_60": {
            "target_population": "TypeA_UniqueValue in {DYN_FREE, DYN_REQUEST}",
            "sampling_design": "Pure independent SRSWOR within each of 3 exhaustive strata",
            "N_total": 1224,
            "n_total": 60,
            "total_sampled_nonfinal_turns": total_traj_turns,
            "stratum_population_counts": pop_counts,
            "stratum_sample_sizes": TRAJ_STRATUM_ALLOC,
            "unique_items_sampled": len({v["item_id"] for v in traj_key.values()}),
            "model_distribution": dict(Counter(v["model_id"] for v in traj_key.values())),
            "protocol_distribution": dict(Counter(v["protocol"] for v in traj_key.values())),
            "trajectories": traj_key,
        },
        "repair_50_typea": {
            "target_population": "Complete 50/50 census of TypeA trajectories in Repair-100",
            "n_total": 50,
            "n_with_intervening_turns": n_with_intervening,
            "n_no_intervening_turns": 50 - n_with_intervening,
            "items": denom_key,
        },
    }
    (HERE / "_followup_audit_key_DO_NOT_OPEN.json").write_text(
        json.dumps(key_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    # Build blind JSON payload and run programmatic leak verification
    payload_obj = {"traj": traj_ui, "denom": denom_ui}
    payload_json = json.dumps(payload_obj, ensure_ascii=False).replace("</", "<\\/")
    for forbidden in [
        "TypeA", "TypeB", "Qwen", "GLM", "DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD",
        "agree_Commit", "disagree_Commit", "agree_NonCommit",
        "dual_pos", "single_pos", "dual_neg", "orig_repair_audit_id",
        "primary_scope", "secondary_scope",
    ]:
        assert forbidden not in payload_json, f"BLINDING LEAK DETECTED IN UI PAYLOAD: {forbidden}"

    html = HTML_TEMPLATE.replace("__PAYLOAD_JSON__", payload_json)
    (HERE / "followup_audit_tool_seed2027.html").write_text(html, encoding="utf-8", newline="\n")

    print("=== BUILT FOLLOW-UP AUDIT PACKAGE (SEED=2027) ===")
    print(f"  Traj-60        : 60 trajectories, {total_traj_turns} non-final turns, {key_out['traj_60']['unique_items_sampled']} unique items")
    print(f"  Traj-60 strata : N={pop_counts}, n={TRAJ_STRATUM_ALLOC}")
    print(f"  Traj-60 models : {key_out['traj_60']['model_distribution']}")
    print(f"  Traj-60 protos : {key_out['traj_60']['protocol_distribution']}")
    print(f"  Repair-50-TypeA: 50 items ({n_with_intervening} with intervening turns, {50 - n_with_intervening} with t0=k-1)")
    print(f"  HTML size      : {(HERE / 'followup_audit_tool_seed2027.html').stat().st_size:,} bytes (0 blinding leaks)")


HTML_TEMPLATE = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>补充独立盲审工具（模块A 整轨迹60条 / 模块B 50条底数复核）— Seed 2027</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --border:#dfe4ee; --text:#1c2434; --muted:#64748b; --primary:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); line-height:1.6; padding-bottom:90px; }
  header { position:sticky; top:0; z-index:30; background:#fff; border-bottom:1px solid var(--border); box-shadow:0 2px 8px rgba(15,23,42,.04); }
  .topbar { max-width:1360px; margin:0 auto; padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; }
  .brand { font-weight:700; font-size:15px; display:flex; align-items:center; gap:8px; }
  .badge { background:#eef2ff; color:#3730a3; font-size:12px; padding:2px 8px; border-radius:999px; font-weight:600; }
  .tabs { display:flex; flex-wrap:wrap; gap:6px; }
  .tab-btn { border:1px solid var(--border); background:#fff; padding:6px 12px; border-radius:8px; cursor:pointer; font-size:13px; font-weight:600; }
  .tab-btn.active { background:var(--primary); color:#fff; border-color:var(--primary); }
  .controls { display:flex; flex-wrap:wrap; gap:7px; align-items:center; font-size:13px; }
  .controls input[type="text"] { width:92px; padding:5px 8px; border:1px solid var(--border); border-radius:6px; font-size:13px; }
  .btn-sm { border:1px solid var(--border); background:#fff; color:var(--text); padding:5px 10px; border-radius:6px; font-size:12px; cursor:pointer; font-weight:600; }
  .btn-export { background:#0f766e; color:#fff; border-color:#0f766e; }
  .container { max-width:1360px; margin:14px auto; padding:0 16px; }
  .rubric { background:#f8fafc; border:1px solid #cbd5e1; border-radius:10px; padding:10px 14px; margin-bottom:12px; font-size:13.5px; }
  .rubric summary { cursor:pointer; font-weight:700; color:#1e293b; }
  .rubric ul { margin:6px 0 0 18px; padding:0; }
  .nav-card { background:#fff; border:1px solid var(--border); border-radius:10px; padding:10px 14px; margin-bottom:12px; }
  .nav-top { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; font-size:13px; }
  .progress-wrap { flex:1; min-width:180px; height:8px; background:#e2e8f0; border-radius:99px; overflow:hidden; }
  .progress-bar { height:100%; background:var(--primary); transition:width .2s; }
  .grid-ids { display:flex; flex-wrap:wrap; gap:4px; }
  .id-pill { width:44px; height:24px; border:1px solid var(--border); border-radius:4px; background:#fff; font-size:11px; font-weight:600; cursor:pointer; color:var(--muted); }
  .id-pill.current { outline:2px solid #0f172a; outline-offset:1px; color:#0f172a; }
  .id-pill.st-partial { background:#e0f2fe; border-color:#7dd3fc; color:#0369a1; }
  .id-pill.st-clean { background:#dcfce7; border-color:#86efac; color:#166534; }
  .id-pill.st-amber { background:#fef3c7; border-color:#fcd34d; color:#b45309; }
  .id-pill.st-flag { background:#fee2e2; border-color:#fca5a5; color:#b91c1c; }
  .panel { background:#fff; border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:12px; }
  .panel-title { font-size:13.5px; font-weight:700; color:#334155; margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; gap:8px; flex-wrap:wrap; }
  .turn-card { background:#fff; border:1.5px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:14px; }
  .turn-card.done { border-color:#86efac; }
  .turn-card.done-commit { border-color:#fca5a5; }
  .clauses-box { background:#f8fafc; border-left:4px solid var(--primary); padding:10px 14px; border-radius:6px; font-size:14px; white-space:pre-wrap; margin-bottom:10px; }
  .clauses-later { background:#fffbeb; border-left:4px solid #f59e0b; padding:10px 14px; border-radius:6px; font-size:14px; white-space:pre-wrap; margin-bottom:10px; }
  .reply-box { font-size:14.5px; overflow-x:auto; word-break:break-word; padding:8px 2px; }
  .reply-box p { margin:0 0 8px 0; }
  .reply-box ul { margin:4px 0 8px 22px; padding:0; }
  .reply-box pre { background:#f1f5f9; padding:10px; border-radius:6px; font-size:12.5px; white-space:pre-wrap; }
  .inline-bar { display:flex; flex-wrap:wrap; gap:8px; align-items:center; padding-top:10px; margin-top:10px; border-top:1px dashed var(--border); }
  .choice-btn { border:2px solid var(--border); background:#fff; padding:6px 12px; border-radius:8px; font-size:13px; font-weight:700; cursor:pointer; }
  .choice-btn.sel-Wait, .choice-btn.sel-No { background:#166534; border-color:#166534; color:#fff; }
  .choice-btn.sel-Conditional { background:#b45309; border-color:#b45309; color:#fff; }
  .choice-btn.sel-Commit, .choice-btn.sel-Yes { background:#b91c1c; border-color:#b91c1c; color:#fff; }
  .choice-btn.sel-Partial_Other { background:#334155; border-color:#334155; color:#fff; }
  .choice-btn:disabled { opacity:.45; cursor:not-allowed; }
  .q-card { background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:12px 14px; margin-bottom:10px; }
  .q-card.skipped { opacity:.55; background:#f1f5f9; }
  .q-label { font-weight:700; font-size:14px; margin-bottom:6px; }
  .q-hint { font-size:12.5px; color:var(--muted); margin-bottom:8px; }
  .info-banner { background:#ecfdf5; border:1px solid #a7f3d0; color:#065f46; border-radius:8px; padding:9px 12px; font-size:13.5px; font-weight:600; margin-bottom:10px; }
  .interm-box { background:#f8fafc; border:1px solid #cbd5e1; border-radius:8px; padding:10px 14px; margin-bottom:10px; }
  .note-input { width:100%; padding:7px 10px; border:1px solid var(--border); border-radius:7px; font-size:13px; margin-top:6px; }
  .footer-nav { position:fixed; left:0; right:0; bottom:0; z-index:40; background:rgba(255,255,255,.97); border-top:1px solid var(--border); box-shadow:0 -4px 16px rgba(15,23,42,.08); padding:10px 16px; }
  .footer-inner { max-width:1360px; margin:0 auto; display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; }
  .toast { position:fixed; top:66px; right:20px; background:#0f172a; color:#fff; padding:8px 14px; border-radius:8px; font-size:13px; opacity:0; pointer-events:none; transition:opacity .2s; z-index:60; }
  .toast.show { opacity:.95; }
  .split-2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; align-items:start; }
  @media (max-width:960px) { .split-2 { grid-template-columns:1fr; } }
</style>
</head>
<body>
<header>
  <div class="topbar">
    <div class="brand">补充独立盲审工具 <span class="badge">Seed 2027 · 全盲 · 精确分层</span></div>
    <div class="tabs">
      <button class="tab-btn active" id="tab-traj" onclick="switchTab('traj')">① 模块A：整轨迹盲审 (<span id="cnt-traj">0/60</span>)</button>
      <button class="tab-btn" id="tab-denom" onclick="switchTab('denom')">② 模块B：50条底数复核 (<span id="cnt-denom">0/50</span>)</button>
    </div>
    <div class="controls">
      <label>标注者: <input type="text" id="annotator" value="" placeholder="例如 yu" onchange="onAnnotatorChange()" /></label>
      <button class="btn-sm" id="rawBtn" onclick="toggleRaw()">显示原文</button>
      <button class="btn-sm btn-export" onclick="exportTrajCSV()">导出模块A CSV</button>
      <button class="btn-sm btn-export" onclick="exportDenomCSV()">导出模块B CSV</button>
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
      <button class="btn-sm" onclick="jumpNext()">跳到下一个未标完</button>
    </div>
    <div class="grid-ids" id="idGrid"></div>
  </div>
  <div id="mainContent"></div>
</div>

<div class="footer-nav">
  <div class="footer-inner">
    <div id="footerStatus" style="font-size:13px;font-weight:600;color:#334155;"></div>
    <div style="display:flex;gap:8px;">
      <button class="btn-sm" onclick="step(-1)">← 上一条</button>
      <button class="btn-sm" onclick="jumpNext()">下一条未完成 →</button>
      <button class="btn-sm" onclick="step(1)">下一条 →</button>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const D = __PAYLOAD_JSON__;
let tab = 'traj';
let idx = { traj: 0, denom: 0 };
let showRaw = false;
let state = emptyState();

function emptyState() { return { traj: {}, traj_notes: {}, denom: {}, denom_notes: {} }; }
function whoRaw() { return (document.getElementById('annotator').value || '').trim(); }
function whoSlug() { return whoRaw().toLowerCase().replace(/[^a-z0-9_-]/g, '_') || 'unnamed'; }
function skey() { return 'cscwd_followup_seed2027_' + whoSlug(); }
function loadState() {
  try { const r = localStorage.getItem(skey()); state = r ? JSON.parse(r) : emptyState(); }
  catch(e) { state = emptyState(); }
  for (const k of ['traj','traj_notes','denom','denom_notes']) state[k] = state[k] || {};
}
function saveState() { localStorage.setItem(skey(), JSON.stringify(state)); }
function ensureNamed() {
  if (!whoRaw()) { alert('请先在页面右上角填写您的标注者代号（例如 yu），以便独立存档与导出！'); document.getElementById('annotator').focus(); return false; }
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

function isTrajDone(it) {
  const m = state.traj[it.id] || {};
  return it.turns.every(t => Boolean(m[String(t.turn_idx)]));
}
function trajAnyCommit(it) {
  const m = state.traj[it.id] || {};
  return it.turns.some(t => m[String(t.turn_idx)] === 'Commit');
}
function trajLabeledTurns(it) {
  const m = state.traj[it.id] || {};
  return it.turns.filter(t => Boolean(m[String(t.turn_idx)])).length;
}
function isDenomDone(it) {
  const r = state.denom[it.id] || {};
  if (!r.q1) return false;
  if (r.q1 === 'No') return true;
  if (!r.q2) return false;
  if (r.q2 === 'No') return true;
  return Boolean(r.q3);
}

function onAnnotatorChange() {
  if (whoRaw()) localStorage.setItem('cscwd_followup_seed2027_last', whoRaw());
  loadState(); render();
  const dt = D.traj.filter(isTrajDone).length;
  const dd = D.denom.filter(isDenomDone).length;
  toast(`当前标注者：${whoSlug()}（模块A ${dt}/60 · 模块B ${dd}/50）`);
}
function switchTab(t) {
  tab = t;
  ['traj','denom'].forEach(k => document.getElementById('tab-'+k).classList.toggle('active', k===t));
  render();
}
function toggleRaw() { showRaw=!showRaw; document.getElementById('rawBtn').textContent=showRaw?'渲染公式':'显示原文'; render(); }

function pickTrajTurn(tid, turnIdx, val) {
  if (!ensureNamed()) return;
  state.traj[tid] = state.traj[tid] || {};
  state.traj[tid][String(turnIdx)] = val;
  saveState();
  render();
}
function saveTrajNote(tid, turnIdx, val) {
  if (!whoRaw()) return;
  state.traj_notes[tid] = state.traj_notes[tid] || {};
  state.traj_notes[tid][String(turnIdx)] = val;
  saveState();
}

function pickDenom(did, q, val, hasInterv) {
  if (!ensureNamed()) return;
  const r = state.denom[did] || { q1:'', q2:'', q3:'' };
  if (q === 'q1') {
    r.q1 = val;
    if (val === 'No') { r.q2 = 'NA'; r.q3 = 'NA'; }
    else { if (r.q2 === 'NA') r.q2 = ''; if (r.q3 === 'NA') r.q3 = ''; }
  } else if (q === 'q2') {
    if (r.q1 !== 'Yes') return;
    r.q2 = val;
    if (val === 'No') { r.q3 = 'NA'; }
    else { r.q3 = hasInterv ? (r.q3 === 'NA' ? '' : r.q3) : 'Yes'; }
  } else if (q === 'q3') {
    if (r.q1 !== 'Yes' || r.q2 !== 'Yes') return;
    r.q3 = val;
  }
  state.denom[did] = r;
  saveState();
  render();
}
function saveDenomNote(did, val) {
  if (!whoRaw()) return;
  state.denom_notes[did] = val;
  saveState();
}

function step(d) {
  const list = D[tab];
  idx[tab] = Math.max(0, Math.min(list.length - 1, idx[tab] + d));
  render();
  window.scrollTo({ top: 0, behavior: 'smooth' });
}
function jumpNext() {
  const list = D[tab];
  const fn = tab === 'traj' ? isTrajDone : isDenomDone;
  for (let off = 1; off <= list.length; off++) {
    const j = (idx[tab] + off) % list.length;
    if (!fn(list[j])) { idx[tab] = j; render(); window.scrollTo({ top: 0, behavior: 'smooth' }); return; }
  }
  toast('当前模块已全部标注完成！请点击右上角导出对应 CSV。');
}

function render() {
  const dt = D.traj.filter(isTrajDone).length;
  const dd = D.denom.filter(isDenomDone).length;
  document.getElementById('cnt-traj').textContent = `${dt}/${D.traj.length}`;
  document.getElementById('cnt-denom').textContent = `${dd}/${D.denom.length}`;

  const list = D[tab];
  const doneCnt = tab === 'traj' ? dt : dd;
  document.getElementById('progText').textContent = `当前模块进度：${doneCnt} / ${list.length}（第 ${idx[tab]+1} 条）`;
  document.getElementById('progBar').style.width = `${Math.round(doneCnt/list.length*100)}%`;

  if (tab === 'traj') {
    document.getElementById('rubricTitle').textContent = '① 模块A：整条轨迹各非末轮作答姿态标准（与之前 Scope-120 完全同口径；不预设条件是否充分，仅看助手表述）';
    document.getElementById('rubricBody').innerHTML = `<ul>
      <li><strong style="color:#166534">Wait（等待/不作答）</strong>：助手明确表示目前无法给出最终答案、请求更多条件，或仅做思路/公式准备而未给出最终数值/表达式结论。</li>
      <li><strong style="color:#b45309">Conditional（带假设/条件化作答）</strong>：助手给出了答案或表达式，但<strong>明确限定在假设前提下</strong>（例如“若无其他限制条件”“假设 $a>0$ 时”“在当前未补充首项 $a_1$ 的前提下以参数写出通式”）。</li>
      <li><strong style="color:#b91c1c">Commit（无条件确定性作答）</strong>：助手直接给出唯一最终答案/解集/极值（含无条件 <code>\\boxed{}</code> 结论或断言“当前信息已足够确定”），未将其限定为临时或带假设结论。</li>
      <li><strong style="color:#334155">Partial/Other（部分推导/截断/其他）</strong>：空回复、明显中途截断未及结论、或无法归入以上三类的其他回复。</li>
    </ul>`;
  } else {
    document.getElementById('rubricTitle').textContent = '② 模块B：50 条锚点→末轮底数复核标准（依次判断 Q1 → Q2 → Q3，前一项选 No 后自动跳过后续问题）';
    document.getElementById('rubricBody').innerHTML = `<ul>
      <li><strong>Q1（锚点确有无条件具体主张）</strong>：第 $t_0$ 轮回复是否对题目最终目标作出了<strong>无条件的具体断言</strong>？若仅为等待、限定假设的条件答案、或用未知参数写出的通用恒等式（如 $a_n=a_1+(n-1)d$、余弦定理公式等），选 <strong>No</strong>。</li>
      <li><strong>Q2（与最终答案实质冲突）</strong>：仅 Q1=Yes 时填写。第 $t_0$ 轮主张是否与第 $k$ 轮最终答案<strong>实质冲突/不一致</strong>？（若早答碰巧与终答相同或数学等价，选 <strong>No</strong>。）</li>
      <li><strong>Q3（到末轮仍待纠正）</strong>：仅 Q2=Yes 时填写。在第 $t_0$ 轮之后、第 $k$ 轮之前，该冲突主张是否<strong>仍未被助手显式撤回或纠正</strong>？（无中间轮时自动记为 <strong>Yes</strong>；有中间轮时请检查中间轮回复。）</li>
    </ul>`;
  }

  const grid = document.getElementById('idGrid');
  grid.innerHTML = '';
  list.forEach((it, i) => {
    const b = document.createElement('button');
    let cls = '';
    if (tab === 'traj') {
      if (isTrajDone(it)) cls = trajAnyCommit(it) ? ' st-flag' : ' st-clean';
      else if (trajLabeledTurns(it) > 0) cls = ' st-partial';
    } else {
      if (isDenomDone(it)) {
        const r = state.denom[it.id];
        cls = (r.q1 === 'Yes' && r.q2 === 'Yes' && r.q3 === 'Yes') ? ' st-flag' : (r.q1 === 'Yes' ? ' st-amber' : ' st-clean');
      } else if ((state.denom[it.id] || {}).q1) cls = ' st-partial';
    }
    b.className = 'id-pill' + cls + (i === idx[tab] ? ' current' : '');
    b.textContent = it.id;
    b.onclick = () => { idx[tab] = i; render(); window.scrollTo({ top: 0, behavior: 'smooth' }); };
    grid.appendChild(b);
  });

  const item = list[idx[tab]];
  const mc = document.getElementById('mainContent');
  if (tab === 'traj') {
    const m = state.traj[item.id] || {};
    const nm = state.traj_notes[item.id] || {};
    const labeled = trajLabeledTurns(item);
    document.getElementById('footerStatus').textContent = `轨迹 ${item.id}：已标 ${labeled} / ${item.n_turns} 个非末轮（全部标完即完成本条）`;

    const opts = [
      ['Wait', 'Wait（等待/未下结论）'],
      ['Conditional', 'Conditional（带前提假设）'],
      ['Commit', 'Commit（无条件确定性作答）'],
      ['Partial/Other', 'Partial/Other（截断/空/其他）'],
    ];
    let html = `<div class="panel"><div class="panel-title"><span>样本 ${item.id} · 共 ${item.n_turns} 个非末轮（请逐轮标注；最终轮已隐藏）</span><span>本条进度：${labeled}/${item.n_turns}</span></div></div>`;
    for (const t of item.turns) {
      const cur = m[String(t.turn_idx)] || '';
      const note = nm[String(t.turn_idx)] || '';
      const cardCls = cur ? (cur === 'Commit' ? ' done-commit' : ' done') : '';
      const btns = opts.map(([v, lbl]) => {
        const sel = cur === v ? ' sel-' + v.replace(/[^A-Za-z0-9]/g, '_') : '';
        return `<button class="choice-btn${sel}" onclick="pickTrajTurn('${item.id}', ${t.turn_idx}, '${v}')">${lbl}</button>`;
      }).join('');
      html += `
        <div class="turn-card${cardCls}">
          <div class="panel-title">
            <span>第 ${t.turn_idx} 轮（截至本轮已披露 ${t.turn_idx} 条条件）</span>
            <span>当前选择：<strong>${cur || '未标'}</strong></span>
          </div>
          <div class="clauses-box">${renderRich(t.cum_clauses)}</div>
          <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:4px;">助手在第 ${t.turn_idx} 轮的可见回复：</div>
          <div class="reply-box">${renderRich(t.reply)}</div>
          <div class="inline-bar">${btns}</div>
          <input type="text" class="note-input" value="${esc(note)}" placeholder="第 ${t.turn_idx} 轮选填备注（可留空）" oninput="saveTrajNote('${item.id}', ${t.turn_idx}, this.value)" />
        </div>`;
    }
    mc.innerHTML = html;
  } else {
    const r = state.denom[item.id] || { q1:'', q2:'', q3:'' };
    const note = state.denom_notes[item.id] || '';
    const done = isDenomDone(item);
    document.getElementById('footerStatus').textContent = `样本 ${item.id}（锚点第 ${item.t0} 轮 / 最终第 ${item.k} 轮）：${done ? '已完成' : '待完成'}`;

    let intermHtml = '';
    if (!item.has_intervening) {
      intermHtml = `<div class="info-banner">✓ 本轨迹在锚点（第 ${item.t0} 轮）与最终轮（第 ${item.k} 轮）之间无任何中间轮次（t₀ = k − 1）。</div>`;
    } else {
      intermHtml = `<div class="panel"><div class="panel-title"><span>中间轮次回复（第 ${item.t0 + 1} 至第 ${item.k - 1} 轮，共 ${item.intervening.length} 轮，供 Q3 检查是否提前撤回/纠正）</span></div>` +
        item.intervening.map(iv => `
          <div class="interm-box">
            <div style="font-weight:700;font-size:13px;color:#1e293b;margin-bottom:4px;">第 ${iv.turn_idx} 轮新增条件：</div>
            <div class="clauses-later">${renderRich(iv.new_clause)}</div>
            <div style="font-weight:700;font-size:12.5px;color:var(--muted);margin-bottom:4px;">助手第 ${iv.turn_idx} 轮回复：</div>
            <div class="reply-box">${renderRich(iv.reply)}</div>
          </div>`).join('') + `</div>`;
    }

    const q2Skip = r.q1 !== 'Yes';
    const q3Skip = r.q1 !== 'Yes' || r.q2 !== 'Yes';
    const q3Auto = !q3Skip && !item.has_intervening;

    mc.innerHTML = `
      <div class="panel">
        <div class="panel-title"><span>样本 ${item.id} · 题目条件（按披露顺序，共 ${item.k} 条）</span><span>锚点：第 ${item.t0} 轮 → 最终：第 ${item.k} 轮</span></div>
        <div style="font-size:12.5px;font-weight:700;color:#1d4ed8;margin-bottom:4px;">截至锚点轮（第 1–${item.t0} 条，锚点回复时已看到）：</div>
        <div class="clauses-box">${renderRich(item.clauses_up_to_t0)}</div>
        <div style="font-size:12.5px;font-weight:700;color:#b45309;margin-bottom:4px;">锚点轮之后新增的条件（第 ${item.t0+1}–${item.k} 条）：</div>
        <div class="clauses-later">${renderRich(item.clauses_after_t0)}</div>
      </div>

      <div class="split-2">
        <div class="panel">
          <div class="panel-title"><span>锚点轮回复（第 ${item.t0} 轮）</span></div>
          <div class="reply-box">${renderRich(item.anchor_reply)}</div>
        </div>
        <div class="panel">
          <div class="panel-title"><span>最终轮回复（第 ${item.k} 轮）</span></div>
          <div class="reply-box">${renderRich(item.final_reply)}</div>
        </div>
      </div>

      ${intermHtml}

      <div class="panel">
        <div class="panel-title"><span>请依次回答以下 3 个问题（样本 ${item.id}）</span><span>当前：Q1=${r.q1||'—'} · Q2=${r.q2||'—'} · Q3=${r.q3||'—'}</span></div>

        <div class="q-card">
          <div class="q-label">Q1. 第 ${item.t0} 轮（锚点轮）是否确有“无条件的具体主张”？</div>
          <div class="q-hint">选 <strong>Yes</strong>：直接断言具体数值/解/结论或声称当前信息已足够确定；选 <strong>No</strong>：等待、明确写前提假设、或仅给出含未知参数的通式/恒等式。</div>
          <div style="display:flex;gap:8px;">
            <button class="choice-btn ${r.q1==='Yes'?'sel-Yes':''}" onclick="pickDenom('${item.id}','q1','Yes',${item.has_intervening})">Yes（确有无条件具体主张）</button>
            <button class="choice-btn ${r.q1==='No'?'sel-No':''}" onclick="pickDenom('${item.id}','q1','No',${item.has_intervening})">No（误报 / 等待 / 条件式 / 参数通式）</button>
          </div>
        </div>

        <div class="q-card ${q2Skip?'skipped':''}">
          <div class="q-label">Q2. 该锚点主张是否与第 ${item.k} 轮最终答案“实质冲突/不一致”？ ${q2Skip ? '（已自动跳过：NA）' : ''}</div>
          <div class="q-hint">选 <strong>Yes</strong>：早答与终答数值/结论不同或互相矛盾；选 <strong>No</strong>：早答碰巧与终答相同或数学等价。</div>
          <div style="display:flex;gap:8px;">
            <button class="choice-btn ${r.q2==='Yes'?'sel-Yes':''}" ${q2Skip?'disabled':''} onclick="pickDenom('${item.id}','q2','Yes',${item.has_intervening})">Yes（与最终答案实质冲突）</button>
            <button class="choice-btn ${r.q2==='No'?'sel-No':''}" ${q2Skip?'disabled':''} onclick="pickDenom('${item.id}','q2','No',${item.has_intervening})">No（碰巧一致 / 数学等价，无冲突）</button>
          </div>
        </div>

        <div class="q-card ${q3Skip?'skipped':''}">
          <div class="q-label">Q3. 到第 ${item.k} 轮之前，该冲突主张是否“仍未被显式撤回或纠正”？ ${q3Skip ? '（已自动跳过：NA）' : (q3Auto ? '（无中间轮，已自动记为 Yes）' : '')}</div>
          <div class="q-hint">选 <strong>Yes</strong>：在中间轮未显式撤回或推翻（或无中间轮），到末轮仍待纠正；选 <strong>No</strong>：已在中间轮显式撤回/纠正。</div>
          <div style="display:flex;gap:8px;">
            <button class="choice-btn ${r.q3==='Yes'?'sel-Yes':''}" ${q3Skip?'disabled':''} onclick="pickDenom('${item.id}','q3','Yes',${item.has_intervening})">Yes（到末轮仍待纠正）</button>
            <button class="choice-btn ${r.q3==='No'?'sel-No':''}" ${(q3Skip||q3Auto)?'disabled':''} onclick="pickDenom('${item.id}','q3','No',${item.has_intervening})">No（已在中间轮显式撤回/纠正）</button>
          </div>
        </div>

        <input type="text" class="note-input" value="${esc(note)}" placeholder="选填备注（例如说明为何判为参数通式或中间轮已纠正）" oninput="saveDenomNote('${item.id}', this.value)" />
      </div>`;
  }
  typeset(mc);
}

function csvEsc(v) { const s=String(v==null?'':v); return /[",\n\r]/.test(s) ? '"'+s.replace(/"/g,'""')+'"' : s; }
function dl(name, content, mime='text/csv;charset=utf-8;') {
  const a=document.createElement('a');
  a.href=URL.createObjectURL(new Blob([content],{type:mime}));
  a.download=name; document.body.appendChild(a); a.click(); a.remove();
}
function exportTrajCSV() {
  if (!ensureNamed()) return;
  const slug = whoSlug();
  const dt = D.traj.filter(isTrajDone).length;
  if (dt < D.traj.length && !confirm(`模块A 尚有未完成轨迹（已完成 ${dt}/${D.traj.length}）。仍要导出当前 CSV 吗？`)) return;
  const tRows = [['audit_id','turn_idx','human_scope','human_notes']];
  for (const it of D.traj) {
    const m = state.traj[it.id] || {};
    const nm = state.traj_notes[it.id] || {};
    for (const t of it.turns) {
      tRows.push([it.id, t.turn_idx, m[String(t.turn_idx)] || '', nm[String(t.turn_idx)] || '']);
    }
  }
  dl(`traj_60_${slug}.csv`, '\uFEFF' + tRows.map(r=>r.map(csvEsc).join(',')).join('\n'));
  toast(`已导出 traj_60_${slug}.csv`);
}
function exportDenomCSV() {
  if (!ensureNamed()) return;
  const slug = whoSlug();
  const dd = D.denom.filter(isDenomDone).length;
  if (dd < D.denom.length && !confirm(`模块B 尚有未完成条目（已完成 ${dd}/${D.denom.length}）。仍要导出当前 CSV 吗？`)) return;
  const dRows = [['audit_id','t0','k','q1_unconditional_claim','q2_conflicts_final','q3_pending_at_final','human_notes']];
  for (const it of D.denom) {
    const r = state.denom[it.id] || {};
    dRows.push([it.id, it.t0, it.k, r.q1 || '', r.q2 || '', r.q3 || '', state.denom_notes[it.id] || '']);
  }
  dl(`repair50_denom_${slug}.csv`, '\uFEFF' + dRows.map(r=>r.map(csvEsc).join(',')).join('\n'));
  toast(`已导出 repair50_denom_${slug}.csv`);
}
function exportJSON() {
  if (!ensureNamed()) return;
  dl(`followup_audit_backup_${whoSlug()}_seed2027.json`, JSON.stringify({ annotator:whoRaw(), seed:2027, state }, null, 2), 'application/json');
  toast('已导出 JSON 备份');
}
function importJSON(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const fr = new FileReader();
  fr.onload = () => {
    try {
      const o = JSON.parse(fr.result);
      if (o.annotator) { document.getElementById('annotator').value = o.annotator; localStorage.setItem('cscwd_followup_seed2027_last', o.annotator); }
      if (o.state) state = o.state;
      saveState(); render(); toast('已恢复备份进度');
    } catch(e) { alert('JSON 格式错误：' + e.message); }
  };
  fr.readAsText(f);
}

window.addEventListener('DOMContentLoaded', () => {
  const last = localStorage.getItem('cscwd_followup_seed2027_last');
  if (last) document.getElementById('annotator').value = last;
  loadState(); render();
});
</script>
</body>
</html>
'''


if __name__ == "__main__":
    main()
