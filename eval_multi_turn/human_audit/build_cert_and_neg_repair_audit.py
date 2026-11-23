#!/usr/bin/env python3
"""Build reproducible SEED=2026 audit package for:
  1. 30-item stratified SymPy formalization-faithfulness audit (20 TypeA + 10 TypeB across G1-G4).
  2. 20-item stratified negative-control Repair audit drawn from the 463 dual-negative E_U=1 trajectories.

Generates:
  - cert_audit_30.csv
  - neg_repair_audit_20.csv
  - _cert_neg_repair_key.json
  - cert_and_neg_repair_audit_seed2026.html (single-file offline KaTeX audit UI)
"""

from __future__ import annotations

import csv
import gzip
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN_TEST = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_test_100.jsonl"
FROZEN_CERTS = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_certificates.jsonl"
FROZEN_REPORTS = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_reports.jsonl"
V13_DIR = ROOT / "eval_multi_turn" / "runs" / "runs_16k_v13_blind"
OUT_DIR = Path(__file__).resolve().parent

SEED = 2026

# Proportional stratification of 30/100 TEST items across (macro_subfield, verified_mechanism)
CERT_QUOTAS = {
    ("G1_Func_Calc_Trig", "TypeA_UniqueValue"): 6,  # out of 19
    ("G2_Alg_Ineq", "TypeA_UniqueValue"): 4,        # out of 13
    ("G3_Seq_Comb_NT", "TypeA_UniqueValue"): 6,     # out of 21
    ("G4_Geom_Analytic", "TypeA_UniqueValue"): 4,   # out of 15
    ("G1_Func_Calc_Trig", "TypeB_Extremum"): 3,     # out of 10
    ("G2_Alg_Ineq", "TypeB_Extremum"): 3,           # out of 10
    ("G3_Seq_Comb_NT", "TypeB_Extremum"): 1,        # out of 3
    ("G4_Geom_Analytic", "TypeB_Extremum"): 3,      # out of 9
}

NEG_REPAIR_MODEL_QUOTAS = {
    "Qwen/Qwen3-8B": 7,
    "zai-org/GLM-Z1-9B-0414": 7,
    "Qwen/Qwen3-14B": 6,
}


def read_jsonl(path: Path) -> list[dict]:
    if path.exists():
        return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]
    gz = Path(str(path) + ".gz")
    if gz.exists():
        with gzip.open(gz, "rt", encoding="utf-8") as f:
            return [json.loads(l) for l in f if l.strip()]
    raise FileNotFoundError(f"Neither {path} nor {gz} exists.")


def format_all_clauses(shards: list[dict], order: list[str]) -> str:
    smap = {s["shard_id"]: (s.get("shard") or s.get("text") or "").strip() for s in shards}
    return "\n".join(f"[第{i + 1}轮披露/{sid}] {smap.get(sid, sid)}" for i, sid in enumerate(order))


def main() -> None:
    rng = random.Random(SEED)

    items = read_jsonl(FROZEN_TEST)
    items_by_id = {x["item_id"]: x for x in items}
    certs_by_id = {c["item_id"]: c["certificate"] for c in read_jsonl(FROZEN_CERTS)}
    reps_by_id = {r["item_id"]: r for r in read_jsonl(FROZEN_REPORTS)}

    # --- 1. Sample 30 items for SymPy formalization-faithfulness audit ---
    buckets: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for it in items:
        buckets[(it["macro_subfield"], it["verified_mechanism"])].append(it)

    picked_certs: list[dict] = []
    for bk in sorted(CERT_QUOTAS.keys()):
        pool = list(buckets[bk])
        rng.shuffle(pool)
        n_q = CERT_QUOTAS[bk]
        assert len(pool) >= n_q
        picked_certs.extend(pool[:n_q])
    rng.shuffle(picked_certs)
    assert len(picked_certs) == 30

    cert_records = []
    cert_key = {}
    for idx, it in enumerate(picked_certs, 1):
        cid = f"C{idx:02d}"
        iid = it["item_id"]
        cert = certs_by_id[iid]
        rep = reps_by_id[iid]
        shards_list = [
            {"id": s["shard_id"], "text": (s.get("shard") or s.get("text") or "").strip()}
            for s in it["shards"]
        ]
        cert_records.append(
            {
                "id": cid,
                "item_id": iid,
                "subfield": it["macro_subfield"],
                "mech": "TypeA" if it["verified_mechanism"] == "TypeA_UniqueValue" else "TypeB",
                "k": int(it["canonical_k"]),
                "question": it["question"],
                "final_answer": str(it["final_answer"]),
                "shards": shards_list,
                "certificate": cert,
                "report": {
                    "full_answer_verified": rep.get("full_answer_verified"),
                    "leave_one_out_verified": rep.get("leave_one_out_verified"),
                },
            }
        )
        cert_key[cid] = {
            "item_id": iid,
            "macro_subfield": it["macro_subfield"],
            "verified_mechanism": it["verified_mechanism"],
            "canonical_k": int(it["canonical_k"]),
        }

    # --- 2. Sample 20 dual-negative Repair controls from the 463 dual-negative E_U=1 trajectories ---
    neg_pools: dict[str, list[dict]] = defaultdict(list)
    total_eu1 = 0
    total_dual_neg = 0
    for jf in sorted(V13_DIR.glob("judged_test100_*_v13_blind.jsonl")):
        if jf.name.endswith("_judge_cache.jsonl"):
            continue
        for tr in read_jsonl(jf):
            if not tr["protocol"].startswith("DYN_"):
                continue
            if tr.get("e_U") != 1:
                continue
            total_eu1 += 1
            rep = tr.get("repair") or {}
            if rep.get("primary_retracted") or rep.get("secondary_retracted"):
                continue
            total_dual_neg += 1
            first_c = next((t for t in tr["turns"][:-1] if t.get("e_U") == 1), None)
            if first_c is None:
                continue
            neg_pools[tr["model_id"]].append(
                {
                    "model_id": tr["model_id"],
                    "trajectory_id": tr["trajectory_id"],
                    "item_id": tr["item_id"],
                    "protocol": tr["protocol"],
                    "verified_mechanism": tr["verified_mechanism"],
                    "canonical_k": int(tr["canonical_k"]),
                    "all_clauses": format_all_clauses(items_by_id[tr["item_id"]]["shards"], tr["disclosure_order"]),
                    "early_turn": int(first_c["turn_idx"]),
                    "early_reply": (first_c.get("visible_reply") or "").strip(),
                    "final_reply": (tr["turns"][-1].get("visible_reply") or "").strip(),
                }
            )
    assert total_eu1 == 489 and total_dual_neg == 463, (total_eu1, total_dual_neg)

    picked_neg: list[dict] = []
    used_items_neg: set[str] = set()
    for mid in sorted(NEG_REPAIR_MODEL_QUOTAS.keys()):
        q_n = NEG_REPAIR_MODEL_QUOTAS[mid]
        pool = list(neg_pools[mid])
        rng.shuffle(pool)
        chosen: list[dict] = []
        # Prefer unique item_ids across the 20 controls
        for rec in pool:
            if len(chosen) >= q_n:
                break
            if rec["item_id"] not in used_items_neg:
                chosen.append(rec)
                used_items_neg.add(rec["item_id"])
        if len(chosen) < q_n:
            seen_t = {(x["model_id"], x["trajectory_id"]) for x in chosen}
            for rec in pool:
                if len(chosen) >= q_n:
                    break
                if (rec["model_id"], rec["trajectory_id"]) not in seen_t:
                    chosen.append(rec)
        picked_neg.extend(chosen)
    rng.shuffle(picked_neg)
    assert len(picked_neg) == 20

    neg_records = []
    neg_key = {}
    for idx, rec in enumerate(picked_neg, 1):
        nid = f"N{idx:02d}"
        neg_records.append(
            {
                "id": nid,
                "early_turn": rec["early_turn"],
                "clauses": rec["all_clauses"],
                "early_reply": rec["early_reply"],
                "final_reply": rec["final_reply"],
            }
        )
        neg_key[nid] = {
            "model_id": rec["model_id"],
            "trajectory_id": rec["trajectory_id"],
            "item_id": rec["item_id"],
            "protocol": rec["protocol"],
            "verified_mechanism": rec["verified_mechanism"],
            "canonical_k": rec["canonical_k"],
            "early_turn": rec["early_turn"],
        }

    # Write blank template CSVs + locked key
    with (OUT_DIR / "cert_audit_30.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "item_id", "verified_mechanism", "macro_subfield", "human_verdict", "human_notes"])
        for r in cert_records:
            w.writerow([r["id"], r["item_id"], r["mech"], r["subfield"], "", ""])

    with (OUT_DIR / "neg_repair_audit_20.csv").open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "early_turn", "human_retracted", "human_notes"])
        for r in neg_records:
            w.writerow([r["id"], r["early_turn"], "", ""])

    key_out = {
        "seed": SEED,
        "cert_quotas": {f"{k[0]}__{k[1]}": v for k, v in CERT_QUOTAS.items()},
        "neg_repair_pool_size": total_dual_neg,
        "neg_repair_unique_items": len({v["item_id"] for v in neg_key.values()}),
        "cert_30": cert_key,
        "neg_repair_20": neg_key,
    }
    (OUT_DIR / "_cert_neg_repair_key.json").write_text(json.dumps(key_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    # Build single-file HTML tool
    payload_json = json.dumps({"certs": cert_records, "neg_repair": neg_records}, ensure_ascii=False).replace("</", "<\\/")

    html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>证书形式化保真度 (30题) + Repair 阴性对照 (20条) 核对工具 — Seed 2026</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --border:#dfe4ee; --text:#1c2434; --muted:#64748b; --primary:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); line-height:1.62; padding-bottom:120px; }
  header { position:sticky; top:0; z-index:30; background:#fff; border-bottom:1px solid var(--border); box-shadow:0 2px 8px rgba(15,23,42,.04); }
  .topbar { max-width:1360px; margin:0 auto; padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; }
  .brand { font-weight:700; font-size:15px; display:flex; align-items:center; gap:8px; }
  .badge { background:#eef2ff; color:#3730a3; font-size:12px; padding:2px 8px; border-radius:999px; font-weight:600; }
  .tabs { display:flex; gap:6px; }
  .tab-btn { border:1px solid var(--border); background:#fff; padding:6px 12px; border-radius:8px; cursor:pointer; font-size:13px; font-weight:600; }
  .tab-btn.active { background:var(--primary); color:#fff; border-color:var(--primary); }
  .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; font-size:13px; }
  .controls input[type="text"] { width:95px; padding:5px 8px; border:1px solid var(--border); border-radius:6px; font-size:13px; }
  .btn-sm { border:1px solid var(--border); background:#fff; color:var(--text); padding:5px 10px; border-radius:6px; font-size:12px; cursor:pointer; font-weight:600; }
  .btn-export { background:#0f766e; color:#fff; border-color:#0f766e; }
  .container { max-width:1360px; margin:14px auto; padding:0 16px; }
  .rubric { background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:10px 14px; margin-bottom:12px; font-size:13px; }
  .rubric summary { cursor:pointer; font-weight:700; color:#92400e; }
  .rubric ul { margin:6px 0 0 18px; padding:0; }
  .nav-card { background:#fff; border:1px solid var(--border); border-radius:10px; padding:10px 14px; margin-bottom:12px; }
  .nav-top { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; font-size:13px; }
  .progress-wrap { flex:1; min-width:180px; height:8px; background:#e2e8f0; border-radius:99px; overflow:hidden; }
  .progress-bar { height:100%; background:var(--primary); transition:width .2s; }
  .grid-ids { display:flex; flex-wrap:wrap; gap:4px; }
  .id-pill { width:40px; height:24px; border:1px solid var(--border); border-radius:5px; background:#fff; font-size:11px; font-weight:600; cursor:pointer; color:var(--muted); }
  .id-pill.current { outline:2px solid #0f172a; outline-offset:1px; color:#0f172a; }
  .id-pill.v-Faithful, .id-pill.v-No { background:#dcfce7; border-color:#86efac; color:#166534; }
  .id-pill.v-Minor { background:#fef3c7; border-color:#fcd34d; color:#b45309; }
  .id-pill.v-Unfaithful, .id-pill.v-Yes { background:#fee2e2; border-color:#fca5a5; color:#b91c1c; }
  .split-2 { display:grid; grid-template-columns:1fr 1fr; gap:12px; align-items:start; }
  @media (max-width:960px) { .split-2 { grid-template-columns:1fr; } }
  .panel { background:#fff; border:1px solid var(--border); border-radius:10px; padding:14px 16px; margin-bottom:12px; }
  .panel-title { font-size:13px; font-weight:700; color:var(--muted); margin-bottom:8px; display:flex; justify-content:space-between; align-items:center; }
  .shard-item { background:#f8fafc; border-left:3px solid var(--primary); padding:6px 10px; margin-bottom:6px; border-radius:4px; font-size:14px; }
  .ans-box { background:#ecfdf5; border:1px solid #a7f3d0; padding:8px 12px; border-radius:6px; margin:8px 0; font-size:14px; }
  .sym-block { background:#f8fafc; border:1px solid #e2e8f0; border-radius:6px; padding:8px 10px; margin-bottom:8px; font-family:ui-monospace,SFMono-Regular,Consolas,monospace; font-size:12.5px; white-space:pre-wrap; word-break:break-word; }
  .eq-note { background:#f0f9ff; border-left:3px solid #0284c7; padding:8px 10px; border-radius:4px; font-size:13px; margin-bottom:8px; }
  .clauses-box { background:#f8fafc; border-left:4px solid var(--primary); padding:10px 14px; border-radius:6px; font-size:14px; white-space:pre-wrap; }
  .reply-box { font-size:14.5px; overflow-x:auto; word-break:break-word; }
  .reply-box p { margin:0 0 8px 0; }
  .reply-box ul { margin:4px 0 8px 22px; padding:0; }
  .reply-box pre { background:#f1f5f9; padding:10px; border-radius:6px; font-size:12.5px; white-space:pre-wrap; }
  .action-bar { position:fixed; left:0; right:0; bottom:0; z-index:40; background:rgba(255,255,255,.97); border-top:1px solid var(--border); box-shadow:0 -4px 16px rgba(15,23,42,.08); padding:10px 16px; }
  .action-inner { max-width:1360px; margin:0 auto; display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; }
  .choice-group { display:flex; flex-wrap:wrap; gap:8px; }
  .choice-btn { border:2px solid var(--border); background:#fff; padding:8px 13px; border-radius:8px; font-size:13.5px; font-weight:700; cursor:pointer; }
  .choice-btn.selected { background:#0f172a; border-color:#0f172a; color:#fff; }
  .note-input { flex:1; min-width:180px; padding:7px 10px; border:1px solid var(--border); border-radius:8px; font-size:13px; }
  .toast { position:fixed; top:66px; right:20px; background:#0f172a; color:#fff; padding:8px 14px; border-radius:8px; font-size:13px; opacity:0; pointer-events:none; transition:opacity .2s; z-index:60; }
  .toast.show { opacity:.95; }
</style>
</head>
<body>
<header>
  <div class="topbar">
    <div class="brand">终稿补充人工核查工具 <span class="badge">Seed 2026 · 30 证书 + 20 阴性对照</span></div>
    <div class="tabs">
      <button class="tab-btn active" id="tab-certs" onclick="switchTab('certs')">① 证书形式化保真 (<span id="cnt-certs">0/30</span>)</button>
      <button class="tab-btn" id="tab-neg_repair" onclick="switchTab('neg_repair')">② Repair 阴性对照 (<span id="cnt-neg_repair">0/20</span>)</button>
    </div>
    <div class="controls">
      <label>核查者: <input type="text" id="annotator" value="cai" onchange="onAnnotatorChange()" /></label>
      <label style="cursor:pointer;"><input type="checkbox" id="autoNext" checked /> 标完自动下一题</label>
      <button class="btn-sm" id="rawBtn" onclick="toggleRaw()">显示原文</button>
      <button class="btn-sm btn-export" onclick="exportAll()">导出两份 CSV</button>
      <button class="btn-sm" onclick="exportJSON()">备份JSON</button>
      <button class="btn-sm" onclick="document.getElementById('imp').click()">导入JSON</button>
      <input type="file" id="imp" accept=".json" style="display:none" onchange="importJSON(event)" />
    </div>
  </div>
</header>

<div class="container">
  <details class="rubric" open>
    <summary id="rubricTitle">核查标准速查（点击可折叠）</summary>
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
    <div class="choice-group" id="choiceGroup"></div>
    <input type="text" class="note-input" id="noteInput" placeholder="简短备注（若标 Minor/Unfaithful 或 Yes 建议写明哪条分片/原因）" oninput="saveNote()" />
    <div style="display:flex;gap:6px;">
      <button class="btn-sm" onclick="step(-1)">← 上一题</button>
      <button class="btn-sm" onclick="step(1)">下一题 →</button>
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const D = __PAYLOAD_JSON__;
let tab = 'certs';
let idx = { certs: 0, neg_repair: 0 };
let showRaw = false;
let state = { certs: {}, neg_repair: {}, notes: {} };

function skey() {
  const who = (document.getElementById('annotator').value || 'default').trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_');
  return 'cscwd_cert_negrep_seed2026_' + (who || 'default');
}
function loadState() {
  try { const r = localStorage.getItem(skey()); state = r ? JSON.parse(r) : { certs:{}, neg_repair:{}, notes:{} }; }
  catch(e) { state = { certs:{}, neg_repair:{}, notes:{} }; }
  state.certs = state.certs || {}; state.neg_repair = state.neg_repair || {}; state.notes = state.notes || {};
}
function saveState() { localStorage.setItem(skey(), JSON.stringify(state)); }
function onAnnotatorChange() {
  localStorage.setItem('cscwd_cert_negrep_last', document.getElementById('annotator').value.trim());
  loadState(); render(); toast('已切换核查者：' + document.getElementById('annotator').value.trim());
}
function toast(m) { const t=document.getElementById('toast'); t.textContent=m; t.classList.add('show'); clearTimeout(window._tt); window._tt=setTimeout(()=>t.classList.remove('show'),1800); }
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
  ['certs','neg_repair'].forEach(k => document.getElementById('tab-'+k).classList.toggle('active', k===t));
  render();
}
function toggleRaw() { showRaw=!showRaw; document.getElementById('rawBtn').textContent=showRaw?'渲染公式':'显示原文'; render(); }

function render() {
  const dc = Object.keys(state.certs).length, dn = Object.keys(state.neg_repair).length;
  document.getElementById('cnt-certs').textContent = `${dc}/${D.certs.length}`;
  document.getElementById('cnt-neg_repair').textContent = `${dn}/${D.neg_repair.length}`;

  if (tab === 'certs') {
    document.getElementById('rubricTitle').textContent = '① SymPy 证书形式化保真度核查标准（重点检查：是否漏译原题隐含条件导致某前缀其实已经充分）';
    document.getElementById('rubricBody').innerHTML = `<ul>
      <li><strong style="color:#166534">1. Faithful（完全忠实）</strong>：右侧 SymPy 变量域、各分片约束和官方目标与左侧中文原题一致，每个 $S_j$ 的留一反例在中文原题下都合法。</li>
      <li><strong style="color:#b45309">2. Minor（细微技术简化，但不充分性成立）</strong>：形式化有轻量技术简化（如函数用有限点探测、反例用了退化圆锥曲线或未显式写某辅助非退化条件），但在中文原题下<strong>每一条 $S_j$ 仍严格必要</strong>（可轻易换成原题合法反例）。</li>
      <li><strong style="color:#b91c1c">3. Unfaithful（实质漏译/原题下不成立）</strong>：实质漏译关键隐含条件导致某条分片在原题下其实冗余（删掉它答案依然唯一/最优不变），或形式化目标与官方答案明显不符。</li>
    </ul>`;
  } else {
    document.getElementById('rubricTitle').textContent = '② Repair 阴性对照核查标准（检验 463 条双阴性轨迹中是否存在漏检的显式撤回；中性呈现、不含模型/协议）';
    document.getElementById('rubricBody').innerHTML = `<ul>
      <li><strong style="color:#b91c1c">1 / Y. Yes（存在显式撤回/修正）</strong>：末轮回复<strong>明确提及并撤回/纠正</strong>了早先轮次给出的错误定论或过度自信断言。</li>
      <li><strong style="color:#166534">2 / N. No（未显式撤回）</strong>：末轮只是基于完整条件直接给出最终推导和答案，没有明确提及或撤回早先结论。</li>
    </ul>`;
  }

  const list = D[tab];
  const map = state[tab];
  const done = list.filter(x => map[x.id]).length;
  document.getElementById('progText').textContent = `进度：${done} / ${list.length}（第 ${idx[tab]+1} 题）`;
  document.getElementById('progBar').style.width = `${Math.round(done/list.length*100)}%`;

  const grid = document.getElementById('idGrid');
  grid.innerHTML = '';
  list.forEach((it, i) => {
    const b = document.createElement('button');
    const v = map[it.id] || '';
    b.className = 'id-pill' + (v ? ' v-'+v : '') + (i===idx[tab] ? ' current' : '');
    b.textContent = it.id;
    b.title = v ? `${it.id}: ${v}` : it.id;
    b.onclick = () => { idx[tab]=i; render(); };
    grid.appendChild(b);
  });

  const item = list[idx[tab]];
  const cur = map[item.id] || '';
  const mc = document.getElementById('mainContent');

  if (tab === 'certs') {
    const c = item.certificate;
    const shardsHtml = item.shards.map(s => `<div class="shard-item"><strong>[${esc(s.id)}]</strong> ${renderRich(s.text)}</div>`).join('');
    const scText = Object.entries(c.shard_constraints || {}).map(([k, arr]) => `${k}: ${JSON.stringify(arr)}`).join('\n');
    const fnText = Object.keys(c.functions || {}).length ? JSON.stringify(c.functions, null, 2) : '(无自定义函数)';
    const looText = JSON.stringify({ leave_one_out: c.leave_one_out, verified_values: item.report.leave_one_out_verified }, null, 2);
    mc.innerHTML = `
      <div class="split-2">
        <div class="panel">
          <div class="panel-title"><span>${item.id} · ${item.item_id} · ${item.mech} · ${item.subfield} (k=${item.k})</span><span>当前：${cur||'未标'}</span></div>
          <div style="margin-bottom:8px;"><strong>原始中文题干：</strong><div class="reply-box" style="margin-top:4px;">${renderRich(item.question)}</div></div>
          <div class="ans-box"><strong>OlympiadBench 官方答案：</strong> ${renderRich(item.final_answer)}</div>
          <div style="margin-top:10px;"><strong>分片（S1–S${item.k}）：</strong><div style="margin-top:6px;">${shardsHtml}</div></div>
        </div>
        <div class="panel">
          <div class="panel-title"><span>SymPy 形式化证书（SymPy 离线校验已 PASS）</span><span>target: <code>${esc(c.target_expr)}</code></span></div>
          <div class="eq-note"><strong>形式化说明：</strong>${esc(c.equivalence_note || '')}</div>
          <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">变量与函数定义</div>
          <div class="sym-block">variables: ${esc(JSON.stringify(c.variables))}\nfunctions: ${esc(fnText)}\nfull_answer_check: ${esc(JSON.stringify(c.full_answer_check))}\nfull_answer_verified: ${esc(String(item.report.full_answer_verified))}</div>
          <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">各分片约束 (shard_constraints)</div>
          <div class="sym-block">${esc(scText)}</div>
          <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">留一法见证 (LOO witnesses & verified outputs)</div>
          <div class="sym-block" style="max-height:280px;overflow:auto;">${esc(looText)}</div>
        </div>
      </div>`;
    const opts = [['Faithful','1: Faithful（完全忠实）'],['Minor','2: Minor（细微简化，必要性仍成立）'],['Unfaithful','3: Unfaithful（实质漏译/不成立）']];
    document.getElementById('choiceGroup').innerHTML = opts.map(([v,lbl])=>
      `<button class="choice-btn ${cur===v?'selected':''}" onclick="pick('${v}')">${lbl}</button>`
    ).join('');
  } else {
    mc.innerHTML = `
      <div class="panel"><div class="panel-title"><span>对照样本 ${item.id} · 完整题目条件</span><span>当前：${cur||'未标'}</span></div><div class="clauses-box">${renderRich(item.clauses)}</div></div>
      <div class="split-2">
        <div class="panel" style="border-top:3px solid #f59e0b"><div class="panel-title"><span>早先第 ${item.early_turn} 轮模型可见回复</span></div><div class="reply-box">${renderRich(item.early_reply)}</div></div>
        <div class="panel" style="border-top:3px solid #2563eb"><div class="panel-title"><span>末轮模型可见回复（判断是否明确提及并撤回/修正左侧结论）</span></div><div class="reply-box">${renderRich(item.final_reply)}</div></div>
      </div>`;
    const opts = [['Yes','1 / Y: Yes（显式撤回/修正）'],['No','2 / N: No（未显式撤回）']];
    document.getElementById('choiceGroup').innerHTML = opts.map(([v,lbl])=>
      `<button class="choice-btn ${cur===v?'selected':''}" onclick="pick('${v}')">${lbl}</button>`
    ).join('');
  }

  document.getElementById('noteInput').value = state.notes[item.id] || '';
  typeset(mc);
  window.scrollTo({top:0, behavior:'instant'});
}

function pick(v) {
  const list = D[tab];
  const item = list[idx[tab]];
  state[tab][item.id] = v;
  saveState();
  toast(`${item.id} -> ${v}`);
  if (document.getElementById('autoNext').checked && idx[tab] < list.length - 1) idx[tab]++;
  render();
}
function saveNote() {
  const item = D[tab][idx[tab]];
  const v = document.getElementById('noteInput').value;
  if (v.trim()) state.notes[item.id] = v; else delete state.notes[item.id];
  saveState();
}
function step(d) { const n = idx[tab]+d; if (n>=0 && n<D[tab].length){ idx[tab]=n; render(); } }
function jumpNext() {
  const list = D[tab], map = state[tab];
  for (let off=1; off<=list.length; off++) {
    const i = (idx[tab]+off) % list.length;
    if (!map[list[i].id]) { idx[tab]=i; render(); return; }
  }
  toast('本页已全部标完！');
}
function csvEsc(v) { const s=String(v==null?'':v); return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s; }
function dl(name, content, mime) {
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([content],{type:mime}));
  a.download=name; document.body.appendChild(a); a.click(); setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove();},200);
}
function exportAll() {
  const who = (document.getElementById('annotator').value || 'annotator').trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_') || 'annotator';
  const cLines = ['audit_id,item_id,verified_mechanism,macro_subfield,human_verdict,human_notes'];
  let cDone = 0;
  for (const r of D.certs) {
    const v = state.certs[r.id] || ''; if (v) cDone++;
    cLines.push([r.id, r.item_id, r.mech, r.subfield, v, state.notes[r.id]||''].map(csvEsc).join(','));
  }
  const nLines = ['audit_id,early_turn,human_retracted,human_notes'];
  let nDone = 0;
  for (const r of D.neg_repair) {
    const v = state.neg_repair[r.id] || ''; if (v) nDone++;
    nLines.push([r.id, r.early_turn, v, state.notes[r.id]||''].map(csvEsc).join(','));
  }
  dl(`cert_audit_30_${who}.csv`, '\uFEFF' + cLines.join('\r\n') + '\r\n', 'text/csv;charset=utf-8');
  setTimeout(()=> dl(`neg_repair_20_${who}.csv`, '\uFEFF' + nLines.join('\r\n') + '\r\n', 'text/csv;charset=utf-8'), 250);
  toast(`已导出 cert_audit_30_${who}.csv (${cDone}/30) 与 neg_repair_20_${who}.csv (${nDone}/20)`);
}
function exportJSON() {
  const who = (document.getElementById('annotator').value || 'annotator').trim() || 'annotator';
  dl(`cert_negrep_backup_${who}.json`, JSON.stringify({ annotator: who, exported_at: new Date().toISOString(), state }, null, 2), 'application/json');
}
function importJSON(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try { const o = JSON.parse(rd.result); const st = o.state || o; state = { certs:st.certs||{}, neg_repair:st.neg_repair||{}, notes:st.notes||{} }; saveState(); render(); toast('导入成功'); }
    catch(e) { alert('JSON 无效：' + e.message); }
  };
  rd.readAsText(f, 'utf-8'); ev.target.value = '';
}
document.addEventListener('keydown', e => {
  if (e.target && e.target.tagName==='INPUT') return;
  if (e.key==='ArrowRight') step(1);
  else if (e.key==='ArrowLeft') step(-1);
  else if (e.key.toLowerCase()==='u') jumpNext();
  else if (tab==='certs') { if(e.key==='1') pick('Faithful'); else if(e.key==='2') pick('Minor'); else if(e.key==='3') pick('Unfaithful'); }
  else { if(e.key==='1'||e.key.toLowerCase()==='y') pick('Yes'); else if(e.key==='2'||e.key.toLowerCase()==='n') pick('No'); }
});
window.addEventListener('DOMContentLoaded', () => {
  const last = localStorage.getItem('cscwd_cert_negrep_last');
  if (last) document.getElementById('annotator').value = last;
  loadState(); render();
});
window.addEventListener('load', () => typeset(document.getElementById('mainContent')));
</script>
</body>
</html>
'''

    out_html = OUT_DIR / "cert_and_neg_repair_audit_seed2026.html"
    out_html.write_text(html.replace("__PAYLOAD_JSON__", payload_json), encoding="utf-8")
    print(f"Generated {out_html} ({out_html.stat().st_size / 1024:.1f} KB)")
    print(f"Cert audit: 30 items (20 TypeA + 10 TypeB), unique={len({r['item_id'] for r in cert_records})}")
    print(f"Neg repair: 20 controls from {total_dual_neg} dual-neg E_U=1 trajectories, unique items={key_out['neg_repair_unique_items']}")


if __name__ == "__main__":
    main()
