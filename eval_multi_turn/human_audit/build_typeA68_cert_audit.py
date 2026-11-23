#!/usr/bin/env python3
"""Build reproducible SEED=2026 full-census audit package for ALL 68 TypeA TEST items.

Generates (in this directory):
  - typeA68_audit_template.csv          (blank template for independent raters)
  - _typeA68_key.json                   (metadata & exact evaluated non-final prefixes per item)
  - typeA68_cert_audit_seed2026.html    (single-file offline KaTeX audit UI; 0 prefilled answers)

Features of the HTML UI:
  - Displays all 68 TypeA_UniqueValue TEST items in deterministic SEED=2026 shuffled order (A01-A68).
  - Left pane: original Chinese problem, OlympiadBench gold answer, exact S1..Sk shards,
    and all 3 evaluated disclosure orders (Rep 0 canonical + Rep 1-2 permutations) with
    explicit non-final prefixes so raters can check both translation faithfulness AND
    whether any actual evaluated prefix becomes sufficient under natural-language semantics.
  - Right pane: SymPy variables, functions, shard_constraints, full_answer_check, and
    leave-one-out WitnessPairs with verified target values.
  - Supports importing prior 30-item CSV (cert_audit_30_cai.csv) by item_id for Cai so
    the 20 previously-audited TypeA items need not be re-entered, while remaining 100% blank
    and un-biased for Tang (or any independent rater).
  - Guards against accidentally exporting an empty CSV.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FROZEN_TEST = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_test_100.jsonl"
FROZEN_CERTS = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_certificates.jsonl"
FROZEN_REPORTS = ROOT / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_reports.jsonl"
DISC_PLAN = ROOT / "eval_multi_turn" / "disclosure_plan_test100.jsonl"
OUT_DIR = Path(__file__).resolve().parent

SEED = 2026


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def main() -> None:
    rng = random.Random(SEED)

    items = [r for r in read_jsonl(FROZEN_TEST) if r["verified_mechanism"] == "TypeA_UniqueValue"]
    assert len(items) == 68, f"Expected 68 TypeA TEST items, got {len(items)}"

    certs_by_id = {r["item_id"]: r for r in read_jsonl(FROZEN_CERTS)}
    reps_by_id = {r["item_id"]: r for r in read_jsonl(FROZEN_REPORTS)}

    orders_by_id: dict[str, list[dict]] = defaultdict(list)
    for row in read_jsonl(DISC_PLAN):
        orders_by_id[row["item_id"]].append({
            "rep": int(row["rep"]),
            "order_kind": row.get("order_kind", ""),
            "order": list(row["disclosure_order"]),
        })
    for iid in orders_by_id:
        orders_by_id[iid].sort(key=lambda x: x["rep"])

    # Deterministic shuffled order across all 68 TypeA items
    items.sort(key=lambda r: r["item_id"])
    rng.shuffle(items)

    records = []
    key_map = {}
    total_eval_prefixes = 0
    total_unique_prefixes = 0

    for i, it in enumerate(items, 1):
        aid = f"A{i:02d}"
        iid = it["item_id"]
        cert = certs_by_id[iid]
        rep = reps_by_id[iid]
        ords = orders_by_id[iid]
        assert len(ords) == 3, f"Expected 3 disclosure orders for {iid}, got {len(ords)}"

        shards = [
            {"id": s["shard_id"], "text": (s.get("shard") or s.get("text") or "").strip()}
            for s in it.get("shards", [])
        ]

        # Compute exact evaluated non-final prefixes across the 3 reps
        eval_orders_view = []
        uniq_prefs = set()
        n_eval_pref_item = 0
        for o in ords:
            seq = o["order"]
            prefs = []
            for t in range(1, len(seq)):
                p_list = seq[:t]
                missing = [x for x in seq if x not in p_list]
                prefs.append({
                    "turn": t,
                    "prefix": p_list,
                    "missing": missing,
                })
                uniq_prefs.add(tuple(sorted(p_list)))
                n_eval_pref_item += 1
            eval_orders_view.append({
                "rep": o["rep"],
                "order": seq,
                "non_final_prefixes": prefs,
            })

        total_eval_prefixes += n_eval_pref_item
        total_unique_prefixes += len(uniq_prefs)

        records.append({
            "id": aid,
            "item_id": iid,
            "subfield": it["macro_subfield"],
            "k": int(it["canonical_k"]),
            "question": it.get("question", ""),
            "final_answer": it.get("final_answer", ""),
            "shards": shards,
            "eval_orders": eval_orders_view,
            "n_eval_prefixes": n_eval_pref_item,
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
                "status": rep.get("status"),
                "full_answer_verified": rep.get("full_answer_verified"),
                "leave_one_out_verified": rep.get("leave_one_out_verified"),
            },
        })

        key_map[aid] = {
            "item_id": iid,
            "macro_subfield": it["macro_subfield"],
            "canonical_k": int(it["canonical_k"]),
            "n_eval_prefixes": n_eval_pref_item,
            "n_unique_eval_prefixes": len(uniq_prefs),
            "disclosure_orders": [o["order"] for o in ords],
        }

    # Write blank CSV template
    with (OUT_DIR / "typeA68_audit_template.csv").open("w", encoding="utf-8", newline="\n") as f:
        w = csv.writer(f)
        w.writerow(["audit_id", "item_id", "macro_subfield", "canonical_k", "human_verdict", "prefix_flip", "affected_prefixes", "human_notes"])
        for r in records:
            w.writerow([r["id"], r["item_id"], r["subfield"], r["k"], "", "", "", ""])

    key_out = {
        "seed": SEED,
        "n_typeA_items": len(records),
        "total_evaluated_nonfinal_prefixes": total_eval_prefixes,
        "total_unique_evaluated_prefixes": total_unique_prefixes,
        "items": key_map,
    }
    (OUT_DIR / "_typeA68_key.json").write_text(json.dumps(key_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8", newline="\n")

    payload_json = json.dumps({"items": records}, ensure_ascii=False).replace("</", "<\\/")

    html = r'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8" />
<meta name="viewport" content="width=device-width, initial-scale=1.0" />
<title>TypeA 全量 68 题 SymPy 证书独立核验工具 — Seed 2026</title>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.css" />
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/katex.min.js"></script>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.11/dist/contrib/auto-render.min.js"></script>
<style>
  :root { --bg:#f5f7fb; --card:#fff; --border:#dfe4ee; --text:#1c2434; --muted:#64748b; --primary:#2563eb; }
  * { box-sizing:border-box; }
  body { margin:0; font-family:-apple-system,BlinkMacSystemFont,"Segoe UI","PingFang SC","Microsoft YaHei",sans-serif; background:var(--bg); color:var(--text); line-height:1.6; padding-bottom:135px; }
  header { position:sticky; top:0; z-index:30; background:#fff; border-bottom:1px solid var(--border); box-shadow:0 2px 8px rgba(15,23,42,.04); }
  .topbar { max-width:1400px; margin:0 auto; padding:10px 16px; display:flex; flex-wrap:wrap; gap:10px; align-items:center; justify-content:space-between; }
  .brand { font-weight:700; font-size:15px; display:flex; align-items:center; gap:8px; }
  .badge { background:#eef2ff; color:#3730a3; font-size:12px; padding:2px 8px; border-radius:999px; font-weight:600; }
  .controls { display:flex; flex-wrap:wrap; gap:8px; align-items:center; font-size:13px; }
  .controls input[type="text"] { width:92px; padding:5px 8px; border:1px solid var(--border); border-radius:6px; font-size:13px; }
  .btn-sm { border:1px solid var(--border); background:#fff; color:var(--text); padding:5px 10px; border-radius:6px; font-size:12px; cursor:pointer; font-weight:600; }
  .btn-export { background:#0f766e; color:#fff; border-color:#0f766e; }
  .btn-imp { background:#eff6ff; color:#1d4ed8; border-color:#bfdbfe; }
  .container { max-width:1400px; margin:14px auto; padding:0 16px; }
  .rubric { background:#fffbeb; border:1px solid #fde68a; border-radius:10px; padding:10px 14px; margin-bottom:12px; font-size:13px; }
  .rubric summary { cursor:pointer; font-weight:700; color:#92400e; }
  .rubric ul { margin:6px 0 0 18px; padding:0; }
  .nav-card { background:#fff; border:1px solid var(--border); border-radius:10px; padding:10px 14px; margin-bottom:12px; }
  .nav-top { display:flex; flex-wrap:wrap; justify-content:space-between; align-items:center; gap:8px; margin-bottom:8px; font-size:13px; }
  .progress-wrap { flex:1; min-width:180px; height:8px; background:#e2e8f0; border-radius:99px; overflow:hidden; }
  .progress-bar { height:100%; background:var(--primary); transition:width .2s; }
  .grid-ids { display:flex; flex-wrap:wrap; gap:4px; }
  .id-pill { width:38px; height:23px; border:1px solid var(--border); border-radius:5px; background:#fff; font-size:11px; font-weight:600; cursor:pointer; color:var(--muted); }
  .id-pill.current { outline:2px solid #0f172a; outline-offset:1px; color:#0f172a; }
  .id-pill.v-Faithful { background:#dcfce7; border-color:#86efac; color:#166534; }
  .id-pill.v-Minor { background:#fef3c7; border-color:#fcd34d; color:#b45309; }
  .id-pill.v-Unfaithful { background:#fee2e2; border-color:#fca5a5; color:#b91c1c; }
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
  .reply-box { font-size:14.5px; overflow-x:auto; word-break:break-word; }
  .reply-box p { margin:0 0 8px 0; }
  .reply-box pre { background:#f1f5f9; padding:10px; border-radius:6px; font-size:12.5px; white-space:pre-wrap; }
  .action-bar { position:fixed; left:0; right:0; bottom:0; z-index:40; background:rgba(255,255,255,.97); border-top:1px solid var(--border); box-shadow:0 -4px 16px rgba(15,23,42,.08); padding:10px 16px; }
  .action-inner { max-width:1400px; margin:0 auto; display:flex; flex-direction:column; gap:8px; }
  .action-row { display:flex; flex-wrap:wrap; gap:8px; align-items:center; justify-content:space-between; }
  .choice-group { display:flex; flex-wrap:wrap; gap:8px; align-items:center; }
  .choice-btn { border:2px solid var(--border); background:#fff; padding:7px 12px; border-radius:8px; font-size:13px; font-weight:700; cursor:pointer; }
  .choice-btn.selected { background:#0f172a; border-color:#0f172a; color:#fff; }
  .flip-btn { border:1.5px solid var(--border); background:#fff; padding:5px 10px; border-radius:7px; font-size:12.5px; font-weight:700; cursor:pointer; }
  .flip-btn.selected-no { background:#dcfce7; border-color:#16a34a; color:#166534; }
  .flip-btn.selected-yes { background:#fee2e2; border-color:#dc2626; color:#991b1b; }
  .note-input { flex:1; min-width:200px; padding:6px 10px; border:1px solid var(--border); border-radius:7px; font-size:13px; }
  .aff-input { width:230px; padding:6px 10px; border:1px solid var(--border); border-radius:7px; font-size:12.5px; }
  .toast { position:fixed; top:66px; right:20px; background:#0f172a; color:#fff; padding:8px 14px; border-radius:8px; font-size:13px; opacity:0; pointer-events:none; transition:opacity .2s; z-index:60; }
  .toast.show { opacity:.95; }
</style>
</head>
<body>
<header>
  <div class="topbar">
    <div class="brand">TypeA 核心 68 题 SymPy 证书双人独立核验工具 <span class="badge">Seed 2026 · 全量 68/68</span></div>
    <div class="controls">
      <label>标注者: <input type="text" id="annotator" value="tang" onchange="onAnnotatorChange()" /></label>
      <label style="cursor:pointer;"><input type="checkbox" id="autoNext" checked /> 标完自动下一题</label>
      <button class="btn-sm" id="rawBtn" onclick="toggleRaw()">显示原文</button>
      <button class="btn-sm btn-imp" onclick="document.getElementById('impCsv').click()" title="cai 可导入已完成的 cert_audit_30_cai.csv 自动填入已标 20 题；tang 请保持全空白独立标注">导入已有CSV</button>
      <input type="file" id="impCsv" accept=".csv" style="display:none" onchange="importCSV(event)" />
      <button class="btn-sm btn-export" onclick="exportCSV()">导出 CSV</button>
      <button class="btn-sm" onclick="exportJSON()">备份JSON</button>
      <button class="btn-sm" onclick="document.getElementById('impJson').click()">导入JSON</button>
      <input type="file" id="impJson" accept=".json" style="display:none" onchange="importJSON(event)" />
    </div>
  </div>
</header>

<div class="container">
  <details class="rubric" open>
    <summary>核验量规与双维度判定说明（点击可折叠）</summary>
    <ul>
      <li><strong style="color:#166534">1. Faithful（严格忠实）</strong>：右侧 SymPy 变量域、各分片约束和官方目标与左侧中文原题完全一致，每个 $S_j$ 的留一见证对在中文原题下都合法且目标值不同（实际评测前缀均不充分）。</li>
      <li><strong style="color:#b45309">2. Minor（细微技术简化，必要性仍成立）</strong>：形式化省略了次要非零/非退化辅助约束或有轻微技术简化，但在中文原题下<strong>每条 $S_j$ 仍严格必要</strong>（可轻易替换为原题合法反例，实际评测前缀仍均不充分）。</li>
      <li><strong style="color:#b91c1c">3. Unfaithful（实质偏差/翻译或切分导致原题下不成立）</strong>：存在实质漏译隐含条件、分片切分导致某分片冗余或留一见证违反保留中文分片。若选此项，请对照左下角<strong>“实际评测的 3 条披露序”</strong>进一步选择：<strong>是否有任一实际评测非最终前缀在中文原题下其实已唯一确定答案（Prefix Flip: Yes / No）</strong>，并在框内注明受影响前缀（如 <code>Rep2-T4</code> 或 <code>{S1,S3,S4}</code>）。</li>
      <li><strong>独立标注提醒</strong>：两位作者请分别填写自己的姓名（<code>cai</code> / <code>tang</code>）并严格独立完成，不要互相参考；<code>cai</code> 可点右上角“导入已有CSV”导入此前已核对的 <code>cert_audit_30_cai.csv</code>（自动匹配其中 20 道 TypeA），再完成余下 48 题即可；<code>tang</code> 请从空白独立标完全部 68 题。</li>
    </ul>
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
      <div class="choice-group" id="flipGroup"></div>
      <div style="display:flex;gap:6px;">
        <button class="btn-sm" onclick="step(-1)">← 上一题</button>
        <button class="btn-sm" onclick="step(1)">下一题 →</button>
      </div>
    </div>
    <div class="action-row">
      <input type="text" class="aff-input" id="affInput" placeholder="若 Prefix Flip=Yes，写明受影响前缀 (如 Rep0-T3)" oninput="saveExtra()" />
      <input type="text" class="note-input" id="noteInput" placeholder="简短备注（若标 Minor/Unfaithful 建议写明分片号与原因）" oninput="saveExtra()" />
    </div>
  </div>
</div>
<div class="toast" id="toast"></div>

<script>
const D = __PAYLOAD_JSON__;
let idx = 0;
let showRaw = false;
// state: { verdicts: {A01:'Faithful'}, flips: {A01:'No'}, affected: {A01:''}, notes: {A01:''} }
let state = { verdicts:{}, flips:{}, affected:{}, notes:{} };

function whoSlug() {
  return (document.getElementById('annotator').value || 'annotator').trim().toLowerCase().replace(/[^a-z0-9_-]/g, '_') || 'annotator';
}
function skey() { return 'cscwd_typeA68_seed2026_' + whoSlug(); }
function loadState() {
  try { const r = localStorage.getItem(skey()); state = r ? JSON.parse(r) : { verdicts:{}, flips:{}, affected:{}, notes:{} }; }
  catch(e) { state = { verdicts:{}, flips:{}, affected:{}, notes:{} }; }
  state.verdicts = state.verdicts || {};
  state.flips = state.flips || {};
  state.affected = state.affected || {};
  state.notes = state.notes || {};
}
function saveState() { localStorage.setItem(skey(), JSON.stringify(state)); }
function onAnnotatorChange() {
  localStorage.setItem('cscwd_typeA68_seed2026_last', document.getElementById('annotator').value.trim());
  loadState(); render(); toast('已切换标注者：' + document.getElementById('annotator').value.trim() + `（已标 ${Object.keys(state.verdicts).length}/68）`);
}
function toast(m) { const t=document.getElementById('toast'); t.textContent=m; t.classList.add('show'); clearTimeout(window._tt); window._tt=setTimeout(()=>t.classList.remove('show'),2000); }
function esc(s) { return String(s||'').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;'); }
function renderRich(text) {
  if (showRaw) return '<pre>' + esc(text) + '</pre>';
  const ph = [];
  let s = String(text||'').replace(/\r\n/g,'\n');
  s = s.replace(/(\$\$[\s\S]*?\$\$|\\\[[\s\S]*?\\\]|\\\([\s\S]*?\\\)|\$(?!\s)([^\n$]+?)(?<!\s)\$)/g, m => { ph.push(m); return '@@M'+(ph.length-1)+'@@'; });
  s = esc(s).replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
  let html='';
  for (let l of s.split('\n')) {
    const tr = l.trim();
    html += tr ? '<p>'+l+'</p>' : '<div style="height:5px"></div>';
  }
  return html.replace(/@@M(\d+)@@/g, (_,i)=>esc(ph[+i]));
}
function typeset(el) {
  if (!showRaw && typeof renderMathInElement==='function') {
    try { renderMathInElement(el, { delimiters:[{left:'$$',right:'$$',display:true},{left:'\\[',right:'\\]',display:true},{left:'$',right:'$',display:false},{left:'\\(',right:'\\)',display:false}], throwOnError:false, strict:false }); } catch(e){}
  }
}
function toggleRaw() { showRaw=!showRaw; document.getElementById('rawBtn').textContent=showRaw?'渲染公式':'显示原文'; render(); }

function render() {
  const list = D.items;
  const done = list.filter(x => state.verdicts[x.id]).length;
  const cF = list.filter(x => state.verdicts[x.id]==='Faithful').length;
  const cM = list.filter(x => state.verdicts[x.id]==='Minor').length;
  const cU = list.filter(x => state.verdicts[x.id]==='Unfaithful').length;
  document.getElementById('progText').textContent = `进度：${done} / ${list.length}（第 ${idx+1} 题 · Faithful ${cF} / Minor ${cM} / Unfaithful ${cU}）`;
  document.getElementById('progBar').style.width = `${Math.round(done/list.length*100)}%`;

  const grid = document.getElementById('idGrid');
  grid.innerHTML = '';
  list.forEach((it, i) => {
    const b = document.createElement('button');
    const v = state.verdicts[it.id] || '';
    b.className = 'id-pill' + (v ? ' v-'+v : '') + (i===idx ? ' current' : '');
    b.textContent = it.id;
    b.title = v ? `${it.id} (${it.item_id}): ${v}` : `${it.id} (${it.item_id})`;
    b.onclick = () => { idx=i; render(); };
    grid.appendChild(b);
  });

  const item = list[idx];
  const cur = state.verdicts[item.id] || '';
  const curFlip = state.flips[item.id] || (cur ? 'No' : '');
  const c = item.certificate;
  const shardsHtml = item.shards.map(s => `<div class="shard-item"><strong>[${esc(s.id)}]</strong> ${renderRich(s.text)}</div>`).join('');
  const ordersHtml = item.eval_orders.map(o => {
    const pStr = o.non_final_prefixes.map(p => `T${p.turn}: <code>{${p.prefix.join(',')}}</code> (缺 ${p.missing.join(',')})`).join(' &nbsp;|&nbsp; ');
    return `<div style="margin-bottom:4px;"><strong>Rep ${o.rep}</strong> [${o.order.join('→')}]：${pStr}</div>`;
  }).join('');
  const scText = Object.entries(c.shard_constraints || {}).map(([k, arr]) => `${k}: ${JSON.stringify(arr)}`).join('\n');
  const fnText = Object.keys(c.functions || {}).length ? JSON.stringify(c.functions, null, 2) : '(无自定义函数)';
  const looText = JSON.stringify({ leave_one_out: c.leave_one_out, verified_values: item.report.leave_one_out_verified }, null, 2);

  const mc = document.getElementById('mainContent');
  mc.innerHTML = `
    <div class="split-2">
      <div class="panel">
        <div class="panel-title"><span>${item.id} · ${item.item_id} · TypeA · ${item.subfield} (k=${item.k})</span><span>当前：${cur||'未标'}${curFlip ? ' · Flip:'+curFlip : ''}</span></div>
        <div style="margin-bottom:8px;"><strong>原始中文题干：</strong><div class="reply-box" style="margin-top:4px;">${renderRich(item.question)}</div></div>
        <div class="ans-box"><strong>OlympiadBench 官方答案：</strong> ${renderRich(item.final_answer)}</div>
        <div style="margin-top:10px;"><strong>分片（S1–S${item.k}）：</strong><div style="margin-top:6px;">${shardsHtml}</div></div>
        <div class="orders-box"><strong>实际评测的 3 条披露序（共 ${item.n_eval_prefixes} 个非最终前缀）：</strong><div style="margin-top:5px;">${ordersHtml}</div></div>
      </div>
      <div class="panel">
        <div class="panel-title"><span>SymPy 形式化证书（SymPy 离线校验已 PASS）</span><span>target: <code>${esc(c.target_expr)}</code></span></div>
        <div class="eq-note"><strong>形式化说明：</strong>${esc(c.equivalence_note || '')}</div>
        <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">变量与函数定义</div>
        <div class="sym-block">variables: ${esc(JSON.stringify(c.variables))}\nfunctions: ${esc(fnText)}\nfull_answer_check: ${esc(JSON.stringify(c.full_answer_check))}\nfull_answer_verified: ${esc(String(item.report.full_answer_verified))}</div>
        <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">各分片约束 (shard_constraints)</div>
        <div class="sym-block">${esc(scText)}</div>
        <div style="font-size:12.5px;font-weight:700;color:var(--muted);margin-bottom:3px;">留一法见证对 (LOO WitnessPairs & verified outputs)</div>
        <div class="sym-block" style="max-height:310px;overflow:auto;">${esc(looText)}</div>
      </div>
    </div>`;

  const opts = [['Faithful','1: Faithful（严格忠实）'],['Minor','2: Minor（细微简化，必要性成立）'],['Unfaithful','3: Unfaithful（实质偏差）']];
  document.getElementById('choiceGroup').innerHTML = opts.map(([v,lbl])=>
    `<button class="choice-btn ${cur===v?'selected':''}" onclick="pick('${v}')">${lbl}</button>`
  ).join('');

  document.getElementById('flipGroup').innerHTML = `<span style="font-size:12.5px;font-weight:700;color:var(--muted);">实际评测前缀受影响?</span>
    <button class="flip-btn ${curFlip==='No'?'selected-no':''}" onclick="pickFlip('No')">No (0条充分)</button>
    <button class="flip-btn ${curFlip==='Yes'?'selected-yes':''}" onclick="pickFlip('Yes')">Yes (有前缀已充分)</button>`;

  document.getElementById('affInput').value = state.affected[item.id] || '';
  document.getElementById('noteInput').value = state.notes[item.id] || '';
  typeset(mc);
  window.scrollTo({top:0, behavior:'instant'});
}

function pick(v) {
  const item = D.items[idx];
  state.verdicts[item.id] = v;
  if (v !== 'Unfaithful') state.flips[item.id] = 'No';
  else if (!state.flips[item.id]) state.flips[item.id] = 'No';
  saveState();
  toast(`${item.id} -> ${v}`);
  // Do not auto-advance on Unfaithful so rater can check Prefix Flip & write a note
  if (v !== 'Unfaithful' && document.getElementById('autoNext').checked && idx < D.items.length - 1) idx++;
  render();
}
function pickFlip(fv) {
  const item = D.items[idx];
  state.flips[item.id] = fv;
  saveState();
  toast(`${item.id} Prefix Flip -> ${fv}`);
  render();
}
function saveExtra() {
  const item = D.items[idx];
  const a = document.getElementById('affInput').value;
  const n = document.getElementById('noteInput').value;
  if (a.trim()) state.affected[item.id] = a; else delete state.affected[item.id];
  if (n.trim()) state.notes[item.id] = n; else delete state.notes[item.id];
  saveState();
}
function step(d) { const n = idx+d; if (n>=0 && n<D.items.length){ idx=n; render(); } }
function jumpNext() {
  for (let off=1; off<=D.items.length; off++) {
    const i = (idx+off) % D.items.length;
    if (!state.verdicts[D.items[i].id]) { idx=i; render(); return; }
  }
  toast('全部 68 题已标完！');
}
function csvEsc(v) { const s=String(v==null?'':v); return /[",\r\n]/.test(s)?'"'+s.replace(/"/g,'""')+'"':s; }
function dl(name, content, mime) {
  const a=document.createElement('a'); a.href=URL.createObjectURL(new Blob([content],{type:mime}));
  a.download=name; document.body.appendChild(a); a.click(); setTimeout(()=>{URL.revokeObjectURL(a.href);a.remove();},200);
}
function exportCSV() {
  const done = D.items.filter(r => state.verdicts[r.id]).length;
  if (done === 0) {
    alert(`当前标注者（${whoSlug()}）的已标数量为 0/68，请先完成标注或检查右上角标注者姓名是否正确！`);
    return;
  }
  const lines = ['audit_id,item_id,macro_subfield,canonical_k,human_verdict,prefix_flip,affected_prefixes,human_notes'];
  for (const r of D.items) {
    const v = state.verdicts[r.id] || '';
    const fl = v ? (state.flips[r.id] || 'No') : '';
    lines.push([r.id, r.item_id, r.subfield, r.k, v, fl, state.affected[r.id]||'', state.notes[r.id]||''].map(csvEsc).join(','));
  }
  const fname = `typeA68_audit_${whoSlug()}.csv`;
  dl(fname, '\uFEFF' + lines.join('\r\n') + '\r\n', 'text/csv;charset=utf-8');
  toast(`已导出 ${fname} (${done}/68)`);
}
function exportJSON() {
  dl(`typeA68_backup_${whoSlug()}.json`, JSON.stringify({ annotator: whoSlug(), exported_at: new Date().toISOString(), state }, null, 2), 'application/json');
}
function parseCSVLine(line) {
  const out=[]; let cur='', inQ=false;
  for (let i=0;i<line.length;i++) {
    const ch=line[i];
    if (inQ) { if (ch==='"' && line[i+1]==='"'){ cur+='"'; i++; } else if (ch==='"'){ inQ=false; } else cur+=ch; }
    else { if (ch==='"') inQ=true; else if (ch===','){ out.push(cur); cur=''; } else cur+=ch; }
  }
  out.push(cur); return out;
}
function importCSV(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    const lines = String(rd.result||'').replace(/^\uFEFF/,'').split(/\r?\n/).filter(l=>l.trim());
    if (lines.length < 2) { alert('CSV 内容为空'); return; }
    const hdr = parseCSVLine(lines[0]).map(x=>x.trim());
    const iItem = hdr.indexOf('item_id');
    const iVerd = hdr.indexOf('human_verdict');
    const iFlip = hdr.indexOf('prefix_flip');
    const iAff = hdr.indexOf('affected_prefixes');
    const iNote = hdr.indexOf('human_notes');
    if (iItem < 0 || iVerd < 0) { alert('CSV 缺少 item_id 或 human_verdict 列'); return; }
    const byItem = {}; D.items.forEach(r => byItem[r.item_id] = r.id);
    let n = 0;
    for (let k=1; k<lines.length; k++) {
      const cols = parseCSVLine(lines[k]);
      const aid = byItem[(cols[iItem]||'').trim()];
      const v = (cols[iVerd]||'').trim();
      if (aid && ['Faithful','Minor','Unfaithful'].includes(v)) {
        state.verdicts[aid] = v;
        state.flips[aid] = (iFlip>=0 && (cols[iFlip]||'').trim()) ? cols[iFlip].trim() : 'No';
        if (iAff>=0 && (cols[iAff]||'').trim()) state.affected[aid] = cols[iAff].trim();
        if (iNote>=0 && (cols[iNote]||'').trim()) state.notes[aid] = cols[iNote].trim();
        n++;
      }
    }
    saveState(); render(); toast(`已导入 ${n} 条已有 TypeA 标注（当前共 ${Object.keys(state.verdicts).length}/68）`);
  };
  rd.readAsText(f, 'utf-8'); ev.target.value = '';
}
function importJSON(ev) {
  const f = ev.target.files[0]; if (!f) return;
  const rd = new FileReader();
  rd.onload = () => {
    try { const o = JSON.parse(rd.result); const st = o.state || o; state = { verdicts:st.verdicts||{}, flips:st.flips||{}, affected:st.affected||{}, notes:st.notes||{} }; saveState(); render(); toast('JSON 导入成功'); }
    catch(e) { alert('JSON 无效：' + e.message); }
  };
  rd.readAsText(f, 'utf-8'); ev.target.value = '';
}
document.addEventListener('keydown', e => {
  if (e.target && e.target.tagName==='INPUT') return;
  if (e.key==='ArrowRight') step(1);
  else if (e.key==='ArrowLeft') step(-1);
  else if (e.key.toLowerCase()==='u') jumpNext();
  else if (e.key==='1') pick('Faithful');
  else if (e.key==='2') pick('Minor');
  else if (e.key==='3') pick('Unfaithful');
});
window.addEventListener('DOMContentLoaded', () => {
  const last = localStorage.getItem('cscwd_typeA68_seed2026_last');
  if (last) document.getElementById('annotator').value = last;
  loadState(); render();
});
window.addEventListener('load', () => typeset(document.getElementById('mainContent')));
</script>
</body>
</html>
'''

    out_html = OUT_DIR / "typeA68_cert_audit_seed2026.html"
    out_html.write_text(html.replace("__PAYLOAD_JSON__", payload_json), encoding="utf-8", newline="\n")
    print(f"Generated {out_html} ({out_html.stat().st_size / 1024:.1f} KB)")
    print(f"TypeA items: {len(records)}, total evaluated non-final prefixes: {total_eval_prefixes}, unique prefixes: {total_unique_prefixes}")


if __name__ == "__main__":
    main()
