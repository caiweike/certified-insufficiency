#!/usr/bin/env python3
"""Input-blind Dual-LLM Scope & Repair re-judger (`exp2-part2-dual-llm-v1.3-input-blind`).

Never overwrites its input file or caches. Reuses the input's final-answer (Y) gradings
verbatim (build_final_arbiter_messages never received mechanism or user_message),
and re-grades non-final Scope C and post-commit Explicit Repair with:
  1. System prompt line 2 stripped of "、题型机制" (scoring rules 1-4 unchanged).
  2. Scope user prompt stripped of "【机制类别】..." and replacing every turn's
     `user_message` with ONLY the raw disclosed clause text from `frozen_*.jsonl`.
  3. Repair user prompt replacing every turn's `user_message` with ONLY the raw
     disclosed clause text from `frozen_*.jsonl`, evaluated on ALL trajectories
     with new v1.3 `e_U == 1`.

Supports resumable batched runs via `--protocols` sharing one v1.3 cache file.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2A_DIR = SCRIPT_DIR.parent / "colab_stage2_sharding"
for p in (SCRIPT_DIR, STAGE2A_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import judge_trajectories as jt  # noqa: E402
from protocol_utils import NONFINAL_SCOPES  # noqa: E402
from run_api_sharding import CapabilityState, resolve_reasoning_effort  # noqa: E402

GRADING_RULE_VERSION_V13 = "exp2-part2-dual-llm-v1.3-input-blind"
DEFAULT_RELEASE_DIR = SCRIPT_DIR.parent / "exp1_codex_gpt61_v1" / "frozen_release_verified_120"

# Exact judge_trajectories.SCOPE_PROMPT with ONLY "、题型机制" removed from line 2.
assert "、题型机制" in jt.SCOPE_PROMPT, "Expected '、题型机制' in judge_trajectories.SCOPE_PROMPT"
SCOPE_PROMPT_V13 = jt.SCOPE_PROMPT.replace("、题型机制", "", 1)
assert "题型机制" not in SCOPE_PROMPT_V13


def _shard_text_for_turn(item: dict[str, Any], tr: dict[str, Any], t: dict[str, Any]) -> tuple[str, str]:
    shard_map = {s["shard_id"]: s["shard"] for s in item["shards"]}
    sid = t.get("disclosed_shard_id") or tr["disclosure_order"][int(t["turn_idx"]) - 1]
    assert sid in shard_map, f"Missing shard {sid} for {tr['item_id']}"
    return sid, shard_map[sid]


def build_v13_scope_messages(item: dict[str, Any], tr: dict[str, Any], turn_idx: int) -> list[dict[str, str]]:
    """Input-blind nonfinal scope prompt: zero mechanism label, zero protocol/rule text in user turns."""
    shard_map = {s["shard_id"]: s["shard"] for s in item["shards"]}
    disclosed_ids = tr["disclosure_order"][:turn_idx]
    disclosed_block = "\n".join(f"[{sid}] {shard_map[sid]}" for sid in disclosed_ids)

    prefix_hist: list[str] = []
    for t in tr["turns"][:turn_idx]:
        sid, stxt = _shard_text_for_turn(item, tr, t)
        prefix_hist.append(f"Turn {t['turn_idx']} User ({sid}):\n{stxt}")
        prefix_hist.append(f"Turn {t['turn_idx']} Assistant (visible):\n{t['visible_reply']}")

    user_content = (
        f"【当前轮次】第 {turn_idx} 轮（非末轮）\n"
        f"【截至当前轮次已披露的题目分片】\n{disclosed_block}\n\n"
        f"【截至第 {turn_idx} 轮的完整可见对话记录】\n"
        + "\n---\n".join(prefix_hist)
        + f"\n\n请只对 **Turn {turn_idx} Assistant (visible)** 给出五分类 Scope 判定。"
    )
    return [{"role": "system", "content": SCOPE_PROMPT_V13}, {"role": "user", "content": user_content}]


def build_v13_repair_messages(item: dict[str, Any], tr: dict[str, Any], commit_turns: list[int]) -> list[dict[str, str]]:
    """Input-blind repair prompt: raw disclosed clauses only in User turns (zero protocol/rule text)."""
    hist: list[str] = []
    for t in tr["turns"]:
        sid, stxt = _shard_text_for_turn(item, tr, t)
        hist.append(f"Turn {t['turn_idx']} User ({sid}):\n{stxt}")
        hist.append(f"Turn {t['turn_idx']} Assistant:\n{t['visible_reply']}")
    user_content = (
        f"【完整题干】\n{item['question']}\n"
        f"【官方答案】{item['final_answer']}\n"
        f"【曾被判为 Commit 的非末轮轮次】{commit_turns}\n\n"
        f"【全过程可见对话】\n" + "\n---\n".join(hist)
    )
    return [{"role": "system", "content": jt.REPAIR_PROMPT}, {"role": "user", "content": user_content}]


def _cohen_kappa(y1: list[Any], y2: list[Any]) -> tuple[float, float | None]:
    n = len(y1)
    if n == 0:
        return 0.0, None
    agree = sum(a == b for a, b in zip(y1, y2)) / n
    c1 = Counter(y1)
    c2 = Counter(y2)
    pe = sum((c1[k] / n) * (c2[k] / n) for k in set(c1) | set(c2))
    kappa = None if abs(1.0 - pe) < 1e-12 else (agree - pe) / (1.0 - pe)
    return agree, kappa


def rejudge_v13_input_blind(
    src_records: list[dict[str, Any]],
    item_map: dict[str, dict[str, Any]],
    primary_client: Any,
    primary_model: str,
    primary_caps: Any,
    secondary_client: Any | None,
    secondary_model: str | None,
    secondary_caps: Any | None,
    protocols_filter: set[str] | None,
    workers: int,
    secondary_workers: int | None,
    max_tokens: int,
    cache_path: Path,
    resume: bool,
    skip_repair: bool = False,
) -> list[dict[str, Any]]:
    p_workers = max(1, int(workers))
    s_workers = max(1, int(secondary_workers if secondary_workers is not None else workers))
    has_secondary = secondary_client is not None and bool(secondary_model)

    cache_map: dict[str, dict[str, Any]] = {}
    cache_lock = threading.Lock()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    if resume:
        cache_map = jt._load_judge_cache(cache_path)
        if cache_map:
            print(f"[Cache v1.3] Loaded {len(cache_map)} completed calls from {cache_path}", flush=True)
        cache_fh = cache_path.open("ab")
    else:
        cache_fh = cache_path.open("wb")

    # Select records to include in output:
    # - Always keep single-turn protocols when protocols_filter is None.
    # - When protocols_filter is specified, only output records matching protocols_filter.
    if protocols_filter is not None:
        selected = [r for r in src_records if r["protocol"] in protocols_filter]
    else:
        selected = list(src_records)

    try:
        # Step 1: Collect all non-empty dynamic nonfinal Scope tasks
        scope_tasks: list[tuple[tuple[str, int], list[dict[str, str]]]] = []
        nonfinal_trunc_map: dict[tuple[str, int], dict[str, bool]] = {}
        for tr in selected:
            if tr["protocol"] in ("FULL", "CONCAT", "CONCAT_MATCHED"):
                continue
            item = item_map[tr["item_id"]]
            for t in tr["turns"]:
                if t["is_final_turn"]:
                    continue
                t_idx = int(t["turn_idx"])
                key = (tr["trajectory_id"], t_idx)
                tflags = jt.classify_turn_truncation(t)
                nonfinal_trunc_map[key] = tflags
                if tflags["empty_trunc"] or tflags["non_trunc_empty"]:
                    continue
                scope_tasks.append((key, build_v13_scope_messages(item, tr, t_idx)))

        scope_primary: dict[Any, dict[str, Any] | None] = {}
        scope_secondary: dict[Any, dict[str, Any] | None] = {}
        if scope_tasks:
            scope_primary, scope_secondary = jt._run_dual_branches(
                lambda: jt._run_parallel_json_tasks(
                    scope_tasks, primary_client, primary_model, primary_caps,
                    stage="scope", role="primary", workers=p_workers, max_tokens=max_tokens,
                    cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                ),
                (
                    lambda: jt._run_parallel_json_tasks(
                        scope_tasks, secondary_client, secondary_model or "", secondary_caps,
                        stage="scope", role="secondary", workers=s_workers, max_tokens=max_tokens,
                        cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                    )
                ) if has_secondary else None,
            )

        # Step 2: Assemble new v1.3 records, reusing the input's final_grading verbatim
        repair_candidates: list[tuple[dict[str, Any], list[int]]] = []
        annotated: list[dict[str, Any]] = []
        src_vs_v13_turns: list[tuple[int, int, str, str]] = []
        src_vs_v13_trajs: list[tuple[int, int, int, int]] = []
        scope_p_fail = 0
        scope_s_fail = 0
        p_bin_list, s_bin_list = [], []
        p_cat_list, s_cat_list = [], []

        for tr in selected:
            rec = json.loads(json.dumps(tr))
            tid = rec["trajectory_id"]
            rec["grading_rule_version"] = GRADING_RULE_VERSION_V13
            rec["final_grading_source_version"] = tr.get("grading_rule_version", "final-answer-judge")
            rec["repair_skipped"] = bool(skip_repair)

            fg = rec["final_grading"]
            y_l = int(fg["y_L"])
            y_u = int(fg["y_U"])

            if rec["protocol"] in ("FULL", "CONCAT", "CONCAT_MATCHED"):
                rec["e_L"] = 0
                rec["e_U"] = 0
                rec["structural_zero_E"] = True
                rec["j_L"] = y_l
                rec["j_U"] = y_u
                rec["repair"] = None
                annotated.append(rec)
                continue

            src_e_l = int(tr["e_L"])
            src_e_u = int(tr["e_U"])
            e_l_traj = 0
            e_u_traj = 0
            commit_turns: list[int] = []

            for t, t_old in zip(rec["turns"], tr["turns"]):
                if t["is_final_turn"]:
                    t["sufficiency_L"] = "Sufficient"
                    t["scope_C"] = None
                    continue
                t["sufficiency_L"] = "Insufficient"
                key = (tid, int(t["turn_idx"]))
                tflags_nf = nonfinal_trunc_map[key]
                t["empty_trunc"] = tflags_nf["empty_trunc"]
                t["vis_trunc"] = tflags_nf["vis_trunc"]
                t["non_trunc_empty"] = tflags_nf["non_trunc_empty"]
                if tflags_nf["empty_trunc"] or tflags_nf["non_trunc_empty"]:
                    t["scope_C"] = "EMPTY_TRUNCATED" if tflags_nf["empty_trunc"] else "NON_TRUNC_EMPTY"
                    t["e_L"] = 0
                    t["e_U"] = 0
                    t["dual_scope_agreed"] = True
                    t["dual_commit_agreed"] = True
                    t["primary_failed"] = False
                    t["secondary_failed"] = False
                    continue

                p_obj = scope_primary.get(key)
                s_obj = scope_secondary.get(key)
                p_valid = isinstance(p_obj, dict) and p_obj.get("scope_label") in NONFINAL_SCOPES
                p_failed = not p_valid
                if p_failed:
                    scope_p_fail += 1
                p_lbl = str(p_obj["scope_label"]) if p_valid else "Unresolved"

                s_valid = isinstance(s_obj, dict) and s_obj.get("scope_label") in NONFINAL_SCOPES
                s_failed = bool(has_secondary and not s_valid)
                if s_failed:
                    scope_s_fail += 1
                s_lbl = (str(s_obj["scope_label"]) if s_valid else "Unresolved") if has_secondary else None

                c_lbl, t_el, t_eu, cat_agreed, commit_agreed = jt._combine_dual_scope(p_lbl, s_lbl)
                t["scope_C"] = c_lbl
                t["primary_scope"] = p_lbl
                t["secondary_scope"] = s_lbl
                t["primary_failed"] = p_failed
                t["secondary_failed"] = s_failed
                t["scope_rationale"] = p_obj.get("rationale", "") if isinstance(p_obj, dict) else ""
                t["secondary_scope_rationale"] = s_obj.get("rationale", "") if isinstance(s_obj, dict) else ""
                t["dual_scope_agreed"] = cat_agreed
                t["dual_commit_agreed"] = commit_agreed
                t["e_L"] = t_el
                t["e_U"] = t_eu
                e_l_traj = max(e_l_traj, t_el)
                e_u_traj = max(e_u_traj, t_eu)
                if t_eu == 1:
                    commit_turns.append(int(t["turn_idx"]))

                if has_secondary and not p_failed and not s_failed:
                    p_bin_list.append(int(p_lbl == "Commit"))
                    s_bin_list.append(int(s_lbl == "Commit"))
                    p_cat_list.append(p_lbl)
                    s_cat_list.append(s_lbl)
                src_vs_v13_turns.append((int(t_old.get("e_L", 0)), t_el, str(t_old.get("scope_C")), c_lbl))

            rec["any_empty_truncated"] = any(bool(t.get("empty_trunc")) for t in rec["turns"])
            rec["any_vis_truncated"] = any(bool(t.get("vis_trunc")) for t in rec["turns"])
            rec["e_L"] = e_l_traj
            rec["e_U"] = e_u_traj
            rec["structural_zero_E"] = False
            rec["j_L"] = y_l * (1 - e_u_traj)
            rec["j_U"] = y_u * (1 - e_l_traj)
            rec["repair"] = None
            src_vs_v13_trajs.append((src_e_l, e_l_traj, src_e_u, e_u_traj))
            if e_u_traj == 1:
                repair_candidates.append((rec, commit_turns))
            annotated.append(rec)

        # Step 3: Input-blind Repair grading on all new v1.3 E_U == 1 trajectories
        rep_p_fail = 0
        rep_s_fail = 0
        if not skip_repair and repair_candidates:
            repair_tasks = [
                (rec["trajectory_id"], build_v13_repair_messages(item_map[rec["item_id"]], rec, cturns))
                for rec, cturns in repair_candidates
            ]
            repair_primary, repair_secondary = jt._run_dual_branches(
                lambda: jt._run_parallel_json_tasks(
                    repair_tasks, primary_client, primary_model, primary_caps,
                    stage="repair", role="primary", workers=p_workers, max_tokens=max_tokens,
                    cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                ),
                (
                    lambda: jt._run_parallel_json_tasks(
                        repair_tasks, secondary_client, secondary_model or "", secondary_caps,
                        stage="repair", role="secondary", workers=s_workers, max_tokens=max_tokens,
                        cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                    )
                ) if has_secondary else None,
            )
            for rec, _ in repair_candidates:
                tid = rec["trajectory_id"]
                p_rep = repair_primary.get(tid)
                s_rep = repair_secondary.get(tid) if has_secondary else None
                comb = jt._combine_dual_repair(p_rep, s_rep, secondary_expected=has_secondary)
                if comb["primary_failed"]:
                    rep_p_fail += 1
                if comb["secondary_failed"]:
                    rep_s_fail += 1
                rec["repair"] = comb

        # Summary telemetry
        c_ag, c_kp = _cohen_kappa(p_bin_list, s_bin_list)
        s_ag, s_kp = _cohen_kappa(p_cat_list, s_cat_list)
        t_el_ag, t_el_kp = _cohen_kappa([x[0] for x in src_vs_v13_turns], [x[1] for x in src_vs_v13_turns])
        tr_el_ag, tr_el_kp = _cohen_kappa([x[0] for x in src_vs_v13_trajs], [x[1] for x in src_vs_v13_trajs])
        tr_eu_ag, tr_eu_kp = _cohen_kappa([x[2] for x in src_vs_v13_trajs], [x[3] for x in src_vs_v13_trajs])
        print(
            f"[v1.3 Summary] scope_turns={len(scope_tasks)} fails(S_P/S_S/R_P/R_S)={scope_p_fail}/{scope_s_fail}/{rep_p_fail}/{rep_s_fail} "
            f"| Dual Commit turn agree={c_ag:.4f} kappa={c_kp} | Dual Scope agree={s_ag:.4f} kappa={s_kp} "
            f"| Repair cand={len(repair_candidates)}",
            flush=True,
        )
        print(
            f"[input vs input-blind] turn_eL agree={t_el_ag:.4f} kappa={t_el_kp} "
            f"| traj_eL agree={tr_el_ag:.4f} kappa={tr_el_kp} "
            f"| traj_eU agree={tr_eu_ag:.4f} kappa={tr_eu_kp} "
            f"| src_eL_sum={sum(x[0] for x in src_vs_v13_trajs)} -> v13_eL_sum={sum(x[1] for x in src_vs_v13_trajs)} "
            f"| src_eU_sum={sum(x[2] for x in src_vs_v13_trajs)} -> v13_eU_sum={sum(x[3] for x in src_vs_v13_trajs)}",
            flush=True,
        )
    finally:
        cache_fh.close()

    return annotated


def _run_self_test() -> None:
    item = {
        "item_id": "test-001",
        "question": "已知 x + y = 3，求 x。",
        "final_answer": "1",
        "shards": [
            {"shard_id": "S1", "shard": "已知 x, y，求 x。"},
            {"shard_id": "S2", "shard": "x + y = 3。"},
            {"shard_id": "S3", "shard": "y = 2。"},
        ],
    }
    tr = {
        "item_id": "test-001",
        "trajectory_id": "test-001___DYN_REQUEST_GUARD___r0",
        "protocol": "DYN_REQUEST_GUARD",
        "verified_mechanism": "TypeA_UniqueValue",
        "disclosure_order": ["S1", "S2", "S3"],
        "turns": [
            {
                "turn_idx": 1,
                "disclosed_shard_id": "S1",
                "is_final_turn": False,
                "user_message": "【任务规则】先验证唯一性再使用\\boxed{}\n已知 x, y，求 x。\n请基于目前已提供的信息给出当前答案。",
                "visible_reply": "条件不足，缺少关于 x 和 y 的方程。",
            },
            {
                "turn_idx": 2,
                "disclosed_shard_id": "S2",
                "is_final_turn": False,
                "user_message": "x + y = 3。\n请基于目前已提供的信息给出当前答案。",
                "visible_reply": "我认为 x = 1。",
            },
            {
                "turn_idx": 3,
                "disclosed_shard_id": "S3",
                "is_final_turn": True,
                "user_message": "y = 2。\n全部条件已给出。",
                "visible_reply": "最终答案 x = 1。",
            },
        ],
    }

    s_msgs = build_v13_scope_messages(item, tr, 2)
    r_msgs = build_v13_repair_messages(item, tr, [2])
    for name, msgs in [("scope", s_msgs), ("repair", r_msgs)]:
        joined = msgs[0]["content"] + "\n" + msgs[1]["content"]
        for forbidden in ("机制类别", "TypeA_UniqueValue", "TypeB_Extremum", "任务规则", "先验证唯一性", "请基于目前已提供的信息给出当前答案", "全部条件已给出", "题型机制"):
            assert forbidden not in joined, f"Leak detected in {name}: {forbidden!r}"
        assert "已知 x, y，求 x。" in msgs[1]["content"]
        assert "x + y = 3。" in msgs[1]["content"]
    print("judge_v13_input_blind.py --self-test PASSED (0 leaks in Scope & Repair prompts).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Input-blind dual-LLM scope & repair judge (reuses final-answer Y gradings from --input-judged).")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--input-judged", type=Path, default=None)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--split", choices=["DEV", "TEST"], default="TEST")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None)
    parser.add_argument("--protocols", type=str, default=None, help="Optional comma-separated protocol filter (e.g. DYN_REQUEST).")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:3425/v1"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--primary-model", type=str, default="codex/gpt-6.1-sol")
    parser.add_argument("--secondary-model", type=str, default="workbuddy/deepseek-v4.1-flash")
    parser.add_argument("--secondary-base-url", type=str, default=None)
    parser.add_argument("--secondary-api-key", type=str, default=None)
    parser.add_argument("--reasoning-effort", type=str, default="low")
    parser.add_argument("--secondary-reasoning-effort", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--secondary-workers", type=int, default=None)
    parser.add_argument("--max-tokens", type=int, default=16384)
    args = parser.parse_args()

    if args.self_test:
        _run_self_test()
        return

    assert args.input_judged is not None and args.output is not None
    assert args.input_judged.resolve() != args.output.resolve(), "Refusing to overwrite the input file!"

    from openai import OpenAI

    items_path = args.release_dir / ("frozen_dev_20.jsonl" if args.split == "DEV" else "frozen_test_100.jsonl")
    item_map = {x["item_id"]: x for x in jt.read_jsonl(items_path)}
    src_records = [r for r in jt.read_jsonl(args.input_judged) if r["item_id"] in item_map]
    pfilter = {x.strip() for x in args.protocols.split(",") if x.strip()} if args.protocols else None

    api_key = args.api_key or ("cache-replay-only" if args.resume else "")
    if not api_key:
        raise SystemExit("Missing API key: pass --api-key or set OPENAI_API_KEY (or pass --resume with a complete cache).")
    p_client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=90.0)
    eff, dis = resolve_reasoning_effort(args.primary_model, args.reasoning_effort)
    p_caps = CapabilityState(eff, True, dis)

    s_client = None
    s_caps = None
    if args.secondary_model:
        s_client = OpenAI(
            api_key=args.secondary_api_key or api_key,
            base_url=args.secondary_base_url or args.base_url,
            timeout=90.0,
        )
        s_eff_arg = args.secondary_reasoning_effort if args.secondary_reasoning_effort is not None else args.reasoning_effort
        seff, sdis = resolve_reasoning_effort(args.secondary_model, s_eff_arg)
        s_caps = CapabilityState(seff, True, sdis)

    cache_path = args.cache_path or args.output.with_name(args.output.stem + "_judge_cache.jsonl")
    t0 = time.time()
    judged = rejudge_v13_input_blind(
        src_records=src_records,
        item_map=item_map,
        primary_client=p_client,
        primary_model=args.primary_model,
        primary_caps=p_caps,
        secondary_client=s_client,
        secondary_model=args.secondary_model,
        secondary_caps=s_caps,
        protocols_filter=pfilter,
        workers=args.workers,
        secondary_workers=args.secondary_workers,
        max_tokens=args.max_tokens,
        cache_path=cache_path,
        resume=args.resume,
        skip_repair=args.skip_repair,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as fh:
        for r in judged:
            fh.write((json.dumps(r, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    print(f"Wrote {len(judged)} v1.3 trajectories -> {args.output} in {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
