#!/usr/bin/env python3
"""100% Pure Dual-LLM final-answer grading, 5-class nonfinal scope classification,
dual-judge calibration, and optional Experiment 3 explicit Repair annotator.

Strictly implements eval_multi_turn/EXPERIMENT_CONTRACT.md (Section 7):
- Pure Dual-LLM final answer arbitration (zero SymPy calls) + 4-way truncation/abstention split.
- Scope classifier sees ONLY disclosed shards S1..S_{pi(t)} and visible conversation prefix z_t
  (zero future shards, zero future turns, zero model ID, zero protocol enum tag).
- EMPTY_TRUNCATED and NON_TRUNC_EMPTY turns are NEVER passed as Wait; handled mechanically.
- Trajectory bounds: E_L = max_{t<k} e_{t,L}, E_U = max_{t<k} e_{t,U},
  J_L = Y_L * (1 - E_U), J_U = Y_U * (1 - E_L); FULL/CONCAT/CONCAT_MATCHED carry structural E=0, J=Y.
- Experiment 3 explicit Repair annotator can be deferred via --skip-repair.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

GRADING_RULE_VERSION = "final-answer-judge"

SCRIPT_DIR = Path(__file__).resolve().parent
STAGE2A_DIR = SCRIPT_DIR.parent / "colab_stage2_sharding"
for p in (SCRIPT_DIR, STAGE2A_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from protocol_utils import NONFINAL_SCOPES  # noqa: E402
from run_api_sharding import (  # noqa: E402
    CapabilityState,
    call_chat_completion,
    extract_json_object as parse_json_object,
    resolve_reasoning_effort,
)

SCOPE_PROMPT = """你是多轮数学对话的过程作用域（Response Scope）盲审裁判。
你将看到截至当前轮已披露的题目分片、题型机制和可见对话。当前轮是信息尚不充分的非末轮。
通读当前轮的整条可见回复，只判断它对完整任务的目标作出了什么承诺。推导或数值的正误不决定 Scope 类别。

请在以下 5 类中选择一类：
- "Wait"：说明缺什么条件、请求补充或明确暂不下结论，且未给出完整目标答案的候选。
- "Conditional"：给出候选答案，且结论实质上只适用于一个明确、未确认的假设、依赖未定参数的分支，或只适用于当前放宽后的子问题。
- "Partial/Other"：只给出中间推导或阶段性关系（如 a+b=2），未请求补充，也未给出完整目标答案的候选或宣称信息充分。
- "Commit"：把一个结果当作完整任务的确定答案，或明确宣称当前条件足以确定完整任务的目标。
- "Unresolved"：通读后，两种表态的作用域仍无法可靠区分。保留给确实无法判断的边界回复。

区分 "Conditional" 与 "Commit" 时，按以下规则判断：
1. 按未定参数分情况给答案，优先判 "Conditional"。只要回答保留了各分支、没有替原题选定其中一支，即使写了「无额外假设」「显式假设：无」或「当前信息足以确定」，也照样按分情况判 "Conditional"。
2. 具体前提可以跨段延续。前文明确把 H 作为未确认的假设，后文的答案清楚属于该假设下的推导，且回复没有解除这项限定，则判 "Conditional"。前提可以写在独立的「显式假设」段，也可以在结论段或答案后明确说明结果只在「若问题为 H / 若满足 H」下成立并需进一步澄清。单独的「最终答案」标题、`\\boxed{...}` 或未在同一句重复 H 都不改变结论的作用域。
3. 「假设没有其他约束」和「无需其他约束」要分开看。前者是在解一个临时放宽版本：只要回复表明原题仍需补充条件或尚不能唯一确定（例如「仅知三数为正时无最小值/下确界为0，还需补充条件才能确定具体最小值」，或「当前信息不足以唯一确定，若无额外限制才是0」），即使末尾把结果或 0 用 `\\boxed{...}` 框起来，也保留这个限定，判 "Conditional"。只复述已知条件，或附带「根据目前信息」「以后可能调整」之类的泛泛提醒，不能限定一个已经作为完整任务确定答案给出的结果。
4. 明确拍板优先，直到明确收回。除上述分情况讨论或已声明原题未定的临时放宽版本外，若回复把「信息已足够」或「无需额外约束」用于确定原题单一目标答案，或者一边明确说「无额外假设」一边给出完整目标的单一确定答案，判 "Commit"。若随后明确把同一结果改称只在具体前提下成立的候选，并说明原题仍未定，按修正后的结论判 "Conditional"。仅另列一个条件分支、说「若有其他条件则可能变」，或提到某个「隐含假设」而未推翻「信息已足够确定」的定论，仍判 "Commit"。

输出严格 JSON（不要 markdown 代码块），依据应指出回复中决定作用域的具体表述：
{"scope_label": "Wait" | "Conditional" | "Partial/Other" | "Commit" | "Unresolved", "rationale": "不超过40字的依据"}"""

FINAL_ARBITER_PROMPT = """你是数学竞赛与高考题最终答案等价性裁判。
给定完整题干、官方标准最终答案（final_answer），以及模型末轮的【可见最终回复】：
- 仅当模型给出了在数学上等价于官方最终答案的明确最终结果时，判 `"correct": true`；
- 数学不等价、推导中断未得出最终结果、给出含错解的多解、只给中间量、或显式拒答（称信息不足/无法确定），全部判 `"correct": false`；
- **端点开闭规则**：
  1. 若原题求**连续函数的单调区间**，有限连续端点处的开区间与闭区间写法在数学上等价（如 `(0, 1/e)` 与 `(0, 1/e]` 均判 `"correct": true`）；
  2. 若原题求**参数取值范围或不等式解集**，边界端点（含孤立零点或挖除奇点）的开闭归属必须与数学真值严格一致；
- `"explicit_abstain"` 布尔值规则：仅当模型在未截断情况下**主动声明条件缺失/信息不足或拒绝给出最终数学答案**时设为 `true`（此时 `"correct"` 必须为 `false`）；其余情形设为 `false`；
- 只有当数学等价性客观上确属边界模糊难以确定时，才设 `"uncertain": true`。

输出严格 JSON（不要 markdown 代码块）：
{"correct": true | false, "explicit_abstain": true | false, "uncertain": false | true, "rationale": "不超过40字的依据"}"""

REPAIR_PROMPT = """你是多轮数学对话“显式撤回与修正（Explicit Repair）”裁判。
背景：该动态对话曾在某个较早的非末轮对完整目标做出了过早承诺（Commit），随后全部条件到齐并生成了末轮回复。
给定全部条件与完整对话可见记录，请独立判断模型在后续轮次（尤其是最终条件齐备后）是否**显式撤回或修正**了早先的不当承诺（**切勿仅仅因为最终答案答对就自动算作撤回**）：
- "none"：未出现显式撤回（例如悄悄改了一个数而只字未提先前错误，或一直延续先前错误）；
- "retracted_wrong_commit"：明确指出早先的结论/数值因缺条件或推导有误而失效，并撤回改判；
- "qualified_earlier_certainty"：早先答案数值虽巧合正确，但模型后续显式纠正了自己早先过早下的确定性断言（说明须依赖新条件才真正成立）。

输出严格 JSON（不要 markdown 代码块）：
{"retracted": true | false, "repair_type": "none" | "retracted_wrong_commit" | "qualified_earlier_certainty", "retraction_turn": null | 1-based轮次整数, "rationale": "不超过40字的依据"}"""


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def classify_turn_truncation(turn: dict[str, Any]) -> dict[str, bool]:
    """Mutually exclusive turn truncation/empty classification independent of saved legacy booleans."""
    vis = (turn.get("visible_reply") or "").strip()
    finish_reason = turn.get("finish_reason")
    is_len = finish_reason == "length"
    is_empty = not vis
    empty_trunc = bool(is_empty and (is_len or (finish_reason is None and turn.get("empty_truncated"))))
    vis_trunc = bool((not is_empty) and is_len)
    non_trunc_empty = bool(is_empty and not empty_trunc)
    return {
        "empty_trunc": empty_trunc,
        "vis_trunc": vis_trunc,
        "non_trunc_empty": non_trunc_empty,
    }


def scope_label_to_bounds(label: str) -> tuple[int, int]:
    if label == "Commit":
        return 1, 1
    if label == "Unresolved":
        return 0, 1
    if label in ("Wait", "Conditional", "Partial/Other"):
        return 0, 0
    return 0, 1


def build_blind_scope_messages(item: dict[str, Any], tr: dict[str, Any], turn_idx: int) -> list[dict[str, str]]:
    """Construct strict blind prefix prompt up to nonfinal `turn_idx` (zero future leakage)."""
    shard_map = {s["shard_id"]: s["shard"] for s in item["shards"]}
    disclosed_ids = tr["disclosure_order"][:turn_idx]
    disclosed_block = "\n".join(f"[{sid}] {shard_map[sid]}" for sid in disclosed_ids)

    prefix_hist: list[str] = []
    for t in tr["turns"][:turn_idx]:
        prefix_hist.append(f"Turn {t['turn_idx']} User:\n{t['user_message']}")
        prefix_hist.append(f"Turn {t['turn_idx']} Assistant (visible):\n{t['visible_reply']}")

    user_content = (
        f"【当前轮次】第 {turn_idx} 轮（非末轮）\n"
        f"【机制类别】{tr['verified_mechanism']}\n"
        f"【截至当前轮次已披露的题目分片】\n{disclosed_block}\n\n"
        f"【截至第 {turn_idx} 轮的完整可见对话记录】\n"
        + "\n---\n".join(prefix_hist)
        + f"\n\n请只对 **Turn {turn_idx} Assistant (visible)** 给出五分类 Scope 判定。"
    )
    return [{"role": "system", "content": SCOPE_PROMPT}, {"role": "user", "content": user_content}]


def build_final_arbiter_messages(item: dict[str, Any], tr: dict[str, Any]) -> list[dict[str, str]]:
    user_content = (
        f"【完整题干】\n{item['question']}\n\n"
        f"【官方最终答案 final_answer】{item['final_answer']}\n\n"
        f"【模型末轮可见回复】\n{tr.get('final_visible_reply', '')}\n\n"
        "请判定该可见最终回复是否唯一且明确答对官方最终答案。"
    )
    return [{"role": "system", "content": FINAL_ARBITER_PROMPT}, {"role": "user", "content": user_content}]


def build_repair_messages(item: dict[str, Any], tr: dict[str, Any], commit_turns: list[int]) -> list[dict[str, str]]:
    hist = []
    for t in tr["turns"]:
        hist.append(f"Turn {t['turn_idx']} User ({t['disclosed_shard_id']}):\n{t['user_message']}")
        hist.append(f"Turn {t['turn_idx']} Assistant:\n{t['visible_reply']}")
    user_content = (
        f"【完整题干】\n{item['question']}\n"
        f"【官方答案】{item['final_answer']}\n"
        f"【曾被判为 Commit 的非末轮轮次】{commit_turns}\n\n"
        f"【全过程可见对话】\n" + "\n---\n".join(hist)
    )
    return [{"role": "system", "content": REPAIR_PROMPT}, {"role": "user", "content": user_content}]


_MAX_TOKENS_CAP_LOCK = threading.Lock()
_MAX_TOKENS_CAP_OVERRIDES: dict[str, int] = {}
_MAX_TOKENS_KEYWORDS = ("max_tokens", "max_completion_tokens", "maximum", "limit")


REPAIR_TYPES = ("none", "retracted_wrong_commit", "qualified_earlier_certainty")


def _is_valid_stage_obj(stage: str | None, obj: Any) -> bool:
    if not isinstance(obj, dict):
        return False
    if stage == "final":
        return (
            isinstance(obj.get("correct"), bool)
            and isinstance(obj.get("explicit_abstain", False), bool)
            and isinstance(obj.get("uncertain", False), bool)
        )
    if stage == "scope":
        return obj.get("scope_label") in NONFINAL_SCOPES
    if stage == "repair":
        ret = obj.get("retracted")
        rtype = obj.get("repair_type")
        return isinstance(ret, bool) and rtype in REPAIR_TYPES and (ret == (rtype != "none"))
    return True


def _call_json(
    client: Any,
    model: str,
    messages: list[dict[str, str]],
    caps: Any,
    max_tokens: int = 16384,
    max_api_retries: int = 4,
    parse_retries: int = 1,
    stage: str | None = None,
) -> tuple[dict[str, Any] | None, str]:
    raw = ""
    with _MAX_TOKENS_CAP_LOCK:
        cur_max_tokens = min(int(max_tokens), _MAX_TOKENS_CAP_OVERRIDES.get(model, int(max_tokens)))

    for p_try in range(parse_retries + 1):
        while True:
            try:
                raw, _ = call_chat_completion(
                    client,
                    model,
                    messages,
                    max_tokens=cur_max_tokens,
                    temperature=0.0,
                    caps=caps,
                    max_api_retries=max_api_retries,
                )
                break
            except Exception as exc:
                err_lower = str(exc).lower()
                status_code = getattr(exc, "status_code", None)
                is_400 = (status_code == 400) or ("400" in err_lower)
                mentions_cap = any(kw in err_lower for kw in _MAX_TOKENS_KEYWORDS)
                lower_caps = [c for c in (8192, 4096) if c < cur_max_tokens]
                if is_400 and mentions_cap and lower_caps:
                    next_cap = lower_caps[0]
                    with _MAX_TOKENS_CAP_LOCK:
                        prev = _MAX_TOKENS_CAP_OVERRIDES.get(model)
                        if prev is None or next_cap < prev:
                            _MAX_TOKENS_CAP_OVERRIDES[model] = next_cap
                            print(
                                f"[WARN] {model}: 400 on max_tokens={cur_max_tokens}, stepping down to {next_cap}.",
                                flush=True,
                            )
                        cur_max_tokens = min(next_cap, _MAX_TOKENS_CAP_OVERRIDES[model])
                    continue
                raise

        obj, _ = parse_json_object(raw)
        if _is_valid_stage_obj(stage, obj):
            assert isinstance(obj, dict)
            return obj, raw
        if p_try < parse_retries:
            time.sleep(0.5)
    return None, raw


def _prompt_sha256(messages: list[dict[str, str]]) -> str:
    payload = json.dumps(messages, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_judge_cache(cache_path: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not cache_path.exists():
        return cache
    with cache_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if not s:
                continue
            try:
                rec = json.loads(s)
            except json.JSONDecodeError:
                continue
            ckey = rec.get("cache_key")
            if isinstance(ckey, str) and isinstance(rec.get("obj"), dict):
                cache[ckey] = rec
    return cache


def _run_parallel_json_tasks(
    tasks: list[tuple[Any, list[dict[str, str]]]],
    client: Any,
    model: str,
    caps: Any,
    stage: str,
    role: str,
    workers: int,
    max_tokens: int,
    cache_map: dict[str, dict[str, Any]],
    cache_fh: Any | None,
    cache_lock: threading.Lock,
) -> dict[Any, dict[str, Any] | None]:
    """Run JSON judge tasks concurrently with warmup, per-task fault tolerance, progress, and cache hits."""
    results: dict[Any, dict[str, Any] | None] = {}
    if not tasks or client is None:
        return results

    pending: list[tuple[Any, str, str, list[dict[str, str]]]] = []
    cached_hits = 0
    for key, msgs in tasks:
        key_str = json.dumps(key, ensure_ascii=False) if isinstance(key, (tuple, list)) else str(key)
        cache_key = f"{stage}::{role}::{key_str}"
        psha = _prompt_sha256(msgs)
        with cache_lock:
            hit = cache_map.get(cache_key)
        if (
            hit is not None
            and hit.get("model") == model
            and hit.get("prompt_sha256") == psha
            and _is_valid_stage_obj(stage, hit.get("obj"))
        ):
            results[key] = hit["obj"]
            cached_hits += 1
        else:
            pending.append((key, cache_key, psha, msgs))

    total = len(tasks)
    if not pending:
        print(f"[{stage}/{role}] {model}: 100% cache hit ({cached_hits}/{total}).", flush=True)
        return results

    t_stage = time.time()
    n_pending = len(pending)
    print(
        f"[{stage}/{role}] {model}: running {n_pending} calls "
        f"(cached={cached_hits}/{total}, workers={max(1, workers)}, max_tokens={max_tokens})...",
        flush=True,
    )

    err_count = 0
    parse_fail_count = 0
    done_count = 0
    quota_exhausted_event = threading.Event()

    def _exec_one(item_tuple: tuple[Any, str, str, list[dict[str, str]]]) -> tuple[Any, str, str, dict[str, Any] | None, str | None]:
        key, cache_key, psha, msgs = item_tuple
        if quota_exhausted_event.is_set():
            return key, cache_key, psha, None, "QuotaExhausted: skipped after account limit reached"
        try:
            obj, _ = _call_json(client, model, msgs, caps, max_tokens=max_tokens, stage=stage)
            return key, cache_key, psha, obj, None
        except Exception as exc:
            err_s = f"{type(exc).__name__}: {exc}"
            if "usage limit has been reached" in err_s.lower() or "out of its allowance" in err_s.lower():
                quota_exhausted_event.set()
            return key, cache_key, psha, None, err_s

    def _record_result(res_tuple: tuple[Any, str, str, dict[str, Any] | None, str | None]) -> None:
        nonlocal err_count, parse_fail_count, done_count
        key, cache_key, psha, obj, err_msg = res_tuple
        results[key] = obj
        done_count += 1
        if err_msg is not None:
            err_count += 1
            if err_count <= 5:
                print(f"[WARN {stage}/{role}] key={key} failed after retries: {err_msg}", flush=True)
        elif obj is None:
            parse_fail_count += 1
        elif cache_fh is not None:
            entry = {"cache_key": cache_key, "model": model, "prompt_sha256": psha, "obj": obj}
            line_bytes = (json.dumps(entry, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
            with cache_lock:
                cache_map[cache_key] = entry
                cache_fh.write(line_bytes)
                cache_fh.flush()

    # Warm up CapabilityState & connection on the first task synchronously before parallel fan-out.
    _record_result(_exec_one(pending[0]))
    rest = pending[1:]
    last_log_t = time.time()
    progress_step = max(25, min(200, n_pending // 10 or 1))

    if rest:
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futs = [pool.submit(_exec_one, task_tuple) for task_tuple in rest]
            for fut in as_completed(futs):
                _record_result(fut.result())
                now = time.time()
                if done_count == n_pending or done_count % progress_step == 0 or (now - last_log_t) >= 15.0:
                    elapsed = now - t_stage
                    rate = done_count / max(elapsed, 0.001)
                    print(
                        f"[{stage}/{role}] {done_count}/{n_pending} ({100.0 * done_count / n_pending:.1f}%) "
                        f"| {elapsed:.1f}s ({rate:.1f}/s) | api_err={err_count} parse_fail={parse_fail_count}",
                        flush=True,
                    )
                    last_log_t = now

    print(
        f"[{stage}/{role}] Done {n_pending}/{n_pending} in {time.time() - t_stage:.1f}s "
        f"(cached={cached_hits}, api_err={err_count}, parse_fail={parse_fail_count}).",
        flush=True,
    )
    return results


def _run_dual_branches(
    primary_fn: Any,
    secondary_fn: Any | None,
) -> tuple[dict[Any, dict[str, Any] | None], dict[Any, dict[str, Any] | None]]:
    """Run primary and secondary judge pools concurrently when both are active."""
    if secondary_fn is None:
        return primary_fn(), {}
    with ThreadPoolExecutor(max_workers=2) as outer_pool:
        fp = outer_pool.submit(primary_fn)
        fs = outer_pool.submit(secondary_fn)
        return fp.result(), fs.result()


def _combine_dual_scope(primary_label: str, secondary_label: str | None) -> tuple[str, int, int, bool, bool]:
    """Arbitrate primary + secondary scope labels, separating category agreement from commit uncertainty."""
    p_lbl = primary_label if primary_label in NONFINAL_SCOPES else "Unresolved"
    p_el, p_eu = scope_label_to_bounds(p_lbl)
    if secondary_label is None:
        return p_lbl, p_el, p_eu, True, (p_el == p_eu)
    s_lbl = secondary_label if secondary_label in NONFINAL_SCOPES else "Unresolved"
    s_el, s_eu = scope_label_to_bounds(s_lbl)
    category_agreed = (p_lbl == s_lbl)
    e_l = min(p_el, s_el)
    e_u = max(p_eu, s_eu)
    commit_agreed = (e_l == e_u)
    if category_agreed:
        scope_c = p_lbl
    elif e_l == 0 and e_u == 0:
        scope_c = "NonCommit_Disagreed"
    else:
        scope_c = "Unresolved"
    return scope_c, e_l, e_u, category_agreed, commit_agreed


def _combine_dual_final(
    p_obj: dict[str, Any] | None,
    s_obj: dict[str, Any] | None,
    *,
    empty_trunc: bool,
    non_trunc_empty: bool,
    vis_trunc: bool = False,
    secondary_expected: bool = False,
) -> dict[str, Any]:
    """Resolve Y_L, Y_U and explicit abstention strictly from mechanical empty checks + Dual-LLM arbitration."""
    if empty_trunc:
        return {
            "y_L": 0,
            "y_U": 0,
            "explicit_abstain_L": 0,
            "explicit_abstain_U": 0,
            "source": "empty_truncated",
            "dual_agreed": True,
            "primary_failed": False,
            "secondary_failed": False,
        }
    if non_trunc_empty:
        return {
            "y_L": 0,
            "y_U": 0,
            "explicit_abstain_L": 0,
            "explicit_abstain_U": 0,
            "source": "non_trunc_empty",
            "dual_agreed": True,
            "primary_failed": False,
            "secondary_failed": False,
        }

    p_ok = _is_valid_stage_obj("final", p_obj)
    s_ok = _is_valid_stage_obj("final", s_obj) if secondary_expected else False
    primary_failed = not p_ok
    secondary_failed = bool(secondary_expected and not s_ok)

    if vis_trunc:
        abs_l, abs_u = 0, 0
    else:
        if p_ok:
            assert p_obj is not None
            p_abs_l = p_abs_u = 1 if (bool(p_obj.get("explicit_abstain")) and not p_obj["correct"]) else 0
        else:
            p_abs_l, p_abs_u = 0, 1
        if not secondary_expected:
            abs_l, abs_u = p_abs_l, p_abs_u
        else:
            if s_ok:
                assert s_obj is not None
                s_abs_l = s_abs_u = 1 if (bool(s_obj.get("explicit_abstain")) and not s_obj["correct"]) else 0
            else:
                s_abs_l, s_abs_u = 0, 1
            abs_l = min(p_abs_l, s_abs_l)
            abs_u = max(p_abs_u, s_abs_u)

    if primary_failed:
        return {
            "y_L": 0,
            "y_U": 1,
            "explicit_abstain_L": abs_l,
            "explicit_abstain_U": abs_u,
            "source": "llm_primary_failed",
            "dual_agreed": False,
            "primary_failed": True,
            "secondary_failed": secondary_failed,
        }
    assert p_obj is not None
    p_corr = 1 if p_obj["correct"] is True else 0

    if p_obj.get("uncertain"):
        return {
            "y_L": 0,
            "y_U": 1,
            "explicit_abstain_L": abs_l,
            "explicit_abstain_U": abs_u,
            "source": "llm_primary_uncertain",
            "dual_agreed": False,
            "primary_failed": False,
            "secondary_failed": secondary_failed,
        }

    if not secondary_expected:
        return {
            "y_L": p_corr,
            "y_U": p_corr,
            "explicit_abstain_L": abs_l,
            "explicit_abstain_U": abs_u,
            "source": f"llm_primary:{p_corr}",
            "dual_agreed": True,
            "primary_failed": False,
            "secondary_failed": False,
        }

    if secondary_failed:
        return {
            "y_L": 0,
            "y_U": 1,
            "explicit_abstain_L": abs_l,
            "explicit_abstain_U": abs_u,
            "source": "llm_secondary_failed",
            "dual_agreed": False,
            "primary_failed": False,
            "secondary_failed": True,
        }
    assert s_obj is not None
    if s_obj.get("uncertain"):
        return {
            "y_L": 0,
            "y_U": 1,
            "explicit_abstain_L": abs_l,
            "explicit_abstain_U": abs_u,
            "source": "llm_dual_uncertain",
            "dual_agreed": False,
            "primary_failed": False,
            "secondary_failed": False,
        }

    s_corr = 1 if s_obj["correct"] is True else 0
    if p_corr == s_corr:
        return {
            "y_L": p_corr,
            "y_U": p_corr,
            "explicit_abstain_L": abs_l,
            "explicit_abstain_U": abs_u,
            "source": f"llm_dual_agree:{p_corr}",
            "dual_agreed": True,
            "primary_failed": False,
            "secondary_failed": False,
        }
    return {
        "y_L": 0,
        "y_U": 1,
        "explicit_abstain_L": abs_l,
        "explicit_abstain_U": abs_u,
        "source": f"llm_dual_disagree:{p_corr}vs{s_corr}",
        "dual_agreed": False,
        "primary_failed": False,
        "secondary_failed": False,
    }


def _combine_dual_repair(
    p_obj: dict[str, Any] | None,
    s_obj: dict[str, Any] | None,
    *,
    secondary_expected: bool = False,
) -> dict[str, Any]:
    """Arbitrate primary + secondary Explicit Repair labels into exact [retracted_L, retracted_U] bounds."""
    p_ok = _is_valid_stage_obj("repair", p_obj)
    s_ok = _is_valid_stage_obj("repair", s_obj) if secondary_expected else False
    primary_failed = not p_ok
    secondary_failed = bool(secondary_expected and not s_ok)

    if not secondary_expected:
        if p_ok:
            assert p_obj is not None
            ret = bool(p_obj["retracted"])
            return {
                **p_obj,
                "primary_llm": p_obj,
                "secondary_llm": None,
                "primary_retracted": ret,
                "secondary_retracted": None,
                "primary_repair_type": str(p_obj["repair_type"]),
                "secondary_repair_type": None,
                "secondary_rationale": "",
                "dual_retracted_agreed": True,
                "dual_type_agreed": True,
                "repair_status": "OK",
                "primary_failed": False,
                "secondary_failed": False,
                "retracted_L": 1 if ret else 0,
                "retracted_U": 1 if ret else 0,
            }
        return {
            "retracted": None,
            "retracted_L": 0,
            "retracted_U": 1,
            "repair_type": "parse_fail",
            "retraction_turn": None,
            "primary_llm": p_obj,
            "secondary_llm": None,
            "primary_retracted": None,
            "secondary_retracted": None,
            "primary_repair_type": None,
            "secondary_repair_type": None,
            "secondary_rationale": "",
            "dual_retracted_agreed": False,
            "dual_type_agreed": False,
            "repair_status": "PARSE_FAIL",
            "primary_failed": True,
            "secondary_failed": False,
            "rationale": "parse_fail",
        }

    if primary_failed or secondary_failed:
        return {
            "retracted": None,
            "retracted_L": 0,
            "retracted_U": 1,
            "repair_type": "parse_fail",
            "retraction_turn": None,
            "primary_llm": p_obj,
            "secondary_llm": s_obj,
            "primary_retracted": bool(p_obj["retracted"]) if p_ok and p_obj is not None else None,
            "secondary_retracted": bool(s_obj["retracted"]) if s_ok and s_obj is not None else None,
            "primary_repair_type": str(p_obj["repair_type"]) if p_ok and p_obj is not None else None,
            "secondary_repair_type": str(s_obj["repair_type"]) if s_ok and s_obj is not None else None,
            "rationale": p_obj.get("rationale", "") if isinstance(p_obj, dict) else "parse_fail",
            "secondary_rationale": s_obj.get("rationale", "") if isinstance(s_obj, dict) else "parse_fail",
            "dual_retracted_agreed": False,
            "dual_type_agreed": False,
            "repair_status": "PARSE_FAIL",
            "primary_failed": primary_failed,
            "secondary_failed": secondary_failed,
        }

    assert p_obj is not None and s_obj is not None
    p_ret = bool(p_obj["retracted"])
    s_ret = bool(s_obj["retracted"])
    p_type = str(p_obj["repair_type"])
    s_type = str(s_obj["repair_type"])
    ret_l = min(1 if p_ret else 0, 1 if s_ret else 0)
    ret_u = max(1 if p_ret else 0, 1 if s_ret else 0)
    ret_agreed = (p_ret == s_ret)
    type_agreed = (p_type == s_type)
    if type_agreed:
        comb_type = p_type
    elif ret_l == 1 and ret_u == 1:
        comb_type = "Retracted_TypeDisagreed"
    else:
        comb_type = "Unresolved"
    ret_turn = p_obj.get("retraction_turn") if p_ret else (s_obj.get("retraction_turn") if s_ret else None)
    return {
        "retracted": p_ret if ret_agreed else None,
        "retracted_L": ret_l,
        "retracted_U": ret_u,
        "repair_type": comb_type,
        "retraction_turn": ret_turn,
        "rationale": p_obj.get("rationale", ""),
        "secondary_rationale": s_obj.get("rationale", ""),
        "primary_llm": p_obj,
        "secondary_llm": s_obj,
        "primary_retracted": p_ret,
        "secondary_retracted": s_ret,
        "primary_repair_type": p_type,
        "secondary_repair_type": s_type,
        "dual_retracted_agreed": ret_agreed,
        "dual_type_agreed": type_agreed,
        "repair_status": "OK",
        "primary_failed": False,
        "secondary_failed": False,
    }


def judge_trajectories(
    rollouts: list[dict[str, Any]],
    item_map: dict[str, dict[str, Any]],
    cert_map: dict[str, dict[str, Any]] | None = None,
    primary_client: Any = None,
    primary_model: str = "codex/gpt-6.1-sol",
    primary_caps: Any = None,
    secondary_client: Any | None = None,
    secondary_model: str | None = None,
    secondary_caps: Any | None = None,
    workers: int = 8,
    dual_nonfinal_limit: int | None = None,
    secondary_workers: int | None = None,
    max_tokens: int = 16384,
    cache_path: Path | None = None,
    resume: bool = False,
    skip_repair: bool = False,
) -> list[dict[str, Any]]:
    """Annotate all trajectories with pure Dual-LLM Y, C(z_t), E, J bounds, and optional Repair."""
    del cert_map  # Part 2 uses pure Dual-LLM final arbitration (no SymPy/certificate target lookup).
    p_workers = max(1, int(workers))
    s_workers = max(1, int(secondary_workers if secondary_workers is not None else workers))
    has_secondary = secondary_client is not None and bool(secondary_model)

    cache_map: dict[str, dict[str, Any]] = {}
    cache_lock = threading.Lock()
    cache_fh = None
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        if resume:
            cache_map = _load_judge_cache(cache_path)
            if cache_map:
                print(f"[Cache] Loaded {len(cache_map)} completed judge calls from {cache_path}", flush=True)
            cache_fh = cache_path.open("ab")
        else:
            cache_fh = cache_path.open("wb")

    try:
        # Step 1: Mechanical 3-way truncation/empty check on every final turn; send ALL non-empty to Dual-LLM.
        final_trunc_map: dict[str, dict[str, bool]] = {}
        final_llm_needed: list[dict[str, Any]] = []
        n_empty_trunc = 0
        n_vis_trunc = 0
        n_non_trunc_empty = 0
        for tr in rollouts:
            last_turn = tr["turns"][-1] if tr["turns"] else {
                "visible_reply": tr.get("final_visible_reply", ""),
                "finish_reason": None,
                "empty_truncated": False,
            }
            if "visible_reply" not in last_turn:
                last_turn = {**last_turn, "visible_reply": tr.get("final_visible_reply", "")}
            tflags = classify_turn_truncation(last_turn)
            final_trunc_map[tr["trajectory_id"]] = tflags
            if tflags["empty_trunc"]:
                n_empty_trunc += 1
            elif tflags["non_trunc_empty"]:
                n_non_trunc_empty += 1
            else:
                if tflags["vis_trunc"]:
                    n_vis_trunc += 1
                final_llm_needed.append(tr)

        print(
            f"[FinalPrep] {len(rollouts)} trajectories: llm_needed={len(final_llm_needed)} "
            f"(vis_trunc={n_vis_trunc}), empty_trunc={n_empty_trunc}, non_trunc_empty={n_non_trunc_empty}",
            flush=True,
        )

        # Step 2: 100% Pure Dual-LLM final arbitration (primary & secondary in parallel).
        final_tasks = [
            (tr["trajectory_id"], build_final_arbiter_messages(item_map[tr["item_id"]], tr))
            for tr in final_llm_needed
        ]
        final_llm_primary: dict[Any, dict[str, Any] | None] = {}
        final_llm_secondary: dict[Any, dict[str, Any] | None] = {}
        if final_tasks and primary_client is not None:
            final_llm_primary, final_llm_secondary = _run_dual_branches(
                lambda: _run_parallel_json_tasks(
                    final_tasks, primary_client, primary_model, primary_caps,
                    stage="final", role="primary", workers=p_workers, max_tokens=max_tokens,
                    cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                ),
                (
                    lambda: _run_parallel_json_tasks(
                        final_tasks, secondary_client, secondary_model or "", secondary_caps,
                        stage="final", role="secondary", workers=s_workers, max_tokens=max_tokens,
                        cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                    )
                ) if has_secondary else None,
            )

        # Step 3: Blind 5-class Scope grading on all non-empty dynamic nonfinal turns.
        scope_tasks: list[tuple[tuple[str, int], list[dict[str, str]]]] = []
        nonfinal_trunc_map: dict[tuple[str, int], dict[str, bool]] = {}
        for tr in rollouts:
            if tr["protocol"] in ("FULL", "CONCAT", "CONCAT_MATCHED"):
                continue
            item = item_map[tr["item_id"]]
            for t in tr["turns"]:
                if t["is_final_turn"]:
                    continue
                t_idx = int(t["turn_idx"])
                key = (tr["trajectory_id"], t_idx)
                tflags = classify_turn_truncation(t)
                nonfinal_trunc_map[key] = tflags
                if tflags["empty_trunc"] or tflags["non_trunc_empty"]:
                    continue
                scope_tasks.append((key, build_blind_scope_messages(item, tr, t_idx)))

        scope_primary: dict[Any, dict[str, Any] | None] = {}
        scope_secondary: dict[Any, dict[str, Any] | None] = {}
        secondary_scope_keys: set[tuple[str, int]] = set()
        if scope_tasks and primary_client is not None:
            sub_tasks = scope_tasks[:dual_nonfinal_limit] if dual_nonfinal_limit else scope_tasks
            if has_secondary:
                secondary_scope_keys = {k for k, _ in sub_tasks}
            scope_primary, scope_secondary = _run_dual_branches(
                lambda: _run_parallel_json_tasks(
                    scope_tasks, primary_client, primary_model, primary_caps,
                    stage="scope", role="primary", workers=p_workers, max_tokens=max_tokens,
                    cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                ),
                (
                    lambda: _run_parallel_json_tasks(
                        sub_tasks, secondary_client, secondary_model or "", secondary_caps,
                        stage="scope", role="secondary", workers=s_workers, max_tokens=max_tokens,
                        cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                    )
                ) if has_secondary else None,
            )

        # Step 4: Assemble per-turn C labels, trajectory E/J bounds, and identify E_U==1 for optional Repair.
        repair_candidates: list[tuple[dict[str, Any], list[int]]] = []
        annotated: list[dict[str, Any]] = []

        for tr in rollouts:
            rec = json.loads(json.dumps(tr))
            tid = rec["trajectory_id"]
            rec["grading_rule_version"] = GRADING_RULE_VERSION
            rec["repair_skipped"] = bool(skip_repair)

            tflags_final = final_trunc_map[tid]
            p_f_obj = final_llm_primary.get(tid)
            s_f_obj = final_llm_secondary.get(tid) if has_secondary else None
            fg_comb = _combine_dual_final(
                p_f_obj,
                s_f_obj,
                empty_trunc=tflags_final["empty_trunc"],
                non_trunc_empty=tflags_final["non_trunc_empty"],
                vis_trunc=tflags_final["vis_trunc"],
                secondary_expected=has_secondary,
            )
            rec["final_grading"] = {
                "grading_rule_version": GRADING_RULE_VERSION,
                "primary_llm": p_f_obj,
                "secondary_llm": s_f_obj,
                "empty_trunc": tflags_final["empty_trunc"],
                "vis_trunc": tflags_final["vis_trunc"],
                "non_trunc_empty": tflags_final["non_trunc_empty"],
                **fg_comb,
            }
            if rec["turns"]:
                rec["turns"][-1]["empty_trunc"] = tflags_final["empty_trunc"]
                rec["turns"][-1]["vis_trunc"] = tflags_final["vis_trunc"]
                rec["turns"][-1]["non_trunc_empty"] = tflags_final["non_trunc_empty"]

            y_l = fg_comb["y_L"]
            y_u = fg_comb["y_U"]

            if rec["protocol"] in ("FULL", "CONCAT", "CONCAT_MATCHED"):
                rec["any_empty_truncated"] = bool(tflags_final["empty_trunc"])
                rec["any_vis_truncated"] = bool(tflags_final["vis_trunc"])
                rec["e_L"] = 0
                rec["e_U"] = 0
                rec["structural_zero_E"] = True
                rec["j_L"] = y_l
                rec["j_U"] = y_u
                rec["repair"] = None
                annotated.append(rec)
                continue

            e_l_traj = 0
            e_u_traj = 0
            commit_turns: list[int] = []
            for t in rec["turns"]:
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
                p_lbl = str(p_obj["scope_label"]) if p_valid else "Unresolved"

                in_sec = key in secondary_scope_keys
                s_valid = isinstance(s_obj, dict) and s_obj.get("scope_label") in NONFINAL_SCOPES
                s_failed = bool(in_sec and not s_valid)
                if in_sec:
                    s_lbl: str | None = str(s_obj["scope_label"]) if s_valid else "Unresolved"
                else:
                    s_lbl = None

                c_lbl, t_el, t_eu, cat_agreed, commit_agreed = _combine_dual_scope(p_lbl, s_lbl)
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

            rec["any_empty_truncated"] = any(bool(t.get("empty_trunc")) for t in rec["turns"])
            rec["any_vis_truncated"] = any(bool(t.get("vis_trunc")) for t in rec["turns"])
            rec["e_L"] = e_l_traj
            rec["e_U"] = e_u_traj
            rec["structural_zero_E"] = False
            rec["j_L"] = y_l * (1 - e_u_traj)
            rec["j_U"] = y_u * (1 - e_l_traj)
            rec["repair"] = None
            if e_u_traj == 1:
                repair_candidates.append((rec, commit_turns))
            annotated.append(rec)

        # Step 5: Optional Experiment 3 Dual-LLM Explicit Repair annotation on E_U == 1 trajectories.
        if not skip_repair and repair_candidates and primary_client is not None:
            repair_tasks = [
                (rec["trajectory_id"], build_repair_messages(item_map[rec["item_id"]], rec, cturns))
                for rec, cturns in repair_candidates
            ]
            repair_primary, repair_secondary = _run_dual_branches(
                lambda: _run_parallel_json_tasks(
                    repair_tasks, primary_client, primary_model, primary_caps,
                    stage="repair", role="primary", workers=p_workers, max_tokens=max_tokens,
                    cache_map=cache_map, cache_fh=cache_fh, cache_lock=cache_lock,
                ),
                (
                    lambda: _run_parallel_json_tasks(
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
                rec["repair"] = _combine_dual_repair(p_rep, s_rep, secondary_expected=has_secondary)
    finally:
        if cache_fh is not None:
            cache_fh.close()

    return annotated


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure Dual-LLM judge rollouts for Y, 5-class Scope C, Dual-Judge Audit, & optional Repair.")
    parser.add_argument("--rollouts", type=Path, default=None)
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--split", choices=["DEV", "TEST"], default="TEST")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--cache-path", type=Path, default=None, help="Incremental JSONL judge cache (defaults to <output>_judge_cache.jsonl).")
    parser.add_argument("--resume", action="store_true", help="Reuse valid completed calls from --cache-path and retry failed/missing ones.")
    parser.add_argument("--skip-repair", action="store_true", help="Skip Experiment 3 Repair annotation (default for Exp 2 Part 2).")
    parser.add_argument("--base-url", type=str, default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:3425/v1"))
    parser.add_argument("--api-key", type=str, default=os.environ.get("OPENAI_API_KEY", ""))
    parser.add_argument("--primary-model", type=str, default="codex/gpt-6.1-sol")
    parser.add_argument("--secondary-model", type=str, default=None)
    parser.add_argument("--secondary-base-url", type=str, default=None)
    parser.add_argument("--secondary-api-key", type=str, default=None)
    parser.add_argument("--dual-nonfinal-limit", type=int, default=None)
    parser.add_argument("--reasoning-effort", type=str, default="low")
    parser.add_argument("--secondary-reasoning-effort", type=str, default=None)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--secondary-workers", type=int, default=None, help="Override thread pool size for secondary judge (defaults to --workers).")
    parser.add_argument("--max-tokens", type=int, default=16384, help="Max completion+reasoning tokens per judge call (default 16384).")
    args = parser.parse_args()

    from openai import OpenAI

    assert args.rollouts is not None and args.output is not None
    items_path = args.release_dir / ("frozen_dev_20.jsonl" if args.split == "DEV" else "frozen_test_100.jsonl")
    item_map = {x["item_id"]: x for x in read_jsonl(items_path)}
    rollouts = [r for r in read_jsonl(args.rollouts) if r["item_id"] in item_map]

    api_key = args.api_key or ("cache-replay-only" if args.resume else "")
    if not api_key:
        raise SystemExit("Missing API key: pass --api-key or set OPENAI_API_KEY (or pass --resume with a complete cache).")
    p_client = OpenAI(api_key=api_key, base_url=args.base_url, timeout=90.0)
    eff, dis = resolve_reasoning_effort(args.primary_model, args.reasoning_effort)
    p_caps = CapabilityState(eff, True, dis)
    s_client = None
    s_caps = None
    if args.secondary_model:
        s_client = OpenAI(api_key=args.secondary_api_key or api_key, base_url=args.secondary_base_url or args.base_url, timeout=90.0)
        s_eff_arg = args.secondary_reasoning_effort if args.secondary_reasoning_effort is not None else args.reasoning_effort
        seff, sdis = resolve_reasoning_effort(args.secondary_model, s_eff_arg)
        s_caps = CapabilityState(seff, True, sdis)

    cache_path = args.cache_path or args.output.with_name(args.output.stem + "_judge_cache.jsonl")
    t0 = time.time()
    judged = judge_trajectories(
        rollouts=rollouts,
        item_map=item_map,
        cert_map=None,
        primary_client=p_client,
        primary_model=args.primary_model,
        primary_caps=p_caps,
        secondary_client=s_client,
        secondary_model=args.secondary_model,
        secondary_caps=s_caps,
        workers=args.workers,
        dual_nonfinal_limit=args.dual_nonfinal_limit,
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
    print(f"Judged {len(judged)} trajectories -> {args.output} in {time.time()-t0:.1f}s")


DEFAULT_RELEASE_DIR = SCRIPT_DIR.parent / "exp1_codex_gpt61_v1" / "frozen_release_verified_120"

if __name__ == "__main__":
    main()
