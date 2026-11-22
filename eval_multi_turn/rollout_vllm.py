#!/usr/bin/env python3
"""Colab A100 vLLM multi-turn rollout engine for Experiments 2–4.

Features:
- Executes any subset of {FULL, CONCAT, CONCAT_MATCHED, DYN_FREE, DYN_REQUEST, DYN_REQUEST_GUARD}.
- Consumes a frozen paired disclosure_plan.jsonl (exact same pi_{i,r} & per-request seeds).
- Places shared TASK_PREAMBLE / SCOPE_GUARD_PREAMBLE inside the first user turn (NO system role).
- Turn-synchronized vLLM batching with incremental checkpointing & exact --resume support.
- Splits `<think>...</think>` / `[/THINK]` cleanly at the last closing tag so private CoT
  never enters subsequent assistant history or scope grading.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from protocol_utils import (  # noqa: E402
    MODEL_PRESETS,
    PROTOCOLS_ALL,
    PROTOCOLS_EXP2,
    build_disclosure_plan,
    build_turn_user_text,
    prompt_manifest,
    safe_history_assistant_text,
    split_thinking_response,
)

DEFAULT_RELEASE_DIR = (
    SCRIPT_DIR.parent / "exp1_codex_gpt61_v1" / "frozen_release_verified_120"
)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            s = line.strip()
            if s:
                rows.append(json.loads(s))
    return rows


def traj_key(item_id: str, protocol: str, rep: int) -> str:
    return f"{item_id}___{protocol}___r{rep}"


def _resolve_model_cfg(
    model_id: str,
    max_tokens: int | None,
    temperature: float | None,
    top_p: float | None,
    max_model_len: int | None = None,
) -> dict[str, Any]:
    cfg = dict(
        MODEL_PRESETS.get(
            model_id,
            {
                "short_name": model_id.split("/")[-1],
                "family": "custom",
                "scale_tier": "unknown",
                "dtype": "bfloat16",
                "temperature": 0.6,
                "top_p": 0.95,
                "top_k": -1,
                "min_p": 0.0,
                "max_tokens": 16384,
                "max_model_len": 32768,
                "max_num_seqs_40gb": 12,
                "enable_thinking": True,
                "tokenizer_mode": "auto",
                "system_prompt_file": None,
                "pure_text_mm_zero": False,
            },
        )
    )
    if max_tokens is not None:
        cfg["max_tokens"] = int(max_tokens)
    if max_model_len is not None:
        cfg["max_model_len"] = int(max_model_len)
    if temperature is not None:
        cfg["temperature"] = float(temperature)
    if top_p is not None:
        cfg["top_p"] = float(top_p)
    return cfg


def render_chat_prompt(
    tokenizer: Any,
    messages: list[dict[str, str]],
    model_id: str,
    enable_thinking: bool,
) -> str:
    """Render chat history with native tokenizer template without altering system/user structure."""
    kwargs: dict[str, Any] = {"tokenize": False, "add_generation_prompt": True}
    if "qwen3" in model_id.lower():
        kwargs["enable_thinking"] = enable_thinking
    try:
        return tokenizer.apply_chat_template(messages, **kwargs)
    except TypeError:
        kwargs.pop("enable_thinking", None)
        return tokenizer.apply_chat_template(messages, **kwargs)


class VLLMBackend:
    """Thin wrapper around vllm.LLM with 40GB/80GB VRAM auto-tuning & pinned HF revision loading."""

    def __init__(
        self,
        model_id: str,
        cfg: dict[str, Any],
        revision: str | None = None,
        allow_unpinned: bool = False,
        gpu_mem_util: float | None = None,
        limit: int | None = None,
    ) -> None:
        import torch
        import vllm
        from huggingface_hub import HfApi
        from vllm import LLM, SamplingParams

        self.SamplingParams = SamplingParams
        self.model_id = model_id
        self.cfg = cfg

        pinned_rev = revision or cfg.get("default_revision")
        if not pinned_rev:
            try:
                pinned_rev = HfApi().model_info(model_id).sha
            except Exception as exc:
                if not allow_unpinned:
                    raise RuntimeError(
                        f"Failed to resolve HuggingFace commit revision for {model_id!r} ({exc}). "
                        "Pass --revision <commit_sha> or --allow-unpinned."
                    ) from exc
                pinned_rev = None

        gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
        vram_gb = (
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2)
            if torch.cuda.is_available()
            else 0.0
        )
        # Candidate concurrency cap on 40GB A100; verify KV capacity & truncation rate in DEV 20.
        is_40gb_tier = 0.0 < vram_gb <= 48.0
        target_ctx_len = int(cfg.get("max_model_len", 32768))
        if gpu_mem_util is not None:
            util = gpu_mem_util
        elif is_40gb_tier and target_ctx_len > 32768:
            util = 0.94
        elif is_40gb_tier:
            util = 0.93
        else:
            util = 0.90

        llm_kwargs: dict[str, Any] = {
            "model": model_id,
            "dtype": cfg.get("dtype", "bfloat16"),
            "max_model_len": target_ctx_len,
            "gpu_memory_utilization": util,
            "trust_remote_code": True,
        }
        if pinned_rev:
            llm_kwargs["revision"] = pinned_rev
            llm_kwargs["tokenizer_revision"] = pinned_rev
        if is_40gb_tier and cfg.get("max_num_seqs_40gb"):
            cap_40gb = int(cfg["max_num_seqs_40gb"])
            if target_ctx_len > 32768:
                cap_40gb = min(cap_40gb, 8)
            llm_kwargs["max_num_seqs"] = cap_40gb
        if cfg.get("tokenizer_mode", "auto") != "auto":
            llm_kwargs["tokenizer_mode"] = cfg["tokenizer_mode"]
        if cfg.get("config_format"):
            llm_kwargs["config_format"] = cfg["config_format"]
        if cfg.get("load_format"):
            llm_kwargs["load_format"] = cfg["load_format"]
        if cfg.get("pure_text_mm_zero"):
            llm_kwargs["limit_mm_per_prompt"] = {"image": 0}

        print(f"Loading vLLM model={model_id} rev={pinned_rev} on {gpu_name} ({vram_gb} GiB) kwargs={llm_kwargs}", flush=True)
        self.llm = LLM(**llm_kwargs)
        self.tokenizer = self.llm.get_tokenizer()
        self.provenance = {
            "model_id": model_id,
            "hf_model_revision": pinned_rev,
            "hf_tokenizer_revision": pinned_rev,
            "vllm_version": getattr(vllm, "__version__", "unknown"),
            "torch_version": getattr(torch, "__version__", "unknown"),
            "cuda_version": getattr(torch.version, "cuda", None),
            "python_version": platform.python_version(),
            "gpu_name": gpu_name,
            "vram_gib": vram_gb,
            "llm_kwargs": llm_kwargs,
            "decoding_cfg": {
                "temperature": cfg["temperature"],
                "top_p": cfg["top_p"],
                "top_k": cfg["top_k"],
                "min_p": cfg["min_p"],
                "max_tokens": cfg["max_tokens"],
                "enable_thinking": cfg["enable_thinking"],
            },
        }

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Execute a synchronized batch of {messages, seed} requests with dynamic context-headroom guard."""
        prompts: list[str] = []
        sampling_list: list[Any] = []
        eff_budgets: list[tuple[int, bool]] = []
        max_ctx = int(self.cfg.get("max_model_len", 32768))
        target_max = int(self.cfg["max_tokens"])
        for req in requests:
            prompt_str = render_chat_prompt(
                self.tokenizer,
                req["messages"],
                self.model_id,
                enable_thinking=bool(self.cfg["enable_thinking"]),
            )
            prompts.append(prompt_str)
            p_len = len(self.tokenizer.encode(prompt_str, add_special_tokens=False))
            avail = max(1, max_ctx - p_len - 8)
            eff_max = min(target_max, avail)
            eff_budgets.append((eff_max, eff_max < target_max))
            sp_kwargs: dict[str, Any] = {
                "temperature": float(self.cfg["temperature"]),
                "top_p": float(self.cfg["top_p"]),
                "max_tokens": eff_max,
                "seed": int(req["seed"]),
            }
            if int(self.cfg.get("top_k", -1)) > 0:
                sp_kwargs["top_k"] = int(self.cfg["top_k"])
            if float(self.cfg.get("min_p", 0.0)) > 0.0:
                sp_kwargs["min_p"] = float(self.cfg["min_p"])
            sampling_list.append(self.SamplingParams(**sp_kwargs))

        outputs = self.llm.generate(prompts, sampling_list)
        results: list[dict[str, Any]] = []
        for prompt_str, out, (eff_max, clamped) in zip(prompts, outputs, eff_budgets):
            first = out.outputs[0]
            raw_text = first.text or ""
            finish_reason = getattr(first, "finish_reason", None)
            p_toks = len(out.prompt_token_ids) if getattr(out, "prompt_token_ids", None) else 0
            c_toks = len(first.token_ids) if getattr(first, "token_ids", None) else 0
            split = split_thinking_response(raw_text, prompt_text=prompt_str, finish_reason=finish_reason)
            results.append(
                {
                    "prompt_sha256": hashlib.sha256(prompt_str.encode("utf-8")).hexdigest(),
                    "raw_reply": raw_text,
                    "finish_reason": finish_reason,
                    "prompt_tokens": p_toks,
                    "completion_tokens": c_toks,
                    "effective_max_tokens": eff_max,
                    "budget_clamped": clamped,
                    **split,
                }
            )
        return results


class MockBackend:
    """Deterministic offline backend simulating thinking, truncation, and incremental turns."""

    def __init__(
        self,
        model_id: str,
        cfg: dict[str, Any],
        truncate_item_id: str | None = None,
        revision: str = "mock-rev-0001",
    ) -> None:
        self.model_id = model_id
        self.cfg = cfg
        self.truncate_item_id = truncate_item_id
        self.provenance = {
            "model_id": model_id,
            "hf_model_revision": revision,
            "hf_tokenizer_revision": revision,
            "vllm_version": "mock-vllm",
            "torch_version": "mock-torch",
            "cuda_version": None,
            "python_version": platform.python_version(),
            "gpu_name": "MOCK_GPU",
            "vram_gib": 40.0,
            "decoding_cfg": dict(cfg),
        }

    def generate_batch(self, requests: list[dict[str, Any]]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for req in requests:
            iid = req["item_id"]
            proto = req["protocol"]
            turn_idx = req["turn_idx"]
            k = req["canonical_k"]
            seed = req["seed"]
            prompt_str = json.dumps(req["messages"], ensure_ascii=False) + "\n<think>\n"
            if self.truncate_item_id and iid == self.truncate_item_id and turn_idx == 1 and proto == "DYN_REQUEST":
                raw = "我正在推导第一步但是还没有想完就被预算打断了"
                finish = "length"
            elif proto in ("FULL", "CONCAT", "CONCAT_MATCHED") or turn_idx == k:
                ans = req.get("mock_final_answer") or r"\frac{\sqrt{5}}{5}"
                raw = f"完整条件已核对(seed={seed})。</think>由题意解得最终答案为 $\\boxed{{{ans}}}$。"
                finish = "stop"
            elif proto == "DYN_REQUEST":
                raw = f"当前条件不足(turn={turn_idx})。</think>在补充条件前无法唯一确定，若暂取 $x=1$ 则当前值为 $\\boxed{{1}}$。"
                finish = "stop"
            elif proto == "DYN_REQUEST_GUARD":
                raw = f"自检发现缺少关键条件(turn={turn_idx})。</think>当前缺少剩余约束，请补充全部条件后再定最终答案。"
                finish = "stop"
            else:
                raw = f"等待补充(turn={turn_idx})。</think>目前条件不足以唯一确定目标，请继续给出下一条件。"
                finish = "stop"
            split = split_thinking_response(raw, prompt_text=prompt_str, finish_reason=finish)
            out.append(
                {
                    "prompt_sha256": hashlib.sha256(prompt_str.encode("utf-8")).hexdigest(),
                    "raw_reply": raw,
                    "finish_reason": finish,
                    "prompt_tokens": 120 + 25 * turn_idx,
                    "completion_tokens": 45 if finish == "stop" else int(self.cfg["max_tokens"]),
                    "effective_max_tokens": int(self.cfg["max_tokens"]),
                    "budget_clamped": False,
                    **split,
                }
            )
        return out


def _build_initial_trajectory(item: dict[str, Any], plan_row: dict[str, Any], protocol: str, model_id: str) -> dict[str, Any]:
    return {
        "trajectory_id": traj_key(item["item_id"], protocol, plan_row["rep"]),
        "item_id": item["item_id"],
        "split": item["split"],
        "bucket": item["bucket"],
        "macro_subfield": item["macro_subfield"],
        "verified_mechanism": item["verified_mechanism"],
        "canonical_k": int(item["canonical_k"]),
        "protocol": protocol,
        "rep": int(plan_row["rep"]),
        "order_kind": plan_row["order_kind"],
        "disclosure_order": plan_row["disclosure_order"],
        "request_seeds": plan_row["request_seeds"][protocol],
        "model_id": model_id,
        "completed": False,
        "turns": [],
        "final_visible_reply": "",
        "total_prompt_tokens": 0,
        "total_completion_tokens": 0,
        "any_empty_truncated": False,
    }


def _write_outputs(
    output_path: Path,
    meta_path: Path,
    trajectories: dict[str, dict[str, Any]],
    ordered_keys: list[str],
    meta: dict[str, Any],
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_out = output_path.with_suffix(".jsonl.tmp")
    with tmp_out.open("wb") as fh:
        for k in ordered_keys:
            row = trajectories[k]
            fh.write((json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8"))
    tmp_out.replace(output_path)

    meta_path.write_text(json.dumps(meta, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _canonical_sha256(obj: Any) -> str:
    payload = json.dumps(obj, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def build_run_signature(
    model_id: str,
    provenance: dict[str, Any],
    cfg: dict[str, Any],
    items: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    empty_history_mode: str,
) -> dict[str, Any]:
    """Build exact configuration & input fingerprint enforced across --resume calls."""
    item_ids = {x["item_id"] for x in items}
    active_plan = [p for p in plan_rows if p["item_id"] in item_ids]
    return {
        "model_id": model_id,
        "hf_model_revision": provenance.get("hf_model_revision"),
        "hf_tokenizer_revision": provenance.get("hf_tokenizer_revision"),
        "splits": sorted({str(x.get("split")) for x in items}),
        "items_sha256": _canonical_sha256(
            [
                {
                    "item_id": x["item_id"],
                    "split": x.get("split"),
                    "canonical_k": x.get("canonical_k"),
                    "question": x.get("question"),
                    "final_answer": x.get("final_answer"),
                    "shards": x.get("shards"),
                }
                for x in items
            ]
        ),
        "plan_sha256": _canonical_sha256(
            [
                {
                    "item_id": p["item_id"],
                    "rep": int(p["rep"]),
                    "disclosure_order": p["disclosure_order"],
                    "request_seeds": p["request_seeds"],
                }
                for p in active_plan
            ]
        ),
        "prompts_sha256": prompt_manifest()["sha256"],
        "empty_history_mode": empty_history_mode,
        "decoding_cfg": {
            "temperature": cfg["temperature"],
            "top_p": cfg["top_p"],
            "top_k": cfg["top_k"],
            "min_p": cfg["min_p"],
            "max_tokens": int(cfg["max_tokens"]),
            "enable_thinking": bool(cfg["enable_thinking"]),
            "max_model_len": int(cfg.get("max_model_len", 32768)),
        },
    }


def _resolve_resume_revision(meta_path: Path | None, cli_revision: str | None) -> str | None:
    """Reuse recorded HF revision on --resume instead of resolving a potentially shifted upstream HEAD."""
    if not meta_path or not meta_path.exists():
        return cli_revision
    prior = json.loads(meta_path.read_text(encoding="utf-8"))
    sig = prior.get("run_signature")
    if not isinstance(sig, dict):
        raise ValueError(f"Refusing to --resume: {meta_path} lacks run_signature metadata.")
    rec_rev = sig.get("hf_model_revision")
    if cli_revision and rec_rev and cli_revision != rec_rev:
        raise ValueError(f"Refusing to --resume: CLI --revision={cli_revision!r} != recorded revision={rec_rev!r}.")
    return cli_revision or rec_rev


def _validate_resume_compatibility(
    output_path: Path,
    meta_path: Path,
    expected_sig: dict[str, Any],
    existing_rows: dict[str, dict[str, Any]],
) -> None:
    if not meta_path.exists():
        raise ValueError(
            f"Refusing to --resume existing output {output_path} without metadata sidecar {meta_path}; "
            "cannot verify model/tokenizer revision, decoding budget, or input hashes."
        )
    prior_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    prior_sig = prior_meta.get("run_signature")
    if not isinstance(prior_sig, dict):
        raise ValueError(f"Refusing to --resume: {meta_path} lacks run_signature.")
    mismatches: list[str] = []
    for k, exp_v in expected_sig.items():
        if prior_sig.get(k) != exp_v:
            mismatches.append(f"{k}: existing={prior_sig.get(k)!r} != new={exp_v!r}")
    if mismatches:
        raise ValueError(
            f"Refusing to --resume {output_path} due to configuration/input mismatch: {'; '.join(mismatches)}. "
            "Use a separate --output path for different budgets, models, prompts, or datasets."
        )
    for tk, row in existing_rows.items():
        if row.get("model_id") != expected_sig["model_id"]:
            raise ValueError(f"Refusing to --resume: trajectory {tk} has model_id={row.get('model_id')!r} != {expected_sig['model_id']!r}.")


def run_rollout(
    items: list[dict[str, Any]],
    plan_rows: list[dict[str, Any]],
    protocols: tuple[str, ...],
    model_id: str,
    cfg: dict[str, Any],
    backend: Any,
    output_path: Path,
    meta_path: Path,
    resume: bool = False,
    empty_history_mode: str = "empty",
) -> dict[str, Any]:
    item_map = {x["item_id"]: x for x in items}
    run_signature = build_run_signature(model_id, backend.provenance, cfg, items, plan_rows, empty_history_mode)

    ordered_keys: list[str] = []
    trajectories: dict[str, dict[str, Any]] = {}
    existing: dict[str, dict[str, Any]] = {}
    if resume and (output_path.exists() or meta_path.exists()):
        if output_path.exists():
            for row in read_jsonl(output_path):
                existing[row["trajectory_id"]] = row
        _validate_resume_compatibility(output_path, meta_path, run_signature, existing)

    for p in plan_rows:
        iid = p["item_id"]
        if iid not in item_map:
            continue
        rep = int(p["rep"])
        for proto in protocols:
            tk = traj_key(iid, proto, rep)
            ordered_keys.append(tk)
            if tk in existing:
                trajectories[tk] = existing[tk]
            else:
                trajectories[tk] = _build_initial_trajectory(item_map[iid], p, proto, model_id)
    for tk, row in existing.items():
        if tk not in trajectories:
            ordered_keys.append(tk)
            trajectories[tk] = row

    max_k = max(int(item_map[p["item_id"]]["canonical_k"]) for p in plan_rows if p["item_id"] in item_map)
    t_start = time.time()

    # Run turn-by-turn across turns 1 .. max_k (single-turn protocols only execute turn 1).
    for turn_idx in range(1, max_k + 1):
        pending_keys: list[str] = []
        batch_reqs: list[dict[str, Any]] = []
        for tk in ordered_keys:
            tr = trajectories[tk]
            if tr["completed"]:
                continue
            target_turns = 1 if tr["protocol"] in ("FULL", "CONCAT", "CONCAT_MATCHED") else tr["canonical_k"]
            done_turns = len(tr["turns"])
            if done_turns >= target_turns:
                tr["completed"] = True
                continue
            if done_turns + 1 != turn_idx:
                continue

            item = item_map[tr["item_id"]]
            user_msg = build_turn_user_text(item, tr["protocol"], tr["disclosure_order"], turn_idx)
            messages: list[dict[str, str]] = []
            for prev_turn in tr["turns"]:
                messages.append({"role": "user", "content": prev_turn["user_message"]})
                messages.append({"role": "assistant", "content": prev_turn["history_assistant_text"]})
            messages.append({"role": "user", "content": user_msg})

            pending_keys.append(tk)
            batch_reqs.append(
                {
                    "trajectory_id": tk,
                    "item_id": tr["item_id"],
                    "protocol": tr["protocol"],
                    "turn_idx": turn_idx,
                    "canonical_k": tr["canonical_k"],
                    "seed": tr["request_seeds"][turn_idx - 1],
                    "messages": messages,
                    "mock_final_answer": item.get("final_answer", ""),
                }
            )

        if not batch_reqs:
            continue

        t0 = time.time()
        print(f"[Turn {turn_idx}/{max_k}] Generating {len(batch_reqs)} requests...", flush=True)
        replies = backend.generate_batch(batch_reqs)
        for tk, req, rep_data in zip(pending_keys, batch_reqs, replies):
            tr = trajectories[tk]
            target_turns = 1 if tr["protocol"] in ("FULL", "CONCAT", "CONCAT_MATCHED") else tr["canonical_k"]
            hist_text = safe_history_assistant_text(rep_data, empty_history_mode=empty_history_mode)
            turn_rec = {
                "turn_idx": turn_idx,
                "disclosed_shard_id": None if tr["protocol"] in ("FULL", "CONCAT", "CONCAT_MATCHED") else tr["disclosure_order"][turn_idx - 1],
                "is_final_turn": turn_idx == target_turns,
                "seed": req["seed"],
                "user_message": req["messages"][-1]["content"],
                "history_assistant_text": hist_text,
                **rep_data,
            }
            tr["turns"].append(turn_rec)
            tr["total_prompt_tokens"] += int(rep_data["prompt_tokens"])
            tr["total_completion_tokens"] += int(rep_data["completion_tokens"])
            if rep_data["empty_truncated"]:
                tr["any_empty_truncated"] = True
            if turn_idx == target_turns:
                tr["completed"] = True
                tr["final_visible_reply"] = rep_data["visible_reply"]

        summary = _build_run_summary(trajectories, ordered_keys, backend.provenance, run_signature, time.time() - t_start)
        _write_outputs(output_path, meta_path, trajectories, ordered_keys, summary)
        print(f"[Turn {turn_idx}/{max_k}] Saved checkpoint ({time.time()-t0:.1f}s).", flush=True)

    summary = _build_run_summary(trajectories, ordered_keys, backend.provenance, run_signature, time.time() - t_start)
    _write_outputs(output_path, meta_path, trajectories, ordered_keys, summary)
    return summary


def _build_run_summary(
    trajectories: dict[str, dict[str, Any]],
    ordered_keys: list[str],
    provenance: dict[str, Any],
    run_signature: dict[str, Any],
    elapsed_sec: float,
) -> dict[str, Any]:
    proto_counts = Counter()
    completed_counts = Counter()
    empty_trunc_turns = Counter()
    clamped_turns = Counter()
    total_turns = Counter()
    for tk in ordered_keys:
        tr = trajectories[tk]
        p = tr["protocol"]
        proto_counts[p] += 1
        if tr["completed"]:
            completed_counts[p] += 1
        for t in tr["turns"]:
            total_turns[p] += 1
            if t.get("empty_truncated"):
                empty_trunc_turns[p] += 1
            if t.get("budget_clamped"):
                clamped_turns[p] += 1
    return {
        "run_signature": run_signature,
        "provenance": provenance,
        "prompt_manifest": prompt_manifest()["sha256"],
        "total_trajectories": len(ordered_keys),
        "completed_trajectories": sum(completed_counts.values()),
        "by_protocol": {
            p: {
                "trajectories": proto_counts[p],
                "completed": completed_counts[p],
                "turns": total_turns[p],
                "empty_truncated_turns": empty_trunc_turns[p],
                "budget_clamped_turns": clamped_turns[p],
            }
            for p in sorted(proto_counts)
        },
        "elapsed_sec": round(elapsed_sec, 2),
    }


def _run_mock_tests() -> None:
    import tempfile

    items = [
        {
            "item_id": "m-1",
            "split": "DEV",
            "bucket": "G1__TypeA",
            "macro_subfield": "G1",
            "verified_mechanism": "TypeA_UniqueValue",
            "canonical_k": 3,
            "question": "原题1",
            "final_answer": "5",
            "shards": [
                {"shard_id": "S1", "shard": "目标1。"},
                {"shard_id": "S2", "shard": "条件2。"},
                {"shard_id": "S3", "shard": "条件3。"},
            ],
        },
        {
            "item_id": "m-2",
            "split": "DEV",
            "bucket": "G2__TypeB",
            "macro_subfield": "G2",
            "verified_mechanism": "TypeB_Extremum",
            "canonical_k": 4,
            "question": "原题2",
            "final_answer": "[1, 2]",
            "shards": [
                {"shard_id": "S1", "shard": "目标2。"},
                {"shard_id": "S2", "shard": "条件A。"},
                {"shard_id": "S3", "shard": "条件B。"},
                {"shard_id": "S4", "shard": "条件C。"},
            ],
        },
    ]
    plan = build_disclosure_plan(items, seed_base=20261024, reps=3, protocols=PROTOCOLS_ALL)
    cfg = _resolve_model_cfg("Qwen/Qwen3-8B", max_tokens=8192, temperature=None, top_p=None)
    backend = MockBackend("Qwen/Qwen3-8B", cfg, truncate_item_id="m-2")

    with tempfile.TemporaryDirectory() as td:
        tdir = Path(td)
        full_out = tdir / "clean.jsonl"
        full_meta = tdir / "clean_meta.json"
        s1 = run_rollout(items, plan, PROTOCOLS_ALL, "Qwen/Qwen3-8B", cfg, backend, full_out, full_meta, resume=False, empty_history_mode="empty")
        assert s1["completed_trajectories"] == len(items) * 3 * len(PROTOCOLS_ALL) == 36
        assert s1["by_protocol"]["DYN_REQUEST"]["empty_truncated_turns"] == 3  # m-2 truncated at turn 1 x 3 reps
        rows_clean = read_jsonl(full_out)
        # Verify genuine empty history string by default (zero synthetic placeholder injection)
        m2_dyn = [r for r in rows_clean if r["item_id"] == "m-2" and r["protocol"] == "DYN_REQUEST"]
        assert all(r["turns"][0]["history_assistant_text"] == "" for r in m2_dyn)

        # Verify missing metadata sidecar fails closed on --resume.
        part_out = tdir / "part.jsonl"
        part_meta = tdir / "part_meta.json"
        partial = []
        for r in rows_clean[:12]:
            rc = json.loads(json.dumps(r))
            if len(rc["turns"]) > 1:
                rc["turns"] = rc["turns"][:1]
                rc["completed"] = False
                rc["final_visible_reply"] = ""
                rc["total_prompt_tokens"] = rc["turns"][0]["prompt_tokens"]
                rc["total_completion_tokens"] = rc["turns"][0]["completion_tokens"]
            partial.append(rc)
        part_out.write_text("".join(json.dumps(x, ensure_ascii=False) + "\n" for x in partial), encoding="utf-8")
        try:
            run_rollout(items, plan, PROTOCOLS_ALL, "Qwen/Qwen3-8B", cfg, backend, part_out, part_meta, resume=True)
            raise AssertionError("Expected ValueError when resuming without meta_path!")
        except ValueError:
            pass

        # Copy matching meta sidecar (simulating Turn-1 checkpoint), verify exact resume equivalence.
        clean_meta_obj = json.loads(full_meta.read_text(encoding="utf-8"))
        part_meta.write_text(json.dumps(clean_meta_obj, ensure_ascii=False, indent=2), encoding="utf-8")

        # Verify mismatched max_tokens (4096 vs 8192), empty_history_mode, revision, or items/plan fail closed & preserve file bytes.
        before_bytes = part_out.read_bytes()
        cfg_4096 = _resolve_model_cfg("Qwen/Qwen3-8B", max_tokens=4096, temperature=None, top_p=None)
        backend_4096 = MockBackend("Qwen/Qwen3-8B", cfg_4096, truncate_item_id="m-2")
        for bad_kwargs in (
            {"cfg": cfg_4096, "backend": backend_4096},
            {"empty_history_mode": "placeholder"},
            {"backend": MockBackend("Qwen/Qwen3-8B", cfg, truncate_item_id="m-2", revision="mock-rev-9999")},
            {"plan_rows": build_disclosure_plan(items, seed_base=999999, reps=3, protocols=PROTOCOLS_ALL)},
        ):
            call_kw = dict(
                items=items, plan_rows=plan, protocols=PROTOCOLS_ALL, model_id="Qwen/Qwen3-8B",
                cfg=cfg, backend=backend, output_path=part_out, meta_path=part_meta,
                resume=True, empty_history_mode="empty",
            )
            call_kw.update(bad_kwargs)
            try:
                run_rollout(**call_kw)
                raise AssertionError(f"Expected ValueError for mismatched resume kwargs={list(bad_kwargs.keys())}!")
            except ValueError:
                pass
            assert part_out.read_bytes() == before_bytes, "Failed --resume mutated existing output file!"

        # Verify _resolve_resume_revision locks back onto recorded revision when CLI --revision is omitted
        assert _resolve_resume_revision(part_meta, None) == "mock-rev-0001"
        try:
            _resolve_resume_revision(part_meta, "different-sha")
            raise AssertionError("Expected ValueError on conflicting CLI revision during resume!")
        except ValueError:
            pass

        run_rollout(items, plan, PROTOCOLS_ALL, "Qwen/Qwen3-8B", cfg, backend, part_out, part_meta, resume=True, empty_history_mode="empty")
        rows_resumed = read_jsonl(part_out)
        assert rows_clean == rows_resumed, "Resumed trajectories differed from single clean pass!"

    print("rollout_vllm.py mock tests PASSED (including --resume lock & exact equivalence).")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Experiments 2–4 multi-turn vLLM evaluation.")
    parser.add_argument("--input", type=Path, default=None, help="Override input JSONL (default picks split from frozen release).")
    parser.add_argument("--release-dir", type=Path, default=DEFAULT_RELEASE_DIR)
    parser.add_argument("--split", choices=["DEV", "TEST"], default="DEV")
    parser.add_argument("--plan", type=Path, default=None, help="Pre-frozen disclosure_plan JSONL.")
    parser.add_argument("--output", type=Path, required=False, default=None)
    parser.add_argument("--meta-output", type=Path, default=None)
    parser.add_argument("--model", type=str, default="Qwen/Qwen3-8B")
    parser.add_argument("--revision", type=str, default=None, help="Pin exact HuggingFace commit SHA for model & tokenizer.")
    parser.add_argument("--allow-unpinned", action="store_true", help="Allow loading without HF commit SHA if offline.")
    parser.add_argument(
        "--empty-history-mode",
        choices=["empty", "placeholder"],
        default="empty",
        help="Assistant history on empty_truncated turn: genuine '' (default) or explicit placeholder.",
    )
    parser.add_argument(
        "--protocols",
        nargs="+",
        default=list(PROTOCOLS_EXP2),
        choices=list(PROTOCOLS_ALL),
    )
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--seed-base", type=int, default=20261024)
    parser.add_argument("--max-tokens", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=None, help="Override context window length (default 32768 for Qwen3/GLM-Z1).")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top-p", type=float, default=None)
    parser.add_argument("--gpu-memory-utilization", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--mock-test", action="store_true")
    args = parser.parse_args()

    if args.mock_test:
        _run_mock_tests()
        return

    assert args.output is not None, "--output is required when not running --mock-test."
    in_path = args.input or (args.release_dir / ("frozen_dev_20.jsonl" if args.split == "DEV" else "frozen_test_100.jsonl"))
    items = read_jsonl(in_path)
    if args.limit:
        items = items[: args.limit]

    if args.plan and args.plan.exists():
        plan_rows = [p for p in read_jsonl(args.plan) if int(p["rep"]) < args.reps]
    else:
        plan_rows = build_disclosure_plan(items, seed_base=args.seed_base, reps=args.reps, protocols=tuple(PROTOCOLS_ALL))

    meta_path = args.meta_output or args.output.with_name(args.output.stem + "_meta.json")
    cfg = _resolve_model_cfg(args.model, args.max_tokens, args.temperature, args.top_p, max_model_len=args.max_model_len)
    pinned_revision = _resolve_resume_revision(meta_path, args.revision) if args.resume else args.revision
    backend = VLLMBackend(
        args.model,
        cfg,
        revision=pinned_revision,
        allow_unpinned=args.allow_unpinned,
        gpu_mem_util=args.gpu_memory_utilization,
        limit=args.limit,
    )
    summary = run_rollout(
        items=items,
        plan_rows=plan_rows,
        protocols=tuple(args.protocols),
        model_id=args.model,
        cfg=cfg,
        backend=backend,
        output_path=args.output,
        meta_path=meta_path,
        resume=args.resume,
        empty_history_mode=args.empty_history_mode,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
