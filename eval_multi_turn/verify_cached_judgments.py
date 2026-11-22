#!/usr/bin/env python3
"""Offline check of every LLM-judge label in the paper's TEST results against the recorded judge responses.

For each TEST trajectory it rebuilds the exact judge prompts from the released data and verifies, with no API calls, that
  (a) a recorded response exists for every final-answer, scope, and repair judgment, made on a prompt with the same SHA-256, and
  (b) the label stored in the judged files equals the recorded response.
Run `python unpack_data.py` first. Uses only the standard library: prompt builders are read from the judging scripts' source.
"""
import ast, collections, hashlib, json
from pathlib import Path
from typing import Any

EVAL = Path(__file__).resolve().parent
REPO = EVAL.parent
RUNS = EVAL / "runs" / "runs_16k_v13_blind"
ITEMS = REPO / "exp1_codex_gpt61_v1" / "frozen_release_verified_120" / "frozen_test_100.jsonl"
FILES = ["qwen3_8b", "qwen3_8b_guard", "glm_z1_9b_0414", "glm_z1_9b_0414_guard", "qwen3_14b", "qwen3_14b_guard"]
DYN = {"DYN_FREE", "DYN_REQUEST", "DYN_REQUEST_GUARD"}


def load_defs(path: Path, names: set, ns: dict) -> dict:
    src = path.read_text(encoding="utf-8")
    parts = [ast.get_source_segment(src, n) for n in ast.parse(src).body
             if (isinstance(n, ast.Assign) and any(getattr(t, "id", None) in names for t in n.targets))
             or (isinstance(n, ast.FunctionDef) and n.name in names)]
    exec("\n\n".join(parts), ns)
    return ns


class _JT:  # stands in for `import judge_trajectories as jt` inside judge_v13_input_blind.py
    pass


def load_cache(path: Path) -> dict:
    cache = {}
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            e = json.loads(line)
            cache[e["cache_key"]] = e
    return cache


def main() -> None:
    jt = load_defs(EVAL / "judge_trajectories.py",
                   {"SCOPE_PROMPT", "REPAIR_PROMPT", "FINAL_ARBITER_PROMPT", "_prompt_sha256", "build_final_arbiter_messages"},
                   {"json": json, "hashlib": hashlib, "Any": Any})
    _JT.SCOPE_PROMPT, _JT.REPAIR_PROMPT = jt["SCOPE_PROMPT"], jt["REPAIR_PROMPT"]
    v13 = load_defs(EVAL / "judge_v13_input_blind.py",
                    {"SCOPE_PROMPT_V13", "_shard_text_for_turn", "build_v13_scope_messages", "build_v13_repair_messages"},
                    {"json": json, "Any": Any, "jt": _JT})
    sha = jt["_prompt_sha256"]
    items = {}
    with ITEMS.open(encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            items[r["item_id"]] = r
    tot = collections.Counter()
    for m in FILES:
        cache = load_cache(RUNS / f"judged_test100_{m}_16384_v13_blind_judge_cache.jsonl")
        cache.update(load_cache(RUNS / f"final_answer_cache_test100_{m}.jsonl"))
        c = collections.Counter()

        def check(stage, key, msgs, field, expected):
            e = cache.get(f"{stage}::{key}")
            c[f"{stage}_total"] += 1
            if e is not None and e.get("prompt_sha256") == sha(msgs):
                c[f"{stage}_hit"] += 1
                c[f"{stage}_match"] += (e.get("obj") or {}).get(field) == expected

        with (RUNS / f"judged_test100_{m}_16384_v13_blind.jsonl").open(encoding="utf-8") as fh:
            for line in fh:
                r = json.loads(line)
                it, tid, fg = items[r["item_id"]], r["trajectory_id"], r["final_grading"]
                fmsgs = jt["build_final_arbiter_messages"](it, r)
                for role, k in (("primary", "primary_llm"), ("secondary", "secondary_llm")):
                    if fg.get(k):  # absent when the reply was empty/truncated (graded without an LLM)
                        check("final", f"{role}::{tid}", fmsgs, "correct", fg[k].get("correct"))
                if r["protocol"] not in DYN:
                    continue
                commit_turns = []
                for t in r["turns"]:
                    if t["is_final_turn"] or t.get("primary_scope") is None:
                        continue
                    ti = int(t["turn_idx"])
                    smsgs = v13["build_v13_scope_messages"](it, r, ti)
                    key = json.dumps([tid, ti], ensure_ascii=False)
                    check("scope", f"primary::{key}", smsgs, "scope_label", t["primary_scope"])
                    check("scope", f"secondary::{key}", smsgs, "scope_label", t["secondary_scope"])
                    if t.get("e_U") == 1:
                        commit_turns.append(ti)
                if r.get("repair"):
                    rmsgs = v13["build_v13_repair_messages"](it, r, commit_turns)
                    check("repair", f"primary::{tid}", rmsgs, "retracted", r["repair"]["primary_retracted"])
                    check("repair", f"secondary::{tid}", rmsgs, "retracted", r["repair"]["secondary_retracted"])
        print(m, dict(c))
        tot.update(c)
    ok = all(tot[f"{s}_total"] == tot[f"{s}_hit"] == tot[f"{s}_match"] for s in ("final", "scope", "repair"))
    print("TOTAL", dict(tot))
    print("ALL JUDGE LABELS REPRODUCED FROM RECORDED RESPONSES" if ok else "MISMATCH: see counts above")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()
