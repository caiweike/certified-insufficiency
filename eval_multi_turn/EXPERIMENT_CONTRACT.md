# Experiments 2–4 Experimental Contract (v2.2 — Final Paper Release Spec)

## 1. Benchmark Inputs & Verified Scale
- Source release: `exp1_codex_gpt61_v1/frozen_release_verified_120/` (`stage2b_rule_version = stage2b-sympy-cert-v1.1.0`)
  - Calibration / Development set: `frozen_dev_20.jsonl` (`N = 20`; prompt/judge/budget calibration ONLY here)
  - Main Test set: `frozen_test_100.jsonl` (`N = 100`; `k in {3: 44, 4: 34, 5: 16, 6: 6}`, `sum(k) = 384`, `TypeA/TypeB = 68/32`, `cluster_id = null` for all 100 items)

## 2. Models & Native Prompt Formatting (A100-80GB BF16)
Three open-weight Chinese-strong reasoning models runnable natively in pure-text BF16 on a single A100 GPU, supporting both a within-family scale contrast (`Qwen3-8B -> Qwen3-14B`) and a near-matched-scale cross-family contrast (`Qwen3-8B` vs `GLM-Z1-9B-0414`):
1. `Qwen/Qwen3-8B` (revision `b968826d9c46dd6066d109eabc6255188de91218`, `enable_thinking=True`, BF16; decoding: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0`)
2. `Qwen/Qwen3-14B` (revision `40c069824f4251a91eefaf281ebe4c544efd3e18`, `enable_thinking=True`, BF16; decoding: `temperature=0.6, top_p=0.95, top_k=20, min_p=0.0`)
3. `zai-org/GLM-Z1-9B-0414` (revision `b221b06fefb23ca320922cf6e68ab5f2fb82de81`, Reasoning `<think>`, BF16; decoding: `temperature=0.6, top_p=0.95, top_k=20`)

Rules:
- **No `system` prompt**: Place the shared Chinese task instructions (`TASK_PREAMBLE`) at the top of the **first `user` message** identically across ALL 3 models, preserving each model's native HF chat template untouched. Cross-model comparisons compare complete released configurations (`GLM-Z1-9B` vs `Qwen3-8B` also reflects `9B` vs `8B` and training differences, not architecture alone).
- **Output & Context Budget**:
  - **Primary full-budget specification (`eval_multi_turn/runs/runs_16k/` and `runs_16k_v13_blind/`)**: `max_tokens = 16384`, `max_model_len = 32768` (supported natively without RoPE/YaRN scaling by all 3 models, adopted after DEV 20 calibration in `eval_multi_turn/runs/dev20/` showed asymmetric single-turn vs multi-turn right-censoring under `4096`/`8192`). Enforces per-turn remaining context guard (`effective_max_tokens`, `budget_clamped_turns == 0`) and exact pinned HF revisions.
  - Concurrency stability (`max_num_seqs`), `budget_clamped_turns == 0`, and `empty_truncated` rates are verified on `frozen_dev_20` before locking `TEST`.
- **Strict thinking separation & truncation history**: Split at the **last `</think>`** (or `[/THINK]` for optional backup models). Store `raw_reply`, `reasoning_trace`, `visible_reply`, `finish_reason`, `prompt_tokens`, `completion_tokens`, `thinking_truncated`, and `empty_truncated`. Pass ONLY `visible_reply` into turn `t+1` assistant history; when `visible_reply` is empty after truncation, default `--empty-history-mode empty` preserves empty `""` assistant content (avoiding artificial text intervention) and marks `EMPTY_TRUNCATED` (NEVER labeled `Wait`). Analyze `reasoning_trace` strictly as generated draft text, never as hidden internal belief.

## 3. Six-Protocol Design, Paired Order, & Call Budget
- Runs per item: `R = 3` (`rep in {0, 1, 2}`):
  - `rep = 0`: canonical order `(S1, S2, ..., Sk)` (independent sampling between `CONCAT` and `CONCAT_MATCHED` using distinct per-protocol seeds — differences reflect sampling variance, NOT order effect).
  - `rep = 1, 2`: forced non-canonical deterministic seeded permutations `pi_{ir}` of `S2..Sk` (`k=3` uses `[S1, S3, S2]` for both runs as two independent samples of the sole non-canonical order; `k >= 4` chooses two distinct non-canonical permutations for `rep=1` and `rep=2`), frozen offline in `disclosure_plan_test100.jsonl` / `disclosure_plan_dev20.jsonl` before inference.
- **Six protocols (`4,356` generation calls / `1,800` trajectories per model; `5,400` trajectories / `13,068` calls across 3 models) sharing `(item_id, rep)` & exact `FINAL_SUFFIX`**:
  1. `FULL`: single-turn original `question` + `FINAL_SUFFIX` (`100 * 3 = 300` calls/model)
  2. `CONCAT`: single-turn fixed canonical order `S1 (+) S2 (+) ... (+) Sk` + `FINAL_SUFFIX` across all 3 reps (`300` calls/model)
  3. `CONCAT_MATCHED`: single-turn paired order `S1 (+) S_{pi_{ir}(2)} (+) ... (+) S_{pi_{ir}(k)}` + `FINAL_SUFFIX` with independent request seeds even at `rep=0` (`300` calls/model)
  4. `DYN_FREE`: multi-turn progressive disclosure following exact `(S1, S_{pi_{ir}(2)}, ..., S_{pi_{ir}(k)})`, no nonfinal answer request (`384 * 3 = 1,152` calls/model)
  5. `DYN_REQUEST`: identical `disclosure_plan` `(S1, S_{pi_{ir}(2)}, ..., S_{pi_{ir}(k)})`, appending `NONFINAL_REQUEST_SUFFIX` to turns `1 .. k-1` and identical `FINAL_SUFFIX` on turn `k` (`384 * 3 = 1,152` calls/model)
  6. `DYN_REQUEST_GUARD` (`GUARD`): identical `disclosure_plan` and turn prompts as `DYN_REQUEST`, adding a zero-leak uniqueness/sufficiency verification clause in `TASK_PREAMBLE` (turn 1 only; sees strictly `S1..St`, no future shard info; `+1,152` calls/model).
- **Five Pre-Specified Contrast Pairs** *(descriptive pairwise contrasts without assuming mechanism independence or additivity)*:
  - `FULL - CONCAT` (`Y`): sharding wording & formatting effect (canonical single-turn baseline)
  - `CONCAT - CONCAT_MATCHED` (`Y`, overall and stratified on canonical `rep=0` vs paired non-canonical `rep in {1, 2}`): non-canonical permutation order effect
  - `CONCAT_MATCHED - DYN_FREE` (`Y`): overall progressive multi-turn effect under matched wording and condition order
  - `DYN_REQUEST - DYN_FREE` (`E`, `Conditional`, `Wait`, `Y`, `J`): intermediate demand effect under matched disclosure order
  - `DYN_REQUEST_GUARD - DYN_REQUEST` (`E`, `J`, `Y`, `Conditional`, `Wait`, `TypeA - TypeB` interaction): prompting-based verification guard effect

## 4. Sufficiency (`L`), 5-Class Scope (`C`), Events (`E`), & Joint Success (`J`)
- **Sufficiency `L(A_{icrt})` and `S1` definition**:
  - `S1` contains the target question, object declarations, and necessary background presuppositions, excludes subsequent conditions `S2..Sk`, and is strictly insufficient to determine the complete-task target answer.
  - Under verified Stage 2B LOO certificates:
    - `TypeA_UniqueValue`: Constraint monotonicity proves every strict prefix `t < k` (missing at least one `S_j`) admits non-uniqueness under the formalized constraints.
    - `TypeB_Extremum`: Removing `S_j` admits a `relaxed_witness` whose objective value strictly improves upon/escapes the complete-task optimum; thus we measure premature commitment relative to the complete original task.
  - Turn `t < k`: `L = Insufficient`; Turn `t = k`: `L = Sufficient` (zero `Unresolved-L`).
- **5-Class Visible Scope `C(z_{icrt})` for nonfinal turns `t < k`** (`7,668` total non-final dynamic turns; `7,655` non-empty, `13` `EMPTY_TRUNCATED`):
  - **Input-Blind Scope & Repair Judge (`judge_v13_input_blind.py`)**: Every user turn in the judge transcript is reconstructed strictly from the raw disclosed shard text `S1..S_{pi(t)}` (stripping `TASK_PREAMBLE`, `NONFINAL_REQUEST_SUFFIX`, `GUARD` instructions, `FINAL_SUFFIX`, item IDs, model IDs, protocol labels, and regime labels `TypeA`/`TypeB`). Residual protocol wording echoed inside assistant replies is `< 0.5%`.
  - Classes:
    1. `Wait` (`e=0`): requests clarification/missing constraints or explains underdetermination without proposing a target answer candidate (excludes truncated empties).
    2. `Conditional` (`e=0`): proposes candidate(s) explicitly restricted to stated unconfirmed assumptions or explicitly scoped only to the currently relaxed problem (including multi-branch parameterized case splits).
    3. `Partial/Other` (`e=0`): deterministic valid intermediate derivations/observations with neither waiting nor a complete-task target claim.
    4. `Commit` (`e=1`): unqualified complete-task target answer or a claim that the available information suffices to determine that target. Read the entire reply to establish whether a premise remains active.
    5. `Unresolved` (`[e_L, e_U] = [0, 1]`): genuinely contested **`Commit` vs non-`Commit`** boundary (or judge parse failure). Disagreements strictly among non-`Commit` classes keep `[e_L, e_U] = [0, 0]` as `NonCommit_Disagreed` (`dual_scope_agreed=False, dual_commit_agreed=True`).
- **Single-turn conventions**:
  - `E_FULL = E_CONCAT = E_CONCAT_MATCHED = 0` (structural zeros: no nonfinal turn opportunity to commit), `J = Y`.
- **Decomposition, Bounds, Confidence Intervals & Reporting Rules**:
  - `Y - J = Y * E` (proportion ending with correct final answer after an earlier premature commitment); reported alongside `P(E=1 | Y=1)`.
  - Trajectory bounds: `E_L = max_{t<k} e_{t,L}`, `E_U = max_{t<k} e_{t,U}`, `J_L = Y_L * (1 - E_U)`, `J_U = Y_U * (1 - E_L)`. Repair parse failures take `[retracted_L, retracted_U] = [0, 1]`.
  - Conditional ratios (`P(E=1 | Y=1)`, explicit/confirmed repair rates, per-turn discrete hazard `h(t) = P(first Commit at t | no Commit before t)`) use exact minimax optimization over unresolved `[0, 1]` entries.
  - Protocol contrasts: `[A_L - B_U, A_U - B_L]`, paired with 10,000 item-cluster bootstrap 95% CIs (`seed = 20261024`).
  - Multiplicity / significance convention: no multiplicity correction; protocol contrasts are called *significant* only when their envelope CI excludes zero; regime-stratified effects and `TypeA - TypeB` interactions are called *significant* when both `E_L` and `E_U` CIs exclude zero, and *suggestive* when only per-judge or single-bound CIs exclude zero.
  - Report `k=6` (`n=6`) descriptively alongside first-commit turn / conditional hazard analysis `h(t)` across depths `k=3..6`.

## 5. Dual-LLM Judging, SymPy Cross-Check, & Explicit Repair
- **Final-Answer Judging (`judge_trajectories.py`)**: Every non-empty final reply (`5,327` of `5,400` TEST trajectories; `73` empty/truncated) is graded independently by primary (`codex/gpt-6.1-sol`) and secondary (`workbuddy/deepseek-v4.1-flash`) judges (`temperature=0.0`, `reasoning_effort="low"`, `max_tokens=16384`), seeing ONLY `question`, `final_answer`, and `final_visible_reply`.
  - `FINAL_ARBITER_PROMPT` treats open vs closed finite continuous endpoints as equivalent for monotonic intervals of continuous functions, while enforcing strict endpoint inclusion/exclusion for parameter ranges and inequality solution sets.
  - Separates `final_empty_trunc`, `final_vis_trunc`, `final_non_trunc_empty`, and LLM-judged `final_explicit_abstain`.
- **Deterministic SymPy Cross-Check (`revision_analysis/verify_final_y_sympy.py` + `sympy_answer_grader.py`)**: Independently audits final-answer `Y` judgments offline on all `5,400` TEST trajectories (`3,177` decided deterministically; `99.94%` agreement with agreed dual-LLM verdicts).
- **Input-Blind Nonfinal Scope & Explicit Repair (`judge_v13_input_blind.py`)**:
  - Runs over all `7,655` non-empty non-final dynamic turns (`15,310` judge calls) and all `489` trajectories with `E_U = 1` (`978` repair judge calls) using the same two judge models (`codex/gpt-6.1-sol` and `workbuddy/deepseek-v4.1-flash`).
  - Explicit `Repair` evaluates whether a later turn explicitly retracts/revises an earlier premature commitment — NEVER inferred from `Y = 1`.
- **Reproducibility & Calibration**:
  - All `26,942` TEST judge responses (final answer, input-blind scope, input-blind repair) are archived in `eval_multi_turn/runs/runs_16k_v13_blind/` and verified offline by `eval_multi_turn/verify_cached_judgments.py` (prompt SHA-256 + exact label replay).
  - DEV 20 calibration thresholds before running TEST: `judge_failures == 0`, `dual_final_kappa >= 0.90`, pooled `dual_commit_kappa >= 0.85`.
  - Fail-closed `--resume` `run_signature` validation pins HF `revision` and `tokenizer_revision`, decoding configuration, `empty_history_mode`, `prompts_sha256`, `items_sha256`, and `plan_sha256`.
