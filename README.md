# Certified Insufficiency and Premature Commitment

Code, benchmark, and judged trajectories for the IEEE CSCWD paper
*"Certified Insufficiency and Premature Commitment: Process–Outcome Evaluation of Reasoning LLMs under Progressive Information Disclosure"*
(Weike Cai, Mohan Tang).

## Contents

| Path | What it is |
|---|---|
| `exp1_codex_gpt61_v1/frozen_release_verified_120/` | **The benchmark used in the paper**: `frozen_test_100.jsonl`, `frozen_dev_20.jsonl`, SymPy certificates, manifest + SHA-256, Stage-2 ledger |
| `exp1_codex_gpt61_v1/stage2b_verified_all469.jsonl` | Complete Stage 2B candidate pool (bit-reproduces `frozen_release_verified_120/` via `stage2b_verification/run_stage2b_freeze.py`) |
| `benchmark_spec_v2_4/`, `colab_stage2_sharding/`, `stage2b_verification/`, `run_exp1_pipeline.py` | Benchmark construction: Stage-1 filtering, clause sharding, certificate generation and offline verification |
| `eval_multi_turn/protocol_utils.py`, `disclosure_plan_*.jsonl` | The six protocols and the paired disclosure orders |
| `eval_multi_turn/rollout_vllm.py`, `Colab_*.ipynb` | Multi-turn rollouts (vLLM, A100-80GB) |
| `eval_multi_turn/judge_trajectories.py` | Final-answer judging: correctness (Y) and abstention |
| `eval_multi_turn/judge_v13_input_blind.py` | Input-blind judging of non-final scope and explicit repair |
| `eval_multi_turn/sympy_answer_grader.py`, `revision_analysis/` | Deterministic SymPy cross-check of final answers |
| `eval_multi_turn/runs/` | All judged trajectories (see *Data*) |
| `eval_multi_turn/analyze_experiments.py`, `revision_analysis_v13/` | Statistics; `extract_tables.py` builds the flat tables the tables and figures are computed from |
| `eval_multi_turn/verify_cached_judgments.py` | Offline check of every judge label against the recorded judge responses |
| `eval_multi_turn/paper_revised/` | Paper source, tables, figures, and `regenerate_v13_assets.py` |
| `MANIFEST.sha256` | SHA-256 checksums for every tracked file in this repository |

## Setup

```bash
python -m pip install -r requirements.txt   # analysis / judging / verification
python unpack_data.py                       # decompresses every runs/*.jsonl.gz next to the archive
```
Rollouts additionally need a GPU with `vllm==0.34.0` and `torch==2.14.0` (CUDA 13.0); the paper's runs used Python 3.13 on A100-80GB.

## Data

`eval_multi_turn/runs/` (all run files ship as `.jsonl.gz`; run `unpack_data.py` first):

| Folder | Content |
|---|---|
| `runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl.gz` | **Paper results.** 5,400 TEST trajectories (3 models × 6 protocols × 100 items × 3 reps; GUARD in the `*_guard_*` files) with full model outputs, final-answer labels, input-blind scope labels, and repair labels |
| `runs_16k_v13_blind/judged_dev20_*` | DEV 20 calibration judging |
| `runs_16k_v13_blind/*_judge_cache.jsonl.gz` | Recorded scope and repair judge responses, keyed by prompt SHA-256 |
| `runs_16k_v13_blind/final_answer_cache_test100_*.jsonl.gz` | Recorded final-answer judge responses (source of Y) |
| `runs_16k/` | DEV 20 rollouts (16K budget) and provenance (`*_meta.json`: model revision, decoding config, library versions, GPU) for every TEST rollout |
| `dev20/` | DEV 20 rollouts with 4K/8K budgets (budget calibration) |

Each line is one trajectory: `model_id`, `trajectory_id`, `item_id`, `protocol`, `rep`, `disclosure_order`, `turns[]` (user message, `reasoning_trace`, `visible_reply`, per-judge scope labels, turn-level `e_L`/`e_U`), `final_grading` (`y_L`, `y_U`, both judges' verdicts), trajectory-level `e_L`, `e_U`, `j_L`, `j_U`, and `repair`.
**`trajectory_id` is not unique across models — key records by (`model_id`, `trajectory_id`).**

## Models and judges

| Role | Model | Details |
|---|---|---|
| Evaluated | `Qwen/Qwen3-8B` | revision `b968826d9c46dd6066d109eabc6255188de91218` |
| Evaluated | `Qwen/Qwen3-14B` | revision `40c069824f4251a91eefaf281ebe4c544efd3e18` |
| Evaluated | `zai-org/GLM-Z1-9B-0414` | revision `b221b06fefb23ca320922cf6e68ab5f2fb82de81` |
| Primary judge | `codex/gpt-6.1-sol` | OpenAI-compatible gateway (`OPENAI_BASE_URL`, default `http://127.0.0.1:3425/v1`), temperature 0, reasoning effort `low` |
| Secondary judge | `workbuddy/deepseek-v4.1-flash` | OpenAI-compatible gateway (`--secondary-base-url` or same `OPENAI_BASE_URL`), temperature 0, reasoning effort `low` |

Decoding for evaluated models: temperature 0.6, top-p 0.95, top-k 20, 16,384 output tokens, 32,768 context, native thinking templates; history carries visible replies only.
Judge access dates: final-answer judging `2026-11-21`; input-blind scope and repair judging `2026-11-22` (Asia/Shanghai).

## Reproducing

Run every command from the repository root, after `python unpack_data.py`; use `--help` for all options.

**Offline (no GPU, no API) — reproduces every number in the paper from the released files:**

0. *(Optional)* Verify repository file integrity: `sha256sum -c MANIFEST.sha256` (Linux/macOS) or verify via Python `hashlib`.
1. Verify the Stage 1 hashes and all 120 SymPy benchmark certificates (~1 min on CPU):
   `python run_exp1_pipeline.py` *(or directly `python stage2b_verification/verify_certificates.py`)*
2. Check every judge label against the recorded judge responses (rebuilds each prompt and matches its SHA-256 across all 26,942 TEST judge calls):
   `python eval_multi_turn/verify_cached_judgments.py`
3. SymPy cross-check of final answers:
   `python eval_multi_turn/revision_analysis/verify_final_y_sympy.py`
4. Rebuild the aggregated statistics and flat analysis tables (`test100_exp2_4_full_summary_16384_v13_blind.json`, `traj.csv`, `turns.csv`):
   ```bash
   python eval_multi_turn/analyze_experiments.py \
     --inputs eval_multi_turn/runs/runs_16k_v13_blind/judged_test100_*_v13_blind.jsonl \
     --output eval_multi_turn/runs/runs_16k_v13_blind/test100_exp2_4_full_summary_16384_v13_blind.json
   python eval_multi_turn/revision_analysis_v13/extract_tables.py
   ```
5. Regenerate all tables, figures, and reported numbers:
   `python eval_multi_turn/paper_revised/regenerate_v13_assets.py`

Steps 4–5 reproduce `test100_exp2_4_full_summary_16384_v13_blind.json`, `revision_analysis_v13/*.csv`, `v13_paper_numbers.json`, and the four `table*_rev.tex` files exactly (up to JSON key order / line endings); PDF figures may differ in embedded metadata/fonts across platforms.

**Full re-run (GPU + judge API):**

6. Rollouts, per model and protocol set:
   `python eval_multi_turn/rollout_vllm.py --split TEST --plan eval_multi_turn/disclosure_plan_test100.jsonl --model Qwen/Qwen3-8B --revision b968826d9c46dd6066d109eabc6255188de91218 --max-tokens 16384 --max-model-len 32768 --temperature 0.6 --top-p 0.95 --output <rollouts.jsonl> --meta-output <meta.json>`
7. Final-answer judging (Y; non-final scope labels written by this step are replaced in step 8):
   `OPENAI_API_KEY=<KEY> python eval_multi_turn/judge_trajectories.py --rollouts <rollouts.jsonl> --split TEST --output <judged.jsonl> --secondary-model workbuddy/deepseek-v4.1-flash --secondary-base-url <URL> --resume`
8. Input-blind scope / repair judging (input = output of step 7):
   `OPENAI_API_KEY=<KEY> python eval_multi_turn/judge_v13_input_blind.py --input-judged <judged.jsonl> --split TEST --output <judged_v13.jsonl> --secondary-model workbuddy/deepseek-v4.1-flash --secondary-base-url <URL> --resume`
   Passing a *copy* of a released `*_judge_cache.jsonl` via `--cache-path` with `--resume` replays the recorded responses offline without calling the API (without `--resume`, an existing cache file is overwritten).

Model sampling and judge APIs are not bit-reproducible; `final_answer_cache_test100_*.jsonl` and `*_v13_blind_judge_cache.jsonl` record the exact judge responses behind every label.

## Licenses

Code: [MIT License](LICENSE). Benchmark items are derived from OlympiadBench (He et al., ACL 2024) and remain subject to its license. Model outputs are subject to the licenses of the respective models (`Qwen/Qwen3-*`, `zai-org/GLM-Z1-9B-0414`).

## Citation

```bibtex
@inproceedings{cai2027certified,
  title     = {Certified Insufficiency and Premature Commitment: Process--Outcome Evaluation of Reasoning {LLMs} under Progressive Information Disclosure},
  author    = {Cai, Weike and Tang, Mohan},
  booktitle = {Proc. IEEE Int. Conf. Computer Supported Cooperative Work in Design (CSCWD)},
  year      = {2027}
}
```
