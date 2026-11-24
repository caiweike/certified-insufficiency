# Pre-Registered Unified Neutral No-Adjudication `[L, U]` Human Audit Protocol (`SEED = 2026`)

This document locks the sampling design, neutral interface rules, and statistical estimators **prior to annotation** for the unified multi-rater human audit of the *Certified Insufficiency and Premature Commitment* benchmark and `v1.3-input-blind` evaluation.

## 1. Core Methodological Principles

1. **Zero Reuse of Legacy Audits:** All three audit modules are freshly sampled under a unified random seed (`SEED = 2026`) and annotated from scratch on a clean, un-prefilled interface.
2. **100% Neutral Framing & Full Blinding:**
   - No leading headers (e.g., no *"premises are incomplete"* or *"earlier premature jump"* hints).
   - No model name, protocol label (`FREE`/`REQ`/`GUARD`), item regime tag (`TypeA`/`TypeB`), or LLM judge verdict is exposed anywhere in the UI or HTML payload.
   - In the Repair module, all judge-flagged positive candidates and dual-negative controls are shuffled into one indistinguishable blind pool.
3. **Zero Post-Hoc Self-Adjudication (`[L, U]` Dual/Multi-Rater Bounds):**
   - Raters ($M \ge 2$) work strictly independently and never negotiate or self-adjudicate disagreements afterward.
   - Every human-estimated quantity is reported as an exact **multi-rater `[L, U]` interval** ($L$: unanimous confirmation across all $M$ raters; $U$: flagged by any rater), directly matching the paper's dual-LLM judge $[E_L, E_U]$ bound framework.
   - When $M \ge 3$ and odd, majority vote is reported alongside $[L, U]$; if external (non-author) raters participate alongside authors, external-vs-author agreement is reported separately to audit confirmation bias.
4. **Distribution-Free Finite-Population Uncertainty:**
   - For census tasks (`Cert-100`), sampling variance is zero; uncertainty is strictly captured by multi-rater `[L, U]` bounds.
   - For finite-population stratified samples (`Repair-100`, `Scope-120`), uncertainty is quantified via exact distribution-free finite-population hypergeometric confidence bounds (for Repair) and Horvitz–Thompson inverse-probability-weighted (IPW) post-stratification with stratified bootstrap CIs (for Scope).

## 2. Module Specifications (`SEED = 2026`)

### Module ①: `Cert-100` (Full 100/100 TEST Certificate Census; 852 Evaluated Prefixes)
- **Population:** All `100` frozen TEST items (`68 TypeA_UniqueValue` + `32 TypeB_Extremum`), covering all `852` evaluated non-final prefixes (`591` TypeA + `261` TypeB across 3 disclosure orders per item).
- **Presentation:** Shuffled deterministically (`C001`–`C100`). Left pane shows the original Chinese problem, OlympiadBench gold answer, exact character-span shards `S1..Sk`, and all 3 evaluated disclosure orders with explicit non-final prefixes. Right pane shows the SymPy formalization and verified leave-one-out certificates.
- **3-Tier Rubric:**
  - `Faithful`: SymPy variable domains, shard constraints, and target faithfully match the Chinese problem, and every LOO witness is valid under the retained Chinese clauses.
  - `Minor`: Technical simplification or omitted secondary non-zero/non-degeneracy side constraint, yet every clause `S_j` remains strictly necessary in natural language (all evaluated prefixes remain insufficient).
  - `Unfaithful`: Substantive omission, translation error, or sharding flaw making a clause redundant or violating retained Chinese clauses. If selected, the rater inspects the 3 evaluated disclosure orders to judge `prefix_flip ∈ {No, Yes}` and list any affected prefixes (`RepX-TY`).
- **Pre-Registered Metrics:** Reported for `TypeA (68)`, `TypeB (32)`, and `ALL (100)`:
  - Multi-rater `[L, U]` rates for strict `Faithful`, necessity preserved (`Faithful + Minor`), and `prefix_flip` items / evaluated prefixes (`out of 591 / 261 / 852`).
  - Inter-rater reliability: exact `% agree`, Gwet's `AC1`, `PABAK`, and Cohen's/Fleiss' `κ`.
  - Exact induced bias on reported `E_L`, `E_U`, and `Y - J` (pp) from all trajectories/turns whose evaluated prefix is flagged by any rater (`U` bound).

### Module ②: `Scope-120` (120 Non-Final Turns; Exhaustive 9-Stratum Neutral Audit)
- **Population:** All `7,668` non-final turns across 3 models × 3 dynamic protocols × 100 items × 3 reps, partitioned into **9 exhaustive, mutually exclusive strata** (100.0% coverage, zero excluded turns):
  1. `agree_Commit_TypeA` (`N = 266`, `n = 18`)
  2. `agree_Commit_TypeB` (`N = 273`, `n = 14`)
  3. `disagree_Commit` (`N = 149`, `n = 16`)
  4. `agree_Wait_TypeA` (`N = 4,036`, `n = 18`)
  5. `agree_Wait_TypeB` (`N = 1,297`, `n = 12`)
  6. `agree_Cond_TypeA` (`N = 491`, `n = 14`)
  7. `agree_Cond_TypeB` (`N = 297`, `n = 10`)
  8. `disagree_Wait_Cond` (`N = 801`, `n = 12`)
  9. `other_noncommit` (`N = 58`, `n = 6`)
  - Total `n = 120` turns (`S001`–`S120`), balanced across `(model, protocol)` and unique items within each stratum, shuffled blindly.
- **Neutral Presentation:** Displays only the cumulative raw clauses disclosed up to Turn $t$ and the assistant's visible reply at Turn $t$ — **no claim that premises are incomplete**. Rater classifies the assistant's response stance into `Wait`, `Conditional`, `Commit`, or `Partial/Other`.
- **Pre-Registered Metrics (No Adjudication):**
  - Inter-rater agreement (binary `Commit` and 4-class scope; unweighted and IPW-weighted `% agree`, `AC1`, `κ`).
  - Direct comparison of human multi-rater bounds `[h_L, h_U]` against dual-LLM judge bounds `[e_L, e_U]`:
    - Binary commit agreement of each individual rater and `[h_L, h_U]` with `e_L`, `e_U`, primary, and secondary judges (unweighted and IPW-weighted).
    - **Interval compatibility rate:** share of turns where `[h_L, h_U] ∩ [e_L, e_U] ≠ ∅` (exact IPW estimate).
    - **IPW turn-level commit prevalence bounds:** human `[h_L, h_U]` vs. LLM judge `[e_L, e_U]` (and 95% stratified bootstrap CIs).

### Module ③: `Repair-100` (Unified Blind Mixed Pool: 26 Flagged + 74 Non-Empty Controls)
- **Population:** All `489` dynamic trajectories with `E_U = 1`, partitioned exhaustively into:
  - `26` judge-flagged candidates (`4` dual-positive + `22` single-judge positive): **100% census (`n = 26`)**.
  - `6` dual-negative trajectories whose final-turn `visible_reply` is deterministically empty (`""`): **deterministically `0/6` visible retractions**.
  - `457` non-empty dual-negative trajectories: **stratified without-replacement sample `n = 74`** (`25` Qwen3-8B, `25` GLM-Z1-9B, `24` Qwen3-14B, maximizing unique items).
  - The `26 + 74 = 100` trajectories (`R001`–`R100`) are shuffled into one completely blind pool.
- **Neutral Presentation:** Displays all complete problem clauses, the assistant's earlier visible reply at Turn $t_0$ (earliest `e_U = 1` turn), and the assistant's final-turn visible reply — **no "premature jump" or judge-status framing**. Rater answers `Yes` (final reply explicitly refers to and retracts/corrects the earlier reply's conclusion) or `No` (merely writes a new derivation/answer without explicit retraction).
- **Pre-Registered Metrics (No Adjudication + Exact Finite-Population 95% Interval):**
  - Inter-rater agreement on all `100` trajectories (`% agree`, `AC1`, `PABAK`, `κ`).
  - Confirmed retractions `[L, U]` separately on the `26` judge-flagged candidates (`k_pos_L`, `k_pos_U`) and the `74` non-empty dual-negative controls (`k_neg_L`, `k_neg_U`).
  - **Exact distribution-free finite-population hypergeometric 95% confidence interval** for total retractions across all `489` trajectories:
    - Exact lower 95% bound on `N = 457` given `k_neg_L` (`K_min`), combined with `k_pos_L` → total lower bound `(k_pos_L + K_min) / 489`.
    - Exact upper 95% bound on `N = 457` given `k_neg_U` (`K_max`), combined with `k_pos_U` → total upper bound `(k_pos_U + K_max) / 489`.
    - *(When `k_neg_U = 0` out of `74`, `K_max = 16/457 = 3.50%`, giving total 95% upper bound `<= (k_pos_U + 16) / 489` — e.g., `[0.2%, 3.5%]` if `k_pos = [1, 1]`.)*

## 3. Post-Audit Design-Based Inference Addendum (Correcting for `balanced_pick`)

During post-audit methodological review, we identified two inference caveats induced by the actual sampler `balanced_pick()` in `build_unified_audit_seed2026.py`:
1. **Unequal Inclusion Probabilities:** Because `balanced_pick()` enforces round-robin balance across `(model, protocol)` buckets of unequal population size and prioritizes unique `item_id`s in Pass 1, unit inclusion probabilities $\pi_i$ are unequal within strata ($\hat{\pi}_j \in [0.079, 0.303]$ across the `457` non-empty dual-negative Repair controls, and up to `23x` spread inside small Scope strata). Thus, the naive SRSWOR hypergeometric formula (`K_max = 16/457 = 3.50%`) does **not** hold a 95% coverage guarantee (`P_design(0 observed | K=16) >= 0.226`).
2. **Why Formal 95% Coverage Is Not Claimed for `Repair-100` or `Scope-120`:**
   - **Repair-100:** Heuristic adversarial search (greedy forward selection + 1-swap refinement + whole-item cluster packing over `M = 50,000` sampler replays) reaches empirical miss probability `< 0.05` at `29/457` (`6.35%`, or `6.13%` including `1/26` flagged across all `489` trajectories). Because local search over finite replays neither proves global optimality across all $\binom{457}{K}$ subsets nor eliminates optimization/simulation slack, **`6.1%` is reported strictly as a heuristic adversarial stress-test benchmark (a lower bound on the worst-case $\alpha = 0.05$ threshold), NOT a proven 95% confidence upper bound**, alongside the exact audited counts (`[0, 1]/26` flagged, `0/74` controls, `0/6` empty) and the design-unbiased Horvitz–Thompson point estimate (`[0.0%, 0.2%]`).
   - **Scope-120:** We estimate unit inclusion probabilities $\hat{\pi}_i$ via `M = 50,000` replays of `balanced_pick()` and report **stratum-calibrated Hájek design-weighted point estimates** ($w_i \propto 1/\hat{\pi}_i$, calibrated to exact $N_h$) alongside **approximate with-replacement linearized Hansen–Hurwitz standard errors (SEs)**. Although within-bucket sampling without replacement and Pass-1 item deduplication induce negative pairwise dependence within buckets ($\pi_{ij} = 0$ for same-item pairs), cross-bucket item blocking couples buckets and small stratum quotas ($n_h = 6\text{--}18$) suffer Wald zero-variance collapse on rare boundary events (e.g., `~62%` nominal-95% Wald coverage on a simulated 5% item-clustered indicator in `agree_Wait_TypeA`, vs. `~92.5%` on dense `turn_idx >= 3`). We therefore report approximate linearized SEs without claiming guaranteed conservative 95% CI coverage.
