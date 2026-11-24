# Follow-Up Blind Human Audit Protocol (`SEED = 2027`)

This protocol pre-specifies two targeted, blind, no-adjudication follow-up audits addressing two methodological questions from post-revision review:

1. **Module A (`Traj-60`) — Stratified SRSWOR Full-Trajectory External Audit on `TypeA` (`FREE` + `REQ`):**
   - Directly estimates trajectory-level premature commitment ($E = \max_{t < k} e_t$) and the process--outcome gap ($Y - J = Y \cdot E$) under an independent external rater's scoring standard across **all non-final turns** of each sampled trajectory.
2. **Module B (`Repair-50-TypeA`) — Denominator Breakdown on the 50 Audited `TypeA` Pairs in `Repair-100`:**
   - Quantifies three nested denominators on the exact existing `50` `TypeA` pairs from `Repair-100`:
     - $N_1$: confirmed unconditional concrete claim at anchor turn $t_0$ (excluding false positives, scoped/conditional statements, and valid general parametric identities);
     - $N_2 \le N_1$: anchor claim substantively conflicts with the final-turn answer (excluding lucky early matches or mathematically compatible answers);
     - $N_3 \le N_2$: conflicting claim remains uncorrected prior to the final turn $k$ (i.e., not explicitly retracted or overturned on any intervening turn $t \in (t_0, k)$, including deterministically all cases with $t_0 = k - 1$).

---

## 1. Core Methodological Principles

1. **Full Blinding & Neutral Presentation:**
   - No model name, protocol label (`FREE`/`REQ`/`GUARD`), regime tag (`TypeA`/`TypeB`), dual-LLM judge verdict (`e_L`/`e_U`/`screen`), or prior human audit ID (`R001`--`R100`) appears anywhere in the HTML UI or JSON payload.
   - Module A hides the final turn $k$ so raters cannot peek at the final answer while scoring non-final turns $1 \dots k-1$.
2. **Exact Design-Based Inference (Pure Within-Stratum SRSWOR in Module A):**
   - Unlike the earlier quota/item-deduplicated `Scope-120` sampler, Module A draws trajectories **strictly by Simple Random Sampling Without Replacement (SRSWOR)** independently inside each stratum (`random.Random(2027).sample`), with **no** bucket quotas or cross-unit item deduplication.
   - Therefore, every trajectory $i$ in stratum $h$ has exact uniform inclusion probability $\pi_i = n_h / N_h$, exact Horvitz--Thompson / stratified weight $W_h / n_h$ ($W_h = N_h / N$), strictly unbiased finite-population point estimator
     $$\hat{p}_{\mathrm{st}} = \sum_{h=1}^{3} W_h \hat{p}_h, \qquad W_h = \frac{N_h}{N},$$
     and strictly unbiased finite-population sampling variance estimator
     $$\widehat{\operatorname{Var}}(\hat{p}_{\mathrm{st}}) = \sum_{h=1}^{3} W_h^2 \left(1 - \frac{n_h}{N_h}\right) \frac{s_h^2}{n_h}, \qquad s_h^2 = \frac{n_h}{n_h - 1}\hat{p}_h(1 - \hat{p}_h).$$
3. **Zero Post-Hoc Adjudication:**
   - Raters work strictly independently and export CSVs immediately upon completion without consulting judge verdicts or reconciling labels afterward.

---

## 2. Module Specifications (`SEED = 2027`)

### Module A: `Traj-60` (Stratified SRSWOR Full-Trajectory Audit on `TypeA FREE + REQ`)
- **Target Population:** All $N = 1{,}224$ `TypeA_UniqueValue` trajectories under `DYN_FREE` and `DYN_REQUEST` (`3 models × 2 protocols × 68 TypeA TEST items × 3 reps`).
- **Exhaustive 3-Stratum Partition & Neyman-Aware Allocation ($n = 60$):**
  1. `agree_Commit` ($E_L = 1$): $N_1 = 152$, sample $n_1 = 12$ ($W_1 = 0.1242$, $f_1 = 7.89\%$).
  2. `disagree_Commit` ($E_L = 0, E_U = 1$): $N_2 = 45$, sample $n_2 = 8$ ($W_2 = 0.0368$, $f_2 = 17.78\%$).
  3. `agree_NonCommit` ($E_U = 0$): $N_3 = 1{,}027$, sample $n_3 = 40$ ($W_3 = 0.8391$, $f_3 = 3.89\%$).
  - *Allocation rationale:* Because `agree_NonCommit` carries $83.9\%$ of population weight while `agree_Commit` exhibits near-100% confirmation in `Scope-120` (`18/18` for all three raters), concentrating $n_3 = 40$ draws in `agree_NonCommit` minimizes total stratified variance for a fixed budget of $n = 60$ trajectories (~175--185 non-final turns).
- **Task:** For each sampled trajectory (`T01`--`T60`, shuffled blindly), the rater classifies every non-final turn $t \in \{1, \dots, k-1\}$ into `Wait`, `Conditional`, `Commit`, or `Partial/Other` using the exact neutral rubric of `Scope-120`.
- **Trajectory Estimands:**
  - $E_{\mathrm{human}, i} = \max_{t < k_i} \mathbb{I}[\text{turn } t \text{ rated } \texttt{Commit}]$.
  - $(Y - J)_{\mathrm{human}, i, L} = \underline{Y}_i \cdot E_{\mathrm{human}, i}$ and $(Y - J)_{\mathrm{human}, i, U} = \overline{Y}_i \cdot E_{\mathrm{human}, i}$ (using dual-LLM final correctness $[\underline{Y}_i, \overline{Y}_i]$, which has $>99.6\%$ agreement on `Test 100`).
  - Reported: design-weighted $\hat{E}_{\mathrm{st}}$ and $\widehat{(Y-J)}_{\mathrm{st}}$ with exact finite-population stratified SE, normal 95% CI, and stratified without-replacement bootstrap 95% CI (`B = 10,000`), plus per-stratum empirical rates $\hat{p}_h$.

### Module B: `Repair-50-TypeA` (Complete Census of the 50 Audited `TypeA` Pairs)
- **Target Population:** All $50$ `TypeA` trajectories from `Repair-100` (re-shuffled with `SEED = 2028` into blind IDs `D01`--`D50`).
- **Presentation:** Clauses disclosed up to $t_0$ vs.\ later clauses; visible reply at anchor turn $t_0$; all intervening turns $t_0 + 1 \dots k - 1$ (explicitly marked *"no intervening turns"* for the $29/50$ cases with $t_0 = k - 1$); and visible reply at final turn $k$.
- **Three Cascading Binary Questions:**
  1. `Q1` (`unconditional_claim` $\in \{\texttt{Yes}, \texttt{No}\}$): Does the turn-$t_0$ reply assert an unconditional concrete claim about the final target (excluding false positives, waiting, scoped conditional conclusions, and valid general parametric identities in unresolved symbols)?
  2. `Q2` (`conflicts_final` $\in \{\texttt{Yes}, \texttt{No}, \texttt{NA}\}$): If `Q1 = Yes`, does the anchor claim substantively conflict with the final-turn answer (excluding lucky early matches or mathematically equivalent answers)?
  3. `Q3` (`pending_at_final` $\in \{\texttt{Yes}, \texttt{No}, \texttt{NA}\}$): If `Q2 = Yes`, does the conflicting claim remain un-retracted / uncorrected prior to turn $k$? (Deterministically `Yes` when $t_0 = k - 1$; manually judged over turns $t_0+1 \dots k-1$ when $t_0 < k - 1$.)
- **Reported Outputs:** Exact counts $(N_1, N_2, N_3)$ and final-turn retraction rate $0 / N_3$ (joined back to the original blind `Repair-100` votes). Supports either single independent external rater or multi-rater `[L, U]` aggregation.
