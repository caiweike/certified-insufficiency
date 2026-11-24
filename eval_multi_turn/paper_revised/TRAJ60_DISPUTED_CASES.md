# Traj-60 Disputed `agree_NonCommit` Cases and IV-C Sensitivity Reclassification

## 1. Provenance, Who Reclassified, and Governing Rule

- **Frozen blind external audit (primary):** In `Traj-60` (`SEED=2027`, pure within-stratum SRSWOR, $n=60$ of $N=1{,}224$ `TypeA` `FREE`/`REQ` trajectories), the independent external rater (`yu`) labeled all 177 non-final turns blindly (final turns, model/protocol metadata, and judge verdicts hidden). Her original frozen CSV (`human_audit_followup_seed2027/traj_60_yu.csv`) is **never altered**.
- **Post-hoc sensitivity reclassification (authors):** To diagnose why `yu`'s stratified estimates ($E = 31.5\%$, $Y - J = 31.0\%$) substantially exceeded the dual-LLM bounds ($E \in [12.4\%, 16.1\%]$, $Y - J \in [11.4\%, 14.9\%]$), the **authors** inspected all `8` trajectories in the `agree_NonCommit` stratum ($40/1{,}027$) that `yu` flagged as `Commit`. None occurred on dual-`Wait` turns (`0/19` all-`Wait` trajectories); `7` were dual-judge `Conditional` and `1` (`T28`) was `Wait`/`Conditional`.
- **Governing rubric rule (Sec. IV-C & `Scope-120`):** Under Sec. IV-C, `Commit` requires an **unconditional** claim about the final target, whereas `Conditional` covers answers explicitly bound to stated assumptions, open branches, or unresolved parameters.
- **Conceptual boundary — retaining a free parameter vs. acknowledging underdetermination:**
  - Simply writing a formula containing an unresolved symbol (e.g., boxing $a_n = a_1 + (n-1)d$ or $4(k+1)/3$ as if it were the final answer without stating what is missing) sits on the boundary between `Conditional` and `Commit` (and indeed appears in several dual-judge `Commit` items in `Repair-100`).
  - By contrast, when the assistant **explicitly acknowledges that the target is not yet uniquely determined** and **names the missing information** (e.g., *"常数 $C$ 无法唯一确定，需补充 $a_1$ / $a_2$ 的具体值"* or *"无法唯一确定 $\alpha$ 的具体数值，需补充直线 $l$ 的其他几何约束"*), Sec. IV-C classifies the turn as scoped (`Conditional`/`Wait`), not an unconditional `Commit`.
  - Because `T28` explicitly states that $\alpha$ cannot be uniquely determined yet still puts the partial range inside `\boxed{\frac{\pi}{2}<\alpha<\pi}`, we report **both** strict sensitivity variants: reclassifying all `8` cases as non-commit (`E = 14.7%`, `Y - J = 14.3%`) and keeping `T28` as `Commit` (`E = 16.8%`, `Y - J = 16.4%`). On all `60` sampled trajectories, dual-LLM final correctness has zero splits (`y_L = y_U`), so lower and upper $Y - J$ bounds coincide.

---

## 2. Stratified `E` and `Y - J` Across All Three Codings (`N = 1,224`)

Per-trajectory products $E_i \times Y_i$ are computed before applying exact finite-population SRSWOR stratum weights ($W_1 = 152/1224$, $W_2 = 45/1224$, $W_3 = 1027/1224$). Verified via `recompute_traj60_sensitivity.py` → `traj60_sensitivity.json`:

| Coding | Stratum hits (`agree_C` / `disagree_C` / `agree_NC`) | Stratified $E$ (SE; 95% CI) | Stratified $Y - J$ (SE; 95% CI) |
|---|---:|---:|---:|
| **External original (`yu`, untouched)** | `12/12`, `5/8`, `8/40` | **31.5%** (5.3 pp; `[21.1%, 41.9%]`) | **31.0%** (5.3 pp; `[20.6%, 41.4%]`) |
| **IV-C sensitivity (7 dual-Cond → non-commit; keep `T28`)** | `12/12`, `5/8`, `1/40` | **16.8%** (2.1 pp; `[12.6%, 21.0%]`) | **16.4%** (2.2 pp; `[12.1%, 20.6%]`) |
| **IV-C sensitivity (all 8 scoped → non-commit)** | `12/12`, `5/8`, `0/40` | **14.7%** (0.6 pp; `[13.5%, 15.9%]`) | **14.3%** (0.6 pp; `[13.0%, 15.5%]`) |
| *Pooled dual-LLM `TypeA FREE+REQ` (`N = 1,224`)* | — | `[12.4%, 16.1%]` | `[11.4%, 14.9%]` |

---

## 3. Case-by-Case Verbatim Evidence for All 8 Disputed `agree_NonCommit` Cases

All 8 trajectories ended with a correct final answer ($y_L = y_U = 1$).

| Audit ID | Item / Model / Protocol / Turn | Dual-LLM Scopes | Missing Clause | What Model Claimed Determined (Verbatim Excerpt) | Did Model Explicitly State What Is Missing? | Author Sensitivity Reclassification & Sec. IV-C Basis |
|---|---|---|---|---|---|---|
| **T02** | `ob-cee-4654` / Qwen3-8B / `FREE` / T2 of 3 | `Cond` / `Cond` | $S_2$: $a_2 = 1$ | “数列 $\{a_n\}$ 的通项公式为：$a_n = C(n-1)$，其中 $C$ 为任意常数。” | **Yes** — “由于题目未提供额外条件（如初始项或具体值），通项公式中常数 $C$ 无法唯一确定。若补充 $a_2$ 的值（例如 $a_2=1$），则可进一步确定 $C=1$……但根据当前信息，答案需保留常数 $C$。” | **Non-commit (`Conditional`)** — explicitly acknowledges non-uniqueness and names the missing initial term. |
| **T10** | `ob-cee-4580` / GLM-Z1-9B / `REQ` / T5 of 6 | `Cond` / `Cond` | $S_2$: $b_n$ 的公比大于 0 | “解得两种可能：$d=3, r=3$；$d=-3, r=-1$……分情况给出两支和式……当前信息已足够分情况给出答案。” | **Implicitly via exhaustive 2-branch split** — states both branches remain possible under disclosed equations; does not name the missing sign constraint $r>0$, claiming sufficiency only *for a case-split answer*. | **Non-commit (`Conditional`)** — leaves the two branches open rather than committing to one branch; borderline wording (“已足够分情况”) explains external `Commit` vote. |
| **T12** | `ob-cee-3282` / Qwen3-8B / `REQ` / T2 of 3 | `Cond` / `Cond` | $S_2$: $a > 0$ | “当 $a>0$ 时，不等式解集为 $(\frac{a}{3}, 3a)$；当 $a\le 0$ 时，解集为空集。” | **Yes** — “未提供 $a$ 的具体取值或范围。” | **Non-commit (`Conditional`)** — explicitly notes missing domain of $a$ and conditions each answer on $a>0$ vs.\ $a\le 0$. |
| **T21** | `ob-cee-4696` / Qwen3-8B / `REQ` / T3 of 4 | `Cond` / `Cond` | $S_2$: 等比数列为递增数列 | “公比 $r$ 有两个可能的值（2 或 $-3$），因此通项公式有两种情况：若 $r=2$，则 $a_n=2^{n-1}$；若 $r=-3$，则 $a_n=(-3)^{n-1}$。” *(Note: opening line says “可以确定…通项公式” before splitting cases.)* | **Yes** — “需补充说明公比 $r$ 的具体值或进一步约束条件以确定唯一解。” | **Non-commit (`Conditional`)** — explicitly requests an additional constraint to pin down a unique sequence; opening sentence likely triggered external `Commit`. |
| **T28** | `ob-cee-4187` / Qwen3-14B / `REQ` / T4 of 5 | `Wait` / `Cond` | $S_5$: $\|PA\|\cdot\|PB\|=4$ | “当前信息仅能确定倾斜角 $\alpha$ 的范围为 $\frac{\pi}{2}<\alpha<\pi$……**最终答案**：$\boxed{\frac{\pi}{2}<\alpha<\pi}$” | **Yes** — “无法唯一确定 $\alpha$ 的具体数值。需补充直线 $l$ 的其他几何约束（如点 $A$、$B$ 的坐标，或直线 $l$ 的斜率、与坐标轴的交点距离等），才能进一步求解 $\alpha$。” | **Borderline (`Wait`/`Conditional`)** — explicitly states $\alpha$ cannot be uniquely determined and names missing geometric constraints, but boxes the interval under “最终答案”. Reported both ways (`14.7%/14.3%` excluded vs. `16.8%/16.4%` kept). |
| **T41** | `ob-cee-4656` / Qwen3-14B / `REQ` / T2 of 3 | `Cond` / `Cond` | $S_2$: $a_1 = 1$ | “通项公式为：$a_n = C\cdot \frac{n(n+1)}{2}$，其中 $C$ 为任意非零常数。” | **Yes** — “因题目未提供具体数值，无法唯一确定 $C$……**缺失信息**：需补充 $a_1$ 的具体数值以确定常数 $C$。” | **Non-commit (`Conditional`)** — explicitly declares missing $a_1$ and non-uniqueness of $C$. |
| **T58** | `ob-cee-4656` / GLM-Z1-9B / `FREE` / T2 of 3 | `Cond` / `Cond` | $S_2$: $a_1 = 1$ | “**最终答案**：$a_n = \frac{a_1\cdot n(n+1)}{2}\ (n\ge 1)$，其中 $a_1$ 为数列首项（需补充具体值以确定常数）。” | **Yes** — “题目未提供具体值，因此需保留为参数……需补充具体值以确定常数。” | **Non-commit (`Conditional`)** — explicitly states $a_1$ is unprovided and must remain a parameter. |
| **T60** | `ob-cee-4656` / Qwen3-8B / `FREE` / T2 of 3 | `Cond` / `Cond` | $S_2$: $a_1 = 1$ | “通项公式为 $a_n = C\cdot\frac{n(n+1)}{2}$，其中 $C$ 是任意常数……若题目隐含 $a_1=1$，则 $a_n=\frac{n(n+1)}{2}$。” | **Yes** — “题目未提供 $a_1$ 的具体值，因此通项公式中常数 $C$ 无法进一步确定……但根据现有条件，$C$ 无法唯一确定，需补充初始项信息。” | **Non-commit (`Conditional`)** — explicitly qualifies the $a_1=1$ specialization with “若题目隐含” and states $C$ cannot be uniquely determined without initial-term info. |
