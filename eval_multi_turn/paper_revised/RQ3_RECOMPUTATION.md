# RQ3 Final-Turn Retraction, Strict-Fidelity Sensitivity, and Follow-Up Audits

This directory contains the revised IEEE CSCWD LaTeX manuscript (`paper_revised.tex`), tables (`table1`--`table4`), figures (`fig1`, `fig2`), and verification scripts.

## 1. Human `Repair-100` Final-Turn Retraction (`recompute_rq3.py`)

This revision uses the original `Repair-100` sample. Each example shows all clauses in disclosure order, the visible reply at the earliest non-final turn with cached upper commit indicator $e_{t,U}=1$ ($t_0$), and the final visible reply ($k$). Three independent blind human raters judged whether turn $k$ explicitly referred back to and withdrew or corrected the turn-$t_0$ assertion. The original dual-LLM whole-dialogue screen defines the sampling strata (`26` screen-positive + `74` quota-sampled nonempty screen-negative pairs).

Run with Python 3.10+ against either the repository root (default `../..`) or a submission `.zip` archive:

    python recompute_rq3.py
    python recompute_rq3.py cscwd_submission_bundle.zip

The output matches `rq3_counts.json` bit-for-bit.

## 2. Unanimous Strict-Fidelity Sensitivity (`recompute_fidelity.py`)

Joins the three `Cert-100` ratings by audit ID, selects the `75` items unanimously rated `Faithful` (`53` `TypeA`, `22` `TypeB`), and recomputes `TypeA` `FREE`/`REQ` ranges (`E = 5.7%--14.5%`, `Y - J = 5.7--13.2 pp`) across all six model--protocol cells (`159` trajectories/cell):

    python recompute_fidelity.py
    python recompute_fidelity.py cscwd_submission_bundle.zip

The output matches `fidelity_counts.json` bit-for-bit.

## 3. SEED=2027 External Full-Trajectory (`Traj-60`) & `TypeA` Denominator (`Repair-50-TypeA`) Audits

See `../human_audit_followup_seed2027/FOLLOWUP_AUDIT_PROTOCOL_SEED2027.md` and run:

    python ../human_audit_followup_seed2027/score_followup_audit_seed2027.py \
      --traj ../human_audit_followup_seed2027/traj_60_yu.csv \
      --denom ../human_audit_followup_seed2027/repair50_denom_yu.csv

The output matches `../human_audit_followup_seed2027/followup_audit_results_seed2027.json` (`Traj-60` exact finite-population stratified `E = 31.5%`, `Y - J = 31.0%`, `SE = 5.3 pp`; `Repair-50-TypeA` nested denominators `N1 = 37`, `N2 = 12`, `N3 = 9`, `0/9` explicit final-turn retractions).

For the `Traj-60` criterion-sensitivity analysis (both `E` and `Y - J` under original external coding vs.\ Sec.\ IV-C reclassification of the 8 disputed `agree_NonCommit` cases) and `TypeA` item concentration:

    python recompute_traj60_sensitivity.py

This reproduces `traj60_sensitivity.json`; verbatim excerpts and reclassification rationales for all 8 disputed `agree_NonCommit` cases appear in `TRAJ60_DISPUTED_CASES.md`.

## 4. Build the Paper

Run `pdflatex` twice on `paper_revised.tex` from this directory with `IEEEtran` and the standard LaTeX packages installed.

## 5. Traj-60 Criterion Sensitivity and TypeA Commit Concentration (`recompute_traj60_sensitivity.py`)

`Traj-60` is a single-external-rater, within-stratum SRSWOR audit of 60 TypeA FREE/REQ trajectories (strata `agree_Commit` 12/152, `disagree_Commit` 8/45, `agree_NonCommit` 40/1,027). The stratified external estimate is E = 31.5%. Of this, 53.3% comes from the 8 flagged `agree_NonCommit` trajectories: 7 were labeled `Conditional` by both judges and 1 `Wait`/`Conditional`; 3 come from one item (`ob-cee-4656`). Each reply leaves the target open (free constant, unresolved case split, or range); the judges labeled all of them `Conditional` (or split `Wait`/`Conditional`). Rescoring these scoped replies as non-commit, as the `Conditional` rubric (Sec. IV) prescribes, gives E = 14.7%, within the dual-LLM TypeA FREE/REQ bounds [12.4%, 16.1%]; keeping the mixed `Wait`/`Conditional` case gives 16.8%, 0.7 pp above the upper bound. The manuscript therefore reports Traj-60 as a sensitivity analysis of the `Conditional`/`Commit` boundary, not as evidence that the judges are conservative. It also cannot rule out judge under-detection: the rater flagged 0/19 sampled `agree_NonCommit` trajectories whose turns were all dual-`Wait`, and each of its 8 `agree_NonCommit` flags falls on a turn that at least one judge labeled `Conditional`. Treating those 8 as scoped, 0/40 misses still leaves a one-sided 95% upper bound of about 7% on the miss rate in that stratum (7.2%, i.e., about 6.1 pp of E).

The script also reports these two control quantities and the item concentration of the 152 confirmed (e_L) TypeA FREE/REQ commits: 29/68 items have at least one; the five most frequent items contribute 42.1%; none of the five is unanimously Faithful in `Cert-100`; excluding them leaves per-cell E at 5.8%-13.8%.

Run: `python recompute_traj60_sensitivity.py [repo_root | bundle.zip]` (writes `traj60_sensitivity.json`).
