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

The output matches `../human_audit_followup_seed2027/followup_audit_results_seed2027.json` (`Traj-60` exact finite-population stratified `E = 31.5%`, `SE = 5.3 pp`, 95% CI `[21.1%, 41.9%]`; `Y - J = 31.0%`, `SE = 5.3 pp`, 95% CI `[20.6%, 41.4%]`; `Repair-50-TypeA` nested denominators `N1 = 37`, `N2 = 12`, `N3 = 9`, `0/9` explicit final-turn retractions).

## 4. Build the Paper

Run `pdflatex` twice on `paper_revised.tex` from this directory with `IEEEtran` and the standard LaTeX packages installed.
