# 31 — C3 efficiency study: tempered-trajectory Hamiltonian component (2026-10-12)

**Dated record of one campaign, written 2026-10-12 from finished runs only.** Runs made 2026-10-12 20:46–22:45 UTC
on the uncommitted working tree of `working_Kuba` (the C3 component of note 30 as implemented that day; no package
file was changed by this study). 207 runs, all return code 0, none hit the 300 s time limit, none hung: 180 grid
runs (the specification) + 27 follow-up runs added after an exact-gradient diagnostic (§2.5; **not in the original
specification**). Raw runs, drivers, tables, figure: `toy_examples/tmp_c3_study_2026-10-12/` (§5).

**Read first (limits of what was measured):**

* **3 repetitions per cell.** Crossing counts per run are small (1–90 for most tempered cells); the min–max over
  repetitions is given everywhere and is often as wide as the difference between neighbouring cells. Only
  differences of several-fold are discussed as findings.
* **Crossings cannot be attributed to the component.** `subchain_stats` has no per-component column (note 30 §7
  asked for one; not added, package read-only for this study). "Crossings" are all mode-to-mode transitions of the
  measured stage, whichever component made them. The tempered component's share is inferred only from the
  difference to the plain baseline.
* **The min-over-parameters ESS/eval (study 29's metric) is meaningless on runs whose chains do not cross.** It is a
  sum of within-chain ESS and does not see that chains disagree: plain H on M3-20 has the highest ESS/eval of the
  whole study (1.21) with 20 of 24 chains never crossing and 3/3 runs "poor". Read ESS/eval only on rows with a
  3/0/0 verdict; the column **mode ESS (ref)** = `n_chains · p(1−p) / mean_c (frac_c − p)²` (per-chain mode
  fractions against the exact mass p) is the crossing-sensitive efficiency used below. It is noisy (8 chains).
* **DAMH-SMU runs are not reproducible** (asynchronous network delivery); repetitions use `seed_offset = 100 000 · rep`
  and `NeuralNetworkUpdater(seed=rep)`.
* **Cheap models.** B2 and M3 cost microseconds; ESS/eval counts exact evaluations (whole run, 32 000 + 200 held-out),
  ESS/s counts network gradients and everything else. They rank differently (§3 ii).
* **The pinned tempered step is the stage-2 adapted step** (carry-over at the end of stage 2; manifest
  `component1.step_size_initial` equals the stage-2 carry-over `step_size` in all 156 grid tempered runs). The adaptive component continues
  its dual averaging into stage 3 ("continued") except where the pooled covariance changed: then its dual averaging
  restarts ("restarted (mass changed)": B2 30/45, M3-20 10/45, M3-10 4/45, M3-5 0/45 grid runs).
* **Error model on** in every DAMH stage (per study 29 §2.7 it costs 2.4× ESS/eval on accurate surrogates). Not varied.
* **The follow-up runs (§2.5) test two settings outside the specified grid** (β = 0.001; a tempered step pinned at
  half the adapted step, given as an explicit number per problem). They are 3 repetitions each on B2, M3-10, M3-20;
  M3-5 was not re-run.

---

## 1. Setup

### 1.1 Problems and references (`scripts/problems.py`, `refs/`)

| name | problem | d | network | held-out | reference |
|---|---|---|---|---|---|
| B2 | study 29's: `Solver_illustrative_local`, prior N(0, I₂), likelihood N(5, 1) | 2 | (32, 32) | 200 | `refs/ref_B2.json` (copied from study 29: 3001² quadrature; left arm {y − x > 1} holds 0.2933) |
| M3-5 / M3-10 / M3-20 | `solver_examples/bimodal3d.make_problem(Δ)`, Δ = 5 / 10 / 20 nats; modes ±(1,1,1), saddle at 0 | 3 | (32, 32) | 200 | `refs/ref_M3-{5,10,20}.json` from `bimodal3d.reference(Δ)`: mean 0, sd 1.266 / 1.280 / 1.286, mode mass 0.5 |

Every run: `mpiexec -n 9` (8 samplers + 1 collector), in-process solver, `use_solvers_pool=False`,
`initial_sample_type="lhs"`, `min_snapshots_initial = 10 d`, `seed_offset = 100 000 · rep`,
`save_snapshots_to_file=False`, `NeuralNetworkUpdater(hidden_layer_sizes=(32, 32), seed=rep)`,
`TestData.generate(problem, 200)`, default `prior_bound` (box ±8). Stages (`scripts/run_case.py`), budget per chain
4 000 exact evaluations:

1. `Stage(proposal=RandomWalk(), max_evaluations=1000, burn_in=True)`
2. `Stage(algorithm="DAMH", proposal=H_ADAPT, max_evaluations=1000, burn_in=True, surrogate_model_updates=True,
   polyak_averaging=True, surrogate_error_model=True, time_limit=300)` with `H_ADAPT = Hamiltonian(integrator=
   "dimension_robust", mass="covariance", integration_time=3.0, integration_time_jitter=0.5, target_rate=0.9)`
   (B2: `target_rate=0.8`)
3. the measured stage, same flags, `max_evaluations=2000`, proposal `P`:
   * **plain H**: `H_ADAPT`;
   * **H + PCN(β=1) 0.8/0.2**: `Mixture([H_ADAPT, PCN(beta=1.0)], probabilities=[0.8, 0.2])`;
   * **temp p_t β f T×k**: `Mixture([H_ADAPT, Hamiltonian(integrator="dimension_robust", mass="covariance",
     integration_time=3k, integration_time_jitter=0.5, adaptive=False, tempered_fraction=f, tempered_beta=β)],
     probabilities=[1 − p_t, p_t])`; grid p_t ∈ {0.1, 0.3} × β ∈ {0.3, 0.1, 0.03} × f ∈ {0.2, 0.35}, k = 3, plus
     (0.3, 0.1, 0.35) with k = 1.

Adapted step at the end of stage 2 (mean, range over the 45 grid runs) and the nominal tempered trajectory at T×3
(L_t = 9/ε; n_t = ⌊f L_t⌋ per end):

| problem | ε (mean, min–max) | nominal L of H_ADAPT | nominal L_t (T×3) | nominal n_t f = 0.2 / 0.35 |
|---|---|---|---|---|
| B2 | 0.563 (0.249–1.141) | 5.3 | 22 (7.9–36) | 3.8 / 7.4 |
| M3-5 | 0.187 (0.175–0.200) | 16 | 48 (45–52) | 9.2 / 16.3 |
| M3-10 | 0.156 (0.139–0.190) | 19 | 58 (47–65) | 11.1 / 19.8 |
| M3-20 | 0.123 (0.112–0.143) | 24 | 74 (63–80) | 14.0 / 25.3 |

`tempered_skipped` (trajectories with n_t = 0) is 0 in every M3 run; on B2 it is ≤ 0.002 of the iterations at T×3
and 0.042 at T×1 (short trajectories when the adapted step is large).

### 1.2 Metrics (`scripts/analyze.py`, `scripts/summary.py`)

Measured stage = stage index 2 only. ESS: per-chain Geyer ESS summed over chains, min over parameters, divided by all
exact evaluations of the run (stages 0–2 + 200 held-out); ESS/s = the same ESS over the run's wall time. Posterior
check as study 29 (max |mean − ref| and |E x² − ref| in ESS-based SE; **"poor" if > 4 SE or the mode mass (M3:
P(t > 0), reference 0.5; B2: left-arm mass, reference 0.293) is off by > 0.1**; "failed" if a DAMH stage stopped on
its time limit — never happened). Crossings: M3 `bimodal3d.crossings` (hysteresis ±0.5 on t) per chain; B2 study
29's `side()` hysteresis on y − x; "/ 1 000" = per 1 000 stage-2 iterations (all chains pooled = the per-chain rate).
Outer acceptance = accepted / (accepted + rejected); pre-rej and oob (out-of-box proposals) per stage-2 iteration;
skipped = `tempered_skipped` per stage-2 iteration; never-crossed chains = chains with 0 crossings out of 24 (3 runs
× 8). The mean leapfrog count of the tempered component is not observable per component (adaptive_stats records
component 0 only); the nominal T_t/ε is given in §1.1.

### 1.3 Timeline (`logs/time_log.txt`, `logs/batch_progress.txt`)

| time (UTC) | event |
|---|---|
| 20:43 | start; scripts adapted from study 29 |
| 20:45 | smoke run (M3-10, tempered, 21 s; deleted) |
| 20:46–22:09 | grid, 180 runs in the order B2, M3-10, M3-20, M3-5 (23–39 s each); total < 2.5 h, so M3-5 kept its f = 0.2 column |
| 22:18–22:29 | re-analysis of all runs with the final `analyze.py`; exact-gradient diagnostic (`diag_exact.json`, `logs/diag_exact.txt`) |
| 22:29–22:45 | follow-up, 27 runs (`jobs/extra.jsonl`) |

Sum of run times 99 min for 207 runs.

---

## 2. Results per problem

Rows: baselines first, then the 12 grid cells (T×3), the T×1 row, and (B2, M3-10, M3-20) the follow-up rows
(§2.5). "mode ESS (ref)" and the verdict column are the ones to read first (§ Read first). Full per-run data:
`table_all.csv`; per-cell: `table_configs.csv`.

### 2.1 B2 (2-D, left arm 0.293, 11-nat barrier)

| configuration | runs | ESS/eval ± sd | ESS/s | mode ESS (ref) | crossings / 1 000 (min–max) | crossings / run | mode mass (reps) | ok/poor/failed | outer acc | pre-rej | oob | skipped | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 0.206 ± 0.077 | 367 | 11.2 | 0.664 (0.249–0.996) | 13 | 0.26;0.37;0.38 | 0/3/0 | 0.94 | 0.20 | 0.048 | – | 11/24 | 18 |
| H + PCN(β=1) 0.8/0.2 | 3 | 0.00946 ± 9.9e-05 | 17.1 | 216 | 6.8 (6.25–7.43) | 166 | 0.28;0.33;0.29 | 3/0/0 | 0.95 | 0.34 | 0.039 | – | 0/24 | 18 |
| temp p_t=0.1 β=0.3 f=0.2 T×3 | 3 | 0.231 ± 0.088 | 396 | 21.1 | 0.615 (0.406–0.806) | 14 | 0.37;0.14;0.18 | 0/3/0 | 0.95 | 0.28 | 0.140 | 0.000 | 11/24 | 19 |
| temp p_t=0.1 β=0.3 f=0.35 T×3 | 3 | 0.2 ± 0.054 | 327 | 13.3 | 0.839 (0.632–0.988) | 19 | 0.26;0.22;0.48 | 0/3/0 | 0.96 | 0.28 | 0.144 | 0.000 | 8/24 | 20 |
| temp p_t=0.1 β=0.1 f=0.2 T×3 | 3 | 0.197 ± 0.066 | 321 | 10.9 | 0.54 (0.316–0.856) | 12 | 0.39;0.30;0.32 | 2/1/0 | 0.95 | 0.28 | 0.141 | 0.001 | 11/24 | 20 |
| temp p_t=0.1 β=0.1 f=0.35 T×3 | 3 | 0.2 ± 0.073 | 320 | 7.71 | 0.885 (0.721–1.08) | 20 | 0.57;0.45;0.41 | 0/3/0 | 0.94 | 0.28 | 0.144 | 0.000 | 10/24 | 20 |
| temp p_t=0.1 β=0.03 f=0.2 T×3 | 3 | 0.221 ± 0.022 | 366 | 16.3 | 0.436 (0.181–0.587) | 10 | 0.44;0.19;0.12 | 0/3/0 | 0.95 | 0.28 | 0.140 | 0.000 | 11/24 | 19 |
| temp p_t=0.1 β=0.03 f=0.35 T×3 | 3 | 0.167 ± 0.04 | 276 | 14.9 | 0.929 (0.316–1.53) | 21 | 0.67;0.09;0.35 | 0/3/0 | 0.95 | 0.28 | 0.142 | 0.000 | 8/24 | 19 |
| temp p_t=0.3 β=0.3 f=0.2 T×3 | 3 | 0.154 ± 0.049 | 221 | 7.75 | 0.536 (0.313–0.948) | 15 | 0.42;0.60;0.42 | 0/3/0 | 0.94 | 0.44 | 0.336 | 0.000 | 9/24 | 22 |
| temp p_t=0.3 β=0.3 f=0.35 T×3 | 3 | 0.163 ± 0.062 | 242 | 21.1 | 0.397 (0.349–0.423) | 11 | 0.41;0.20;0.21 | 0/3/0 | 0.95 | 0.44 | 0.333 | 0.000 | 10/24 | 21 |
| temp p_t=0.3 β=0.1 f=0.2 T×3 | 3 | 0.142 ± 0.025 | 202 | 11.4 | 0.71 (0.35–1.11) | 20 | 0.26;0.42;0.61 | 0/3/0 | 0.95 | 0.44 | 0.336 | 0.002 | 7/24 | 23 |
| temp p_t=0.3 β=0.1 f=0.35 T×3 | 3 | 0.168 ± 0.038 | 244 | 15.5 | 0.55 (0.244–1.02) | 16 | 0.21;0.39;0.22 | 0/3/0 | 0.95 | 0.44 | 0.335 | 0.000 | 9/24 | 22 |
| temp p_t=0.3 β=0.03 f=0.2 T×3 | 3 | 0.16 ± 0.026 | 227 | 7.78 | 1.13 (0.8–1.37) | 32 | 0.54;0.50;0.42 | 0/3/0 | 0.94 | 0.44 | 0.336 | 0.001 | 8/24 | 23 |
| temp p_t=0.3 β=0.03 f=0.35 T×3 | 3 | 0.148 ± 0.047 | 220 | 17.3 | 0.5 (0.383–0.591) | 14 | 0.50;0.24;0.20 | 0/3/0 | 0.95 | 0.44 | 0.335 | 0.000 | 9/24 | 21 |
| temp p_t=0.3 β=0.1 f=0.35 T×1 | 3 | 0.13 ± 0.11 | 198 | 17.8 | 0.57 (0.246–0.936) | 16 | 0.38;0.11;0.49 | 0/3/0 | 0.95 | 0.44 | 0.320 | 0.042 | 9/24 | 21 |
| temp p_t=0.3 β=0.03 f=0.35 T×3 ε_t=0.235 (pinned) [follow-up] | 3 | 0.164 ± 0.043 | 229 | 11.8 | 0.489 (0.244–0.667) | 14 | 0.21;0.45;0.34 | 0/3/0 | 0.95 | 0.44 | 0.338 | 0.000 | 9/24 | 23 |
| temp p_t=0.3 β=0.001 f=0.35 T×3 [follow-up] | 3 | 0.191 ± 0.018 | 280 | 10.1 | 0.628 (0.386–0.837) | 18 | 0.61;0.22;0.40 | 0/3/0 | 0.94 | 0.44 | 0.336 | 0.000 | 11/24 | 22 |
| temp p_t=0.3 β=0.001 f=0.35 T×3 ε_t=0.235 (pinned) [follow-up] | 3 | 0.198 ± 0.067 | 274 | 12 | 0.546 (0.245–0.801) | 16 | 0.24;0.22;0.40 | 0/3/0 | 0.95 | 0.44 | 0.335 | 0.000 | 13/24 | 23 |


* **No tempered configuration helps on B2.** Crossings stay at plain's level, 0.4–1.1 per 1 000 (plain 0.66), and 46 of
  the 48 tempered runs (39 grid + 9 follow-up) are "poor" (left-arm mass 0.09–0.67 against 0.293). Only the prior
  draw passes (3/3, 6.8 crossings per 1 000, mode ESS 216 vs 11 for plain).
* **Mechanism: the tempered proposals leave the box.** The oob share rises from 0.048 (plain) to 0.14 (p_t = 0.1)
  and 0.34 (p_t = 0.3); attributing plain's rate to the adaptive component, (oob − (1 − p_t) · oob_plain) / p_t ≈
  **0.95–1.01 of the tempered proposals end outside ±8** in every B2 cell, at every β, with T×1 and with the step
  halved (follow-up). On B2 the component is a no-op that costs 10–30 % of the iterations.
* Plain H crosses far less here (13 per run) than in study 29's best cell (1 408 per run over a single 3 000-
  evaluation stage). The layouts differ (here the measured stage starts with a tuned step and a trained network;
  study 29 §2.4 found its B2 stage spent ~500 updates at steps 10–100× the final one). Consistent with that
  transient doing the crossing in study 29; **not verified**.

### 2.2 M3-5 (Δ = 5 nats)

| configuration | runs | ESS/eval ± sd | ESS/s | mode ESS (ref) | crossings / 1 000 (min–max) | crossings / run | mode mass (reps) | ok/poor/failed | outer acc | pre-rej | oob | skipped | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 0.00543 ± 0.00087 | 14.1 | 94.9 | 4.03 (3.71–4.55) | 72 | 0.55;0.50;0.60 | 3/0/0 | 1.00 | 0.10 | 0.007 | – | 0/24 | 12 |
| H + PCN(β=1) 0.8/0.2 | 3 | 0.0275 ± 0.0019 | 69.6 | 578 | 21.4 (20.8–21.7) | 453 | 0.50;0.48;0.53 | 3/0/0 | 1.00 | 0.24 | 0.006 | – | 0/24 | 13 |
| temp p_t=0.1 β=0.3 f=0.2 T×3 | 3 | 0.00606 ± 0.0014 | 13.6 | 159 | 5 (4.6–5.66) | 92 | 0.49;0.45;0.56 | 3/0/0 | 1.00 | 0.13 | 0.018 | 0.000 | 0/24 | 14 |
| temp p_t=0.1 β=0.3 f=0.35 T×3 | 3 | 0.00696 ± 0.00081 | 15.5 | 81.6 | 5.03 (4.5–5.48) | 91 | 0.50;0.54;0.51 | 3/0/0 | 1.00 | 0.12 | 0.015 | 0.000 | 0/24 | 14 |
| temp p_t=0.1 β=0.1 f=0.2 T×3 | 3 | 0.00897 ± 0.00037 | 19.2 | 128 | 6.49 (5.58–6.96) | 122 | 0.54;0.55;0.57 | 3/0/0 | 1.00 | 0.15 | 0.026 | 0.000 | 0/24 | 15 |
| temp p_t=0.1 β=0.1 f=0.35 T×3 | 3 | 0.00918 ± 0.00075 | 20.2 | 129 | 6.69 (5.31–7.61) | 123 | 0.56;0.45;0.47 | 3/0/0 | 1.00 | 0.13 | 0.023 | 0.000 | 0/24 | 15 |
| temp p_t=0.1 β=0.03 f=0.2 T×3 | 3 | 0.00866 ± 0.00065 | 18.5 | 102 | 6.29 (5.21–7.26) | 121 | 0.44;0.51;0.46 | 3/0/0 | 1.00 | 0.17 | 0.043 | 0.000 | 0/24 | 15 |
| temp p_t=0.1 β=0.03 f=0.35 T×3 | 3 | 0.0103 ± 0.004 | 22.1 | 151 | 7.75 (6.29–10.5) | 145 | 0.53;0.49;0.54 | 3/0/0 | 1.00 | 0.15 | 0.037 | 0.000 | 0/24 | 15 |
| temp p_t=0.3 β=0.3 f=0.2 T×3 | 3 | 0.0088 ± 0.00031 | 15.5 | 221 | 8.02 (7.55–8.76) | 157 | 0.50;0.54;0.46 | 3/0/0 | 1.00 | 0.18 | 0.033 | 0.000 | 0/24 | 18 |
| temp p_t=0.3 β=0.3 f=0.35 T×3 | 3 | 0.00932 ± 0.00073 | 17.1 | 208 | 7.8 (7.46–8.1) | 147 | 0.47;0.54;0.55 | 3/0/0 | 1.00 | 0.15 | 0.027 | 0.000 | 0/24 | 18 |
| temp p_t=0.3 β=0.1 f=0.2 T×3 | 3 | 0.0158 ± 0.0014 | 26.3 | 249 | 12 (10.5–13.7) | 254 | 0.53;0.50;0.52 | 3/0/0 | 1.00 | 0.24 | 0.062 | 0.000 | 0/24 | 19 |
| temp p_t=0.3 β=0.1 f=0.35 T×3 | 3 | 0.0155 ± 0.0021 | 27.1 | 455 | 13.8 (10.7–15.5) | 273 | 0.50;0.48;0.45 | 3/0/0 | 1.00 | 0.19 | 0.053 | 0.000 | 0/24 | 18 |
| temp p_t=0.3 β=0.03 f=0.2 T×3 | 3 | 0.0159 ± 0.0042 | 25.1 | 190 | 11 (8.17–15.3) | 252 | 0.46;0.55;0.50 | 3/0/0 | 1.00 | 0.30 | 0.117 | 0.000 | 0/24 | 20 |
| temp p_t=0.3 β=0.03 f=0.35 T×3 | 3 | 0.0231 ± 0.0069 | 38.4 | 748 | 16.9 (13.9–22.3) | 358 | 0.48;0.46;0.51 | 3/0/0 | 1.00 | 0.25 | 0.103 | 0.000 | 0/24 | 19 |
| temp p_t=0.3 β=0.1 f=0.35 T×1 | 3 | 0.0127 ± 0.0014 | 27.1 | 182 | 8.6 (7.7–9.54) | 191 | 0.48;0.51;0.53 | 3/0/0 | 1.00 | 0.28 | 0.032 | 0.000 | 0/24 | 15 |


* All 45 runs pass. Every tempered cell crosses more than plain (5.0–16.9 vs 4.0 per 1 000); the best,
  (0.3, 0.03, 0.35), gives 16.9 (4.2× plain) and ESS/eval 0.023 (4.3× plain's 0.0054), close to the prior draw
  (21.4 crossings, ESS/eval 0.0275, 1.2× the best tempered).
* Monotone trends visible above the replicate spread: p_t = 0.3 > 0.1 (every β, f), β = 0.1/0.03 > 0.3.

### 2.3 M3-10 (Δ = 10 nats)

| configuration | runs | ESS/eval ± sd | ESS/s | mode ESS (ref) | crossings / 1 000 (min–max) | crossings / run | mode mass (reps) | ok/poor/failed | outer acc | pre-rej | oob | skipped | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 0.889 ± 0.15 | 1.83e+03 | 11 | 0.244 (0.113–0.393) | 4 | 0.79;0.80;0.69 | 0/3/0 | 1.00 | 0.10 | 0.009 | – | 15/24 | 16 |
| H + PCN(β=1) 0.8/0.2 | 3 | 0.0159 ± 0.0014 | 32.4 | 181 | 11.9 (11.3–12.2) | 255 | 0.48;0.45;0.47 | 3/0/0 | 1.00 | 0.26 | 0.006 | – | 0/24 | 16 |
| temp p_t=0.1 β=0.3 f=0.2 T×3 | 3 | 0.476 ± 0.28 | 913 | 12 | 0.4 (0.264–0.661) | 7 | 0.35;0.69;0.32 | 0/3/0 | 1.00 | 0.13 | 0.024 | 0.000 | 10/24 | 17 |
| temp p_t=0.1 β=0.3 f=0.35 T×3 | 3 | 0.525 ± 0.097 | 925 | 11.6 | 0.324 (0.272–0.424) | 6 | 0.27;0.62;0.69 | 0/3/0 | 1.00 | 0.13 | 0.035 | 0.000 | 11/24 | 18 |
| temp p_t=0.1 β=0.1 f=0.2 T×3 | 3 | 0.266 ± 0.26 | 473 | 16.8 | 0.682 (0.582–0.796) | 13 | 0.70;0.55;0.35 | 0/3/0 | 1.00 | 0.16 | 0.047 | 0.000 | 5/24 | 18 |
| temp p_t=0.1 β=0.1 f=0.35 T×3 | 3 | 0.236 ± 0.16 | 434 | 27.5 | 0.821 (0.417–1.29) | 15 | 0.39;0.52;0.66 | 1/2/0 | 1.00 | 0.15 | 0.046 | 0.000 | 5/24 | 18 |
| temp p_t=0.1 β=0.03 f=0.2 T×3 | 3 | 0.231 ± 0.28 | 425 | 27.9 | 0.693 (0.306–0.993) | 13 | 0.47;0.55;0.49 | 1/2/0 | 1.00 | 0.17 | 0.059 | 0.000 | 5/24 | 19 |
| temp p_t=0.1 β=0.03 f=0.35 T×3 | 3 | 0.126 ± 0.12 | 231 | 31.5 | 1.2 (0.633–2.3) | 23 | 0.47;0.45;0.47 | 1/2/0 | 1.00 | 0.16 | 0.054 | 0.000 | 3/24 | 18 |
| temp p_t=0.3 β=0.3 f=0.2 T×3 | 3 | 0.128 ± 0.22 | 197 | 51.7 | 0.907 (0.215–1.31) | 18 | 0.34;0.47;0.59 | 2/1/0 | 1.00 | 0.22 | 0.098 | 0.000 | 4/24 | 23 |
| temp p_t=0.3 β=0.3 f=0.35 T×3 | 3 | 0.14 ± 0.14 | 212 | 18.6 | 0.945 (0.372–1.44) | 18 | 0.62;0.48;0.42 | 1/2/0 | 1.00 | 0.19 | 0.080 | 0.000 | 4/24 | 22 |
| temp p_t=0.3 β=0.1 f=0.2 T×3 | 3 | 0.144 ± 0.24 | 211 | 42 | 1.32 (0.342–1.83) | 29 | 0.23;0.63;0.50 | 1/2/0 | 1.00 | 0.28 | 0.128 | 0.000 | 4/24 | 24 |
| temp p_t=0.3 β=0.1 f=0.35 T×3 | 3 | 0.0751 ± 0.12 | 117 | 69.2 | 2.12 (0.53–3.07) | 44 | 0.66;0.50;0.40 | 1/2/0 | 0.99 | 0.24 | 0.122 | 0.000 | 2/24 | 23 |
| temp p_t=0.3 β=0.03 f=0.2 T×3 | 3 | 0.15 ± 0.17 | 214 | 40.9 | 1.82 (0.243–3.06) | 42 | 0.51;0.30;0.53 | 2/1/0 | 1.00 | 0.32 | 0.173 | 0.000 | 5/24 | 25 |
| temp p_t=0.3 β=0.03 f=0.35 T×3 | 3 | 0.00884 ± 0.0014 | 12.1 | 100 | 4.17 (1.18–7.01) | 89 | 0.38;0.48;0.56 | 2/1/0 | 1.00 | 0.27 | 0.148 | 0.000 | 0/24 | 24 |
| temp p_t=0.3 β=0.1 f=0.35 T×1 | 3 | 0.00448 ± 0.00072 | 7.79 | 56.7 | 1.73 (1.54–1.97) | 39 | 0.45;0.59;0.51 | 3/0/0 | 1.00 | 0.30 | 0.061 | 0.000 | 0/24 | 19 |
| temp p_t=0.3 β=0.03 f=0.35 T×3 ε_t=0.078 (pinned) [follow-up] | 3 | 0.0143 ± 0.00024 | 15.9 | 234 | 14.6 (14.4–14.8) | 277 | 0.48;0.48;0.46 | 3/0/0 | 0.99 | 0.16 | 0.005 | 0.000 | 0/24 | 29 |
| temp p_t=0.3 β=0.001 f=0.35 T×3 [follow-up] | 3 | 0.0101 ± 0.0028 | 13 | 127 | 5.62 (1.43–8.83) | 135 | 0.47;0.42;0.50 | 3/0/0 | 1.00 | 0.34 | 0.234 | 0.000 | 0/24 | 25 |
| temp p_t=0.3 β=0.001 f=0.35 T×3 ε_t=0.078 (pinned) [follow-up] | 3 | 0.0469 ± 0.0026 | 47.3 | 1.73e+03 | 41.2 (39.9–42.4) | 883 | 0.50;0.50;0.48 | 3/0/0 | 0.99 | 0.25 | 0.031 | 0.000 | 0/24 | 32 |


* Plain H: 0.24 crossings per 1 000, 15/24 chains never cross, 3/3 poor (mode mass 0.69–0.80). Prior draw: 11.9,
  3/3 ok.
* Grid: crossings rise with p_t and with smaller β/larger f up to 4.17 (0.3, 0.03, 0.35; 17× plain, 0.35× prior
  draw), but **12 of the 36 T×3 grid runs pass** and no grid cell is 3/3 ok; the T×1 row is 3/3 ok with 1.73 crossings.
* **Follow-up, tempered step pinned at ½ the adapted step, β = 0.001: 41.2 crossings per 1 000 (39.9–42.4), 3/3 ok,
  mode ESS 1 730 (prior draw 181), ESS/eval 0.047 (prior draw 0.016), ESS/s 47 (prior draw 32).** With the step
  halved, oob falls from 0.15–0.23 to 0.005–0.03.

### 2.4 M3-20 (Δ = 20 nats)

| configuration | runs | ESS/eval ± sd | ESS/s | mode ESS (ref) | crossings / 1 000 (min–max) | crossings / run | mode mass (reps) | ok/poor/failed | outer acc | pre-rej | oob | skipped | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 1.21 ± 0.14 | 1.91e+03 | 8.87 | 0.0749 (0.0561–0.112) | 1 | 0.56;0.61;0.52 | 0/3/0 | 1.00 | 0.10 | 0.008 | – | 20/24 | 20 |
| H + PCN(β=1) 0.8/0.2 | 3 | 0.011 ± 0.0012 | 17.8 | 150 | 8.07 (7.75–8.3) | 175 | 0.52;0.47;0.43 | 3/0/0 | 1.00 | 0.26 | 0.005 | – | 0/24 | 20 |
| temp p_t=0.1 β=0.3 f=0.2 T×3 | 3 | 1.01 ± 0.13 | 1.48e+03 | 8.5 | 0.0542 (0–0.109) | 1 | 0.40;0.25;0.28 | 0/3/0 | 1.00 | 0.14 | 0.024 | 0.000 | 21/24 | 22 |
| temp p_t=0.1 β=0.3 f=0.35 T×3 | 3 | 0.887 ± 0.26 | 1.26e+03 | 9.04 | 0.106 (0.0511–0.211) | 2 | 0.73;0.38;0.43 | 0/3/0 | 1.00 | 0.15 | 0.052 | 0.000 | 18/24 | 23 |
| temp p_t=0.1 β=0.1 f=0.2 T×3 | 3 | 0.844 ± 0.2 | 1.18e+03 | 10.4 | 0.122 (0.0509–0.208) | 2 | 0.47;0.56;0.56 | 0/3/0 | 1.00 | 0.17 | 0.063 | 0.000 | 17/24 | 23 |
| temp p_t=0.1 β=0.1 f=0.35 T×3 | 3 | 0.823 ± 0.34 | 1.16e+03 | 10.9 | 0.122 (0.051–0.262) | 2 | 0.45;0.30;0.55 | 0/3/0 | 1.00 | 0.16 | 0.060 | 0.000 | 17/24 | 23 |
| temp p_t=0.1 β=0.03 f=0.2 T×3 | 3 | 0.67 ± 0.46 | 975 | 13 | 0.242 (0.0508–0.519) | 5 | 0.65;0.31;0.49 | 1/2/0 | 1.00 | 0.18 | 0.067 | 0.000 | 14/24 | 22 |
| temp p_t=0.1 β=0.03 f=0.35 T×3 | 3 | 0.867 ± 0.31 | 1.24e+03 | 9.92 | 0.138 (0–0.209) | 3 | 0.59;0.25;0.60 | 0/3/0 | 1.00 | 0.17 | 0.072 | 0.000 | 18/24 | 23 |
| temp p_t=0.3 β=0.3 f=0.2 T×3 | 3 | 0.673 ± 0.13 | 768 | 9.55 | 0.0664 (0–0.152) | 1 | 0.63;0.74;0.29 | 0/3/0 | 0.99 | 0.23 | 0.079 | 0.000 | 20/24 | 28 |
| temp p_t=0.3 β=0.3 f=0.35 T×3 | 3 | 0.462 ± 0.23 | 512 | 13 | 0.207 (0.0906–0.409) | 5 | 0.37;0.50;0.74 | 1/2/0 | 0.99 | 0.30 | 0.186 | 0.000 | 13/24 | 29 |
| temp p_t=0.3 β=0.1 f=0.2 T×3 | 3 | 0.466 ± 0.14 | 514 | 10.5 | 0.2 (0.0795–0.345) | 4 | 0.54;0.51;0.79 | 1/2/0 | 1.00 | 0.29 | 0.144 | 0.000 | 14/24 | 29 |
| temp p_t=0.3 β=0.1 f=0.35 T×3 | 3 | 0.379 ± 0.14 | 422 | 12.5 | 0.337 (0.161–0.521) | 8 | 0.32;0.46;0.61 | 0/3/0 | 0.99 | 0.30 | 0.186 | 0.000 | 11/24 | 30 |
| temp p_t=0.3 β=0.03 f=0.2 T×3 | 3 | 0.506 ± 0.17 | 567 | 10.9 | 0.15 (0.0399–0.207) | 4 | 0.38;0.57;0.57 | 0/3/0 | 1.00 | 0.35 | 0.234 | 0.000 | 15/24 | 30 |
| temp p_t=0.3 β=0.03 f=0.35 T×3 | 3 | 0.203 ± 0.26 | 224 | 26.7 | 0.639 (0.159–1.01) | 15 | 0.59;0.67;0.63 | 1/2/0 | 1.00 | 0.34 | 0.248 | 0.000 | 6/24 | 31 |
| temp p_t=0.3 β=0.1 f=0.35 T×1 | 3 | 0.189 ± 0.16 | 252 | 21 | 0.53 (0.423–0.725) | 13 | 0.38;0.46;0.72 | 1/2/0 | 1.00 | 0.32 | 0.120 | 0.000 | 4/24 | 25 |
| temp p_t=0.3 β=0.03 f=0.35 T×3 ε_t=0.061 (pinned) [follow-up] | 3 | 0.00442 ± 0.00099 | 4.02 | 48.9 | 2.11 (2.02–2.21) | 40 | 0.54;0.66;0.57 | 2/1/0 | 0.99 | 0.15 | 0.005 | 0.000 | 0/24 | 35 |
| temp p_t=0.3 β=0.001 f=0.35 T×3 [follow-up] | 3 | 0.203 ± 0.27 | 236 | 30 | 0.668 (0.159–1.36) | 17 | 0.47;0.54;0.61 | 1/2/0 | 1.00 | 0.36 | 0.273 | 0.000 | 6/24 | 30 |
| temp p_t=0.3 β=0.001 f=0.35 T×3 ε_t=0.061 (pinned) [follow-up] | 3 | 0.0366 ± 0.003 | 29.5 | 748 | 31.6 (29.4–33.9) | 669 | 0.49;0.50;0.54 | 3/0/0 | 0.99 | 0.25 | 0.025 | 0.000 | 0/24 | 40 |


* Plain H: 0.075 per 1 000, 20/24 chains never cross, 3/3 poor. Prior draw: 8.1, 3/3 ok.
* Grid: best 0.64 per 1 000 (0.3, 0.03, 0.35; 8.5× plain, 0.08× prior draw); 4 of the 36 T×3 grid runs pass; 6–21 of
  24 chains never cross in every grid cell. The estimated oob share of the tempered proposals is 0.24–0.81 at
  p_t = 0.3 (same attribution as §2.1).
* **Follow-up, ½ step, β = 0.001: 31.6 per 1 000 (29.4–33.9), 3/3 ok, mode ESS 748 (prior 150), ESS/eval 0.037
  (prior 0.011), ESS/s 29.5 (prior 17.8)**, wall 40 s vs 20 s. ½ step with β = 0.03: 2.1 per 1 000, 2/3 ok.
  β = 0.001 at the carried (full) step: 0.67 per 1 000, 1/3 ok (oob 0.27) — both changes are needed.

### 2.5 Exact-gradient diagnostic and why the follow-up was run (`scripts/diag_exact.py`)

Local, no surrogate: the library's runtime `HamiltonianInfinite` (dimension_robust) with the EXACT gradient of
−log L, mass = inverse reference covariance (B2: covariance of the prior-draw runs' samples), 1 500 proposals from
states in one mode, at the geometric-mean adapted step ε of the grid runs and at ε/2, ε/4. Per proposal:
P(end out of box), mean exact acceptance, and E[acceptance · ends in the other mode] (= crossings per proposal of an
exact-MH chain using only that proposal). Full table: `logs/diag_exact.txt`, `diag_exact.json`.

| problem | proposal | P(oob) at ε / ε/2 | mean acc at ε / ε/2 | crossings per proposal at ε / ε/2 |
|---|---|---|---|---|
| M3-10 | H T = 3 (untempered) | 0.030 / 0.000 | 0.81 / 0.97 | 0.0002 / 0.0000 |
| M3-10 | temp T = 9, f 0.35, β 0.1 | 0.33 / 0.000 | 0.50 / 0.82 | 0.0038 / 0.0085 |
| M3-10 | temp T = 9, f 0.35, β 0.03 | 0.51 / 0.000 | 0.32 / 0.73 | 0.0062 / 0.046 |
| M3-10 | temp T = 9, f 0.35, β 0.001 | 0.87 / 0.10 | 0.05 / 0.40 | 0.0041 / 0.126 |
| M3-20 | H T = 3 (untempered) | 0.10 / 0.000 | 0.74 / 0.97 | 0.0012 / 0.0000 |
| M3-20 | temp T = 9, f 0.35, β 0.03 | 0.64 / 0.000 | 0.26 / 0.74 | 0.0019 / 0.0074 |
| M3-20 | temp T = 9, f 0.35, β 0.001 | 0.92 / 0.063 | 0.03 / 0.43 | 0.0007 / 0.094 |
| B2 | H T = 3 (untempered) | 0.98 / 0.85 (ε/4: 0.12) | 0.001 / 0.047 | 0 / 0 |
| B2 | temp T = 9, f 0.35, β 0.03 | 1.00 / 1.00 (ε/4: 0.83) | 0 / 0.0004 | 0 / 0.0004 |

Reading: (a) the step the dual averaging gives (tuned on the surrogate sub-chain with the error model at target
0.8–0.9) is near the stability limit of the exact dynamics already untempered (M3-20: 10 % out of box); a heated
trajectory moves faster and leaves the box in 19–92 % of the proposals at that step (β 0.3 → 0.001); at ε/2 almost none do.
(b) At ε/2 the crossing rate per tempered proposal grows 2.7× from β = 0.03 to 0.001 on M3-10 and 13× on M3-20; the
grid's warmest-to-coldest β range (0.3–0.03) is too warm for Δ = 20. (c) B2's exact dynamics are unstable at the
adapted step even untempered; the DAMH runs survive on B2 only because the surrogate (with error model) is smoother.
These predictions motivated the 27 follow-up runs (p_t = 0.3, f = 0.35, T×3; β = 0.001 at the carried step; β = 0.03
and 0.001 with the step pinned at ½ of the grid's geometric-mean adapted step: B2 0.235, M3-10 0.078, M3-20 0.061).
The DAMH runs confirm (a) and (b) on M3 and the B2 failure. Exact-MH numbers here are not DAMH numbers (the DAMH
trajectory runs on network gradients); they explain the mechanism, not the rates.

---

## 3. Cross-problem answers

Headline (crossings per 1 000 stage-2 iterations, mean of 3 runs; verdict ok/3; mode ESS (ref)):

| problem | plain H | H + PCN(β=1) | best grid tempered (cell) | follow-up: β 0.001, ½ step |
|---|---|---|---|---|
| B2 (11 nats, d = 2) | 0.66, 0/3, 11 | **6.8, 3/3, 216** | 1.13, 0/3, 8 (0.3, 0.03, 0.2) | 0.55, 0/3, 12 |
| M3-5 | 4.0, 3/3, 95 | **21.4, 3/3, 578** | 16.9, 3/3, 748 (0.3, 0.03, 0.35) | not run |
| M3-10 | 0.24, 0/3, 11 | 11.9, 3/3, 181 | 4.17, 2/3, 100 (0.3, 0.03, 0.35) | **41.2, 3/3, 1 730** |
| M3-20 | 0.075, 0/3, 9 | 8.1, 3/3, 150 | 0.64, 1/3, 27 (0.3, 0.03, 0.35) | **31.6, 3/3, 748** |

Figure: `figures/fig_crossings_vs_depth.png` (crossings per 1 000 vs Δ for plain, prior draw, best grid cell,
both β = 0.001 follow-ups; log axis, min–max bars).

**(i) Does the tempered component increase crossings, and does it scale with the barrier?** On M3, yes: the best grid
cell gives 4.2× / 17× / 8.5× plain's crossings at Δ = 5 / 10 / 20. But within the specified grid it does **not**
scale with the barrier: from Δ = 5 to 20 the best grid cell falls 26× (16.9 → 0.64), about like plain (54×), while
the prior draw falls only 2.7× (21.4 → 8.1) and beats the grid's best at every depth (1.3× / 2.9× / 13×). With the
two follow-up changes (β = 0.001 and the tempered step at ½ the adapted step) the component falls only 1.3× from
Δ = 10 to 20 (41.2 → 31.6), crosses 3.5× / 3.9× more often than the prior draw and 170× / 420× more than plain, and
all 6 runs pass. On B2 it does nothing at any setting tried (§2.1: ≈ all tempered proposals leave the box).
Caveat: d ≤ 3 here, where the prior draw is at its best; the dimension-scaling question of note 30 (prior draw
collapsing with d) is **not tested**.

**(ii) Cost.** Exact-evaluation cost is nil by construction (one exact evaluation per non-pre-rejected iteration,
whichever component proposes). The costs are (a) pre-rejections: the tempered iterations that leave the box are lost
iterations (pre-rej 0.10 → 0.15–0.35 on M3 at p_t = 0.3 in the grid; 0.15–0.25 with ½ step); (b) network gradients:
T×3 at p_t = 0.3 roughly doubles the gradients per iteration, ½ step doubles the tempered part again; wall time per
run 16 s (plain) → 22–25 s (M3-10 grid, p_t = 0.3) → 29–32 s (½ step); on M3-20 20 → 28–31 → 35–40 s. On rows that pass, per evaluation and per
second the ½-step β = 0.001 setting is the most efficient sampler of the study on M3-10/20: ESS/eval 0.047 / 0.037
vs prior draw 0.016 / 0.011 (2.9× / 3.3×), ESS/s 47 / 29.5 vs 32 / 17.8 (1.5× / 1.7×). On M3-5 the best grid cell
reaches ESS/eval 0.023 and ESS/s 38 vs the prior draw's 0.0275 and 70 (0.84× / 0.55×). Plain H's large ESS/eval on
M3-10/20/B2 is not efficiency (chains do not mix; § Read first).

**(iii) Best (p_t, β, f) and a single setting.** Within the grid the same cell is best by crossings on all three M3
depths: **p_t = 0.3, β = 0.03, f = 0.35** (on B2 nothing works; its nominal best (0.3, 0.03, 0.2) is within the
replicate spread of plain). Trends: p_t 0.3 > 0.1 everywhere on M3; smaller β better on M3-10/20; f matters less
than β. Only at Δ = 5 is the grid's best competitive. Recommendation for the C6 placeholders of note 30 D-C3-3
(`p_t = 0.2, β = 0.1, f = 0.3, T_t = 3T`), from these runs:

| placeholder | note 30 | recommended | evidence |
|---|---|---|---|
| p_t | 0.2 | 0.3 | p_t = 0.3 > 0.1 in every M3 cell; values > 0.3 not tried |
| β | 0.1 | **0.001** (with ½ step) | β 0.1 → 2.1 (M3-10) / 0.34 (M3-20) crossings per 1 000 in the grid, β 0.001 + ½ step → 41 / 32; exact diag §2.5 |
| f | 0.3 | 0.35 | crossings at f 0.35 ≥ f 0.2 in 15 of 18 M3 (p_t, β) pairs; differences mostly within spread |
| T_t | 3T | 3T (weak) | (iv) |
| tempered step | the carried adapted step | **½ of the carried step** | estimated oob of tempered proposals 0.24–0.81 at the carried step (M3-20 grid, p_t 0.3), total oob 0.025 at ½ step; β 0.001 at the full step: 1/3 ok on M3-20 |

The last row is not expressible today: a pinned component takes either the carried step (`step_size=None`) or an
absolute number; the follow-up used absolute numbers (½ of the grid's geometric-mean step per problem). A relative
step (e.g. a factor on the carried step) would be needed for Auto. **The recommended setting is supported by 6 runs
(M3-10, M3-20) and fails on B2 (3/3 poor); it was not run on M3-5, nor at other p_t or f.** It does not replace the
prior draw in d ≤ 3 on B2-like posteriors (the tempered proposals leave the box there at every setting tried).

**(iv) T×1 vs T×3** at (0.3, 0.1, 0.35): M3-5 8.6 vs 13.8, M3-10 1.73 vs 2.12, M3-20 0.53 vs 0.34, B2 0.57 vs 0.55
crossings per 1 000 (ranges overlap except M3-5). T×1 has fewer out-of-box proposals (M3-10 oob 0.061 vs 0.122,
M3-20 0.120 vs 0.186) and passes more often (M3-10 3/3 vs 1/3, M3-20 1/3 vs 0/3) at lower wall time. The longer T
pays only at Δ = 5 in this grid; whether it pays with the ½ step was not tested.

**(v) Failed, hung or poor runs.** No run failed, hung or hit the 300 s limit (`stage{1,2}_time_limit_hit` all
false; non-finite proposals 0). Poor runs: grid 104 of 180 (B2 40/45, M3-10 27/45, M3-20 37/45, M3-5 0/45);
follow-up 12 of 27 (B2 9/9, M3-20 3/9). The mode mass is off by > 0.1 in 73 of the 116 poor runs (B2 0.09–0.67, M3
0.23–0.80); the other 43 fail on the mean (> 4 ESS-SE) with the mode mass within 0.1 (M3: E x² is never off by
> 4 SE). Poor runs cross at most 2.8 times per 1 000 iterations and 111 of 116 have at least one chain that
never crossed: mixing failures. Conversely some "ok" runs cross only 0.08–0.3 times per 1 000 (stuck chains split
about evenly between the modes by chance), so ok-counts in cells below ~3 crossings per 1 000 are not evidence of
mixing; the mode ESS (ref) column is the better guide there. Plain H is poor on B2, M3-10, M3-20 in every repetition. The
`RuntimeWarning: overflow encountered in cast` from `surrogates/torch_perceptron_minibatches.py` appears in 154 of the
180 grid logs (pre-existing network-training warning; not investigated).

---

## 4. Caveats

* 3 repetitions; small crossing counts; DAMH-SMU not reproducible.
* Crossings are not attributed to components (no per-component column in `subchain_stats`).
* d ≤ 3 only: the "scales with dimension" claim of note 30 (prior draw collapses with d, tempering does not) is
  untested; in d ≤ 3 the prior draw is strong.
* The ½-step and β = 0.001 settings are follow-ups outside the specification, 3 runs each, on 3 of the 4 problems;
  the ½ step is an absolute number per problem, not ½ of each run's own adapted step.
* Error model on everywhere; stage-2 budget 1 000 per chain (the adaptive component's dual averaging restarts in
  stage 3 when the covariance changed, mostly on B2; effect not measured).
* The exact-gradient diagnostic uses a reference-covariance mass and exact gradients; the DAMH runs use the pooled
  run covariance and network gradients.
* Items for `10_manual_review_notes.md` (not written there by this study): (a) a pinned tempered component at the
  carried step mostly leaves the prior box (M3 0.16–0.81, B2 ≈ 1.0 of tempered proposals); a relative step for pinned
  components is needed for C6; (b) the `Hamiltonian` docstring's "enough to climb a barrier of order log(1/β) nats"
  (β = 0.03 → 3.5 nats) understates what the runs show (crossings of 10–20-nat valleys at β = 0.03–0.001); the
  relation between β and the barrier is not settled by this study; (c) min-over-parameters ESS/eval rewards
  non-mixing runs on multimodal problems (study 29's metric; use a between-chain-aware measure).

## 5. Paths

`toy_examples/tmp_c3_study_2026-10-12/`:
* `runs/<name>/` (format-v2 output + `study_case.json` + `metrics.json`), `runs/<name>.log` (progress lines stripped);
  names `B2_*`, `M3-{5,10,20}_*` (grid), `x_*` (follow-up);
* `jobs/grid.jsonl` (180), `jobs/extra.jsonl` (27);
* `refs/ref_B2.json` (copy of study 29's), `refs/ref_M3-{5,10,20}.json`;
* `table_all.csv` (per run), `table_configs.csv` (per cell), `tables.md` (the §2 tables), `diag_exact.json`;
* `figures/fig_crossings_vs_depth.png`;
* `logs/time_log.txt`, `logs/batch_progress.txt`, `logs/diag_exact.txt` (no `failed.txt`: no failures);
* `scripts/{problems,run_case,batch,jobs,analyze,collect,summary,diag_exact,figures}.py` (adapted from study 29;
  `figures.py` is study 29's, unused).
