# 33 — Flattened-likelihood vs tempered-momentum Hamiltonian component: comparison study (2026-10-13)

**Dated record of one campaign, written 2026-10-13 from finished runs only.** The study of note 32 §2, run
2026-10-13 15:57–18:42 UTC on the uncommitted working tree of `working_Kuba` (the `likelihood_weight` option and the
`subchain_stats` `component` column as implemented that day; the last package-code change was 15:42 UTC, before the
first run; no package file was changed by this study). 252 runs (14 configurations × 6 problems × 3 repetitions), all
return code 0, none hit the 300 s time limit, no non-finite proposals. Raw runs, drivers, tables, figures:
`toy_examples/tmp_flattened_study_2026-10-13/` (§5).

**Result first.** The flattened component with a small likelihood weight — `τ = 0.05` (or `0`), `T_t = T`, the carried
step, `p = 0.3` — is the only mechanism of the study that mixes on **all six problems (18/18 runs "mixed")**, with
23–75 crossings per 1 000 measured-stage iterations, 90–100 % of them made by the flattened component itself. The
tempered-momentum reference (study 31's best) fails on B2 (0.3 per 1 000, 0/3) and drops 20× from d = 3 to d = 30
(42.6 → 2.2, 0/3 mixed at d = 30); the prior draw collapses with d as predicted (11.8 → 0.95 → 0.05 at Δ = 10). The
flattened component does **not** need the halved step. Combining it with tempering never beats the flattened rows. The
mechanism is visible analytically on these problems (§3 (i)): at `τ ≤ d/(2Δ + d)` the flattened potential has no
barrier at all; the crossing rate of every flattened row follows that barrier height. Because the M-family's barrier
is largely made by the prior (the modes sit in the prior's tail at radius √d), **how far the recommendation carries to
problems whose barrier is made by the likelihood alone was not tested** (§4).

**Read first (limits of what was measured):**

* **Deviation from the brief — the half step of M3-5 and of the new problems.** The brief said to halve the mean of
  study 31's `step_end_stage2` column (plain runs) for M3-5. In study 31's `analyze.py` that column is the step at the
  end of the *measured* stage; the step the pinned component carries (and that study 31's 0.235 / 0.078 / 0.061 are
  half of) is the column `step_end_stage1` (end of the second stage). I halved the **carried** step, to keep "½ carried
  step" consistent with the three study-31 numbers: M3-5 0.1876 / 2 → **0.094** (the literal reading gives 0.087).
  Likewise for M10-10 / M30-10: half of the mean carried step of this study's three plain runs, **0.0629 / 0.0645**
  (the end-of-measured-stage means would give 0.0666 / 0.0647). All numbers in `half_steps.json`.
* **3 repetitions per cell**; ranges (min–max over reps) are given. Only several-fold differences are discussed.
* **The "½ step" rows pin an ABSOLUTE step per problem** (no relative-step option); in the individual runs it is
  0.43–0.57 of that run's own carried step on the M problems and 0.22–0.91 on B2 (B2's carried step varies 4× between
  runs, as in study 31).
* **DAMH-SMU runs are not reproducible** (asynchronous network delivery); repetitions use `seed_offset = 100 000 · rep`
  and `NeuralNetworkUpdater(seed=rep)`.
* **Cheap models**: an exact evaluation costs microseconds; "per evaluation" and "per second" rank differently. Wall
  times are **inflated by other load on the host during 16:00–17:10 UTC** (load average up to 27 on 32 cores; M3-10
  and M10-10 blocks): M10-10 plain took 48 s vs 45 s for M30-10 plain; ESS/s and s/crossing on M3-10 / M10-10 are
  therefore pessimistic and noisy (max single run 137 s, M10-10 combined r0). Every d = 30 run took ≤ 49 s, so the
  d = 30 budget was **not** reduced (4 000 per chain everywhere).
* **Error model on, Polyak on** in every DAMH stage (as study 31). The adaptive component's dual averaging restarted in
  stage 3 ("mass changed") in all M10-10 and M30-10 runs, 29/42 B2, 10/42 M3-20, 0 M3-5/M3-10 (manifest).
* **The min-over-parameters ESS/eval rewards non-mixing runs** (plain H has the highest ESS/eval on every problem
  with 0/3 mixed). It is reported, not used for ranking; the verdict columns are "mixed" (R̂_mode < 1.05 and every
  chain crossed) and mode ESS (ref).
* **The M-family is symmetric** under u → −u around the prior mean, and its two modes sit at the same prior radius;
  τ ≈ 0 makes the flattened component close to a prior rotation, which connects such modes. B2 (asymmetric, d = 2) is
  the only asymmetric problem here.

---

## 1. Setup

### 1.1 Problems (`scripts/problems.py`, `refs/`)

| name | problem | d | network | `min_snapshots_initial` | held-out | reference |
|---|---|---|---|---|---|---|
| B2 | study 29's `Solver_illustrative_local`, prior N(0, I₂), likelihood N(5, 1); left arm holds 0.2933 | 2 | (32, 32) | 20 | 200 | `refs/ref_B2.json` (copied from study 31) |
| M3-5 / M3-10 / M3-20 | `bimodal3d.make_problem(Δ)`, Δ = 5 / 10 / 20 nats | 3 | (32, 32) | 30 | 200 | `refs/ref_M3-{5,10,20}.json` (copied from study 31) |
| M10-10 | `bimodal3d.make_problem(10, no_parameters=10)`: modes ±(1,…,1), t_mode = √10 | 10 | (32, 32) | 100 | 200 | `refs/ref_M10-10.json` = `bimodal3d.reference(10, no_parameters=10)` (sd per coordinate 1.368) |
| M30-10 | `bimodal3d.make_problem(10, no_parameters=30)`, t_mode = √30 | 30 | (64, 64) | 300 | 200 | `refs/ref_M30-10.json` (sd 1.393) |

Mode indicator / crossings: M problems `t = u·1/√d`, mode = `1[t > 0]`, crossings with `bimodal3d.crossings`'
hysteresis ±0.5 on t; B2 study 29's arm label on y − x (hysteresis 0.75 / 1.25), mode = left arm.

Every run (`scripts/run_case.py`, exactly study 31's layout): `mpiexec -n 9` (8 samplers + 1 collector), in-process
solver, `use_solvers_pool=False`, `initial_sample_type="lhs"`, `min_snapshots_initial = 10 d`,
`seed_offset = 100 000 · rep`, `save_snapshots_to_file=False`, `NeuralNetworkUpdater(hidden_layer_sizes, seed=rep)`,
`TestData.generate(problem, 200)`, default prior box ±8. Stages, budget 4 000 exact evaluations per chain:
1. `Stage(proposal=RandomWalk(), max_evaluations=1000, burn_in=True)`;
2. `Stage(algorithm="DAMH", proposal=H_ADAPT, max_evaluations=1000, burn_in=True, surrogate_model_updates=True,
   polyak_averaging=True, surrogate_error_model=True, time_limit=300)`, `H_ADAPT = Hamiltonian(integrator=
   "dimension_robust", mass="covariance", integration_time=3, integration_time_jitter=0.5, target_rate=0.9)` (B2 0.8);
3. measured stage, same flags, `max_evaluations=2000`, proposal per configuration (all mixtures `[H_ADAPT, component]`,
   component = `Hamiltonian(integrator="dimension_robust", mass="covariance", integration_time=T_t,
   integration_time_jitter=0.5, adaptive=False, ...)` unless stated):

| label | component | p |
|---|---|---|
| plain H | none (`H_ADAPT` alone) | – |
| prior draw | `PCN(beta=1.0)` | 0.2 |
| tempered ref | `tempered_fraction=0.35, tempered_beta=0.001`, `T_t = 9`, step = ½ (absolute) | 0.3 |
| flat τ T_t carried (8) | `likelihood_weight=τ`, τ ∈ {0.5, 0.2, 0.05, 0}, `T_t ∈ {3, 9}`, `step_size=None` (carried) | 0.3 |
| flat τ ½ step (2) | `likelihood_weight=τ`, τ ∈ {0.2, 0.05}, `T_t = 9`, step = ½ (absolute) | 0.3 |
| combined | `likelihood_weight=0.2, tempered_fraction=0.35, tempered_beta=0.03`, `T_t = 9`, step = ½ | 0.3 |

Half steps used (absolute `step_size` of the component; `half_steps.json`):

| problem | half step | source |
|---|---|---|
| B2 | 0.235 | study 31 follow-up |
| M3-5 | 0.094 | study 31 `table_all.csv`, plain runs, mean `step_end_stage1` (= carried step) 0.1876 / 2 (deviation, see Read first) |
| M3-10 | 0.078 | study 31 follow-up |
| M3-20 | 0.061 | study 31 follow-up |
| M10-10 | 0.0629 | this study's plain runs, carried steps 0.1245 / 0.1280 / 0.1249, mean / 2 |
| M30-10 | 0.0645 | this study's plain runs, carried steps 0.1384 / 0.1285 / 0.1201, mean / 2 |

Checked in every run's manifest: the carried-step rows' component took exactly the stage-2 carry-over step (144/144
runs), the half-step rows exactly the pinned number (72/72), and `component1.likelihood_weight` equals the configured τ.

### 1.2 Metrics (`scripts/analyze.py`, `scripts/summary.py`)

Study 31's metrics (measured stage = stage index 2; ESS/eval = min-over-parameters per-chain Geyer ESS summed over
chains / all exact evaluations of the run incl. 200 held-out; ESS/s = same ESS / wall; posterior verdict ok/poor/failed:
"poor" if a mean or second moment is off by > 4 ESS-SE or the mode mass by > 0.1; mode ESS (ref) = `n_chains · p(1−p) /
mean_c (frac_c − p)²`), plus:

* **R̂_mode**: classic Gelman–Rubin on the mode indicator, each chain split in 2 halves (16 sequences); `inf` when all
  sequences are constant but differ. **mixed** = R̂_mode < 1.05 and all 8 chains crossed ≥ 1 time.
* **Per component** from the `subchain_stats` `component` column (no exact steps in this study, so every row is a
  delayed-acceptance row): share of iterations, sub-chain acceptance (`subchain_acceptance_rate` mean; sub-chain length
  1, so the share of iterations whose surrogate test passed), outer acceptance (accepted / iterations with a moved
  sub-chain), pre-rejection.
* **Crossings attributed to the component of the switching iteration.** Alignment: the measured stage's initial state
  is carried over (its first `samples` row drops the leading +1), so the decompressed chain has exactly one state per
  outer iteration and state j is the state *after* `subchain_stats` row j of that rank (`rank_world` ↔ the run's
  `chain_ranks`). A crossing (hysteresis label change between states j−1 and j) is attributed to row j's `component`.
  **Verified in every one of the 234 mixture runs, every chain**: the row count equals the chain length, and state j
  equals state j−1 exactly iff row j has `outer_accepted = 0` (0 mismatches; `align_*` columns in `table_all.csv`).
  A crossing whose hysteresis band (|t| < 0.5) was entered in an earlier iteration is attributed to the iteration that
  left the band (rare with these steps; not separated).
* **Cost**: nominal gradients per iteration = share₀ · mean L of `H_ADAPT` (its `num_steps` in `adaptive_stats`) +
  share₁ · `T_t / ε_component` (0 for PCN); measured-stage exact evaluations, nominal gradients and wall seconds per
  crossing (cell totals / cell crossings). `tempered_skipped` was 0 in every tempered run.

### 1.3 Timeline (`logs/time_log.txt`, `logs/batch_progress.txt`)

| time (UTC) | event |
|---|---|
| 15:53 | start; scripts adapted from study 31 |
| 15:54–15:56 | smoke runs (M3-10 flat τ 0.2 T×3, 55 s, alignment 0 mismatches; M30-10 plain, 44 s) — kept in `smoke/`, not in the tables |
| 15:57–16:03 | phase 1: plain H × 3 on M10-10 and M30-10 → half steps |
| 16:03–18:42 | phase 2: 246 runs in the order M3-10, M10-10, M30-10, B2, M3-20, M3-5 |
| 18:43–18:55 | re-analysis of all 252 runs with the final `analyze.py` (0 errors), tables, figures |

Sum of run wall times 130 min.

---

## 2. Results per problem

Rows: the two baselines, the tempered reference, the 8 carried-step flattened rows, the 2 half-step flattened rows,
the combined row. First table per problem: mixing and efficiency ("sub-chain acc" / "outer acc": `H_ADAPT` /
the second component). Second table: per-component attribution and cost ("crossings made by comp." = share of all
crossings made in iterations of the second component; "own iters" = crossings per 1 000 iterations of that component).
Full per-run data `table_all.csv`, per-cell `table_configs.csv`.

### 2.1 B2

| configuration | runs | crossings / 1 000 (min–max) | mode mass (reps) | R̂_mode (reps) | mixed | ok/poor/failed | mode ESS (ref) | ESS/eval | ESS/s | sub-chain acc H_ADAPT / comp. | outer acc H_ADAPT / comp. | oob | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 1.28 (0.948–1.6) | 0.54;0.31;0.41 | 2.11;1.70;3.16 | 0/3 | 1/2/0 | 10.3 | 0.167 | 291 | – / – | – / – | 0.049 | 7/24 | 18 |
| prior draw PCN(β=1) p=0.2 | 3 | 7.32 (6.18–7.89) | 0.31;0.33;0.31 | 1.02;1.05;1.06 | 2/3 | 3/0/0 | 248 | 0.0088 | 15.4 | 0.79 / 0.081 | 0.953 / 0.945 | 0.041 | 0/24 | 18 |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 3 | 0.303 (0.0695–0.596) | 0.20;0.42;0.13 | 2.29;3.11;5.09 | 0/3 | 0/3/0 | 11.6 | 0.248 | 348 | 0.8 / 3.9e-05 | 0.954 / 0.333 | 0.334 | 15/24 | 23 |
| flat τ=0.5 T_t=T carried | 3 | 1.07 (0.853–1.46) | 0.31;0.28;0.35 | 1.27;1.70;1.78 | 0/3 | 2/1/0 | 24.2 | 0.105 | 166 | 0.8 / 0.039 | 0.956 / 0.97 | 0.267 | 6/24 | 21 |
| flat τ=0.5 T_t=3T carried | 3 | 0.407 (0.211–0.521) | 0.43;0.17;0.25 | 2.00;2.09;5.38 | 0/3 | 0/3/0 | 11.7 | 0.185 | 269 | 0.8 / 0.0013 | 0.949 / 0.646 | 0.332 | 11/24 | 23 |
| flat τ=0.2 T_t=T carried | 3 | 3.52 (3.35–3.81) | 0.28;0.30;0.32 | 1.07;1.06;1.07 | 0/3 | 3/0/0 | 124 | 0.0057 | 9.29 | 0.8 / 0.17 | 0.953 / 0.95 | 0.123 | 0/24 | 20 |
| flat τ=0.2 T_t=3T carried | 3 | 1.98 (0.766–3.44) | 0.28;0.35;0.44 | 1.07;1.24;1.81 | 0/3 | 2/1/0 | 56.5 | 0.0307 | 46.9 | 0.8 / 0.1 | 0.931 / 0.92 | 0.242 | 1/24 | 23 |
| flat τ=0.05 T_t=T carried | 3 | 22.7 (16.6–26.9) | 0.28;0.32;0.33 | 1.01;1.01;1.02 | 3/3 | 3/0/0 | 2e+03 | 0.0274 | 45.8 | 0.8 / 0.13 | 0.953 / 0.934 | 0.049 | 0/24 | 19 |
| flat τ=0.05 T_t=3T carried | 3 | 14.4 (3.48–20.9) | 0.30;0.32;0.27 | 1.01;1.02;1.05 | 2/3 | 3/0/0 | 674 | 0.0177 | 25.2 | 0.8 / 0.14 | 0.949 / 0.931 | 0.096 | 0/24 | 22 |
| flat τ=0 T_t=T carried | 3 | 26.4 (25.1–28.1) | 0.29;0.28;0.30 | 1.01;1.01;1.00 | 3/3 | 3/0/0 | 912 | 0.0338 | 55.6 | 0.8 / 0.17 | 0.95 / 0.931 | 0.036 | 0/24 | 20 |
| flat τ=0 T_t=3T carried | 3 | 18.6 (16.2–20.6) | 0.26;0.28;0.28 | 1.02;1.01;1.01 | 3/3 | 3/0/0 | 923 | 0.0231 | 34.3 | 0.8 / 0.18 | 0.95 / 0.933 | 0.036 | 0/24 | 22 |
| flat τ=0.2 T_t=3T ½ step | 3 | 5.38 (3.04–7.73) | 0.34;0.33;0.31 | 1.12;1.09;1.08 | 0/3 | 3/0/0 | 78.8 | 0.00737 | 10.4 | 0.8 / 0.3 | 0.943 / 0.945 | 0.078 | 0/24 | 23 |
| flat τ=0.05 T_t=3T ½ step | 3 | 20 (19.5–20.9) | 0.28;0.26;0.28 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 894 | 0.0246 | 32.2 | 0.8 / 0.2 | 0.95 / 0.941 | 0.041 | 0/24 | 25 |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 3 | 4.97 (4.26–5.79) | 0.32;0.29;0.24 | 1.05;1.06;1.08 | 0/3 | 3/0/0 | 113 | 0.00866 | 11.3 | 0.8 / 0.12 | 0.947 / 0.946 | 0.240 | 0/24 | 25 |

| configuration | comp. share | crossings made by comp. (share) | crossings / 1 000 own iters: H_ADAPT / comp. | ε comp. | nominal L comp. | mean L H_ADAPT | grad / iter | measured-stage exact evals / crossing | grads / crossing | s / crossing | pre-rej | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | – | – | – / – | – | – | 32.7 | 32.7 | 623 | 2.55e+04 | 0.72 | 0.20 | – |
| prior draw PCN(β=1) p=0.2 | 0.2 | 0.86 | 1.24 / 31.8 | – | 0 | 32.2 | 25.8 | 89.2 | 3.53e+03 | 0.103 | 0.35 | – |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 0.3 | 0 | 0.432 / 0 | 0.235 | 38.3 | 31.2 | 33.3 | 1.85e+03 | 1.1e+05 | 2.65 | 0.44 | 0.000 |
| flat τ=0.5 T_t=T carried | 0.3 | 0.39 | 0.932 / 1.4 | 0.56 | 7.38 | 31.9 | 24.5 | 533 | 2.29e+04 | 0.688 | 0.43 | – |
| flat τ=0.5 T_t=3T carried | 0.3 | 0.057 | 0.549 / 0.0772 | 0.586 | 21.9 | 31.9 | 28.9 | 1.37e+03 | 7.09e+04 | 1.96 | 0.44 | – |
| flat τ=0.2 T_t=T carried | 0.3 | 0.78 | 1.11 / 9.16 | 0.583 | 7.33 | 32.3 | 24.8 | 173 | 7.03e+03 | 0.213 | 0.39 | – |
| flat τ=0.2 T_t=3T carried | 0.3 | 0.73 | 0.757 / 4.85 | 0.541 | 23 | 32.8 | 29.9 | 306 | 1.55e+04 | 0.432 | 0.41 | – |
| flat τ=0.05 T_t=T carried | 0.3 | 0.95 | 1.6 / 72 | 0.582 | 7.41 | 31.6 | 24.4 | 26.4 | 1.08e+03 | 0.0315 | 0.40 | – |
| flat τ=0.05 T_t=3T carried | 0.3 | 0.9 | 1.95 / 43.7 | 0.576 | 22.2 | 32.2 | 29.2 | 42.4 | 2.06e+03 | 0.058 | 0.40 | – |
| flat τ=0 T_t=T carried | 0.3 | 0.95 | 1.92 / 84 | 0.582 | 7.36 | 32.2 | 24.8 | 23.1 | 937 | 0.0284 | 0.39 | – |
| flat τ=0 T_t=3T carried | 0.3 | 0.93 | 1.85 / 58 | 0.558 | 22.1 | 32.3 | 29.2 | 33.1 | 1.57e+03 | 0.0448 | 0.39 | – |
| flat τ=0.2 T_t=3T ½ step | 0.3 | 0.86 | 1.08 / 15.5 | 0.235 | 38.3 | 31.9 | 33.8 | 121 | 6.28e+03 | 0.174 | 0.35 | – |
| flat τ=0.05 T_t=3T ½ step | 0.3 | 0.91 | 2.57 / 61 | 0.235 | 38.3 | 33.4 | 34.9 | 30.9 | 1.74e+03 | 0.0475 | 0.38 | – |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 0.3 | 0.92 | 0.602 / 15.2 | 0.235 | 38.3 | 32.3 | 34.1 | 120 | 6.87e+03 | 0.185 | 0.40 | 0.000 |

* Plain H 1.28 per 1 000, 0/3 mixed. Prior draw 7.3, 2/3 mixed (R̂ 1.06 in one run). **Tempered ref: 0.30, 0/3** —
  its component made 0 crossings, sub-chain acceptance 4·10⁻⁵, oob 0.334: the tempered proposals leave the box (study 31).
* Flattened at the carried step: τ 0.5 and 0.2 also send many proposals out of the box (oob 0.12–0.33; B2's carried
  step is large, 0.54–0.59 for the component) and do not mix; **τ 0.05 and τ 0 do (22.7 / 26.4 per 1 000 at T_t = T,
  3/3 each, oob 0.036–0.049 ≈ plain)**, 3–3.6× the prior draw, 90–95 % of the crossings by the component.
* ½ step helps here (the only problem where it does): τ 0.05 3T 14.4 → 20.0, τ 0.2 2.0 → 5.4 (still not mixed).


### 2.2 M3-5

| configuration | runs | crossings / 1 000 (min–max) | mode mass (reps) | R̂_mode (reps) | mixed | ok/poor/failed | mode ESS (ref) | ESS/eval | ESS/s | sub-chain acc H_ADAPT / comp. | outer acc H_ADAPT / comp. | oob | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 3.65 (3.25–4.05) | 0.43;0.52;0.60 | 1.19;1.10;1.10 | 0/3 | 2/1/0 | 87.6 | 0.00532 | 13.5 | – / – | – / – | 0.007 | 0/24 | 13 |
| prior draw PCN(β=1) p=0.2 | 3 | 21.2 (20.4–21.9) | 0.51;0.51;0.52 | 1.02;1.02;1.02 | 3/3 | 3/0/0 | 456 | 0.0262 | 65.6 | 0.9 / 0.18 | 0.998 / 0.994 | 0.006 | 0/24 | 13 |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 3 | 50.6 (50.4–50.8) | 0.48;0.50;0.50 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 1.36e+03 | 0.0589 | 68.8 | 0.9 / 0.37 | 0.998 / 0.997 | 0.041 | 0/24 | 28 |
| flat τ=0.5 T_t=T carried | 3 | 10.6 (10.3–11) | 0.49;0.54;0.45 | 1.04;1.07;1.05 | 2/3 | 3/0/0 | 201 | 0.0107 | 24.6 | 0.9 / 0.69 | 0.998 / 0.997 | 0.005 | 0/24 | 14 |
| flat τ=0.5 T_t=3T carried | 3 | 9.08 (8.67–9.56) | 0.55;0.45;0.46 | 1.04;1.06;1.01 | 2/3 | 3/0/0 | 226 | 0.0109 | 19.3 | 0.9 / 0.68 | 0.998 / 0.999 | 0.006 | 0/24 | 18 |
| flat τ=0.2 T_t=T carried | 3 | 76.3 (75.1–78.1) | 0.52;0.50;0.50 | 1.00;1.00;1.00 | 3/3 | 3/0/0 | 3.87e+03 | 0.0888 | 187 | 0.9 / 0.4 | 0.998 / 0.998 | 0.005 | 0/24 | 15 |
| flat τ=0.2 T_t=3T carried | 3 | 54 (53.2–55.1) | 0.51;0.48;0.51 | 1.00;1.00;1.01 | 3/3 | 3/0/0 | 2.13e+03 | 0.0637 | 105 | 0.9 / 0.4 | 0.997 / 0.998 | 0.005 | 0/24 | 20 |
| flat τ=0.05 T_t=T carried | 3 | 75.4 (73.8–76.4) | 0.52;0.51;0.50 | 1.00;1.01;1.00 | 3/3 | 3/0/0 | 2.82e+03 | 0.0907 | 205 | 0.9 / 0.49 | 0.998 / 0.997 | 0.005 | 0/24 | 14 |
| flat τ=0.05 T_t=3T carried | 3 | 66.1 (62.8–67.8) | 0.52;0.53;0.47 | 1.01;1.00;1.00 | 3/3 | 3/0/0 | 1.12e+03 | 0.0784 | 128 | 0.9 / 0.41 | 0.997 / 0.998 | 0.005 | 0/24 | 20 |
| flat τ=0 T_t=T carried | 3 | 65.8 (64.4–67.6) | 0.51;0.51;0.54 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 826 | 0.0733 | 166 | 0.9 / 0.47 | 0.998 / 0.998 | 0.005 | 0/24 | 14 |
| flat τ=0 T_t=3T carried | 3 | 73.1 (70.1–74.9) | 0.49;0.51;0.50 | 1.00;1.00;1.01 | 3/3 | 3/0/0 | 1.58e+03 | 0.087 | 143 | 0.9 / 0.43 | 0.997 / 0.998 | 0.005 | 0/24 | 20 |
| flat τ=0.2 T_t=3T ½ step | 3 | 55.1 (53.5–56.4) | 0.51;0.50;0.51 | 1.01;1.00;1.01 | 3/3 | 3/0/0 | 1.56e+03 | 0.0697 | 81.4 | 0.9 / 0.41 | 0.997 / 0.997 | 0.005 | 0/24 | 28 |
| flat τ=0.05 T_t=3T ½ step | 3 | 66.4 (65.1–68.2) | 0.49;0.52;0.47 | 1.01;1.00;1.00 | 3/3 | 3/0/0 | 1.77e+03 | 0.0791 | 94.2 | 0.9 / 0.41 | 0.997 / 0.998 | 0.005 | 0/24 | 27 |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 3 | 49.8 (48.8–51.8) | 0.51;0.51;0.50 | 1.00;1.01;1.01 | 3/3 | 3/0/0 | 1.6e+03 | 0.0657 | 75.6 | 0.9 / 0.33 | 0.997 / 0.997 | 0.005 | 0/24 | 28 |

| configuration | comp. share | crossings made by comp. (share) | crossings / 1 000 own iters: H_ADAPT / comp. | ε comp. | nominal L comp. | mean L H_ADAPT | grad / iter | measured-stage exact evals / crossing | grads / crossing | s / crossing | pre-rej | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | – | – | – / – | – | – | 17.7 | 17.7 | 246 | 4.84e+03 | 0.198 | 0.10 | – |
| prior draw PCN(β=1) p=0.2 | 0.2 | 0.83 | 4.41 / 88.9 | – | 0 | 17.8 | 14.2 | 35.7 | 671 | 0.0287 | 0.24 | – |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 0.3 | 0.95 | 3.39 / 162 | 0.094 | 95.7 | 17.8 | 41 | 14.6 | 810 | 0.0252 | 0.26 | 0.000 |
| flat τ=0.5 T_t=T carried | 0.3 | 0.7 | 4.54 / 25 | 0.188 | 16 | 17.9 | 17.4 | 78.7 | 1.63e+03 | 0.0691 | 0.16 | – |
| flat τ=0.5 T_t=3T carried | 0.3 | 0.68 | 4.13 / 20.7 | 0.185 | 48.6 | 17.9 | 27.1 | 91.8 | 2.98e+03 | 0.105 | 0.17 | – |
| flat τ=0.2 T_t=T carried | 0.3 | 0.96 | 4.16 / 246 | 0.188 | 16 | 17.8 | 17.3 | 9.83 | 227 | 0.0094 | 0.25 | – |
| flat τ=0.2 T_t=3T carried | 0.3 | 0.95 | 3.92 / 172 | 0.191 | 47.3 | 17.9 | 26.6 | 13.9 | 493 | 0.017 | 0.25 | – |
| flat τ=0.05 T_t=T carried | 0.3 | 0.96 | 4.01 / 244 | 0.187 | 16.1 | 17.6 | 17.1 | 10.3 | 227 | 0.00915 | 0.22 | – |
| flat τ=0.05 T_t=3T carried | 0.3 | 0.96 | 4.16 / 212 | 0.187 | 48.2 | 17.8 | 26.8 | 11.4 | 406 | 0.0141 | 0.25 | – |
| flat τ=0 T_t=T carried | 0.3 | 0.96 | 4.2 / 211 | 0.19 | 15.8 | 17.8 | 17.2 | 11.7 | 262 | 0.0104 | 0.23 | – |
| flat τ=0 T_t=3T carried | 0.3 | 0.96 | 4.03 / 236 | 0.186 | 48.5 | 17.9 | 27 | 10.4 | 369 | 0.0127 | 0.24 | – |
| flat τ=0.2 T_t=3T ½ step | 0.3 | 0.94 | 4.51 / 175 | 0.094 | 95.7 | 17.8 | 41 | 13.7 | 744 | 0.0236 | 0.25 | – |
| flat τ=0.05 T_t=3T ½ step | 0.3 | 0.95 | 4.32 / 213 | 0.094 | 95.7 | 17.8 | 41 | 11.4 | 618 | 0.0192 | 0.25 | – |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 0.3 | 0.94 | 4.07 / 158 | 0.094 | 95.7 | 17.8 | 41 | 14.7 | 823 | 0.0257 | 0.27 | 0.000 |

* Everything except plain mixes (plain 3.65 per 1 000, R̂ 1.10–1.19). Best: flat τ 0.2 / 0.05 at T_t = T, 75–76
  per 1 000, 1.5× the tempered ref (50.6), 3.6× the prior draw (21.2). τ 0.5 (barrier 1.8 nats) 9–11.


### 2.3 M3-10

| configuration | runs | crossings / 1 000 (min–max) | mode mass (reps) | R̂_mode (reps) | mixed | ok/poor/failed | mode ESS (ref) | ESS/eval | ESS/s | sub-chain acc H_ADAPT / comp. | outer acc H_ADAPT / comp. | oob | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 0.187 (0.0563–0.281) | 0.29;0.48;0.47 | 3.84;2.34;2.25 | 0/3 | 0/3/0 | 10.9 | 0.968 | 1.41e+03 | – / – | – / – | 0.007 | 16/24 | 24 |
| prior draw PCN(β=1) p=0.2 | 3 | 11.8 (11.6–12.2) | 0.49;0.45;0.52 | 1.02;1.03;1.03 | 3/3 | 3/0/0 | 367 | 0.0151 | 24.4 | 0.9 / 0.12 | 0.998 / 0.997 | 0.006 | 0/24 | 23 |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 3 | 42.6 (41–44.1) | 0.52;0.52;0.48 | 1.01;1.00;1.01 | 3/3 | 3/0/0 | 1.1e+03 | 0.0507 | 39 | 0.9 / 0.39 | 0.993 / 0.994 | 0.031 | 0/24 | 49 |
| flat τ=0.5 T_t=T carried | 3 | 0.441 (0.266–0.685) | 0.38;0.36;0.55 | 1.50;2.34;2.06 | 0/3 | 0/3/0 | 15.5 | 0.425 | 758 | 0.9 / 0.73 | 0.997 / 0.996 | 0.005 | 8/24 | 24 |
| flat τ=0.5 T_t=3T carried | 3 | 0.477 (0.319–0.578) | 0.52;0.35;0.50 | 1.76;2.28;1.85 | 0/3 | 2/1/0 | 16 | 0.281 | 313 | 0.9 / 0.73 | 0.996 / 0.997 | 0.006 | 8/24 | 32 |
| flat τ=0.2 T_t=T carried | 3 | 13.8 (13.2–14.2) | 0.50;0.54;0.49 | 1.03;1.06;1.03 | 2/3 | 3/0/0 | 241 | 0.0177 | 24.1 | 0.9 / 0.41 | 0.998 / 0.997 | 0.004 | 0/24 | 27 |
| flat τ=0.2 T_t=3T carried | 3 | 11 (10.6–11.3) | 0.51;0.49;0.47 | 1.04;1.03;1.03 | 3/3 | 3/0/0 | 237 | 0.015 | 16.1 | 0.9 / 0.38 | 0.996 / 0.995 | 0.005 | 0/24 | 34 |
| flat τ=0.05 T_t=T carried | 3 | 63.6 (62–66.7) | 0.48;0.49;0.47 | 1.01;1.00;1.00 | 3/3 | 3/0/0 | 2.02e+03 | 0.074 | 103 | 0.9 / 0.37 | 0.997 / 0.997 | 0.004 | 0/24 | 27 |
| flat τ=0.05 T_t=3T carried | 3 | 49.2 (47.4–51) | 0.49;0.51;0.51 | 1.01;1.01;1.00 | 3/3 | 3/0/0 | 1.4e+03 | 0.0605 | 62.3 | 0.9 / 0.32 | 0.996 / 0.996 | 0.005 | 0/24 | 35 |
| flat τ=0 T_t=T carried | 3 | 59.7 (52.2–73.4) | 0.48;0.53;0.50 | 1.01;1.00;1.00 | 3/3 | 3/0/0 | 2.26e+03 | 0.0671 | 111 | 0.9 / 0.38 | 0.998 / 0.998 | 0.004 | 0/24 | 20 |
| flat τ=0 T_t=3T carried | 3 | 58.5 (57–60.5) | 0.50;0.51;0.49 | 1.01;1.01;1.00 | 3/3 | 3/0/0 | 1.75e+03 | 0.0704 | 94.8 | 0.9 / 0.35 | 0.995 / 0.996 | 0.005 | 0/24 | 24 |
| flat τ=0.2 T_t=3T ½ step | 3 | 10.4 (9.76–11.4) | 0.52;0.55;0.45 | 1.02;1.03;1.04 | 3/3 | 3/0/0 | 277 | 0.0144 | 13.9 | 0.9 / 0.38 | 0.993 / 0.993 | 0.005 | 0/24 | 33 |
| flat τ=0.05 T_t=3T ½ step | 3 | 49.2 (47.4–51.1) | 0.52;0.50;0.48 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 1.3e+03 | 0.0606 | 57.9 | 0.9 / 0.32 | 0.992 / 0.993 | 0.005 | 0/24 | 34 |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 3 | 34.1 (32.9–35.5) | 0.48;0.51;0.48 | 1.02;1.01;1.00 | 3/3 | 3/0/0 | 831 | 0.0434 | 42.1 | 0.9 / 0.33 | 0.993 / 0.994 | 0.005 | 0/24 | 33 |

| configuration | comp. share | crossings made by comp. (share) | crossings / 1 000 own iters: H_ADAPT / comp. | ε comp. | nominal L comp. | mean L H_ADAPT | grad / iter | measured-stage exact evals / crossing | grads / crossing | s / crossing | pre-rej | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | – | – | – / – | – | – | 24.4 | 24.4 | 4.8e+03 | 1.3e+05 | 7.22 | 0.10 | – |
| prior draw PCN(β=1) p=0.2 | 0.2 | 0.98 | 0.27 / 58.6 | – | 0 | 24.2 | 19.4 | 62.7 | 1.64e+03 | 0.0892 | 0.26 | – |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 0.3 | 1 | 0.265 / 142 | 0.078 | 115 | 23.8 | 51.1 | 17.5 | 1.2e+03 | 0.0535 | 0.25 | 0.000 |
| flat τ=0.5 T_t=T carried | 0.3 | 0.88 | 0.0753 / 1.3 | 0.155 | 19.6 | 23.9 | 22.6 | 1.92e+03 | 5.12e+04 | 2.86 | 0.15 | – |
| flat τ=0.5 T_t=3T carried | 0.3 | 0.81 | 0.126 / 1.3 | 0.156 | 58.4 | 24.7 | 34.7 | 1.78e+03 | 7.27e+04 | 3.59 | 0.15 | – |
| flat τ=0.2 T_t=T carried | 0.3 | 0.99 | 0.246 / 45.7 | 0.156 | 19.5 | 23.9 | 22.6 | 54.8 | 1.64e+03 | 0.0935 | 0.25 | – |
| flat τ=0.2 T_t=3T carried | 0.3 | 0.99 | 0.221 / 36.5 | 0.157 | 57.9 | 24.1 | 34.2 | 67.8 | 3.1e+03 | 0.143 | 0.25 | – |
| flat τ=0.05 T_t=T carried | 0.3 | 1 | 0.309 / 213 | 0.155 | 19.6 | 23.4 | 22.2 | 11.6 | 349 | 0.0196 | 0.26 | – |
| flat τ=0.05 T_t=3T carried | 0.3 | 1 | 0.216 / 165 | 0.156 | 58.2 | 24.2 | 34.3 | 14.7 | 698 | 0.0325 | 0.27 | – |
| flat τ=0 T_t=T carried | 0.3 | 1 | 0.154 / 200 | 0.148 | 20.3 | 23.2 | 22.3 | 12.4 | 373 | 0.0155 | 0.26 | – |
| flat τ=0 T_t=3T carried | 0.3 | 1 | 0.306 / 196 | 0.157 | 58.1 | 24.3 | 34.3 | 12.6 | 587 | 0.0189 | 0.27 | – |
| flat τ=0.2 T_t=3T ½ step | 0.3 | 0.98 | 0.265 / 34.3 | 0.078 | 115 | 24.6 | 51.6 | 71.6 | 4.96e+03 | 0.149 | 0.25 | – |
| flat τ=0.05 T_t=3T ½ step | 0.3 | 1 | 0.194 / 165 | 0.078 | 115 | 24.7 | 51.7 | 14.8 | 1.05e+03 | 0.0311 | 0.27 | – |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 0.3 | 1 | 0.0867 / 114 | 0.078 | 115 | 24.6 | 51.6 | 21.4 | 1.51e+03 | 0.0444 | 0.27 | 0.000 |

* Plain 0.19, 0/3, 16/24 chains never cross. Prior draw 11.8, tempered ref 42.6 (study 31: 41.2), both 3/3.
* τ 0.05 / 0 (no flattened barrier): 49–64 per 1 000, 3/3 at every T and step; τ 0.2 (barrier ≈ 1 nat) 10–14, 2–3/3
  mixed; τ 0.5 (4.3 nats) ≈ plain (0.44–0.48, 0/3).


### 2.4 M3-20

| configuration | runs | crossings / 1 000 (min–max) | mode mass (reps) | R̂_mode (reps) | mixed | ok/poor/failed | mode ESS (ref) | ESS/eval | ESS/s | sub-chain acc H_ADAPT / comp. | outer acc H_ADAPT / comp. | oob | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 0.0187 (0–0.0561) | 0.37;0.37;0.57 | inf;inf;5.17 | 0/3 | 0/3/0 | 8.36 | 1.4 | 2.31e+03 | – / – | – / – | 0.006 | 23/24 | 20 |
| prior draw PCN(β=1) p=0.2 | 3 | 7.98 (7.45–8.68) | 0.50;0.50;0.51 | 1.07;1.04;1.02 | 2/3 | 3/0/0 | 312 | 0.0111 | 17.5 | 0.9 / 0.081 | 0.997 / 0.997 | 0.005 | 0/24 | 20 |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 3 | 31.8 (30.5–33) | 0.48;0.51;0.47 | 1.01;1.02;1.02 | 3/3 | 3/0/0 | 491 | 0.0363 | 28.9 | 0.9 / 0.42 | 0.993 / 0.992 | 0.019 | 0/24 | 40 |
| flat τ=0.5 T_t=T carried | 3 | 0.0528 (0–0.105) | 0.15;0.62;0.75 | 3.05;12.02;inf | 0/3 | 0/3/0 | 8.23 | 1.16 | 1.89e+03 | 0.9 / 0.75 | 0.996 / 0.997 | 0.005 | 21/24 | 20 |
| flat τ=0.5 T_t=3T carried | 3 | 0.0355 (0–0.0535) | 0.62;0.73;0.26 | inf;4.02;5.57 | 0/3 | 0/3/0 | 8.25 | 0.807 | 993 | 0.9 / 0.75 | 0.995 / 0.995 | 0.008 | 22/24 | 26 |
| flat τ=0.2 T_t=T carried | 3 | 0.546 (0.486–0.627) | 0.38;0.47;0.50 | 1.33;2.23;1.77 | 0/3 | 1/2/0 | 22.6 | 0.232 | 348 | 0.9 / 0.48 | 0.996 / 0.996 | 0.006 | 5/24 | 22 |
| flat τ=0.2 T_t=3T carried | 3 | 0.6 (0.436–0.684) | 0.52;0.47;0.52 | 1.59;1.56;2.00 | 0/3 | 3/0/0 | 19.6 | 0.139 | 155 | 0.9 / 0.49 | 0.996 / 0.996 | 0.006 | 4/24 | 29 |
| flat τ=0.05 T_t=T carried | 3 | 50.7 (47.7–54.8) | 0.49;0.51;0.47 | 1.01;1.00;1.01 | 3/3 | 3/0/0 | 1.8e+03 | 0.0665 | 90 | 0.9 / 0.26 | 0.996 / 0.997 | 0.005 | 0/24 | 24 |
| flat τ=0.05 T_t=3T carried | 3 | 35.6 (32.8–37.3) | 0.54;0.51;0.50 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 1.98e+03 | 0.0431 | 44 | 0.9 / 0.24 | 0.996 / 0.996 | 0.006 | 0/24 | 32 |
| flat τ=0 T_t=T carried | 3 | 43.2 (40.8–47.5) | 0.51;0.53;0.50 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 1.01e+03 | 0.0508 | 71.4 | 0.9 / 0.31 | 0.996 / 0.994 | 0.006 | 0/24 | 23 |
| flat τ=0 T_t=3T carried | 3 | 48.1 (47.4–48.7) | 0.49;0.52;0.50 | 1.01;1.00;1.01 | 3/3 | 3/0/0 | 1.1e+03 | 0.0623 | 64 | 0.9 / 0.28 | 0.994 / 0.995 | 0.004 | 0/24 | 31 |
| flat τ=0.2 T_t=3T ½ step | 3 | 0.437 (0.293–0.531) | 0.55;0.57;0.64 | 1.84;2.76;2.21 | 0/3 | 0/3/0 | 16.7 | 0.23 | 187 | 0.9 / 0.5 | 0.992 / 0.992 | 0.006 | 6/24 | 40 |
| flat τ=0.05 T_t=3T ½ step | 3 | 33.5 (31.5–36.2) | 0.55;0.50;0.48 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 818 | 0.044 | 32.8 | 0.9 / 0.23 | 0.992 / 0.991 | 0.006 | 0/24 | 43 |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 3 | 17.5 (16.5–19.2) | 0.46;0.52;0.52 | 1.02;1.04;1.04 | 3/3 | 3/0/0 | 339 | 0.0187 | 14.9 | 0.9 / 0.42 | 0.992 / 0.994 | 0.006 | 0/24 | 40 |

| configuration | comp. share | crossings made by comp. (share) | crossings / 1 000 own iters: H_ADAPT / comp. | ε comp. | nominal L comp. | mean L H_ADAPT | grad / iter | measured-stage exact evals / crossing | grads / crossing | s / crossing | pre-rej | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | – | – | – / – | – | – | 33.3 | 33.3 | 4.8e+04 | 1.78e+06 | 59 | 0.10 | – |
| prior draw PCN(β=1) p=0.2 | 0.2 | 1 | 0.0191 / 40.1 | – | 0 | 33.5 | 26.8 | 92.1 | 3.36e+03 | 0.117 | 0.26 | – |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 0.3 | 1 | 0 / 107 | 0.061 | 148 | 33.4 | 67.4 | 23.8 | 2.12e+03 | 0.0603 | 0.24 | 0.000 |
| flat τ=0.5 T_t=T carried | 0.3 | 0.33 | 0.0501 / 0.0599 | 0.123 | 24.5 | 33.1 | 30.6 | 1.6e+04 | 5.74e+05 | 19.8 | 0.15 | – |
| flat τ=0.5 T_t=3T carried | 0.3 | 0 | 0.0506 / 0 | 0.126 | 72.1 | 34.7 | 45.8 | 2.4e+04 | 1.29e+06 | 39.3 | 0.15 | – |
| flat τ=0.2 T_t=T carried | 0.3 | 0.97 | 0.023 / 1.78 | 0.122 | 24.8 | 33.7 | 31.1 | 1.41e+03 | 5.69e+04 | 1.92 | 0.23 | – |
| flat τ=0.2 T_t=3T carried | 0.3 | 0.95 | 0.0462 / 1.91 | 0.125 | 72.5 | 35.2 | 46.3 | 1.3e+03 | 7.73e+04 | 2.35 | 0.22 | – |
| flat τ=0.05 T_t=T carried | 0.3 | 1 | 0.0835 / 170 | 0.123 | 24.7 | 34.2 | 31.4 | 13.9 | 619 | 0.0207 | 0.29 | – |
| flat τ=0.05 T_t=3T carried | 0.3 | 1 | 0.0208 / 119 | 0.122 | 74.6 | 33.6 | 45.8 | 19.8 | 1.29e+03 | 0.0389 | 0.30 | – |
| flat τ=0 T_t=T carried | 0.3 | 1 | 0.0214 / 145 | 0.128 | 23.7 | 34.5 | 31.3 | 16.8 | 724 | 0.024 | 0.28 | – |
| flat τ=0 T_t=3T carried | 0.3 | 1 | 0.0212 / 161 | 0.123 | 74 | 33.8 | 45.8 | 14.8 | 953 | 0.0292 | 0.29 | – |
| flat τ=0.2 T_t=3T ½ step | 0.3 | 0.96 | 0.023 / 1.41 | 0.061 | 148 | 34.8 | 68.4 | 1.78e+03 | 1.56e+05 | 4.43 | 0.22 | – |
| flat τ=0.05 T_t=3T ½ step | 0.3 | 1 | 0 / 112 | 0.061 | 148 | 33.6 | 67.6 | 20.8 | 2.01e+03 | 0.0563 | 0.30 | – |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 0.3 | 1 | 0 / 58.9 | 0.061 | 148 | 34 | 67.8 | 43 | 3.87e+03 | 0.108 | 0.24 | 0.000 |

* Plain 0.02 (23/24 chains never cross). Prior draw 8.0 (2/3 mixed); tempered ref 31.8 (3/3; study 31: 31.6).
* τ 0.05 / 0: 33–51 per 1 000, 3/3 everywhere; τ 0.2 (2.9-nat barrier) 0.44–0.60, 0/3; τ 0.5 (9.3 nats) ≈ plain.
  Combined 17.5 (3/3).


### 2.5 M10-10

| configuration | runs | crossings / 1 000 (min–max) | mode mass (reps) | R̂_mode (reps) | mixed | ok/poor/failed | mode ESS (ref) | ESS/eval | ESS/s | sub-chain acc H_ADAPT / comp. | outer acc H_ADAPT / comp. | oob | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 0.318 (0.168–0.45) | 0.56;0.54;0.64 | 3.51;2.27;1.88 | 0/3 | 0/3/0 | 12.5 | 0.631 | 419 | – / – | – / – | 0.009 | 11/24 | 48 |
| prior draw PCN(β=1) p=0.2 | 3 | 0.948 (0.866–0.992) | 0.54;0.58;0.42 | 1.51;1.46;1.23 | 0/3 | 2/1/0 | 28.9 | 0.048 | 35.5 | 0.9 / 0.0081 | 0.999 / 1 | 0.008 | 1/24 | 34 |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 3 | 20.1 (19–20.9) | 0.47;0.47;0.53 | 1.02;1.03;1.02 | 3/3 | 3/0/0 | 324 | 0.0306 | 17.9 | 0.9 / 0.18 | 0.999 / 0.999 | 0.030 | 0/24 | 69 |
| flat τ=0.5 T_t=T carried | 3 | 2.44 (2.21–2.81) | 0.43;0.41;0.47 | 1.14;1.22;1.11 | 0/3 | 3/0/0 | 73 | 0.00572 | 6.44 | 0.9 / 0.45 | 0.999 / 0.997 | 0.008 | 0/24 | 32 |
| flat τ=0.5 T_t=3T carried | 3 | 2.62 (2.27–3.22) | 0.63;0.47;0.45 | 1.16;1.18;1.10 | 0/3 | 2/1/0 | 57.9 | 0.0204 | 10.4 | 0.9 / 0.47 | 0.999 / 0.999 | 0.008 | 0/24 | 50 |
| flat τ=0.2 T_t=T carried | 3 | 48 (38.2–65) | 0.53;0.50;0.48 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 1.12e+03 | 0.0687 | 68.8 | 0.9 / 0.26 | 0.999 / 0.999 | 0.006 | 0/24 | 36 |
| flat τ=0.2 T_t=3T carried | 3 | 40.8 (38.6–43.2) | 0.51;0.51;0.49 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 3.2e+03 | 0.0567 | 37.6 | 0.9 / 0.28 | 0.998 / 0.999 | 0.008 | 0/24 | 60 |
| flat τ=0.05 T_t=T carried | 3 | 32.6 (30.1–34.5) | 0.51;0.47;0.48 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 822 | 0.046 | 51.8 | 0.9 / 0.29 | 0.999 / 0.998 | 0.007 | 0/24 | 39 |
| flat τ=0.05 T_t=3T carried | 3 | 43.3 (41.5–44.3) | 0.48;0.51;0.51 | 1.01;1.00;1.01 | 3/3 | 3/0/0 | 1.46e+03 | 0.0614 | 41.7 | 0.9 / 0.29 | 0.999 / 0.999 | 0.008 | 0/24 | 58 |
| flat τ=0 T_t=T carried | 3 | 31.3 (29.9–32.7) | 0.52;0.47;0.53 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 1.04e+03 | 0.0451 | 38.8 | 0.9 / 0.3 | 0.998 / 0.999 | 0.008 | 0/24 | 50 |
| flat τ=0 T_t=3T carried | 3 | 49.6 (48.4–51.6) | 0.49;0.49;0.50 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 905 | 0.0668 | 47 | 0.9 / 0.3 | 0.998 / 0.999 | 0.007 | 0/24 | 56 |
| flat τ=0.2 T_t=3T ½ step | 3 | 41.9 (40.7–42.8) | 0.50;0.53;0.51 | 1.01;1.02;1.01 | 3/3 | 3/0/0 | 965 | 0.0565 | 34 | 0.9 / 0.28 | 0.999 / 0.998 | 0.007 | 0/24 | 69 |
| flat τ=0.05 T_t=3T ½ step | 3 | 42.5 (38.8–47) | 0.49;0.47;0.49 | 1.01;1.01;1.00 | 3/3 | 3/0/0 | 1e+03 | 0.0591 | 38.7 | 0.9 / 0.3 | 0.999 / 0.999 | 0.007 | 0/24 | 67 |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 3 | 22 (21.1–23.5) | 0.52;0.50;0.52 | 1.04;1.01;1.02 | 3/3 | 3/0/0 | 433 | 0.0319 | 17.7 | 0.9 / 0.15 | 0.999 / 1 | 0.008 | 0/24 | 75 |

| configuration | comp. share | crossings made by comp. (share) | crossings / 1 000 own iters: H_ADAPT / comp. | ε comp. | nominal L comp. | mean L H_ADAPT | grad / iter | measured-stage exact evals / crossing | grads / crossing | s / crossing | pre-rej | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | – | – | – / – | – | – | 24.7 | 24.7 | 2.82e+03 | 7.77e+04 | 8.48 | 0.10 | – |
| prior draw PCN(β=1) p=0.2 | 0.2 | 0.84 | 0.188 / 4.01 | – | 0 | 26.2 | 21 | 762 | 2.21e+04 | 1.64 | 0.28 | – |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 0.3 | 0.99 | 0.285 / 66.7 | 0.0629 | 143 | 26 | 61 | 34.1 | 3.03e+03 | 0.148 | 0.31 | 0.000 |
| flat τ=0.5 T_t=T carried | 0.3 | 0.92 | 0.295 / 7.49 | 0.125 | 23.9 | 26.5 | 25.8 | 314 | 1.05e+04 | 0.631 | 0.23 | – |
| flat τ=0.5 T_t=3T carried | 0.3 | 0.95 | 0.183 / 8.35 | 0.126 | 71.4 | 26.9 | 40.1 | 294 | 1.53e+04 | 0.927 | 0.23 | – |
| flat τ=0.2 T_t=T carried | 0.3 | 1 | 0.147 / 161 | 0.126 | 23.9 | 21.9 | 22.5 | 14.8 | 469 | 0.0328 | 0.29 | – |
| flat τ=0.2 T_t=3T carried | 0.3 | 1 | 0.277 / 136 | 0.126 | 71.6 | 26 | 39.6 | 17.6 | 970 | 0.0653 | 0.28 | – |
| flat τ=0.05 T_t=T carried | 0.3 | 1 | 0.214 / 109 | 0.126 | 23.8 | 25.7 | 25.1 | 22 | 770 | 0.0542 | 0.28 | – |
| flat τ=0.05 T_t=3T carried | 0.3 | 1 | 0.234 / 145 | 0.126 | 71.6 | 26 | 39.6 | 16.5 | 913 | 0.0604 | 0.28 | – |
| flat τ=0 T_t=T carried | 0.3 | 1 | 0.215 / 105 | 0.126 | 23.8 | 26.1 | 25.4 | 23.1 | 812 | 0.0728 | 0.28 | – |
| flat τ=0 T_t=3T carried | 0.3 | 1 | 0.172 / 166 | 0.127 | 71.1 | 26.2 | 39.6 | 14.6 | 798 | 0.0512 | 0.28 | – |
| flat τ=0.2 T_t=3T ½ step | 0.3 | 1 | 0.234 / 140 | 0.0629 | 143 | 25 | 60.2 | 17.1 | 1.44e+03 | 0.0741 | 0.28 | – |
| flat τ=0.05 T_t=3T ½ step | 0.3 | 1 | 0.128 / 143 | 0.0629 | 143 | 23 | 58.8 | 17 | 1.38e+03 | 0.071 | 0.28 | – |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 0.3 | 0.99 | 0.242 / 73.2 | 0.0629 | 143 | 26.2 | 61.1 | 30.8 | 2.77e+03 | 0.144 | 0.32 | 0.000 |

* Prior draw collapses: 0.95 per 1 000, 0/3 mixed (sub-chain acceptance of the PCN iterations 0.0081). Tempered ref
  20.1, 3/3. Every flattened row with τ ≤ 0.2 (no barrier at d = 10) gives 31–50, 3/3; τ 0.5 (2.8 nats) 2.4–2.6,
  0/3 mixed although ok.


### 2.6 M30-10

| configuration | runs | crossings / 1 000 (min–max) | mode mass (reps) | R̂_mode (reps) | mixed | ok/poor/failed | mode ESS (ref) | ESS/eval | ESS/s | sub-chain acc H_ADAPT / comp. | outer acc H_ADAPT / comp. | oob | never-crossed chains | wall s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | 3 | 0.112 (0.0563–0.168) | 0.37;0.71;0.69 | 2.82;3.38;12.15 | 0/3 | 0/3/0 | 9.49 | 0.908 | 653 | – / – | – / – | 0.000 | 19/24 | 45 |
| prior draw PCN(β=1) p=0.2 | 3 | 0.0455 (0–0.136) | 0.75;0.18;0.38 | inf;2.35;inf | 0/3 | 0/3/0 | 9.48 | 0.706 | 1.21e+03 | 0.9 / 7.8e-05 | 0.984 / 0.333 | 0.000 | 21/24 | 19 |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 3 | 2.16 (1.94–2.49) | 0.43;0.42;0.45 | 1.25;1.21;1.15 | 0/3 | 3/0/0 | 65.5 | 0.00721 | 5.17 | 0.9 / 0.021 | 0.975 / 0.977 | 0.048 | 0/24 | 45 |
| flat τ=0.5 T_t=T carried | 3 | 24.2 (23.6–25.4) | 0.47;0.47;0.51 | 1.01;1.02;1.01 | 3/3 | 3/0/0 | 460 | 0.037 | 53.4 | 0.9 / 0.19 | 0.98 / 0.98 | 0.000 | 0/24 | 22 |
| flat τ=0.5 T_t=3T carried | 3 | 27.9 (27.4–28.3) | 0.52;0.49;0.49 | 1.01;1.01;1.02 | 3/3 | 3/0/0 | 436 | 0.0414 | 45.2 | 0.9 / 0.19 | 0.982 / 0.981 | 0.000 | 0/24 | 29 |
| flat τ=0.2 T_t=T carried | 3 | 27.4 (26.3–28.5) | 0.51;0.50;0.49 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 802 | 0.0395 | 56.4 | 0.9 / 0.18 | 0.982 / 0.979 | 0.000 | 0/24 | 23 |
| flat τ=0.2 T_t=3T carried | 3 | 29.1 (28.6–30.1) | 0.48;0.50;0.51 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 614 | 0.0434 | 47 | 0.9 / 0.19 | 0.98 / 0.981 | 0.000 | 0/24 | 30 |
| flat τ=0.05 T_t=T carried | 3 | 33.2 (32.6–33.7) | 0.45;0.51;0.49 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 802 | 0.0491 | 69.6 | 0.9 / 0.19 | 0.982 / 0.977 | 0.000 | 0/24 | 23 |
| flat τ=0.05 T_t=3T carried | 3 | 25.8 (24.3–28.6) | 0.49;0.51;0.53 | 1.01;1.01;1.02 | 3/3 | 3/0/0 | 703 | 0.0376 | 40.5 | 0.9 / 0.19 | 0.979 / 0.984 | 0.000 | 0/24 | 30 |
| flat τ=0 T_t=T carried | 3 | 31 (27.8–33) | 0.47;0.50;0.52 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 659 | 0.0456 | 64.8 | 0.9 / 0.17 | 0.981 / 0.982 | 0.000 | 0/24 | 23 |
| flat τ=0 T_t=3T carried | 3 | 23.2 (21.4–24.4) | 0.48;0.52;0.50 | 1.02;1.01;1.02 | 3/3 | 3/0/0 | 796 | 0.0367 | 40.3 | 0.9 / 0.17 | 0.978 / 0.975 | 0.000 | 0/24 | 29 |
| flat τ=0.2 T_t=3T ½ step | 3 | 29.7 (29.1–30.7) | 0.50;0.54;0.53 | 1.02;1.01;1.01 | 3/3 | 3/0/0 | 532 | 0.0448 | 34.5 | 0.9 / 0.2 | 0.972 / 0.975 | 0.000 | 0/24 | 42 |
| flat τ=0.05 T_t=3T ½ step | 3 | 24.7 (23.5–25.7) | 0.51;0.52;0.47 | 1.01;1.01;1.01 | 3/3 | 3/0/0 | 825 | 0.0339 | 26.4 | 0.9 / 0.18 | 0.97 / 0.963 | 0.000 | 0/24 | 41 |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 3 | 6.66 (6.25–7.24) | 0.48;0.59;0.54 | 1.02;1.05;1.06 | 1/3 | 3/0/0 | 310 | 0.0133 | 9.72 | 0.9 / 0.043 | 0.972 / 0.954 | 0.000 | 0/24 | 44 |

| configuration | comp. share | crossings made by comp. (share) | crossings / 1 000 own iters: H_ADAPT / comp. | ε comp. | nominal L comp. | mean L H_ADAPT | grad / iter | measured-stage exact evals / crossing | grads / crossing | s / crossing | pre-rej | skipped |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| plain H | – | – | – / – | – | – | 25 | 25 | 8e+03 | 2.23e+05 | 22.4 | 0.10 | – |
| prior draw PCN(β=1) p=0.2 | 0.2 | 0 | 0.0564 / 0 | – | 0 | 25.9 | 20.8 | 1.6e+04 | 4.6e+05 | 18.9 | 0.28 | – |
| tempered ref (f 0.35, β 0.001, 3T, ½ step) p=0.3 | 0.3 | 0.96 | 0.132 / 6.94 | 0.0645 | 140 | 26.3 | 60.1 | 294 | 2.78e+04 | 0.825 | 0.36 | 0.000 |
| flat τ=0.5 T_t=T carried | 0.3 | 1 | 0.164 / 80.8 | 0.133 | 22.6 | 26.3 | 25.2 | 28.5 | 1.04e+03 | 0.0397 | 0.31 | – |
| flat τ=0.5 T_t=3T carried | 0.3 | 1 | 0.163 / 93 | 0.132 | 68.2 | 25.4 | 38.2 | 24.7 | 1.37e+03 | 0.0454 | 0.31 | – |
| flat τ=0.2 T_t=T carried | 0.3 | 0.99 | 0.203 / 91.3 | 0.131 | 22.9 | 26.2 | 25.2 | 25 | 921 | 0.0353 | 0.31 | – |
| flat τ=0.2 T_t=3T carried | 0.3 | 1 | 0.204 / 97 | 0.131 | 68.9 | 26.3 | 39 | 23.7 | 1.34e+03 | 0.0441 | 0.31 | – |
| flat τ=0.05 T_t=T carried | 0.3 | 1 | 0.122 / 111 | 0.131 | 22.8 | 26.5 | 25.4 | 20.7 | 764 | 0.0294 | 0.31 | – |
| flat τ=0.05 T_t=3T carried | 0.3 | 1 | 0.122 / 86.3 | 0.132 | 68.2 | 25.7 | 38.4 | 26.6 | 1.48e+03 | 0.0496 | 0.31 | – |
| flat τ=0 T_t=T carried | 0.3 | 1 | 0.203 / 103 | 0.132 | 22.8 | 26.4 | 25.3 | 22 | 818 | 0.0312 | 0.32 | – |
| flat τ=0 T_t=3T carried | 0.3 | 0.99 | 0.284 / 77.2 | 0.133 | 67.8 | 25.5 | 38.1 | 29.4 | 1.64e+03 | 0.0539 | 0.32 | – |
| flat τ=0.2 T_t=3T ½ step | 0.3 | 1 | 0.184 / 99.3 | 0.0645 | 140 | 26 | 59.9 | 23.2 | 2.01e+03 | 0.0605 | 0.31 | – |
| flat τ=0.05 T_t=3T ½ step | 0.3 | 1 | 0.122 / 82.4 | 0.0645 | 140 | 26.1 | 60 | 27.7 | 2.43e+03 | 0.0715 | 0.32 | – |
| combined τ 0.2 + f 0.35 β 0.03, 3T, ½ step | 0.3 | 0.98 | 0.23 / 21.8 | 0.0645 | 140 | 25.9 | 59.8 | 96.8 | 8.99e+03 | 0.266 | 0.36 | 0.000 |

* Prior draw 0.05 per 1 000 (0/3; 21/24 chains never cross; its component made no crossing). **Tempered ref 2.16,
  0/3 mixed** (R̂ 1.15–1.25; posterior check 3/3 ok). Every flattened row 23–33 per 1 000, 3/3, including τ 0.5
  (barrier 0.3 nats at d = 30). Combined 6.7 (1/3).

---

## 3. Answers to note 32 §2.4

Headline (crossings per 1 000 measured-stage iterations, mean of 3 runs; mixed / 3; mode ESS (ref)):

| problem | plain H | prior draw p 0.2 | tempered ref | flat τ 0.05, T_t = T, carried | flat τ 0, T_t = T, carried | combined |
|---|---|---|---|---|---|---|
| B2 (d 2) | 1.28, 0/3, 10 | 7.32, 2/3, 248 | 0.30, 0/3, 12 | **22.7, 3/3, 2 000** | 26.4, 3/3, 912 | 4.97, 0/3, 113 |
| M3-5 | 3.65, 0/3, 88 | 21.2, 3/3, 456 | 50.6, 3/3, 1 357 | **75.4, 3/3, 2 822** | 65.8, 3/3, 826 | 49.8, 3/3, 1 595 |
| M3-10 | 0.19, 0/3, 11 | 11.8, 3/3, 367 | 42.6, 3/3, 1 097 | **63.6, 3/3, 2 022** | 59.7, 3/3, 2 257 | 34.1, 3/3, 831 |
| M3-20 | 0.02, 0/3, 8 | 7.98, 2/3, 312 | 31.8, 3/3, 491 | **50.7, 3/3, 1 796** | 43.2, 3/3, 1 013 | 17.5, 3/3, 339 |
| M10-10 | 0.32, 0/3, 13 | 0.95, 0/3, 29 | 20.1, 3/3, 324 | **32.6, 3/3, 822** | 31.3, 3/3, 1 036 | 22.0, 3/3, 433 |
| M30-10 | 0.11, 0/3, 9 | 0.05, 0/3, 9 | 2.16, 0/3, 65 | **33.2, 3/3, 802** | 31.0, 3/3, 659 | 6.66, 1/3, 310 |

Reproducibility check against study 31 (same settings): plain M3-10 0.19 vs 0.24, prior draw M3-10 11.8 vs 11.9,
tempered ref M3-10 42.6 vs 41.2 and M3-20 31.8 vs 31.6 per 1 000; B2 plain 1.28 vs 0.66 (within B2's spread).

Figures: `figures/fig_crossings_vs_dimension.png` (d = 3, 10, 30 at Δ = 10) and `figures/fig_crossings_vs_depth.png`
(d = 3, Δ = 5, 10, 20): plain, prior draw, tempered ref, best flattened row per point (selected after the fact, label
on the point) and the fixed row flat τ 0.05 T_t = T carried; log axis, bars = min–max, 0 plotted at 0.01.

**(i) Crossings and mixing verdict per mechanism.**
* *Prior draw*: mixes in d ≤ 3 (10/12 runs; 2 runs at R̂ 1.06–1.07), collapses with d: 11.8 → 0.95 → 0.05 per
  1 000 at Δ = 10 (sub-chain acceptance of its iterations 0.12 → 0.0081 → 0.00008); at d = 30 the 3 crossings of
  the 3 runs were all made by `H_ADAPT`. Prediction of note 32 confirmed.
* *Tempered ref*: mixes on M3-5/10/20 and M10-10 (12/12), fails on B2 (0.30 per 1 000, 0/3; its sub-chain
  acceptance 4·10⁻⁵, oob 0.334 ≈ p_t — essentially every tempered proposal leaves the box, as in study 31) and drops at
  d = 30 (2.16, 0/3 mixed, R̂ 1.15–1.25, although 3/3 "ok"): sub-chain acceptance of its iterations 0.39 → 0.18 →
  0.021 for d = 3 → 10 → 30. Prediction "tempering should not fall with d" **not confirmed**: it falls 20× from d = 3 to 30.
* *Flattened*: the small-τ rows mix everywhere — τ 0.05 T carried, τ 0 T carried, τ 0 3T carried and τ 0.05 3T ½ step
  are 18/18 mixed each (τ 0.05 3T carried 17/18). τ = 0.2 and 0.5 work only where the flattened barrier vanishes
  (next item). **90–100 % of the crossings are made by the flattened component** on every problem for τ ≤ 0.05 (B2: 0.90–0.95;
  M: 0.95–1.00); its own crossing rate is 44–246 per 1 000 of its iterations, against 0–2.6 for `H_ADAPT` in the same runs.
* *Why τ matters, analytically.* Along t the flattened potential is `U_τ(t) = t²/2 + τ (t²/d − y)²/(2σ²)`; since
  `log L(mode) − log L(0) = Δ + d/2`, the barrier at the saddle is about `max(0, τ(Δ + d/2) − d/2)`, i.e. it vanishes for
  `τ ≲ d/(2Δ + d)`. Numerically (1-D grid, `U_τ(0) − min U_τ`):

| problem | τ* = d/(2Δ+d) | barrier at τ = 1 / 0.5 / 0.2 / 0.05 / 0 (nats) | crossings / 1 000, carried T_t = T, τ = 0.5 / 0.2 / 0.05 / 0 |
|---|---|---|---|
| M3-5 | 0.23 | 5.0 / 1.81 / 0.16 / 0 / 0 | 10.6 / 76.3 / 75.4 / 65.8 |
| M3-10 | 0.13 | 10 / 4.28 / 0.98 / 0 / 0 | 0.44 / 13.8 / 63.6 / 59.7 |
| M3-20 | 0.07 | 20 / 9.26 / 2.89 / 0.08 / 0 | 0.05 / 0.55 / 50.7 / 43.2 |
| M10-10 | 0.33 | 10 / 2.81 / 0 / 0 / 0 | 2.44 / 48.0 / 32.6 / 31.3 |
| M30-10 | 0.60 | 10 / 0.31 / 0 / 0 / 0 | 24.2 / 27.4 / 33.2 / 31.0 |

  Every row with a barrier ≤ 0.3 nats crosses 24–76 per 1 000; ≈ 1 nat 11–14; 1.8–2.9 nats 0.5–11; ≥ 4 nats like plain.
  This also explains the "inverse" dimension dependence of τ = 0.5 (0.44 at d = 3, 24 at d = 30): in the M-family the
  modes sit at prior radius √d, the prior pulls them together, and the flattened trajectory sees mainly the prior.

**(ii) Cost per crossing** (measured-stage exact evaluations / nominal network gradients / seconds per crossing, cell
totals; from the "components and cost" tables):

| problem | prior draw | tempered ref | flat τ 0.05 T carried | flat τ 0 T carried |
|---|---|---|---|---|
| B2 | 89 / 3.5e3 / 0.10 | 1 850 / 1.1e5 / 2.65 | **26 / 1.1e3 / 0.032** | 23 / 940 / 0.028 |
| M3-5 | 36 / 670 / 0.029 | 15 / 810 / 0.025 | **10 / 230 / 0.009** | 12 / 260 / 0.010 |
| M3-10 | 63 / 1.6e3 / 0.089 | 18 / 1.2e3 / 0.054 | **12 / 350 / 0.020** | 12 / 370 / 0.016 |
| M3-20 | 92 / 3.4e3 / 0.12 | 24 / 2.1e3 / 0.060 | **14 / 620 / 0.021** | 17 / 720 / 0.024 |
| M10-10 | 762 / 2.2e4 / 1.64 | 34 / 3.0e3 / 0.15 | **22 / 770 / 0.054** | 23 / 810 / 0.073 |
| M30-10 | 1.6e4 / 4.6e5 / 18.9 | 294 / 2.8e4 / 0.83 | **21 / 760 / 0.029** | 22 / 820 / 0.031 |

Exact evaluations per iteration are the same for every configuration (one per non-pre-rejected iteration); the
flattened component at `T_t = T` and the carried step costs about as many gradients per iteration as `H_ADAPT`
(17–31 per iteration in total, ≈ plain), the tempered ref at ½ step and 3T 1.4–2.4× that (33–67). On every problem
the two small-τ rows are the cheapest per crossing in all three currencies (2–750× cheaper than the prior draw, 1.2–70×
than the tempered ref). ESS/s of the τ 0.05 row: 46 / 205 / 103 / 90 / 52 / 70 (B2, M3-5, M3-10, M3-20, M10-10, M30-10)
vs prior draw 15 / 66 / 24 / 18 / 36 (not mixed) / 1 210 (not mixed) and tempered ref 348 (not mixed) / 69 / 39 / 29 /
18 / 5 (not mixed). ESS/s on M3-10 / M10-10 is depressed by host load (Read first).

**(iii) Does the flattened component need the halved step? No.** Half vs carried step at T_t = 3T, crossings per 1 000:
τ 0.05 — B2 20.0 vs 14.4, M3-5 66.4 vs 66.1, M3-10 49.2 vs 49.2, M3-20 33.5 vs 35.6, M10-10 42.5 vs 43.3, M30-10 24.7
vs 25.8; τ 0.2 — 5.4 vs 2.0, 55.1 vs 54.0, 10.4 vs 11.0, 0.44 vs 0.60, 41.9 vs 40.8, 29.7 vs 29.1. The halved step
doubles the component's gradients for no gain on the M-family; only on B2 it lowers the out-of-box share (0.096 → 0.041
at τ 0.05) and raises crossings 1.4–2.7×. Out-of-box at the carried step is 0.000–0.008 (cell means; max single run 0.014) on all M problems for
every flattened row (plain 0.000–0.009), unlike the tempered component (study 31: 0.15–0.27 at the carried step).
Also `T_t = T` ≥ `3T` on 5 of 6 problems at τ 0.05 (B2 22.7 vs 14.4, M3-5 75 vs 66, M3-10 64 vs 49, M3-20 51 vs 36,
M30-10 33 vs 26; M10-10 33 vs 43) at a third of the component's gradients.

**(iv) Dimension curve d = 3 → 10 → 30 at Δ = 10** (`figures/fig_crossings_vs_dimension.png`):

| mechanism | d = 3 | d = 10 | d = 30 | mixed (3/10/30) | component's sub-chain acceptance (3/10/30) |
|---|---|---|---|---|---|
| prior draw | 11.8 | 0.95 | 0.05 | 3/0/0 | 0.12 / 0.0081 / 0.00008 |
| tempered ref | 42.6 | 20.1 | 2.16 | 3/3/0 | 0.39 / 0.18 / 0.021 |
| flat τ 0.05 T carried (fixed row) | 63.6 | 32.6 | 33.2 | 3/3/3 | 0.37 / 0.29 / 0.19 |
| best flattened row (per d, after the fact) | 63.6 (τ 0.05 T) | 49.6 (τ 0 3T) | 33.2 (τ 0.05 T) | 3/3/3 | – |

Prior draw falls ~240× over d = 3 → 30, tempered 20×, flattened 1.9× (and not at all from 10 to 30). Who crossed:
prior-draw component 0.98 / 0.84 / 0 of the crossings, tempered 1.00 / 0.99 / 0.96, flattened τ 0.05 1.00 / 1.00 /
1.00. Caveat: on this family the flattened barrier shrinks with d (table in (i)); the curve says the flattened
component survives a growing dimension *with the barrier made by the prior*; a family whose barrier stays at Δ in
the flattened potential was not run.

**(v) Does combining help? No.** The combined row (τ 0.2 + tempering β 0.03, f 0.35, 3T, ½ step) gives fewer crossings
than flat τ 0.05 (any variant) on every problem (B2 5.0, M3-5 50, M3-10 34, M3-20 17.5, M10-10 22, M30-10 6.7) and fails
on B2 (0/3 mixed). Against flat τ 0.2 ½ step alone it helps where τ 0.2 leaves a barrier (M3-10 34 vs 10.4, M3-20 17.5
vs 0.44) and hurts where it does not (M10-10 22 vs 42, M30-10 6.7 vs 30: the tempering drops the component's sub-chain
acceptance to 0.04 at d = 30). It never beats lowering τ.

**(vi) Recommendation for the C6 placeholders.** From these 252 runs: replace the tempered-momentum component of the
C6 placeholder by the flattened one,

| placeholder | recommended | evidence |
|---|---|---|
| component | `Hamiltonian(integrator="dimension_robust", mass="covariance", adaptive=False, likelihood_weight=0.05)` | the only setting mixed on all six problems (18/18); cheapest per crossing on all six (ii) |
| τ | **0.05** (τ = 0 equivalent within the spread: 18/18, 26–66 per 1 000) | τ ≥ 0.2 fails where τ(Δ + d/2) − d/2 > ~1 nat (M3-10 partially, M3-20) |
| T_t | **= T of the adaptive component** (not 3T) | T ≥ 3T on 5/6 problems at ⅓ the gradients |
| step | **the carried step** (`step_size=None`) — no relative-step option needed | ½ step: no gain on M, ×1.4 on B2, double gradients (iii) |
| p | 0.3 | only value run for the flattened rows (no p scan in note 32's grid) |

Not supported by this study: (a) problems whose barrier is made by the likelihood inside the prior bulk (there τ ≈ 0
makes the trajectory ignore the barrier but also the modes, and the exact-energy acceptance pays (1 − τ) Δ log L for a
jump between unequal modes — expected to fail for strongly unequal modes; not run); (b) other p, other T; (c) d > 30;
(d) unequal mode weights in the M-family. The tempered-momentum component is **not robust enough** for C6 (B2 fails at
every setting tried in studies 31 and 33; d = 30 not mixed).

---

## 4. Caveats

* 3 repetitions; DAMH-SMU not reproducible; ranges given.
* The M-family's barrier is mostly made by the prior (modes at prior radius √d) and the family is symmetric about the
  prior mean; the success of τ → 0 (close to a prior rotation) may be specific to that geometry. **Sharper (manager's addition,
  2026-10-13): with τ = 0 and `dimension_robust` the trajectory is the exact prior rotation by the angle T; at T = π it is
  the antipodal map u → −u, which sends one M-mode EXACTLY onto the other. The flattened rows use T_t = T = 3 with jitter
  0.5, i.e. T ∈ [1.5, 4.5], which contains π, and τ = 0.05 is close to that. So on the M-family a large part of the
  flattened component's crossings can come from the problem's point symmetry rather than from a generic mechanism; the
  T_t = 3T rows (T ∈ [4.5, 13.5], containing 3π ≈ 9.4) have the same property. B2 is the only run where this symmetry is
  absent, and there the flattened component still mixed 3/3 with 3× the prior draw's crossings — but d = 2. The
  recommendation below therefore needs a test on an asymmetric, likelihood-made barrier in d ≥ 10 (e.g. modes at different
  prior radii, or a non-centred second mode) before it is adopted.** B2 (asymmetric, d = 2,
  τ 0.05: 22.7 per 1 000, 3/3) is the only counter-check.
* Half steps are absolute per problem (0.40–0.51 of each run's carried step), and M3-5/M10-10/M30-10's were derived
  from the carried step, not from the column named in the brief (Read first).
* Host load inflated wall times on the M3-10 and M10-10 blocks; ESS/s and s/crossing there are pessimistic.
* "ok" (posterior check) and "mixed" (R̂_mode) disagree on 40 runs (all "ok but not mixed", e.g. tempered ref M30-10
  3/3 ok, 0/3 mixed; plain M3-5 2/3 ok, 0/3 mixed); "mixed" is the stricter verdict used above. No run was "mixed but poor".
* Crossing attribution uses the iteration that completed the hysteresis transition; the per-component outer acceptance
  is ≥ 0.93 everywhere except the components that almost never move (B2 tempered 0.33, M30-10 prior draw 0.33 on a
  handful of moves).
* Per-component out-of-box share is not observable (oob is counted per chain in `notes`); totals are given.
* `RuntimeWarning: overflow encountered in cast` from the network training appears in 50 of 252 logs (pre-existing,
  as in study 31; not investigated).
* Items for `10_manual_review_notes.md` (not written there by this study): (a) recommendation (vi) changes the C6
  placeholder from tempered to flattened momentum — needs an author decision and a test on a likelihood-made barrier;
  (b) the tempered component falls 20× from d = 3 to 30 at Δ = 10 — note 30's "scales with dimension" claim is not
  supported; (c) the brief/note-31 naming "stage 2" is ambiguous (1-based "second stage" vs study 31's index-2
  `step_end_stage2`); the half steps here follow the carried step.

## 5. Paths

`toy_examples/tmp_flattened_study_2026-10-13/`:
* `runs/<problem>_<label>_r<rep>/` (format-v2 output + `study_case.json` + `metrics.json`), `runs/<name>.log`
  (progress lines stripped); labels `plain`, `prior`, `tempref`, `flat_tau<τ>_T<1|3>`, `flathalf_tau<τ>_T3`, `comb`;
* `jobs/phase1.jsonl` (6), `jobs/phase2.jsonl` (246); `half_steps.json`;
* `refs/ref_B2.json`, `refs/ref_M3-{5,10,20}.json` (copied from study 31), `refs/ref_M10-10.json`, `refs/ref_M30-10.json`;
* `table_all.csv` (per run, 252 rows), `table_configs.csv` (per cell), `tables.md` (the §2 tables);
* `figures/fig_crossings_vs_dimension.png`, `figures/fig_crossings_vs_depth.png`;
* `logs/time_log.txt`, `logs/batch_progress.txt`, `logs/batch_phase2.out` (no `failed.txt`: no failures);
* `scripts/{problems,run_case,batch,jobs,analyze,collect,summary}.py` (adapted from study 31);
* `smoke/` (2 smoke runs, not in the tables).
