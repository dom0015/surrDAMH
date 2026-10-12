# 27 — Efficiency and robustness with/without Polyak averaging (S3) and the error model (S1) (2026-10-09)

**Dated record of one campaign.** Everything below was produced by running the code on 2026-10-09
(working tree of `working_Kuba` with the uncommitted automatic mode, S0, S1, S2, S3; the package was not
modified). Raw runs, drivers, tables and figures: `toy_examples/tmp_s3_s1_study_2026-10-09/`
(`runs/<name>/` format-v2 output + `study_case.json` + `metrics.json`, `runs/<name>.log`;
`runs/reps/` the repetitions; `ref_P1.json`, `ref_P2.json`; `tables_single_run.md`,
`tables_repetitions.md`, `table_*.csv`; `fig_*.png`; `scripts/{problems,run_case,analyze,tables,reps,
ref_P1_quadrature,make_ref_P2}.py`, `scripts/batch*.sh`). Untracked, 785 MB.

**Read first (limits of what was measured):**

* **P2/P2b is a weakly informed problem: the posterior is close to the prior** (reference sd 0.80–1.02
  in every coordinate, |mean| ≤ 0.45). The study used the problem the brief specified
  (`Solver_diffusion_GRF(no_parameters=20)` defaults, noise sd 0.1); a harder GRF setting (note 19's P1)
  was not run. Consequently **P2b did not produce a surrogate blind spot** (`audit_hidden = 0` in all 40
  P2b runs, with and without S1), and "does S1 help in P2b" is answered "could not check" (§4).
* The forward models are cheap (P2: 0.2 ms per solve), so **wall time measures surrogate/MPI overhead**, and
  the number of networks the collector delivers per exact evaluation depends on the sampler's speed: in the
  robust mode on P1 a whole run takes 0.8 s and the chain installs only 4–6 networks per chunk. An
  expensive model would deliver many more networks per evaluation; this regime was **not** studied.
* Runs are not bitwise reproducible: the warm-up and the held-out set are identical in every run of a
  problem (checked: `cmp` of `alg0000_MH-adaptive/rank0000.csv` and `surrogate_test_data.npz` across runs
  and configurations), but the DAMH phase depends on when the asynchronously trained networks arrive.
  Each configuration was therefore run **5 times with the library's seeds** (brief: once; the 4 extra
  runs were possible because time allowed); the spread over the 5 runs is the DAMH-phase variability only
  (the LHS design / warm-up is fixed), so it **underestimates** the total Monte-Carlo error.

---

## 1. Setup

| | P1 (2-D, nonlinear) | P2 (GRF, d = 20) | P2b (P2, far start) |
|---|---|---|---|
| forward model | `exp-sin` of `damh_surrogate_from_data_2d.ipynb`, `G(u) = (u1·e^{0.3 u2}, u2 + 0.5 sin 2u1)`, `scripts/problems.py::ExpSinSolver` | `grf_diffusion.py::Solver_diffusion_GRF(no_parameters=20)` (defaults: Matérn, ℓ 0.7, σ 0.1, 3×3 observations) | same |
| prior / data / noise | N(0, I); **noise-free** data `G(1.0, −0.8)` (the notebook adds a noise draw from an unseeded-here rng; not reproduced); sd 0.1 | N(0, I) (dimension-free, box ±8 by S0); data `G(ones)`; sd 0.1 | same; `initial_sample_type="user_specified"`, `initial_samples_distribution = Normal(mean=−2·ones, sd=1e-8)` (= the point −2·ones for both chains; Auto keeps a user-chosen type) |
| reference | quadrature on [−4, 4]², 2401² points: mean (0.9757, −0.7883), sd (0.1177, 0.0995); change vs a 1201² grid 5·10⁻¹³ sd | adaptive RW MH, 2 chains × 150 000 evaluations, first 20 000 per chain excluded (`runs/ref_P2_mh`, 37 s): min ESS 3936 (of 260 002 samples), R̂ 1.002 | as P2 |
| reference error | negligible | MC error of the mean ≤ 1/√3936 = **0.016 sd**, of the sd ≈ 1/√(2·3936) = **0.011** (relative) | as P2 |
| budget (total, incl. held-out set) | 4 000 | 12 000 | 12 000 |
| Auto plan (2 chains) | held-out 80, B = 1960, warm-up 98, chunks 465/465/465/467 | held-out 240, B = 5880, warm-up 400, chunks 1370 ×3 / 1370 | as P2 |

The reference MH for P2 ran with `mpiexec -n 2` and `use_collector=False` (2 chains, no collector needed);
every other run used `mpiexec -n 3` (2 samplers + collector, in-process solver), sequentially.

**What varies.** For each problem and mode (`robust`, `fast`) `plan_auto(problem, conf, budget, mode)` was
resolved, `conf._set_by_auto(**plan.conf_settings)` applied, and on every DAMH chunk only
`stage.polyak_averaging` and `stage.surrogate_error_model` were set to the four combinations; then
`problem.run_sampling(conf, plan.stages, surrogate_updater=plan.surrogate_updater,
surrogate_test_data=plan.surrogate_test_data)` (`scripts/run_case.py`). S2 stayed at Auto's 0.05/0.05.

**Metrics** (`scripts/analyze.py`): posterior = the 4 DAMH chunks (warm-up is burn-in), both chains pooled;
err mean = max_j |mean_j − ref_j| / sd_ref_j, err sd = max_j |sd_j − sd_ref_j| / sd_ref_j; ESS = minimum
over parameters of `Samples.calculate_effective_sample_size(stages_to_disp=chunks)`; **ESS/eval = min ESS /
all exact evaluations of the run** (warm-up + chunks + audits + held-out set = the budget); ESS/s with
the wall time of `run_sampling` on rank 0; outer acceptance and pre-rejection rate summed over the chunks;
exact steps, audits, `audit_hidden`, `polyak_updates` from `notes`; rmse from the last row of
`surrogate_quality.csv` (snapshots) and `surrogate_quality_test.csv` (held-out), `tr Σ_surr` =
`error_cov_trace`. For calibration, the max over d parameters of |mean error| from ESS independent
draws is about √(2 ln 2d)/√ESS (≈ 2.4/√ESS for d = 20, ≈ 1.7/√ESS for d = 2); the tables give 1/√ESS.

---

## 2. P1 (exp-sin, 2-D, budget 4 000)

Single run per configuration (the brief's matrix; `runs/P1_<mode>_polyak<0|1>_em<0|1>/`):

| mode | Polyak | S1 | err mean | err sd | 1/√ESS | ESS min | ESS/eval | ESS/s | wall s | exact evals | R̂ max | outer acc | prerej | exact steps acc/all | audited (hidden) | Polyak installs | rmse snap | rmse held-out | tr Σ_surr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust | on | on | 0.072 | 0.023 | 0.044 | 511.5 | 0.128 | 611 | 0.837 | 4000 | 1.009 | 0.485 | 0.493 | 76/348 | 176 (0) | 19 | 0.0195 | 0.209 | 0.00555 |
| robust | on | off | 0.1 | 0.073 | 0.09 | 123.7 | 0.0309 | 173 | 0.715 | 4000 | 1.014 | 0.462 | 0.594 | 85/408 | 264 (76) | 17 | 0.0383 | 0.273 | (off) |
| robust | off | on | 0.054 | 0.028 | 0.042 | 554.3 | 0.139 | 680 | 0.816 | 4000 | 1.003 | 0.526 | 0.525 | 86/366 | 202 (0) | – | 0.0154 | 0.21 | 0.00557 |
| robust | off | off | 0.039 | 0.039 | 0.062 | 260.1 | 0.065 | 364 | 0.715 | 4000 | 1.001 | 0.549 | 0.623 | 113/463 | 282 (48) | – | 0.0312 | 0.271 | (off) |
| fast | on | on | 0.051 | 0.033 | 0.034 | 848.1 | 0.212 | 73.4 | 11.6 | 4000 | 1.002 | 0.755 | 0.188 | 41/258 | 15 (1) | 373 | 0.0024 | 0.0533 | 0.000203 |
| fast | on | off | 0.022 | 0.057 | 0.045 | 488.2 | 0.122 | 44.9 | 10.9 | 4000 | 1.002 | 0.763 | 0.195 | 33/245 | 29 (2) | 372 | 0.000817 | 0.0614 | (off) |
| fast | off | on | 0.031 | 0.038 | 0.022 | 2136 | 0.534 | 191 | 11.2 | 4000 | 1 | 0.869 | 0.191 | 30/246 | 29 (0) | – | 0.00328 | 0.0491 | 0.000178 |
| fast | off | off | 0.016 | 0.036 | 0.023 | 1972 | 0.493 | 186 | 10.6 | 4000 | 1.004 | 0.882 | 0.198 | 30/248 | 24 (1) | – | 0.000999 | 0.0602 | (off) |

Five runs per configuration (median [min–max]; run 1 is the row above, runs 2–5 in `runs/reps/`;
"audit hidden" = sum over the 5 runs):

| mode | Polyak | S1 | n | ESS/eval | err mean | err sd | outer acc | audit hidden (sum) | wall s |
|---|---|---|---|---|---|---|---|---|---|
| robust | on | on | 5 | 0.162 [0.128–0.181] | 0.044 [0.011–0.072] | 0.023 [0.01–0.042] | 0.487 [0.484–0.505] | 0 | 0.8 [0.741–0.837] |
| robust | on | off | 5 | 0.0445 [0.0309–0.0512] | 0.1 [0.069–0.12] | 0.046 [0.0097–0.085] | 0.477 [0.454–0.493] | 305 | 0.694 [0.686–0.715] |
| robust | off | on | 5 | 0.153 [0.136–0.197] | 0.038 [0.0069–0.054] | 0.024 [0.0089–0.033] | 0.515 [0.489–0.529] | 0 | 0.746 [0.697–0.816] |
| robust | off | off | 5 | 0.065 [0.0435–0.0802] | 0.039 [0.01–0.21] | 0.037 [0.019–0.048] | 0.541 [0.492–0.557] | 267 | 0.694 [0.69–0.718] |
| fast | on | on | 5 | 0.237 [0.208–0.318] | 0.04 [0.015–0.051] | 0.048 [0.029–0.063] | 0.763 [0.755–0.765] | 3 | 11.4 [11.3–11.6] |
| fast | on | off | 5 | 0.131 [0.107–0.189] | 0.022 [0.01–0.042] | 0.057 [0.016–0.082] | 0.765 [0.755–0.775] | 7 | 10.7 [10.5–10.9] |
| fast | off | on | 5 | 0.534 [0.51–0.605] | 0.031 [0.0044–0.049] | 0.038 [0.02–0.057] | 0.874 [0.864–0.878] | 2 | 11.2 [11.1–11.4] |
| fast | off | off | 5 | 0.555 [0.493–0.655] | 0.025 [0.016–0.049] | 0.042 [0.02–0.058] | 0.884 [0.876–0.89] | 3 | 10.5 [10.3–10.6] |

Outer acceptance per chunk (single runs, chunks 1–4) shows where Polyak averaging costs:

| run | chunk 1 | chunk 2 | chunk 3 | chunk 4 | Polyak installs per chunk (both chains) |
|---|---|---|---|---|---|
| fast, Polyak on, S1 on | 0.24 | 0.907 | 0.932 | 0.942 | 93, 94, 95, 91 |
| fast, Polyak on, S1 off | 0.257 | 0.916 | 0.931 | 0.943 | 92, 94, 95, 91 |
| fast, Polyak off, S1 on | 0.698 | 0.914 | 0.923 | 0.94 | – |
| fast, Polyak off, S1 off | 0.742 | 0.921 | 0.927 | 0.939 | – |
| robust, Polyak on, S1 on | 0.271 | 0.401 | 0.606 | 0.679 | 4, 4, 6, 5 |
| robust, Polyak on, S1 off | 0.132 | 0.282 | 0.711 | 0.746 | 4, 5, 4, 4 |
| robust, Polyak off, S1 on | 0.277 | 0.549 | 0.595 | 0.697 | – |
| robust, Polyak off, S1 off | 0.186 | 0.54 | 0.734 | 0.75 | – |

Findings (P1):

* **S1 helps the robust mode 2.4–3.6× in ESS/eval** (medians 0.162 vs 0.0445 with Polyak, 0.153 vs 0.065
  without; the ranges do not overlap) and removes the hidden-mass signal: `audit_hidden` 0 in all 10 S1-on
  robust runs vs 305 and 267 (sums over 5 runs) with S1 off. Accuracy is within MC error either way
  (err mean ≤ 0.12 except one S1-off/Polyak-off run at 0.21; 1/√ESS 0.04–0.09). The robust network is poor on
  P1 (held-out rmse 0.21–0.27 vs noise 0.1; snapshot rmse 0.015–0.04) because the collector trains only a few
  networks during the 0.8 s run; S1 inflates the surrogate likelihood accordingly (tr Σ_surr 0.0056).
* **Polyak averaging costs the fast mode 2.3–4.2× in ESS/eval** (0.237 vs 0.534 with S1, 0.131 vs 0.555
  without; ranges do not overlap). The cost is entirely in chunk 1: outer acceptance 0.24–0.26 with Polyak
  vs 0.70–0.74 without, chunks 2–4 equal within 0.01. In chunk 1 the running mean includes the first
  networks (trained on ~200 warm-up snapshots) for the whole chunk, while the plain install follows the
  improving network.
* With Polyak on in the fast mode, S1 recovers part of the loss (0.237 vs 0.131); with Polyak off S1 changes
  nothing measurable (0.534 vs 0.555).
* Robust mode, Polyak: no consistent effect (0.162 vs 0.153 with S1; 0.0445 vs 0.065 without S1, ranges
  [0.031–0.051] vs [0.044–0.080]); only 4–6 installs per chunk, so the averaging barely acts here.
* Fast vs robust (best flags each): 0.555 vs 0.162 ESS/eval, 3.4×.

---

## 3. P2 (GRF, d = 20, budget 12 000)

Single run per configuration:

| mode | Polyak | S1 | err mean | err sd | 1/√ESS | ESS min | ESS/eval | ESS/s | wall s | exact evals | R̂ max | outer acc | prerej | exact steps acc/all | audited (hidden) | Polyak installs | rmse snap | rmse held-out | tr Σ_surr |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust | on | on | 0.22 | 0.19 | 0.066 | 226.5 | 0.0189 | 65 | 3.48 | 12000 | 1.031 | 0.884 | 0.651 | 343/1438 | 927 (0) | 315 | 0.000841 | 0.000476 | 9.14e-07 |
| robust | on | off | 0.11 | 0.14 | 0.062 | 256 | 0.0213 | 80.8 | 3.17 | 12000 | 1.011 | 0.884 | 0.649 | 341/1431 | 926 (0) | 311 | 0.000396 | 0.000451 | (off) |
| robust | off | on | 0.14 | 0.16 | 0.064 | 245.7 | 0.0205 | 75.4 | 3.26 | 12000 | 1.015 | 0.887 | 0.652 | 352/1449 | 925 (0) | – | 0.000383 | 0.000452 | 8.84e-07 |
| robust | off | off | 0.16 | 0.14 | 0.082 | 147.2 | 0.0123 | 48.6 | 3.03 | 12000 | 1.021 | 0.885 | 0.648 | 336/1466 | 890 (0) | – | 0.000321 | 0.000462 | (off) |
| fast | on | on | 0.038 | 0.037 | 0.011 | 8295 | 0.691 | 229 | 36.2 | 12000 | 1.001 | 0.944 | 0.191 | 150/677 | 82 (0) | 553 | 0.000425 | 0.000373 | 8.75e-07 |
| fast | on | off | 0.046 | 0.053 | 0.011 | 8233 | 0.686 | 245 | 33.6 | 12000 | 1.001 | 0.945 | 0.191 | 156/650 | 106 (0) | 554 | 0.000246 | 0.000354 | (off) |
| fast | off | on | 0.044 | 0.033 | 0.011 | 8604 | 0.717 | 242 | 35.5 | 12000 | 1 | 0.947 | 0.189 | 146/661 | 98 (0) | – | 0.000325 | 0.000356 | 9.04e-07 |
| fast | off | off | 0.049 | 0.03 | 0.011 | 7876 | 0.656 | 238 | 33 | 12000 | 1 | 0.949 | 0.189 | 157/665 | 88 (0) | – | 0.000224 | 0.000396 | (off) |

Five runs per configuration:

| mode | Polyak | S1 | n | ESS/eval | err mean | err sd | outer acc | audit hidden (sum) | wall s |
|---|---|---|---|---|---|---|---|---|---|
| robust | on | on | 5 | 0.0205 [0.0152–0.0217] | 0.14 [0.14–0.22] | 0.15 [0.14–0.19] | 0.883 [0.881–0.886] | 0 | 3.38 [3.32–3.48] |
| robust | on | off | 5 | 0.0146 [0.013–0.0213] | 0.17 [0.099–0.19] | 0.14 [0.13–0.15] | 0.884 [0.883–0.888] | 0 | 3.15 [3–3.2] |
| robust | off | on | 5 | 0.0205 [0.0157–0.0223] | 0.13 [0.11–0.15] | 0.14 [0.12–0.16] | 0.887 [0.887–0.887] | 0 | 3.26 [3.19–3.39] |
| robust | off | off | 5 | 0.0149 [0.0108–0.0217] | 0.15 [0.12–0.18] | 0.15 [0.14–0.16] | 0.885 [0.883–0.89] | 0 | 3.03 [3.01–3.13] |
| fast | on | on | 5 | 0.691 [0.685–0.693] | 0.046 [0.038–0.058] | 0.036 [0.032–0.05] | 0.945 [0.944–0.945] | 0 | 35.7 [35–36.3] |
| fast | on | off | 5 | 0.677 [0.646–0.686] | 0.05 [0.046–0.058] | 0.041 [0.029–0.053] | 0.945 [0.943–0.946] | 0 | 33.6 [32.4–33.9] |
| fast | off | on | 5 | 0.699 [0.685–0.717] | 0.05 [0.044–0.054] | 0.037 [0.024–0.047] | 0.949 [0.947–0.951] | 0 | 34.8 [34.6–35.5] |
| fast | off | off | 5 | 0.661 [0.656–0.722] | 0.047 [0.043–0.052] | 0.031 [0.03–0.042] | 0.95 [0.948–0.952] | 0 | 32.7 [32.3–33.1] |

Findings (P2):

* The surrogate is accurate from the first chunk on (held-out rmse 3.4–5·10⁻⁴ against noise sd 0.1), so
  **S1 is inert**: tr Σ_surr ≈ 9·10⁻⁷ against tr Σ_noise = 9 × 0.01 = 0.09. ESS/eval with S1 on vs off:
  robust 0.0205 vs 0.0146 (Polyak on), 0.0205 vs 0.0149 (off) — ranges overlap ([0.015–0.022] vs
  [0.011–0.022]); fast 0.691 vs 0.677 and 0.699 vs 0.661 (ranges overlap). **No measurable cost in ESS/eval;
  wall time +6–8 %** (fast: 35.7 vs 33.6 s, 34.8 vs 32.7 s; robust: 3.38 vs 3.15 s, 3.26 vs 3.03 s).
* **Polyak costs nothing measurable on P2**: fast 0.691/0.677 (on) vs 0.699/0.661 (off); robust
  0.0205/0.0146 vs 0.0205/0.0149; chunk-1 outer acceptance 0.925–0.931 (on) vs 0.946–0.948 (off) in the
  single runs, equal afterwards. Wall time +1 s (3 %) in the fast mode.
* Errors are at the MC level of each configuration: fast err mean 0.04–0.06 with 1/√ESS = 0.011 (expected
  max over 20 ≈ 0.027, plus the reference's own 0.016); robust 0.10–0.22 with 1/√ESS ≈ 0.06–0.08
  (expected max ≈ 0.15–0.2). No configuration is biased beyond that.
* **The robust mode is about as efficient as plain MH here**: 0.015–0.021 ESS/eval vs 0.0151 for the
  reference MH (3936 / 260 002 post-burn-in evaluations, 0.0131 counting its burn-in). Pre-rejection
  0.65 and outer acceptance 0.88: the sub-chain-1 random walk on the surrogate wastes the surrogate. The
  fast mode is 0.66–0.70 ESS/eval, **~34× the robust mode and ~45× MH per evaluation**.
* The fast mode's Hamiltonian proposals leave the S0 box (±8) in 2–7 % of the DAMH iterations (P2 fast
  4.1 %, P2b fast 3.9 %, P1 fast 7.4 %, manual Hamiltonian 1.9–2.3 %; robust 0 %), counted in `notes`
  `out_of_bounds`. On posteriors with sd ≤ 1 this means diverging trajectories; the cause was **not
  investigated** (could not check).

---

## 4. P2b (P2 started at −2·ones, budget 12 000)

Single run per configuration ("evals to 0.5 sd" = exact evaluations of the chain, warm-up included,
until the running mean of the chain from the first DAMH chunk on is within 0.5 sd_ref of the reference
mean in every coordinate; per chain; exact evaluations per iteration from `subchain_stats`
(`outer_proposed_changed` or `kernel = 1`), audits spread proportionally to match `notes`):

| mode | Polyak | S1 | err mean | err sd | 1/√ESS | ESS min | ESS/eval | ESS/s | wall s | exact evals | R̂ max | outer acc | prerej | exact steps acc/all | audited (hidden) | Polyak installs | rmse snap | rmse held-out | tr Σ_surr | evals to 0.5 sd (chain 0, 1) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| robust | on | on | 0.17 | 0.13 | 0.074 | 181 | 0.0151 | 55.5 | 3.26 | 12000 | 1.011 | 0.882 | 0.649 | 344/1437 | 914 (0) | 237 | 0.000545 | 0.000458 | 9.56e-07 | 897, 787.3 |
| robust | on | off | 0.13 | 0.16 | 0.068 | 213.9 | 0.0178 | 68.8 | 3.11 | 12000 | 1.023 | 0.878 | 0.651 | 333/1441 | 923 (0) | 201 | 0.000526 | 0.000499 | (off) | 1700, 865 |
| robust | off | on | 0.12 | 0.16 | 0.065 | 235.1 | 0.0196 | 69.1 | 3.4 | 12000 | 1.019 | 0.887 | 0.648 | 314/1402 | 939 (0) | – | 0.000632 | 0.000459 | 8.88e-07 | 903.8, 802.7 |
| robust | off | off | 0.15 | 0.13 | 0.096 | 108.2 | 0.00902 | 35.5 | 3.05 | 12000 | 1.04 | 0.887 | 0.654 | 328/1405 | 977 (0) | – | 0.000301 | 0.000453 | (off) | 1298, 1301 |
| fast | on | on | 0.065 | 0.032 | 0.011 | 8101 | 0.675 | 228 | 35.5 | 12000 | 1 | 0.943 | 0.195 | 160/655 | 115 (0) | 554 | 0.00021 | 0.000343 | 8.51e-07 | 427.1, 418.3 |
| fast | on | off | 0.045 | 0.04 | 0.011 | 7664 | 0.639 | 232 | 33.1 | 12000 | 1 | 0.944 | 0.191 | 172/674 | 89 (0) | 554 | 0.000152 | 0.000381 | (off) | 413.1, 433.4 |
| fast | off | on | 0.05 | 0.037 | 0.011 | 7898 | 0.658 | 226 | 35 | 12000 | 1 | 0.95 | 0.189 | 164/657 | 102 (0) | – | 0.000221 | 0.000367 | 8.2e-07 | 413.1, 423.3 |
| fast | off | off | 0.039 | 0.029 | 0.011 | 8128 | 0.677 | 246 | 33.1 | 12000 | 1.001 | 0.949 | 0.192 | 166/671 | 94 (0) | – | 0.00017 | 0.000411 | (off) | 417.2, 409.1 |

Five runs per configuration (the 10 chain values pooled):

| mode | Polyak | S1 | n | ESS/eval | err mean | err sd | outer acc | audit hidden (sum) | wall s | evals to 0.5 sd (both chains pooled) |
|---|---|---|---|---|---|---|---|---|---|---|
| robust | on | on | 5 | 0.0182 [0.0151–0.0218] | 0.17 [0.16–0.19] | 0.14 [0.13–0.16] | 0.883 [0.879–0.883] | 0 | 3.29 [3.26–3.35] | 1135 [787.3–1679] |
| robust | on | off | 5 | 0.0178 [0.0138–0.0188] | 0.12 [0.11–0.15] | 0.16 [0.15–0.16] | 0.879 [0.878–0.881] | 0 | 3.11 [2.99–3.13] | 1242 [853–1796] |
| robust | off | on | 5 | 0.0196 [0.0107–0.021] | 0.12 [0.099–0.14] | 0.16 [0.14–0.18] | 0.886 [0.884–0.889] | 0 | 3.23 [3.16–3.4] | 1021 [802.7–1565] |
| robust | off | off | 5 | 0.018 [0.00902–0.0191] | 0.14 [0.11–0.2] | 0.13 [0.12–0.16] | 0.886 [0.884–0.887] | 0 | 3 [2.93–3.17] | 1147 [763.6–1573] |
| fast | on | on | 5 | 0.675 [0.664–0.704] | 0.065 [0.035–0.078] | 0.033 [0.031–0.046] | 0.943 [0.94–0.944] | 0 | 35 [34.6–35.5] | 420.2 [403–457.8] |
| fast | on | off | 5 | 0.693 [0.639–0.714] | 0.045 [0.038–0.054] | 0.035 [0.029–0.04] | 0.942 [0.941–0.946] | 0 | 33 [32.8–34.3] | 433.4 [413.1–452.4] |
| fast | off | on | 5 | 0.685 [0.658–0.714] | 0.051 [0.039–0.061] | 0.045 [0.036–0.046] | 0.949 [0.948–0.95] | 0 | 35 [34.4–35.2] | 420.8 [403–446.6] |
| fast | off | off | 5 | 0.677 [0.668–0.711] | 0.055 [0.039–0.058] | 0.035 [0.029–0.041] | 0.948 [0.947–0.949] | 0 | 33 [32.5–33.3] | 420.1 [409.1–441.5] |

![P2b running mean](../toy_examples/tmp_s3_s1_study_2026-10-09/fig_P2b_running_mean.png)

`fig_P2b_running_mean.png`: max_j |running mean_j − ref_j| / sd_ref_j vs exact evaluations, single runs,
solid chain 0, dashed chain 1. The first chunk starts at error ≈ 2 (the 400-evaluation warm-up from
−2·ones has not reached the bulk); the fast mode crosses 0.5 within ~20 evaluations of chunk 1
(median 420 incl. the 400 warm-up), the robust mode after median 1021–1242.

Findings (P2b):

* **No blind spot arose**: `audit_hidden = 0` in all 40 runs; the network trained on the far warm-up is
  already accurate in the bulk (held-out rmse 3.4–5·10⁻⁴). The S1 question for P2b is therefore
  **"could not check"** on this problem. Measured: S1 changes neither the recovery (fast 420.2/420.8 with
  S1 vs 433.4/420.1 without; robust 1135/1021 vs 1242/1147, all ranges overlapping) nor ESS/eval.
* Polyak: no measurable effect on recovery or ESS/eval (ranges overlap in every pair).
* ESS/eval matches P2 (fast 0.675–0.693, robust 0.018–0.020 medians): the far start costs only the
  warm-up on this near-prior posterior.

---

## 5. Manual GRF settings vs Auto (P2, budget 12 000)

Explicit stage lists with Auto's network, held-out set (240) and snapshot settings: warm-up = adaptive RW MH
2 000 evaluations per chain (burn-in), then one DAMH-SMU stage of 3 880 per chain (`runs/P2_manual_*`,
`runs/reps/P2_manual_*_rep{2..5}`). (a) adaptive RW, sub-chain 20; (b) `Hamiltonian(num_steps ∈ {10, 30},
integrator="dimension_robust", mass=1.0)`, sub-chain 1; (c) = (b) with `polyak_averaging` and
`surrogate_error_model` on. No S2 in the manual settings. Five runs each:

| setting | n | ESS/eval | ESS/s | err mean | err sd | outer acc | prerej | wall s |
|---|---|---|---|---|---|---|---|---|
| (a) RW, sub-chain 20 | 5 | 0.139 [0.13–0.142] | 172 [161–179] | 0.051 [0.045–0.067] | 0.032 [0.023–0.042] | 0.996 [0.995–0.996] | 0.655 [0.652–0.655] | 9.56 [9.49–9.7] |
| (b) Hamiltonian L 10 | 5 | 0.295 [0.282–0.309] | 306 [298–329] | 0.052 [0.034–0.061] | 0.037 [0.029–0.039] | 0.997 [0.996–0.997] | 0.199 [0.196–0.203] | 11.4 [11.3–11.9] |
| (b) Hamiltonian L 30 | 5 | 0.609 [0.583–0.638] | 289 [277–301] | 0.047 [0.039–0.065] | 0.032 [0.029–0.047] | 0.997 [0.996–0.997] | 0.198 [0.196–0.203] | 25.3 [25.2–25.4] |
| (c) L 10 + Polyak + S1 | 5 | 0.3 [0.268–0.311] | 298 [268–313] | 0.044 [0.038–0.063] | 0.038 [0.032–0.042] | 0.993 [0.992–0.994] | 0.2 [0.196–0.201] | 12.1 [11.9–12.1] |
| (c) L 30 + Polyak + S1 | 5 | 0.604 [0.598–0.629] | 266 [261–281] | 0.043 [0.04–0.057] | 0.04 [0.036–0.057] | 0.993 [0.992–0.994] | 0.201 [0.198–0.202] | 27.2 [26.8–27.4] |
| Auto robust (Polyak off, S1 on) | 5 | 0.0205 [0.0157–0.0223] | 75 [56–82] | 0.13 [0.11–0.15] | 0.14 [0.12–0.16] | 0.887 | 0.65 | 3.26 [3.19–3.39] |
| Auto fast (Polyak off, S1 on) | 5 | 0.699 [0.685–0.717] | 240 [236–247] | 0.05 [0.044–0.054] | 0.037 [0.024–0.047] | 0.949 [0.947–0.951] | 0.19 | 34.8 [34.6–35.5] |

(Auto rows from §3; prerej from the single runs.)

* **Best per exact evaluation: Auto fast** (0.699, L 30, warm-up 400) — 1.15× manual (b) L 30 (0.609)
  and 1.16× (c) L 30 (0.604); the difference is about the warm-up share (Auto 400 vs manual 2 000 of
  5 880 per chain, i.e. 0.93 vs 0.66 of the chain budget spent in DAMH). Per DAMH-stage evaluation the
  order reverses: manual L 30 0.609 × 12 000 / 7 760 = 0.94 vs Auto fast 0.699 × 12 000 / 10 960 = 0.77
  (the manual stage has no S2 exact steps and a surrogate trained on 4 000 warm-up snapshots instead of 800;
  which of the two matters was not separated). L 30 gives 2.1× L 10 per evaluation (0.609 vs 0.295).
* **Manual RW with sub-chain 20 is 6.8× Auto robust** (0.139 vs 0.0205) — the robust mode's sub-chain
  length 1 is what makes it no better than MH on this problem.
* Polyak + S1 in a single long DAMH stage (c vs b): 0.604 vs 0.609 (L 30), 0.300 vs 0.295 (L 10): no
  measurable effect on ESS/eval; ESS/s −3 % (L 10) and −8 % (L 30), wall +0.7 s and +1.9 s.
* Per second (cheap model only): manual (b) L 10 is best (306 ESS/s), 1.27× Auto fast (240).

---

## 6. Conclusions

1. **Polyak averaging (S3) costs efficiency where the surrogate is still learning, and only there.** On
   P1 the fast mode loses 2.3–4.2× ESS per exact evaluation (0.237 vs 0.534 with S1, 0.131 vs 0.555
   without; 5 runs each, ranges disjoint); the loss is confined to the first DAMH chunk (single runs: outer acceptance
   0.24–0.26 vs 0.70–0.74; chunks 2–4 equal within 0.01). On P2/P2b, where the network is accurate from the
   start (held-out rmse ≈ 4·10⁻⁴ vs noise 0.1), the cost is not measurable (fast 0.691/0.677 vs
   0.699/0.661; manual L 30 0.604 vs 0.609) and the wall time grows by ~3 %. In the robust mode on P1 the
   chain installed only 4–6 networks per chunk, so S3 barely acts there (no consistent effect).
2. **The error model (S1) helps where the surrogate is inaccurate and is free where it is accurate.** P1
   robust: 2.4–3.6× ESS/eval (0.162 vs 0.0445, 0.153 vs 0.065) and `audit_hidden` 0 vs 267–305 per 5 runs;
   P1 fast with Polyak: 1.8× (0.237 vs 0.131); P1 fast without Polyak: no change (0.534 vs 0.555). P2:
   tr Σ_surr ≈ 9·10⁻⁷ vs 0.09 noise, ESS/eval unchanged within run-to-run spread, wall time +6–8 %.
   **P2b: could not check** — the far start produced no blind spot on this near-prior posterior
   (`audit_hidden = 0` in all 40 runs, recovery to 0.5 sd unchanged by S1: fast ≈ 420, robust ≈ 1000–1250
   exact evaluations per chain).
3. **Recommendations for the Auto defaults, from these numbers:**
   * `AUTO_SURROGATE_ERROR_MODEL = True` (keep): largest gain measured in this study (P1 robust 2.4–3.6×),
     no ESS cost on P2, +6–8 % wall time on a 0.2 ms model (relatively less on an expensive one).
   * `AUTO_POLYAK_AVERAGING`: **do not apply it in the first DAMH chunk** (or start the average after a
     burn-in of the chunk's received networks); measured basis: the whole P1 loss is in chunk 1, chunks
     2–4 are unaffected and P2 shows no cost. If S3 is kept on in every chunk for the diminishing-adaptation
     argument, the price on a P1-like problem is 2.3–4.2× in the fast mode. A per-chunk or delayed variant
     needs a package change and was not tested.
   * S2 weights (0.05/0.05) were not varied in this study; nothing to recommend from it.
   * The robust mode's layout is the larger issue than either flag: on P2 it reaches only 0.015–0.021
     ESS/eval (≈ MH's 0.015), while the same random walk with sub-chain length 20 reaches 0.139 (6.8×); the
     fast mode reaches 0.66–0.70 (~34×). Auto fast beats the best manual Hamiltonian setting by 1.15× per
     exact evaluation of the whole budget (0.699 vs 0.609; its warm-up is 400 instead of 2 000 per chain),
     while per DAMH-stage evaluation the manual stage is ahead (0.94 vs 0.77, §5).
4. Open, not checked here: the expensive-model regime (many more network installs per evaluation, where S3
   presumably matters more), a harder GRF posterior (note 19 P1) where a blind spot can occur, and the cause of the
   2–7 % out-of-box Hamiltonian proposals.

---

## 7. Exact paths of every run

All under `toy_examples/tmp_s3_s1_study_2026-10-09/`. Each run directory has `sampling_output/` (format v2),
`study_case.json` (case, resolved stages, wall time) and `metrics.json`; the launch log is `<run>.log`, the
analysis log `<run>.analyze.log`.

| what | runs |
|---|---|
| P1 reference | `ref_P1.json` (quadrature, `scripts/ref_P1_quadrature.py`) |
| P2 reference | `runs/ref_P2_mh/` (+ `ref_P2.json`, `scripts/make_ref_P2.py`) |
| matrix, run 1 | `runs/{P1,P2,P2b}_{robust,fast}_polyak{1,0}_em{1,0}/` (24) |
| matrix, runs 2–5 | `runs/reps/{P1,P2,P2b}_{robust,fast}_polyak{1,0}_em{1,0}_rep{2,3,4,5}/` (96) |
| manual, run 1 | `runs/P2_manual_{a,b10,b30,c10,c30}/` (5) |
| manual, runs 2–5 | `runs/reps/P2_manual_{a,b10,b30,c10,c30}_rep{2,3,4,5}/` (20) |
| extra reproducibility check | `runs/repro_check_P1_fast_polyak0_em0/` (same configuration as `runs/P1_fast_polyak0_em0`; different DAMH samples, identical warm-up) |
| batch logs | `runs/batch_matrix.log`, `runs/batch_manual.log`, `runs/batch_reps.log`, `runs/batch_manual_reps.log` |
| tables / figures | `tables_single_run.md`, `tables_repetitions.md`, `table_{P1,P2,P2b}.csv`, `table_repetitions.csv`, `fig_P2b_running_mean.png`, `fig_ess_per_eval_repetitions.png` |

No run failed; nothing was skipped.
