# 29 — Step-size tuning study, Part 1 only (2026-10-11/12)

**Dated record of one campaign, written 2026-10-12 from the runs that finished.** The runs were made overnight
2026-10-11 22:02 to 2026-10-12 00:07 (working tree of `working_Kuba`, uncommitted; see §1.3 for the two package
changes the study added). The batch stopped after the Part-1 diagnostics. **Parts 2 and 3 of the specification
(exact safety kernel, re-tune with the kernel on) were never started: `jobs/p2.jsonl` holds 153 queued jobs and
no `runs/p2_*` directory exists. The author decided on 2026-10-12 not to continue the runs and to report only
what is done.** The fixed-step summary (`scripts/scan_summary.py`, `part1_scan.json`) was produced on 2026-10-12
during the write-up; it reads the finished runs only.

Raw runs, drivers, tables, figures: `toy_examples/tmp_step_size_study_2026-10-12/` (`runs/<name>/` format-v2
output + `study_case.json` + `metrics.json` + `trace.npz`, `runs/<name>.log`; `jobs/*.jsonl`; `refs/ref_{L10,B2,G20}.json`;
`part1_grid.csv`, `part1_best.json`, `part1_eps_star.json`, `part1_scan.json`; `figures/*.png`; `logs/{time_log,
batch_progress,failed}.txt`, `logs/pytest_{unit,mpi}.log`; `scripts/{problems,run_case,batch,jobs,analyze,collect,
part1_summary,part1_details,scan_summary,figures,ref_b2_quadrature,ref_l10,diag_outer}.py`; `diag/exactsur_T{1,5}/`).
Untracked, ~0.9 GB. 139 analysed runs (72 grid + 4 best-cell repetitions + 3 MH-Hamiltonian + 54 fixed-step +
5 diagnostics + 1 G20 reference) plus 2 local exact-surrogate diagnostics and one unanalysed smoke run.

**Read first (limits of what was measured):**

* **Only Part 1 was run** (is the adaptive step tuned correctly?), plus nine follow-up runs on 2026-10-12 (§2.7:
  the S1 error-model effect on L10 confirmed with 3 + 3 runs, and one smoke run per problem of the exact safety
  kernel). Nothing else here measures `exact_step_probability`, `exact_step_scale` or `audit_prerejected`.
* **The ESS/eval metric counts exact evaluations only** (whole budget incl. the held-out set). Surrogate
  trajectories are free in it. For the two cheap models (L10 linear, B2 1 µs) this is the metric of the
  specification, not of wall time; ESS/s is given alongside and ranks differently (§3).
* **Both grids end at their edges.** The best target acceptance is 0.9 on L10 and G20 (the largest tried),
  the best T is 5 on L10 (the largest tried), and the best fixed step is ε*/4 on L10 and G20 (the smallest
  tried). The optima are therefore bounds, not located.
* **The MH-Hamiltonian comparison (question vi) failed on L10 and B2** (§2.6): the step collapsed to 10⁻⁵–10⁻³
  within the first updates and never recovered in the budget. Only G20 gives a usable comparison.
* **3 repetitions, and on B2 the spread is as large as the effect** (best cell 0.020 / 0.032 / 0.052 ESS/eval).
  Single-run cells (12 of 16 per problem) are not ranked against each other below the replicate spread.
* **The G20 reference has its own Monte-Carlo error** (8 × 45 000 adaptive-RW states, min ESS 5 581); it is
  included in the SE of every G20 deviation. The L10 reference is exact, B2 is quadrature (3001² grid,
  quadrature change of the left-arm mass 1.4·10⁻⁴).
* DAMH-SMU runs are not reproducible (asynchronous network delivery); repetitions use
  `Configuration.seed_offset = 100 000 · rep` (new, §1.3), so they also shift the LHS starts and the warm-up.

---

## 1. Setup

### 1.1 Problems and references (`scripts/problems.py`, `refs/`)

| name | problem | d | network | held-out | warm-up / chain | budget / chain | reference |
|---|---|---|---|---|---|---|---|
| L10 | anisotropic linear-Gaussian (posterior sd log-spaced 0.5 → 0.05, random rotation, seed 20261011; the C1 "aniso10" setup) | 10 | (32, 32) | 200 | 1 000 | 4 000 | closed form (`ref_L10.json`) |
| B2 | `Solver_illustrative_local`, prior N(0, I), likelihood N(5, 1) (notes 27/28) | 2 | (32, 32) | 200 | 1 000 | 4 000 | trapezoid quadrature on [−8, 8]², 3001² (`ref_B2.json`); left arm {y − x > 1} holds 0.2933 |
| G20 | shipped GRF diffusion, `template_experiment_manual_test.py` setup, data at ones, noise sd 0.1 | 20 | (64, 64) | 256 | 400 | 2 000 | adaptive-RW MH, 8 chains × (5 000 + 45 000), `runs/ref_G20_mh` (`ref_G20.json`, with `se_mean`, `se_m2`) |

Every run: `mpiexec -n 9` (8 samplers + 1 collector), in-process solver, LHS starts, `min_snapshots_initial = 10 d`,
`NeuralNetworkUpdater(seed=rep)`, `polyak_averaging=True`, `surrogate_error_model=True`, default `prior_bound`.
Stage list: `Stage(proposal=RandomWalk(), max_evaluations=warm-up, burn_in=True)` then one
`Stage(algorithm="DAMH", proposal=Hamiltonian(integrator="dimension_robust", mass="covariance", integration_time=T,
integration_time_jitter=0.5, target_rate=a), max_evaluations=budget − warm-up, surrogate_model_updates=True)` with
`subchain_length=1` (default). The mass is the inverse pooled posterior covariance at stage start (manifest:
`mass_source = "covariance (n=100)"`, the burn-in statistics discounted to an effective 100 states), the dual
averaging starts fresh from `step_size_initial = 0.1` with `mu = log(10 · 0.1)`, `num_steps = max(1, round(T u / ε))`
capped at 1 000, `u ~ U(0.5, 1.5)` per trajectory.

Metrics (`scripts/analyze.py`): per-chain Geyer ESS summed over chains, min over parameters, for x and x²; divided by
exact evaluations = accepted + rejected + audited + held-out. Posterior check: max over parameters of
|mean − ref| in reference sd and in SE, |E x² − ref| in SE; **"poor" = any of these > 4 SE, or (B2) left-arm mass
off by > 0.1**. Outer acceptance = accepted / (accepted + rejected) (pre-rejections excluded); pre-rejection rate =
pre-rejected / all outer iterations; `oob` = out-of-bounds proposals / all; non-finite proposals parsed from the log.
Dual-averaging trace per chain from `adaptive_stats` (live `log_step_size`, frozen `log_step_size_bar`, mean
sub-chain acceptance per 10 updates, `num_steps`); `eps_bar` = exp of the chain-mean frozen log step at stage end.

### 1.2 Timeline (`logs/time_log.txt`, `logs/batch_progress.txt`)

| time | event |
|---|---|
| 22:02 | start; prerequisites P1/P2 implemented; unit 770 passed / 1 skipped (4:17), MPI 58 passed (6:20) |
| 22:12–22:19 | references (G20 MH 13 s, quadrature, closed form) |
| 22:19–22:45 | Part-1 grid, 72 runs |
| 22:45–23:24 | best-cell repetitions + MH-Hamiltonian runs; the B2 MH run took 1 010 s, so the MH comparison was cut to 1 run per problem (`jobs/p1best2.jsonl`) |
| 23:24–23:36 | fixed-step scan started without a time limit; `p1fix_B2_T3_f2_r0` killed after ~8 min (328 816 iterations, 327 091 pre-rejected, 1 726 of 3 000 exact evaluations per chain; `logs/p1fix_B2_T3_f2_r0_untimed_killed.log`); scan restarted with `time_limit = 90 s` on the DAMH stage |
| 23:36–00:07 | fixed-step scan, 54 runs, and 5 diagnostics; all return code 0 |
| 00:07 | batch ends; Part 2 not started |

### 1.3 Package changes made for the study (prerequisites P1, P2; by the study session, verified by me only through the test logs and the CHANGELOG entries)

* `Configuration.seed_offset: int = 0` — added to every seed stream of a run, recorded in the manifest; default
  byte-identical (CHANGELOG 2026-10-12, `tests/unit/test_seed_offset_exact_step_scale.py`).
* `Stage.exact_step_scale: float | None = None` — non-adaptive exact random walk with covariance
  `scale · 2.38²/d · Σ̂`; `None` = today's adaptive kernel, byte-identical (same CHANGELOG entry and test file).
  **Not exercised by any run of this study** (Part 2 would have).

---

## 2. Part 1 — adaptive step: grid over target acceptance a and integration time T

ESS/eval = min-over-parameters ESS per exact evaluation (mean over runs; ± = sd over 3 repetitions where present).
Full tables with ESS(x²), ESS/s, wall time: `part1_grid.csv`; per-cell markdown: `scripts/part1_details.py`.
Figure: `figures/fig_part1_ess_grid.png`.

### 2.1 L10 (ESS/eval; `eps_bar`; mean leapfrog steps L; outer acceptance)

| a \ T | 1 | π/2 | 3 | 5 |
|---|---|---|---|---|
| 0.6 | 0.055 ± 0.006 (3); ε 0.54, L 1.9, outer 0.31 | 0.064; ε 0.50, L 3.2, outer 0.32 | 0.083; ε 0.48, L 6.2, outer 0.46 | 0.111; ε 0.47, L 10.7, outer 0.52 |
| 0.7 | 0.049; ε 0.45, L 2.2, outer 0.27 | 0.060 ± 0.016 (3); ε 0.49, L 3.3, outer 0.33 | 0.077; ε 0.44, L 6.8, outer 0.44 | 0.098; ε 0.43, L 11.6, outer 0.50 |
| 0.8 | 0.049 (poor, E x² 5.2 SE); ε 0.39, L 2.6, outer 0.25 | 0.049; ε 0.38, L 4.1, outer 0.30 | 0.093 ± 0.036 (3); ε 0.41, L 7.4, outer 0.47 | 0.099; ε 0.37, L 13.5, outer 0.50 |
| 0.9 | 0.053; ε 0.28, L 3.6, outer 0.27 | 0.071; ε 0.29, L 5.5, outer 0.33 | 0.078; ε 0.28, L 10.6, outer 0.46 | **0.116 ± 0.039 (3)**; ε 0.30, L 16.9, outer 0.53 |

T dominates: every column of T = 5 beats T = 3 beats π/2 beats 1, at every a. The dependence on a is weak
(T = 5: 0.111, 0.098, 0.099, 0.116). Best cell (a = 0.9, T = 5) replicates: 0.100 / 0.088 / 0.161 with
`eps_bar` 0.278 / 0.346 / 0.277. Pre-rejection rate = 1 − a within 0.003 everywhere; `oob` ≤ 0.005; non-finite 0.
1 poor run of 22 (a = 0.8, T = 1: E x² off by 5.2 SE, mean fine at 1.4 SE).

### 2.2 B2

| a \ T | 1 | π/2 | 3 | 5 |
|---|---|---|---|---|
| 0.6 | 0.015 ± 0.006 (3, **2 poor**); ε 0.95, L 1.4, outer 0.14, oob 0.008 | 0.026 (**poor**); ε 0.33, L 4.3, outer 0.28, oob 0.057 | 0.015; ε 0.20, L 13.8, outer 0.46, oob 0.13 | 0.007; ε 0.15, L 32, outer 0.60, oob 0.18 |
| 0.7 | 0.021; ε 0.36, L 2.6, outer 0.22 | 0.019 ± 0.007 (3, **2 poor**); ε 0.57, L 3.2, outer 0.17 | 0.013; ε 0.18, L 15.4, outer 0.46, oob 0.083 | 0.009; ε 0.15, L 32, outer 0.58, oob 0.12 |
| 0.8 | 0.021; ε 0.30, L 3.3, outer 0.23 | 0.025; ε 0.23, L 6.4, outer 0.34 | **0.035 ± 0.017 (3)**; ε 0.29, L 10.8, outer 0.28, oob 0.043 | 0.009; ε 0.14, L 35, outer 0.57, oob 0.061 |
| 0.9 | 0.016; ε 0.20, L 4.7, outer 0.28 | 0.027 ± 0.008 (3); ε 0.30, L 5.4, outer 0.18 | 0.016; ε 0.15, L 18.9, outer 0.46 | 0.029 ± 0.020 (3, 1 poor); ε 0.19, L 28, outer 0.38 |

Noisy: the three best-cell replicates are 0.020 / 0.032 / 0.052 with `eps_bar` 0.18 / 0.29 / 0.39, and the cell
(0.9, π/2) at 0.027 ± 0.008 is not distinguishable from it. After the single-run grid the best cell was (0.9, π/2);
it became (0.8, 3) after the repetitions (`jobs/p1best*.jsonl` use (0.9, π/2) for the MH comparison for that reason).
6 poor runs of 22, all at a ≤ 0.7 or T = 5: left-arm mass 0.18–0.37 against 0.293, mean off by 3–9 SE. Arm crossings
per run: 410–1 400 over 8 chains, most at T = 3 (1 408 in the best cell) and fewest at T = 5 (410–440), where
`oob` reaches 0.06–0.18 (trajectories leave the box of ±8 and are pre-rejected).

### 2.3 G20

| a \ T | 1 | π/2 | 3 | 5 |
|---|---|---|---|---|
| 0.6 | 0.63 ± 0.11 (3, **3 poor**); ε 2.94, L 1.0 | 0.54 (**poor**); ε 2.69, L 1.0 | 0.83 (**poor**); ε 2.59, L 1.2 | 0.68; ε 2.18, L 2.3 |
| 0.7 | 0.38 (**poor**); ε 2.09, L 1.0 | 0.51 ± 0.10 (3, 1 poor); ε 2.32, L 1.0 | 0.87; ε 2.03, L 1.5 | 0.72; ε 1.93, L 2.6 |
| 0.8 | 0.23; ε 1.65, L 1.0 | 0.32; ε 1.66, L 1.0 | 0.95 ± 0.06 (3); ε 1.89, L 1.6 | 0.72; ε 1.71, L 3.0 |
| 0.9 | 0.15; ε 1.25, L 1.0 | 0.27; ε 1.29, L 1.1 | **1.09 ± 0.04 (3)**; ε 1.52, L 2.1 | 0.79 ± 0.04 (3); ε 1.52, L 3.3 |

Outer acceptance is 0.97–0.98 in every cell (near-prior posterior, accurate network). The adapted step is 1.2–2.9,
larger than T = 1 and T = π/2, so those columns run a single leapfrog step (L = 1) and ESS/eval falls with a
(0.63 → 0.15 at T = 1): with L = 1, a larger a only shortens the jump. At T = 3 and 5 the trend reverses
(0.83 → 1.09 at T = 3). Best cell (0.9, 3) replicates 1.05 / 1.10 / 1.12, `eps_bar` 1.41 / 1.66 / 1.50.
7 poor runs of 22, all at a ≤ 0.7 and T ≤ 3: mean off by 3.7–6.8 SE (0.05–0.10 reference sd), E x² by 5–10 SE.
Diagnostic with the (0.6, 1) step held fixed (ε = 2.936, 3 runs, `p1diag_G20_fixed_T1_eps2.936_r*`): mean off by
2.4–3.0 SE, E x² by 3.2 / 9.7 / 2.0 SE, so 1 of 3 poor against 3 of 3 adaptive. **Not resolved** whether the
low-target bias comes from the adaptation or from the large step itself; 3 runs do not separate them.

### 2.4 Answers to the Part-1 questions

**(iii) Reached acceptance vs target.** The second-half mean sub-chain acceptance equals the target within
0.001 on every problem and target (`part1_details.py`, column `alpha2`). The dual averaging hits its target.

**(ii) Does the live step keep drifting after the frozen one settled?** L10 and G20: no. The second-half drift
of the live log-step (mean of the last 10 % minus the first 10 % of the second half) is −0.04 … +0.02, and the
live and frozen geometric means agree within 2 %. **B2: yes.** The drift is −0.27 (a = 0.9) to −0.49 (a = 0.7)
in log units per half stage, i.e. the step is still falling by 24–39 % during the second half of 3 000
evaluations per chain, and the frozen step follows it (`figures/fig_part1_traces_best.png`, middle panel). The
B2 adaptation has not converged at this budget, which is also why its replicates differ by 2× in `eps_bar`.

**(iv) Restart transient.** The stage starts at ε₀ = 0.1 with `mu = log(10 ε₀)`, which pushes the step up
first (by design, note 26). Geometric mean of the live step over the first 200 iterations vs the final frozen
step, and the largest live step seen in that window:

| problem | a = 0.6 | a = 0.9 |
|---|---|---|
| L10 | 1.02 vs 0.51 (max 16.9) | 0.43 vs 0.29 (max 1.5) |
| B2 | 16.8 vs 0.59 (max 673) | 3.0 vs 0.23 (max 70) |
| G20 | 2.6 vs 2.7 (max 24) | 1.2 vs 1.5 (max 4.4) |

On B2 the first ~500 dual-averaging updates run at steps 10–100× the final value (trace figure, log axis
from 10⁻¹ to 10²); these iterations are mostly pre-rejected (out-of-box, `oob` up to 0.18 per run at low a).
On L10 the overshoot is a factor 2 for 200 iterations, on G20 none.

**(v) Non-finite / out-of-box vs target.** Non-finite proposals: 0 in every run. Out-of-box fraction falls
with the target: B2 0.066 / 0.048 / 0.037 / 0.011 for a = 0.6 / 0.7 / 0.8 / 0.9 (mean over T), and grows with T
(B2 a = 0.6: 0.008 at T = 1 to 0.18 at T = 5); L10 ≤ 0.001, G20 ≤ 0.0003.

**(i) Adapted vs optimal step — the fixed-step scan.** §3.

**(vi) Does the sub-chain feedback bias the step vs an MH stage on the exact model?** §2.6.

### 2.5 Exact-surrogate and no-error-model diagnostics on L10

* `diag/exactsur_T{1,5}` (`scripts/diag_outer.py`, local run, 1 chain, a = 0.8, the exact linear map as the
  surrogate, error model on): outer acceptance **1.000** at T = 1 and at T = 5 (2 000 accepted, 0 rejected,
  499–500 pre-rejected). With a perfect surrogate the DAMH correction never rejects, as it must. The 0.25–0.53
  outer acceptance of the L10 runs is therefore surrogate error, not the integrator.
* `p1diag_L10_noem_a0.8_T{1,5}_r0` (error model **off**, single runs) against the a = 0.8 grid runs (error
  model on, single runs):

| T | error model | ESS/eval | outer acc | `eps_bar` | L | verdict |
|---|---|---|---|---|---|---|
| 1 | on | 0.049 | 0.25 | 0.39 | 2.6 | poor (E x² 5.2 SE) |
| 1 | off | 0.105 | 0.59 | 0.35 | 2.9 | ok |
| 5 | on | 0.099 | 0.50 | 0.37 | 13.5 | ok |
| 5 | off | 0.248 | 0.71 | 0.36 | 14.3 | ok |

Turning the error model off doubles the outer acceptance and the ESS/eval on L10 at both T, with the step
unchanged. **Single runs; flagged, not established.** Note 27 found the opposite sign on its P1 (S1 gives
2.4–3.6× in robust mode); on a linear-Gaussian problem with an accurate network the inflated sub-chain target
appears to cost rather than help. Part 2/3 of this study or a dedicated repetition would settle it.

### 2.6 MH-Hamiltonian comparison (question vi) — failed on L10 and B2

One run per problem, `algorithm="MH"`, Hamiltonian proposal on the network gradients, same a, T, network and
budget (`runs/p1mh_*`, `figures/fig_part1_traces_mh_vs_damh.png`):

| problem | (a, T) | DAMH `eps_bar` (reps) | MH `eps_bar` | MH L | MH ESS/eval | DAMH ESS/eval | MH wall |
|---|---|---|---|---|---|---|---|
| G20 | (0.9, 3) | 1.41 / 1.66 / 1.50 | 1.38 | 2.2 | 0.96 | 1.09 ± 0.04 | 11 s |
| L10 | (0.9, 5) | 0.28 / 0.35 / 0.28 | 0.0055 | 547 | 0.092 | 0.116 ± 0.039 | 671 s |
| B2 | (0.9, π/2) | 0.30 ± 0.08 (cell mean) | 3.5·10⁻⁵ | 1 000 (cap) | 0.0024 | 0.027 ± 0.008 | 996 s |

* **G20**: the MH step (1.38) sits inside the DAMH replicate range (1.41–1.66) and the ESS/eval match. No sign
  that the sub-chain feedback biases the step on this problem.
* **L10 and B2: run failed as a comparison.** The step collapsed to 10⁻⁵–10⁻⁴ within the first 10 updates
  (median at the first logged point, m = 10: 2·10⁻⁵ on L10, 8·10⁻⁶ on B2) and the recovery was slow or absent
  (L10: three chains reach 0.1 after 2 000 updates, five stay at 3·10⁻⁴; B2: all chains stay at 10⁻⁵–10⁻⁴ for
  the whole stage). `num_steps = T/ε` saturates at the 1 000-step cap, so each proposal costs up to 1 000 network
  gradients: 996 s for B2 and 671 s for L10 against 10–27 s for the DAMH runs of the same budget. The reached
  acceptance is 0.87–0.90 only because the trajectory barely moves. ESS/eval 0.0024 (B2) is 11× below DAMH, but
  the comparison measures the collapse, not the feedback bias the question asked about.
* **Cause (corrected 2026-10-12; the first version of this note blamed an untrained network at stage start,
  which is wrong: the network had the 8 000 warm-up snapshots).** The collapse is structural. In an MH stage the
  trajectory follows the network's gradients but the acceptance uses the exact posterior. As ε → 0 at fixed T the
  trajectory converges to the surrogate's exact dynamics, which conserve the surrogate energy, not the true one,
  so the acceptance does not go to 1: it levels off at a ceiling set by the surrogate error along the path,
  roughly the DAMH outer acceptance at the same settings (≈ 0.5 on L10, 0.2–0.3 on B2, 0.98 on G20). With the
  ceiling below the target 0.9, the dual averaging (γ = 0.05, t0 = 10) pushes ε down without limit; replaying
  its recursion with the acceptance held at 0.5 gives ε = 3·10⁻⁶ after 10 updates. Only when `num_steps` hits
  the 1 000-step cap does a smaller ε shorten the trajectory (L10: effective length 1 000 ε = 0.02–0.5 instead
  of 5; B2: 0.01–0.05 instead of π/2), and the short moves finally reach the target. Recovery is slow because the
  recursion's memory of the early deficit decays like 1/m: even with the acceptance exactly at target, ε needs
  ~1 000 updates to climb back to 0.08, and in these runs it hovered at 0.87–0.88, just under target. G20
  does not collapse because its ceiling (0.98) is above the target. DAMH stages are immune: the step adapts on
  the sub-chain acceptance, surrogate against itself, which does go to 1 as ε → 0. None of the 78 adaptive DAMH
  runs fell below ε = 0.057 or above 88 leapfrog steps.
* **Scope (low priority).** MH + surrogate-driven adaptive Hamiltonian is a supported, documented combination
  (`docs/stages.md`, WS7) but not a planned production usage: no example and no Auto layout builds it. The
  `docs/stages.md` warning ("pushes the adapted step size down … give such a stage enough evaluations")
  understates it: more evaluations do not help while the ceiling is below the target. Options, cheapest first:
  sharpen that warning (adaptive step unsuitable here; use a fixed step or DAMH); warn or resolve to a fixed step
  at stage construction; stop the adaptation and report when `num_steps` saturates at the cap. A clean answer to
  question (vi) would need an MH stage on the exact model's own gradients (possible only for L10 here).

### 2.7 Follow-up runs 2026-10-12 (author request): S1 error-model check on L10, P2 smoke runs

Nine runs added after the report was written (`jobs/s1check.jsonl`, `jobs/p2smoke.jsonl`; `logs/batch_progress.txt`
17:32–17:35; all return code 0, all pass the posterior check). Same driver and setup as Part 1, `time_limit = 120 s`
guard on the DAMH stage (never reached).

**S1 check: surrogate error model on vs off, L10 best cell (a = 0.9, T = 5), 3 repetitions each (`runs/s1chk_L10_em{1,0}_r*`).**
Confirms the single-run diagnostic of §2.5:

| error model | ESS/eval (3 reps) | mean ± sd | ESS(x²)/eval | outer acc | pre-rej | `eps_bar` | L | verdict |
|---|---|---|---|---|---|---|---|---|
| on | 0.108 / 0.090 / 0.140 | 0.113 ± 0.026 | 0.094 / 0.089 / 0.135 | 0.49 / 0.51 / 0.57 | 0.10 | 0.277 / 0.346 / 0.278 | 18 / 14 / 18 | 3 ok |
| off | 0.222 / 0.259 / 0.322 | **0.267 ± 0.051** | 0.201 / 0.273 / 0.235 | 0.70 / 0.79 / 0.80 | 0.10 | 0.267 / 0.324 / 0.270 | 19 / 16 / 19 | 3 ok |

The error model costs **2.4× ESS per exact evaluation on L10** (every "off" replicate beats every "on" replicate),
entirely through the outer acceptance (0.52 → 0.77); the adapted step and the trajectory length are unchanged
(same seeds per replicate: `eps_bar` agrees within 7 %). The "on" replicates also reproduce the Part-1 best-cell
runs (0.100 / 0.088 / 0.161 there) to within the DAMH-SMU non-reproducibility. Mechanism, consistent with the data
but not measured directly: on a problem where the network is accurate (the exact-surrogate diagnostic of §2.5 gives
outer acceptance 1.0), inflating the sub-chain likelihood by the residual covariance makes the sub-chain accept
proposals the exact posterior then rejects. This is the opposite sign to note 27's P1 result (2.4–3.6× gain in
robust mode) and leaves the S1 default-on of Auto (note 25) problem dependent: it pays when the surrogate is biased
and costs when it is good. Decision for the author; a conditional default (error model on only while the
held-out residual is large relative to the noise) is the obvious candidate and is not implemented.

**P2 smoke runs: `Stage.exact_step_scale = 1.0`, `exact_step_probability = 0.1`, `audit_prerejected = 0.05`, best
cell per problem, 1 run each (`runs/p2smoke_{L10,B2,G20}_pe0.1_se1_r0`).** First runs that exercise prerequisite P2
in a real stage. Every new field is present and consistent: manifest `proposal_resolution` carries `exact_step_scale`,
`exact_step_covariance_source` ("covariance (n=100)" L10/B2, "(n=200)" G20), `exact_step_covariance_trace`;
`Stage.describe()` shows the field; `notes` carries `exact_steps`, `exact_accepted`, `audited`, `audit_hidden`;
`subchain_stats` has the `kernel` column (L10 rank 0: 300 exact / 2 998 DA iterations ≈ 0.09); no warning or
traceback in the logs.

| problem | exact steps (share of exact evals) | exact-step acceptance | audits (hidden) | ESS/eval | Part-1 cell | outer acc | verdict |
|---|---|---|---|---|---|---|---|
| L10 | 2 597 (0.081) | **0.026** | 112 (0) | 0.081 | 0.116 ± 0.039 | 0.42 | ok |
| B2 | 2 910 (0.091) | 0.070 | 195 (0) | 0.019 | 0.035 ± 0.017 | 0.38 | ok |
| G20 | 1 382 (0.086) | 0.28 | 70 (0) | 0.85 | 1.09 ± 0.04 | 0.90 | ok |

**Finding for Part 2's design: the fixed exact-step covariance is only as good as the warm-up's statistics.**
`exact_step_covariance_trace` is the trace of `scale · 2.38²/d · Σ̂`, so Σ̂'s trace at stage start was 3.20 on L10
against the true posterior trace 0.62 (5× too wide; the warm-up MH with prior-scale steps has not resolved the
0.05–0.5 posterior sds), 3.45 on B2 against 3.31, 15.5 on G20 against 18.9. The exact random walk on L10 therefore
proposes 2.3× too far in every direction and accepts 2.6 %; it is non-adaptive by design (P2), so nothing corrects it
within the stage. The DAMH stage proposal uses the same Σ̂ as its mass and is unaffected, because the dual averaging
rescales the step. Consequences: (i) on L10 the Part-2 scale grid {0.5, 1, 2, 4} lies entirely above the useful
range — scales down to ~0.05 would be needed, or the scale should be defined relative to the statistics at the END
of the previous chunk rather than the discounted warm-up; (ii) the single-run cost of the kernel at p = 0.1 is
0.70× (L10), 0.53× (B2, within noise), 0.78× (G20) of the Part-1 ESS/eval — the exact steps spend 8–9 % of the
evaluations and almost never move on L10/B2; (iii) no audit found a hidden region (0 of 377), as expected on these
problems (§ "Read first": no blind spot by construction).


---

## 3. Fixed-step scan at the best (a, T): adapted vs optimal step

Best cell per problem, ε* = pooled `eps_bar` over its 3 replicates; then `Hamiltonian(step_size = f ε*,
adaptive=False)` for f ∈ {¼, ½, 1, 2, 4, 8}, 3 repetitions, same T, **`time_limit = 90 s` on the DAMH stage**
(after the untimed f = 2 B2 run never terminated, §1.2). "budget" = fraction of the exact-evaluation budget used
before the time limit; ESS/eval is per evaluation actually made. Source: `scripts/scan_summary.py`, `part1_scan.json`.

**L10** (a = 0.9, T = 5, ε* = 0.300; adaptive best cell 0.116 ESS/eval)

| f | step | ESS/eval ± sd | ESS(x²)/eval | ESS/s | sub-chain acc | outer acc | pre-rej | budget | wall s | oob | poor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ¼ | 0.075 | **0.251 ± 0.043** | 0.196 | 221 | 0.994 | 0.72 | 0.006 | 1.00 | 37 | 0 | 0 |
| ½ | 0.150 | 0.165 ± 0.024 | 0.163 | 274 | 0.978 | 0.63 | 0.022 | 1.00 | 19 | 0 | 0 |
| 1 | 0.300 | 0.127 ± 0.037 | 0.116 | **353** | 0.909 | 0.54 | 0.091 | 1.00 | 12 | 0 | 0 |
| 2 | 0.601 | 0.090 ± 0.011 | 0.089 | 129 | 0.251 | 0.51 | 0.75 | 0.69 | 63 | 0.063 | 0 |
| 4 | 1.202 | 0.027 ± 0.046 | 0.027 | 2.7 | 0.001 | 0.005 | 0.999 | 0.26 | 90 | 0.94 | 2 |
| 8 | 2.403 | 0.001 ± 0.000 | 0.001 | 0.1 | 0.000 | 0.000 | 1.000 | 0.26 | 90 | 0.77 | 2 |

**B2** (a = 0.8, T = 3, ε* = 0.285; adaptive 0.035)

| f | step | ESS/eval ± sd | ESS(x²)/eval | ESS/s | sub-chain acc | outer acc | pre-rej | budget | wall s | oob | poor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ¼ | 0.071 | 0.014 ± 0.006 | 0.033 | 19 | 0.991 | 0.50 | 0.009 | 1.00 | 24 | 0 | 0 |
| ½ | 0.143 | 0.033 ± 0.022 | 0.034 | 82 | 0.968 | 0.38 | 0.032 | 1.00 | 13 | 0.0005 | 0 |
| 1 | 0.285 | 0.033 ± 0.024 | 0.027 | 120 | 0.639 | 0.32 | 0.36 | 1.00 | 40 | 0.27 | 0 |
| 2 | 0.570 | 0.028 ± 0.018 | 0.025 | 90 | 0.340 | 0.23 | 0.66 | 0.87 | 38 | 0.48 | 0 |
| 4 | 1.140 | 0.034 ± 0.012 | 0.030 | 124 | 0.287 | 0.22 | 0.71 | 0.90 | 35 | 0.40 | 1 |
| 8 | 2.281 | 0.028 ± 0.019 | 0.024 | 77 | 0.127 | 0.23 | 0.87 | 0.87 | 63 | 0.26 | 2 |

**G20** (a = 0.9, T = 3, ε* = 1.523; adaptive 1.09)

| f | step | ESS/eval ± sd | ESS(x²)/eval | ESS/s | sub-chain acc | outer acc | pre-rej | budget | wall s | oob | poor |
|---|---|---|---|---|---|---|---|---|---|---|---|
| ¼ | 0.381 | **1.42 ± 0.04** | 0.301 | 5 643 | 0.996 | 0.98 | 0.004 | 1.00 | 4.1 | 0 | 0 |
| ½ | 0.761 | 1.29 ± 0.08 | 0.301 | 7 089 | 0.982 | 0.98 | 0.018 | 1.00 | 3.0 | 0 | 0 |
| 1 | 1.523 | 1.06 ± 0.06 | 0.290 | **7 331** | 0.915 | 0.98 | 0.085 | 1.00 | 2.3 | 0 | 0 |
| 2 | 3.045 | 0.60 ± 0.12 | 0.081 | 3 418 | 0.547 | 0.98 | 0.45 | 1.00 | 2.9 | 0 | 0 |
| 4 | 6.090 | 0.026 ± 0.012 | 0.062 | 31 | 0.104 | 0.99 | 0.90 | 1.00 | 16 | 0 | 3 |
| 8 | 12.18 | 0.031 ± 0.015 | 0.058 | 31 | 0.091 | 1.00 | 0.91 | 1.00 | 45 | 0 | 3 |

**Answer to (i), adapted/optimal step ratio.**

| problem | fixed f = 1 vs adaptive ESS/eval | best fixed f (grid; 3-point log-quadratic fit) | adapted / best-fixed step | ESS/eval gain of best fixed over adaptive |
|---|---|---|---|---|
| L10 | 0.127 vs 0.116 (same) | ¼ (edge; fit ¼) | **≥ 4** | 2.2× |
| B2 | 0.033 vs 0.035 (same) | 4 (fit 3.9), but ½ … 4 all within 0.028–0.034 ± 0.02 | ≈ 0.25 nominal, **flat** | 0.97× (none) |
| G20 | 1.06 vs 1.09 (same) | ¼ (edge; fit ¼) | **≥ 4** | 1.3× |

* The frozen adapted step reproduces the adaptive run's efficiency (f = 1 row = adaptive cell, within noise)
  on all three problems: the adaptation delivers the step it targets, and freezing it costs nothing.
* **"Steps are forced too small" is false for ESS per exact evaluation.** On L10 and G20 the adapted step is
  at least 4× *larger* than the best fixed step in the scan; the ESS/eval still rises at the smallest step
  tried, where the sub-chain acceptance is 0.99 and the pre-rejection rate below 1 %. With `subchain_length = 1`
  every outer iteration spends one exact evaluation unless pre-rejected, and a smaller step buys a higher outer
  acceptance (L10: 0.72 at ε*/4 vs 0.54 at ε*) at no cost in exact evaluations, only in network gradients
  (L = T/ε grows 4×). A target acceptance of 0.9 is therefore too low for this metric; the ESS/eval optimum
  lies at a sub-chain acceptance ≥ 0.99, outside the a-grid.
* **Per second the picture reverses**: ESS/s peaks at f = 1 on both L10 (353/s) and G20 (7 331/s) and is 1.3–1.6×
  lower at ε*/4. With these cheap models the network gradients are the cost, and the adapted step is right.
  For an expensive exact model (where ESS/eval is the relevant metric) the step should be smaller than the
  dual averaging at a = 0.9 gives.
* **Steps above ε*** fail in a specific way: at f ≥ 2 the pre-rejection rate is 0.45–1.0 and most of it is
  out-of-box (L10 f = 4: 94 % of proposals leave the prior box); the stage then does not reach its budget in the
  90 s limit (L10 f ≥ 4: 26 % of the budget, G20 f ≥ 4: budget reached but 3/3 poor with the mean off by
  3–14 SE). Without a time limit such a stage never terminates (§1.2). This is the same failure note 28 saw with
  ε = 0.4 on B2.
* **B2 is flat** from ε*/2 to 4ε* (0.028–0.034, sd 0.01–0.02) and only ε*/4 is clearly worse (0.014); steps ≥ 4ε*
  start to fail the left-arm check (1/3, 2/3 poor). The posterior's two arms, not the step, limit the ESS (note 28).

---

## 4. Correctness summary

28 poor runs of 139 (`scripts/collect.py verdict`; thresholds in §1.1):

| group | poor / runs | what failed |
|---|---|---|
| L10 grid + reps | 1 / 22 | (0.8, 1): E x² 5.2 SE |
| B2 grid + reps | 6 / 22 | all a ≤ 0.7 (5) or T = 5 (1): left-arm mass 0.18–0.37 vs 0.293, mean 3–9 SE |
| G20 grid + reps | 7 / 22 | all a ≤ 0.7 and T ≤ 3: mean 3.7–6.8 SE (0.05–0.10 sd), E x² 5–10 SE |
| MH-Hamiltonian | 0 / 3 | (B2 and L10 pass the check despite the step collapse: the exact MH acceptance keeps them correct, only slow) |
| fixed-step scan | 13 / 54 | L10 f ≥ 4 (4/6, mean 0.5–0.6 sd off, budget 26 %), B2 f ≥ 4 (3/6, left arm 0.20–0.33), G20 f ≥ 4 (6/6, mean 0.12–0.62 sd) |
| diagnostics | 1 / 5 | G20 fixed ε = 2.936 r1: E x² 9.7 SE |

Pattern: on the two non-Gaussian-or-rough problems (B2, G20) the low targets (a ≤ 0.7) give biased runs in 8 of
16 cells (B2 3, G20 5), the high targets (a ≥ 0.8) in 1 of 16 (B2 (0.9, 5) r0). Whether that is the adaptation or the step size
is not settled (§2.3). Steps ≥ 4ε* are unusable on every problem.

---

## 5. Recommendations for the placeholders (Part 1 evidence only)

* **Target acceptance `a`: 0.9 or higher**, not 0.8 (the current note-26 placeholder, the b2 cell (0.8, 3) is the
  single exception and within noise of (0.9, π/2)). Low targets (0.6, 0.7) are both less efficient on G20 at
  the useful T and biased on B2/G20. For ESS per exact evaluation the data point to a > 0.9; **a grid point at
  0.95 and 0.98 is needed before fixing the value.**
* **Integration time `T`: problem dependent, 3 (G20) to ≥ 5 (L10; the edge).** T = 1 and π/2 are too short
  on L10 and G20 (on G20 they degenerate to a single leapfrog step because ε > T). On B2, T = 5 hurts at a ≤ 0.8
  (out-of-box pre-rejections 6–18 %). The note-26 plan (T from a U-turn diagnostic) is the right direction;
  nothing here tests it.
* **Jitter 0.5: untested** (fixed in every run).
* **Exact-step probability and scale: no data** (Part 2 not run).
* **Defects / items for `10_manual_review_notes.md`:** (a) MH-stage adaptive Hamiltonian step collapse (low priority) and
  cost (§2.6); (b) fixed steps ≥ 2ε* can make a DAMH stage non-terminating (§1.2, §3) — a cap on pre-rejections
  per accepted sample, or on wall time by default, would turn this into a reported failure; (c) B2 dual averaging
  not converged at 3 000 evaluations per chain and a 10–100× start transient (§2.4); (d) the surrogate error model
  costs 2.4× ESS/eval on L10 (3 + 3 runs, §2.7), opposite to note 27's P1 result — S1 default-on is problem
  dependent; (e) a fixed `exact_step_scale` inherits the warm-up's covariance error (L10: 5× too wide, exact-step
  acceptance 2.6 %, §2.7) — Part 2's scale grid must go below 0.5 or re-base on the previous chunk's statistics.

---

## 6. Caveats

* Cheap models: L10 and B2 cost microseconds, G20 ~0.2 ms; ESS/eval is the specification's metric, ESS/s is the
  one that matters on these machines and ranks the step differently (§3).
* G20's posterior is close to the prior (outer acceptance 0.98 everywhere); its reference has Monte-Carlo error
  (min ESS 5 581), included in the SE.
* DAMH-SMU runs are not reproducible; 3 repetitions; B2 replicate spread ≈ effect size.
* The fixed-step scan ran under a 90 s time limit; the f ≥ 2 rows use 26–90 % of the budget.
* The a- and T-grids and the f-scan all have their optimum at an edge (§ "Read first").
* Parts 2 and 3 were not run; nothing about the exact safety kernel is measured here.

## 7. Paths

`toy_examples/tmp_step_size_study_2026-10-12/` — see the header. Specification of the study (the brief the runs
followed): the previous session's scratchpad `STUDY_SPEC_step_size.md` (not in the tree; its "Common setup",
Part 1–3 definitions and report structure are reproduced in §1 and the section headings above).
