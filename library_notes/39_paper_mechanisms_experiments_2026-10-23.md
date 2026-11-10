# Paper-mechanism experiments: results log (started 2026-10-23 22:33 UTC; observed results only)

Plan: `38_experiment_plan_paper_mechanisms_2026-10-23.md`. Drivers, jobs, runs and analysis:
`toy_examples/tmp_paper_experiments_2026-10-23/` (`README.md` there; `scripts/{problems,run_case,launch,run_queue,
make_jobs,analyze,segments}.py`; `jobs/<phase>/*.json`; `runs/<name>/` = format-v2 output + `spec.json`,
`timing.json`, `launch_log.txt`, `DONE`/`FAILED`; `analysis/<phase>/{metrics,posterior,summary}.csv`).
Library state: the six mechanisms of note 37 plus the version-id tooling (T1: `Evaluator.version`,
`installations/` column `collector_version`, `versions/` CSV), uncommitted.

## 0. Smoke phase (9 tiny runs, 4 chains, budgets 200/300/600/200; `runs/smoke_*`)

Observed (agent report, 2026-10-23 22:35 UTC):
- All 9 jobs DONE; the analysis wrote 17 metrics rows. Wall: P1 F0/F1/U0/U 37-48 s, A-avg 5 s, REF 2.5 s,
  BA10 A-rand 131 s, P1 A-rand 420 s.
- **Defect found (deferred adaptation, randomised rule + adaptive Hamiltonian sub-chain):** with the live step
  frozen between installs, the dual-averaging recursion is fed acceptances of the frozen step (open loop) and
  runs off: `log_step_size` down to -51.9 (the clip), `num_steps` at the cap 1000, carried step into S3
  2.4e-4 vs 0.066 after S1 (`runs/smoke_P1_A-rand_hmc5_w0.05_nl500-al0.75_S0-200_S1-300_S2-600_S3-200_c4_r0/
  sampling_output/adaptive_stats/alg0002_DAMH-SMU/rank0000.csv`, `carry_over/alg0002_DAMH-SMU.npz`); the
  same on BA10 (min log step -50.5). The random-walk log-scale and the pCN logit have the same open-loop
  structure (slower drift). Fix in progress: in deferred mode the feedback recursions advance once per
  installation on the mean acceptance since the previous one (closed loop at the installation time scale).
  Every A-rand run waits for the fix; paper §6.5 sentence on the recursions to be corrected accordingly.
- P1 F1 smoke, frozen tail S3 (4 chains, 800 evaluations): prior step accepted 0/50; f_pre = 0 (HMC K = 5
  never pre-rejects); ESS/eval min over 16 directions 0.112; tau max 8.95; test RMSE 0.022. Moment errors
  against the 600-evaluation toy reference are meaningless (listed only to show the pipeline runs).
- P1 A-rand smoke S2: 743 installs of 743 received (every version installed: coin bound 2759 > versions),
  cost ratio 1.001. BA10 A-rand S2 far-mode mass 0.151 (quadrature reference 0.1409).
- Replicate-variance ESS: implemented as Var_ref(g)/Var_rep(mean_g) (note 38 §D corrected).

## 1. Phase 1 (`jobs/phase1_norand`, 228 jobs: refs 3, A1 48, A6 without A-rand 80, B1 without A-rand 15,
B2 72; launched 2026-10-23 22:33 UTC, 3 parallel, max 30 ranks)

(results appended by the supervision ticks)

### Fix verified (2026-10-23 22:38 UTC): per-installation recursion

`runs/smoke_P1_A-rand_hmc5_w0.05_nl500-al0.75_S0-200_S1-300_S2-600_S3-200_c4_r0` re-run after the fix
(the defective output kept as `..._DEFECT_before_fix`):

| quantity | before fix | after fix |
|---|---|---|
| S2 log step size min / max | -51.9 / (clip) | -2.78 / -1.61 |
| S2 num_steps max | 1000 (cap) | 25 |
| carried step into S3 | 2.4e-4 | 0.154 |
| S2 wall (600 evaluations/chain, 4 chains) | 282 s | 13.8 s |
| S2 installs / versions received | 743 / 743 | 64 / 64 |

S1 (the preliminary DAMH-SMU stage, standard dual averaging, not deferred) shows a transient
log-step minimum of -12.85 and num_steps 1000 on this tiny budget (46 s): the usual early behaviour of
the dual averaging on a coarse network, not related to the rule. Unit suite after the fix: 906 passed.
Paper §6.5 sentence on the recursions corrected (advance once per installation).

## 2. Queue re-prioritised (2026-10-23 22:44 UTC)

The three reference runs finished by 22:42 UTC (P1 400 000 and P1-50/P1-100 200 000 evaluations per
chain, 8 chains each, exact adaptive random-walk MH; `runs/P1_REF_S2-400000`, `runs/P1-50_REF_S2-200000`,
`runs/P1-100_REF_S2-200000`). The phase-1 queue process was then stopped (18 A1 jobs had completed
under it, 3 more finished as orphans) and the merged queue `jobs/all_ordered` (605 jobs: A1, A2, A6, A8,
B1, B2, B7, B8, B3, B4, B6 in this order, names de-duplicated, DONE jobs skipped) was started at 22:44 UTC with 3 parallel jobs (two earlier attempts
to start it through a waiter failed on a self-matching pgrep pattern; no run was affected).

## 3. Reference runs (analysed 2026-10-23 22:45 UTC; `analysis/refs/metrics.csv`, caches `refs/*_reference.npz`)

Exact adaptive random-walk MH, 8 chains, first 25 % of every chain dropped before caching.

| problem | run | exact evaluations | states cached | min ESS/eval (16 observed directions) | ESS in the worst direction |
|---|---|---|---|---|---|
| P1 (d = 20) | `runs/P1_REF_S2-400000` | 3 200 000 | 2 400 008 | 0.00311 | ≈ 9 900 (≈ 7 500 after the drop) |
| P1-50 | `runs/P1-50_REF_S2-200000` | 1 600 000 | 1 200 008 | 0.00124 | ≈ 2 000 (≈ 1 500 after the drop) |
| P1-100 | `runs/P1-100_REF_S2-200000` | 1 600 000 | 1 200 008 | 0.00121 | ≈ 1 900 (≈ 1 450 after the drop) |

Reference precision: the relative standard error of a reference sd in its worst direction is about
1/sqrt(2 ESS) ≈ 0.8 % (P1), 1.8 % (P1-50), 1.9 % (P1-100); moment-error comparisons below the 5 % level at
d = 50/100 would need longer references (range rule: extend if B7's errors approach that level).
Queue at 22:45 UTC: 605 jobs, 18 done (A1), 3 running, 0 failed.

## 4. A1 — prior step exact on the closed-form toy (48 runs DONE by 22:52 UTC; `analysis/A1/`)

LG2 and LG10 (A = I + 0.5 superdiagonal, obs sd 0.3, prior N(0, I)), code F1 (frozen network, prior step
at rate w), HMC sub-chain K ∈ {1, 5}, 8 chains, S2 = 20 000 evaluations per chain, 3 replicates each.
z = (posterior mean of the run − closed form) / (closed-form sd / sqrt(ESS)), ESS of the coordinate from the
S2 autocorrelation (the F1 stage is time-homogeneous, so the estimate is justified), coordinates x1–x5:

| problem | w | prior-step acceptance | ESS min per coordinate | max abs z (n) | fraction abs z > 3 | max rel. sd error |
|---|---|---|---|---|---|---|
| LG2 | 0.02 / 0.1 / 0.5 | 0.113–0.117 | 151 000 / 132 000 / 60 000 | 1.66 / 1.85 / 2.34 (12 each) | 0 | 0.6 % |
| LG2 | 1 | 0.117 | 11 400 | 1.98 (12) | 0 | 1.4 % |
| LG10 | 0.02 / 0.1 / 0.5 | 1e-4 | 132 000 / 111 000 / 47 000 | 2.00 / 2.44 / 1.82 (30 each) | 0 | 0.6 % |
| LG10 | 1 | 1e-4 | 66 | 4.55 (30) | 0.23 | 49 % |

Verdict: the prior step is exact (A1 pass) for every w < 1 and for w = 1 at d = 2. At d = 10 the pure
prior independence sampler (w = 1) accepts 1 in 10 000 draws (about 16 moves per chain in 20 000
evaluations), so its chains barely move: the "errors" there are a non-converged chain and an
autocorrelation estimate that cannot see it (paper §7.2 caveat), not a bias. This is the expected collapse
of the prior-step acceptance with the dimension (paper Rem. 5.3); w = 1 is the grid boundary by definition,
no extension. Efficiency of F1 on the toy, frozen tail S3, min ESS per exact evaluation: LG2 1.01 / 0.87 /
0.39 / 0.075 and LG10 0.82 / 0.73 / 0.29 / 0.004 for w = 0.02 / 0.1 / 0.5 / 1 (K = 1; K = 5 within 10 %),
i.e. the prior-step cost is close to the paper's multiplier 1 − w at small w (0.98, 0.9, 0.5 expected
for tau ≈ 1). Wall per run 12–120 s.

## 5. A2 — coin mechanics with a fast collector (9 runs DONE by 23:15 UTC; `analysis/A2/`, own check script in the tick log)

P1, A-rand, HMC K = 5, w = 0.05, 8 chains, production stage 20 000 evaluations per chain (≈ 20 100 outer
iterations per chain, f_pre ≈ 0.007), collector with `iterations_batch = 20`: 7 400–8 900 versions published
per run, 5 300–6 300 received per run (one per ≈ 27 iterations per chain), 1 replicate. Installs compared with
the pending-conditioned coin prediction Σ_i λ_{i−1} 1{a version is pending at i} (from `versions/` and
`installations/`), z = (installs − expected)/binomial SE:

| n_λ | a_λ | received | installs | expected | z | installs / received | distinct versions across the 8 chains at iteration 10⁵ (sum) |
|---|---|---|---|---|---|---|---|
| 100 | 0.5 / 0.75 / 1 | 6 293 / 6 138 / 6 156 | 5 976 / 4 377 / 2 640 | 6 115 / 4 445 / 2 610 | −1.97 / −1.08 / 0.63 | 0.95 / 0.71 / 0.43 | 8 / 8 / 8 |
| 1 000 | 0.5 / 0.75 / 1 | 5 866 / 6 025 / 5 841 | 5 866 / 6 006 / 5 649 | 5 862 / 5 951 / 5 641 | 0.08 / 0.93 / 0.13 | 1.00 / 1.00 / 0.97 | 7 / 7 / 8 |
| 10 000 | 0.5 / 0.75 / 1 | 5 340 / 5 258 / 4 775 | 5 340 / 5 258 / 4 775 | 5 349 / 5 241 / 4 754 | −0.58 / 0.87 / 1.15 | 1 / 1 / 1 | 7 / 8 / 7 |

Verdict: A2 pass — the install counts agree with the coin law for all nine schedules (|z| ≤ 2), chains hold
different versions at the same time, and superseded versions are dropped exactly when the coin binds.
Within a 20 000-step production stage and a version every 27 steps, the coin binds only for n_λ = 100
(expected gap n/n_λ exceeds the arrival interval for n > 2 700 at a_λ = 1); for n_λ ≥ 1 000 the installed
lag is the arrival interval, not the coin. **Range rule (note 38 §C, trigger "behaviour confirmed only at one
end of the grid")**: 6 follow-up jobs `jobs/A2x` with n_λ ∈ {10, 30} × a_λ ∈ {0.5, 0.75, 1} generated, to run
after the main queue; B3 will need the same downward extension. Single-replicate efficiency figures (S3 tail,
min ESS per evaluation 0.14–0.38, two low outliers at nl100-al0.5 and nl1000-al0.75) are not interpretable
without replicates (A6/B1). Moment errors in the observed directions: mean rms 0.008–0.011, sd rms
0.005–0.007 (reference sds 0.03–1.0); judged with replicates in A6. Wall 235–246 s per run.

## 6. A6 — posterior correctness and efficiency of every production code (110 runs DONE by 02:03 UTC 2026-10-24; `analysis/A6/`)

P1 (d = 20) and BA10, codes F0 / F1 / A-rand / A-avg / U / U0, HMC K = 5 (dimension-robust, mass from the
pooled statistics, dual-averaged step) and pCN K = 5; w = 0.05, RandomisedInstallation(500, 0.75),
AveragedInstallation(5, 0.8, 10) with pCN β = 0.2 PINNED in S1–S3 and `input_clip = 8`; 8 chains; budgets
1 000 / 3 000 / 20 000 / 5 000 evaluations per chain; 5 replicates; references: the 3.2 M-evaluation P1
run (§3) and the quadrature of the 1-D marginal on BA10 (far-mode mass 0.1409).

### P1, Hamiltonian sub-chain (S2 = production stage, S3 = frozen tail)

| code | S3 min ESS / exact eval (16 directions) ± SE | S2 f_pre | prior-step acc. | test RMSE at end | S2 wall [s] | S3 wall [s] | S2 installs / received |
|---|---|---|---|---|---|---|---|
| F0 (frozen, no exact step) | 0.375 ± 0.040 | 0.000 | – | 0.0197 | 539 | 136 | – |
| F1 (frozen + prior step) | 0.347 ± 0.021 | 0.000 | 0.0000 | 0.0196 | 506 | 128 | – |
| A-rand (theorem) | 0.352 ± 0.018 | 0.009 | 0.0000 | 0.0090 | 233 | 56 | 4 507 / 4 582 |
| U (prior step, no DA) | 0.371 ± 0.019 | 0.006 | 0.0000 | 0.0092 | 248 | 57 | – |
| U0 (no exact step, no DA) | 0.390 ± 0.061 | 0.006 | – | 0.0207 (*) | 271 | 60 | – |

(*) U0's `test_rmse_final` is read from a different row of `surrogate_quality_test.csv` than U's; not
interpreted. Observations: (i) ESS per exact evaluation is the same for all five codes within one SE
(0.35–0.39); the prior step costs nothing measurable at w = 0.05 here (its acceptance is 0: at d = 20 with
noise sd 0.01 a prior draw never fits; 1 000 prior draws per chain were all rejected); (ii) continued
training halves the WALL time of both the production stage and the frozen tail (233–271 s vs 506–539 s):
the network frozen after 4 000 evaluations (test RMSE 0.020) gives rougher gradients, a smaller
dual-averaged step and more leapfrog steps per trajectory than the one trained on 24 000 (RMSE 0.009), at
equal ESS per exact evaluation; (iii) f_pre < 1 % for every Hamiltonian code, so the cost ratio is 1.000
(A7 is not informative here).

Posterior correctness (S2, 16 observed singular directions, 5 replicates). Against the 3.2 M reference
every Hamiltonian code shows the SAME offset (rms 0.8 % of the posterior sd, max 1.3 %), i.e. z up to 9–28
with replicate SEs: that is the reference's own error (its worst-direction SE is 1.2 % of sd, §3), not a
bias of the samplers; a 10× longer reference (`jobs/refs2`, 4 M evaluations per chain) is queued. The
decisive test is pairwise against F0 (classically exact, same budgets), replicate SE of the difference:

| code vs F0 | max abs z over 16 directions | directions with abs z > 3 | mean abs difference / sd |
|---|---|---|---|
| F1 | 2.36 | 0 | 0.0021 |
| A-rand | 1.75 | 0 | 0.0018 |
| U | 4.47 | 3 | 0.0034 |
| U0 | 4.59 | 2 | 0.0030 |

The two guaranteed codes agree with the frozen reference code; the two codes WITHOUT diminishing
adaptation (every version installed, continuous tuning) differ from it at 4.5 SE in 2–3 directions, by
0.3 % of a posterior sd. With 5 replicates (t with ≈ 8 degrees of freedom, 16 directions) this is
suggestive, not conclusive: 5 more replicates of P1 U and U0 are queued (`jobs/A6x`).

### BA10 (far-mode mass, reference 0.1409; crossings of t = c per 1 000 outer steps, sum over 8 chains)

| code | sub-chain | far-mode mass ± SE (5 reps) | z vs 0.1409 | crossings / 1 000 | S3 min ESS/eval (t, mode indicator) | S2 wall [s] |
|---|---|---|---|---|---|---|
| F0 | hmc5 | 0.102 ± 0.033 | −1.2 (chains stuck: 0.03 crossings) | 0.03 | 0.0019 | 297 |
| F1 | hmc5 | 0.149 ± 0.004 | 2.3 | 2.4 | 0.0086 | 284 |
| A-rand | hmc5 | 0.137 ± 0.002 | −1.8 | 2.8 | 0.014 | 230 |
| U | hmc5 | 0.128 ± 0.003 | −3.9 | 2.8 | 0.0070 | 290 |
| U0 | hmc5 | 0.099 ± 0.014 | −3.1 (0.6 crossings) | 0.6 | 0.0095 | 261 |
| F0 / F1 / A-rand / U / U0 | pcn5 | 0.1409 / 0.1409 / 0.1410 / 0.1415 / 0.1410 (SE ≤ 0.0012) | all within 1.6 | 80–135 | 0.56–0.60 (U, U0: 0.16–0.20) | 14–25 |
| A-avg (β = 0.2 pinned) | pcn5 | 0.143 ± 0.006 | 0.4 | 2.2 | 0.0068 | 21 |

Observations: (i) with a Hamiltonian sub-chain the prior step is what crosses the likelihood barrier
(acceptance 0.177 per prior draw on BA10; 2.4–2.8 crossings per 1 000 steps with it, 0.03–0.6 without):
F0 and U0 fail the far-mode mass (chains stuck in one mode), F1 and A-rand reproduce it within 2.3 SE,
and U (no diminishing adaptation) sits 3.9 SE below the reference with 5 replicates (more replicates
queued, `jobs/A6x`); (ii) the pCN sub-chain with the carried β crosses the barrier by itself (80–135
crossings per 1 000 steps) and every pCN code is exact on BA10; (iii) A-avg with the PINNED β = 0.2 is 80×
less efficient than the pCN codes with the carried β (0.0068 vs 0.56–0.60 ESS per evaluation) and on P1
likewise poor (0.0046 vs 0.003 for the other pCN codes, all hopeless on P1's anisotropic posterior): the
pinned β, not the averaged rule, is the cause, so the averaged-rule sweep is re-generated with the carried
β (`jobs/B6c`, 64 jobs incl. A6's A-avg with the carried β) and the pinned-β B6 block is dropped.

Replicate-variance ESS per exact evaluation (note 38 §D, min over the first 5 directions, whole S2 stage
including the adaptation): P1 hmc5 A-rand 0.196, F0 0.223, F1 0.138, U 0.203, U0 0.247 (SE ≈ 35 % with 5
replicates; consistent with one another and with the tail estimates); BA10 hmc5: A-rand 0.032, F1 0.012, U
0.013, F0 0.0001, U0 0.0008.

### Queue re-prioritised (02:20 UTC)

After A6 the old queue was stopped (in-flight jobs finish) and `jobs/all_ordered2` started: A6x (30: BA10
F1/A-rand/U/U0 hmc5 and P1 U/U0 hmc5, replicates 5–9), refs2 (P1 reference 4 M evaluations per chain), A2x
(n_λ 10, 30), then B1, B2, B7, B8, B3, B4, B6c.

## 7. A8 — blind-spot escape (30 runs DONE by 02:44 UTC 2026-10-24; `analysis/A8/`)

BA10, all 8 chains started at the NEAR mode (t = c − r, sd 1e-3), preliminary budget 500 evaluations per
chain (S0 200 + S1 300: the network knows one mode only), production 20 000 + frozen tail 5 000, HMC K = 5,
w = 0.05, 10 replicates; reference far-mode mass 0.1409.

| code | far-mode mass, mean ± SE (10 reps) | min / max over reps | replicates with mass < 0.02 | crossings / 1 000 steps | last chain's first passage to the far mode [outer iterations] | prior-step acceptance |
|---|---|---|---|---|---|---|
| F0 (frozen, no exact step) | 0.022 ± 0.014 | 0.000 / 0.124 | 8 of 10 | 0.002 | never in 8 reps; one chain late in 2 | – |
| F1 (frozen + prior step) | 0.144 ± 0.003 | 0.126 / 0.156 | 0 | 2.22 | 971–4 228 (median ≈ 1 700) | 0.178 |
| A-rand (theorem) | 0.142 ± 0.003 | 0.125 / 0.158 | 0 | 2.16 | 720–3 481 (median ≈ 1 700) | 0.177 |

Verdict: A8 pass. With the network blind to the far mode the usual practice (F0) stays in the start mode
in 8 of 10 replicates (two replicates had one chain reach it late, giving 0.08–0.12); with the exact prior
step every replicate of F1 and A-rand reaches the far mode in every chain within the first 5 000 outer
iterations and reproduces its mass within 2 SE (the earlier A6 estimate of F1, 0.149 ± 0.004, and this one,
0.144 ± 0.003, bracket the reference). The crossings here come from the prior step (2.2 per 1 000 steps at
acceptance 0.18), not from the Hamiltonian sub-chain. Wall 86–92 s per run.

## 8. A6 with the extra replicates (A6x DONE by 04:03 UTC 2026-10-24; `analysis/A6_10reps/`, A6 + A6x)

### BA10, Hamiltonian sub-chain, far-mode mass (reference 0.1409), 10 replicates for F1 / A-rand / U / U0

| code | far-mode mass ± SE | per-replicate values | z vs reference | crossings / 1 000 |
|---|---|---|---|---|
| F1 (frozen + prior step) | 0.1437 ± 0.0034 | 0.126–0.158 | +0.8 | 2.29 |
| A-rand (theorem) | 0.1329 ± 0.0027 | 0.114–0.145 | −3.0 | 2.79 |
| U (prior step, every version, continuous tuning) | 0.1288 ± 0.0021 | 0.119–0.140 | −5.7 | 2.76 |
| U0 (no exact step) | 0.0990 ± 0.0078 | stuck chains | −5.4 | 0.58 |
| F0 (5 reps) | 0.1018 ± 0.0330 | stuck chains | −1.2 | 0.03 |

Observed: at a production budget of 20 000 evaluations per chain the frozen code with the prior step is
unbiased, while BOTH adaptive codes with the prior step underestimate the far-mode mass, U by 9 % (5.7 SE)
and A-rand by 6 % (3.0 SE). The adaptive codes also scatter less between replicates than F1 (sd 0.007–0.009
vs 0.011). This is a finite-run effect of adaptation on a bimodal target (the network is retrained on
data dominated by the mode the chains occupy); the theorem makes A-rand's bias vanish with the stage
length, U's need not. **Range rule (budget):** `jobs/A6y` = BA10 F1 / A-rand / U, HMC K = 5, production
80 000 evaluations per chain, 5 replicates each, to see whether A-rand's deficit shrinks with the
budget (expected) and U's persists.

### P1, Hamiltonian sub-chain, pairwise bias test vs F0 (U and U0 now 10 replicates, F0/F1/A-rand still 5)

| code vs F0 | max abs z (16 directions) | directions with abs z > 3 |
|---|---|---|
| F1 | 2.36 | none |
| A-rand | 1.75 | none |
| U | 4.83 | v1, v2, v3, v6 |
| U0 | 6.89 | v1, v2, v3 |
| pooled U + U0 (20) vs pooled F0 + F1 + A-rand (15) | 7.48 | v1, v2, v3, v6, v12, v15 |

Offsets to the 3.2 M reference in units of the posterior sd (mean ± replicate SE), leading directions:
v1: F0 +0.008, F1 +0.007, A-rand +0.008 vs U −0.002, U0 −0.003 (SE 0.001–0.003); v3: F0 +0.005, F1 +0.007,
A-rand +0.002 vs U −0.006, U0 −0.008. The three exact/guaranteed codes share one offset (the reference's
own error, §6); the two no-DA codes sit about 1 % of a posterior sd away from them with the same sign in
both, in the best-constrained directions. B1 adds 5 replicates of F0, F1, A-rand, U on P1; refs2 (4 M
evaluations) will replace the reference. The effect is small (1 % of sd) but systematic across 10 + 10
replicates and two problems (BA10 above), which is the expected signature of adaptation without
diminishing adaptation.

## 9. A2x — n_λ extended downwards (6 runs DONE by 04:03 UTC 2026-10-24; `analysis/A2x/`)

Same setting as A2 (P1, A-rand, HMC K = 5, fast collector, 1 replicate). Coin law and lag:

| n_λ | a_λ | received | installs | expected | z | installs / received | mean install gap, 2nd half of the stage [steps] | S3 min ESS/eval | test RMSE at end |
|---|---|---|---|---|---|---|---|---|---|
| 10 | 0.5 / 0.75 / 1 | 6 238 / 6 347 / 7 362 | 4 032 / 1 374 / 491 | 4 065 / 1 350 / 469 | −0.5 / 0.7 / 1.1 | 0.65 / 0.22 / 0.07 | 56 / 187 / 573 | 0.35 / 0.30 / 0.35 | 0.0115 / 0.0112 / 0.0110 |
| 30 | 0.5 / 0.75 / 1 | 7 576 / 7 343 / 5 735 | 5 816 / 2 717 / 979 | 5 843 / 2 699 / 990 | −0.4 / 0.4 / −0.4 | 0.77 / 0.37 / 0.17 | 37 / 94 / 284 | 0.38 / 0.34 / 0.37 | 0.0108 / 0.0111 / 0.0108 |

The coin law holds where the coin binds hard (A2 pass completed); an installation lag of up to ≈ 570
steps (61 installs per chain in 20 000 steps at n_λ = 10, a_λ = 1) costs no measurable efficiency on P1
(tail ESS per evaluation 0.30–0.38 vs 0.33–0.38 in A2) and leaves the final network quality unchanged
(test RMSE 0.011): on P1 the surrogate changes little during the production stage, as paper §6.4 expects.

## 10. refs2 — P1 reference with 4 M evaluations per chain (DONE 04:17 UTC; `runs/P1_REF_S2-4000000`, 32 min)

32 M exact evaluations, 24 M states cached (`refs/P1_reference.npz`; the 3.2 M cache kept as
`P1_reference_3p2M.npz`), min ESS/eval 0.00316 → ≈ 100 000 effective samples in the worst direction
(reference SE ≈ 0.3 % of sd there). The two references differ by rms 0.9 % of sd over the 16 directions
(the 3.2 M reference's own error, as suspected in §6). A6 + A6x re-tested against the 4 M reference
(`analysis/A6_10reps_ref4M/`), offsets of the run means in units of the posterior sd:

| code | reps | rms offset | max abs offset | max abs z | directions with abs z > 3 |
|---|---|---|---|---|---|
| F0 | 5 | 0.0033 | 0.0060 | 4.5 | 3 (v2, v5, v10) |
| F1 | 5 | 0.0028 | 0.0062 | 4.7 | 2 (v12, v16) |
| A-rand | 5 | 0.0027 | 0.0054 | 4.0 | 3 (v9, v10, v12) |
| U | 10 | 0.0044 | 0.0103 | 8.6 | 7 |
| U0 | 10 | 0.0048 | 0.0107 | 8.6 | 9 |

With the better reference the exact and the guaranteed codes sit within 0.3 % of sd (rms) of it, the
residual z ≈ 4–4.7 in 2–3 directions being of the size of the reference's own error; the two no-DA
codes are 1.6× further away (rms 0.45 %, max 1.0 % of sd) with 7–9 of 16 directions beyond 3 SE. Together
with the pairwise test (§8) this is the clearest validity result so far: adaptation without diminishing
adaptation leaves a small bias (≈ 0.5–1 % of a posterior sd on P1, 9 % of the far-mode mass on BA10),
the guaranteed stage does not on P1 and shows a smaller, budget-dependent deficit on BA10 (A6y queued).

Queue re-prioritised at 04:20 UTC: `jobs/all_ordered4` = B1 (remaining), A6y, B7, B8, B2, B3, B4, B6c.

## 11. B1 — headline comparison on P1 with 10 replicates per code (B1 DONE by 05:17 UTC 2026-10-24; `analysis/P1_10reps/` = A6 + A6x + B1, reference = 4 M run)

P1 (d = 20), HMC K = 5 sub-chain, w = 0.05, RandomisedInstallation(500, 0.75), 8 chains, budgets
1 000 / 3 000 / 20 000 / 5 000 per chain, 10 replicates per code.

| code | guarantee | S3 min ESS / exact eval ± SE | τ max | replicate-variance ESS/eval (S2, min of 5 dirs; SE ≈ 24 %) | S2 wall [s] | S3 wall [s] | test RMSE at end | vs F0: max abs z (dirs > 3) | vs 4 M reference: rms offset / max [sd] |
|---|---|---|---|---|---|---|---|---|---|
| F0 | classical (frozen, no exact step) | 0.406 ± 0.022 | 2.55 | 0.363 | 545 | 129 | 0.0197 | – | 0.0029 / 0.0056 |
| F1 | uniformly ergodic (frozen + prior step) | 0.368 ± 0.019 | 2.78 | 0.261 | 524 | 133 | 0.0193 | 2.0 (0) | 0.0022 / 0.0040 |
| A-rand | Thm 6.11 | 0.370 ± 0.012 | 2.74 | 0.255 | 228 | 56 | 0.0089 | 2.8 (0) | 0.0020 / 0.0045 |
| U | SUE, no DA | 0.346 ± 0.031 | 3.64 | 0.215 | 249 | 57 | 0.0090 | 6.0 (4: v1, v2, v3, v6) | 0.0044 / 0.0103 |
| U0 | none | 0.434 ± 0.033 | 2.52 | 0.383 | 269 | 61 | 0.0209 | 8.1 (3: v1, v2, v3) | 0.0048 / 0.0107 |

Observed: (i) validity — F1 and A-rand agree with the classically exact F0 in every direction (max 2.0
and 2.8 replicate SE, 10 vs 10) and sit within 0.2 % of a posterior sd (rms) of the 4 M reference; U and
U0 differ from F0 at 6–8 SE in the three best-constrained directions and are twice as far from the
reference (rms 0.45 %, max 1.0 % of sd); (ii) efficiency per exact evaluation — the prior step costs
9 % of the tail ESS (F1/F0 = 0.91 ± 0.07; the paper's bound 1 − w = 0.95), continued training changes the
tail ESS per evaluation by nothing measurable (A-rand/F1 = 1.01 ± 0.06) and the replicate-variance
ESS of the whole production stage is the same for F1 and A-rand (0.26 vs 0.26) and lower than F0's 0.36
by the prior-step cost plus the adaptation transient (SE 24 %, so only the direction is established);
(iii) wall time — the adaptive codes run 2.3× faster than the frozen ones at equal budgets (S2 228–269 s
vs 524–545 s; tail 56–61 s vs 129–133 s) because the better network (test RMSE 0.009 vs 0.019) allows a
larger leapfrog step and fewer gradient evaluations per trajectory; at a cheap forward model this is
the whole gain of continued training on P1. Run directories: `runs/P1_{F0,F1,A-rand,U,U0}_hmc5_*_r{0..9}`.

## 12. A6y — BA10 at four times the production budget (15 runs DONE by 06:59 UTC 2026-10-24; `analysis/A6y/`)

BA10, HMC K = 5, w = 0.05, production stage 80 000 evaluations per chain (640 000 per run), 5 replicates;
reference far-mode mass 0.1409. Compared with the 20 000-evaluation stage (§8, 10 replicates):

| code | far-mode mass at 20 000 (10 reps) | deficit | far-mode mass at 80 000 (5 reps) | deficit | z vs reference at 80 000 | per-replicate values at 80 000 |
|---|---|---|---|---|---|---|
| F1 (frozen + prior step) | 0.1437 ± 0.0034 | +2 % | 0.1409 ± 0.0038 | 0 % | 0.0 | 0.134–0.153 |
| A-rand (theorem) | 0.1329 ± 0.0027 | −6 % | 0.1368 ± 0.0021 | −3 % | −1.9 | 0.133–0.145 |
| U (no DA) | 0.1288 ± 0.0021 | −9 % | 0.1286 ± 0.0020 | −9 % | −6.1 | 0.121–0.132 |

Observed: quadrupling the stage halves the guaranteed code's deficit (6 % → 3 %, now within 2 SE of the
reference) and leaves the no-DA code's deficit unchanged (9 %, 6 SE): the signature the theory predicts
(the adaptation transient of A-rand is a finite-run effect that decays with the stage length; U's bias
is a property of its kernel sequence). Crossings per 1 000 steps 2.8–2.9 for all three. Wall 915–1 093 s
per run. The frozen tails (5 000 evaluations, SE 0.004–0.013) are too short to add anything. Range rule:
a further doubling (160 000) for A-rand and U would pin the decay rate; not queued now (cost ≈ 40 min per
run), to be decided after B4 (production-budget sweep on P1).

## 13. B7 — dimension (24 runs DONE by 08:40 UTC 2026-10-24; `analysis/B7/`)

P1-50 and P1-100 (same solver, 50 / 100 KL modes, 4×4 observations, noise sd 0.01), codes F0 / F1 /
A-rand / U, HMC K = 5, w = 0.05, 3 replicates, budgets as in B1 (tail 5 000).

| d | code | S3 min ESS/eval ± SE | τ max | S2 wall [s] | test RMSE at end | prior draws / accepted | vs F0: max abs z (dirs > 3; 3 vs 3 reps) | offset to the 200 k reference, rms [sd] |
|---|---|---|---|---|---|---|---|---|
| 50 | F0 | 0.013 ± 0.004 | 96 | 633 | 0.030 | – | – | 0.031 |
| 50 | F1 | 0.016 ± 0.007 | 86 | 606 | 0.032 | 8 064 / 0 | 4.2 (2) | 0.030 |
| 50 | A-rand | 0.025 ± 0.007 | 48 | 321 | 0.015 | 8 086 / 0 | 3.2 (1) | 0.026 |
| 50 | U | 0.030 ± 0.007 | 39 | 358 | 0.015 | 8 103 / 0 | 7.9 (1) | 0.024 |
| 100 | F0 | 0.012 ± 0.002 | 91 | 697 | 0.039 | – | – | 0.055 |
| 100 | F1 | 0.012 ± 0.002 | 85 | 657 | 0.036 | 8 064 / 0 | 1.9 (0) | 0.053 |
| 100 | A-rand | 0.007 ± 0.001 | 160 | 423 | 0.018 | 8 081 / 0 | 1.7 (0) | 0.049 |
| 100 | U | 0.015 ± 0.004 | 81 | 449 | 0.018 | 8 101 / 0 | 2.1 (0) | 0.053 |

Observed: (i) the prior step is never accepted on this problem at d = 20, 50 or 100 (0 of ≈ 8 000 draws
per run), so its acceptance curve cannot be shown on P1 (the LG toy, §4, shows 0.117 → 1e-4 from d = 2 to
10; a P1 variant at d = 5–10 is the natural extension and is NOT yet in the drivers); (ii) ESS per exact
evaluation falls 15–30× from d = 20 to 50 and a further 1.5–2× to 100 for every code; the adaptive codes
are 2× more efficient than the frozen ones at d = 50 (0.025–0.030 vs 0.013–0.016) and run 1.6–1.9× faster
in wall time at both dimensions; at d = 100 the tail of 5 000 evaluations is only ≈ 250 τ long, so the τ
estimates (and A-rand's low 0.007) are not reliable; (iii) validity: all codes, F0 included, sit 2.4–3 % (d =
50) and 5 % (d = 100) of a posterior sd from the 200 k references, i.e. at the references' own error (§3);
the pairwise tests with 3 replicates are inconclusive (U at 7.9 SE in one direction at d = 50; everything
within 2.1 SE at d = 100). **Range rule / precision:** `jobs/B7x` generated: references P1-50 and P1-100 with
2 M evaluations per chain (10×), and replicates 3–5 of every B7 configuration with a 20 000-evaluation
tail (26 jobs), to be queued after B8.
Added 08:55 UTC: the drivers now know `P1-5` and `P1-10` (same solver, 5 / 10 KL modes; `scripts/problems.py`
`P1_FAMILY`); `jobs/B7y` = references (400 k evaluations per chain) and F1 / A-rand runs at d = 5 and 10
(3 replicates) for the prior-step acceptance curve. Queue re-ordered (`jobs/all_ordered5`): B8 (remaining),
B7y, B7x, B2, B3, B4, B6c.

## 14. B8 — multimodal posterior, rate w and dimension (70 runs DONE by 10:50 UTC 2026-10-24; `analysis/B8/`)

BA10 and BA30 (far-mode mass 0.1409 at both d: the 1-D marginal along the unit direction does not
depend on d), codes F0 (w = 0) / F1 / A-rand, w ∈ {0.02, 0.05, 0.1}, HMC K = 5, 5 replicates (the w = 0.05
rows reuse A6's runs), budgets as in B1.

| d | code | w | far-mode mass ± SE | z vs 0.1409 | crossings / 1 000 | prior-step acc. | f_pre | test RMSE | S3 ESS of the mode indicator / eval ± SE | S2 wall [s] |
|---|---|---|---|---|---|---|---|---|---|---|
| 10 | F0 | 0 | 0.102 ± 0.033 | −1.2 (stuck) | 0.03 | – | 0.000 | 0.004 | 0.0019 | 297 |
| 10 | F1 | 0.02 / 0.05 / 0.1 | 0.138 ± 0.011 / 0.149 ± 0.004 / 0.142 ± 0.004 | −0.3 / 2.3 / 0.2 | 1.4 / 2.4 / 5.4 | 0.18 | 0.00–0.01 | 0.004–0.008 | 0.0045 / 0.0086 / 0.0131 (± 0.001) | 271–290 |
| 10 | A-rand | 0.02 / 0.05 / 0.1 | 0.121 ± 0.005 / 0.137 ± 0.002 / 0.134 ± 0.004 | −4.3 / −1.7 / −1.8 | 1.6 / 2.8 / 5.0 | 0.18 | 0.002 | 0.002–0.005 | 0.0081 / 0.0143 / 0.0174 (± 0.002–0.004) | 212–248 |
| 30 | F0 | 0 | 0.191 ± 0.017 | +3.0 | 2.5 | – | 0.055 | 0.043 | 0.0048 ± 0.0025 | 327 |
| 30 | F1 | 0.02 / 0.05 / 0.1 | 0.1367 ± 0.0008 / 0.1305 ± 0.0020 / 0.136 ± 0.004 | −5.2 / −5.3 / −1.4 | 3.9 / 4.9 / 4.3 | 0.17–0.18 | 0.06 / 0.06 / 0.00 | 0.045 | 0.0049 / 0.0067 / 0.0124 (± 0.0005) | 279–323 |
| 30 | A-rand | 0.02 / 0.05 / 0.1 | 0.137 ± 0.011 / 0.132 ± 0.007 / 0.139 ± 0.005 | −0.3 / −1.3 / −0.4 | 1.7 / 2.8 / 4.9 | 0.17–0.18 | 0.002 | 0.005–0.008 | 0.0085 / 0.0114 / 0.0155 (± 0.001–0.004) | 239–252 |

Observed: (i) the prior-step acceptance is 0.17–0.18 at both dimensions (the likelihood constrains one
direction only, so it does not fall with d here), and the barrier crossings scale with w (≈ 50 w per 1 000
steps): on this problem the exact step IS the mode-jumping mechanism, and the effective sample size
of the mode indicator per exact evaluation rises monotonically with w for both codes at both d (F1
0.0045 → 0.0131, A-rand 0.0081 → 0.0174 at d = 10). **Range rule:** the optimum sits at the upper grid
edge w = 0.1 → `jobs/B8x` adds w ∈ {0.2, 0.4} (F1 and A-rand, both d, 5 replicates). (ii) A-rand's
finite-run deficit of the far-mode mass at d = 10 grows as w shrinks (14 % at w = 0.02, 2–3 % at 0.05
and 0.1), as expected from the mechanism (fewer forced visits of the far mode, stronger feedback of the
chains' occupation on the network). (iii) At d = 30 the frozen network from the 4 000 preliminary
evaluations is poor (test RMSE 0.045, f_pre 6 %) and the FROZEN code F1 underestimates the far-mode
mass by 3–7 % at w ≤ 0.05 with tiny replicate scatter (a shared initial transient: all replicates start
from the same kind of preliminary-stage state and the within-mode mixing under the poor network is
slow), while A-rand, whose network keeps improving (RMSE 0.005–0.008), is within 1.3 SE at every w; F0
at d = 30 even overshoots (0.19 ± 0.02, z + 3). So at d = 30 continued training buys convergence speed
within the 20 000-step stage, which here outweighs the adaptation transient. `jobs/B8x` also adds BA30
F1 and A-rand at w = 0.05 with an 80 000-evaluation stage (5 replicates) to separate the transient from
a bias. (iv) The adaptive code is 1.1–1.3× faster in wall time on BA than the frozen one (the gap is
smaller than on P1 because the Hamiltonian step is cheap in 10–30 dimensions).

## 15. B7y — P1 at d = 5 and 10 (14 runs DONE by 11:05 UTC 2026-10-24; `analysis/B7y/`, references 400 k evaluations per chain)

| d | code | prior draws / accepted | S3 min ESS/eval ± SE (3 reps) | τ max | S2 wall [s] | test RMSE | offset to reference, rms [sd] |
|---|---|---|---|---|---|---|---|
| 5 | F1 | 8 064 / 0 | 0.864 ± 0.007 | 1.16 | 151 | 0.0012 | 0.0041 |
| 5 | A-rand | 8 074 / 0 | 0.850 ± 0.010 | 1.18 | 101 | 0.0005 | 0.0047 |
| 10 | F1 | 8 064 / 0 | 0.752 ± 0.028 | 1.33 | 361 | 0.0080 | 0.0025 |
| 10 | A-rand | 8 077 / 0 | 0.702 ± 0.010 | 1.43 | 136 | 0.0030 | 0.0018 |

Observed: the prior step is not accepted once even at d = 5 on P1: with noise sd 0.01 and 16 observations
the evidence Z is astronomically small at every d, so the acceptance δ ≈ Z e^{−B₊} of paper Rem. 5.3 is
invisible on this problem; the dimension is the wrong axis for the acceptance curve (the LG toy, §4, shows
the collapse 0.117 → 1e-4 from d = 2 to 10 at noise sd 0.3). **Range rule:** the relevant axis is the noise
level; a P1 variant with noise sd ∈ {0.03, 0.1, 0.3, 1} at d = 20 is the follow-up (drivers to be extended,
`jobs/B7z`). The ESS-per-evaluation curve over d for the HMC K = 5 codes is now complete: 0.85–0.86 (d = 5),
0.70–0.75 (10), 0.37–0.41 (20), 0.013–0.030 (50), 0.007–0.015 (100); the adaptive code is 1.5× (d = 5) to
2.7× (d = 10) faster in wall time than the frozen one.
Added 12:00 UTC: noise-level variants `P1-sd0.03`, `P1-sd0.1`, `P1-sd0.3`, `P1-sd1` (d = 20, `scripts/problems.py`
`P1_NOISE`); `jobs/B7z` = their references (400 k) and F1 / A-rand runs (w = 0.05, 3 replicates) for the
prior-step acceptance against the noise level. Queue re-ordered (`jobs/all_ordered7`): B7x, B8x, B7z, B2,
B3, B4, B6c.

## 16. B7x — d = 50 and 100 with 10× references and six replicates (26 runs DONE by 14:02 UTC 2026-10-24; `analysis/B7x/`, `analysis/B7_6reps/` = B7 + B7x)

References rebuilt from `runs/P1-50_REF_S2-2000000` and `runs/P1-100_REF_S2-2000000` (16 M evaluations each,
12 M states cached; min ESS/eval 0.00118 and 0.00080 → ≈ 19 000 and 13 000 effective samples in the worst
direction, SE ≈ 0.5–0.6 % of a posterior sd; the 200 k caches kept as `*_reference_200k.npz`). Replicates
3–5 ran with a 20 000-evaluation frozen tail (4× the previous one).

| d | code | S3 min ESS/eval, 5 000-tail (reps 0–2) | S3 min ESS/eval, 20 000-tail (reps 3–5) ± SE | τ max (20 000 tail) | tail wall [s] | vs F0, 6 vs 6 reps: max abs z | offset to the 2 M reference: rms / max [sd] (dirs > 3 SE) |
|---|---|---|---|---|---|---|---|
| 50 | F0 | 0.013 | 0.0072 ± 0.0034 | 297 | 626 | – | 0.019 / 0.043 (0) |
| 50 | F1 | 0.016 | 0.0091 ± 0.0040 | 221 | 595 | 2.8 | 0.014 / 0.027 (0) |
| 50 | A-rand | 0.025 | 0.0217 ± 0.0072 | 56 | 322 | 1.9 | 0.010 / 0.024 (1) |
| 50 | U | 0.030 | 0.0254 ± 0.0109 | 57 | 322 | 2.2 | 0.010 / 0.023 (1) |
| 100 | F0 | 0.012 | 0.0041 ± 0.0004 | 248 | 528 | – | 0.019 / 0.039 (0) |
| 100 | F1 | 0.012 | 0.0034 ± 0.0011 | 350 | 635 | 1.8 | 0.026 / 0.052 (0) |
| 100 | A-rand | 0.007 | 0.0064 ± 0.0016 | 174 | 398 | 1.6 | 0.010 / 0.021 (0) |
| 100 | U | 0.015 | 0.0075 ± 0.0028 | 232 | 420 | 1.8 | 0.010 / 0.022 (0) |

Observed: (i) the 5 000-evaluation tails underestimated τ for the frozen codes at d ≥ 50 (a tail of
≈ 50 τ); with 20 000-evaluation tails the adaptive codes are 2.5–3× (d = 50) and 1.5–2× (d = 100) more
efficient per exact evaluation than the frozen ones, and 1.5–2× faster in wall time, because the
network trained on 24 000 evaluations is much better than the one frozen after 4 000 (test RMSE 0.015 vs
0.030 at d = 50, 0.018 vs 0.037 at d = 100, §13); (ii) validity: with six replicates every code agrees with
F0 within 2.8 SE in all 16 directions at both d, and the offsets to the 2 M references (rms 1–2.6 % of sd,
at most one direction beyond 3 SE) are at the level of the runs' own precision (ESS ≈ 1 500–3 500 per run
in the worst direction, i.e. 1.7–2.6 % of sd per run); the 0.5–1 % no-DA effect seen at d = 20 (§11) is
below the resolution here and would need ≈ 20 replicates to test at d ≥ 50. Not pursued further now.

## 17. B8x — larger w and the longer stage at d = 30 (50 runs DONE by 16:40 UTC 2026-10-24; `analysis/B8_all/` = B8 + B8x)

| d | code | w | far-mode mass ± SE (5 reps) | z vs 0.1409 | crossings / 1 000 | S3 ESS of the mode indicator / eval ± SE | S2 wall [s] |
|---|---|---|---|---|---|---|---|
| 10 | F1 | 0.2 / 0.4 | 0.140 ± 0.003 / 0.139 ± 0.003 | −0.3 / −0.9 | 8.7 / 17.5 | 0.024 ± 0.001 / 0.038 ± 0.003 | 238 / 185 |
| 10 | A-rand | 0.2 / 0.4 | 0.146 ± 0.003 / 0.138 ± 0.002 | +1.8 / −1.9 | 9.6 / 17.8 | 0.042 ± 0.016 / 0.053 ± 0.011 | 185 / 143 |
| 30 | F1 | 0.2 / 0.4 | 0.140 ± 0.002 / 0.138 ± 0.002 | −0.4 / −1.4 | 9.7 / 17.8 | 0.021 ± 0.002 / 0.035 ± 0.002 | 253 / 188 |
| 30 | A-rand | 0.2 / 0.4 | 0.140 ± 0.003 / 0.138 ± 0.002 | −0.3 / −1.4 | 9.5 / 17.8 | 0.026 ± 0.002 / 0.036 ± 0.003 | 216 / 170 |
| 30, S2 = 80 000 | F1 | 0.05 | 0.1401 ± 0.0024 | −0.3 | 2.3 | 0.0062 ± 0.0006 | 1 147 |
| 30, S2 = 80 000 | A-rand | 0.05 | 0.1367 ± 0.0015 | −2.8 | 3.0 | 0.0120 ± 0.0029 | 1 021 |

Observed: (i) the mode-indicator efficiency keeps rising with w through the extended grid (F1: 0.0045 →
0.038, A-rand: 0.008 → 0.053 per exact evaluation from w = 0.02 to 0.4 at d = 10; the same factor at
d = 30); crossings are 44 w per 1 000 steps, i.e. the prior step makes essentially all of them, and the
runs get faster with w (fewer sub-chains). The optimum is still at the grid edge: **range rule →
`jobs/B8y` with w ∈ {0.6, 0.8, 1.0}** (w = 1 is the pure prior independence sampler, acceptance 0.18
here). (ii) At w ≥ 0.1 both codes reproduce the far-mode mass within 2 SE at both d. (iii) The d = 30
deficit of the FROZEN code F1 at w = 0.05 (−5.3 SE with 20 000 steps, §14) is gone at 80 000 steps
(0.1401 ± 0.0024): it was the initial transient under the poor frozen network. A-rand at 80 000 shows the
same ≈ 3 % residual deficit as at d = 10 (0.1367 ± 0.0015 here, 0.1368 ± 0.0021 in §12), i.e. the
guaranteed code's transient decays roughly like n^{−1/2} between 20 000 and 80 000 steps at
n_λ = 500, a_λ = 0.75 (λ_n ≈ 0.02 at n = 80 000: an installation every ≈ 45 steps). The theory
predicts that a faster-decaying λ_n shortens this transient: **`jobs/B8z`** = BA10 A-rand, w = 0.05,
(n_λ, a_λ) ∈ {(10, 1), (50, 1), (500, 1), (5 000, 0.5)}, 5 replicates, to test it directly.

## 18. B7z — prior-step acceptance against the noise level (28 runs DONE by 16:47 UTC 2026-10-24; `analysis/B7z/`)

P1 at d = 20 with noise sd ∈ {0.03, 0.1, 0.3, 1} (`P1-sd*`; same observations, 16 of them), references
400 k evaluations per chain (min ESS/eval 0.006–0.012 → 20 000–37 000 effective samples in the worst
direction, SE 0.5–0.7 % of sd), codes F1 and A-rand, HMC K = 5, w = 0.05, 3 replicates.

| noise sd | prior-step acceptance (F1 / A-rand) | S3 min ESS/eval F1 / A-rand | τ max | test RMSE F1 / A-rand | S2 wall [s] | offset to reference, rms [sd] (F1 / A-rand) |
|---|---|---|---|---|---|---|
| 0.01 (P1, §11) | 0 / 0 | 0.37 / 0.37 | 2.8 | 0.019 / 0.009 | 524 / 228 | 0.002 / 0.002 (4 M reference) |
| 0.03 | 1.2e-4 / 0 | 0.55 / 0.59 | 1.7–1.8 | 0.016 / 0.009 | 184 / 148 | 0.0034 / 0.0038 |
| 0.1 | 0.058 / 0.059 | 0.67 / 0.70 | 1.4–1.5 | 0.010 / 0.008 | 112 / 119 | 0.0041 / 0.0048 |
| 0.3 | 0.64 / 0.64 | 0.86 / 0.88 | 1.1–1.2 | 0.006 / 0.006 | 93 / 99 | 0.0033 / 0.0032 |
| 1 | 0.954 / 0.955 | 0.96 / 0.35 ± 0.31 (one replicate with τ ≈ 22, not understood; the other two ≈ 0.95) | 1.0 / 22 | 0.006 / 0.006 | 93 / 96 | 0.0030 / 0.0041 |

Observed: the acceptance of the exact prior step rises from 0 at the paper's noise level through
1e-4 (sd 0.03), 0.06 (0.1), 0.64 (0.3) to 0.95 (1.0): it is governed by how much the data constrain the
posterior relative to the prior (the evidence), as paper Rem. 5.3 states, and not by the dimension
(§13, §15). Offsets to the references are 0.3–0.5 % of a posterior sd for both codes, with 4–8 of 16
directions beyond 3 replicate SE for both alike (reference precision 0.5–0.7 %), so no bias is resolved
at this precision. One A-rand replicate at sd 1 has τ ≈ 22 in its frozen tail (the mean of the other two is
0.95): recorded, not investigated (`runs/P1-sd1_A-rand_hmc5_w0.05_nl500-al0.75_r*`).

## 19. B8y — w up to 1 on the bimodal problem (60 runs DONE by 18:10 UTC 2026-10-24; `analysis/B8_full/` = B8 + B8x + B8y)

Full curve, HMC K = 5, 5 replicates per point, both codes; far-mode mass within 2 SE of 0.1409 at every
w ≥ 0.1 for both codes and both d (the only deviations remain A-rand at d = 10, w = 0.02, §14, and the
frozen code's initial transient at d = 30, w ≤ 0.05, §17).

| w | crossings / 1 000 (both d) | mode-indicator ESS / exact eval, d = 10: F1 / A-rand | d = 30: F1 / A-rand | S2 wall [s], d = 10: F1 / A-rand |
|---|---|---|---|---|
| 0.02 | 1.4–1.7 | 0.0045 / 0.0081 | 0.0049 / 0.0085 | 290 / 248 |
| 0.05 | 2.4–2.8 | 0.0086 / 0.0143 | 0.0067 / 0.0114 | 284 / 230 |
| 0.1 | 4.3–5.4 | 0.0131 / 0.0174 | 0.0124 / 0.0155 | 271 / 212 |
| 0.2 | 8.7–9.7 | 0.0238 / 0.0423 | 0.0207 / 0.0261 | 238 / 185 |
| 0.4 | 17.5–17.8 | 0.0380 / 0.0528 | 0.0354 / 0.0359 | 185 / 143 |
| 0.6 | 25.9–26.3 | 0.0584 / 0.0682 | 0.0573 / 0.0686 | 122 / 96 |
| 0.8 | 35.0–35.1 | 0.0775 / 0.0749 | 0.0750 / 0.0785 | 62 / 53 |
| 1 (pure prior sampler) | 43.4–43.6 | 0.0916 / 0.0916 | 0.0956 / 0.0956 | 3 / 16 |

Observed: on this problem the exact prior step (acceptance 0.18) is the whole mode-mixing mechanism:
crossings are 44 w per 1 000 steps at every w, the mode-indicator efficiency per exact evaluation rises
monotonically to the pure prior sampler at w = 1 (0.09–0.10), and the surrogate-driven Hamiltonian
sub-chain contributes nothing to it (A-rand's advantage over F1 at small w, 1.3–1.8×, is within-mode
mixing on a better network and vanishes above w = 0.6). The range w ∈ [0, 1] is exhausted, so the rule
of note 38 §C stops here: the "optimum at the edge" is the degenerate no-surrogate sampler. For the
paper this problem is the regime δ ≈ 0.18 of Thm 5.1 (the prior step is useful), the complement of P1
(δ = 0, the prior step is pure insurance, §11, §18); a problem between the two (several constrained
directions AND a barrier) is not in the drivers. The adaptation bias of A-rand at w = 0.02 (§14) and
the λ-schedule test `jobs/B8z` (running) complete this block.

## 20. B8z — installation schedule against the adaptation bias on BA10 (20 runs DONE by 19:02 UTC 2026-10-24; `analysis/B8z/`)

BA10, A-rand, HMC K = 5, w = 0.05, production 20 000 evaluations per chain, 5 replicates per schedule
(the (500, 0.75) row is A6 + A6x, 10 replicates); versions received ≈ 21 000–24 500 per run.

| (n_λ, a_λ) | installs per run | far-mode mass ± SE | deficit | z vs 0.1409 | per-replicate values | S3 mode-indicator ESS/eval ± SE | test RMSE at end |
|---|---|---|---|---|---|---|---|
| (5 000, 0.5) ≈ every version | 22 776 | 0.1328 ± 0.0042 | −6 % | −1.9 | 0.125–0.146 | 0.0080 ± 0.0011 | 0.0024 |
| (500, 0.75) | 15 348 | 0.1329 ± 0.0027 | −6 % | −3.0 | 0.114–0.145 | 0.0143 ± 0.0040 | 0.0023 |
| (500, 1) | 10 735 | 0.1358 ± 0.0050 | −4 % | −1.0 | 0.119–0.149 | 0.0105 ± 0.0009 | 0.0023 |
| (50, 1) | 1 992 | 0.1407 ± 0.0035 | 0 % | 0.0 | 0.129–0.148 | 0.0188 ± 0.0062 | 0.0027 |
| (10, 1) | 553 | 0.1478 ± 0.0070 | +5 % | +1.0 | 0.125–0.169 | 0.0124 ± 0.0037 | 0.0020 |

Observed: the finite-run deficit of the far-mode mass decreases monotonically as the installation
schedule decays faster (fewer installations per run), from 6 % when essentially every version is
installed to 0 % at n_λ = 50 (≈ 250 installations per chain, a gap of ≈ 80 steps at the end of the
stage), with no loss of efficiency (the mode-indicator ESS per evaluation is highest at n_λ = 50) and
the same final network quality: the installation lag is the lever that the theory names, and this is
the direct experimental counterpart of Prop. 6.2 / Thm 6.11 on a problem where adaptation without the
rule is biased (§8). At n_λ = 10 the mass overshoots by 5 % (1 SE; one replicate at 0.169). Range rule:
the zero crossing is interior (between n_λ = 50 and 10) but the replicate SEs are of the size of the
effect, so 5 more replicates of (50, 1) and (10, 1) are queued (`jobs/B8z2`).

### B8z with ten replicates for the two fastest schedules (B8z2 DONE by 19:56 UTC 2026-10-24; `analysis/B8z_10reps/`)

| (n_λ, a_λ) | reps | installs per run | far-mode mass ± SE | deficit | z | S3 mode-indicator ESS/eval ± SE |
|---|---|---|---|---|---|---|
| (5 000, 0.5) | 5 | 22 776 | 0.1328 ± 0.0042 | −6 % | −1.9 | 0.0080 ± 0.0011 |
| (500, 0.75) | 10 | 15 348 | 0.1329 ± 0.0027 | −6 % | −3.0 | 0.0143 ± 0.0040 |
| (500, 1) | 5 | 10 735 | 0.1358 ± 0.0050 | −4 % | −1.0 | 0.0105 ± 0.0009 |
| (50, 1) | 10 | 2 000 | 0.1385 ± 0.0027 | −2 % | −0.9 | 0.0229 ± 0.0095 |
| (10, 1) | 10 | 535 | 0.1427 ± 0.0042 | +1 % | +0.4 | 0.0225 ± 0.0095 |

With ten replicates the two fastest schedules are unbiased within 1 SE and the trend with the number of
installations is monotone (−6 %, −6 %, −4 %, −2 %, +1 %); the earlier +5 % at n_λ = 10 was replicate
noise. The schedules with the fewest installations are also the most efficient per exact evaluation
(0.022 vs 0.008–0.014). Block closed.

## 21. Time cap (author, 2026-10-24 20:46 UTC): at most 4 more hours

The remaining queue (B2 67, B3 54, B4 144, B6c 64 jobs, ≈ 20 h) was replaced at 20:55 UTC by
`jobs/final4h` (113 jobs, ≈ 3 h with 3 slots): B6c (61: the averaged rule with the carried pCN β, the first
valid test of that rule), A5 (6: input clip b ∈ {3, 2} on that configuration), B4 trimmed (24: F0 vs A-rand,
preliminary budget {300, 3 000} × production budget {5 000, 20 000}, 3 reps), B2 trimmed (21: K = 1 only,
w ∈ {0, 0.02, 0.05, 0.1, 0.2} at the full preliminary budget and w ∈ {0, 0.1} at 300, 3 reps), A9 (1: a
100 000-evaluation production stage under the rule). Three in-flight K = 10 B2 runs were stopped
(markers cleared; their partial output is not used). Dropped for now, optionally later: the rest of B2
(K = 10, S1 = 300 × all w), B3 (lag vs learning curve on P1; A2/A2x and B8z cover the lag question), the
rest of B4, B5, the continuation part of A9. B2 runs already DONE (7, K = 10) are kept and analysed with
the trimmed set.

## 22. B6c — averaged installation with the carried pCN β (61 runs DONE by 21:20 UTC 2026-10-24; `analysis/B6c_A5/`)

A-avg code: pCN K = 5 with β adapted in S1 and FIXED at the carried value in S2/S3, prior step w = 0.05,
`AveragedInstallation(j0, a_zeta, weight_box)`, `input_clip = 8`, P1 with j0 ∈ {1, 5, 20} × a_ζ ∈ {0.6, 0.8, 1} ×
ω ∈ {None, 10} (3 replicates each; (5, 0.8, 10) with 5) and BA10 with (5, 0.8, 10) (5 replicates).
Comparison: the pCN K = 5 codes of A6 (F0 / F1 / U).

| problem | configuration | S3 min ESS/eval ± SE | S2 mean error rms [abs] | f_pre | versions averaged per chain | test RMSE at end | S2 wall [s] |
|---|---|---|---|---|---|---|---|
| P1 | all 18 (j0, a_ζ, ω) combinations | 0.0030–0.0038 (SE ≤ 0.0005) | 0.030–0.043 | 0.31–0.33 | 447–460 | 0.0112–0.0118 | 47–54 |
| P1 | A6: F0 / F1 / U pcn5 | 0.0031 / 0.0029 / 0.0029 | 0.044 / 0.044 / 0.031 | 0.28 | – | 0.016 / 0.016 / 0.012 | 23 / 22 / 39 |
| BA10 | (5, 0.8, 10) | 0.610 ± 0.007 | 0.0024 (far-mode mass 0.1410 ± 0.0005) | 0.39 | 818 | 0.0097 | 28 |
| BA10 | A6: F0 / F1 / U pcn5 | 0.593 / 0.559 / 0.197 ± 0.074 | 0.0018 / 0.0016 / 0.0020 (far 0.1409 / 0.1409 / 0.1415) | 0.40 / 0.40 / 0.36 | – | 0.010 / 0.013 / 0.009 | 14 / 14 / 23 |

Observed: with the sub-chain parameters fixed as the paper requires, the averaged rule is valid (P1
errors equal to the exact pCN codes'; BA10 far-mode mass exact to 0.0005) and costs nothing: on BA10 it
is as efficient as the frozen codes (0.61 vs 0.56–0.59 per exact evaluation) and 3× more efficient
than the un-guaranteed U (0.20 ± 0.07, whose continuous β tuning hurts), on P1 all pCN codes are
equally inefficient (τ ≈ 400–500: the sub-chain, not the rule, is the limit). None of j0, a_ζ or the
weight box ω = 10 makes a measurable difference (the box never binds: results identical with and
without it), so no interior optimum exists on this grid and the range rule does not apply (the rule's
knobs are inert at this budget). The pinned-β failure of A6 is confirmed as a tuning artefact.

## 23. A5 — input clip (6 runs DONE by 21:40 UTC 2026-10-24; `analysis/A5/`, P1, A-avg pcn5c (5, 0.8, 10), 3 replicates)

| input clip b | fraction of production states with some abs(x_k) > b | test RMSE at end | f_pre | S2 mean error rms | S3 min ESS/eval ± SE |
|---|---|---|---|---|---|
| 8 (B6c) | 0 (max abs(x_k) ≈ 3.5) | 0.0116 | 0.33 | 0.035 | 0.0031 ± 0.0002 |
| 3 | 0.011 | 0.0125 | 0.35 | 0.043 | 0.0030 ± 0.0004 |
| 2 | 0.856 | 0.0186 | 0.33 | 0.046 | 0.0030 ± 0.0003 |

Observed: the clip changes the surrogate, not the target: even with b = 2, where 86 % of the chain's
states have a clipped coordinate, the posterior errors stay at the level of the exact pCN codes
(0.03–0.046) and the efficiency per exact evaluation is unchanged (0.0030); the network trained on
clipped inputs is worse (test RMSE 0.019 vs 0.012) and f_pre rises slightly. A5 pass; b = 8 is a safe
default in the standard-normal internal space (max abs(x_k) ≈ 3.5 here).

## 24. B4 trimmed — when continued training beats freezing (24 runs DONE by 22:15 UTC 2026-10-24; `analysis/B4/`)

P1, HMC K = 5, F0 (frozen, no exact step) vs A-rand (w = 0.05, RandomisedInstallation(500, 0.75)),
preliminary DAMH budget S1 ∈ {300, 3 000} × production budget S2 ∈ {5 000, 20 000} evaluations per chain,
3 replicates (the (3 000, 20 000) cells reuse A6 replicates 0–2), reference = the 4 M run.

| S1 (network at the start of S2) | S2 | code | test RMSE at end | S3 min ESS/eval ± SE | S2 mean error rms [abs] ± SE | S2 wall [s] | S3 wall [s] |
|---|---|---|---|---|---|---|---|
| 300 (RMSE ≈ 0.020) | 5 000 | F0 | 0.0199 | 0.103 ± 0.026 | 0.0225 ± 0.0066 | 178 | 176 |
| 300 | 5 000 | A-rand | 0.0107 | 0.336 ± 0.026 | 0.0100 ± 0.0015 | 72 | 69 |
| 300 | 20 000 | F0 | 0.0195 | 0.076 ± 0.016 | 0.0096 ± 0.0009 | 704 | 173 |
| 300 | 20 000 | A-rand | 0.0090 | 0.393 ± 0.018 | 0.0047 ± 0.0001 | 271 | 66 |
| 3 000 (RMSE ≈ 0.020) | 5 000 | F0 | 0.0201 | 0.293 ± 0.103 | 0.0070 ± 0.0004 | 140 | 140 |
| 3 000 | 5 000 | A-rand | 0.0111 | 0.328 ± 0.033 | 0.0092 ± 0.0008 | 62 | 58 |
| 3 000 | 20 000 | F0 | 0.0192 | 0.393 ± 0.043 | 0.0047 ± 0.0007 | 544 | 136 |
| 3 000 | 20 000 | A-rand | 0.0090 | 0.358 ± 0.028 | 0.0048 ± 0.0003 | 238 | 58 |

Observed: the crossover of paper §6.5(ii) is the quality of the network the production stage starts
with. After a short preliminary stage (300 evaluations per chain) the frozen code is 3.5–5× less
efficient per exact evaluation than the continuously trained one (0.08–0.10 vs 0.34–0.39), twice as
inaccurate at equal budget, and 2.5× slower in wall time; after 3 000 preliminary evaluations the two
have the same ESS per exact evaluation and the same accuracy, and continued training only buys the
2.3× wall-time gain. (Both preliminary networks show the same end-of-S1 test RMSE ≈ 0.02 because the
network keeps training during S1 in either case; the difference is how far it is from the 0.009 that
24 000 evaluations reach.) The cost of the guarantee (w = 0.05, randomised installation) is invisible
in these numbers.

## 25. B2 trimmed — cost of the prior step on P1 (21 K = 1 runs + 8 earlier K = 10 runs DONE by 22:38 UTC 2026-10-24; `analysis/B2/`)

F1 (frozen network + prior step), HMC K = 1, S1 ∈ {3 000, 300}, 3 replicates; the prior step is never
accepted on P1, so its evaluations are pure cost. Multiplier = tail ESS per exact evaluation relative
to w = 0; formula = (1−w) τ/(τ+w) · (1−f_pre)/(w + (1−w)(1−f_pre)) of paper §6.5 with the measured τ and f_pre.

| S1 | w | S3 min ESS/eval ± SE | τ max | f_pre | multiplier (observed) | formula | S2 wall [s] |
|---|---|---|---|---|---|---|---|
| 3 000 | 0 | 0.160 ± 0.025 | 6.6 | 0.015 | 1 | 1 | 127 |
| 3 000 | 0.02 | 0.183 ± 0.010 | 5.6 | 0.015 | 1.14 | 0.98 | 125 |
| 3 000 | 0.05 | 0.163 ± 0.007 | 6.2 | 0.016 | 1.02 | 0.94 | 122 |
| 3 000 | 0.1 | 0.130 ± 0.018 | 8.0 | 0.015 | 0.81 | 0.89 | 118 |
| 3 000 | 0.2 | 0.128 ± 0.016 | 8.1 | 0.015 | 0.80 | 0.78 | 101 |
| 300 | 0 | 0.017 ± 0.001 | 61 | 0.022 | 1 | 1 | 132 |
| 300 | 0.1 | 0.011 ± 0.003 | 102 | 0.022 | 0.68 | 0.90 | 124 |
| 3 000, K = 10 | 0.01 | 0.432 ± 0.029 | 2.3 | 0.000 | – | 0.99 | 1 069 |
| 300, K = 10 | 0.01 / 0.02 | 0.160 ± 0.028 / 0.145 ± 0.018 | 6.7 / 7.0 | 0.000 | – | 0.99 / 0.98 | 1 229 / 1 575 |

Observed: with the good network the measured multiplier follows the formula within the replicate
noise (SE 6–15 %): no measurable cost up to w = 0.05, 20 % at w = 0.1–0.2 (formula 0.89–0.78). With the
poor network (S1 = 300, τ ≈ 60–100, a 5 000-evaluation tail of only ≈ 50–80 τ) the measured loss at w = 0.1
(0.68) exceeds the formula (0.90), but those τ estimates are not reliable. K = 10 is the most efficient
sub-chain length per exact evaluation on P1 with a good network (0.43 vs 0.37 for K = 5 and 0.16 for K = 1)
at 4–5× the wall time of K = 5 (surrogate gradients); K = 10 with w ∈ {0, 0.2} runs in the slack queue.

## 26. A9 — 100 000-evaluation production stage under the randomised rule (1 run DONE by 22:38 UTC; `analysis/A9/`, `runs/P1_A-rand_hmc5_S2-100000_w0.05_nl500-al0.75_r0`)

800 000 exact evaluations over 8 chains (806 514 outer iterations), 4 917 versions published by the
collector, 8 173 received and 8 052 installed in total over the chains, f_pre 0.009, prior-step
acceptance 0, status ok; S2 wall 666 s (0.83 ms per exact evaluation, versus 1.4 ms per evaluation in
the 20 000-evaluation stages: no slowdown with the stage length, the buffers stay bounded); mean error
rms 0.0034 (0.0048 at 20 000, §24) and sd error rms 0.0021 against the 4 M reference; tail ESS per
evaluation 0.316, τ 3.2. A9 (long-run robustness) pass; the continuation part was not run.

### B4 key contrast with six replicates (B4x DONE by 23:08 UTC 2026-10-24; `analysis/B4x/`)

P1, preliminary budget 300, production 20 000 evaluations per chain, HMC K = 5:

| code | reps | S3 min ESS/eval ± SE | τ max | S2 mean error rms ± SE | test RMSE at end | S2 wall [s] | S3 wall [s] |
|---|---|---|---|---|---|---|---|
| F0 (frozen) | 6 | 0.085 ± 0.015 | 13.9 | 0.0108 ± 0.0010 | 0.020 | 711 | 177 |
| A-rand (theorem) | 6 | 0.401 ± 0.011 | 2.5 | 0.0044 ± 0.0002 | 0.009 | 270 | 65 |

With six replicates: starting the production stage from a network trained on 300 evaluations per
chain, the guaranteed continuously trained stage is 4.7 ± 0.8× more efficient per exact evaluation,
2.5× more accurate at equal budget and 2.6× faster in wall time than freezing that network.

### B2 with K = 10 (slack queue DONE by 23:58 UTC 2026-10-24; `analysis/B2_K10/`)

| S1 | w | reps | S3 min ESS/eval ± SE | τ max | multiplier (observed) | formula | S2 wall [s] |
|---|---|---|---|---|---|---|---|
| 3 000 | 0 | 3 | 0.439 ± 0.038 | 2.3 | 1 | 1 | 1 065 |
| 3 000 | 0.01 | 3 | 0.432 ± 0.029 | 2.3 | 0.98 | 0.99 | 1 069 |
| 3 000 | 0.2 | 3 | 0.288 ± 0.008 | 3.5 | 0.66 | 0.76 | 858 |
| 300 | 0.01 / 0.02 | 3 / 2 | 0.160 ± 0.028 / 0.145 ± 0.018 | 6.7 / 7.0 | – | 0.99 / 0.98 | 1 229 / 1 575 |

With K = 10 the prior step at w = 0.2 costs 34 % of the ESS per exact evaluation (formula 24 %: the lazy
chain's τ grows from 2.3 to 3.5 instead of the predicted 3.1), at w ≤ 0.05 nothing measurable. K = 10 is
the most efficient sub-chain length on P1 (0.44 per exact evaluation vs 0.37 for K = 5 and 0.16 for K = 1)
at 4–5× the wall time of K = 5 for this cheap forward model.

## 27. Final summary (2026-10-25 00:00 UTC; campaign closed under the author's 4-hour cap)

**What ran.** 684 runs (68G under `runs/`, smoke runs excluded), 0 failures, between 2026-10-23
22:33 and 2026-10-24 23:58 UTC on 32 cores, 3 MPI jobs of 8 chains + collector at a time. Phases:
references (P1 3.2 M and 4 M, P1-50/P1-100 200 k and 2 M, P1-5/10 and noise variants 400 k), A1, A2 (+A2x),
A5, A6 (+A6x, A6y), A8, A9 (long run only), B1, B2 (K = 1 full grid at S1 = 3 000, corners at S1 = 300, K = 10 at
w ∈ {0, 0.01, 0.02, 0.2}), B4 (F0 vs A-rand, S1 × S2 2×2 grid, key cell with 6 reps), B6c (+A5 clip), B7
(+B7x, B7y, B7z), B8 (+B8x, B8y, B8z, B8z2). Not run: B3 (installation lag on P1; covered by A2/A2x and B8z),
the rest of B2 (K = 10 × S1 = 300 × all w) and B4 (other budgets, F1/U), B5, A3/A4 beyond the unit tests,
A9's continuation, B9 (a calculation from the timings, not done). Validation set: A1, A2, A5, A6, A7
(on the pCN runs), A8, A9 pass.

**Headline findings (numbers in the sections above).**
1. Validity. The guaranteed codes (frozen + prior step, F1; randomised installation + prior step +
   deferred tuning, A-rand) agree with the classically exact frozen code on P1 in all 16 observed
   directions with 10 replicates (max 2.0 and 2.8 SE) and sit within 0.2 % of a posterior sd of the 4 M
   reference; the codes without diminishing adaptation (U, U0) differ at 6–8 SE in the best-constrained
   directions (0.5–1 % of sd) (§11). On the bimodal problem the no-DA code underestimates the far-mode
   mass by 9 % (5.7 SE, 10 reps) at every tested budget, the guaranteed code by 6 % at 20 000 steps and 3 %
   at 80 000 (§8, §12), and the deficit vanishes when the installation schedule decays faster (6 % → 0 %
   from n_λ = 5 000 to 50, 10 reps; §20): the installation lag is the lever the theory names. The averaged
   rule with fixed sub-chain parameters is exact and as efficient as freezing (§22); the input clip
   changes only the surrogate (§23). The prior step is exact on the closed-form toy (§4).
2. The exact prior step. Its acceptance is set by the evidence, not the dimension: 0 on P1 at every d
   and noise sd 0.01, rising to 0.95 at noise sd 1 (§18); 0.18 on the bimodal problem at d = 10 and 30,
   where it is the entire mode-mixing mechanism (crossings 44 w per 1 000 steps up to w = 1, §19) and the
   only thing that frees chains from a mode the network cannot see (A8: 8 of 10 frozen replicates stuck,
   all guaranteed replicates mixed, §7). Its cost where it is never accepted follows the lazy-chain
   formula within noise: none up to w = 0.05, 20–34 % at w = 0.1–0.2 (§11, §25).
3. Continued training. Equal ESS per exact evaluation to freezing when the preliminary network is
   good (P1, 3 000 preliminary evaluations), but 2.3× faster in wall time; 4.7 ± 0.8× more efficient,
   2.5× more accurate and 2.6× faster when the preliminary network is poor (300 evaluations, §24); 2.5–3×
   (d = 50) and 1.5–2× (d = 100) more efficient at higher dimension (§16); at d = 30 on the bimodal problem
   it removes the frozen code's initial transient (§17). The randomised rule's lag (even 570 steps) costs
   no efficiency on P1 (§9) and the coin law holds exactly (§5, §9).
4. Sub-chain and dimension. ESS per exact evaluation of the HMC K = 5 codes falls 0.86 (d = 5), 0.73 (10),
   0.37 (20), 0.02 (50), 0.007 (100) (§13, §15, §16); K = 10 beats K = 5 by 15–20 % per evaluation at 4–5× the
   wall time (§25); pCN is hopeless on P1 (τ ≈ 450) but crosses the bimodal barrier by itself (§6).

**Caveats.** Frozen tails of 5 000 evaluations underestimate τ when τ ≳ 50 (d ≥ 50, poor networks);
the 20 000-tail replicates are the reliable ones (§16). Replicate SEs with 3 replicates are rough;
every headline number above has 5–10. References: P1 4 M, P1-50/100 2 M, others 400 k (their precision is
stated per section). One unexplained outlier (A-rand at noise sd 1, τ ≈ 22, §18). The three K = 10 runs
stopped at the cap are not used. All numbers are from `analysis/<phase>/{metrics,posterior,summary}.csv`;
run directories as named in the sections; drivers and job files in `scripts/` and `jobs/`.

## 28. PDF report (2026-10-26)

`pdf_reports/paper_experiments_2026-10-23/report.pdf` (38 pages; source `report.tex`, build `make`; figures
regenerated by `make_figures.py` run from `toy_examples/`, 14 vector figures under `figures/` with a
`CAPTIONS` dict). Part I = validation (A1–A9 and the deferred-adaptation defect), Part II = the paper set
(B1, B2, B4, B6c/A5, B7, B8 and their extensions), then what was not run, caveats, a suggested content
of the paper's Section 8, the run inventory and the range-extension record. Every number is copied
from this log; figure annotations that go beyond the log (replicate-0 far masses, posterior sds of
v1/v2/v16, per-run install counts) are computed by the figure script from the run outputs.
