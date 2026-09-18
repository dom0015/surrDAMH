# 14 — GRF-diffusion validation runs (2026-09-18)

Longer validation runs of the maintained GRF-diffusion example, on branch `working_Kuba`,
commit `1ded39e` (dirty working tree), format v2. Everything below was produced by running
the code; **no file under `surrDAMH/`, `tests/` or `toy_examples/` was modified**, and no git
operation was performed. Drivers live outside the repository (scratchpad, reproduced in §1.3).

All outputs: `toy_examples/out_grf_validation_2026-09-17/` (one sub-directory per run, 13.4 GB
in total, dominated by `raw_data/`). Wall clock: **00:00:43 → 01:08:45 UTC = 68 min**, against
a 3 h budget.

---

## 1. Setup

### 1.1 Problem

`Solver_diffusion_GRF(xa=-1, xb=1, ya=-1, yb=1, nx=20, ny=20,
covariance_type='squared_exponential', length_scale=0.1, sigma=0.5, nu=1.0, no_parameters=20,
observations_per_dim=2, positivity_transform_factor=1.0, u_left=1.0, u_right=0.0,
source_strength=0.0)`; prior = 20 independent standard normals (internal = physical space);
`observations = solver.generate_artificial_observations(seed=11)`; likelihood
`Normal(mean=observations, sd=0.03)`.

The **true coefficient vector is recoverable**: `generate_artificial_observations(seed=11)` is
literally `np.random.seed(11); z = np.random.normal(size=20)`, so re-running that RNG sequence
reproduces it. Verified: re-evaluating the solver at that `z` reproduces the observations with
`max|Δ| = 0.0`.

```
z_true      = [ 1.749455 -0.286073 -0.484565 -2.653319 -0.008285 -0.319631 -0.536629  0.315403
                0.421051 -1.065603 -0.886240 -0.475733  0.689682  0.561192 -1.305549 -1.119475
                0.736837  1.574634 -0.031075 -0.683447]
observations = [0.64685136 0.29370273 0.65382741 0.30765483]   (noise-free solver output at z_true)
```

### 1.2 Cost of one solver evaluation

Single process, `/dolfinx-env/bin/python3`, 5 warm-up evaluations then 100 timed evaluations at
independent prior draws:

| quantity | value |
|---|---|
| solver construction (mesh + covariance + KL) | 0.110 s |
| **time per evaluation (`set_parameters` + `get_observations`)** | **0.190 ms** |
| 100 evaluations | 0.019 s |

This is the number that shaped every budget below — see §10, finding F1.

### 1.3 Runs

All runs: `use_solvers_pool=False`, `save_snapshots_to_file=True`, `min_snapshots_initial=0`,
`min_snapshots_to_update=0`, `initial_sample_type="lhs"` (except R-D(ii)), surrogate =
`NeuralNetworkUpdaterMinibatches` with exactly the example's hyper-parameters
(`(16,16,16)`, silu, adamw, lr 1e-4, batch 1024, replay 1.0/4096, clip 2.0, wd 1e-3;
`weighting="uniform"`, `output_normalization="likelihood"` — applied automatically from the
likelihood), test data `TestData.generate(..., size=128, seed=11)` on the collector.

Command pattern (from `/workspaces/surrDAMH/toy_examples`):

```
timeout <s> /usr/local/bin/mpiexec -n <k> /dolfinx-env/bin/python3 -m mpi4py <driver>.py
```

| run | dir | `-n` | stages (all `time_limit`-bounded, with an iteration cap) | planned wall | actual wall |
|---|---|---|---|---|---|
| R-A | `R-A_reference_MH` | 6 (6 samplers, no collector) | `MH/pCN β=0.2, time_limit=2700, max_samples=1_000_000` | ≈60 min | **43.1 min** (6.5 sampling + 36.6 report) |
| R-B | `R-B_DAMH_hamiltonian` | 7 (6 + collector) | `MH/pCN β=0.2, time_limit=480, max_samples=200_000` → `DAMH/HamiltonianInfinite, num_steps=100, step=0.05, subchain_max_length=1, surrogate_model_updates=True, time_limit=2700, max_samples=300_000, is_excluded=True` | ≈60 min | **56.2 min** (48.6 + 7.6) |
| R-C | `R-C_DAMH_SMU_pCN` | 7 (6 + collector) | `MH/pCN β=0.2, time_limit=480, max_samples=200_000` → `DAMH/pCN β=0.2, subchain_max_length=5, SMU=True, time_limit=1200, max_samples=500_000, is_excluded=True` → `DAMH/pCN β=0.2, subchain_max_length=5, SMU=False, time_limit=900, max_samples=500_000` | ≈45 min | **61.5 min** (36.2 + 25.3) |
| R-D(i) | `R-D1_repro_run1`, `R-D1_repro_run2` | 3 each (2 + collector) | R-B's warm-up, `max_samples=100_000` | 2 × 3 min | 1.7 + 1.6 min |
| R-D(ii) | `R-D2_continuation` | 6 (no collector) | `MH/pCN β=0.2, time_limit=300, max_samples=200_000`, `initial_sample_type="continued"`, `continued_from_dir=<R-A dir>` | 5 min | 9.3 min (1.5 + 7.7 report) |

R-A, R-B and R-C ran **concurrently** (6+7+7 = 20 of 32 cores; `uptime` load ≈ 19 during the
overlap), R-D(i) alongside them (23 ranks), R-D(ii) after R-A's `last_sample` was written.

**Two deliberate deviations from the plan, both documented rather than hidden:**

1. Every stage got an **iteration cap (`max_samples`) in addition to `time_limit`**. With a
   0.19 ms solver a plain MH sampler runs at 1000–3600 iterations/s/chain, so a 45-minute stage
   would produce 20–40 M states per run; `write_report` cannot post-process that inside the
   budget (§10 F1). The caps bound the chain length only — they cannot change the sampled
   distribution.
2. Consequently the pCN warm-ups ended after ~2.5 min (200 k iterations/chain) instead of the
   planned 8 min, and R-A sampled for 6.5 min instead of 45. The resulting chains are still
   very long (R-A: 6 000 000 states, R̂ ≤ 1.00017, pooled ESS ≈ 40 000 per parameter).

Reference throughput measured in a 3-stage, 3-rank, 60 s-per-stage pilot
(`out_grf_validation_2026-09-17/pilot`): MH/pCN 1490 it/s/chain (69 % acceptance),
DAMH/HamiltonianInfinite 57 it/s/chain, DAMH/pCN with `subchain_max_length=5` 1270 it/s/chain.

---

## 2. Validation item 1 — mechanics

| run | exit | orphan procs | `run_manifest.json` `finished_at` | `format_version` | `report_extended.html` | field statistics in report | measurement-point field | `surrogate_quality_test.csv` |
|---|---|---|---|---|---|---|---|---|
| R-A | 0 | 0 | 00:13:39 | 2 | yes, 3 333 849 B | yes | yes | n/a (`use_collector=False`) |
| R-B | 0 | 0 | 00:55:47 | 2 | yes, 5 430 380 B | yes | yes | yes |
| R-C | 0 | 0 | 00:43:26 | 2 | yes, 7 168 240 B | yes | yes | yes |
| R-D1 run1 | 0 | 0 | 00:09:20 | 2 | not requested | — | — | yes |
| R-D1 run2 | 0 | 0 | 00:11:00 | 2 | not requested | — | — | yes |
| R-D2 | 0 | 0 | 00:25:45 | 2 | yes, 3 621 940 B | yes | yes | n/a |

Every `mpiexec` exited 0 well inside its `timeout`; `ps` afterwards showed **0** surviving
`mpiexec`/`mpi4py` processes. `write_report(observations=..., ranking_mode="likelihood")` also
wrote `summary.csv` and the two `best_fit_solver_visualization_*.png` per reporting run.
"field statistics in report" = the strings *Posterior diffusion coefficient field* and
*Posterior solution at measurement points* are present in the HTML, i.e. the
`field_builder`/`coords`/`measurement_points` path in `core.py` was taken.

`write_report` was not requested for the two R-D(i) runs (their purpose is byte-identity, and
report generation writes only into `post_processing_output/`).

### Stage counters (summed over the 6 chains)

| run | stage | accepted | rejected | prerejected | Σ = iterations | full-model evaluations |
|---|---|---|---|---|---|---|
| R-A | `alg0000_MH` | 4 142 014 | 1 857 986 | 0 | 6 000 000 | 6 000 000 |
| R-B | `alg0000_MH` | 828 138 | 371 862 | 0 | 1 200 000 | 1 200 000 |
| R-B | `alg0001_DAMH-SMU` | 662 669 | 14 312 | 3 757 | 680 738 | 676 981 |
| R-C | `alg0000_MH` | 826 922 | 373 078 | 0 | 1 200 000 | 1 200 000 |
| R-C | `alg0001_DAMH-SMU` | 2 933 138 | 57 967 | 8 895 | 3 000 000 | 2 991 105 |
| R-C | `alg0002_DAMH` | 1 461 239 | 26 125 | 4 434 | 1 491 798 | 1 487 364 |
| R-D2 | `alg0000_MH` | 827 791 | 372 209 | 0 | 1 200 000 | 1 200 000 |

---

## 3. Validation item 2 — convergence of the reference R-A

6 chains × 1 000 001 states, 10 % burn-in, batch means with 64 batches per chain
(`tests/helpers_statistics.py` formulas; `gelman_rubin` imported from there, the pooled
mean/variance/SE code re-implemented diagonal-only with pandas I/O because `np.loadtxt` is too
slow for 286 MB sample files — identical formulas to `pooled_chain_statistics`).

**R̂ max = 1.00017, R̂ mean = 1.00009** over the 20 parameters.

| par | R̂ | posterior mean | posterior sd | sd / prior sd | min ESS per chain | pooled ESS | z of `z_true` |
|---|---|---|---|---|---|---|---|
| 0 | 1.00016 | +0.0780 | 0.9691 | 0.969 | 5 498 | 41 726 | +1.72 |
| 1 | 1.00011 | −0.0283 | 0.9831 | 0.983 | 5 902 | 41 121 | −0.26 |
| 2 | 1.00009 | −0.1413 | 0.9254 | 0.925 | 4 311 | 38 063 | −0.37 |
| 3 | 1.00001 | −0.0223 | 0.9697 | 0.970 | 6 014 | 43 141 | −2.71 |
| 4 | 1.00008 | −0.0853 | 0.9680 | 0.968 | 5 252 | 42 796 | +0.08 |
| 5 | 1.00005 | +0.1048 | 0.9777 | 0.978 | 5 725 | 40 763 | −0.43 |
| 6 | 1.00002 | +0.0699 | 0.9882 | 0.988 | 4 945 | 36 507 | −0.61 |
| 7 | 1.00009 | −0.0565 | 0.9654 | 0.965 | 5 684 | 43 777 | +0.39 |
| 8 | 1.00014 | +0.0467 | 0.9962 | 0.996 | 5 625 | 37 991 | +0.38 |
| 9 | 1.00009 | −0.1428 | 0.9314 | 0.931 | 5 616 | 38 977 | −0.99 |
| 10 | 1.00010 | −0.0527 | 0.9729 | 0.973 | 6 085 | 46 202 | −0.86 |
| 11 | 1.00012 | −0.1444 | 0.9412 | 0.941 | 5 421 | 37 847 | −0.35 |
| 12 | 1.00002 | −0.0784 | 0.9869 | 0.987 | 6 029 | 40 235 | +0.78 |
| 13 | 1.00011 | −0.0904 | 0.9537 | 0.954 | 5 962 | 40 763 | +0.68 |
| 14 | 1.00014 | +0.0835 | 0.9846 | 0.985 | 5 055 | 36 734 | −1.41 |
| 15 | 1.00009 | −0.1351 | 0.9621 | 0.962 | 4 690 | 36 769 | −1.02 |
| 16 | 1.00003 | +0.0324 | 0.9710 | 0.971 | 4 785 | 38 653 | +0.73 |
| 17 | 1.00017 | −0.0692 | 0.9507 | 0.951 | 5 521 | 37 107 | +1.73 |
| 18 | 1.00009 | +0.1498 | 0.9399 | 0.940 | 4 260 | 38 415 | −0.19 |
| 19 | 1.00004 | +0.0704 | 0.9666 | 0.967 | 6 302 | 42 469 | −0.78 |

SE of the pooled mean: 0.0045–0.0052 per parameter. `summary.csv` reports integrated
autocorrelation time 140.6 for this stage, consistent with pooled ESS ≈ 6 000 000/140 ≈ 43 000.

**Which KL modes are informed? None, individually.** The smallest marginal posterior sd is
0.9254 (mode 2), i.e. at most a 7.5 % shrinkage against the prior sd 1; no mode falls below
0.9. This is not a sampling failure — it is a property of the example, quantified next.

### Why (identifiability of this inverse problem)

| diagnostic | value |
|---|---|
| prior-predictive sd of the 4 observations (500 prior draws, `default_rng(7)`) | 0.0453, 0.0906, 0.0493, 0.0985 |
| ... in units of the noise sd 0.03 | 1.51, 3.02, 1.64, 3.28 |
| singular values of ∂G/∂z at z = 0 (forward differences, h = 1e−3) | 1.516e−1, 2.357e−2, 6.6e−11, 2.0e−11 |
| ... signal-to-noise (σ/0.03) | **5.06, 0.79, 2e−9, 7e−10** |
| variance fraction of the field captured by the first 20 KL modes (of 400 DOF) | 0.240 |

The 4 observations sit on a 2×2 grid, and with `u_left=1, u_right=0, source_strength=0` the
solution is essentially a function of x, so the two y-levels duplicate each other
(obs₀ = 0.6469 vs obs₂ = 0.6538, obs₁ = 0.2937 vs obs₃ = 0.3077). The Jacobian therefore has
**effective rank 2**, and only one direction carries signal above the noise. Projecting the
posterior onto the two leading right singular vectors v₁, v₂ (prior sd = 1 in each):

| direction | R-A posterior mean | R-A posterior sd | linearised-Gaussian prediction 1/√(1+SNR²) |
|---|---|---|---|
| v₁ | +0.3778 | **0.2116** | 0.194 |
| v₂ | +0.1673 | **0.7769** | 0.786 |

The sampled posterior matches the linearised prediction closely in both directions — an
independent confirmation that the sampler is targeting the right distribution. The per-mode
marginals stay near the prior simply because a rank-2 constraint spread over 20 coordinates
shrinks no single coordinate much.

---

## 4. Validation item 3 — DAMH equals MH on the real problem (decisive check)

Pooled over the 6 chains, 10 % burn-in, batch-means SE (64 batches per chain, combined in
quadrature with length weights); the DAMH SE and the R-A SE are combined in quadrature.
**Pass criterion fixed before looking: no parameter beyond 4 SE**, on either the 20 means or
the 20 variances.

| comparison | n pooled | max \|Δmean\|/SE | >2 SE | >3 SE | >4 SE | max \|Δvar\|/SE | >2 SE | >3 SE | >4 SE |
|---|---|---|---|---|---|---|---|---|---|
| R-B `alg0001_DAMH-SMU` (Hamiltonian, SMU) | 612 668 | **2.65** | 3 | 0 | **0** | **2.06** | 1 | 0 | **0** |
| R-C `alg0001_DAMH-SMU` (pCN, subchain 5, SMU) | 2 700 000 | **2.43** | 2 | 0 | **0** | **1.61** | 0 | 0 | **0** |
| R-C `alg0002_DAMH` (pCN, subchain 5, frozen surrogate) | 1 342 620 | **1.64** | 0 | 0 | **0** | **2.01** | 1 | 0 | **0** |

**PASS.** Nothing exceeds 3 SE, let alone 4. Across the 120 comparisons (3 stages × 20 means +
3 × 20 variances) 7 exceed 2 SE, against ≈5.5 expected for independent normal deviates.

Mean deviation over the 20 parameters (a check for a *common* shift, which per-parameter
maxima would not reveal):

| comparison | mean Δmean/SE (neg/20) | mean Δvar/SE (neg/20) |
|---|---|---|
| R-B `alg0001_DAMH-SMU` | −0.38 (12) | −0.60 (14) |
| R-C `alg0001_DAMH-SMU` | −0.11 (10) | −0.34 (12) |
| R-C `alg0002_DAMH` | −0.18 (13) | −0.25 (13) |

All three show the same mild negative offset, including `alg0002_DAMH`, whose frozen surrogate
makes it an *exactly* correct DAMH kernel. Since all three are compared against the **same**
R-A realisation, a common offset of a few tenths of an SE is the expected signature of R-A's
own estimation error, not of a DAMH bias. It is reported here because it is the only
systematic-looking feature in the data.

### Same comparison in the identifiable directions (where a surrogate bias would concentrate)

| run | stage | dir | mean | sd | Δmean/SE vs R-A | Δvar/SE vs R-A |
|---|---|---|---|---|---|---|
| R-A | `alg0000_MH` | v₁ | +0.3778 | 0.2116 | reference | reference |
| R-A | `alg0000_MH` | v₂ | +0.1673 | 0.7769 | reference | reference |
| R-B | `alg0001_DAMH-SMU` | v₁ | +0.3776 | 0.2115 | −0.40 | −0.23 |
| R-B | `alg0001_DAMH-SMU` | v₂ | +0.1813 | 0.7726 | +1.11 | −0.60 |
| R-C | `alg0001_DAMH-SMU` | v₁ | +0.3775 | 0.2108 | −0.74 | **−3.10** |
| R-C | `alg0001_DAMH-SMU` | v₂ | +0.1702 | 0.7749 | +0.80 | −1.07 |
| R-C | `alg0002_DAMH` | v₁ | +0.3777 | 0.2114 | −0.16 | −0.54 |
| R-C | `alg0002_DAMH` | v₂ | +0.1717 | 0.7754 | +1.02 | −0.71 |

The one value worth naming: R-C's DAMH-SMU stage has the v₁ **variance** 3.10 SE below R-A's
(sd 0.21077 vs 0.21158, i.e. −0.4 % in sd, −0.8 % in variance). It is below the 4 SE criterion,
it is not reproduced by the other SMU stage (R-B, −0.23 SE) nor by the frozen-surrogate stage
(−0.54 SE), and this direction was chosen *a posteriori* as the most sensitive one, so it is
not a 1-in-16 event drawn blind. It is flagged, not claimed as a defect: DAMH-SMU updates the
surrogate inside the stage, which is the one place where exactness is not guaranteed, so this
is the number to re-measure if the question is ever revisited.

### Posterior predictive

Mean of the **exact** observations of `accepted` rows in `raw_data` (`prerejected` rows
excluded — they have no exact observations), minus the reference `observations`, in units of
the noise sd 0.03:

| run | stage | accepted rows | (mean obs − data)/0.03 | max |
|---|---|---|---|---|
| R-A | `alg0000_MH` | 4 142 014 | +0.120, +0.241, −0.059, −0.118 | 0.241 |
| R-B | `alg0001_DAMH-SMU` | 662 669 | +0.120, +0.241, −0.053, −0.105 | 0.241 |
| R-C | `alg0001_DAMH-SMU` | 2 933 138 | +0.123, +0.246, −0.054, −0.109 | 0.246 |
| R-C | `alg0002_DAMH` | 1 461 239 | +0.122, +0.245, −0.054, −0.108 | 0.245 |

All four agree to within 0.006 noise sd of each other, and all lie within 0.25 noise sd of the
data: the observed data sit comfortably inside the posterior predictive band.

---

## 5. Validation item 4 — efficiency (reported, not judged)

| run | stage | wall (sampling) | full-model evals | accepted | rejected | prerejected | `summary.csv` autocorr | `summary.csv` CpUS | pooled ESS (median par) | **ESS per full-model eval** |
|---|---|---|---|---|---|---|---|---|---|---|
| R-A | `alg0000_MH` | 6.5 min | 6 000 000 | 4 142 014 | 1 857 986 | 0 | 140.64 | 140.64 | 39 606 | 0.0066 |
| R-B | `alg0000_MH` | ~2.5 min | 1 200 000 | 828 138 | 371 862 | 0 | 138.67 | 138.67 | — | — |
| R-B | `alg0001_DAMH-SMU` | ~46 min | 676 981 | 662 669 | 14 312 | 3 757 | **3.68** | **3.66** | 171 658 | **0.2536 (38× MH)** |
| R-C | `alg0000_MH` | ~2.5 min | 1 200 000 | 826 922 | 373 078 | 0 | 140.72 | 140.72 | — | — |
| R-C | `alg0001_DAMH-SMU` | ~14 min | 2 991 105 | 2 933 138 | 57 967 | 8 895 | **28.55** | **28.46** | 93 751 | **0.0313 (4.7× MH)** |
| R-C | `alg0002_DAMH` | ~19 min | 1 487 364 | 1 461 239 | 26 125 | 4 434 | **28.62** | **28.54** | 46 894 | **0.0315 (4.8× MH)** |

Sub-chain diagnostics from `summary.csv`: R-B `subchain_acc_rate` 0.994, `outer_acc_given_move`
0.979; R-C `subchain_acc_rate` 0.691 / 0.690, `subchain_move_rate` 0.997, `outer_acc_given_move`
0.981 / 0.982.

Reading: the surrogate almost never pre-rejects (0.3–0.6 % of iterations), because it is far
more accurate than the noise sd — so DAMH here does **not** save full-model evaluations by
pre-rejecting. It pays off through the *proposal*: the Hamiltonian proposal on the surrogate
cuts the autocorrelation time from 140.6 to 3.68 (38× more effective samples per full-model
evaluation), and a 5-step pCN sub-chain cuts it to 28.6 — a factor of 4.9, i.e. exactly the
sub-chain length, as it should be.

Caveat, stated plainly: with a 0.19 ms forward model none of this is a wall-clock win — R-A
produced 6 M states in 6.5 min while R-B produced 0.68 M in 46 min. The ESS-per-evaluation
column is what transfers to an expensive solver; the wall-clock column is not.

---

## 6. Validation item 5 — surrogate quality

From `sampling_output/surrogate_quality_test.csv` (held-out set: 128 points, seed 11, the same
set for both runs), against the noise sd 0.03:

| run | updates logged | final `snapshots_total` | RMSE first | RMSE last | RMSE min | RMSE last / 0.03 | weighted RMSE first | last | min |
|---|---|---|---|---|---|---|---|---|---|
| R-B | 29 298 | 1 876 981 | 0.08535 | **0.00119** | 0.00116 | **0.0395** | 0.04707 | **0.00048** | 0.00039 |
| R-C | 13 104 | 5 678 469 | 0.08308 | **0.00158** | 0.00156 | **0.0527** | 0.03907 | **0.00044** | 0.00035 |

The network converges to an RMSE of 4–5 % of the observation noise sd, and the final RMSE is
within 3 % of the best value ever reached (no late degradation). This is the direct explanation
of the near-zero pre-rejection rate in §5.

---

## 7. Validation item 6 — truth recovery (diagnostic)

z-scores of the true coefficients under R-A's posterior are in §3's last column:
max |z| = **2.71** (mode 3, `z_true` = −2.653, posterior mean −0.022, sd 0.970); 18 of 20 lie
within 2. Because the marginal posteriors are essentially the prior (§3), these z-scores are
almost exactly the prior z-scores of `z_true` itself, so this diagnostic carries little
information for this example — a value near 2.7 for a draw that is itself 2.65 prior sd from
zero is expected, not a sign of a mis-targeted posterior. The posterior predictive check (§4)
is the meaningful truth-recovery statement here, and it passes.

---

## 8. Validation item 7 — R-D reproducibility and continuation

**(i) Two identical runs, `max_samples`-bounded (not time-bounded, so the chain length itself is
deterministic).** SHA-256 of every output file of `R-D1_repro_run1` vs `R-D1_repro_run2`:

| file | sha256 (first 12) both runs | bytes | identical |
|---|---|---|---|
| `samples/alg0000_MH/rank0000.csv` | `ebef97c78692` | 28 721 798 | **yes** |
| `samples/alg0000_MH/rank0001.csv` | `7cbe2318d11a` | 28 690 144 | **yes** |
| `notes/alg0000_MH/rank0000.csv` | `e344ff2f438b` | 65 | **yes** |
| `notes/alg0000_MH/rank0001.csv` | `d46187dbabc7` | 66 | **yes** |
| `raw_data/alg0000_MH/rank0000.csv` | `b07e940270cd` | 53 692 231 | **yes** |
| `raw_data/alg0000_MH/rank0001.csv` | `ab57a8651c56` | 53 693 357 | **yes** |
| `last_sample/alg0000_MH/rank000{0,1}.npz` | (array compare) | (20,) | **yes** |

Byte-identical including `raw_data`, i.e. the accept/reject sequence and every proposal match,
with the `lhs` starts and the per-rank seeds. Note this holds even though a collector was
running and retraining the surrogate throughout — as it must, since an MH stage never consults
the surrogate (`Stage.__post_init__` forces `surrogate_model_updates=False` for `MH`).

**(ii) Continuation.** `initial_sample_type="continued"`,
`continued_from_dir=.../R-A_reference_MH`, 6 samplers:

| rank | max\|first sample − R-A `last_sample`\| | bitwise equal |
|---|---|---|
| 0 | 0.000e+00 | **yes** |
| 1 | 0.000e+00 | **yes** |
| 2 | 0.000e+00 | **yes** |
| 3 | 0.000e+00 | **yes** |
| 4 | 0.000e+00 | **yes** |
| 5 | 0.000e+00 | **yes** |

The continued run's `alg0000_MH` counters (827 791 / 372 209 / 0) are statistically
indistinguishable from R-A's per-iteration rates, and its `summary.csv` autocorrelation (140.56)
matches R-A's (140.64).

---

## 9. Files to look at

```
toy_examples/out_grf_validation_2026-09-17/
  R-A_reference_MH/      sampling_output/{samples,raw_data,notes,last_sample,run_manifest.json}
                         post_processing_output/{report_extended.html,summary.csv,best_fit_solver_visualization_*.png}
  R-B_DAMH_hamiltonian/  + sampling_output/surrogate_quality{,_test}.csv, surrogate_test_data.npz
  R-C_DAMH_SMU_pCN/      + sampling_output/surrogate_quality{,_test}.csv, surrogate_test_data.npz
  R-D1_repro_run1/, R-D1_repro_run2/, R-D2_continuation/
  pilot/                 throughput pilot (3 stages × 60 s, 3 ranks)
```

---

## 10. Conclusions

### Pass / fail per criterion

| # | criterion | verdict |
|---|---|---|
| 1 | mechanics: exit 0, no orphans, manifest finalized, report + field statistics, surrogate CSVs | **PASS** (all 6 runs) |
| 2 | R-A convergence (R̂, ESS, informed modes) | **PASS** — R̂ ≤ 1.00017, pooled ESS ≈ 37 000–46 000/parameter; *no* mode individually informed, explained and quantified (rank-2 Jacobian) |
| 3 | DAMH ≡ MH: no parameter beyond 4 SE | **PASS** — worst is 2.65 SE (means) / 2.06 SE (variances); nothing beyond 3 SE |
| 3b | posterior predictive within the noise scale | **PASS** — ≤ 0.25 noise sd for every run |
| 4 | efficiency reported | done — 38× (Hamiltonian DAMH-SMU) and 4.7–4.8× (pCN sub-chain 5) ESS per full-model evaluation |
| 5 | surrogate quality | **PASS** — held-out RMSE 4–5 % of the noise sd, no late degradation |
| 6 | truth recovery | diagnostic only; max \|z\| = 2.71, uninformative here by construction |
| 7 | R-D byte identity + continuation | **PASS** — byte-identical reruns incl. `raw_data`; continuation exact on all 6 chains |

### Findings worth acting on

**F1 — `write_report` cost is linear in the *decompressed* chain length, with a forward-model
call per state; it dominated every long run here.** `SamplingFramework._write_report_rank0`
calls `compute_posterior_field_statistics(solver.set_parameters_and_get_observations)`, which
loops over **every decompressed posterior state** (multiplicities expanded, all stages, all
chains) and evaluates the solver once per state. Measured: R-A, 6 000 000 states →
**36.6 min of report generation for 6.5 min of sampling**; R-C, 5 691 798 states → 25.3 min for
36.2 min of sampling; R-B, 1 880 738 states → 7.6 min. `compute_posterior_field_statistics`
already accepts `n_max_samples` (with a seeded RNG for sub-sampling), but `write_report` does
not expose it and passes nothing, so the sub-sampling path is unreachable from the public API.
Consequence for users: a run that fills an hour of wall time with a cheap solver cannot be
post-processed by its own driver. Cheapest fix would be to give `write_report` an
`n_max_samples`/`field_statistics_max_samples` argument forwarded to both
`compute_posterior_field_statistics` calls. *Not filed as a bug in a behaviour sense — the
result is correct, the cost is the issue.*

**F2 — for this example the surrogate never pre-rejects, so DAMH's benefit is entirely in the
proposal.** Pre-rejection rates 0.55 % (R-B), 0.30 % (R-C SMU), 0.30 % (R-C frozen): the
network's held-out RMSE (0.0012–0.0016) is ~25× smaller than the noise sd, so the surrogate
posterior and the true posterior almost never disagree about a proposal. The measured gain
comes from the sub-chain/Hamiltonian proposal decorrelating the chain (autocorrelation
140.6 → 3.68 / 28.6). Worth knowing when quoting "DAMH saves solver calls".

**F3 — the example's inverse problem is effectively rank 2 in 20 unknowns.** 2×2 sensors with
`u_left=1, u_right=0, source_strength=0` duplicate each other across y; the Jacobian singular
values are 1.5e−1, 2.4e−2, 6.6e−11, 2.0e−11, and only the first exceeds the noise
(SNR 5.06 vs 0.79). Combined with the first 20 KL modes capturing only 24 % of the field
variance (length scale 0.1 on a 2×2 domain), the posterior is ≈ the prior in every marginal.
Not a defect — but any future test that asserts "mode k is informed" on this example will fail,
and anyone using it to demonstrate inversion should probably raise `observations_per_dim`,
add a source term, or lengthen `length_scale`.

**F4 (mild, flagged not claimed).** R-C's DAMH-SMU variance in the single identifiable
direction v₁ is 3.10 SE below R-A's (−0.8 % in variance). Below the criterion, not reproduced
by the other two DAMH stages, and found in an a-posteriori-chosen projection. The
frozen-surrogate stage in the same run agrees with R-A to −0.54 SE, which is the evidence
against this being a DAMH-SMU bias.

### What I could not check

- **Runs of the originally planned length** (45–60 min of *sampling* each). Blocked by F1: the
  reports for such runs would not have fit the 3 h budget. What was run is 6 M / 0.68 M / 4.5 M
  states per run, and the statistical conclusions rest on those.
- **Whether the mild common negative offset in §4 is R-A estimation error or a real shared
  effect.** Distinguishing them needs a second independent reference MH run; not run here.
- **`surrogate_quality.csv` (training-set) trends** were not analysed; only the held-out
  `surrogate_quality_test.csv` was.
- **Contents of the HTML reports beyond the presence of the field-statistics sections**
  (file size and section markers were checked; the plots were not inspected).
- Throughput numbers in §1.3 come from a 3-rank pilot and the production runs shared 32 cores,
  so per-stage rates in §5 are contended and should not be read as isolated performance.
