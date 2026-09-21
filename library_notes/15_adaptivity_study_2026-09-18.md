# 15 — Usability of the adaptive proposal (adaptivity study, 2026-09-18)

**Dated record of a one-time study.** Question from the author: how usable is the library's
current adaptivity (the adaptive random-walk proposal) for the intended main workflow —
**(1) preliminary stage → (2) DAMH with a Hamiltonian proposal and an NN surrogate updated on
the fly → (3) DAMH with the surrogate frozen** — and for the other sampling options used for
comparison; which knobs must the user guess from preliminary runs; which lightweight
adaptivity options would be worth adding later. Budget 6 h; used ≈ 1 h 20 min wall clock
(toy track 35 min and GRF track 48 min in parallel, then synthesis).

Two tracks, run by two agents in parallel; every strong claim below was independently
re-checked by the manager (code read, a numerical check, or the raw CSV):

* **Toy track** — linear-Gaussian problems with a *closed-form posterior* (exact oracle
  comparisons): 600 runs, `d ∈ {2, 10, 20}`, condition numbers 1/10/100, posterior
  correlations up to 0.98, `run_local` plus one `mpiexec -n 4` run.
  `toy_examples/out_adaptivity_study_2026-09-18/toy/RESULTS_TOY.md` (+ `results_toy.csv`, 7 figures, drivers). 4.9 MB.
* **GRF track** — the maintained FEniCSx example `toy_examples/sampling_diffusion_grf.py`
  (20 KL parameters, 4 observations, in-process 0.19 ms solver, NN surrogate): 32 MPI runs
  (`-n 4`, reference `-n 7`), a 600 s six-chain pCN reference (R-hat ≤ 1.0003), stage
  budgets 90/180 s. `…/grf/RESULTS_GRF.md` (+ `results_grf.csv`, 3 figures, drivers, every
  run's config JSON and log). **4.0 GB, nothing deleted — the author decides** (852 MB are
  logs dominated by the library's per-iteration `Progress:` prints, most of the rest is
  `raw_data/` from runs with `save_snapshots_to_file=True`). It is an `out_*` directory:
  off-limits to later sessions unless named explicitly.

No library code, test, example or doc was modified by the study. Every number below was
computed; none is filled in. ESS from one 90 s stage scatters ±~40 %, so only ratios ≳ 1.5×
are read as real (the load-bearing GRF comparisons were repeated at 180 s and reproduced
within ~15 %); toy figures use 3 seeds where marked, with 2–3× seed spread at d ≥ 10.

---

## 1. What adaptivity exists today (from the code)

| Aspect | State on 2026-09-18 |
|---|---|
| What adapts | Only `Stage(adaptive=True)` with `proposal_type="RWMH"` → `GaussRandomWalk_adaptive` (`modules/proposals.py`). pCN forces `adaptive=False`; block proposals raise; **`Hamiltonian`/`HamiltonianInfinite` have no adaptivity at all** — `hamiltonian_step_size`, `hamiltonian_num_steps` and the mass (`proposal_sd_or_cov`, default `1.0`) are fixed for the stage. |
| Knobs | `adaptive_target_rate` (0.25), `adaptive_corr_limit` (0.3 — estimated correlations are **clipped** entrywise to \|ρ\| ≤ 0.3), `adaptive_sample_limit` (None = unbounded history). `period = 10` (adapt every 10 proposals) is hardcoded in the class and exposed nowhere. |
| Mechanism | Every `period` proposals: acceptance-weighted `np.cov` of all (bounded) *proposed* samples → correlations clipped → scaled by `coef`; `coef *= ratio**(2/d)` (capped 2.0/0.5) **only** when the *lifetime* mean acceptance leaves `[target/1.2, target·1.2]`. Adaptation never stops within a stage. |
| Carry-over | The final covariance of an `adaptive=True` stage (Allreduce-averaged over sampler ranks under MPI; single chain in `run_local`) becomes `prev_cov` for later stages, consumed **only** by an RWMH stage with `proposal_sd_or_cov=None`. Set only after adaptive stages and then kept, so `adaptive → fixed → fixed(None)` hands the *first* stage's covariance to the third. An adaptive **first** stage must be given `proposal_sd_or_cov` (else `assert prev_cov is not None`). **Never reaches a Hamiltonian mass** (`build_proposal` hardcodes `1.0`). |
| Inside DAMH | `adapt()` runs once per outer iteration, on the exact-posterior Metropolis ratio between the outer state and the sub-chain endpoint, and **only on iterations whose sub-chain moved** — pre-rejected iterations are invisible to it (`Algorithm_DAMH.run`). |
| Persistence / diagnostics | The adapted covariance is **persisted nowhere** (`run_manifest.json` stores the *input* `proposal_sd_or_cov`, `null` for a carry-over stage; `notes/` only counters; `last_sample/` only parameters). The only trace is a `prop_cov` line the MPI sampler prints at the end of the stage; `run_local` prints nothing. No per-period adaptation log, no ESS in `summary.csv`. |

Also noticed: the shipped GRF example sets `surrogate_model_updates=True` on its pCN
warm-up stage, where it is refused with a printed warning (no gradient proposal) — harmless,
but it is in every log.

---

## 2. Results — toy track (closed-form posteriors)

Baselines: *oracle* = fixed RW with `(2.38²/d)·C_post` (optimal scaling); *naive* = fixed
isotropic RW `sd = 1` (the prior sd — what one writes with no preliminary run). Metric:
min-over-parameters ESS per full-model evaluation (Geyer, 10 % burn-in); covariance *shape*
error `‖P/trP − C/trC‖_F/‖C/trC‖_F`; posterior check in batch-means SE (tolerance 4).

### 2.1 The target acceptance is hit; the efficiency is not (A1, 5000 evaluations)

| problem | adaptive / oracle | naive / oracle |
|---|---|---|
| d2 κ1 · κ10 (ρ 0.82) · κ100 (ρ 0.98) | 0.87 · 0.56 · 0.18 | 0.87 · 0.49 · 0.19 |
| d10 κ1 · κ10 · κ100 | 0.55 · **0.53** · 0.12 | 0.36 · **0.05** · 0.05 |
| d20 κ1 · κ10 · κ100 | 0.13 · 0.12 · 0.08 | 0.12 · 0.12 · 0.08 |

Acceptance sits at 0.21–0.28 (target 0.25) everywhere; at d = 20 the adapted proposal is
indistinguishable from the naive isotropic guess on every κ — the *scale* is recovered, the
*shape* essentially not (shape error stays > 1 at d = 20, and several rows are identical at
1000 and 5000 evaluations because the covariance stopped being updated). The clear win is the
middle ground (d10, κ 10–100: 2.5–10× the naive guess). At d = 10/20 the acceptance
trajectory oscillates between ≈ 0 and ≈ 0.7 with a ~2000-evaluation period and has not
settled by 5000 (`fig_A1_acceptance_trajectory.png`).

### 2.2 Two separable defects in `adapt()` (A1b) — both verified by the manager

1. **The covariance *shape* update is gated behind the *scale* trigger.** `set_covariance`
   is called only inside the two out-of-band branches; whenever the acceptance rate is already
   within ±20 % of target the freshly computed estimate is discarded, shape included. On
   `d2 κ10` the covariance was installed in 8 of 500 periods, the last at proposal 660 of
   5000. *Verified by reading `adapt()`: there is no `else` branch.* Ungating alone (~2 lines)
   takes `d20 κ1` from 0.18 → 0.75 of the oracle and `d10 κ1` from 0.46 → 0.86 (§2.6).
2. **Clipping correlations entrywise to \|ρ\| ≤ 0.3 breaks positive semi-definiteness.** At
   d = 20 the clipped matrix had a negative eigenvalue in 491/500 (κ10) and 495/500 (κ100)
   periods; `RandomState.multivariate_normal` then warns and samples via an SVD
   factorisation, i.e. the chain runs with a proposal covariance nobody specified (MH stays
   valid; the knob is undefined exactly where shape matters). *Verified by the manager:
   200/200 random rank-2 20-D correlation matrices became indefinite after clipping;
   harmless at d = 2.* On the correlated d = 2 problems the clip is the whole shape error
   (0.40 → 0.02 without it).

Also: the unbounded history makes `adapt()` O(n²·d) — 1.6–2.9× wall-time overhead at 5000
evaluations with a microsecond solver; irrelevant next to a FEM solve, real memory in long
stages (and the GRF track measured a 2.6× throughput loss over 90 s, §3.1).

### 2.3 Knob sensitivity (A2, 20 000 evaluations, 3 seeds) — no default is right, one knob is unsafe

| knob | d2 κ10 ρ 0.82 (oracle ESS 2221) | d10 κ100 (oracle 470) |
|---|---|---|
| defaults | 1350 | 56 |
| `corr_limit` 0.9 / 1.0 | **1907 / 1907** (shape error 0.40 → 0.03) | **20 / 6** (covariance near-singular, proposal collapses) |
| `target_rate` 0.1 / 0.234 / 0.4 | 673 / 1052 / 1112 | 20 / 43 / 49 |
| `sample_limit` 100 / 500 | 657 / **25** — scale 27 500× the oracle, all 3 seeds | 41 / 12 |
| `period` 50 / 200 (monkeypatched) | 1202 / 1363 | 30 / 62 |

* `adaptive_corr_limit` has **opposite signs** in low and high dimension (+41 % ESS at
  d = 2, 3–10× loss at d = 10): the clip doubles as a crude regulariser of a near-singular
  sample covariance. It cannot be chosen without knowing `d` and the true correlation
  structure — i.e. the answer.
* `adaptive_sample_limit` is **unstable**: with a bounded window the windowed sample
  covariance ≈ the current proposal covariance, so `P ← coef·P` becomes an unanchored
  multiplicative recursion (the unbounded history is anchored by the early samples). No guard
  exists.
* `target_rate` 0.234/0.25/0.4 are within ~25 %; 0.1 is bad. `period` is within seed noise —
  exposing it is not worth doing.

### 2.4 Adapt-then-freeze — the "preliminary stage" use case (A3)

Stage 1 adaptive for N1 evaluations → stage 2 fixed RWMH with `proposal_sd_or_cov=None`,
20 000 evaluations, 3 seeds; stage-2 ESS:

| problem | N1 = 200 / 500 / 1000 / 3000 | naive | oracle |
|---|---|---|---|
| d2 κ10 ρ 0.82 | 950 / 974 / 1028 / 1245 | 1060 | 2221 |
| d10 κ100 | 11 / 10 / 16 / 27 (spread ≈ mean) | 6 | 470 |
| d20 κ100 | 5.4 / 5.9 / 5.3 / 5.5 | 5.9 | 210 |

At d = 2 the carried covariance needs 3000 adaptive evaluations to beat the naive guess, by
17 %; at d = 20 it never does. The frozen stage's acceptance landed anywhere in 0.02–0.65
although stage 1 ended near 0.25: `current_rate` is a **lifetime** mean of the acceptance
weights, a lagging indicator, so the hand-over scale suits a mixture of past regimes. MPI
(`-n 4`, N1 = 1000): the Allreduce is exactly the arithmetic mean of the four ranks'
covariances, R-hat ≤ 1.010, pooled posterior within 2.24 SE — but the four inputs differed
by **6.6× in trace**, so the mean is dominated by the widest rank. The manifest records
`null` for the carry-over stage: such a run is not reproducible from its own manifest.

### 2.5 Adaptivity inside DAMH (A4; fixed wrong surrogate `A' = 1.3A`, 10 000 exact evaluations)

| K | fraction of iterations that moved | adapted scale vs per-step oracle | ESS per exact eval, adaptive vs oracle |
|---|---|---|---|
| 1 | 0.002–0.19 | **3.6–4.7× too large** | 0.133 vs 0.013 (d2, but 5.8 M surrogate calls vs 23 k); 0.0014 vs 0.0024 (d10) |
| 5 | 0.04–0.37 | **2.4–9.3× too large** | 0.012 vs 0.008 (d2); 0.0007 vs 0.0016 (d10) |
| 20 | 0.99–1.00 | **5–10× too small** | 0.0005 vs 0.0024 (d2); equal (d10) |

`adapt()` hit 0.23–0.29 in every case; what it scaled *to* flips sign with K. At K = 1/5 most
iterations are pre-rejected and `adapt()` only sees proposals that already passed the
surrogate — a selection-biased sample it *over*-scales to satisfy. At K = 20 nearly every
iteration moves and the documented endpoint-vs-per-step under-scaling appears, 2–3× stronger
than the `sqrt(K·inner_rate)` rule of thumb. Per exact evaluation the over-scaled K = 1
proposal *looks* like a 10× win; per unit of work it is a ~50× loss (250× more surrogate
calls). On the toy the surrogate is free, which is why this did not diverge — on GRF it did (§3.3).

### 2.6 What a small change would buy (A7 prototypes, library untouched; ESS / oracle, 5000 evaluations, 3 seeds)

| problem | shipped | ungate only | **ungate + vanishing shrinkage** `C ← (1−δ)C + δ·diag C`, `δ = min(1, 2d/n)` | textbook AM `(2.38²/d)(C+εI)` |
|---|---|---|---|---|
| d2 κ10 ρ 0.82 | 0.56 | 0.51 | **1.13** | 0.97 |
| d2 κ100 ρ 0.98 | 0.16 | 0.18 | **0.51** | 0.60 |
| d10 κ1 · κ10 · κ100 | 0.46 · 0.40 · 0.12 | 0.86 · 0.62 · 0.10 | **0.90** · 0.41 · 0.06 | 0.07 · 0.04 · 0.05 |
| d20 κ1 · κ10 · κ100 | 0.18 · 0.15 · 0.08 | 0.75 · 0.22 · 0.09 | 0.50 · **0.28** · 0.09 | 0.44 · 0.10 · 0.09 |

Ungating is nearly free and never hurt beyond seed noise; ungate + shrinkage beats the shipped
rule on 7 of 9 problems and removes the indefinite matrices. **Textbook adaptive Metropolis
is a disaster from d = 10 up here** — the `coef` feedback is load-bearing, because the
covariance is estimated from *proposed* samples (which carry the proposal's own spread).
Differences under ~1.5× are unresolved.

### 2.7 Hamiltonian proposals — the cost of guessing (A5; d10 κ100, exact gradients, 3000 evaluations)

| mass \ `step_size` | 0.01 | 0.05 | 0.1 | 0.2 | 0.5 |
|---|---|---|---|---|---|
| I (default), 5 / 20 / 100 steps | 8 / 56 / 435 | 78 / 550 / 434 | 215 / 57 / 181 | **1 / 1 / 1** | **1 / 1 / 1** |
| inv(posterior cov), 5 / 20 / 100 | 4 / 19 / 720 | 24 / 720 / 1364 | 102 / **2701** / 2701 | 738 / 2701 / 1104 | 2701 / 2701 / 27 |
| posterior cov itself, all 15 cells | **1** | **1** | **1** | **1** | **1** |

(2701 = the whole post-burn-in chain, i.e. independent samples.) With the default mass the
usable region is a narrow band; `step_size ≥ 0.2` gives acceptance exactly 0 — a single-point
"chain" — and the library defaults (0.1, 10 steps) sit at the edge of the cliff, 13–50×
below the best cell. `HamiltonianInfinite` behaved like `Hamiltonian` on this
fully-informed target. See §3.2 for why the mass row that "froze" here helped on GRF.

### 2.8 Correctness (A6)

100 000 evaluations, 3 seeds, ESS 1900–6800: the never-stopping adaptive chain matches the
closed form within 1.8 / 2.7 SE (mean / covariance), as does the oracle — no detectable bias.
Every large "deviation" elsewhere in the toy track comes with ESS < 64, where the SE is itself
unreliable: recorded, not evidence of bias.

---

## 3. Results — GRF-diffusion track (the realistic problem)

Reference: `b4_reference_pcn`, 6 chains × 600 s, 2.67 M exact evaluations, R-hat ≤ 1.0003.
Posterior covariance eigenvalues **0.044, 0.60, then 0.95–1.12** (one strongly informed
direction, one weakly, 18 prior-like; 25× anisotropy), largest parameter correlation only
**0.171**. Metric: ESS of the projection on the best-informed direction v1 (the hardest
direction), per exact evaluation and per second; `dev` = max over parameters of
|mean − reference| in combined SE (≲ 4 = consistent).

### 3.1 The intended workflow: does the preliminary stage matter? (B1)

Stage 1 preliminary (90 s) → stage 2 DAMH `HamiltonianInfinite` (L 100, step 0.05, K 1, SMU,
90 s) → stage 3 same, surrogate frozen (60 s); `-n 4` (3 chains).

| preliminary stage | prelim: exact evals · acc · ESS v1/eval · R-hat | stage 2 (SMU) ESS/eval | stage 3 (frozen) ESS/eval | surrogate test RMSE at end |
|---|---|---|---|---|
| pCN β 0.2 (shipped) | 364 566 · 0.690 · 0.0507 · 1.002 | 0.424 | 0.537 | 0.00131 |
| pCN β 0.2, **30 s only** | 99 687 · 0.689 · 0.0455 · 1.005 | 0.386 | 0.492 | 0.00155 |
| RWMH fixed sd 0.3 | 363 829 · 0.358 · **0.0583** · 1.002 | 0.392 | 0.519 | 0.00130 |
| RWMH adaptive, defaults, from sd 0.3 | 137 419 · **0.249** · **0.0208** · 1.007 | 0.471 | 0.610 | 0.00109 |
| RWMH adaptive, from sd 1.0 | 183 959 · 0.244 · 0.0138 · 1.003 | 0.390 | 0.469 | 0.00160 |
| RWMH adaptive, `corr_limit=1.0` | 135 193 · 0.254 · 0.0178 · **2.349** (dev 45.8 SE) | 0.401 | 0.490 | 0.00193 |
| RWMH adaptive, `sample_limit=1000` | 335 652 · 0.352 · **0.0026** · 1.031 | 0.547 | 0.584 | 0.00186 |

* **The adaptive scale converges fast** (acceptance 0.244–0.254 within ~5000 iterations from
  either start) **and still makes the preliminary stage worse**: ESS/eval 0.0208 vs 0.0583
  for the fixed proposal it started from (2.8×) and 0.0507 for pCN (2.4×). Two causes: the
  adapted covariances, recovered from the logs (`adapted_cov.py` — nothing persists them),
  score 10–393× when whitened by the true posterior (1 = perfect; their Allreduce mean 11.8×),
  with anisotropy not aligned with the posterior's; and the unbounded history cost
  **2.6× throughput** (137 k vs 364 k evaluations in 90 s).
* **The `corr_limit=0.3` clip binds everywhere** (max \|ρ\| = 0.300 exactly in every adapted
  matrix) although the true posterior's largest correlation is 0.171 — it truncates estimation
  noise, not real structure. Removing it (`corr_limit=1.0`) makes the sample covariance
  numerically singular and the three chains **disagree** (R-hat 2.35, 45.8 SE) — the same
  high-d collapse as the toy §2.3. With the clip on, `multivariate_normal` still warned three
  times about an indefinite covariance (one per chain).
* **`adaptive_sample_limit=1000`** restores throughput but the proposal scale oscillates by
  ~1.6× for the whole stage, acceptance never reaches target (0.35) and ESS/eval is
  **22× worse than fixed**.
* **Chains disagree on what to adapt to**: per-rank proposal sds at stage end differ by up to
  2.3× (defaults) and **17×** (`sample_limit`); only their Allreduce mean survives.
* **For the intended workflow the preliminary stage barely matters**: every DAMH-SMU stage
  landed at ESS/eval 0.39–0.55 and every frozen stage at 0.47–0.61 regardless of what came
  before — even the 30 s pCN warm-up (3.7× fewer snapshots) — and all agree with the
  reference to ≤ 4 SE. The surrogate crosses the likelihood noise (0.03) at ~5000 snapshots
  and is an order of magnitude below it by ~20 000, long before any preliminary stage ends
  (`fig_surrogate_rmse_vs_snapshots.png`). The only preliminary stage that hurt is the
  non-converged `corr_limit=1.0` one (slowest surrogate, 0.00193). Freezing the surrogate
  gains ~15 %.

### 3.2 Hamiltonian knobs on the real problem — nothing adapts them (B2; same start and same restored surrogate for every row)

| knob | ESS v1/eval | ESS v1/s | acc | pre-rej | note |
|---|---|---|---|---|---|
| step 0.005 / 0.01 / **0.02** / 0.05 (shipped) / 0.1 / 0.2 (L 100) | 0.65 / 0.28 / **1.18** / 0.42 / 0.53 / 0.27 | 104 / 44 / **145** / 56 / 79 / 41 | 0.97–0.98 → 0.90 | 0 → 0.07 | acceptance is flat where ESS varies 4× |
| step **0.5** | — | — | **0.000** | **1.000** | **14 783 proposals all pre-rejected: zero exact evaluations, a constant chain, no warning, no error** |
| L 20 / 100 / 400 (step 0.05) | 0.27 / 0.42 / 0.14 | **129** / 56 / **4.9** | 0.97 | ≤ 0.004 | L 400 is 26× worse per second than L 20 |
| 180 s repeats: step 0.05 mass 1 → step 0.02 | 0.362 → **1.086** | 56 → **178** | | | the 3× is reproducible |
| 180 s: mass Σ̂ (posterior cov from the preliminary stage) | **1.292** | **185** | 0.884 | 0.093 | **3.6× / 3.3× over mass 1** |
| 180 s: mass Σ̂⁻¹ (textbook HMC) | 0.290 | 43 | 0.976 | 0.0002 | worse than the default |
| 90 s: mass diag(Σ̂)⁻¹ | 0.105 | 16 | 0.973 | 0.002 | worse still |

* A **silent total failure** sits 10× above the shipped step size; at 0.2 the pre-rejection
  rate is already 7 %.
* **The acceptance rate carries no information about the step size** in the usable band
  (0.96–0.98 from 0.005 to 0.1 while ESS/eval varies 4×): a user cannot tune it from what the
  run prints; only an ESS calculation reveals it, and a single 90 s point is too noisy
  (non-monotone 0.65 / 0.28 / 1.18 for 0.005 / 0.01 / 0.02).
* **The mass carry-over and the toy/GRF "contradiction".** On GRF, `mass = Σ̂` (what the
  adaptation estimates) is worth 3.3–3.6× and `mass = Σ̂⁻¹` (the textbook HMC choice) is
  worse than the default; on the toy (§2.7) `mass = Σ` froze the chain in all 30 cells and
  `Σ⁻¹` gave the plateau. Both are real measurements. Manager's reconciliation (a reading of
  the code, not a separate experiment): for **`Hamiltonian`** the position step is
  `ε·M⁻¹p`, so `M ≈ Σ⁻¹` is right and `M = Σ` is the inverse of what is wanted — the toy's
  result, for both proposal types on that fully-informed target. For **`HamiltonianInfinite`**
  the mass enters the *prior-preserving rotation angle* `ε/√λ_M` per eigen-direction: on
  GRF's target (2 informed directions, 18 prior-like) `M = Σ̂` rotates exactly the narrow
  directions faster (0.24 rad/step instead of 0.05) and leaves the 18 others at the default,
  which helps; on the toy's target (all 10 directions informed, λ down to 0.005) the same
  choice spins the narrow directions through several radians per step and the chain freezes.
  **There is no blanket transform that makes "wire the adapted covariance into the mass"
  correct** — it depends on the proposal class and on how many directions the data inform.
  Any such feature needs an explicit per-class rule and a pre-rejection guard behind it.

### 3.3 The proposal inside the DAMH stage (B3; single 90 s DAMH-SMU stage, same start/surrogate)

| proposal | K | iterations | exact evals | pre-rej | ESS v1/eval | ESS v1/s | R-hat | dev (SE) |
|---|---|---|---|---|---|---|---|---|
| pCN β 0.2 | 5 | 300 064 | 299 208 | 0.003 | 0.239 | **796** | 1.000 | 1.6 |
| RWMH fixed sd 0.3 | 1 | 909 854 | 326 065 | 0.64 | 0.122 | 441 | 1.001 | 1.3 |
| HamiltonianInfinite (shipped) | 1 | 11 998 | 11 941 | 0.005 | 0.419 | 56 | 1.003 | 1.7 |
| HamiltonianInfinite | 5 | 3 034 | 3 034 | 0.000 | **0.687** | 23 | 1.004 | 1.7 |
| **RWMH adaptive**, from sd 0.3 | 1 | **2 775 431** | **229** | **0.99992** | (meaningless) | 2.8 | **1.72** | **11.8** |
| RWMH adaptive | 5 | 614 540 | **327** | 0.99947 | (meaningless) | 3.1 | 1.28 | 5.9 |
| RWMH adaptive, `corr_limit=1.0` | 1 | 2 979 786 | **567** | 0.99981 | (meaningless) | 3.2 | 2.55 | 25.3 |

* **`adaptive=True` inside a DAMH stage diverges** — the sharpest result of the study,
  reproduced for K = 1, K = 5 and `corr_limit=1.0`. The proposal sd grew from 0.3 to 1.0–3.9
  per component (largest eigenvalue 38, a −0.325 eigenvalue on one rank); 99.99 % of
  proposals were pre-rejected; the chains went nowhere. Mechanism (the toy §2.5 saw the same
  sign at K = 1/5, only bounded because its surrogate was free): `adapt()` only sees the
  iterations whose sub-chain moved, and on those the exact ratio accepts 94–96 %, so the
  measured rate is ≫ 0.25 on a tiny surviving sample → `coef` grows → more pre-rejection →
  an even more biased sample. Nothing in the loop sees the 99.99 % that were thrown away. The
  identical non-adaptive stage (`RWMH fixed sd 0.3`) is healthy and has the second-best
  throughput of the campaign.
* **On this cheap solver the shipped Hamiltonian configuration loses in wall clock.** Per
  exact solve it is the best proposal (0.42–0.69 vs 0.24 for pCN K 5); per second it is the
  worst by an order of magnitude (56 vs 796 ESS/s), because 100 surrogate-gradient
  evaluations per proposal cost far more than one 0.19 ms FEM solve. Per sampler process the
  shipped stage 2 delivers 18.5 ESS/s against 34.3 for plain 6-chain pCN with no surrogate at
  all — half the speed of not using the library's machinery. This is a property of *this*
  example's solver cost (the earlier GRF campaign's finding F2 said the same: the surrogate
  pre-rejects < 1 %, the gain is proposal quality); DAMH-Hamiltonian is designed for the regime
  where an exact solve costs ≫ 100 surrogate gradients. `subchain_max_length=5` helps pCN
  by exactly 5× per evaluation and hurts the Hamiltonian proposal (sub-chain acceptance is
  already 0.995 — nothing to filter).
* Correctness: every non-degenerate stage in the whole GRF campaign agrees with the reference
  to ≤ 4 SE. The failures (`corr_limit=1.0` preliminary, `step_size=0.5`, the three adaptive
  DAMH runs) are sampling failures — chains that never mixed or never moved — not bias.

---

## 4. Discussion

### 4.1 What adaptivity the library offers now

One mechanism: covariance adaptation of the plain Gaussian random walk, usable as a
preliminary MH stage whose final covariance can seed a *later RWMH* stage. Measured behaviour:
it finds the **scale** reliably (target acceptance hit in every run, on both tracks) and the
**shape** poorly above d ≈ 10 — because the shape estimate is discarded whenever the scale is
acceptable (§2.2 defect 1), because the 0.3 correlation clip truncates noise and breaks
positive-definiteness in high d (defect 2; on GRF it binds everywhere despite a true max
correlation of 0.17), and because a bounded history is unstable (§2.3, §3.1). Where it works
(moderate d, moderate anisotropy) it is a 2.5–10× win over guessing; on the 20-D GRF target
it was a 2.8× *loss* against the fixed proposal it started from within a 90 s stage, and its
carry-over reaches only an RWMH stage. **Inside a DAMH stage it is not usable at all**
(§3.3). Nothing adapts pCN's β, the Hamiltonian `step_size`/`num_steps`/mass,
`subchain_max_length`, the NN hyper-parameters or the retrain thresholds. No run output
reports an ESS, so none of the efficiency differences above is visible to a user without
post-processing by hand.

### 4.2 What the user must guess today (and what a preliminary run does and does not tell them)

| knob | stage | evidence of sensitivity | does a preliminary run help? |
|---|---|---|---|
| `hamiltonian_step_size` | 2, 3 | 3× ESS between 0.02 and the shipped 0.05 on GRF; **silent total failure at 0.5** (100 % pre-rejection, zero exact evals); toy: frozen chain at ≥ 0.2 with the default mass | **No** — the printed acceptance is ≈ 0.97 across the whole usable range; only an ESS scan (several minutes per point) finds it |
| `hamiltonian_num_steps` | 2, 3 | 26× in ESS/s between L 20 and L 400 (GRF); per evaluation L 100 best, per second L 20 best | No — depends on the solver-vs-surrogate cost ratio, which nothing reports |
| Hamiltonian mass (`proposal_sd_or_cov`) | 2, 3 | 3.3–3.6× available on GRF from the preliminary stage's covariance; the "obvious" inverse loses a further ~4×; on the fully-informed toy the opposite | Partly — the covariance exists (printed once) but can't be carried, and the right transform depends on proposal class and target (§3.2) |
| `proposal_sd_or_cov` of an adaptive / first stage | 1 | mandatory; sets `coef`'s anchor, which the adaptation never fully forgets (toy: 1350 vs 1145 ESS from sd 0.1 vs 1.0) | It *is* the preliminary run |
| `adaptive_corr_limit` | 1 | +41 % at d 2, −3 to −10× at d 10, chains disagree (R-hat 2.35) at 1.0 on GRF | No — the right value needs the true correlation structure |
| `adaptive_sample_limit` | 1 | 4-orders-of-magnitude blow-up (toy), 22× loss (GRF) | No — unsafe at any value tested; leave `None` |
| `adaptive_target_rate` | 1 | 0.234 / 0.25 / 0.4 within 25 %; 0.1 bad | Default is fine |
| length of the preliminary adaptive stage | 1 | toy: > 3000 evaluations to beat naive at d 2, never at d 20; GRF: the surrogate is good after ~5000 snapshots regardless | For the surrogate, 30 s was enough; for the covariance, 90 s was not |
| `subchain_max_length` | 2, 3 | 5× gain with pCN, 2.4× wall-clock loss with Hamiltonian (GRF); flips the sign of adaptive mis-scaling (toy) | No — depends on the sub-chain acceptance, visible only in `subchain_stats` |
| whether `adaptive=True` is safe in a DAMH stage | 2 | it is not (§3.3, §2.5) | — |
| whether the proposal family fits the solver cost | 2 | per evaluation Hamiltonian wins 3.4×, per second loses 14× on GRF | No — nothing prints ESS/s or the surrogate-call count |

### 4.3 Lightweight adaptivity options worth implementing later (prioritised; each tied to a measurement)

1. **Ungate the covariance install** — call `set_covariance` every period, keep the `coef`
   update gated (~2 lines). Toy: 0.18 → 0.75 of oracle on d20 κ1, 0.46 → 0.86 on d10 κ1, no
   observed harm (§2.6).
2. **Replace the entrywise correlation clip by shrinkage toward the diagonal**
   (`(1−δ)C + δ·diag C`, vanishing δ, or an eigenvalue ridge) and **warn on an indefinite
   covariance** (one `eigvalsh` per period). Removes the non-PSD matrices entirely, fixes
   the correlated-problem shape error 0.40 → 0.015, wins on 7/9 toy problems (§2.6), would not
   bind on GRF where the true correlations are 0.17 (§3.1). Not textbook AM — that collapsed
   from d = 10 up (§2.6).
3. **Make `adaptive=True` in a DAMH stage either safe or impossible**: count every outer
   iteration in `adapt()`, scoring a pre-rejected one as acceptance 0 (so the loop sees what
   it throws away), or refuse the combination. Today it diverges (§3.3).
4. **A pre-rejection / no-move guard**: a stage that pre-rejects > 90 % (or accepts nothing)
   over its first few hundred iterations should print a loud warning naming the knob. Would
   have caught `step_size=0.5` (zero evaluations, silent) and the adaptive-DAMH divergence
   instantly (§3.2, §3.3).
5. **Persist the adapted covariance** (`adaptive_covariance/<stage>/rank%04d.npz` next to
   `last_sample/`, plus the Allreduce mean in the manifest) and a per-period
   `adaptive_stats.csv` (period, recent and lifetime acceptance, `coef`, `tr(cov)`, min
   eigenvalue). Today a `proposal_sd_or_cov=None` run is not reproducible from its manifest
   and the oscillation in `fig_A1_acceptance_trajectory.png` is invisible (§2.4).
6. **Guard `adaptive_sample_limit`** (refuse, or cap the per-period scale change and warn when
   `tr(cov)` grows > 10× within a stage) and **bound the cost differently** — a running
   (Welford) weighted covariance removes the O(n²) history and the 2.6× throughput loss
   without the window's instability (§2.3, §3.1).
7. **Use a recent-window acceptance estimate for the hand-over** (report both), so the frozen
   stage does not inherit a scale tuned to a lifetime average (§2.4); and **pool the ranks'
   samples instead of averaging their covariances** (6.6× / 17× disagreement, §2.4, §3.1).
8. **Report ESS per exact evaluation and per second in `summary.csv`** (and the number of
   surrogate calls per stage). Every conclusion here needed it and none of it is printed
   (§3.3, §2.5).
9. **Hamiltonian family**: (a) let `proposal_sd_or_cov=None` on a Hamiltonian stage pick up
   the carried covariance, but only through an explicit, per-class transform with the guard of
   item 4 behind it — `Σ̂⁻¹` for `Hamiltonian`, and for `HamiltonianInfinite` the evidence is
   split (Σ̂ helped 3.3× on GRF, froze the fully-informed toy; §3.2) so it needs a study of its
   own before becoming a default; (b) a step-size adaptation cannot be driven by the acceptance
   rate (flat at 0.97 over a 20× step range on GRF) — it would have to maximise expected
   squared jump distance per evaluation, a design decision rather than a patch.
10. **Do not** expose `period` (within seed noise, §2.3), and do not adopt a windowed rate
    estimate alone (helps at d 10/20, hurts at d 2, §2.6).

For the intended 3-stage workflow specifically: the preliminary stage's job is to feed the
surrogate, and for that even a 30 s fixed-proposal or pCN stage was enough on GRF; the
adaptive RW brought nothing to that stage and cannot reach the Hamiltonian stage. The knobs
that actually decided the workflow's efficiency were `hamiltonian_step_size` (3×), the mass
(3.3×), `hamiltonian_num_steps` (26× in wall clock) and the proposal family vs. solver cost
(14× in wall clock) — none of which is adapted or diagnosed today.

## 5. Limits of this study

Toy: 3 seeds on the headline cases (2–3× spread at d ≥ 10; rankings within ~1.5× unresolved),
one seed for the DAMH and Hamiltonian sweeps, prototypes at one budget. GRF: one run per
configuration, 90 s stages (±~40 % ESS scatter; the load-bearing comparisons were repeated at
180 s), chains started from a common continued state for B2/B3, machine load 4.5–21 during
the runs (per-second numbers ±~20 %, per-evaluation numbers unaffected), one problem whose
posterior is effectively rank-2 — a stress test for shape adaptation, not a typical one. Not
checked: the toy/GRF mass reconciliation (§3.2) as an experiment of its own; DAMH stages with
NN *gradients* under adaptation; anything about pCN's β, the NN hyper-parameters, or
`min_snapshots_*`, none of which adapt. No library change was made; the prototype rules of
§2.6 live only in `toy/run_A7.py`.
