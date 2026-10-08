# 19 — Sampling-efficiency study on the corrected GRF-diffusion problem (2026-09-28)

**Dated record of one campaign**, requested by the author: fix the GRF solver, choose a test
problem, and compare proposals, sub-chain lengths, surrogates and stage layouts by
effective samples per exact-model evaluation and by cost. Everything below was produced by
running the code on 2026-09-28. Raw output, drivers, tables and figures:
`toy_examples/tmp_efficiency_study_2026-09-28/` (a `tmp_*` directory on purpose, so later
sessions may read it: `runs/<scheme>/` format-v2 output + `scheme.json`, `timing.json`,
`launch_log.txt`; `metrics.csv`, `posterior.csv`, `directions_P1.npz`, `figures/*.png` with
`figures/captions.md`; `scripts/{problem,run_scheme,launch,batch.sh,analyze,plots}.py`).
The proposal this study started from is kept as §1–§2 (updated to what was actually run).

---

## 0. The GRF solver evaluated every observation in mesh cell 0 (fixed)

`toy_examples/grf_diffusion.py::Solver_diffusion_GRF.get_observations` passed
`cells = np.zeros(...)` ("dummy cell indices") to `Function.eval`; dolfinx evaluates *in the
given cell*, so every observation was the bilinear extrapolation of cell 0's four nodal values.
Verified before the fix (scratch script, 2026-09-28):

| check | result |
|---|---|
| constant field (z = 0), 3×3 points | solver = proper = (1 − x)/2 exactly (the profile is linear, extrapolation is exact) |
| random field (seed 11), 3×3 points | max \|solver − proper\| = 0.045 (the examples' noise sd: 0.03) |
| finite-difference Jacobian at z_true, any number of points | **rank 2** (cell 0 touches the left Dirichlet boundary: two free nodal values) |

**Fix (applied, author-approved):** the colliding cells are computed once in `__init__`
(`geometry.bb_tree` → `compute_collisions_points` → `compute_colliding_cells`) and passed to
`eval`. Check after the fix: solver output = independent evaluation (max \|Δ\| = 0.0), Jacobian
rank 16 with 16 observation points. Consequences: the artificial observations and the posterior
of every GRF example changed; notes 14 and 18 sampled a rank-2 forward map, so their "posterior
≈ prior in 18 of 20 directions" and "adaptive pCN runs to β = 1" were artefacts. Their
*qualitative* findings are re-measured here (§4). Recorded in `10_manual_review_notes.md` §3.

---

## 1. Test problems

### P1 (main): 20 parameters, 4×4 observations, noise sd 0.01

```
Solver_diffusion_GRF(xa=-1, xb=1, ya=-1, yb=1, nx=20, ny=20,
    covariance_type='squared_exponential', length_scale=0.2, sigma=1.0,
    no_parameters=20, observations_per_dim=4,
    positivity_transform_factor=1.0, u_left=1.0, u_right=0.0, source_strength=0.0)
prior        = 20 × NormalComponent(0, 1)          (internal = physical space)
observations = solver.generate_artificial_observations(seed=11)    # 16 values, noise-free
likelihood   = Normal(mean=observations, sd=0.01)
```

Laplace approximation at z_true (finite-difference Jacobian, `directions_P1.npz`): posterior sd
along the 16 observed singular directions 0.032, 0.057, 0.085, 0.106, 0.116, 0.16, 0.18, 0.19,
0.22, 0.25, 0.30, 0.42, 0.54, 0.55, 0.95, 0.99; four directions unconstrained. Coefficient field at
z_true spans 0.34–9.1 (σ = 1 makes exp(·) clearly nonlinear). One solver evaluation costs
**0.20 ms** (20×20 grid), so wall time measures surrogate + MPI overhead; the headline metric is
ESS per exact evaluation and §6 converts it to cost for an expensive model.

**The true posterior is far from the Laplace approximation**: the reference (§3) gives sd 0.149
along v₁ (Laplace 0.032) and 0.125–0.16 along v₂–v₅ (Laplace 0.06–0.12). The posterior is
non-Gaussian, roughly isotropic at sd ≈ 0.14 in the observed directions and sd 1 in the four
unobserved ones — a 7× anisotropy rather than the Laplace 31×.

Candidates considered and rejected (Laplace diagnostics, corrected evaluator): the old note-18
problem with the fixed evaluator (2×2, noise 0.03, σ 0.5, ℓ 0.1) is nearly the prior (all
directions sd > 0.5); 4×4 with σ 0.5 (weakly nonlinear field 0.7–2.0); 4×4 with noise 0.005;
Matérn ν = 1 (similar to P1, rougher).

### P2 (confirmation): 40 parameters, 5×5 observations

As P1 with `no_parameters=40, observations_per_dim=5`, noise sd 0.01. Laplace: 25 observed
directions with sd 0.026–0.88, 15 unconstrained.

---

## 2. Experiment design as executed

Common: `use_solvers_pool=False`, 4 sampler ranks (+ 1 collector where a surrogate is used),
`save_snapshots_to_file=True`, `initial_sample_type="prior"`, `torch_threads=1`,
`min_snapshots_initial=500`, `min_snapshots_to_update=0`, surrogate test set
`TestData.generate(size=512, seed=11)` (512 prior draws), `send_snapshots_to_collector=False`
and `surrogate_model_updates=False` on the final frozen stage. MPICH `mpiexec`, up to 6 runs
concurrently on 32 cores (`OMP/MKL/OPENBLAS_NUM_THREADS=1`). Budgets are exact-model
evaluations **per chain**; they were raised from the proposal after a first Phase A showed
autocorrelation times of 200–300 (40 000 evaluations would have been < 200 τ):

| group | stages | evaluations per chain | ranks |
|---|---|---|---|
| `A_*` stage-1 proposal | 1 × MH | 200 000 | 4 |
| `R1`, `R2` reference | MH `RandomWalk()` adaptive 100 000 → MH `RandomWalk(adaptive=False)` (carried covariance) 400 000 | 500 000 (8 chains) | 8 |
| `B_*`, `C_*`, `D_*`, `E_*`, `F_*` | MH `RandomWalk()` 20 000 → DAMH-SMU 50 000 → DAMH frozen 50 000 | 120 000 | 5 (9 for `D_chains_8`) |
| `G_slow_*` | same layout, 10 000 / 15 000 / 15 000, solver `sleep_time=0.01` | 40 000 | 5 / 4 |

Default surrogate: `NeuralNetworkUpdater(hidden_layer_sizes=(80, 80), silu, adamw, lr 1e-3,
batch_size=None (inferred), replay 1.0 / cap 12 800, clip 2.0, wd 1e-3, seed 0)` — note 18
§3.1's recommendation. Proposal objects: stage 2 adapts (`RandomWalk()`, `PCN()`,
`Hamiltonian(num_steps=L, integrator=...)` with the step carried from stage 1 where the family
matches), stage 3 is the same family with `adaptive=False` (step carried from stage 2). Scheme
definitions: `scripts/run_scheme.py` (`SCHEMES`, `register_followups`, `register_phase_e`).

Metrics per stage, pooled over chains (`scripts/analyze.py`): exact evaluations = accepted +
rejected; states = all proposals (a pre-rejected proposal is a chain state at no solver cost);
τ = emcee integrated autocorrelation time of the chain-averaged autocorrelation function, mean
and max over parameters; **ESS = states / τ; ESS per exact evaluation** (headline); the same
along the first five Jacobian singular directions v₁–v₅; split-R̂ over the 4 chains;
posterior mean/sd per parameter against the reference; surrogate test-set RMSE (prior points)
and posterior-weighted RMSE; per-stage wall time (max over ranks of the library's stage-end
line, measured by `launch.py`). Single-stage runs drop their first 25 % as burn-in; later
stages continue a warmed-up chain and use everything. One run per scheme unless stated;
run-to-run spread is measured in §5 (Phase F).

---

## 3. Reference posterior

`R1` (P1): 8 chains × 400 000 states in the fixed-kernel stage, τ = 202, pooled ESS 15 800,
split-R̂ ≤ 1.001. Mean along v₁ 0.523 (per chain 0.518–0.526), sd 0.149; sd along v₂–v₅
0.125, 0.146, 0.162, 0.137. Its adaptive warm-up stage alone (τ = 212, ESS 2 800) already agrees
to 0.01 sd. Monte-Carlo SE of a reference mean ≈ 0.008 sd, so scheme-vs-reference deviations
below ≈ 0.03 sd are noise.

---

## 4. Results on P1

### 4.1 Phase A — which proposal for the exact-model stage

Figures: `fig_A_ess_per_eval.png`, `fig_adaptation_traces.png`.

Single MH stage, 4 chains × 200 000 evaluations, first 25 % dropped; "ref" = `R1`.

| run | proposal | acceptance | τ | **ESS/eval** | ESS/eval (min over params) | τ along v₁ | max R̂ | max \|mean−ref\|/sd | wall (s) |
|---|---|---|---|---|---|---|---|---|---|
| `A_rw` | `RandomWalk()` adaptive, target 0.234 | 0.236 | 219 | **0.0034** | 0.0027 | 224 | 1.00 | 0.051 | 84 |
| `A_rw_t35` | `RandomWalk(target_rate=0.35)` | 0.352 | 304 | 0.0025 | 0.0019 | 268 | 1.00 | 0.037 | 84 |
| `A_pcn` | `PCN()` adaptive, target 0.234 | 0.233 | 2560 | 0.0003 | 0.0002 | 1490 | 1.03 | 0.122 | 83 |
| `A_pcn_b02` | `PCN(beta=0.2)` fixed | 0.009 | 4562 | 0.0002 | 0.0001 | 3219 | 1.12 | 0.184 | 80 |
| `A_rw_fixed` | `RandomWalk(adaptive=False)` (default step 2.38²/d · I) | 0.0003 | 15550 | 0.00005 | 0.00004 | 10068 | 5.54 | 1.11 | 79 |

- **The adaptive random walk wins by 12×** over adaptive pCN. pCN is isotropic in the prior's
  internal space; with posterior sd 0.14 in 16 directions and 1 in four, its β settles at
  0.068 (Robbins–Monro on the 0.234 target) and every step is tiny in the unconstrained
  directions. The adaptive random walk learns the covariance (σ scale → 0.67 with the
  shrinkage-AM estimator) and moves each direction at its own scale.
- Target 0.35 costs 27 %. The two fixed defaults are unusable on this posterior: the
  library's default random-walk step accepts 0.03 % of proposals and never converges (R̂ 5.5);
  the maintained example's `PCN(beta=0.2)` accepts 0.9 %.
- Stage 1 of every later scheme is therefore `RandomWalk()` (adaptive); its covariance is
  carried into stage 2 where the family matches.
- The opposite of note 18 §2 (there adaptive pCN won 4.3× and ran to β = 0.89): that problem was
  the rank-2 artefact of §0, whose posterior equalled the prior in 18 directions.

### 4.2 Phase B — DAMH proposal × sub-chain length

Figures: `fig_B_ess_per_eval.png`, `fig_B_tau_vs_subchain.png`, `fig_acceptance.png`.

Stage 1 is identical in every run (4 × 20 000 evaluations, ESS/eval 0.0012 with its burn-in).
Stage 2 = DAMH-SMU 50 000, stage 3 = DAMH frozen 50 000, network surrogate (80, 80).
"Speed-up" = stage-3 ESS/evaluation over `A_rw`'s 0.0034. "τ per exact eval" = τ × evaluations /
states (a pre-rejected proposal is a state but costs no evaluation).

| scheme (stages 2–3) | sub-chain | ESS/eval stage 2 | **ESS/eval stage 3** | speed-up | τ per exact eval | accepted | pre-rejected | sub-chain acc. | adapted step (end of stage 2) | test RMSE (prior pts) | wall s2 / s3 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `B_hmc_L30_1` Hamiltonian L = 30, dimension-robust | 1 | 0.549 | **0.604** | **177×** | 1.7 | 0.72 | 0.19 | 0.81 | ε = 0.049 | 0.0162 | 726 / 968 |
| `B_hmc_L10_5` Hamiltonian L = 10, dimension-robust | 5 | 0.248 | **0.260** | 76× | 3.8 | 0.90 | 0.00 | 0.77 | ε = 0.049 | 0.0204 | 1151 / 972 |
| `B_rw_50` random walk | 50 | 0.149 | **0.149** | 44× | 6.7 | 0.31 | 0.65 | 0.01 | σ = 1.84 | 0.0225 | 491 / 1497 |
| `B_rw_20` random walk | 20 | 0.106 | **0.115** | 34× | 8.7 | 0.31 | 0.65 | 0.02 | σ = 1.54 | 0.0241 | 237 / 580 |
| `B_hmc_leap_1` Hamiltonian L = 10, leapfrog | 1 | 0.071 | 0.077 | 22× | 13.1 | 0.68 | 0.24 | 0.76 | ε = 0.050 | 0.0182 | 464 / 410 |
| `B_hmc_L10_1` Hamiltonian L = 10, dimension-robust | 1 | 0.068 | 0.075 | 22× | 13.4 | 0.68 | 0.24 | 0.76 | ε = 0.050 | 0.0167 | 399 / 411 |
| `B_mh_hmc` **MH** with Hamiltonian L = 10 (exact accept/reject) | – | 0.011 | 0.058 | 17× | 17.3 | 0.82 | 0 | – | ε = 0.045 | 0.0227 | 347 / 300 |
| `B_rw_5` random walk | 5 | 0.055 | 0.057 | 17× | 17.4 | 0.30 | 0.66 | 0.08 | σ = 1.10 | 0.0244 | 153 / 253 |
| `B_pcn_50` pCN | 50 | 0.015 | 0.016 | 5× | 63 | 0.30 | 0.66 | 0.01 | β = 0.198 | 0.0159 | 550 / 693 |
| `B_rw_1` random walk | 1 | 0.015 | 0.015 | 4× | 69 | 0.30 | 0.68 | 0.32 | σ = 0.59 | 0.0268 | 95 / 81 |
| `B_pcn_20` pCN | 20 | 0.010 | 0.010 | 3× | 103 | 0.31 | 0.66 | 0.02 | β = 0.162 | 0.0184 | 259 / 255 |
| `B_pcn_5` pCN | 5 | 0.004 | 0.005 | 1.4× | 216 | 0.30 | 0.67 | 0.08 | β = 0.111 | 0.0232 | 149 / 102 |
| `B_pcn_1` pCN | 1 | 0.001 | 0.001 | 0.3× | 853 | 0.29 | 0.68 | 0.32 | β = 0.056 | 0.0281 | 83 / 54 |

Reading:

- **A Hamiltonian proposal on the network surrogate is the most efficient by far**: 30
  dimension-robust leapfrog steps per proposal give τ = 1.7 exact evaluations, i.e. nearly
  independent samples at 72 % acceptance, 177× the best MH. Ten steps are not enough (L = 10:
  22×; the trajectory is too short to decorrelate) and a 5-step sub-chain of L = 10 proposals
  (76×) is worse than one L = 30 proposal at 5/3 of the gradient cost. Plain leapfrog and the
  dimension-robust splitting are indistinguishable at d = 20 (0.077 vs 0.075).
- **Delayed acceptance matters even for HMC**: the same L = 10 Hamiltonian proposal used in an
  *exact* MH stage (`B_mh_hmc`, gradients from the surrogate but every proposal evaluated
  exactly) gives 0.058 against 0.075 with DAMH. Its stage 2 is much worse (0.011): the step
  size adapted there on exact acceptance probabilities while the surrogate was still changing,
  and ended at ε = 0.045 against 0.050 in the DAMH runs — a plausible cause, not verified.
- **Random-walk DAMH: the sub-chain law holds** — τ per evaluation 69 → 17 → 8.7 → 6.7 for
  sub-chain 1 → 5 → 20 → 50, with diminishing returns beyond 20 (the stage-3 wall time triples
  from 20 to 50 while ESS/eval gains 30 %). But the sub-chain itself barely moves: the random
  walk adapts on the *outer* acceptance probability (0.31, kept near 0.234 by pre-rejection), so
  σ inflates to 1.5–1.8 and the sub-chain accepts 1–2 % of its steps; 50 sub-chain steps
  produce about one accepted surrogate move. The 44× comes from that single move being a
  large one, not from the sub-chain exploring the surrogate posterior. An adaptation target
  on the sub-chain acceptance (note 16's α-scoring point) would test the alternative — not
  done here.
- **pCN is the wrong proposal for this posterior** in DAMH as in MH: β ≤ 0.2 and isotropic
  steps make its sub-chain-50 scheme 5× MH at best, sub-chain 1 is *worse* than MH.
- Stage 2 (surrogate retrained) and stage 3 (frozen) give the same efficiency in every
  scheme except the HMC ones (0.55 → 0.60, 0.011 → 0.058 for `B_mh_hmc`), where the step-size
  adaptation and the surrogate both matured during stage 2.
- **Accuracy**: every scheme reproduces `R1` — max over parameters of \|mean − ref\|/sd
  0.013–0.079, of \|sd/ref − 1\| 0.011–0.094 (the worst two are the pCN sub-chain-1/5 runs with
  ESS < 1 000), sd along v₁ 0.146–0.150 (ref 0.149), split-R̂ ≤ 1.05 (`fig_posterior_accuracy.png`,
  `fig_posterior_accuracy_summary.png`, `fig_directions.png`; posterior mean and sd fields of the
  best scheme against the reference and the truth: `fig_posterior_fields.png`).
- **The surrogate is worse than the noise on prior draws** (`fig_surrogate_quality.png`): test
  RMSE 0.016–0.028 against noise sd 0.01 after 280 000 snapshots, in every run. The posterior
  region is where it counts, see the posterior-weighted RMSE in §4.3 and the surrogate
  comparison there. Pre-rejection is nevertheless accurate enough: 65 % of random-walk
  proposals are screened at no cost and the exact acceptance of what passes is 0.31/0.35 = 89 %.
- **Frozen stage slower than the retraining stage** for the random-walk schemes (1497 vs
  491 s at sub-chain 50, 580 vs 237 s at 20) although both do the same surrogate work and the
  frozen stage sends no snapshots — the opposite of note 18 §5's trap. Not explained here;
  §4.4's `D_smu_only` / `D_frozen_only` runs isolate it.

**Additional Phase-B points.** Three runs added after the first Phase-B batch (same layout; `B_hmc_L30_5s` has stages 2–3 of
10 000 evaluations each because the full-budget run was projected at 7 h and killed):

| scheme | sub-chain | **ESS/eval stage 3** | ESS/eval (min over params) | τ per exact eval (mean / max over params) | accepted | pre-rejected | wall s2 / s3 (s) |
|---|---|---|---|---|---|---|---|
| `B_hmc_L50_1` Hamiltonian L = 50 | 1 | **0.867** | 0.332 | 1.43 / 3.7 | 0.73 | 0.19 | 1096 / 1935 |
| `B_hmc_L30_5s` Hamiltonian L = 30 | 5 | **0.722** | 0.652 | 1.39 / 1.5 | 0.89 | 0.00 | 357 / 283 |
| `B_hmc_L30_1` (from above) | 1 | 0.604 | 0.453 | 2.05 / 2.7 | 0.72 | 0.19 | 726 / 968 |
| `B_rw_100` random walk | 100 | **0.180** | 0.151 | 5.5 / 6.6 | 0.32 | 0.65 | 1793 / 2834 |

L = 50 is nominally the best (0.87) but uneven: eleven parameters have τ < 1 (anti-correlated
successive states, τ down to 0.6 — the trajectory overshoots to the far side of the posterior)
while parameters 15–16 have τ = 3.7; the minimum-over-parameters efficiency (0.33) is *below*
L = 30's (0.45). L = 30 with a 5-step sub-chain (0.72, min 0.65, essentially independent states
at 89 % acceptance) is the most uniform sampler of the campaign, at 1.5× the gradient cost of
L = 30 alone. Random walk with sub-chain 100 continues the diminishing-returns curve (0.15 →
0.18 for twice the surrogate work).

### 4.3 Phase C — surrogate choice

Figure: `fig_surrogate_quality.png`.

`C_best_*`: the Hamiltonian L = 30 scheme (needs gradients, therefore networks only);
`C_second_*`: the random-walk sub-chain-20 scheme (any surrogate; chosen over sub-chain 50
because the polynomial/RBF evaluation cost scales with the sub-chain length). RMSE on 512 prior
draws and posterior-weighted RMSE (self-normalised importance weights of the same points), both
in observation units against noise sd 0.01; "retrainings" = `train()` calls on the collector.

| run | scheme | surrogate | **ESS/eval stage 3** | test RMSE (prior) | weighted RMSE (posterior) | retrainings | wall stage 1 / 2 / 3 (s) |
|---|---|---|---|---|---|---|---|
| `C_best_nn128` | HMC L = 30 | network (128, 128, 128), lr 1e-3 | **0.644** | 0.0222 | 0.0154 | 11 193 | 53 / 1367 / 834 |
| `B_hmc_L30_1` | HMC L = 30 | network (80, 80), lr 1e-3 (default) | 0.604 | 0.0162 | 0.0171 | 17 327 | 23 / 726 / 968 |
| `C_best_nn16` | HMC L = 30 | network (16, 16, 16), lr 1e-4, batch 1024 (the old example) | **0.013** | 0.0501 | 0.1015 | 16 903 | 26 / 793 / 723 |
| `C_second_poly3` | RW sub-chain 20 | polynomial degree 3 (1 771 terms) | **0.120** | 0.0288 | 0.0128 | 1 053 | **552 / 8176** / 906 |
| `B_rw_20` | RW sub-chain 20 | network (80, 80) | 0.115 | 0.0241 | 0.0117 | 9 491 | 22 / 237 / 580 |
| `C_second_poly2` | RW sub-chain 20 | polynomial degree 2 (231 terms) | 0.087 | 0.0345 | 0.0435 | 1 388 | 50 / 666 / 722 |
| `C_second_rbf` | RW sub-chain 20 | RBF thin-plate, 50 neighbours | 0.066 (stage 2 only) | – | – | – | **702 / > 13 000** — timed out at 4 h in stage 3 |
| `C_second_nn16` | RW sub-chain 20 | network (16, 16, 16), lr 1e-4 | 0.029 | 0.0569 | 0.0849 | 5 479 | 26 / 135 / 303 |

- **The old example's network is 50× worse than the default one on the same scheme** (0.013 vs
  0.60 with HMC; 0.029 vs 0.115 with the random walk): its posterior-weighted RMSE is 10× the
  noise, the Hamiltonian trajectories follow a wrong energy surface (acceptance 15 %), and the
  step-size adaptation compensates with ε = 0.060 that the exact model then rejects. Note 18
  §3.1's recommendation (wider layers, lr 1e-3, inferred batch) is confirmed on a harder problem.
- The (128, 128, 128) network is not better than (80, 80) at this budget (0.64 vs 0.60, inside the
  ±4 % repeat spread of §4.5) and costs 1.9× per gradient; its prior-set RMSE is *worse* (0.022 vs
  0.016) with half the training passes. Surrogate accuracy is not the ceiling of the HMC scheme
  here — the leapfrog trajectory is (§4.2).
- Polynomial degree 3 equals the network for the random-walk scheme (0.120 vs 0.115; its
  posterior-weighted RMSE 0.013 is the best of all surrogates) but **the collector's refits stall
  the samplers**: 552 s for the 20 000-evaluation warm-up that takes 22 s with the network and
  8 176 s for stage 2. Each refit of 1 771 ridge terms on up to 280 000 snapshots takes seconds,
  `min_snapshots_to_update=0` refits after every batch, and the samplers block once their 100
  pending `isend` requests are full. Degree 2 is 25 % less efficient (its weighted RMSE 0.044 is
  4× the noise: the log-normal field is not quadratic in 20 parameters).
- The RBF (thin-plate spline, 50 neighbours) is the same trap made worse: 702 s of warm-up, a
  stage-2 efficiency of 0.066 while being refitted, and the frozen stage did not finish in the
  remaining 3 h (≈ 1 M local RBF solves per chain). **Run failed (timeout)**; no stage-3 number.
  Both fits would need `min_snapshots_to_update` in the thousands to be usable — not tested.

### 4.4 Phase D — stage layout knobs, on the HMC L = 30 scheme

| run | change against `B_hmc_L30_1` | **ESS/eval last stage** | ESS/eval stage 2 | wall total (s) | note |
|---|---|---|---|---|---|
| `B_hmc_L30_1` | – | 0.604 | 0.549 | 1717 | |
| `D_s1_5k` | warm-up 5 000 instead of 20 000 evaluations | 0.610 | 0.513 | 1373 | first surrogate from 20 000 snapshots |
| `D_s1_60k` | warm-up 60 000 | 0.582 | 0.574 | 1972 | |
| `D_init_2k` | `min_snapshots_initial=2000` | 0.605 | 0.545 | 1391 | |
| `D_smu_only` | one DAMH-SMU stage of 100 000 (never frozen) | 0.592 | – | 1420 | |
| `D_chains_8` | 8 samplers instead of 4 | 0.598 | 0.536 | 2181 | 2× the ESS for 1.27× the wall time |
| `D_frozen_only` | one *frozen* DAMH stage of 100 000 straight after the warm-up | **dead chain** | – | 1136 | 0 accepted of 400 000; see below |
| `D_frozen_only_eps` | same with `Hamiltonian(step_size=0.049, adaptive=False)` | **dead chain** | – | 544 | 0 accepted of 400 000; the step size was not the cause |
| `X_smu_eps` (diagnostic, 10 000 evals) | as `D_frozen_only_eps` but `surrogate_model_updates=True` | 0.489 (67 % accepted, 19 % pre-rejected; τ = 2.5 with the surrogate still training) | – | 69 | works |
| `X_frozen_rw` (diagnostic, 10 000 evals) | frozen DAMH straight after MH with `RandomWalk(adaptive=False)`, sub-chain 20 | **0.0015** (0.5 % accepted, 0 pre-rejected; = plain MH) | – | 21 | |

- **None of the knobs matters at ±4 %** once the scheme is HMC on a trained network: a 5 000-
  evaluation warm-up is enough (the surrogate keeps learning during stage 2 and the HMC step
  size adapts within a few hundred iterations), the first-surrogate threshold is irrelevant, and
  retraining during the whole run neither helps nor hurts (the network converged early,
  `fig_surrogate_quality.png`). Eight chains scale almost linearly (the collector is not the
  bottleneck: 8 samplers × 30 gradients per evaluation are 1.3× slower per chain than 4).
- **`D_frozen_only` exposed a library bug: a frozen DAMH stage that directly follows an MH
  stage samples with the *first* surrogate ever trained, not the latest one.** Evidence: (i)
  `D_frozen_only` and `D_frozen_only_eps` (tuned step 0.049) both accepted 0 of 400 000 with only
  0.3 % pre-rejected — the surrogate let everything through and the exact model rejected
  everything; (ii) the same layout with `surrogate_model_updates=True` (`X_smu_eps`) accepts 67 %
  from the first iterations (0.49 ESS/eval over a 10 000-evaluation stage), and a frozen random-walk stage after MH (`X_frozen_rw`) is nearly
  dead as well (0.5 % accepted, no pre-rejection), so neither the step size nor the proposal is
  the cause; (iii) `raw_data` stores the surrogate prediction next to the exact observation of
  every proposal: RMSE(surrogate − exact) is **0.134 / 0.142** in the two frozen-after-MH runs
  and 0.024 in `X_frozen_rw`, against 0.0006 in `B_hmc_L30_1`'s frozen stage 3 and 0.0004 in
  `B_rw_20`'s — and 0.13 is exactly the test RMSE of the network after its first training on
  ~500 snapshots (`fig_surrogate_quality.png`, leftmost point). Mechanism (from the code, not
  yet pinned by a unit test): the sampler posts one evaluator request at start-up, the collector
  answers it with the first evaluator it trains (`min_snapshots_initial=500`, seconds into the
  MH stage), and that message waits in the buffer; a DAMH stage with
  `surrogate_model_updates=False` calls `get_evaluator()` once in
  `_initialize_current_approximation` and never polls again, so it keeps the stale one. A
  DAMH-SMU stage polls every sub-chain and replaces it within a few iterations, and a frozen
  stage *after* an SMU stage inherits the SMU stage's last evaluator — which is why every
  `B_*` stage 3 is fine. Consequences: `Stage(algorithm="DAMH", surrogate_model_updates=False)`
  right after an MH stage is broken in the current library (the note-18 layout never had it);
  the fix is to drain the request queue / request a fresh evaluator at the start of a
  non-updating DAMH stage. Recorded as a **[bug]** in `10_manual_review_notes.md` §3, not fixed
  in this session. The dead chain also went unnoticed for 20 minutes: a warning when a DAMH
  stage's exact acceptance is exactly zero after ~1 000 iterations is worth having (§7).

### 4.5 Phase F — repeatability

Figure: `fig_F_spread.png`.

`B_hmc_L30_1` and three repeats with shifted library seeds (`F_rep1..3`: 1–3 one-evaluation
excluded stages in front, which change every seed through the stage count, `modules/seeds.py`):
ESS/eval **0.604, 0.651, 0.616, 0.617** (mean 0.622, ±4 %), acceptance 0.724–0.755, sd along v₁
0.1480–0.1487, max \|mean − ref\|/sd 0.014–0.020. Differences below ≈ 10 % between two runs of
this campaign are therefore not significant; the ranking of §4.2 (factors of 2–10 between
families) is.

---

## 5. Confirmation on P2 (40 parameters, 5×5 observations)

Reference `R2`: 8 × 400 000 states, τ = 560, pooled ESS 5 700, split-R̂ 1.004, sd along v₁
0.183 (Laplace 0.026 — again strongly non-Gaussian). Its adaptive random walk achieves
0.0018 ESS/eval (P1: 0.0034).

| run | scheme (network (80, 80)) | **ESS/eval stage 3** | ESS/eval stage 2 | speed-up vs `R2` | τ per exact eval | accepted | pre-rejected | step at end of stage 2 | test RMSE prior / weighted | max \|mean−ref\|/sd | wall s2 / s3 (s) |
|---|---|---|---|---|---|---|---|---|---|---|---|
| `E_hmc30_1_nn80` | HMC L = 30, sub-chain 1 | **0.204** | 0.075 | **113×** | 4.9 | 0.51 | 0.19 | ε = 0.041 | 0.030 / 0.069 | 0.036 | 722 / 704 |
| `E_hmc10_5_nn80` | HMC L = 10, sub-chain 5 | 0.103 | 0.065 | 57× | 9.7 | 0.63 | 0.00 | ε = 0.041 | 0.033 / 0.045 | 0.034 | 1320 / 1793 |
| `E_rw_50_nn80` | random walk, sub-chain 50 | 0.046 | 0.037 | 26× | 22 | 0.27 | 0.58 | σ = 1.55 | 0.041 / 0.060 | 0.058 | 383 / 3042 |

The ranking of P1 holds at twice the dimension, with every scheme 3× less efficient than on P1:
the surrogate is the difference. Its posterior-weighted RMSE is 4.5–7× the noise (P1: 1–2×),
the Hamiltonian acceptance falls to 51 % (P1: 72 %) and the stage-2 numbers are well below
stage 3 in all three runs (the network was still improving when stage 2 ended,
`fig_surrogate_quality.png`). A d = 40 problem wants a longer stage 2 or a larger/longer-trained
network before freezing; not tested within the budget. Accuracy against `R2` is unaffected
(max mean error 0.06 sd, sd along v₁ 0.185–0.186 vs 0.183).

---

## 6. Cost: what transfers to an expensive model

Figures: `fig_cost_model.png`, `fig_wall_time.png`.

Wall times in this campaign are surrogate + MPI + Python overhead (the solver costs 0.2 ms) and
were measured with up to six 5-rank runs sharing 32 cores, so they are upper bounds by perhaps
1.5×. The cost model used here is deliberately simple: *cost per exact evaluation = t_solver +
overhead*, where overhead is a stage's measured wall time divided by its exact evaluations per
chain and therefore already contains every surrogate call, gradient and message of that stage;
ESS per unit cost = (ESS per exact evaluation) / cost. Break-even = the solver cost above which a
scheme beats `A_rw` per second of wall time.

| scheme | ESS/eval | overhead per exact eval (ms) | break-even solver cost | ESS/s measured (0.2 ms solver, 4 chains) | ESS/s at a 10 ms solver (model) | ESS/s at a 1 s solver (model) |
|---|---|---|---|---|---|---|
| `A_rw` plain MH | 0.0034 | 0.4 | – | 33 | 1.3 | 0.014 |
| `B_rw_5` | 0.057 | 5.1 | always | 45 | 15 | 0.23 |
| `B_rw_20` | 0.115 | 11.6 | always | 40 | 21 | 0.46 |
| `B_rw_50` | 0.149 | 30 | 0.3 ms | 20 | 15 | 0.58 |
| `B_rw_100` | 0.180 | 57 | 0.7 ms | 13 | 11 | 0.68 |
| `B_hmc_L10_5` | 0.260 | 19 | always | 54 | 35 | 1.02 |
| `B_hmc_L30_1` | 0.604 | 19 | always | **125** | **82** | 2.37 |
| `B_hmc_L30_5s` | 0.722 | 28 | always | 102 | 75 | 2.81 |
| `B_hmc_L50_1` | 0.867 | 39 | always | 90 | 71 | **3.34** |
| `C_best_nn128` | 0.644 | 17 | always | 154 | 97 | 2.53 |
| `C_second_poly3` | 0.120 | 18 | 0.1 ms | 26 | 17 | 0.47 |
| `E_hmc30_1_nn80` (P2) | 0.204 | 14 | always | 58 | 34 | 0.81 |

- **Even on the 0.2 ms model the HMC-DAMH schemes win by wall clock** (125 vs 33 ESS/s): the
  Hamiltonian's 30 network gradients cost 19 ms per exact evaluation but buy a 177× better
  sample. Random-walk sub-chains ≥ 50 and the polynomial need a solver of at least ≈ 0.3–0.7 ms
  to pay off; at a 10 ms solver all DAMH schemes are ahead, and from ≈ 1 s on the ranking is
  simply the ESS-per-evaluation ranking of §4.2.
- **Check with a slow solver (`G_*`, `sleep_time=0.01`, 10 ms per evaluation, reduced budgets):**
  `G_slow_best` (HMC L = 30, stages 10 000 / 15 000 / 15 000) gives 0.570 ESS/eval at 21.3 ms
  wall per evaluation (10 ms solver + 11 ms overhead, as the model predicts from the fast run's
  19 ms), i.e. **107 ESS/s**; `G_slow_mh` (plain adaptive MH, 40 000 evaluations) gives 0.0023
  ESS/eval at 11.2 ms per evaluation, **0.8 ESS/s**. Measured speed-up 130× against the model's
  82/1.3 = 63× — the MH run is short (τ = 330 over 40 000 evaluations) and its ESS estimate is
  the less reliable of the two. The model is right within a factor of 2 in the region where it
  matters.
- Per-stage wall times (`timing.json`, `fig_wall_time.png`): the frozen stage of the random-walk
  schemes is consistently 2–3× slower than the retraining stage that does the same surrogate
  work (`B_rw_20` 237 → 580 s, `B_rw_50` 491 → 1497 s, `B_rw_100` 1793 → 2834 s, `E_rw_50` 383 →
  3042 s), while the HMC schemes show no such gap (726 → 968, 1096 → 1935 for L = 50 only). The
  frozen stage sends no snapshots and receives no evaluators; the remaining difference between
  the two is that the collector sits idle in a receive loop. Not explained by this campaign;
  worth a profile (§7).

---

**Check with a 1 s solver (added 2026-09-30, `H_1s_*`, `sleep_time=1.0`, 4 chains, reduced budgets).**
`H_1s_mh` (plain adaptive MH, 4 000 evaluations per chain, 4 005 s): ESS 40.9 (τ = 294, first 25 %
dropped), **0.010 ESS/s**; its posterior means are still up to 1.4 sd from the reference — the run is
far from converged. `H_1s_best` (HMC L = 30, sub-chain 1, stages 3 000 / 1 500 / 1 500 per chain):
frozen stage ESS 3 134 in 1 520 s, 0.52 ESS/eval, **2.1 ESS/s**, max |mean − ref|/sd 0.035; the
surrogate overhead is 13 ms per exact evaluation (1.013 s wall per evaluation). The §6 model
predicted 0.014 and 2.37 ESS/s; measured ratio 200× against the model's 170×. MH's ESS estimate
rests on ~10 τ per chain and is the less reliable of the two. Rows are in `metrics.csv`.

## 7. Recommendations and follow-ups

For this problem class (smooth but nonlinear model, informative posterior, tens of parameters):

1. **Exact-model stage: adaptive random walk** (`RandomWalk()`), never a fixed step and not pCN.
   On a posterior that differs from the prior, pCN's isotropic step is the wrong shape; the
   random walk learns the covariance. (The reverse of note 18, whose problem was the rank-2
   artefact of §0.)
2. **Surrogate stage: DAMH with a Hamiltonian proposal on the network surrogate**, 30
   dimension-robust leapfrog steps, adaptive step size (target 0.8), sub-chain 1 — 0.60 ESS per
   exact evaluation, 177× the best MH, 125 ESS/s even at a 0.2 ms model; a 5-step sub-chain of
   the same proposal is the most uniform sampler (0.72, min-over-parameters 0.65) at 1.5× the
   gradient cost. Fifty leapfrog steps look better on average (0.87) but mix unevenly. Ten steps
   are too few (0.075).
3. **If gradients are not available** (polynomial, RBF, or a black-box surrogate): random walk
   with a sub-chain of 20–50 (0.12–0.15, 34–44×); longer sub-chains give diminishing returns and
   the random walk's adaptation should then target the *sub-chain* acceptance, which today is
   1–2 % because the outer, pre-screened acceptance is what it sees. pCN sub-chains are 5× at
   best.
4. **Surrogate:** the network with (80, 80) hidden units, lr 1e-3, inferred batch size,
   `min_snapshots_to_update=0` (the current defaults); the previous (16, 16, 16)/lr 1e-4 network
   is 50× worse on the same scheme. Polynomial degree 3 matches the network's efficiency but
   refitting it after every batch stalls the samplers; use it only with a large
   `min_snapshots_to_update`. RBF with 50 neighbours is impractical at this snapshot count.
5. **Stage layout hardly matters** once 1–2 hold: a 5 000-evaluation warm-up, any first-surrogate
   threshold, retraining throughout or freezing after 50 000 — all within ±4 %. Do **not** freeze straight after the MH warm-up until the stale-evaluator
   bug of §4.4 is fixed: put at least a short DAMH-SMU stage in between.
6. **Budget by ESS per exact evaluation**, converting with the measured per-evaluation overhead
   (≈ 19 ms for 30 network gradients here): the DAMH-HMC scheme is ahead of MH at any solver cost,
   random-walk sub-chains from ≈ 0.5 ms per evaluation on.
7. At d = 40 the ranking holds but everything is 3× slower per evaluation because the surrogate is
   3–4× less accurate relative to the noise; the lever there is surrogate capacity/training time,
   untested.

Library follow-ups suggested by the campaign (none implemented; recorded for the manual-review
list): (0) **[bug]** a non-updating DAMH stage after an MH stage installs the stale first
evaluator (§4.4); (a) a warning when a DAMH stage's exact acceptance stays at zero for many
iterations (§4.4); (b) an option for the adaptive random walk / pCN in a DAMH sub-chain to adapt on the
sub-chain acceptance (§4.2, note 16); (c) the frozen-stage slowdown of random-walk DAMH (§6); (d)
a start-up note when a polynomial/RBF updater is combined with `min_snapshots_to_update=0` and
more than ~10⁴ snapshots (§4.3).

**Follow-up study (2026-09-29):** `21_pcn_investigation_2026-09-29.md` checks whether pCN's poor
showing is an implementation or tuning problem — it is neither (isotropic proposal on a 12×
anisotropic posterior; auto-tuners within 20 % of the fixed optima) — and adds cost per
effective sample with surrogate values and gradients priced.

Not checked: repetition of any scheme other than the winner; sub-chain > 5 for Hamiltonian
proposals; HMC targets other than 0.8 or a mass matrix from the adapted covariance; longer
stage 2 or wider networks on P2; the effect of running fewer jobs per node on the wall times.

## 8. Files

`toy_examples/tmp_efficiency_study_2026-09-28/`: `scripts/problem.py` (P1/P2),
`scripts/run_scheme.py` (`SCHEMES`), `scripts/launch.py` (timestamped log, timeout, per-stage
wall times → `timing.json`), `scripts/batch.sh`, `scripts/batch2.sh`, `scripts/analyze.py` →
`metrics.csv` (one row per run and stage, 115 rows), `posterior.csv`, `directions_P1.npz`,
`directions_P2.npz`; `scripts/plots.py` → `figures/*.png` + `figures/captions.md`;
`scripts/make_html.py` → `report.html` (this note with the figures embedded); `runs/<scheme>/`
(46 runs incl. two `X_*` diagnostics, format v2 + `scheme.json`, `timing.json`, `launch_log.txt`; `C_second_rbf` incomplete;
`B_hmc_L30_5` killed and removed). Library changes: `toy_examples/grf_diffusion.py` (§0) only.
Total: 46 runs, 29 M exact evaluations (3.2 M in each reference), 5 h 5 min of wall clock
(2026-09-28 20:29 → 2026-09-29 01:35 UTC, up to 6 runs at a time).
