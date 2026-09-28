# 18 — Sampling schemes on the GRF-diffusion problem (2026-09-22)

**Dated record of one comparison campaign**, requested by the author: same problem and
therefore the same posterior in every run, only the sampling scheme differs. Every scheme
has three stages — a short adaptive MH stage, DAMH with the surrogate retrained (DAMH-SMU),
DAMH with the surrogate frozen — and the stage-1 proposal was chosen from a preliminary
single-stage comparison. Raw output, drivers and figures:
`toy_examples/out_scheme_study_2026-09-22/` (4.3 GB, dominated by `raw_data/`; an `out_*`
directory, off-limits for future sessions unless the author asks). Everything below was
produced by running the code; the two library changes made during the campaign are listed
in §7 and in `CHANGELOG.md`.

---

## 1. Setup

### 1.1 Problem (identical in every run)

Exactly `14_grf_validation_2026-09-17.md` §1.1: `Solver_diffusion_GRF(nx=20, ny=20,
covariance_type='squared_exponential', length_scale=0.1, sigma=0.5, nu=1.0,
no_parameters=20, observations_per_dim=2, u_left=1, u_right=0, source_strength=0)`; prior
= 20 independent standard normals (`PriorIndependentComponents`, internal = physical
space); observations = `solver.generate_artificial_observations(seed=11)` (4 values);
likelihood `Normal(mean=observations, sd=0.03)`. One solver evaluation costs ≈ 0.2 ms
(measured again: 1 ms first call, then below the timer resolution), so **wall-clock time
measures surrogate and MPI overhead, not the forward model**; the metric that transfers to an
expensive model is effective sample size per exact-model evaluation (§1.4).

The posterior is close to the prior in 18 of 20 directions (Jacobian at z = 0 has singular
values 0.152, 0.024, ~0, ~0; noise sd 0.03): only the leading right singular vector v₁ is
constrained (posterior sd 0.21 vs prior 1), v₂ weakly (sd 0.78). Accuracy is therefore
judged along v₁/v₂ and by per-parameter deviations, as in note 14.

### 1.2 Runs

All runs: `use_solvers_pool=False` (solver in-process on every sampler), 3 sampler ranks
(+1 collector for the three-stage schemes), `save_snapshots_to_file=True`,
`min_snapshots_initial=2000`, `min_snapshots_to_update=2000`, `torch_threads=1`,
`initial_sample_type="prior"` (library seeds), surrogate test set
`TestData.generate(size=256, seed=11)`. Budgets are exact-model evaluations **per chain**:

| group | stages | evaluations per chain | ranks |
|---|---|---|---|
| `P_*` preliminary | 1 × MH | 40 000 | 3 |
| `R_ref` reference | 1 × MH, `PCN(beta=0.2)` | 400 000 (6 chains) | 6 |
| `S_pcn_*` schemes | MH → DAMH-SMU → DAMH | 5 000 / 20 000 / 20 000 | 4 |

Stage-1 candidates (`P_*`): adaptive random walk (`RandomWalk()`, target 0.234), adaptive
random walk with target 0.4, adaptive pCN (`PCN()`, target 0.234), fixed pCN β = 0.2 (the
maintained example's choice). Three-stage schemes: stage 1 = the preliminary winner;
stages 2–3 either the same proposal family (adaptive in stage 2, frozen carry-over in
stage 3) with sub-chain length 1 or 5, or `Hamiltonian(step_size=0.05, num_steps=100,
integrator="dimension_robust")` in both (only on the network surrogate, the only one with
gradients). Surrogates: `PolynomialSklearnUpdater(max_degree=2)`,
`RBFInterpolationUpdater()` (thin-plate spline, `max_neighbors=50`),
`KDTreeUpdater(no_nearest_neighbors=8)`, `NeuralNetworkUpdater((16,16,16), silu, adamw,
lr 1e-4, batch 1024, replay 1.0/4096, clip 2.0, wd 1e-3, seed 0)` — the example's network.
Two extra runs (`*_nosend`) repeat the RBF and k-d-tree 1-step schemes with
`send_snapshots_to_collector=False` in the frozen stage (§5). One extra run
(`S_pcn_nn_same_sub5_upd0`, added 2026-09-23 at the author's request) repeats the network 5-step
scheme with `min_snapshots_to_update=0` -- the collector retrains on every loop instead of after
every 2 000 new snapshots -- and nothing else changed (§3.1).

Commands (from `toy_examples/`; `launch.py` timestamps stdout and records per-stage wall
times in `runs/<scheme>/timing.json`):

```
/dolfinx-env/bin/python3 out_scheme_study_2026-09-22/scripts/launch.py <scheme> <ranks> [timeout_s]
/dolfinx-env/bin/python3 out_scheme_study_2026-09-22/scripts/analyze.py   # -> metrics.csv, posterior.csv
/dolfinx-env/bin/python3 out_scheme_study_2026-09-22/scripts/plots.py     # -> figures/*.png
```

Scheme definitions: `scripts/run_scheme.py` (`SCHEMES`). 20 runs, 4.75 M exact evaluations
in total, 1 h 59 min of sampling wall time (runs were executed 4–5 at a time on 32 cores).

### 1.3 Reference posterior

`R_ref`: 6 chains × 400 001 states, autocorrelation time 141.2 (note 14's R-A: 140.6),
pooled ESS 17 018 (SE of a pooled mean ≈ 0.008 prior sd). Along v₁: mean 0.3772, sd
0.2116 (R-A: 0.3778, 0.2116); v₂ sd 0.7777 (R-A: 0.7769).

### 1.4 Metrics

Per stage, pooled over the 3 chains, from `read_run` / `post_processing.Samples`:
- **exact evaluations** = accepted + rejected proposals (a pre-rejected proposal never
  reaches the solver);
- **ESS** = states / integrated autocorrelation time (`emcee`, mean over the 20
  parameters); **ESS per exact evaluation** is the headline number; ESS per wall-clock
  second is reported with the caveat of §1.1;
- acceptance / rejection / pre-rejection fractions of all proposals;
- surrogate test-set RMSE (256 held-out prior points, in observation units; noise sd 0.03);
- posterior mean/sd per parameter and along v₁, v₂, compared with `R_ref`.

One run per scheme, no repetition: differences below ≈ 10 % in ESS are within noise
(DAMH-SMU retrain timing is not even deterministic, see note 10 §2.31).

---

## 2. Preliminary: which proposal for stage 1

`figures/fig_prelim_ess_per_eval.png`

| run | proposal | acceptance | autocorr. time | ESS per exact evaluation |
|---|---|---|---|---|
| `P_pcn` | adaptive pCN, target 0.234 | 0.239 | **15.4** | **0.0652** |
| `P_rw` | adaptive random walk, target 0.234 | 0.234 | 67.0 | 0.0150 |
| `P_rw_t40` | adaptive random walk, target 0.4 | 0.399 | 78.9 | 0.0128 |
| `P_pcn_b02` | pCN β = 0.2 fixed (the example) | 0.690 | 132.3 | 0.0076 |

Adaptive pCN is 4.3× the adaptive random walk and 8.6× the example's fixed pCN. The
adaptation drove β to **0.89** (carry-over file): with a posterior that equals the prior in
18 directions, near-independence proposals are best, and β = 0.2 is far too cautious. All
four agree with the reference along v₁ (mean 0.373–0.378, sd 0.208–0.212).
**Stage 1 of every scheme = `PCN()` (adaptive).**

---

## 3. Three-stage schemes: efficiency

`figures/fig_ess_per_eval.png`, `figures/fig_acceptance.png`. Stage 1 is the same in every
scheme (15 000 evaluations, 0.062 ESS/evaluation, β → 0.86); the table shows stages 2 and 3.
"Speed-up" = stage-3 ESS/evaluation over the MH-only 0.0652.

| scheme (stage 2–3 proposal, sub-chain) | surrogate | ESS/eval stage 2 | ESS/eval stage 3 | speed-up vs MH | pre-rejected (stage 3) | accepted | autocorr. (stage 3) | surrogate RMSE |
|---|---|---|---|---|---|---|---|---|
| pCN, sub-chain 5 | polynomial | 0.701 | **0.691** | **10.6×** | 35 % | 64 % | 2.2 | 0.0013 |
| pCN, sub-chain 5, `min_snapshots_to_update=0` | network | 0.654 | **0.680** | **10.4×** | 35 % | 63 % | 2.2 | 0.0012 |
| … + hidden layers (80, 80) | network | 0.657 | 0.705 | 10.8× | 35 % | 65 % | 2.2 | 0.00036 |
| … + learning rate 1e-3 | network | 0.703 | **0.712** | **10.9×** | 35 % | 65 % | 2.2 | 0.00024 |
| … + batch 256 (inferred), replay cap 12 800 | network | 0.703 | 0.711 | 10.9× | 35 % | 65 % | 2.2 | 0.00041 |
| pCN, sub-chain 5 | RBF | 0.573 | 0.574 | 8.8× | 34 % | 57 % | 2.6 | 0.0052 |
| pCN, sub-chain 5 | network | 0.398 | 0.541 | 8.3× | 34 % | 56 % | 2.8 | 0.0039 |
| Hamiltonian (0.05 × 100) | network | 0.274 | 0.312 | 4.8× | 0.5 % | 86 % | 5.2 | 0.0039 |
| pCN, sub-chain 5 | k-d tree | 0.146 | 0.170 | 2.6× | 6 % | 33 % | 6.3 | 0.050 |
| pCN, sub-chain 1 | polynomial | 0.162 | 0.161 | 2.5× | 71 % | 28 % | 21.4 | 0.0013 |
| pCN, sub-chain 1 | RBF | 0.134 | 0.142 | 2.2× | 70 % | 26 % | 23.7 | 0.0055 |
| pCN, sub-chain 1 | network | 0.112 | 0.136 | 2.1× | 70 % | 25 % | 24.6 | 0.0039 |
| pCN, sub-chain 1 | k-d tree | 0.058 | 0.061 | 0.9× | 41 % | 22 % | 28.1 | 0.050 |

Reading:

- **Sub-chain length is the dominant factor.** With one surrogate step per exact
  evaluation, DAMH gains 2.1–2.5× over MH purely from pre-rejection (70 % of proposals are
  screened out for free). Five surrogate steps between exact evaluations turn that into
  8–11×: the autocorrelation time per exact evaluation drops from ≈ 22 to ≈ 2.2–2.8, i.e.
  by about the sub-chain length, which is what the DAMH construction predicts and what
  note 14 §5 measured for β = 0.2 (28.6 → 4.9× from a 5-step sub-chain).
- **Surrogate accuracy sets the ceiling.** Polynomial (degree 2, RMSE 0.0013 — 25× below
  the noise sd), RBF (0.005) and network (0.004) are all far more accurate than the noise
  and give similar efficiency; the k-d tree (RMSE 0.050, *worse than the noise*) pre-rejects
  the wrong proposals: with sub-chain 1 it is no better than plain MH, with sub-chain 5 it
  reaches only 2.6×. Below the noise level, more surrogate accuracy buys little (polynomial
  vs network: 0.69 vs 0.54, partly the network's 2× larger error, partly noise).
- **Adaptive pCN in the DAMH sub-chain runs to β = 1.0** (carry-over of stage 2 in every
  5-step scheme: 0.99999…), i.e. the sub-chain proposes *independent prior draws* and the
  surrogate does all the filtering. The target acceptance 0.234 is unreachable from above
  (acceptance stays 0.56–0.64 even at β = 1), so the Robbins–Monro recursion saturates at
  the boundary. This is the correct behaviour for a posterior that nearly equals the prior,
  but it means "adaptive pCN in DAMH" degenerates into an independence sampler here; on a
  more informative problem β would settle inside (0, 1). The 1-step schemes settled at
  β ≈ 0.74–0.86.
- **Hamiltonian on the network surrogate is good but not the best**: 4.8× MH, with
  almost no pre-rejection (0.5 %) and 86 % acceptance — the leapfrog trajectories land in
  high-posterior regions — but its autocorrelation per exact evaluation (5.2) is twice the
  5-step pCN sub-chain's, and it costs 100 surrogate gradient evaluations per exact one
  (13 min of wall time vs 42 s for pCN sub-chain 5 on the same network). Note 14 reported
  38× for a Hamiltonian DAMH-SMU stage; that run had β = 0.2 pCN (0.0076 ESS/eval) as its
  baseline, i.e. the *same* absolute efficiency class (0.25–0.31 ESS/eval) measured against
  a much weaker MH — the two campaigns agree.
- DAMH-SMU (stage 2) and frozen DAMH (stage 3) give the same efficiency for every scheme
  except the network 5-step one (0.40 → 0.54), whose surrogate was still improving during
  stage 2 (RMSE 0.0075 at 15 000 snapshots → 0.0039 at the end, `fig_surrogate_quality.png`).

### 3.1 The network was under-trained, not worse (added 2026-09-23)

The author asked why the polynomial beat the network. `NeuralNetworkUpdater.iterations_batch=100`
is the number of optimizer steps per `train()` call, and the collector calls `train()` only
after `min_snapshots_to_update` new snapshots: with the study's 2 000 the network received 67
retrainings, i.e. about 6 700 optimizer steps at learning rate 1e-4 in the whole run, and its
test RMSE was still falling at the end. `S_pcn_nn_same_sub5_upd0` sets `min_snapshots_to_update=0`
(retrain on every collector loop, otherwise identical): **646 retrainings, final RMSE 0.0012**
(was 0.0039; the polynomial has 0.0013), stage-3 ESS/evaluation **0.680** (was 0.541; polynomial
0.691), pre-rejection and acceptance now equal to the polynomial's (35 % / 63 %), wall time
unchanged at 15 s per DAMH stage -- the training happens on the otherwise idle collector rank.
It is the best scheme of the study by wall clock (2 680 ESS/s). The frozen stage 3 also improved
over its own stage 2 less than before (0.654 → 0.680), because the surrogate was already
converged when stage 2 started. So the difference in §3 was a training-budget artefact of
`min_snapshots_to_update`, not a property of the surrogate class -- which is what the author
suspected when asking for `min_snapshots_to_update` to default to 0 for the network (note 10 §8a).

**Three cumulative setting changes on top of `upd0`** (author-requested 2026-09-23; the session's
recommendations, everything else identical, one run each):

| variant | final test RMSE | RMSE / noise sd | ESS/eval stage 3 | wall per DAMH stage |
|---|---|---|---|---|
| `upd0` (16,16,16), lr 1e-4, batch 1024, replay cap 4 096 | 0.00119 | 0.040 | 0.680 | 15 s |
| `upd0_arch`: hidden layers (80, 80) | 0.00036 | 0.012 | 0.705 | 27 s |
| `upd0_lr`: + learning rate 1e-3 | 0.00024 | 0.008 | 0.712 | 27 s |
| `upd0_batch`: + batch size inferred (256), replay cap 12 800 | 0.00041 | 0.014 | 0.711 | 15 s |
| polynomial, degree 2 (for comparison) | 0.00126 | 0.042 | 0.691 | 45 s |

Each change improved the surrogate as predicted -- the wider network cuts the error 3.3×, the higher
learning rate another 1.5× -- but **the sampling efficiency is saturated**: 0.68 → 0.71 ESS per
evaluation is within the ≈ 10 % run-to-run noise of this campaign, and equals the polynomial's
0.69–0.70. Once the surrogate error is a few percent of the noise sd, DAMH with a 5-step pCN
sub-chain at β = 1 has reached its ceiling (autocorrelation 2.15–2.2 per exact evaluation) and
further accuracy buys nothing; the lever from here is a longer sub-chain, not a better surrogate.
The batching change is still worth having: the 80-wide network doubled the wall time of a DAMH
stage (27 s) because every surrogate call is more expensive, and the inferred batch of 256 makes
each optimizer step cheap enough to bring it back to 15 s at the same efficiency -- the best
scheme of the study by wall clock (2 890 ESS/s) with a surrogate 3× more accurate than `upd0`.
Posterior unchanged in all three (v₁ sd 0.210–0.212, v₂ sd 0.772–0.778).

---

## 4. Accuracy: the same posterior everywhere

`figures/fig_posterior_accuracy.png`. Stage 3 of every scheme against `R_ref`:

| quantity | range over the 11 three-stage runs | reference / expectation |
|---|---|---|
| posterior mean along v₁ | 0.375 – 0.380 | 0.3772 (SE ≈ 0.003–0.005 per run) |
| posterior sd along v₁ | 0.2084 – 0.2123 | 0.2116 |
| posterior sd along v₂ | 0.757 – 0.801 | 0.7777 |
| max over 20 parameters of \|mean − ref mean\| / ref sd | 0.011 – 0.043 | Monte-Carlo SE 0.005–0.017 per run |
| max over 20 parameters of \|sd / ref sd − 1\| | 0.010 – 0.027 | — |

No scheme is biased at the resolution of this campaign, including the k-d tree schemes
whose surrogate error exceeds the noise: delayed acceptance corrects the surrogate exactly,
a poor surrogate only costs efficiency. The four preliminary MH runs deviate more
(0.03–0.065) because their ESS is smaller.

---

## 5. Wall-clock time, and a trap in the frozen stage

`figures/fig_ess_per_second.png`, `figures/fig_wall_time.png`. With a 0.2 ms model, wall
time is surrogate cost: network 5-step 42 s (1 850 ESS/s; 2 680 ESS/s with
`min_snapshots_to_update=0`, §3.1), polynomial 5-step 94 s (920 ESS/s), RBF 5-step 1 594 s (36 ESS/s), k-d tree 5-step 1 014 s (15 ESS/s),
Hamiltonian 804 s (47 ESS/s). Only the ESS-per-evaluation column transfers to an expensive
solver; the ranking by ESS/s is specific to a millisecond model.

**The frozen stage was 2–3× slower than the retraining stage for RBF and k-d tree**
(831 vs 397 s, 555 vs 196 s) although both run the same 60 000 exact evaluations. Cause:
samplers keep streaming snapshots (`send_snapshots_to_collector=True` by default) and the
collector keeps refitting on every 2 000 new ones — 64–67 refits per run, up to 135 000
snapshots — a surrogate that no stage will ever pick up; a refit of an RBF or k-d tree on
100 k points takes seconds, and the samplers stall on their snapshot queue meanwhile. The
`*_nosend` reruns confirm it: frozen-stage time 831 → 386 s (RBF) and 555 → 243 s (k-d
tree), with ESS/evaluation unchanged (0.142 vs 0.141; 0.061 vs 0.059). The library now
prints a start-up note for this layout and the `Stage` docstring says when to switch the
flag off (§7); it does **not** drop the snapshots by itself, because a later run may want
them for `SurrogateRestart(mode="data")`.

---

## 6. Recommendations for this problem class (smooth model, posterior ≈ prior in most directions)

1. Stage 1: **adaptive pCN**, not a fixed β and not a random walk; let it run to its own β.
2. Stages 2–3: **pCN with a 5-step sub-chain** (or longer — untested) on the polynomial
   or network surrogate: ≈ 10× the effective samples per model evaluation of the best MH.
   Try longer sub-chains before trying Hamiltonian proposals.
3. Surrogate: any of polynomial (degree 2), RBF, network is accurate enough; choose by
   cost — polynomial for few parameters and a smooth model, network when gradients are
   wanted or the model is not polynomial-like. For the network: `min_snapshots_to_update=0`
   (now the default), two hidden layers of `clip(4·no_parameters, 32, 256)` units, learning
   rate 1e-3, batch size left to the updater (256) and a replay cap of a few times the steps
   × batch (§3.1). Do not use the k-d tree at this dimension.
4. Set `send_snapshots_to_collector=False` on the frozen final stage unless the snapshots
   are wanted for a restart.
5. Judge schemes by ESS per exact evaluation; compare wall time only for models of the
   same cost class.

Untested here (candidates for a follow-up): sub-chain lengths 10–50; a network-gradient
Hamiltonian with fewer leapfrog steps and on the fully trained network of §3.1; adaptive
Hamiltonian step size; more than 3 chains.

---

## 7. Bugs and traps found, and what was done

- **Deadlock (fixed).** The smoke run of the driver (300 evaluations in stage 1,
  `min_snapshots_initial=2000`, then DAMH) hung: every sampler blocked waiting for its
  first surrogate, the collector waiting for snapshots only those samplers could send. New
  protocol message `TAG_EVALUATOR_NEEDED`; the collector trains on what it has once every
  active sampler is blocked, with a printed warning; `run_local` does the same. Regression
  test `tests/mpi/test_mpi_hangs.py::test_i3b_...`. Details: note 10 §2.38, CHANGELOG.
- **Useless retraining in a frozen tail (trap, made visible).** `stages.wasted_snapshot_notes`
  prints a `[note]` at start-up; `Stage.send_snapshots_to_collector` documents the case.
  Quantified in §5.
- **Adaptive pCN saturating at β = 1** in a DAMH sub-chain (observation, not a bug): the
  target acceptance cannot be reached from above. Worth a printed note from the proposal
  when β pins to a boundary for many periods — not implemented.
- The launcher's first version could not kill a silent hang (the reader blocked on a quiet
  pipe); rewritten with a reader thread and a process-group kill.

## 8. Files

`toy_examples/out_scheme_study_2026-09-22/`: `scripts/{run_scheme,launch,analyze,plots}.py`,
`scripts/batch_*.sh`, `runs/<scheme>/` (format-v2 output, `timing.json`, `launch_log.txt`,
`scheme.json`), `metrics.csv`, `posterior.csv`, `v1v2.npy`, `figures/*.png`,
`figures/captions.md`. Library changes: `surrDAMH/modules/communication.py`,
`algorithm_interfaces*.py`, `algorithms.py`, `process_COLLECTOR.py`, `stages.py`, `core.py`,
`tests/mpi/test_mpi_hangs.py`, `tests/unit/test_config_stages.py`.

Not checked: repetition of any scheme with other seeds; sub-chain lengths above 5; the
Hamiltonian step-size/leapfrog trade-off; behaviour with an expensive (seconds) model,
where the collector's refit time would no longer dominate.
