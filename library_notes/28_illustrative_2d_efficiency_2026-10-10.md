# 28 — Efficiency study on the 2-D illustrative example (2026-10-10)

**Dated record of one campaign.** Everything below was produced by running the code on 2026-10-10 (working tree
of `working_Kuba` with the uncommitted automatic mode, S0, S1, S2, S3, S6; the package was not modified). Raw runs,
drivers, tables and figures: `toy_examples/tmp_illustrative_2d_study_2026-10-10/` (`runs/<name>/` format-v2 output +
`study_case.json` + `metrics.json`, `runs/<name>.log`, `runs/<name>.analyze.log`, `runs/batch_*.out`;
`ref_quadrature.{json,npz}`; `tables.md`, `table_all.csv`; `figs/*.png`; `scripts/{configs,candidates,run_case,analyze,
tables,ref_quadrature,ref_saddle,fig_traces,peek}.py`, `scripts/batch*.sh`). Untracked, ~300 MB. 116 runs in total.

**Read first (limits of what was measured):**

* **The posterior is bimodal with a deep barrier**: two local maxima, (−1.81, 0.11) and (1.39, −0.80), separated
  by a saddle at 1.4·10⁻⁵ of the maximum density (≈ 11 nats); the narrow left arm holds 30 % of the mass.
  At 20 000 evaluations every sampler sees only a few to a few hundred arm-to-arm transitions per chain, so
  **the accuracy of a run is dominated by how often it crosses between the arms**, and single runs are very noisy
  (the round-1 ranking is not reliable on its own; round 2 used 5 repetitions instead of the brief's 3 because time allowed).
* **The ESS estimator is not trustworthy when a chain does not cross**: several runs report ESS in the thousands
  with R̂ ≈ 4 (each chain stuck in a different arm; e.g. `eps0.1`: ESS 4718, R̂ 3.85, mean off by 26 SE).
  ESS/eval is only compared among runs that passed the correctness check, and the medians over repetitions are
  reported.
* **The forward model costs 1.1 µs, not 0.2 ms** (measured: 20 000 calls of `Solver_illustrative_local` in a loop;
  the 0.2 ms of the brief is the GRF model of note 27). An MH run of 20 000 evaluations takes 0.23 s; **ESS/s
  measures library overhead only** (network training, Hamiltonian trajectories on network gradients, MPI).
* **The brief's "result poor" rule is strict for this budget**: plain adaptive MH with 20 000 evaluations fails it in
  6 of 10 seeds (`runs/mhcal__g0..g9`), because the threshold (2× the Kolmogorov distance of the 400 000-evaluation
  reference MH) is far below the Monte-Carlo error of a 20 000-evaluation run on this posterior. "poor" here mostly
  means "the arm weights are off by more than the reference's own error".
* **Repetitions shift every library seed** (`NAME__gK`: `Configuration.lineage_generation = K`, which adds 10⁶·K to
  every seed incl. the LHS design — a field meant for continuations, used here only as a seed offset; for Auto via
  `plan_auto` + `conf._set_by_auto` + `run_sampling`, the same route as note 27). So the spread over repetitions
  includes the starting points and the warm-up, unlike note 27. `__g0` has the library's default seeds (the same
  starts as the round-1 run). DAMH-SMU runs are not reproducible anyway (asynchronous network delivery); MH is
  (`mh__g0` = `mhcal__g0` = `mh`, identical numbers).

---

## 1. Problem and reference

`Solver_illustrative_local` (G(x, y) = (x² − y)·log((x − y)² + 1)), prior N(0, I₂), likelihood `Normal(mean=5.0, sd=1.0)`,
default `prior_bound=8` — the setup of `toy_examples/hamiltonian_neural_network_example.py`. Every run: in-process
solver, `use_collector=True`, `mpiexec -n 3` (2 samplers + collector), `initial_sample_type="lhs"`; MH-only runs
(`mh*`, `ref_mh`) used `mpiexec -n 2` with `use_collector=False` (2 chains, no collector needed). All runs sequential.

![posterior](../toy_examples/tmp_illustrative_2d_study_2026-10-10/figs/fig_posterior_quadrature.png)

`figs/fig_posterior_quadrature.png`: a curved ridge that splits into two arms — a broad main ridge running from
(−1, −3) to (3, 3) with its maximum at (1.39, −0.80), and a narrow (sd ≈ 0.1 in x) vertical arm at x ≈ −1.8 with a
local maximum at (−1.81, 0.11) of relative density 0.71; a third, detached piece at (2.5, 4) has density < 10⁻³.
**Multimodal inside the box: yes** — the highest path between the two maxima drops to 1.36·10⁻⁵ of the maximum
(`scripts/ref_saddle.py`); the left basin holds 0.301 of the mass (0.302 for {x < −1.4}, the "left-arm mass" used
below). The x-marginal is bimodal (sharp peak at −1.8, broad one at 1.4).

| | quadrature (3001², trapezoid on [−8, 8]²) | long adaptive-RW MH (`runs/ref_mh`) |
|---|---|---|
| mean | (0.18458, −0.69279) | (0.0508, −0.6571) |
| sd | (1.47472, 1.06426) | (1.5011, 1.0758) |
| corr | −0.212 | |
| own error | change vs a 1501² grid ≤ 3·10⁻¹⁶ sd; mass with \|x\| or \|y\| > 7: 3·10⁻¹⁴ | min ESS 89 (x) / 259 (y) of 360 000 samples; R̂ 1.02; 2 × 200 000 evaluations, first 20 000 per chain excluded (burn-in stage); 4.3 s |
| agreement | | mean dev 0.091 / 0.034 sd = **0.84 / 0.53 SE**; sd dev 1.8 % / 1.1 %; KS 0.0414 / 0.0213; TV40 0.054; left-arm mass 0.344 (chains 0.27 / 0.42); 172 / 137 arm transitions |

The MH reference agrees with the quadrature within its own Monte-Carlo error, which is large (1/√89 = 0.11 sd for
the mean of x) because of the arm switching. **All correctness numbers below are against the quadrature**; the MH
reference only sets the Kolmogorov threshold. Checks per run (`scripts/analyze.py`, posterior = all stages after the
warm-up, both chains pooled): mean deviation in reference sd and in SE = sd/√ESS of the run; Kolmogorov distance of
each marginal (empirical CDF vs quadrature CDF); TV distance of a 40×40 histogram over the 99.9 % marginal box
(x ∈ [−2.73, 2.86], y ∈ [−3.10, 2.65], plus one outside bin) vs the quadrature mass. **"poor" = max |mean dev| > 3 SE
or KS_x > 0.0829 or KS_y > 0.0426** (2× the reference). Also reported: left-arm mass per chain and arm transitions
per chain (hysteresis: left after x < −1.6, main after x > −1.0; a crude count that also counts jitter at the arm
foot). TV40 of a 20 000-evaluation run is dominated by MC noise: MH at this budget gives 0.152–0.326 (median 0.165);
the best runs reach 0.06–0.10.

**Budget.** Every study run: 20 000 exact evaluations in total = held-out set (`TestData.generate(problem, 200)`,
evaluated before `run_sampling`, outside the timed region; ~0.2 ms) + 2 chains × 9 900. Hence B0's DAMH stage has
`max_evaluations = 9900 − warm-up = 8900` (brief: 9000; reduced so that the held-out set is inside the 20 000).
MH runs: 2 × 10 000 (1000 burn-in + 9000), no held-out set. Auto: `budget=20000` (held-out 256, 2 × 9872). Every
run's exact-evaluation count was checked from `notes` (accepted + rejected + audited + held-out = 20 000 in every
finished run). Metrics: ESS = min over the 2 parameters of `Samples.calculate_effective_sample_size` on the posterior
stages; **ESS/eval = ESS / 20 000**; ESS/s with the wall time of `run_sampling` on rank 0 (Auto: incl. its held-out
evaluation); outer acceptance and pre-rejection rate over the DAMH stages; `oob` = out-of-bounds proposals; exact
steps, audited (`audit_hidden`), Polyak installs from `notes`; rmse = last row of `surrogate_quality_test.csv`.

**B0** = `[Stage(proposal=RandomWalk(), max_evaluations=1000, burn_in=True), Stage(algorithm="DAMH",
proposal=Hamiltonian(integrator="dimension_robust"), max_evaluations=8900)]` (adaptive step, 10 steps, mass 1),
`NeuralNetworkUpdater(no_parameters=2, no_observations=1, hidden_layer_sizes=(32, 32))` (default seed `None`),
held-out 200, no safeguards, default `min_snapshots_*` (1 / 0).

---

## 2. Round 1 — one factor at a time from B0 (1 run each)

`runs/<name>/`, names in `scripts/configs.py`. Columns as in §1; "left-arm mass" per chain (quadrature 0.302).
B0 is shown once (it is the reference level of every group).

| run | ESS min | ESS/eval | ESS/s | wall s | R̂ | outer acc | prerej | oob | exact steps | audited (hidden) | Polyak | rmse held-out | mean dev (sd) | mean dev (SE) | KS x / y | TV40 | left-arm mass ch0/ch1 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **base** | | | | | | | | | | | | | | | | | | | |
| B0 | 26 | 0.0013 | 1 | 20.9 | 1.03 | 0.97 | 0.2 | 244 | 0 | 0 (0) | 0 | 0.0574 | 1.16 | 9.1 | 0.587 / 0.346 | 0.592 | 0.83 / 0.95 | poor: mean 9.1 SE, KS0 0.587, KS1 0.346 |
| **warm-up length** | | | | | | | | | | | | | | | | | | | |
| warm-up 500 | 29 | 0.0014 | 1 | 21.9 | 1.71 | 0.94 | 0.2 | 277 | 0 | 0 (0) | 0 | 0.0838 | 0.68 | 3.7 | 0.346 / 0.204 | 0.350 | 0.30 / 1.00 | poor: mean 3.7 SE, KS0 0.346, KS1 0.204 |
| warm-up 2000 | 25 | 0.0012 | 1 | 19.1 | 1.08 | 0.97 | 0.2 | 284 | 0 | 0 (0) | 0 | 0.0459 | 0.03 | 0.2 | 0.019 / 0.024 | 0.069 | 0.16 / 0.45 | ok |
| warm-up 4000 | 88 | 0.0044 | 6 | 14.8 | 1.02 | 0.98 | 0.2 | 155 | 0 | 0 (0) | 0 | 0.0408 | 1.32 | 71.7 | 0.670 / 0.398 | 0.675 | 0.95 / 1.00 | poor: mean 71.7 SE, KS0 0.670, KS1 0.398 |
| **num_steps** | | | | | | | | | | | | | | | | | | | |
| num_steps 3 | 970 | 0.0485 | 69 | 14.0 | 4.04 | 0.95 | 0.21 | 18 | 0 | 0 (0) | 0 | 0.0674 | 0.35 | 11.8 | 0.198 / 0.149 | 0.212 | 0.01 / 0.99 | poor: mean 11.8 SE, KS0 0.198, KS1 0.149 |
| num_steps 5 | 82 | 0.0041 | 4 | 19.0 | 2.81 | 0.97 | 0.2 | 74 | 0 | 0 (0) | 0 | 0.0486 | 0.33 | 7.5 | 0.171 / 0.111 | 0.182 | 0.00 / 0.94 | poor: mean 7.5 SE, KS0 0.171, KS1 0.111 |
| num_steps 20 | 50 | 0.0025 | 1 | 58.8 | 1 | 0.98 | 0.2 | 641 | 0 | 0 (0) | 0 | 0.0291 | 0.45 | 4.7 | 0.221 / 0.137 | 0.232 | 0.10 / 0.07 | poor: mean 4.7 SE, KS0 0.221, KS1 0.137 |
| num_steps 30 | 34 | 0.0017 | 0 | 86.3 | 1.71 | 0.98 | 0.2 | 855 | 0 | 0 (0) | 0 | 0.0578 | 0.67 | 4.0 | 0.347 / 0.205 | 0.351 | 0.30 / 1.00 | poor: mean 4.0 SE, KS0 0.347, KS1 0.205 |
| **integrator** | | | | | | | | | | | | | | | | | | | |
| leapfrog | 21 | 0.0010 | 1 | 32.7 | 1.02 | 0.96 | 0.2 | 233 | 0 | 0 (0) | 0 | 0.0889 | 1.04 | 6.1 | 0.528 / 0.308 | 0.532 | 0.78 / 0.88 | poor: mean 6.1 SE, KS0 0.528, KS1 0.308 |
| **step size** | | | | | | | | | | | | | | | | | | | |
| ε = 0.05 fixed | 1214 | 0.0607 | 45 | 26.9 | 4 | 0.97 | 0.01 | 0 | 0 | 0 (0) | 0 | 0.0341 | 0.37 | 12.6 | 0.199 / 0.137 | 0.204 | 0.00 / 0.99 | poor: mean 12.6 SE, KS0 0.199, KS1 0.137 |
| ε = 0.1 fixed | 4718 | 0.2359 | 173 | 27.2 | 3.85 | 0.96 | 0.046 | 0 | 0 | 0 (0) | 0 | 0.055 | 0.40 | 26.4 | 0.204 / 0.127 | 0.210 | 0.00 / 0.99 | poor: mean 26.4 SE, KS0 0.204, KS1 0.127 |
| ε = 0.2 fixed | 83 | 0.0042 | 2 | 35.5 | 1.01 | 0.98 | 0.53 | 5043 | 0 | 0 (0) | 0 | 0.0454 | 0.11 | 1.0 | 0.056 / 0.035 | 0.123 | 0.21 / 0.29 | ok |
| ε = 0.4 fixed | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | – | run failed |
| **network** | | | | | | | | | | | | | | | | | | | |
| (16,16) | 19 | 0.0010 | 1 | 20.7 | 1.41 | 0.97 | 0.2 | 332 | 0 | 0 (0) | 0 | 0.0561 | 0.04 | 0.2 | 0.023 / 0.012 | 0.065 | 0.58 / 0.01 | ok |
| (64,64) | 18 | 0.0009 | 1 | 20.8 | 1.01 | 0.97 | 0.2 | 269 | 0 | 0 (0) | 0 | 0.0438 | 0.82 | 3.7 | 0.417 / 0.247 | 0.422 | 0.69 / 0.75 | poor: mean 3.7 SE, KS0 0.417, KS1 0.247 |
| (128,) | 20 | 0.0010 | 1 | 16.8 | 1.39 | 0.94 | 0.2 | 290 | 0 | 0 (0) | 0 | 0.173 | 0.84 | 4.1 | 0.428 / 0.250 | 0.433 | 0.47 / 0.99 | poor: mean 4.1 SE, KS0 0.428, KS1 0.250 |
| **iterations_batch** | | | | | | | | | | | | | | | | | | | |
| iterations_batch 50 | 17 | 0.0009 | 1 | 21.7 | 1.04 | 0.98 | 0.2 | 217 | 0 | 0 (0) | 0 | 0.0419 | 0.92 | 4.4 | 0.469 / 0.278 | 0.474 | 0.86 / 0.68 | poor: mean 4.4 SE, KS0 0.469, KS1 0.278 |
| iterations_batch 300 | 20 | 0.0010 | 1 | 21.5 | 1.48 | 0.98 | 0.2 | 269 | 0 | 0 (0) | 0 | 0.0564 | 0.80 | 3.8 | 0.405 / 0.239 | 0.409 | 0.42 / 1.00 | poor: mean 3.8 SE, KS0 0.405, KS1 0.239 |
| **min_snapshots_to_update** | | | | | | | | | | | | | | | | | | | |
| min_snapshots_to_update 20 | 23 | 0.0011 | 1 | 21.0 | 1.13 | 0.98 | 0.2 | 317 | 0 | 0 (0) | 0 | 0.0331 | 0.30 | 1.7 | 0.145 / 0.080 | 0.160 | 0.00 / 0.31 | poor: KS0 0.145, KS1 0.080 |
| min_snapshots_to_update 100 | 56 | 0.0028 | 3 | 20.7 | 1.91 | 0.94 | 0.2 | 301 | 0 | 0 (0) | 0 | 0.0883 | 0.18 | 1.7 | 0.099 / 0.066 | 0.117 | 0.00 / 0.80 | poor: KS0 0.099, KS1 0.066 |
| **subchain_length** | | | | | | | | | | | | | | | | | | | |
| subchain_length 3 | 30 | 0.0015 | 1 | 49.2 | 1.28 | 0.98 | 0.027 | 567 | 0 | 0 (0) | 0 | 0.0516 | 0.95 | 8.4 | 0.480 / 0.274 | 0.484 | 0.57 / 0.99 | poor: mean 8.4 SE, KS0 0.480, KS1 0.274 |
| **safeguards** | | | | | | | | | | | | | | | | | | | |
| exact_step_probability 0.05 | 20 | 0.0010 | 1 | 19.7 | 1.3 | 0.93 | 0.19 | 272 | 1.06e+03 | 0 (0) | 0 | 0.0701 | 0.40 | 1.7 | 0.210 / 0.132 | 0.217 | 0.78 / 0.25 | poor: KS0 0.210, KS1 0.132 |
| audit_prerejected 0.05 | 19 | 0.0010 | 1 | 20.5 | 1.16 | 0.96 | 0.2 | 324 | 0 | 186 (1) | 0 | 0.0651 | 0.15 | 0.7 | 0.068 / 0.037 | 0.101 | 0.06 / 0.42 | ok |
| polyak_averaging | 35 | 0.0017 | 2 | 21.1 | 1.03 | 0.95 | 0.2 | 318 | 0 | 0 (0) | 1.01e+03 | 0.0299 | 0.41 | 3.5 | 0.203 / 0.112 | 0.216 | 0.03 / 0.17 | poor: mean 3.5 SE, KS0 0.203, KS1 0.112 |
| surrogate_error_model | 104 | 0.0052 | 5 | 22.3 | 1 | 0.98 | 0.2 | 236 | 0 | 0 (0) | 0 | 0.0388 | 1.32 | 92.3 | 0.671 / 0.389 | 0.675 | 0.96 / 0.98 | poor: mean 92.3 SE, KS0 0.671, KS1 0.389 |
| all four | 88 | 0.0044 | 4 | 20.9 | 1.12 | 0.60 | 0.19 | 441 | 1.05e+03 | 175 (1) | 1.02e+03 | 0.0366 | 0.07 | 0.7 | 0.037 / 0.018 | 0.107 | 0.44 / 0.12 | ok |
| **layout** | | | | | | | | | | | | | | | | | | | |
| 4 equal chunks | 20 | 0.0010 | 1 | 20.7 | 1.03 | 0.98 | 0.2 | 565 | 0 | 0 (0) | 0 | 0.0516 | 0.60 | 2.7 | 0.308 / 0.177 | 0.313 | 0.70 / 0.52 | poor: KS0 0.308, KS1 0.177 |
| SMU half + frozen half | 58 | 0.0029 | 3 | 20.4 | 1.03 | 0.96 | 0.2 | 493 | 0 | 0 (0) | 0 | 0.169 | 0.48 | 8.5 | 0.235 / 0.128 | 0.245 | 0.13 / 0.00 | poor: mean 8.5 SE, KS0 0.235, KS1 0.128 |

**What round 1 shows.**

* **Almost every Hamiltonian DAMH-SMU run is "poor" because a chain is trapped in one arm**: in 12 of the 27
  finished Hamiltonian runs one chain has left-arm mass ≥ 0.94 (trapped in the narrow left arm — e.g. `warm4000`
  0.95 / 1.00 with 1 / 0 transitions, `em` 0.96 / 0.98, B0 0.83 / 0.95), in others a chain never visits it
  (`eps0.05`, `eps0.1`, `L3`, `L5`: one chain at 0.00, the other at 0.99, R̂ ≈ 3–4). `figs/fig_traces_x.png` shows the
  traces. The adapted step sizes are ε ≈ 0.14–0.17 (`adaptive_stats`, log step −1.75 to −2.0) in both arms, so
  the trapping is not a step-size collapse; my reading (not verified) is that a 10-step trajectory of length ≈ 1.5
  rarely carries enough energy over an 11-nat barrier, and the gradient surrogate is poorest exactly there.
  Whether the over-weighting of the left arm in some runs is pure trapping or also a DAMH-SMU bias (no
  diminishing adaptation without Polyak) **could not be checked** with these data.
* **Only 5 Hamiltonian settings passed**: `safe_all` (all four safeguards, ESS/eval 0.0044), `eps0.2` (fixed ε = 0.2,
  0.0042), `warm2000` (0.0012), `nn16` (0.00097), `audit` (0.00096). Their ESS (19–88) is at the level of plain MH
  (28 in `mh`). The only setting with clearly more arm transitions is `safe_all` (83 / 87 vs 7 / 4 in B0): the exact
  random-walk steps of S2 (1054 of them) cross the barrier that the Hamiltonian proposal does not, at the price of
  ~5 % of the budget. Fixed ε = 0.2 crosses too (54 / 69) but proposes out of the ±8 box in 5 043 of ~11 000 outer
  iterations (pre-rejection 0.53).
* **ε = 0.4 fixed: run failed (did not terminate)** — killed by the 900 s timeout (`runs/eps0.4.log`); the repetition
  `eps0.4__g1` was killed by a 240 s timeout (`runs/eps0.4__g1.log`). Diagnostic run `eps0.4_t30` (same, with
  `time_limit=30` on the DAMH stage): in 30 s each chain did 17 800 outer iterations, **96 % of the proposals were out
  of bounds** (17 113 / 17 001) and only 66 / 60 exact evaluations happened (`notes`), i.e. at that rate the 8 900
  evaluations would take ≈ 70 min per chain. Out-of-bounds proposals are rejected without an exact evaluation, so a
  stage stopped only by `max_evaluations` makes no progress towards its stopping rule. (Flagged in §7.)
* Single-factor effects are otherwise not separable from the arm-crossing noise: e.g. warm-up 2000 "ok" but 500 and
  4000 "poor"; (16,16) "ok", (32,32)/(64,64)/(128,) "poor"; num_steps 3/5 have the highest ESS of the group but are
  wrong (stuck chains, R̂ 2.8–4.0). `subchain_length=3` and `num_steps` 20 / 30 cost 2.4–4.1× the wall time.
  `chunks4` vs one stage and the frozen second half: no measurable gain (both poor here).

**Selection for round 2 (the brief's rule: 4 best Hamiltonian settings by ESS/eval among the correct runs):**
`safe_all`, `eps0.2`, `warm2000`, `nn16` (`audit` is 5th by 1.5 %). Candidate combinations (`scripts/candidates.py`):
`cand1` = all four safeguards + warm-up 2000 + (16,16), adaptive ε; `cand2` = `cand1` + fixed ε = 0.2;
`cand3` = all four safeguards + fixed ε = 0.2 (warm-up 1000, (32,32)).

## 3. Round 1b — other variants (1 run each)

| run | ESS min | ESS/eval | ESS/s | wall s | R̂ | outer acc | prerej | oob | exact steps | audited (hidden) | Polyak | rmse held-out | mean dev (sd) | mean dev (SE) | KS x / y | TV40 | left-arm mass ch0/ch1 | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DAMH-SMU RW sub 1 | 44 | 0.0022 | 12 | 3.5 | 1.18 | 0.85 | 0.77 | 225 | 0 | 0 (0) | 0 | 0.151 | 0.28 | 2.2 | 0.143 / 0.085 | 0.161 | 0.00 / 0.32 | poor: KS0 0.143, KS1 0.085 |
| DAMH-SMU RW sub 5 | 89 | 0.0044 | 22 | 4.0 | 1.21 | 0.86 | 0.31 | 134 | 0 | 0 (0) | 0 | 0.0801 | 0.09 | 0.9 | 0.048 / 0.028 | 0.097 | 0.04 / 0.47 | ok |
| DAMH-SMU RW sub 20 | 89 | 0.0045 | 13 | 6.7 | 1.23 | 0.92 | 0.016 | 6 | 0 | 0 (0) | 0 | 0.129 | 0.11 | 1.4 | 0.055 / 0.035 | 0.090 | 0.03 / 0.47 | ok |
| DAMH-SMU pCN sub 1 | 292 | 0.0146 | 77 | 3.8 | 1 | 0.86 | 0.84 | 0 | 0 | 0 (0) | 0 | 0.234 | 0.03 | 0.8 | 0.012 / 0.015 | 0.091 | 0.32 / 0.28 | ok |
| DAMH-SMU pCN sub 20 | 301 | 0.0151 | 42 | 7.2 | 1.02 | 0.95 | 0.04 | 0 | 0 | 0 (0) | 0 | 0.0551 | 0.07 | 1.6 | 0.040 / 0.031 | 0.072 | 0.28 / 0.41 | ok |
| RW sub 20, polynomial deg 3 | 127 | 0.0064 | 6 | 20.7 | 1.01 | 0.75 | 0.012 | 0 | 0 | 0 (0) | 0 | 2.9 | 0.42 | 6.4 | 0.210 / 0.109 | 0.239 | 0.06 / 0.13 | poor: mean 6.4 SE, KS0 0.210, KS1 0.109 |
| RW sub 20, RBF | 331 | 0.0166 | 13 | 24.7 | 1.01 | 1.00 | 0.016 | 0 | 0 | 0 (0) | 0 | 0.0451 | 0.05 | 0.9 | 0.024 / 0.008 | 0.059 | 0.26 / 0.39 | ok |
| MH + Hamiltonian on NN gradients (SMU) | 20 | 0.0010 | 1 | 15.9 | 1.46 | 0.80 | – | 171 | – | – (–) | – | 0.122 | 0.05 | 0.3 | 0.012 / 0.020 | 0.076 | 0.00 / 0.62 | ok |
| Auto fast | 22 | 0.0011 | 0 | 57.3 | 1.3 | 0.87 | 0.19 | 1225 | 1.13e+03 | 150 (0) | 1.86e+03 | 0.0465 | 0.03 | 0.2 | 0.022 / 0.022 | 0.076 | 0.55 / 0.05 | ok |
| Auto robust | 63 | 0.0032 | 13 | 4.9 | 1.03 | 0.56 | 0.73 | 327 | 3.14e+03 | 2.28e+03 (3) | 58 | 0.151 | 0.20 | 4.1 | 0.112 / 0.088 | 0.167 | 0.28 / 0.11 | poor: mean 4.1 SE, KS0 0.112, KS1 0.088 |
| MH adaptive RW (baseline) | 28 | 0.0014 | 121 | 0.2 | 1.05 | 0.22 | – | 0 | – | – (–) | – | – | 0.47 | 2.4 | 0.237 / 0.144 | 0.270 | 0.63 / 0.44 | poor: KS0 0.237, KS1 0.144 |

* **pCN in DAMH-SMU is the clear winner** (ESS/eval 0.0146 / 0.0151 for sub-chain 1 / 20; 254–369 arm transitions per
  chain; adapted β ≈ 0.64 in `pcn_sub1`, i.e. large, nearly prior-scale moves that jump between the arms; the
  surrogate pre-screens them at 0.84 pre-rejection). Sub-chain 20 buys nothing over 1 for pCN and doubles the wall
  time.
* **RW sub-chain 20 with the RBF surrogate** is as good per evaluation (0.0166, 104 / 103 transitions; outer
  acceptance 1.00 — the interpolant is exact at the snapshots), but 3.7× slower in wall time than with the network (24.7 s vs 6.7 s).
  With the network RW sub-chain 5 / 20 are 0.0044 / 0.0045 (ok), sub-chain 1 is poor; polynomial degree 3 is poor
  (held-out rmse 2.9 — the model is not a cubic).
* **MH with the Hamiltonian on network gradients** (`mh_hamgrad`, exact acceptance, 80 %) passed but has ESS 20 —
  no better than plain MH (28).
* **Auto fast passed this time, Auto robust did not** (single runs; see round 2 for the spread).
* **MH baseline: poor** in this run (arm masses 0.63 / 0.44); at this budget MH itself fails the rule in 6 of 10
  seeds (`runs/mhcal__g*`, table in `tables.md`): KS_x 0.013–0.300 (median 0.069), KS_y 0.016–0.175 (median 0.054),
  ESS/eval 0.0011–0.0657 (median 0.0023; the 0.0657 is seed 4 with one chain that never left an arm, R̂ 1.0, mean off
  by 43 SE).

## 4. Round 2 — finalists with repetitions (5 runs each, seeds shifted per repetition)

`runs/<name>__g0 … __g4`. Median [min–max] over the 5 repetitions; "verdicts" lists each repetition in order g0…g4.
Sorted by median ESS/eval.

| setting | n | ESS min | ESS/eval | ESS/s | wall s | R̂ | outer acc | prerej | oob | exact steps | audited | hidden | Polyak | rmse held-out | mean dev (sd) | mean dev (SE) | KS x | KS y | TV40 | verdicts |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| DAMH-SMU pCN sub 1 | 5 | 259 [120–359] | 0.0129 [0.00601–0.0179] | 66 [32.7–85.9] | 3.8 [3.67–4.37] | 1 [1–1.01] | 0.86 [0.84–0.92] | 0.85 [0.83–0.87] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0.113 [0.0922–0.163] | 0.016 [0.015–0.064] | 0.37 [0.26–1] | 0.017 [0.011–0.0356] | 0.0134 [0.00908–0.0268] | 0.0944 [0.0869–0.102] | ok, ok, ok, ok, ok |
| RW sub 20, RBF | 5 | 193 [96.7–285] | 0.00964 [0.00483–0.0143] | 8.07 [3.92–11.4] | 24.7 [23.9–25] | 1 [1–1.17] | 1 [0.99–1] | 0.015 [0.013–0.016] | 0 [0–8] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0.0335 [0.0172–0.0744] | 0.23 [0.035–0.28] | 2.3 [0.76–4.7] | 0.118 [0.0165–0.144] | 0.0668 [0.0178–0.0888] | 0.128 [0.0623–0.155] | poor, poor, ok, poor, ok |
| cand2 | 5 | 76.9 [42–257] | 0.00385 [0.0021–0.0129] | 4.2 [2.42–16.4] | 17.4 [15.7–20.5] | 1.01 [1–1.09] | 0.63 [0.4–0.75] | 0.12 [0.029–0.24] | 50 [0–497] | 875 [816–1.01e+03] | 100 [24–229] | 0 [0–1] | 814 [440–1e+03] | 0.0241 [0.0205–0.0693] | 0.084 [0.03–0.12] | 0.72 [0.49–1] | 0.047 [0.0268–0.0625] | 0.0313 [0.0122–0.0389] | 0.125 [0.107–0.165] | ok, ok, ok, ok, ok |
| Auto robust | 5 | 75.3 [30–126] | 0.00376 [0.0015–0.00632] | 17.2 [6.8–28.9] | 4.37 [4.24–4.41] | 1.08 [1–1.11] | 0.56 [0.51–0.59] | 0.73 [0.73–0.74] | 419 [211–614] | 3.1e+03 [3.05e+03–3.13e+03] | 2.22e+03 [2.16e+03–2.31e+03] | 8 [1–37] | 60 [59–60] | 0.157 [0.1–0.442] | 0.2 [0.11–0.37] | 2.3 [1.5–6.7] | 0.105 [0.0587–0.187] | 0.077 [0.0358–0.111] | 0.159 [0.13–0.219] | poor, poor, poor, poor, ok |
| ε = 0.2 fixed | 5 | 53 [46.3–129] | 0.00265 [0.00232–0.00646] | 1.52 [1.29–3.46] | 35.4 [34.6–37.4] | 1.02 [1–1.13] | 0.98 [0.98–0.99] | 0.54 [0.52–0.55] | 5.22e+03 [4.64e+03–5.3e+03] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0.0637 [0.029–0.105] | 0.044 [0.0031–0.17] | 0.56 [0.035–1.9] | 0.0204 [0.00896–0.0829] | 0.0147 [0.01–0.0443] | 0.0963 [0.0875–0.122] | poor, ok, ok, ok, ok |
| MH adaptive RW (baseline) | 5 | 48.5 [22.8–1.31e+03] | 0.00243 [0.00114–0.0656] | 217 [101–5.76e+03] | 0.226 [0.224–0.229] | 1.4 [1–1.94] | 0.22 [0.22–0.24] | – | 1 [0–3] | – | – | – | – | – | 0.15 [0.011–0.6] | 1 [0.078–43] | 0.0728 [0.0128–0.3] | 0.0453 [0.0155–0.175] | 0.166 [0.152–0.326] | poor, ok, poor, ok, poor |
| all four | 5 | 47.4 [37–77.1] | 0.00237 [0.00185–0.00385] | 2.26 [1.8–3.66] | 20.9 [20.6–21.1] | 1.01 [1–1.14] | 0.7 [0.6–0.74] | 0.19 [0.19–0.19] | 454 [356–508] | 1.08e+03 [1.05e+03–1.11e+03] | 186 [179–215] | 0 [0–0] | 982 [942–999] | 0.0383 [0.0215–0.0891] | 0.15 [0.025–0.28] | 1.1 [0.23–1.9] | 0.0824 [0.0169–0.143] | 0.0621 [0.0284–0.0892] | 0.119 [0.0939–0.167] | poor, ok, poor, poor, poor |
| cand3 | 5 | 46.8 [38.4–88.5] | 0.00234 [0.00192–0.00443] | 2.17 [1.93–4.98] | 19.6 [17.8–24.2] | 1.01 [1–1.09] | 0.62 [0.49–0.81] | 0.12 [0.048–0.32] | 52 [5–818] | 1.01e+03 [946–1.24e+03] | 119 [45–391] | 0 [0–0] | 896 [515–1.16e+03] | 0.022 [0.0181–0.119] | 0.079 [0.037–0.37] | 0.82 [0.23–4.5] | 0.0464 [0.0258–0.19] | 0.0278 [0.00898–0.114] | 0.136 [0.0992–0.206] | poor, poor, ok, ok, ok |
| cand1 | 5 | 31.4 [23.5–336] | 0.00157 [0.00117–0.0168] | 1.68 [1.23–17.9] | 18.8 [18.7–19.1] | 1.07 [1–1.39] | 0.64 [0.42–0.75] | 0.19 [0.19–0.19] | 403 [306–657] | 979 [923–1e+03] | 162 [154–179] | 0 [0–0] | 964 [952–979] | 0.041 [0.0302–0.106] | 0.27 [0.042–0.48] | 2.2 [0.2–2.8] | 0.143 [0.0259–0.243] | 0.0799 [0.0139–0.144] | 0.174 [0.0932–0.263] | poor, ok, poor, poor, poor |
| (16,16) | 5 | 28.7 [18.9–46.6] | 0.00144 [0.000946–0.00233] | 1.38 [0.913–2.26] | 20.7 [20.5–20.9] | 1.11 [1.03–1.93] | 0.97 [0.96–0.98] | 0.2 [0.2–0.2] | 297 [221–337] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0.0535 [0.0314–0.109] | 0.45 [0.017–1.1] | 4.9 [0.074–8.3] | 0.229 [0.0115–0.559] | 0.133 [0.00959–0.321] | 0.236 [0.0669–0.563] | poor, poor, poor, poor, ok |
| Auto fast | 5 | 26.7 [23.4–73.3] | 0.00134 [0.00117–0.00366] | 0.474 [0.41–1.31] | 56.6 [55.9–57.1] | 1.07 [1–2.02] | 0.84 [0.81–0.87] | 0.19 [0.19–0.19] | 1.36e+03 [1.28e+03–1.42e+03] | 1.17e+03 [1.13e+03–1.18e+03] | 144 [136–154] | 0 [0–1] | 1.86e+03 [1.86e+03–1.86e+03] | 0.0531 [0.0212–0.169] | 0.18 [0.074–0.55] | 0.93 [0.54–6] | 0.0957 [0.0474–0.278] | 0.0615 [0.0399–0.164] | 0.115 [0.0948–0.284] | poor, ok, poor, poor, poor |
| warm-up 2000 | 5 | 21.5 [18.5–167] | 0.00108 [0.000924–0.00836] | 1.15 [0.982–8.94] | 18.7 [18.6–18.8] | 1.18 [1.03–3.22] | 0.97 [0.96–0.98] | 0.2 [0.2–0.2] | 255 [223–303] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0.0565 [0.0346–0.13] | 0.35 [0.039–1.3] | 3.2 [0.17–43] | 0.187 [0.026–0.659] | 0.122 [0.0132–0.375] | 0.197 [0.07–0.664] | poor, ok, poor, poor, poor |
| B0 | 5 | 21.3 [18.8–8.42e+03] | 0.00106 [0.000938–0.421] | 1.02 [0.903–403] | 20.8 [20.5–20.9] | 1.1 [1–3.9] | 0.98 [0.95–0.98] | 0.2 [0.2–0.2] | 303 [284–347] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0 [0–0] | 0.0652 [0.0408–0.108] | 0.39 [0.025–0.44] | 1.8 [0.11–34] | 0.199 [0.014–0.218] | 0.116 [0.00842–0.128] | 0.207 [0.0665–0.229] | ok, ok, poor, poor, poor |

![round 2](../toy_examples/tmp_illustrative_2d_study_2026-10-10/figs/fig_round2_ess_vs_wall.png)

`figs/fig_round2_ess_vs_wall.png`: ESS/eval vs wall time per repetition, black ring = "poor". The isolated high points
of B0 (0.42) and MH (0.066) are runs with a chain stuck in one arm (R̂ 3.9 / 1.0): their ESS is an artefact.

**Ranking (median ESS/eval, number of "ok" repetitions out of 5):**

| setting | ESS/eval | × MH | ok | ESS/s | × MH |
|---|---|---|---|---|---|
| DAMH-SMU pCN sub-chain 1, NN (32,32) | 0.0129 | 5.3 | 5/5 | 66 | 0.30 |
| DAMH-SMU RW sub-chain 20, RBF | 0.0096 | 4.0 | 2/5 | 8.1 | 0.04 |
| `cand2` (Hamiltonian ε 0.2 fixed, safeguards, warm 2000, NN (16,16)) | 0.00385 | 1.6 | 5/5 | 4.2 | 0.02 |
| Auto robust | 0.00376 | 1.5 | 1/5 | 17 | 0.08 |
| ε = 0.2 fixed (B0 otherwise) | 0.00265 | 1.1 | 4/5 | 1.5 | 0.007 |
| MH adaptive RW | 0.00243 | 1 | 2/5 | 217 | 1 |
| `safe_all` | 0.00237 | 0.98 | 1/5 | 2.3 | 0.01 |
| `cand3` | 0.00234 | 0.96 | 3/5 | 2.2 | 0.01 |
| `cand1` | 0.00157 | 0.65 | 1/5 | 1.7 | 0.008 |
| (16,16) | 0.00144 | 0.59 | 1/5 | 1.4 | 0.006 |
| Auto fast | 0.00134 | 0.55 | 1/5 | 0.47 | 0.002 |
| warm-up 2000 | 0.00108 | 0.44 | 1/5 | 1.2 | 0.005 |
| B0 | 0.00106 | 0.44 | 2/5 | 1.0 | 0.005 |

The round-1 winners `safe_all`, `warm2000`, `nn16` did **not** hold up over repetitions (1/5 ok each): their round-1
"ok" was arm-crossing luck. Only `cand2` (5/5) and `eps0.2` (4/5) among the Hamiltonian settings pass consistently.
Per second, MH wins everything (the model costs 1.1 µs). From the medians, pCN sub 1 overtakes MH in ESS/s once the
model costs more than ≈ 30 µs per evaluation (5.3·(0.23 s + 20 000 c) > 3.8 s + 20 000 c), `cand2` once it costs more
than ≈ 1.5 ms (rough break-even estimates, assuming the overhead does not grow with the model cost).

## 5. Correctness summary

Every finished run, its deviations and verdict: `table_all.csv` (column `verdict`) and the tables above. Totals:
round 1 + 1b: 38 configurations, 37 finished, **12 ok, 25 poor, 1 run failed** (`eps0.4`; plus its repetition and
the diagnostic run). Round 2: 65 repetitions, **29 ok, 36 poor**, 0 failed. MH calibration: 4/10 ok. Reference MH
(`ref_mh`): agrees with the quadrature (0.84 / 0.53 SE).

The "poor" verdicts, by cause (cause read off the per-chain left-arm mass, transitions and R̂ in `metrics.json`):

* **A chain trapped in the narrow left arm** (left-arm mass ≥ 0.94 in at least one chain, few transitions): round 1
  B0, `warm500`, `warm4000`, `L30`, `em`, `it300`, `nn128`, `sub3`, `L5`, `L3`, `eps0.05`, `eps0.1`; this is
  the dominant failure of every Hamiltonian setting without S2 exact steps, also in round 2 (B0, `warm2000`, (16,16)).
* **A chain that never visits the left arm** (mass ≤ 0.03 with the other chain far above 0.3): `eps0.05`, `eps0.1`,
  `L3`, `L5`, `ms20`, `ms100`, `rw_sub1`.
* **Both chains under-weight the left arm** (≤ 0.17 in both): `L20`, `polyak`, `frozen`.
* **Arm weights off with both chains crossing** (too few transitions for 20 000 evaluations): `leapfrog`, `chunks4`, `exact`,
  `it50`, `nn64`, `auto_robust`, the poor MH runs, and the poor repetitions of RBF, `safe_all`, `cand1`, `cand3`, Auto fast.
* **Wrong surrogate shape**: `rw_sub20_poly3` (held-out rmse 2.9; both chains under-weight the arm, 0.06 / 0.13).
* No run was poor for a reason other than the arm weights: in all poor runs but two (`mhcal__g8`, `safe_all__g4`)
  the KS distance of x (the bimodal marginal) is the larger one; a relative sd deviation above 0.3 occurs in only 6
  runs (B0, `em`, `frozen`, `warm4000`, `warm2000__g2`, `mh__g4`, max 0.66), all with a chain at left-arm mass ≈ 0 or ≥ 0.93;
  and `audit_hidden` is 0–1 in all Hamiltonian runs (Auto robust: 1–37 per run, the only setting where the audit
  found surrogate blind spots).

## 6. Best setting and how far Auto is from it

**Best overall (5/5 ok, 5.3× MH per evaluation): DAMH-SMU with an adaptive pCN proposal, sub-chain 1.** Copy:

```python
from surrDAMH.proposals import RandomWalk, PCN
from surrDAMH.stages import Stage
from surrDAMH.modules.test_data import TestData
conf = surrDAMH.Configuration(output_dir=..., use_solvers_pool=False, use_collector=True, initial_sample_type="lhs")
stages = [
    Stage(proposal=RandomWalk(), max_evaluations=1000, burn_in=True),
    Stage(algorithm="DAMH", proposal=PCN(), subchain_length=1, max_evaluations=8900),
]
updater = surrDAMH.surrogates.NeuralNetworkUpdater(no_parameters=2, no_observations=1, hidden_layer_sizes=(32, 32))
run = problem.run_sampling(conf, stages, surrogate_updater=updater, surrogate_test_data=TestData.generate(problem, 200))
```

**Best Hamiltonian setting (5/5 ok, 1.6× MH per evaluation): `cand2`.**

```python
stages = [
    Stage(proposal=RandomWalk(), max_evaluations=2000, burn_in=True),
    Stage(algorithm="DAMH", proposal=Hamiltonian(num_steps=10, integrator="dimension_robust", step_size=0.2, adaptive=False),
          max_evaluations=7900, exact_step_probability=0.05, audit_prerejected=0.05,
          polyak_averaging=True, surrogate_error_model=True),
]
updater = surrDAMH.surrogates.NeuralNetworkUpdater(no_parameters=2, no_observations=1, hidden_layer_sizes=(16, 16))
# + TestData.generate(problem, 200), initial_sample_type="lhs" as above
```

Note ε = 0.2 is close to the non-terminating ε = 0.4 (§2); in `cand2` the out-of-bounds count is 0–497 per run
(vs ~5 000 without the safeguards and (16,16)), which I attribute to the error model/smaller network (not separated).

**Auto** (plan in `runs/auto_*/study_case.json`: held-out 256, warm-up 493 per chain, 4 DAMH-SMU chunks of 2344–2347,
NN (64,64) seed 0, S2 0.05/0.05, Polyak, error model, `min_snapshots_initial/to_update` = 20):

* **Auto fast**: 0.00134 ESS/eval, **9.6× below pCN sub 1 and 2.9× below `cand2`**, 1/5 ok, and the slowest run of the
  study (57 s: 30-step trajectories on the network). Its fixed choices are exactly the levels that did worst in
  round 1 — 30 leapfrog steps (`L30` poor, 4× the wall time of L10), warm-up ≈ 500 (`warm500` poor), (64,64)
  (`nn64` poor), adaptive ε (all adaptive-ε Hamiltonian settings trapped at least once) — and its S2 exact steps
  (≈ 1 170 per run) are too few to fix the arm weights at 1/5 ok.
* **Auto robust**: 0.00376 ESS/eval, 3.4× below pCN sub 1, ≈ `cand2` per evaluation and 4× faster in wall time, but
  1/5 ok. Its RW sub-chain 1 (`rw_sub1` poor in round 1b) pre-rejects 73 % of the proposals; the safeguards then spend
  ≈ 3 100 exact steps + 2 200 audits = **27 % of the chain budget** (S2 draws per outer iteration, and most outer
  iterations are cheap pre-rejections); the audits found 1–37 hidden spots per run.
* Neither Auto mode tries pCN, which is what this posterior needs (large prior-scale moves that hop between the arms).

---

## 7. Caveats

* **Cheap model**: 1.1 µs per evaluation, so ESS/s is overhead-dominated (network training on the collector,
  Hamiltonian trajectories on network gradients, MPI). MH is the fastest per second by a factor 3 (vs pCN) to 460
  (vs Auto fast). The break-even model costs in §4 are rough estimates, not measurements.
* **Bimodality dominates everything**: at 20 000 evaluations the results are decided by the number of arm-to-arm
  transitions. The brief's correctness threshold (2× the long-MH Kolmogorov distance) is below the MC error of plain
  MH at this budget (6/10 MH seeds fail it), and ESS from stuck chains is meaningless (R̂ up to 4 with ESS 8 400).
  Five repetitions are still few for such a heavy-tailed outcome (B0's ESS/eval spans 0.0009–0.42).
* **Pending package items that may change the numbers**: the dual-averaging state of the Hamiltonian step size and
  the Welford proposal statistics are not carried across chunks (affects `chunks4`, Auto); Polyak averaging starts at
  the first evaluator of each stage (no delay), so early bad networks keep weight `1/k` (affects `polyak`, `safe_all`,
  the candidates, Auto).
* **DAMH-SMU runs are not reproducible** (asynchronous network delivery; the NN updater seed is `None` in the manual
  runs, 0 in Auto). Repetitions additionally shift the library seeds via `lineage_generation` (a continuation field used
  as a seed offset — not its intended use; it changes only seeds and the LHS design).
* **Budget accounting**: B0's DAMH stage has 8 900 instead of the brief's 9 000 evaluations so that the 200-point
  held-out set fits into the 20 000; the manual held-out set is evaluated outside the timed region (≈ 0.2 ms), Auto's
  inside.
* **Non-termination (flag for the manual-review list)**: a DAMH stage stopped only by `max_evaluations` does not
  terminate in practice when nearly all proposals are pre-rejected without an exact evaluation (here: Hamiltonian
  ε = 0.4, 96 % out of the ±8 box). No package change was made; a `time_limit` on the stage, or a cap/warning on
  consecutive pre-rejections, would avoid it.
* Single problem, d = 2, one noise level; NN updater defaults otherwise; `min_snapshots_initial` 1 in the manual runs,
  20 in Auto.

## 8. Paths

* Study directory: `toy_examples/tmp_illustrative_2d_study_2026-10-10/` (untracked, ~300 MB).
* Reference: `ref_quadrature.json`, `ref_quadrature.npz` (`scripts/ref_quadrature.py`, `scripts/ref_saddle.py`),
  `runs/ref_mh/` (+ `metrics.json`).
* Runs: round 1 / 1b `runs/<name>/` (names in `scripts/configs.py`), round 2 `runs/<name>__g0..g4/`, MH calibration
  `runs/mhcal__g0..g9/`, failed runs `runs/eps0.4.log`, `runs/eps0.4__g1.log`, diagnostic `runs/eps0.4_t30/`;
  batch logs `runs/batch_round1.out`, `runs/batch_round2.out`, `runs/batch_round2b.out`, `runs/batch_diag.out`.
* Tables: `tables.md` (all tables incl. the MH calibration), `table_all.csv` (every run, every metric, verdict).
* Figures: `figs/fig_posterior_quadrature.png`, `figs/fig_traces_x.png`, `figs/fig_round2_ess_vs_wall.png`.
* Drivers: `scripts/run_case.py NAME OUT` (one run; `mpiexec -n 3 /dolfinx-env/bin/python -m mpi4py …`),
  `scripts/analyze.py RUN`, `scripts/batch3.sh NAME…` (sequential run + analyse, 240 s timeout), `scripts/tables.py`.
