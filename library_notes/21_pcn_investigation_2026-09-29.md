# 21 — Why pCN loses on the corrected GRF problem: implementation, tuning, cost (2026-09-29)

**Dated record**, requested by the author after note 19 reported adaptive pCN 12× behind the
adaptive random walk as an exact-model proposal and ≤ 5× MH as a DAMH sub-chain proposal.
Question: is that an implementation or auto-tuning problem? Everything below was produced by
running the code on 2026-09-29; runs, tables and figures are in
`toy_examples/tmp_efficiency_study_2026-09-28/` (runs `P_*`, `Q_*`, plus the note-19 runs they
are compared with; `metrics.csv` now also carries surrogate call counts and cost per effective
sample; figures `fig_pcn_stage1.png`, `fig_pcn_stage2.png`, `fig_cost_per_ess.png`;
`report_pcn.html`). §8 (added 2026-09-30) side-tests the generalised pCN that would fix it.

**Answer in one paragraph.** The pCN implementation is correct and its auto-tuning lands within
a factor ≈ 1.2 of the best fixed β. pCN is slow here because its proposal is *isotropic in the
prior's internal space* while the posterior is not: the reference posterior covariance has
eigen-standard-deviations from 1.02 down to 0.087 (12× anisotropy, 16 directions narrowed by the
data, 4 left at the prior). A pCN step must be small enough for the narrowest direction, so it
crawls in the wide ones; the adaptive random walk learns the covariance and moves every direction
at its own scale. A pure-Gaussian target with exactly that covariance reproduces the gap without
any library code (best pCN τ = 965, covariance random walk τ = 61). In a DAMH sub-chain the same
geometry applies, and pCN additionally suffers from the outer-acceptance adaptation target. pCN
is the right proposal when the posterior is close to the prior (the rank-2 problem of notes
14/18, where it won 4×), not on an informative posterior. The fix that would make pCN
competitive is a *generalised* pCN whose reference Gaussian is the (adapted) posterior
covariance, which the library does not have.

---

## 1. Implementation review (`surrDAMH/modules/proposals.py`)

- `PCN.propose_sample`: `x' = m + sqrt(1 − β²)(x − m) + β η`, `η ~ N(0, C₀)` with `m, C₀` the
  internal prior's mean and covariance (standard normal for every shipped prior) — the standard
  Crank–Nicolson proposal. `get_log_acceptance_probability` returns the likelihood ratio and a
  prior part of exactly 0, which is the pCN acceptance rule (the prior cancels by construction).
  `build_proposal` only constructs a `PCN` for a Gaussian internal prior. Correct.
- `PCN_adaptive.adapt`: Robbins–Monro on `logit β` with gain `n^−0.7`, fed the acceptance
  probability `min(1, exp(log α))` of every proposal (pre-rejected DAMH proposals scored 0);
  the chains pool their logits at stage end. Target 0.234 by default. Correct as a recursion;
  whether 0.234 is the right target is a tuning question (§2).
- Posterior check: the chains sampled with pCN reproduce the reference posterior (note 19 §4:
  `A_pcn` max \|mean − ref\|/sd 0.12 with an ESS of 235, sd along v₁ 0.152 vs 0.149; the DAMH
  pCN runs 0.03–0.08). A wrong acceptance rule (e.g. the prior ratio counted twice) would
  narrow the four unconstrained directions from sd 1 to 0.71 — not observed (max relative sd
  error 7 %, within the noise of that short run).

Nothing in the code is wrong.

## 2. Geometry: the reference posterior is 12× anisotropic

Eigen-sd of the reference covariance (`R1`, 3.2 M states, `R1_cov.npy`):
1.02, 1.00, 0.99, 0.97, 0.91, 0.85, 0.53, 0.49, 0.42, 0.29, 0.25, 0.22, 0.20, 0.20, 0.17, 0.13,
0.125, 0.11, 0.098, 0.087. Per-parameter sd 0.32–0.94 with strong correlations (the eigenvectors
mix parameters), so no diagonal rescaling helps either.

A pCN proposal moves every direction by β (in prior units). To be accepted it must respect the
narrowest directions (sd ≈ 0.09–0.13): β ≈ 0.07–0.1. In the six directions of sd ≈ 1 that is a
relative step of 0.07–0.1, giving an autocorrelation time of order 1/β² ≈ 100–200 *per accepted
move*, i.e. several hundred to a thousand iterations. The adaptive random walk proposes
`N(0, s² Σ̂)` with `Σ̂` the learned covariance and `s² ≈ 2.38²/d` — every direction gets a relative
step ≈ 0.5.

**Synthetic check (no surrogate, no adaptation, no library code):** Gaussian target
`N(0, C_R1)` written as prior `N(0, I)` × likelihood, 200 000 steps of each sampler in NumPy,
τ from emcee:

| sampler | acceptance | τ mean over parameters | τ max | ESS per step |
|---|---|---|---|---|
| pCN β = 0.02 | 0.82 | 6 214 | 21 070 | 0.00016 |
| pCN β = 0.04 | 0.65 | 2 034 | 3 254 | 0.00049 |
| pCN β = 0.07 | 0.43 | 1 317 | 3 274 | 0.00076 |
| **pCN β = 0.10** | 0.27 | **965** | 1 721 | **0.00104** |
| pCN β = 0.15 | 0.12 | 1 013 | 2 576 | 0.00099 |
| pCN β = 0.25 | 0.02 | 2 712 | 4 433 | 0.00037 |
| RW, 0.5 × 2.38²/d × C | 0.41 | 73 | 85 | 0.0137 |
| **RW, 1 × 2.38²/d × C** | 0.25 | **61** | 72 | **0.0164** |
| RW, 2 × 2.38²/d × C | 0.11 | 73 | 84 | 0.0136 |

The best pCN is 16× behind the covariance random walk on the Gaussian alone, at an optimal
acceptance of ≈ 0.27 — the same ratio and the same optimum the real problem shows below. On the
real posterior the adaptive pCN settled at β = 0.068, acceptance 0.233, i.e. within the flat
top of this curve.

## 3. Stage 1 (exact model): pCN against the random walk, auto-tuned and at their optima

Figure: `fig_pcn_stage1.png`. Single MH stage, 4 chains × 200 000 evaluations (first 25 %
dropped), problem P1 of note 19, `A_*` runs from note 19, `P_*` new. "cost/ESS" = seconds of
exact-model time per effective sample at the measured 0.22 ms per evaluation (no surrogate in
these runs); "CPU/ESS" = measured sampler wall time × chains / ESS.

### 3.1 Auto-tuned variants

| run | proposal, target acceptance | β or scale at the end | acceptance | τ (mean / max over parameters) | **ESS/eval** | cost/ESS (s) | CPU/ESS (s) |
|---|---|---|---|---|---|---|---|
| `P_rw_t15` | random walk, target 0.15 | learned Σ̂, log s = −0.63 | 0.151 | 216 / 261 | **0.0035** | 0.063 | 0.11 |
| `A_rw` | random walk, target 0.234 (default) | learned Σ̂, log s = −0.40 | 0.236 | 219 / 279 | **0.0034** | 0.064 | 0.12 |
| `A_rw_t35` | random walk, target 0.35 | learned Σ̂ | 0.352 | 304 / 388 | 0.0025 | 0.089 | 0.17 |
| `P_pcn_t10` | pCN, target 0.10 | β = 0.102 | 0.095 | 2 243 / 4 340 | **0.00032** | 0.66 | 1.12 |
| `A_pcn` | pCN, target 0.234 (default) | β = 0.068 | 0.233 | 2 560 / 4 126 | 0.00029 | 0.75 | 1.41 |
| `P_pcn_t35` | pCN, target 0.35 | β = 0.052 | 0.349 | 2 427 / 3 948 | 0.00030 | 0.71 | 1.25 |
| `P_pcn_t50` | pCN, target 0.50 | β = 0.036 | 0.499 | 3 173 / 5 163 | 0.00021 | 0.93 | 1.67 |

The adaptation does what it is asked: every target is hit to two digits and β follows it
monotonically (0.036 → 0.102 for targets 0.5 → 0.1). The efficiency is flat between targets
0.1 and 0.35 (0.00029–0.00032) — the recursion is not the problem.

### 3.2 Fixed ("optimal") variants

| run | proposal | acceptance | τ mean / max | **ESS/eval** | cost/ESS (s) |
|---|---|---|---|---|---|
| `P_pcn_b02` | pCN β = 0.02 | 0.700 | 5 492 / 7 539 | 0.00013 | 1.61 |
| `P_pcn_b04` | pCN β = 0.04 | 0.458 | 2 641 / 4 404 | 0.00027 | 0.77 |
| `P_pcn_b07` | pCN β = 0.07 | 0.223 | 2 309 / 3 333 | 0.00031 | 0.68 |
| **`P_pcn_b10`** | **pCN β = 0.10** | 0.107 | 2 085 / 3 327 | **0.00036** | **0.61** |
| `P_pcn_b15` | pCN β = 0.15 | 0.030 | 3 746 / 7 532 | 0.00019 | 1.10 |
| `A_pcn_b02` | pCN β = 0.20 (the old example's value) | 0.009 | 4 562 / 8 160 | 0.00016 | 1.33 |
| `P_pcn_b25` | pCN β = 0.25 | 0.003 | 7 177 / 11 462 | 0.00010 | 2.10 |
| `P_pcn_b40` | pCN β = 0.40 | 0.0003 | 13 680 / 17 224 | 0.00005 | 4.00 |
| `P_rw_c0.25` | RW, 0.25 × 2.38²/d × Σ_ref | 0.372 | 218 / 300 | 0.0034 | 0.064 |
| **`P_rw_c0.5`** | **RW, 0.5 × 2.38²/d × Σ_ref** | 0.224 | 187 / 247 | **0.0040** | **0.055** |
| `P_rw_c1` | RW, 1 × 2.38²/d × Σ_ref (the textbook scaling) | 0.103 | 220 / 310 | 0.0034 | 0.064 |
| `P_rw_c2` | RW, 2 × | 0.032 | 388 / 521 | 0.0019 | 0.113 |
| `P_rw_c4` | RW, 4 × | 0.006 | 1 176 / 1 670 | 0.0006 | 0.344 |
| `A_rw_fixed` | RW, library default 2.38²/d × I | 0.0003 | 15 550 / 20 621 | 0.00005 | 4.55 |

Σ_ref = the reference posterior covariance (`R1_cov.npy`), i.e. the best covariance a random walk
could learn. Reading:

- **pCN's optimum is β ≈ 0.10 at 11 % acceptance, 0.00036 ESS/eval** — 1.2× its auto-tuned
  value (0.00029 at target 0.234, 0.00032 at target 0.1). The optimal acceptance rate of pCN on
  this posterior is ≈ 0.1–0.2, lower than the 0.234 default, but the curve is flat: targets
  0.1–0.35 are all within 10 % of the optimum. A target of 0.1 would be the marginally better
  default *for this kind of posterior*; note 16's prototype found the opposite on posteriors
  close to the prior. Not worth changing.
- **The random walk's optimum is 0.5 × the textbook scaling, 0.0040 ESS/eval** — 1.2× its
  auto-tuned value (0.0034, which learns Σ̂ from scratch and ends at log s = −0.40, i.e. 0.67 ×,
  between the two best grid points). The optimal acceptance is ≈ 0.22, and the library's default
  target 0.234 is right.
- **At their optima the random walk is 11× pCN** (0.0040 vs 0.00036); auto-tuned, 12×. The
  synthetic Gaussian of §2 gave 16×. So neither the implementation nor the tuning explains the
  gap: an isotropic proposal on this posterior is simply 10–16× less efficient than one shaped
  like the posterior, and both auto-tuners land within 20 % of their own optimum.
- The library's *fixed* default random-walk step (2.38²/d × I, no covariance) is as bad as the
  worst pCN (acceptance 0.03 %) — adaptivity is what makes the random walk good here, not the
  family.

## 4. Stage 2 (DAMH on the network surrogate): pCN against the Hamiltonian proposal

Figure: `fig_pcn_stage2.png`. Three-stage runs of note 19 (`MH RandomWalk()` 20 000 → DAMH-SMU
50 000 → DAMH frozen 50 000, network (80, 80)); the table shows the frozen stage 3. `B_*` from
note 19, `Q_*` new. "sub-chain acc." = acceptance inside the surrogate sub-chain; "β" = the value
in stage 3 (adaptive runs: the value stage 2 ended with). Surrogate calls are counted per stage
(values: one per sub-chain step; gradients: one `vjp` per leapfrog step); the cost columns use the
micro-benchmarked single-point costs of §5.

| run | pCN variant | sub-chain | β | **ESS/eval** | τ per exact eval | accepted | pre-rejected | sub-chain acc. | wall s3 (s) |
|---|---|---|---|---|---|---|---|---|---|
| `B_pcn_1` | adaptive, target 0.234 | 1 | 0.056 | 0.0012 | 853 | 0.29 | 0.68 | 0.32 | 54 |
| `B_pcn_5` | adaptive | 5 | 0.111 | 0.0046 | 216 | 0.30 | 0.67 | 0.08 | 102 |
| `B_pcn_20` | adaptive | 20 | 0.162 | 0.0098 | 103 | 0.31 | 0.66 | 0.02 | 255 |
| `B_pcn_50` | adaptive | 50 | 0.198 | 0.0158 | 63 | 0.30 | 0.66 | 0.010 | 693 |
| `Q_pcn_200` | adaptive | 200 | 0.253 | 0.0273 | 37 | 0.32 | 0.65 | 0.003 | 944 |
| `Q_pcn_t50_50` | adaptive, target 0.5 | 50 | 0.145 | 0.0139 | 72 | 0.66 | 0.25 | 0.035 | 243 |
| `Q_pcn_b05_50` | fixed β = 0.05 | 50 | 0.05 | 0.0138 | 72 | 0.90 | 0.00 | 0.36 | 200 |
| **`Q_pcn_b10_50`** | **fixed β = 0.10** | 50 | 0.10 | **0.0166** | 60 | 0.89 | 0.02 | 0.11 | 197 |
| `Q_pcn_b20_50` | fixed β = 0.20 | 50 | 0.20 | 0.0151 | 67 | 0.29 | 0.67 | 0.009 | 456 |
| `Q_pcn_b40_50` | fixed β = 0.40 | 50 | 0.40 | **run failed**: stage 2 did not reach 50 000 evaluations in 1 h (sub-chain acceptance ≈ 0, pre-rejection ≈ 1) | | | | | |
| **`Q_pcn_b10_200`** | **fixed β = 0.10** | 200 | 0.10 | **0.0631** | 16 | 0.91 | 0.00 | 0.11 | 356 |
| `B_rw_50` | random walk, adaptive | 50 | – | 0.149 | 6.7 | 0.31 | 0.65 | 0.010 | 1497 |
| `B_rw_100` | random walk, adaptive | 100 | – | 0.180 | 5.6 | 0.32 | 0.65 | 0.005 | 2834 |
| `B_hmc_L10_5` | Hamiltonian L = 10, adaptive ε | 5 | – | 0.260 | 3.9 | 0.90 | 0.00 | 0.77 | 972 |
| **`B_hmc_L30_1`** | **Hamiltonian L = 30, adaptive ε** | 1 | – | **0.604** | 1.7 | 0.72 | 0.19 | 0.81 | 968 |
| `B_hmc_L50_1` | Hamiltonian L = 50 | 1 | – | 0.867 | 1.4 | 0.73 | 0.19 | 0.81 | 1935 |

Reading:

- **The auto-tuner is not the limit at sub-chain 50**: fixed β = 0.10 gives 0.0166 against the
  adaptive 0.0158 (β = 0.198) and 0.0139 at target 0.5 (β = 0.145) — all within the ±10 % noise.
  β = 0.05, 0.10, 0.20 are equivalent at this sub-chain length.
- **The auto-tuner *is* the limit for long sub-chains.** With 200 surrogate steps per exact
  evaluation, the adaptive pCN (β → 0.25, sub-chain acceptance 0.3 %) reaches 0.027, while the
  fixed β = 0.10 (sub-chain acceptance 11 %) reaches **0.063**, 2.3× more, with 91 % exact
  acceptance and no pre-rejection at all. Cause: in a DAMH stage the pCN (and the random walk)
  adapt on the *outer* acceptance probability, which delayed acceptance keeps high whatever the
  sub-chain does; the recursion therefore inflates β until the outer acceptance drops to the
  target, at which point the sub-chain accepts almost nothing and the whole sub-chain amounts to
  one large jump. The right feedback for a sub-chain proposal is the sub-chain acceptance (note
  16's α-scoring item, now measured): with 11 % sub-chain acceptance and 200 steps, the
  sub-chain actually explores the surrogate posterior. The same mechanism limits the random
  walk (sub-chain acceptance 0.5–1 % at 50–100 steps, note 19 §4.2).
- **Even at its best, pCN is 10× behind the Hamiltonian proposal on the same surrogate**
  (0.063 vs 0.60) — and at the same surrogate cost: 200 network values per exact evaluation
  cost 200 × 0.017 ms = 3.4 ms, 30 leapfrog gradients cost 30 × 0.117 ms = 3.5 ms. The
  Hamiltonian proposal uses the surrogate's gradient to move along the posterior's ridges in one
  trajectory; 200 isotropic pCN steps at β = 0.1 make a random walk of 200 × 0.1 = a few prior
  sd in the wide directions and get rejected in the narrow ones. The random walk with the learned
  covariance (0.15–0.18 at 50–100 steps) sits in between.
- Why pCN gains from sub-chains at all: on the surrogate, pre-rejection screens the isotropic
  proposals for free, so the exact model only sees the survivors; the gain per exact evaluation
  grows with the sub-chain (0.0012 → 0.027 adaptive, 0.063 fixed β = 0.1 at 200) but the
  surrogate work grows linearly with it, see §5.
- Posterior accuracy is unaffected in every run (max \|mean − ref\|/sd 0.02–0.08, split-R̂ ≤ 1.05;
  the β = 0.4 run never finished).

## 5. Cost per effective sample, surrogate calls included

Figure: `fig_cost_per_ess.png`. Micro-benchmarked single-point call costs on this machine
(`surrogate_call_costs.json`, one thread, the way the sampler calls them): exact solver 0.22 ms;
network (80, 80) value 0.017 ms, gradient (`vjp`) 0.117 ms; network (128, 128, 128) 0.022 /
0.156 ms; polynomial degree 3 0.10 ms per value. Analytic cost of a stage =
N_exact · t_exact + N_values · t_value + N_gradients · t_vjp, divided by the stage's ESS; three
assumed solver costs. "CPU/ESS" = measured sampler wall time × chains / ESS — the real cost on
this machine, including MPI and Python overhead (which dominates: the analytic surrogate cost
of the HMC stage is 4.4 ms per exact evaluation, the measured overhead 19 ms).

| run (stage 3) | ESS/eval | surrogate work per exact eval | analytic surrogate cost per exact eval (ms) | **cost/ESS, solver 0.2 ms (s)** | **cost/ESS, solver 10 ms (s)** | **cost/ESS, solver 1 s (s)** | measured CPU/ESS (s) |
|---|---|---|---|---|---|---|---|
| `A_rw` (plain MH, best auto-tuned) | 0.0034 | – | 0 | 0.064 | 2.92 | 292 | 0.12 |
| `P_rw_c0.5` (plain MH, best fixed) | 0.0040 | – | 0 | 0.055 | 2.49 | 249 | 0.12 |
| `A_pcn` (plain MH, auto-tuned) | 0.0003 | – | 0 | 0.75 | 34.1 | 3 413 | 1.41 |
| `P_pcn_b10` (plain MH, best fixed) | 0.0004 | – | 0 | 0.61 | 27.8 | 2 780 | 1.06 |
| `B_pcn_50` (adaptive) | 0.0158 | 147 values | 2.5 | 0.172 | 0.79 | 63.4 | 0.88 |
| `Q_pcn_b10_50` (fixed β 0.1) | 0.0166 | 51 values | 0.87 | 0.065 | 0.65 | 60.3 | 0.24 |
| `Q_pcn_200` (adaptive) | 0.0273 | 570 values | 9.7 | 0.364 | 0.72 | 37.0 | 0.69 |
| **`Q_pcn_b10_200`** (fixed β 0.1) | **0.0631** | 200 values | 3.4 | **0.058** | **0.21** | **15.9** | **0.11** |
| `B_rw_20` | 0.115 | 58 values | 0.98 | 0.010 | 0.095 | 8.7 | 0.10 |
| `B_rw_50` | 0.149 | 144 values | 2.4 | 0.018 | 0.084 | 6.7 | 0.20 |
| `B_rw_100` | 0.180 | 286 values | 4.9 | 0.028 | 0.083 | 5.6 | 0.32 |
| `B_hmc_L10_5` | 0.260 | 5 values + 50 gradients | 6.0 | 0.024 | 0.061 | 3.9 | 0.075 |
| **`B_hmc_L30_1`** | **0.604** | 1.2 values + 37 gradients | 4.4 | **0.0076** | **0.024** | **1.66** | **0.032** |
| `B_hmc_L50_1` | 0.867 | 1.2 values + 62 gradients | 7.3 | 0.0087 | 0.020 | 1.16 | 0.045 |
| `C_best_nn128` (HMC L = 30, wider net) | 0.644 | 1.2 values + 37 gradients | 5.8 | 0.0094 | 0.025 | 1.56 | 0.026 |

("values per exact eval" exceed the sub-chain length where pre-rejected proposals — states
without an exact evaluation — carry their own sub-chains.)

- **On every cost scale the ranking is Hamiltonian < random walk < pCN < plain MH**, and the
  gap widens with the solver cost: at 0.2 ms the best pCN scheme costs 8× the best Hamiltonian
  scheme per effective sample, at 10 ms 9×, at 1 s 10×. The best pCN DAMH scheme is 5× cheaper
  than the best plain-MH random walk from a 10 ms solver on (and 6× at 1 s), but 3–6× more
  expensive than random-walk DAMH.
- Surrogate cost is never the decisive term with this network: the analytic surrogate cost per
  exact evaluation stays at 1–10 ms for every scheme, i.e. below the 19 ms of measured Python/MPI
  overhead per exact evaluation and far below a 1 s solver. A gradient (0.117 ms) costs 7 values,
  so 30 gradients ≈ 200 values — the Hamiltonian and the 200-step pCN sub-chain are equally
  priced, and the Hamiltonian returns 10× the effective samples.
- Measured CPU per effective sample agrees with the 0.2 ms column within a factor ≈ 4 (the
  overhead), and preserves the ranking.

## 6. Conclusions

1. **No implementation problem.** pCN proposal, acceptance rule and prior handling are the
   textbook ones; the sampled posteriors match the reference.
2. **No auto-tuning problem in the exact-model stage.** The logit Robbins–Monro reaches any
   target; efficiency is flat for targets 0.1–0.35 and within 20 % of the best fixed β (0.10).
   The random walk's auto-tuning is likewise within 20 % of its best fixed scaling.
3. **pCN is 10–16× less efficient than the covariance random walk on this posterior because
   it is isotropic** — shown on the real problem (11× at the optima), on a Gaussian with the
   same covariance (16×), and explained by the 12× eigen-anisotropy. It won on the rank-2
   problem of note 18 because that posterior *was* the prior in 18 directions. The generalised
   pCN with the learned covariance as reference measure would close the gap; the library has no
   such proposal (design item for `10_manual_review_notes.md`).
4. **One genuine tuning defect, in DAMH sub-chains:** adapting on the outer acceptance drives β
   (and the random walk's scale) up until the sub-chain accepts < 1 %; a fixed β = 0.1 with a
   200-step sub-chain is 2.3× the adaptive pCN. Sub-chain proposals should adapt on the
   sub-chain acceptance. (Follow-up (b) of note 19 §7, now with a measured effect.)
5. **Against the Hamiltonian proposal pCN loses 10× at equal surrogate cost**, on every solver
   cost scale; cost per effective sample favours Hamiltonian DAMH by 8–10× over the best pCN
   DAMH and by 100× over plain MH at a 1 s solver.

Not checked: pCN on a posterior close to the prior in this corrected setup (e.g. noise sd 0.05,
where pCN should win again); a generalised pCN; sub-chain-acceptance adaptation (needs code).

## 7. Files

New runs `P_*` (16, single stage), `Q_*` (7, three stages; `Q_pcn_b40_50` timed out);
`surrogate_call_costs.json`; `R1_cov.npy`; `analyze.py` gained `n_surrogate_values`,
`n_surrogate_gradients`, `surrogate_cost_s`, `cost_per_ess_{0p2ms,10ms,1s}`, `wall_per_ess`;
`plots.py` gained `fig_pcn_stage1.png`, `fig_pcn_stage2.png`, `fig_cost_per_ess.png`;
`report_pcn.html` = this note with figures. Wall clock ≈ 1 h 20 min for the 23 runs. No library
code changed.

---

## 8. Side test (2026-09-30): would a generalised pCN with the learned covariance help?

Author's question after §6: can pCN use the covariance the adaptive random-walk stage builds,
and would it pay? Stand-alone test, **no library change**: `scripts/side_gpcn.py` runs a single
chain per configuration on the real P1 posterior with the exact solver (200 000 evaluations,
first 20 000 dropped, start = a reference posterior state), results in `side_gpcn/`
(`results_seed1.json` = `results_pass1_misscaled_learned.json` for the reference/plain rows, the
"learned" rows of that pass used a 3.5× too small covariance and are discarded;
`results_seed2.json` is the corrected full grid).

Generalised pCN (gpCN; Rudolf & Sprungk 2018, the DILI construction of Cui, Law & Marzouk 2016
when the covariance is `I + low rank`): reference Gaussian N(m, Σ̂), proposal
`y = m + sqrt(1 − β²)(x − m) + β L ξ`, `Σ̂ = L Lᵀ`, acceptance
`log α = [ℓ(y) − ℓ(x)] + [log p₀(y) − log p₀(x)] − [log N(y; m, Σ̂) − log N(x; m, Σ̂)]`
(likelihood ratio plus the prior/reference correction; with Σ̂ = I, m = 0 it is plain pCN).
Two covariances: the reference posterior covariance Σ_ref (`R1_cov.npy`, the best case) and the
one the adaptive random-walk stage `A_rw` actually learned (its carry-over `base_cov × d/2.38²`
and `mean`, the practical case); each used in full and as `I + low rank` (eigen-directions
with sd < 0.9 kept: rank 15 for Σ_ref, 16 for the learned one).

| proposal | setting | acceptance | τ mean / max | **ESS/eval** seed 1 | **ESS/eval** seed 2 | max \|mean−ref\|/sd |
|---|---|---|---|---|---|---|
| plain pCN | β = 0.05 / 0.10 / 0.15 | 0.36 / 0.11 / 0.03 | 2 200–4 700 / up to 11 900 | 0.00032 / 0.00042 / 0.00026 | 0.00021 / 0.00046 / 0.00029 | 0.23–0.44 (ESS ≈ 50) |
| random walk, Σ_ref | c = 0.5 / 1.0 | 0.23 / 0.10 | 178–217 / 229–347 | 0.0046 / 0.0047 | 0.0056 / 0.0048 | 0.06–0.11 |
| random walk, learned Σ̂ | c = 0.5 / 1.0 | 0.21 / 0.09 | 200–230 / 248–295 | – | 0.0050 / 0.0044 | 0.07 |
| gpCN, Σ_ref full | β = 0.2 / 0.4 / 0.6 / 0.8 / 1.0 | 0.61 / 0.36 / 0.22 / 0.15 / 0.11 | 61–242 / 85–693 | 0.0044 / 0.0073 / 0.0088 / 0.0052 / **0.0163** | 0.0041 / 0.0078 / **0.0110** / 0.0087 / **0.0149** | 0.03–0.08 |
| gpCN, Σ_ref `I + rank 15` | β = 0.2 / 0.4 / 0.6 / 0.8 / 1.0 | 0.60 / 0.36 / 0.21 / 0.14 / 0.11 | 64–253 / 79–675 | 0.0040 / **0.0100** / 0.0082 / 0.0095 / **0.0156** | 0.0043 / 0.0093 / 0.0081 / 0.0052 / **0.0157** | 0.03–0.09 |
| gpCN, learned Σ̂ full | β = 0.2 / 0.4 / 0.6 / 0.8 / 1.0 | 0.59 / 0.33 / 0.19 / 0.12 / 0.09 | 101–253 / 117–590 | – | 0.0040 / **0.0099** / 0.0064 / 0.0095 / 0.0051 | 0.05–0.08 |
| gpCN, learned Σ̂ `I + rank 16` | β = 0.2 / 0.4 / 0.6 / 0.8 / 1.0 | 0.59 / 0.33 / 0.20 / 0.12 / 0.09 | 82–223 / 101–603 | – | 0.0045 / 0.0094 / **0.0121** / 0.0070 / 0.0046 | 0.05–0.09 |

Single chains: individual numbers carry ≈ ±30 % (compare the two seeds), τ_max outliers of
300–700 mark a direction where the Gaussian reference fits the posterior poorly and the
near-independence proposal (β ≥ 0.8) occasionally sticks. Reading:

- **Yes, an improvement is to be expected.** With the covariance the adaptive random walk
  actually learned, gpCN at β = 0.4–0.6 gives 0.010–0.012 ESS/eval: **2–2.5× the covariance
  random walk (0.005) and 25× plain pCN (0.0004)**, with the posterior reproduced (mean errors
  ≤ 0.09 sd, as the random walk's). With the exact posterior covariance the gain is 2–3.5×
  over the random walk, up to 0.016 at β = 1.
- The `I + low rank` form loses nothing (0.0121 vs 0.0099 learned; 0.0156 vs 0.0163 reference)
  and is the form that keeps the function-space robustness (prior in the uninformed directions,
  §"dimension robustness" discussion of 2026-09-30), so it is the one to implement.
- **β should stay at 0.4–0.6 with a learned covariance.** β = 1 (independence sampler from the
  Gaussian approximation) is the best setting only when the reference is the true posterior
  covariance; with the learned one it drops to 0.005 and sticks (τ_max 600), because the
  posterior is not Gaussian (note 19 §1: sd along v₁ 0.15 vs Laplace 0.03) and a slightly
  wrong Gaussian is a poor independence proposal. Robbins–Monro on β with a target acceptance
  of ≈ 0.2–0.35 (as today) would land in the right range (β 0.4–0.6 have acceptance 0.33–0.19).
- The learned covariance is good enough: the random walk with it (0.0050) equals the random
  walk with Σ_ref (0.0056 / 0.0046), and gpCN with it reaches 75 % of gpCN with Σ_ref at β ≤ 0.6.
- Expected place in the ranking: still an exact-model (stage-1) improvement of the same order
  as tuning; in DAMH stage 2 it would presumably land near the random-walk numbers
  (0.15–0.18 at sub-chains 50–100) and stay several times below the Hamiltonian proposal
  (0.60); not tested.

**On the spread of τ in the gpCN rows (author's question, 2026-09-30).** Part of the range is
the β dependence (τ falls from ≈ 240 at β = 0.2 to ≈ 60 at β = 1). The rest is the independence-
sampler effect: at large β a proposal is mostly a fresh draw from the reference Gaussian, and
where the posterior has more mass than that Gaussian (it is non-Gaussian, note 19 §1) the chain
sticks until it draws its way out — heavy-tailed holding times, hence τ_max ≫ τ_mean and 4×
seed-to-seed swings of τ_max at β ≥ 0.8 (e.g. 193/693 vs 116/175 for Σ_ref full β = 0.8). The
random walk has no such episodes (its acceptance is a local density ratio). This *is* a warning
sign: τ_mean is optimistic for those runs; the β = 1 independence limit is geometrically ergodic
only if the reference has heavier tails than the posterior everywhere (Mengersen & Tweedie 1996),
which a fitted Gaussian cannot guarantee, and a learned covariance that underestimates the spread
in some direction makes it worse. Guidance for an implementation: keep β adaptive (the current
target lands at 0.4–0.6, where the seeds agree), inflate the low-rank part of the reference by a
safety factor ≈ 1.5 rather than fitting it tightly, and report τ_max in the diagnostics. The
2–2.5× gain above is read from the β 0.4–0.6 rows; the β = 1 rows with Σ_ref are an upper bound.

Implementation sketch (not done): `PCN` gets an optional reference `(mean, cov)`; when the
previous adaptive stage hands over a random-walk covariance the builder uses
`I + low rank(cov)` and the RW mean as the reference, the acceptance adds the prior/reference
log-ratio, β keeps its Robbins–Monro adaptation. About 30 lines plus a posterior-recovery test
(`tests/validation`). Recorded in `10_manual_review_notes.md` §8 (b4).
