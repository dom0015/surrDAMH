# 26 — Hamiltonian proposal in DAMH: the chosen plan (step size, trajectory length, jitter, sub-chain length, momentum) (2026-10-08)

**Design note / plan**, nothing implemented. Written for: the author, as the Hamiltonian part of the
fast mode in note 25 (§4 "fast" column, §7 phase 3). Sources: note 19 §4.2/§4.4 (efficiency study,
GRF problem, network surrogate), note 21 §4, note 25, the theory discussion of 2026-10-08 (chat; only
its conclusions are recorded here), and the check in Appendix A. Code referred to:
`surrDAMH/modules/proposals.py` (`Hamiltonian`, `HamiltonianInfinite`, `_DualAveragingStepSize`),
`surrDAMH/modules/algorithms.py` (`Algorithm_DAMH._propose_new_sample_using_subchain`),
`surrDAMH/proposals.py` (`Hamiltonian` spec), `surrDAMH/stages.py` (`Stage.subchain_length`).
**Addendum 2026-10-09:** §10, robust estimation of the mass matrix (identity plus a low-rank correction,
prior shrinkage, within-chain pooling, clipping, fallbacks); §1, §3.3, §7, §8, §9 updated to point to it.

---

## 1. The plan in one table

| component | setting | status today | section |
|---|---|---|---|
| step size `eps` | dual averaging on the sub-chain (surrogate) acceptance, target 0.8, one update per trajectory, frozen at the stage boundary, carried over | implemented (`_DualAveragingStepSize`, `adapt()` on every sub-chain step) | §2 |
| trajectory length | an integration time `T` per stage, *not* a step count; `T` estimated in the adaptive stage from the U-turn diagnostic (default) or set from the mass matrix (`T = pi` when the mass is the inverse carried covariance); frozen per stage | not implemented (`num_steps` fixed, never tuned) | §3 |
| jitter | a new `L ~ Uniform{1, ..., L_max}`, `L_max = round(T / eps)`, for **every trajectory**, drawn from the proposal's own RNG stream, independent of the state | not implemented | §4 |
| sub-chain length `K` | **deterministic**, set at each stage boundary from measured costs: `K* = argmax_K (1 - rho^K) / (R + K)`, `R = c_exact / c_traj`, `rho = 0.5`, `1 <= K <= 10`; `K = 1` in robust mode and in the first DAMH chunk; user may pin an integer | `Stage.subchain_length: int = 1`, user-given only | §5 |
| momentum | fresh `p ~ N(0, M)` at the start of every trajectory (full refresh), also between the `K` trajectories of a sub-chain | implemented (`propose_sample`) | §6 |
| mass `M` | `M = I + low-rank correction`, the inverse of a regularised posterior covariance estimated at each stage boundary from the outer (exact-posterior) chain states: shrinkage towards the prior `I`, informed eigen-directions only, eigenvalues clipped to `[lambda_floor, 1]`, within-chain pooling, diagonal / previous-`M` fallbacks; frozen per stage. Until implemented: `I` | `I` (`mass=1.0`), not tuned | §10 (and §3.3) |

What is *not* in the plan and why, in one line each: NUTS (variable-cost tree builder, no gain over an
estimated `T` once `L` is jittered, §3.4); persistent or partially refreshed momentum (valid only with a
flip on rejection, lower acceptance than one long trajectory, backtracks after a rejection, §6); a random
sub-chain length (valid but fixes nothing, §5.4); a state-dependent stopping rule such as "stop at the
U-turn" (invalid, drifts, Appendix A).

---

## 2. Step size: dual averaging (unchanged)

`_DualAveragingStepSize` (Hoffman & Gelman 2014 recursion, `gamma = 0.05`, `t0 = 10`, `kappa = 0.75`,
`mu = log(10 eps_0)`, target 0.8) already receives one acceptance probability per trajectory: in a DAMH
sub-chain `Algorithm_DAMH.run` calls `adapt()` with every entry of `subchain_log_acceptance_probabilities`,
and a non-finite trajectory scores 0. Nothing changes here except two consequences of §3–§4:

* **`T` fixed, `eps` moving**: the step count `L_max = round(T / eps)` is recomputed from the *current*
  `eps` whenever the dual averaging changes it (today `num_steps` is a constant and the trajectory length
  `L * eps` drifts with the adaptation; note 25 §2.1 already asks for `T` instead of `num_steps`).
* **Jittered `L` does not bias the recursion**: in the stable regime the leapfrog energy error at the
  endpoint is `O(eps^2)` whatever the length, so the per-trajectory acceptance the recursion sees depends
  on `eps`, not on the drawn `L`. (Checked on the Gaussian of Appendix A: acceptance 1.000 at every `L`.)

The frozen value `log_eps_bar` is what the next stage inherits (carry-over by proposal family, as today).

---

## 3. Trajectory length: integration time `T` from the U-turn

### 3.1 Why a half period, and why not exactly

For a direction of the posterior with standard deviation `sigma` (unit mass) the Hamiltonian flow is
harmonic with period `2 pi sigma`. The displacement from the start, averaged over the start and the
momentum, is `E|q(t) - q(0)|^2 = 2 sigma^2 (1 - cos(t / sigma))`, maximal at the half period
`T = pi sigma`. Beyond it the trajectory comes back. The slowest direction `sigma_max` sets the useful
`T`; faster directions have oscillated and sit at a phase `T / sigma_i` that is deterministic for fixed
`T`, which is the resonance problem (§4). Exactly at `T = pi sigma` a Gaussian direction maps `q -> -q`
and the chain in that direction is period-2, i.e. never explores (Appendix A, row "fixed T"): the mild
form of this is the `L = 50` anticorrelation of note 19 §4.2. Hence `T` at the half period of the slowest
direction, **always with the jitter of §4**.

### 3.2 Estimation of `T` in the adaptive stage: the U-turn diagnostic (default)

For a trajectory started at `(q_0, p_0)` record the first leapfrog index `i` at which

    (q_i - q_0) . M^{-1} p_i < 0          (the momentum points back towards the start)

and the corresponding time `t_u = i * eps`. This is a *statistic*: the proposal still uses the endpoint of
the drawn `L`, so nothing about reversibility or delayed acceptance is touched. Rules:

* Diagnostic trajectories are a fixed subset (every 10th trajectory, chosen by a counter, not by the state);
  on them, if the drawn `L` ends before the U-turn, integration continues *for the diagnostic only* up to
  `L_diag_max = 4 L_max`, the proposal keeps the endpoint at `L`; the extra gradients are at most
  ~10 % × 4 = 40 % of the trajectory cost on 10 % of the trajectories, i.e. ≤ 4 % overall. Censored
  values (no U-turn by `L_diag_max`) are recorded as `L_diag_max * eps`.
* At the stage boundary the `t_u` of all chains are pooled (`Allgather`, like the carried covariance) and
  `T_next = median(t_u)`. The median rather than the mean because the U-turn time of non-Gaussian
  posteriors is heavy-tailed (tails, ridges).
* `T` is frozen within a stage (stage freeze = the diminishing-adaptation device the library already
  uses); the first adaptive Hamiltonian stage starts from `T_0 = pi * sqrt(lambda_max(M^{-1} Sigma_hat))`
  if a carried covariance exists, else from the user's `T` / today's `num_steps * eps`.
* Alternative, not chosen as default: draw `L` from the pooled *empirical* distribution of `i` instead of
  `Uniform{1..L_max}` (eHMC, Wu, Stoehr & Robert 2018). It merges estimation and jitter; it is a
  one-line change once the diagnostic exists, worth a comparison in the validation (§9).

### 3.3 Set from the mass matrix (fallback / cross-check)

With `M = Sigma_hat^{-1}` (the inverse estimated covariance, note 25 phase 3; how it is estimated robustly is
§10) every Gaussian direction has unit
scale and half period `pi`; `T = pi` then needs no estimation and the U-turn diagnostic only corrects for
non-Gaussian shape. Two remarks: (i) `HamiltonianInfinite` (`integrator="dimension_robust"`) solves the
prior + kinetic flow as a rotation that assumes the N(0, I) internal prior with the given mass; a general
`M` makes the rotation angles per eigen-direction of `M` (`q'' = -M^{-1} q`), a contained change to
`_apply_prior_kinetic_flow` but not free; (ii) in that integrator the prior-dominated directions have
half period `pi` by construction and the informed directions turn earlier, so the U-turn diagnostic (§3.2)
is the more general route and is the default.

### 3.4 Alternatives considered

* **ChEES** (Hoffman, Radul & Sountsov 2021): gradient adaptation of `T` across parallel chains, with the
  same `Uniform(0, T]` jitter. Matches NUTS in their benchmarks. Deferred: more code than §3.2 for the same
  quantity; revisit if the median U-turn time proves a poor estimate on non-Gaussian problems.
* **NUTS**: valid as a sub-chain step (the kernel is `pi~`-reversible), but variable cost, a tree builder
  with multinomial/slice selection, and it would have to live inside `_propose_new_sample_using_subchain`.
  Not worth it when §3.2 + §4 deliver the length.

---

## 4. Jitter: a fresh `L` for every trajectory

**Decision: new `L ~ Uniform{1, ..., L_max}` per trajectory**, also for each of the `K` trajectories within
one sub-chain. Not one `L` per sub-chain, not one per stage.

* *Validity*: `L` is drawn from the proposal's RNG stream independently of the state, so one trajectory is
  a mixture of `pi~`-reversible kernels `P_L`, hence reversible; `K` of them is `P_bar^K`, reversible as a
  power of a reversible kernel. `Q(y -> x) / Q(x -> y) = pi~(y) / pi~(x)` holds and the telescoped
  correction is unchanged. (One `L` per sub-chain would be a mixture over `L` of `P_L^K`, also reversible;
  validity does not decide this.)
* *Why per trajectory*: a constant `L` within a sub-chain reproduces the resonance inside the sub-chain.
  At the half period a Gaussian direction maps `q -> -q`, so a sub-chain of even `K` with the same `L`
  returns the slow coordinate to where it started; more generally the `K` endpoints share one phase set and
  the sub-chain explores a lattice of phases, not the surrogate posterior. A fresh `L` per trajectory
  randomises every phase and costs nothing.
* *Why `Uniform{1..L_max}` and not `L_max * (1 ± 0.2)`*: for the slow direction the correlation of the
  endpoint with the start is `E cos(T_rand / sigma)`, which is 0 for `T_rand ~ Uniform(0, pi sigma]` and
  ≈ −0.9 for the narrow jitter (antithetic, which looks good for means but keeps a deterministic phase for
  the directions with `sigma_max / sigma_i` between 1 and ~5). The uniform jitter also halves the mean
  gradient cost per trajectory (`L_bar = (L_max + 1) / 2`), which enters `c_traj` in §5.
* *Reproducibility*: the draw consumes the proposal's generator, so a run is still fixed by its seed; the
  manifest records `T` and `L_max` per stage.

---

## 5. Sub-chain length `K`: by cost ratio, deterministic

### 5.1 Model

Per outer DAMH iteration the chain pays `c_exact * (1 - r_pre) + K * c_traj`, with `c_traj = L_bar * c_grad
+ c_surr_overhead` and the pre-rejection rate `r_pre ≈ 0` for `K >= 2` (note 19: 0.19 at `K = 1`, 0.00 at
`K = 5`). The gain is ESS per outer iteration `e(K) = e_inf * (1 - rho^K)`: a concave curve with the
ceiling `e_inf` of the independence sampler with proposal `pi~` (reached as `K -> inf`; it depends only on
the surrogate quality through `w = pi / pi~`, ≈ 0.89 on the note-19 GRF problem, where `K = 1` already
reaches 0.60 and `K = 5` 0.72). `rho` is the correlation left after one jittered U-turn-length trajectory
plus its rejection probability; with target acceptance 0.8 it is at least 0.2, and `rho = 0.5` is the
conservative default. Efficiency per wall-second is then proportional to

    f(K) = (1 - rho^K) / (R + K),      R = c_exact / c_traj,

and `K* = argmax_{1 <= K <= K_max} f(K)`, evaluated numerically (ten values). For `rho = 0.5`:

| `R = c_exact / c_traj` | 0.06 (note-19 GRF: 0.2 ms solver, 30 × 0.117 ms gradients) | 1 | 10 | 100 | 1000 |
|---|---|---|---|---|---|
| `K*` | 1 | 1–2 | 3–4 | 6 | 9–10 |

The note-19 wall-clock ranking (`K = 1` best at 0.2 ms) is the first column; a PDE solver at seconds per
evaluation is the last, where `K = 10` costs nothing visible and takes the chain to the surrogate ceiling.
`K_max = 10`: beyond it `1 - rho^K` is within 0.1 % of the ceiling for `rho = 0.5`, and long sub-chains
trade away the robustness of local moves (a fully `pi~`-decorrelated proposal has the independence-sampler
acceptance `E min(1, w(y) / w(x))`, which collapses where `log w` varies by more than O(1); a local move
has `w(y) / w(x) ≈ 1` wherever the surrogate error is smooth).

### 5.2 User-given, measured or estimated?

* **Measured costs, model `rho`, user override.** `Stage.subchain_length: int | "auto"`; `"auto"` is the
  default in fast mode, `1` in robust mode (note 25 §4), an integer pins it as today.
* `c_exact` = the sampler's mean wait for an exact evaluation (send to receive, i.e. what the chain
  actually pays, queueing included) over the previous stage; `c_traj` = the sampler's mean wall time of one
  sub-chain trajectory over the previous *DAMH* stage. Both are pooled across chains at the stage boundary
  with the other carried statistics and written to the manifest with the resolved `K`.
* `rho` is **not** measured by default (its estimate from the adaptive stage, the lag-1 autocorrelation of
  consecutive sub-chain states projected on the leading eigenvector of `Sigma_hat`, is cheap but noisy and
  only shifts `K*` by ±1 at the `R` values that matter); it is a module constant, exposed for study only.
* `K` is set **only at stage boundaries** from the previous stage's timings. In the note-25 chunked layout
  the first DAMH chunk runs `K = 1` (robust; it also measures `c_traj`), and every later chunk uses `K*`.
  Changing `K` inside a stage would be a kernel change mid-chain; a one-off change at a fixed iteration
  is harmless in burn-in but has no place in a production stage.

### 5.3 What the user sees

Printed at the start of each DAMH stage, and in the manifest:
`subchain_length=auto -> K=6 (c_exact=2.1 s, c_traj=4.3 ms, R=490, rho=0.5)`.

### 5.4 Should `K` be randomised? No.

A random `K` drawn independently of the state is valid (a mixture over `K` of `P_bar^K`, each
`pi~`-reversible), so nothing forbids it; but there is no mechanism it repairs. The resonance that
motivates jitter lives in the trajectory phase and is already broken by the fresh `L` and the fresh
momentum of every trajectory; consecutive sub-chains are joined by the exact accept/reject, not by a
deterministic map. A random `K` would only make the cost per exact evaluation variable and the
pre-rejection statistics harder to read. Keep `K` deterministic. (Note 25 §2 lists "randomised sub-chain
length" among the deferred items; this closes it for the Hamiltonian proposal. For random-walk
sub-chains the same argument applies: the proposal noise already randomises each step.)

---

## 6. Momentum: full refresh, also between the trajectories of a sub-chain

A fresh `p ~ N(0, M)` at the start of every trajectory (today's `propose_sample`). The alternatives were
examined and rejected:

* **Carry the momentum through the sub-chain, flip it on rejection** (Horowitz 1991 inside the sub-chain,
  any partial-refresh angle). Valid as a DAMH proposal: each step is skew-reversible under the momentum
  flip `F`, the sequence "fresh draw, `K` steps, forget `p`" is a palindrome under the `F`-adjoint, and
  the fresh draw and the marginalisation are adjoints of each other, so the position kernel is
  `pi~`-reversible and the telescoped correction (likelihood parts only, no kinetic terms) is unchanged.
  When every segment is accepted it *is* the `K L`-step trajectory; otherwise it is worse than one long
  trajectory: the segmented acceptance `prod_i min(1, e^{-dH_i}) <= min(1, e^{-sum_i dH_i})` penalises the
  oscillating leapfrog energy error at every checkpoint, and a rejection reverses the direction so the
  sub-chain retraces its own path. The one gain (a late divergence keeps the progress up to the previous
  checkpoint) does not pay for this.
* **Carry the momentum, no flip on rejection.** Invalid (the proposal map is no longer an involution, the
  reverse move has zero density, and the same point is re-proposed after a rejection).
* **Carry the momentum across outer exact steps.** Would need the flip on exact rejection as well and
  backtracks at the 10–30 % exact rejection rate. Excluded.

---

## 7. Validity conditions the plan relies on (checklist for the implementation)

1. Every random choice of the proposal (`p`, `L`) is drawn independently of the state, from the proposal's
   generator.
2. `eps`, `T`, `L_max`, `M`, `K` are constant within a production stage; they change only at stage
   boundaries (stage freeze) or, for `eps` and hence `L_max`, under the dual averaging of an adaptive stage.
3. The surrogate evaluator is refreshed once per sub-chain, before the first trajectory (finding 1.2,
   unchanged).
4. The U-turn diagnostic never changes the proposal's endpoint; its extra integration is discarded.
5. The DA correction stays the telescoped sum of accepted surrogate likelihood ratios; kinetic terms stay in
   the "prior part" of `get_log_acceptance_probability` and never enter the correction.
6. The mass matrix is estimated only from samples of *previous* stages (outer chain states, §10.3) and is
   symmetric positive definite by construction (eigenvalues clipped to `[lambda_floor, 1]`, §10.1).

---

## 8. Interface and size (estimate)

| change | where | lines (incl. tests) |
|---|---|---|
| `Hamiltonian(T=..., num_steps=None, jitter=True)`: `T` replaces `num_steps` as the primary knob (`num_steps` kept as "fixed length, no jitter" for experiments, mutually exclusive with `T`); `L_max = round(T / eps)` recomputed on every `eps` change | `proposals.py` spec, `modules/proposals.py` (`Hamiltonian`, `HamiltonianInfinite`, `_DualAveragingStepSize`) | ~80 |
| per-trajectory `L` draw from the proposal RNG; manifest fields `T`, `L_max` | `modules/proposals.py`, `modules/manifest.py` | ~30 |
| U-turn diagnostic on every 10th trajectory, pooled at the stage boundary, `T_next = median`, carried over | `modules/proposals.py` (`adapted_state`/`adapted_summary`), `process_SAMPLER.py` carry-over, `stages.py` | ~120 |
| `Stage.subchain_length: int | "auto"`, sampler-side timing of exact waits and trajectories, pooling, `K*` rule, print + manifest | `stages.py`, `modules/algorithms.py`, `process_SAMPLER.py`, `modules/manifest.py` | ~150 |
| `Hamiltonian(mass="auto")`: robust mass estimation at the stage boundary (§10: shrinkage, eigen-truncation, clipping, within-chain pooling, fallbacks, dual-averaging restart), manifest fields | `modules/proposals.py`, `process_SAMPLER.py` carry-over, `modules/manifest.py` | ~150 |
| `HamiltonianInfinite` with a non-identity mass: exact prior + kinetic rotation per eigen-direction of `M` (§10.1) | `modules/proposals.py` (`_apply_prior_kinetic_flow`) | ~60 |
| total | | ~590 |

Behaviour-change evidence required (CLAUDE.md): with `Hamiltonian(num_steps=L, jitter=False)` and
`subchain_length=K` the run must stay byte-identical to today; every default change (`T`, jitter, `"auto"`)
is listed in `CHANGELOG.md` with the note-19 scheme re-run before and after.

---

## 9. Validation before adopting the defaults

On the note-19 GRF set-up (network surrogate, 0.2 ms solver) and on one case with an artificially slow
solver (sleep to `R ≈ 100`), stage 3 ESS per exact evaluation (mean and min over parameters) and per
wall-second, three repetitions each:

1. `L = 30` fixed (today's winner) vs `T = 30 eps` with `Uniform{1..L_max}` jitter at `K = 1`: the jitter
   must remove the `L = 50`-type anticorrelation/unevenness without losing the mean.
2. `T` from the U-turn median vs `T = 30 eps` vs eHMC-style empirical `L`.
3. `K = 1, 3, 6, 10` at `R ≈ 0.06` and `R ≈ 100` against the §5.1 prediction; `rho` estimated from the
   runs to check the 0.5 default.
4. One non-Gaussian problem (the banana/mixture toy of `toy_examples/`) for the median U-turn estimate.
5. Mass matrix (§10): `I` vs dense `Sigma_hat^{-1}` vs the §10 low-rank estimate on the GRF set-up, with the
   estimate made from 10², 10³ and 10⁴ effective samples (the dense inverse is expected to degrade first);
   within-chain vs total pooling on a two-mode toy with chains started in both modes.

---

## 10. Mass matrix: robust estimation (added 2026-10-09)

The plain inverse sample covariance is right for a small, well-sampled Gaussian (the 2-D demo
`toy_examples/hmc_illustration_2d.ipynb`: identity mass 20.5, estimated mass 51.7 minimum ESS per 1 000
gradients, one seed, with the trajectory length also changed from 3 to `pi`) but fragile in a real problem:
it needs O(d²) well-mixed samples, it breaks the prior structure the dimension-robust integrator relies on,
and it is distorted by stuck chains and by chains in different modes. The rules below make it robust. All
numerical thresholds are placeholders for the validation of §9 item 5.

### 10.1 Form: identity plus a low-rank correction

In surrDAMH's internal coordinates the prior is N(0, I). For a near-Gaussian posterior the covariance is
`Sigma ≈ (I + H)^{-1}` with `H` the data-misfit Hessian, and the data inform only a few directions (rank
`r << d`). The mass matrix should therefore equal `I` (the prior) in the uninformed directions and be
stiffer only in the informed ones:

    M = V diag(1 / lambda_tilde) V^T = I + V_r diag(1 / lambda_i - 1) V_r^T

Estimator, from the pooled covariance `S` of §10.2 with `n_eff` effective samples:

1. **Shrink towards the prior**: `S_shr = (1 - a) S + a I`, `a` from the Ledoit–Wolf formula with the
   identity as target. The prior is the natural target here, not the scaled identity of the textbook
   version.
2. **Eigendecompose** `S_shr = V diag(lambda) V^T`.
3. **Keep only the informed directions**: direction `i` is informed iff
   `lambda_i < (1 - sqrt(d / n_eff))² * (1 - c * sqrt(2 / n_eff))`, `c = 3`. The first factor is the
   Marchenko–Pastur lower edge: with few samples the smallest eigenvalues of a noisy estimate of the
   identity fall well below 1 by chance, and without this test pure-prior directions would be mistaken
   for informed ones. Every other eigenvalue is set to exactly 1.
4. **Clip** to `[lambda_floor, 1]`, `lambda_floor = 1e-4`. The cap at 1 encodes "the posterior is not wider
   than the prior"; it is a heuristic (a nonlinear posterior can be wider in a direction) and costs only
   efficiency, never exactness. The floor stops a nearly singular estimate producing a huge mass and frozen
   trajectories.
5. **Cap the rank**: `r <= n_eff / 10`, keeping the smallest eigenvalues.

Consequences:

* Directions the samples cannot resolve fall back to the correct prior scale instead of to noise.
* The sample requirement scales with `r`, not with `d²`, so the estimate is usable on the GRF problems with
  hundreds of coefficients.
* The dimension-robust integrator (`HamiltonianInfinite`) stays valid. Its prior + kinetic flow
  `q'' = -M^{-1} q` remains an exact rotation in the eigenbasis of `M`, with angular frequency
  `sqrt(lambda_tilde_i)` per direction; in the uninformed directions it is today's unit-frequency rotation.
  This is the contained change to `_apply_prior_kinetic_flow` that §3.3 mentions. A dense estimated inverse
  would put estimation noise into thousands of irrelevant directions of that rotation.

### 10.2 Pooling across chains

* **Within-chain pooling for the Hamiltonian**: `S = W`, the `n_eff`-weighted average of the per-chain
  covariances (the `W` of R-hat), not the total covariance of all chains. With chains in nearby modes the
  total covariance contains the between-mode spread, which makes `M` far too soft inside each mode, and a
  Hamiltonian trajectory does not cross the barrier anyway. The random-walk component of the robust mode
  and the exact safety kernel S2 (note 25) keep using the *total* covariance, which spans the modes.
* **Exclude stuck chains**: chains with acceptance below 0.05 in the stage, or flagged by the stagnation
  check of S5 (note 25), are left out. Their near-zero spread would shrink `S`.
* **Mode indicator**: if the trace of the between-chain part exceeds that of `W` (an R-hat-like ratio), the
  stage summary reports "chains in different modes"; the Hamiltonian still uses `W`.

### 10.3 Which samples

* Only **outer DAMH chain states**, which follow the exact posterior `pi`, including the repeats of rejected
  iterations (multiplicity is part of the MCMC estimator; with the library's `weighting` option, use the
  weighted covariance). Never surrogate sub-chain states (they follow `pi~`) and never pre-rejected
  proposals.
* The **stage just finished**, after its first 10 % as burn-in. Adding earlier stages is allowed when their
  between-stage difference is small, but it is not the default, because early stages carry burn-in bias.
* `n_eff` = total kept iterations divided by the largest integrated autocorrelation time over the
  parameters (conservative; the projection on the leading eigenvectors is a cheaper alternative).

### 10.4 Fallbacks

| available `n_eff` | mass used for the next stage |
|---|---|
| `>= 10 r` for the detected rank `r` | the §10.1 low-rank estimate |
| enough for per-parameter variances (`>= 50`) but not for the above | diagonal: per-parameter variances, shrunk towards 1 and clipped to `[lambda_floor, 1]` |
| `< 50`, or every chain excluded | the previous stage's `M` (`I` at the first Hamiltonian stage) |

The manifest records which row applied, `r`, the kept eigenvalues, `a`, `n_eff` and the excluded chains.

### 10.5 Schedule and interaction with the other knobs

1. Re-estimate `M` at every stage boundary that ends a stage with outer chain states (each DAMH chunk of
   note 25 §5). Freeze it inside the stage.
2. After a change of `M` **restart the dual averaging** of `eps` (§2) with `mu = log(10 eps_previous)`; the
   optimal step changes with the mass.
3. `T`: after whitening the half period is about `pi` in every direction (§3.3); the U-turn diagnostic (§3.2)
   remains the default and corrects for non-Gaussian shape. `L` jitter (§4) unchanged.
4. User control: `Hamiltonian(mass="auto")` is the fast-mode default; a scalar, vector or matrix pins it as
   today.

### 10.6 Validity

`M` enters only the proposal. Any symmetric positive definite `M` that is fixed within a stage gives an
exact HMC kernel; building it from previous stages' samples is the same stage-boundary adaptation the
library already uses for the carried covariance. In DAMH the kinetic terms stay in the prior part of the
sub-chain acceptance and never enter the correction (§7 items 5–6).

### 10.7 What no constant mass can fix

With position-dependent curvature (banana, funnel, strongly nonlinear forward maps) any constant `M` is a
compromise, and `eps` is limited by the tightest region. The robust answer is not a cleverer mass but the
exact random-walk kernel S2 and the diagnostics S5 of note 25, with the self-demotion to the robust mode.
Riemannian-manifold HMC (position-dependent `M`, implicit integrator) is excluded as a default: it needs
the Hessian or Fisher metric of the surrogate, costs a linear solve per step, and is fragile.

### 10.8 Open decisions for the author

* **MM1** Within-chain pooling for the Hamiltonian while the robust components use the total covariance
  (§10.2) — or one covariance for everything?
* **MM2** Cap the eigenvalues at 1 (§10.1 step 4), or allow a posterior wider than the prior?
* **MM3** Estimate from the last stage only (§10.3), or accumulate stages after warm-up?

---

## Appendix A — the fixed-`T` / jitter / state-dependent-stop check (2026-10-08)

2-D Gaussian, standard deviations (3, 1), unit mass, leapfrog `eps = 0.05`, 40 000 iterations, first
5 000 discarded; `L_half = round(pi * 3 / 0.05) = 188`. Script reproduced below (was run from the session
scratchpad with `/dolfinx-env/bin/python`).

| scheme | sd of slow coordinate (true 3) | sd of fast coordinate (true 1) | mean kinetic energy at the endpoint (true 1) | acceptance |
|---|---|---|---|---|
| fixed `L = L_half` | 2.04 | 1.06 | 1.00 | 1.000 |
| `L ~ Uniform{1..L_half}` per trajectory | 2.99 | 1.00 | 1.00 | 1.000 |
| stop at the first U-turn, endpoint MH-accepted | 286 | 1.30 | 0.90 | 1.000 |

Reading: fixed `T` at the half period is period-2 in the slow direction (it moves only through the
discretisation error); the jittered chain is exact in both marginals and in the endpoint kinetic energy
(there is no "stops where it loses kinetic energy" bias: the flow preserves `pi~(q) N(p)` at any
state-independent length); a state-dependent stopping rule is not biased by a little but drifts without
bound, because every step lands at the farthest point of its trajectory and no reverse move exists.

```python
import numpy as np
rng = np.random.default_rng(0)
sig = np.array([3.0, 1.0]); eps = 0.05
grad = lambda q: q / sig**2
U = lambda q: 0.5 * np.sum(q**2 / sig**2)
def leap(q, p, L):
    p = p - 0.5*eps*grad(q)
    for i in range(L):
        q = q + eps*p
        if i < L-1: p = p - eps*grad(q)
    p = p - 0.5*eps*grad(q)
    return q, p
def leap_uturn(q0, p, Lmax=2000):
    q = q0.copy(); p = p - 0.5*eps*grad(q)
    for i in range(Lmax):
        q = q + eps*p
        p_full = p - 0.5*eps*grad(q)
        if np.dot(q - q0, p_full) < 0:        # first U-turn: stop here (INVALID as a proposal)
            return q, p_full
        p = p - eps*grad(q)
    return q, p + 0.5*eps*grad(q)
def run(mode, n=40000):
    q = np.zeros(2); out = []; Kend = []; acc = 0
    Lhalf = int(round(np.pi*sig.max()/eps))   # half period of the slow direction
    for _ in range(n):
        p = rng.standard_normal(2); H0 = U(q) + 0.5*p@p
        if mode == "fixed_T": qn, pn = leap(q, p, Lhalf)
        elif mode == "jitter": qn, pn = leap(q, p, rng.integers(1, Lhalf+1))
        else: qn, pn = leap_uturn(q, p)
        H1 = U(qn) + 0.5*pn@pn
        if np.log(rng.random()) < H0 - H1: q, p = qn, pn; acc += 1
        out.append(q.copy()); Kend.append(0.5*p@p)
    out = np.array(out)[5000:]
    return out.std(0), np.mean(Kend[5000:]), acc/n
for m in ["fixed_T", "jitter", "stop_at_uturn"]:
    sd, K, a = run(m); print(m, sd, K, a)
```
