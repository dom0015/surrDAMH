# 22 — The surrogate blind spot: a network trained away from the high-posterior region, and how to let the chain find it (2026-10-06)

**Design note**, not a record of implemented work. Setting: a `NeuralNetworkUpdater` surrogate trained
on-the-fly from the chains' exact evaluations (snapshots), and, from the first trained network on, a
`Hamiltonian` proposal driven by the network's autograd gradient inside DAMH (`Stage(algorithm="DAMH",
proposal=Hamiltonian(...))`). The question, raised by the author on 2026-10-01: *can the first surrogate
be uninformed exactly where the posterior is high, make the posterior look low there, so that proposals
into that region are rejected in the first (surrogate) stage, the exact model is never evaluated there,
and the surrogate therefore never gets the data to correct itself?* Short answer: yes; §1 says why, §2
what the library does and does not do about it today, §3 where the literature treats it, §4 what to
implement. A standalone 2-D demonstration is `toy_examples/hmc_surrogate_error_model_2d.ipynb`.

Nothing here has been implemented or measured on the library's problems; the numbers quoted come from the
notebook and from note 19.

---

## 1. The problem

### 1.1 Mechanism

Notation as in `docs/concepts.md`: internal state `u ~ N(0, I)` a priori, exact model `G`, surrogate
`G̃`, likelihood `L(y − G(u))` with noise covariance `Σ_noise`, surrogate likelihood `L̃(u) = L(y − G̃(u))`,
surrogate posterior `π̃ ∝ L̃ · prior`. One DAMH iteration from the current state `x`:

1. a proposal `y*` is drawn (here: the end point of a Hamiltonian trajectory integrated in the potential
   `Ũ(u) = −log L̃(u) + ½|u|²`) and accepted or rejected **against `π̃`** with probability
   `α₁ = min{1, π̃(y*) q(y*→x) / (π̃(x) q(x→y*))}` (for HMC: `exp(H(x,p) − H(y*,p*))`);
2. only if accepted, `G(y*)` is evaluated and the proposal is accepted with
   `α₂ = min{1, [L(y*)/L(x)] / [L̃(y*)/L̃(x)]}`.

If `G̃` is wrong in a region `R` in the direction that makes `L̃` *small* there (the network's extrapolated
output is far from `y`), then for `y* ∈ R` the first-stage ratio carries a factor of order

    exp( −½ |G(y*) − G̃(y*)|²_{Σ_noise⁻¹} )   (relative to a correct surrogate),

so `α₁ ≈ 0`, step 2 never runs, no snapshot from `R` reaches the collector, the network keeps extrapolating
the same way in `R`, and the chain never learns that `R` is where the posterior mass is. Theoretically the
chain is still ergodic — `α₁ > 0` whenever `L̃(y*) > 0`, and once a proposal passes, `α₂ ≥ 1` in `R` — but
the expected waiting time is the inverse of that factor. With the test problem's noise sd 0.01 and the
first network's typical error 0.1 (note 19, `fig_surrogate_quality.png`, leftmost point), the factor is
`exp(−½ (0.1/0.01)²) = e⁻⁵⁰`. "Never" in practice.

### 1.2 Why it does not fix itself: the asymmetry

The surrogate's errors are corrected by the data the chain generates, and the chain generates data only
where it evaluates the exact model. Two cases:

- `G̃` *overestimates* the posterior in `R` (optimistic error): proposals into `R` pass stage 1, `G` is
  evaluated there, stage 2 mostly rejects them, but every evaluation is a snapshot (rejected proposals are
  trained on with multiplicity 0; `weighting="uniform"`). The network is corrected in `R` and the optimism
  disappears.
- `G̃` *underestimates* the posterior in `R` (pessimistic error): nothing is evaluated in `R`, nothing is
  learned, the pessimism stays.

Training data therefore accumulate only where the surrogate is optimistic or where the chain already is.
This is the one direction of surrogate error that on-the-fly training cannot repair by itself. The sign of
a network's extrapolation error is essentially random (the notebook's first attempts with a real MLP
ensemble happened to extrapolate optimistically twice and no trap appeared), so in a given run the
blind spot either exists or not, and nothing in the output tells which.

### 1.3 Why the Hamiltonian proposal makes it worse, not better

A random-walk proposal is blind to the surrogate: it proposes isotropically, so `R` is proposed into at a
rate set by its distance and the step; only stage 1 blocks it. The Hamiltonian trajectory is *driven by*
`∇Ũ`, i.e. by the surrogate's landscape: where the surrogate is flat or low the trajectory is not
attracted, and where the surrogate has spurious maxima (a network extrapolating a smooth function
produces rugged outputs: see the notebook's surrogate-posterior panel, a lattice of narrow spikes with the
true mode between them) the trajectory is attracted to those. With the `dimension_robust` integrator
the only surrogate-independent component of the motion is the prior rotation, which does move the chain on
the prior scale; but the stage-1 test then still rejects the end point if it lies in `R`.

Note 19 did not observe the blind spot (all schemes matched the reference posterior) because its warm-up
(20 000 adaptive random-walk evaluations per chain on the exact model) covered the posterior region before
any surrogate was used. That is a property of that run, not of the algorithm.

### 1.4 When to expect it

- A short warm-up, or a warm-up started far from the posterior mass (prior-drawn initial sample in a
  high-dimensional problem with an informative likelihood) that ends before the chain converges.
- A multimodal posterior where the warm-up found one mode: the surrogate is then confident around that
  mode only. (Escaping a genuine mode is also hard for exact HMC; the surrogate adds a second barrier.)
- `d` large relative to the number of snapshots: the network interpolates well inside the convex hull of
  the data and arbitrarily outside; the posterior region needs to lie inside that hull.
- A layout that freezes the surrogate early (`surrogate_model_updates=False`): the blind spot is then
  permanent for the stage, even for a random-walk proposal.

---

## 2. What the library does today

Protects:
- The MH warm-up runs on the exact model; all its evaluations (accepted and rejected) are snapshots.
  **This is the main protection**, and it is only as good as the warm-up's convergence.
- In DAMH stages, proposals rejected in stage 2 are snapshots too (the exact model was called).
- The held-out test set (`TestData`, prior draws evaluated exactly; `surrogate_quality_test.csv`) reports
  the surrogate's RMSE weighted by the *exact* posterior, which would expose a bad surrogate in `R` — if
  some of the 512 prior draws fall into `R`, which in 20+ dimensions they need not.

Does not protect:
- A pre-rejected proposal is never evaluated, so it is never a snapshot. This is the gap of §1.1.
- `raw_data` holds `obs` and `obs_approx` only for proposals that passed stage 1, i.e. a sample biased in
  favour of the surrogate; the blind spot is invisible in it.
- No diagnostic distinguishes "the posterior is here" from "the surrogate says the posterior is here".

---

## 3. Literature

The blind spot is the known weak point of delayed acceptance and of adaptive surrogate MCMC in general;
the remedies below are standard ingredients, not new ideas. Bibliographic details to be checked before
citing in a paper.

Delayed acceptance and its dependence on the approximation
- Christen, J. A., Fox, C. (2005). Markov chain Monte Carlo using an approximation. *J. Comput. Graph.
  Statist.* 14(4), 795–810. The original DA paper; the approximation affects efficiency only, and a poor
  approximation makes the chain slow — the blind spot is the extreme of "slow".
- Banterle, M., Grazian, C., Lee, A., Robert, C. P. (2019). Accelerating Metropolis–Hastings algorithms by
  delayed acceptance. *Found. Data Sci.* 1(2), 103–128. Spectral-gap bounds for DA in terms of the first-stage
  kernel; shows how a first stage that rejects too much degrades the whole chain.
- Sherlock, C., Golightly, A., Henderson, D. A. (2017). Adaptive, delayed-acceptance MCMC for targets with
  expensive likelihoods. *J. Comput. Graph. Statist.* 26(2), 434–444. Adaptive DA with a kNN surrogate
  updated from the chain's own evaluations; the convergence argument surrDAMH's note 18/20 relies on.
- Quiroz, M., Tran, M.-N., Villani, M., Kohn, R. (2018). Speeding up MCMC by delayed acceptance and data
  subsampling. *J. Comput. Graph. Statist.* 27(1), 12–22. DA with a cheap surrogate likelihood; discusses
  the trade-off between first-stage selectivity and total cost.

Local approximations refined where the chain needs them
- Conrad, P. R., Marzouk, Y. M., Pillai, N. S., Smith, A. (2016). Accelerating asymptotically exact MCMC for
  computationally intensive models via local approximations. *J. Amer. Statist. Assoc.* 111(516), 1591–1607.
  The closest treatment of the present problem: the surrogate is refined (a new exact evaluation is
  requested) when a cross-validation error indicator at the *proposed* point is large **and also at random
  with a probability `β_t` decaying in `t`**; the random refinement is exactly what guarantees that regions
  the surrogate underrates are eventually evaluated, and the decay is what keeps the chain asymptotically
  exact. Proposal P1 below is this idea transplanted to the collector.
- Conrad, P. R., Davis, A. D., Marzouk, Y. M., Pillai, N. S., Smith, A. (2018). Parallel local approximation
  MCMC for expensive models. *SIAM/ASA J. Uncertain. Quantif.* 6(1), 339–373. Same framework with shared
  evaluations across parallel chains — structurally the collector of surrDAMH.

Approximation (model) error in the likelihood
- Kaipio, J., Somersalo, E. (2005). *Statistical and Computational Inverse Problems*. Springer. Ch. 7, the
  approximation error approach: `y = G̃(u) + ε(u) + noise`, `ε` modelled as Gaussian with empirical mean
  and covariance, folded into the likelihood.
- Kaipio, J., Somersalo, E. (2007). Statistical inverse problems: discretization, model reduction and inverse
  crimes. *J. Comput. Appl. Math.* 198(2), 493–504.
- Arridge, S. R., Kaipio, J. P., Kolehmainen, V., Schweiger, M., Somersalo, E., Tarvainen, T., Vauhkonen, M.
  (2006). Approximation errors and model reduction with an application in optical diffusion tomography.
  *Inverse Problems* 22(1), 175–195.
- Cui, T., Fox, C., O'Sullivan, M. J. (2011). Bayesian calibration of a large-scale geothermal reservoir model
  by a new adaptive delayed acceptance Metropolis Hastings algorithm. *Water Resour. Res.* 47, W10521. DA
  with an approximation error model whose mean and covariance are **updated from the chain's own
  (exact − surrogate) residuals during sampling** — the "enhanced error model". Proposal P3 below.
- Cui, T., Fox, C., O'Sullivan, M. J. (2019). A posteriori stochastic correction of reduced models in
  delayed-acceptance MCMC, with application to multiphase subsurface inverse problems. *Int. J. Numer.
  Methods Eng.* 118(10), 578–605.
- Lykkegaard, M. B., Dodwell, T. J., Fox, C., Mingas, G., Scheichl, R. (2023). Multilevel delayed acceptance
  MCMC. *SIAM/ASA J. Uncertain. Quantif.* 11(1), 1–30. Uses the adaptive error model of Cui et al. at every
  level of a multilevel DA hierarchy; the implementation discussion (how the error statistics are
  accumulated and when they are updated) is directly transferable.

Uncertainty of a neural-network surrogate (for the local error model)
- Lakshminarayanan, B., Pritzel, A., Blundell, C. (2017). Simple and scalable predictive uncertainty
  estimation using deep ensembles. *NeurIPS* 30. Ensemble spread as predictive uncertainty; grows away
  from the data, but is known to be miscalibrated.
- Kuleshov, V., Fenner, N., Ermon, S. (2018). Accurate uncertainties for deep learning using calibrated
  regression. *ICML*. Post-hoc calibration of regression uncertainties on a held-out set (the notebook's
  calibration factor of ≈ 20 on the raw ensemble variance is an instance of why this is needed).

Validity of the proposed kernels
- Tierney, L. (1994). Markov chains for exploring posterior distributions. *Ann. Statist.* 22(4), 1701–1762.
  Mixtures and cycles of `π`-invariant kernels are `π`-invariant (proposal P2).
- Roberts, G. O., Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC. *J. Appl. Probab.* 44(2),
  458–475. Diminishing adaptation + containment; an error model whose parameters are re-estimated during
  the run is one more adaptive component and falls under the same conditions (see note 18, A5, and note 20).
- Beskos, A., Pinski, F. J., Sanz-Serna, J. M., Stuart, A. M. (2011). Hybrid Monte Carlo on Hilbert spaces.
  *Stoch. Process. Appl.* 121(10), 2201–2230. The `dimension_robust` integrator; its exact prior rotation is
  what turns the Hamiltonian proposal into a prior-like move where the likelihood force is damped (§4.3).

Overview
- Peherstorfer, B., Willcox, K., Gunzburger, M. (2018). Survey of multifidelity methods in uncertainty
  propagation, inference, and optimization. *SIAM Rev.* 60(3), 550–591. §5 covers DA and model adaptation.

---

## 4. Proposals for surrDAMH

Ordered by implementation effort. P1 and P3 are the recommended first steps; they are independent and
complementary (P1 generates data where the surrogate is pessimistic, P3 lets proposals get there).

### P1 — Audit of pre-rejected proposals (snapshot only, chain law unchanged)

**Mechanism.** In a DAMH stage with `surrogate_model_updates=True`, after a proposal has been pre-rejected
in stage 1, evaluate the exact model at it anyway with probability `p_audit` and send `(y*, G(y*))` to the
collector as a snapshot with multiplicity 0. The chain's decision is **not** revisited: the proposal stays
rejected. The evaluation serves training only.

**Validity.** The transition kernel is untouched (the decision was taken before the audit, and the audit's
outcome is not used by the chain), so the stage's sampled law is exactly what it was. The surrogate's
training set changes, which is already covered by the DAMH-SMU adaptation argument (note 18: the kernel
depends on the surrogate, the surrogate on the data; the same A5 diminishing-adaptation requirement
applies, no new condition). This is the "random refinement with probability `β_t`" of Conrad et al. 2016,
without their decay — the decay is unnecessary here because the audit does not alter the kernel.

**Effect on the blind spot.** Pessimistic errors now get corrected at rate `p_audit × (pre-rejection rate)`
per iteration. If `L(y*) ≫ L̃(y*)` is found at audited points, the next retraining raises `L̃` in `R`, the
first-stage acceptance in `R` rises, and the trap opens. The audit statistic itself is the diagnostic that
is missing today (§4.6).

**Cost.** `p_audit × pre-rejected` extra exact evaluations. In the note-19 Hamiltonian scheme 19 % of
proposals are pre-rejected, so `p_audit = 0.1` costs 2 % more exact evaluations; for random-walk
sub-chains (65 % pre-rejected) the same `p_audit` costs 6.5 %.

**Implementation.** `Algorithm_DAMH.run`, in the branch that calls `_transition_to_prerejected()`: draw from
the per-rank generator (a *separate* stream, so that the existing sample streams stay bit-identical when
`p_audit = 0`), call the solver, send the snapshot through the existing `CommSnapshots`/local path with
multiplicity 0 and a flag `audit=True` in `raw_data` (new `state_type` value, format v2 permits new state
types without breaking readers — check `post_processing/loading`). New field `Stage.audit_prerejected:
float = 0.0`. Without a collector, or in a frozen stage, the field is ignored with a start-up note. Unit
test: with `p_audit = 0` the sample stream is unchanged (compare CSVs); with `p_audit = 1` every
pre-rejected proposal has an exact evaluation in `raw_data`. Manual check: the notebook's trap replayed in
the library (a 2-parameter solver with the constructed error is not possible since the surrogate is the
network; instead: short warm-up on the GRF problem started from a prior draw, DAMH-SMU with HMC, compare
time to reach the reference posterior mean with `p_audit ∈ {0, 0.05, 0.2}`).

### P2 — Mixture kernel: an exact MH step with probability `p_exact`

**Mechanism.** In every DAMH iteration, with probability `p_exact` perform a plain MH step on the exact
model (random walk with the carried covariance, or the same Hamiltonian proposal accepted with the exact
likelihood) instead of the DA step.

**Validity.** A mixture of `π`-invariant kernels is `π`-invariant (Tierney 1994); both components are
reversible, so the mixture is. Works in frozen stages too, where P1 is useless.

**Effect.** Unlike P1 it lets the chain *move* into `R` even while the surrogate is still wrong there,
because the exact step ignores the surrogate entirely. It is the only remedy that acts in a frozen stage.

**Cost.** `p_exact` exact evaluations per iteration regardless of pre-rejection, i.e. `p_exact / (1 −
pre-rejection rate)` relative to the DA step's exact evaluations — more expensive than P1 for the same
`p`.

**Implementation.** `Stage.exact_step_probability: float = 0.0`; in `Algorithm_DAMH.run` branch on a draw
from the separate stream; the exact branch is `Algorithm_MH`'s body with the same proposal object. The
adaptive proposal must be told which acceptance it is seeing (exact vs surrogate) — for dual averaging on a
Hamiltonian proposal the exact-step acceptance should probably be excluded (it adapts on the sub-chain
acceptance by design, note 16 §5).

### P3 — Approximation error model in the surrogate likelihood (global)

**Mechanism.** Replace `L̃(u) = N(y; G̃(u), Σ_noise)` by

    L̃(u) = N( y ; G̃(u) + ε̄ , Σ_noise + Σ_surr ),

with `ε̄` and `Σ_surr` the empirical mean and covariance of the residuals `G(u_i) − G̃(u_i)` on a set of
points with exact evaluations (Kaipio–Somersalo; Cui et al. 2011). The same `L̃` is used in stage 1, in the
Hamiltonian potential, and in the stage-2 correction, so DAMH remains exact for the true posterior with
`Σ_noise` — the error model only changes *where the chain goes*, not *what it samples*.

**Effect on the blind spot.** The log-likelihood gap in `R` shrinks from `½|G − G̃|²/σ²` to
`½|G − G̃|²/(σ² + σ_surr²)`; with `σ = 0.01`, `|G − G̃| ≈ 0.1 ≈ σ_surr`, that is 50 nats → 0.5 nat, i.e.
`α₁` from `e⁻⁵⁰` to ≈ 0.6. Proposals into `R` pass, `G` is evaluated there, snapshots arrive, the network
corrects itself, `Σ_surr` shrinks at the next re-estimation, stage 1 sharpens again. The adaptive loop
closes in the right direction for *both* signs of error. Price: where the surrogate is good, stage 1 passes
more proposals than necessary (lower pre-rejection, more exact evaluations). With `Σ_surr` re-estimated as
the network improves, that price falls over the run.

**Effect on the Hamiltonian proposal (important).** `∇Ũ = J̃ᵀ (Σ_noise + Σ_surr)⁻¹ (G̃ + ε̄ − y) + u`. The
likelihood force is damped by `Σ_noise (Σ_noise + Σ_surr)⁻¹` (≈ 1/100 in the example); the barriers of the
spurious surrogate landscape shrink by the same factor, and with the `dimension_robust` integrator the
motion becomes dominated by the exact prior rotation. The proposal therefore explores on the prior scale
where the surrogate is untrustworthy. The notebook: fraction of iterations in which the chain moves, from
0.5 % (surrogate only) to 37 % (global error model) on the default example; the fraction of iterations with
an exact evaluation at a point of higher exact posterior than the current state, from 11 % to 37 %.

**Where `ε̄`, `Σ_surr` come from, in the library.** (i) The held-out `TestData` (prior draws, exact
evaluations already stored): residuals are available after every retraining at no extra solver cost;
`surrogate_quality_test.csv` already holds their RMSE. Prior draws over-weight regions the posterior does
not care about, so use the posterior-weighted residuals (the weights are there: `weighted_rmse`) or a mix.
(ii) The run's own residuals `obs − obs_approx` in `raw_data` (Cui et al.'s choice): free, posterior-located,
but biased in favour of the surrogate (only stage-1 survivors) — an under-estimate. Use (i) as the floor
and (ii) for the mean `ε̄`. A full `Σ_surr` (observations are correlated through the field) rather than a
diagonal; regularise with the shrinkage already used for the adaptive covariance.

**Validity under on-the-fly updates.** `ε̄, Σ_surr` change at every retraining, so the first-stage kernel
changes; this is the same adaptation as the network change itself and is covered by notes 18/20
(sampler-side freeze per sub-chain; diminishing adaptation per stage if Polyak-averaged). In a frozen
stage they are frozen with the network.

**Implementation.** The library computes `log_likelihood_approx` and the surrogate gradient with
`self.likelihood` (`Algorithm_DAMH`, `_compute_surrogate_log_likelihood_gradient`); introduce a separate
`likelihood_surrogate` object (`SamplingFramework(likelihood_surrogate=...)`, default `None` = current
behaviour), a `Normal(mean=y − ε̄, cov=Σ_noise + Σ_surr)` whose parameters the collector updates together
with the evaluator (ship them inside the evaluator message, `TAG_EVALUATOR_OBJECT`, as two attributes of
the evaluator — no new MPI message type; the sampler rebuilds the `Normal` on install). `TestData` already
lives on the collector. `run_local` mirrors it. Posterior-affecting flag: no (the target is unchanged), but
acceptance rates and sample streams change whenever it is on — document as such. Tests: closed-form
Gaussian toy (`tests/unit` V-series) with a deliberately biased linear surrogate: posterior unchanged,
acceptance changed in the predicted direction.

### P4 — Local error model `Σ_surr(u)`

**Why.** A global `Σ_surr` is an estimate of the error *where there are data*; the extrapolation error in
`R` can be much larger, so the blind spot is weakened, not removed, and the global flattening is paid
everywhere. A local envelope that grows away from the data is sharp where the network is trusted and flat
where it is not — the notebook's "local error model" raises the fraction of moving iterations to 61 %.

**Two cheap estimators.**
- *Deep ensemble*: the collector trains `K` (3–5) networks with different seeds — it is idle anyway
  (`min_snapshots_to_update=0` already means "train continuously") — and ships them as one evaluator
  returning mean and variance. `Σ_surr(u) = c · diag(var_k(u))`, with the calibration factor `c` fitted
  on the held-out residuals (Kuleshov et al.; the notebook needed `c ≈ 20`). Autograd goes through the mean
  and the variance alike, so the Hamiltonian gradient includes the `−½ log det` term that pushes the
  trajectory *out of* confident-but-wrong regions. `K`× the gradient cost per leapfrog step (19 ms → ~60–
  100 ms per exact evaluation in the note-19 scheme; still cheap for a solver of seconds).
- *Distance to data*: `σ_surr(u) = s_global · min(1, d(u)/d₀)` with `d(u)` the distance to the nearest
  snapshot (`surrogates/nearest_kdtree.py` already does this) and `d₀` a scale from the snapshot cloud.
  No extra training; the gradient through `d(u)` is piecewise and can be dropped (treat the envelope as
  frozen along a trajectory — the stage-1 ratio then uses `L̃` with the envelope, the trajectory the
  damped force only). In 20+ dimensions nearest-neighbour distances concentrate, so this is coarse.

**Caveats, measured in the notebook.** (i) An uncapped envelope turns the whole unexplored space into a
plateau *above* the data well (an uncertain point has `L̃ ≤ 1/√det(2πΣ_total)` which can exceed an
imperfect fit's likelihood), and in a 2-D box "unexplored" is almost everything: the chain then wanders
into the prior tails and every exact evaluation is wasted. Cap `Σ_surr(u)` at the held-out error, or
switch the local model off in frozen stages. (ii) With one of the five constructed errors
(`tanh-quadratic` × `smooth bump`) the local model did *worse* than the plain surrogate (3 % vs 20 %
moving iterations), because the plateau carries a spurious ridge the exact model rejects. An error model
is only as good as its envelope; calibrate it, and monitor the audit statistic (P1) regardless.

### P5 — Step-size adaptation under an error model

Dual averaging tunes `ε` on the sub-chain acceptance. On a flattened plateau the acceptance is ≈ 1, so
`ε` would inflate; with P3/P4 switched on, adapt only from iterations whose likelihood force is not
negligible (e.g. `|Σ_noise (Σ_noise + Σ_surr)⁻¹|` above a threshold), or keep the adaptation to the
warm-up DAMH stage and freeze `ε` afterwards (already supported: `Hamiltonian(adaptive=False)` with the
carried step). Note that the gradient damping of P3 also keeps a trajectory stable at an `ε` tuned for the
undamped surrogate, so divergence is not the risk; inflation is.

### P6 — Diagnostics (no change to the sampled law)

- From P1: the number of audited points with `log L(y*) − log L̃(y*) > κ` (say 5 nats), per stage and rank,
  in `notes/` and the HTML report. This is the direct measurement of "how much posterior mass the
  surrogate is hiding"; today nothing measures it.
- From the held-out set: report `weighted_ess` (already written) prominently — a low value says the test
  set barely covers the posterior region and the weighted RMSE is unreliable, which is exactly the
  situation of §1.4.
- Posterior of the warm-up stage vs posterior of the frozen stage: a shift beyond what their lengths
  explain is a blind-spot symptom. Cheap to add to `write_report`.
- The warning proposed in note 19 §7(a) (a DAMH stage whose exact acceptance stays at zero) catches the
  extreme case; the blind spot usually shows as a *normal* acceptance in the wrong place, so it is
  necessary, not sufficient.

### P7 — Stage layout, without code changes

- Do not shorten the exact warm-up to save solver calls: it is the only data the first network has.
  Prefer `RandomWalk()` adaptive there (note 19: pCN's isotropic step is the wrong shape).
- Keep a DAMH-SMU stage long enough for the network to be retrained many times before freezing, and
  never freeze directly after the warm-up (bug fixed 2026-09-30, but the layout is still fragile).
- Interleave short exact MH stages between DAMH stages when the solver cost allows: a manual, coarse
  version of P2 that needs nothing new.

---

## 5. Recommended order and how to test

1. **P1** (audit) — smallest change, no effect on the sampled law, gives the missing diagnostic (P6). One
   `Stage` field, one branch in `Algorithm_DAMH`, one regression test for bit-identity at `p_audit = 0`.
2. **P3** (global error model from `TestData`) — the systematic fix; a `likelihood_surrogate` object and
   two extra attributes on the evaluator message. Test on the Gaussian toy and on the GRF problem with a
   deliberately trapped start (short warm-up from a prior draw, 4 chains, HMC DAMH-SMU): compare the time
   until the frozen-stage posterior mean is within 0.1 sd of the reference, with and without P3, with and
   without P1.
3. **P2** (mixture kernel) — if frozen stages are to be protected as well.
4. **P4** (local envelope, ensemble first) — only if 2 shows that the global model's flattening costs
   too much efficiency where the surrogate is good; measure with the note-19 scheme (0.60 ESS per exact
   evaluation is the number to keep).

What to measure throughout: the audit statistic of P6, the pre-rejection rate, ESS per exact evaluation,
and the surrogate's posterior-weighted RMSE over the run.

---

## 6. Relation to the existing notes

- Note 18 (validity of DAMH-SMU) is unaffected by P1 (kernel unchanged) and covers P3/P4 as additional
  adaptive components under A5; P2 is a `π`-invariant mixture and needs no new argument.
- Note 20 (diminishing adaptation per stage): if the network's weights are Polyak-averaged before
  publication, `ε̄` and `Σ_surr` should be averaged on the same schedule.
- Note 19: none of its runs had a blind spot (warm-up covered the posterior); its scheme is the baseline
  whose efficiency P3/P4 must not destroy where the surrogate is good.
- Note 21 (pCN): irrelevant to the mechanism, but pCN's isotropic proposals are the least surrogate-driven
  option and therefore the least affected by §1.3.

Demonstration: `toy_examples/hmc_surrogate_error_model_2d.ipynb` (2-D, constructed surrogate error with
four forward models and five error structures; exact-model reference, surrogate only, global and local
error models; one-iteration statistics and trajectory plots).
