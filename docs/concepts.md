# Concepts

## The Bayesian inverse problem

surrDAMH samples the posterior of

    f(u | y) ∝ f_noise(y − G(u)) · f_prior(u)

where `u` are unknown parameters, `G` an expensive forward model (a PDE solver, e.g.),
`y` observed data, and `f_noise` the observation-noise density (the `likelihood`
argument's `logpdf`, evaluated at `y − G(u)` or however the user's `Distribution`
subclass defines it — the library never subtracts `y` itself, that is the caller's
`likelihood`'s job). One MPI rank runs one Markov chain; several ranks run several
chains of the same posterior in parallel, independently (no inter-chain coupling beyond
adaptive-proposal covariance averaging, see `docs/stages.md`).

## Internal vs. physical space

The chain's state is a vector in an *internal* parameter space; `prior.transform`
maps it to the *physical* parameters that the forward model and the on-disk output
understand. Concretely, `Distribution` is the function composition
`transform ∘ internal_prior`:

- `prior.rvs()`, `prior.logpdf()`, `prior.grad_logpdf()`, the proposal, and the MH
  acceptance ratio all operate on the INTERNAL sample.
- `prior.transform(sample)` produces the PHYSICAL parameters passed to
  `Solver.set_parameters`/`Evaluator.__call__` (when `Configuration.transform_before_surrogate=True`)
  and written to `samples/*.csv` (when `Configuration.transform_before_saving=True`, the default).

For `PriorIndependentComponents`, the internal prior is *exactly* the standard normal
`N(0, I)` by construction: `logpdf(sample) == -0.5 * sample @ sample`, with **no
Jacobian/change-of-variables term** for the (possibly nonlinear) `transform`. This is a
deliberate design choice, not an oversight — acceptance ratios are computed entirely in
the internal N(0, I) space, so the Jacobian would cancel between numerator and
denominator of any ratio even if it were included, and every `logpdf` in this library is
defined "up to an additive constant" for the same reason (see
`surrDAMH/distributions/parent.py`, pinned by
`tests/unit/test_distributions.py::TestPriorIndependentComponentsLogpdfDesign`). `Normal`
and `FromScipy` skip the transform (identity) and work directly in what is then a single
space.

## Metropolis-Hastings (MH)

Standard MH (`Algorithm_MH`): propose `y ~ Q(x, ·)`, evaluate the exact model at `y`,
accept with probability `min(1, exp(log α))` where
`log α = [log L(y) − log L(x)] + [log prior(y) − log prior(x)]` (the two parts returned
separately by `Proposal.get_log_acceptance_probability`, so an asymmetric proposal like
pCN can fold a proposal-density ratio into the second term). Every proposed point costs
one exact forward-model evaluation.

## Delayed-acceptance MH (DAMH)

DAMH (`Algorithm_DAMH.run`) replaces most exact evaluations with a cheap surrogate `L~`,
and only confirms the result with the exact model occasionally. Per outer iteration:

1. Run a **sub-chain** of up to `Stage.subchain_max_length` MH steps that use *only* the
   surrogate, starting from the current exact state `x`. This inner MH kernel is
   reversible with respect to the surrogate posterior `π~`, so its transition density
   satisfies `Q(y→x)/Q(x→y) = π~(y)/π~(x) = [L~(y)·prior(y)] / [L~(x)·prior(x)]`.
2. If at least one sub-chain step was accepted, its endpoint `y` is evaluated with the
   *exact* model and accepted with

       log α = [log L(y) − log L(x)] − [log L~(y) − log L~(x)]

   i.e. the prior cancels, and the surrogate correction is the telescoped sum of the
   accepted sub-chain steps' surrogate log-likelihood ratios (`correction_log_ratio` in
   the code). If no sub-chain step was accepted, the outer iteration is **pre-rejected**
   with no exact evaluation at all — this is DAMH's speed-up.

The telescoping in step 2 is only exact if the surrogate is the *same* `L~` throughout
the sub-chain. `subchain_max_length=1` makes DAMH degenerate to a fixed-surrogate
delayed-acceptance scheme with one sub-chain step per outer iteration.

## DAMH-SMU (surrogate model updates during sampling)

With `Stage.surrogate_model_updates=True` (DAMH-SMU) the surrogate keeps retraining on
the collector while the chain runs. To keep the telescoping in step 2 valid, the
evaluator is **frozen for the duration of one sub-chain**: it is refreshed at most once,
right before the sub-chain starts, and held fixed until the sub-chain ends (a new
evaluator that arrives mid-sub-chain is only picked up at the start of the *next* outer
iteration). This is what makes `correction_log_ratio` telescope to
`log L~final(y) − log L~final(x)` for a single, well-defined surrogate — see the
derivation comment in `Algorithm_DAMH.run` and
`library_notes/10_manual_review_notes.md` §2.5 for the regression test and the
before/after sample-stream comparison (the freeze changed DAMH-SMU streams for
`subchain_max_length > 1`; `subchain_max_length=1` is unaffected).

## `state_dependent_approximation` (unverified)

`Configuration.state_dependent_approximation=True` uses the surrogate as an *additive
correction* around the current exact state (`observations_approx + G(x) − G~(x)`)
instead of using the surrogate values directly. This is **not validated** for
`Stage.subchain_max_length > 1` (the shifted sub-chain kernel's reversibility has not
been derived, `library_notes/06_findings_consolidated.md` finding 1.1); the default is
`False`, and setting it to `True` raises a `RuntimeWarning` at `Configuration`
construction. Do not use it for `subchain_max_length > 1` unless you have checked the
theory yourself (`library_notes/09_improvement_plan.md` §3 decision 1, §4 item 1).

## What "posterior-affecting" means in this documentation

Throughout `docs/configuration.md`/`docs/stages.md` and the docstrings, a field is
called **posterior-affecting** if changing it can change the stationary distribution
the chain targets, or the accept/reject sequence that produces samples from it (hence
also the acceptance rate and effective sample size) — as opposed to fields that only
change performance, buffering, or what gets written to disk. A field can be
posterior-affecting only for certain algorithm/stage combinations (e.g.
`min_snapshots_initial` only matters when a surrogate is trained at all).
