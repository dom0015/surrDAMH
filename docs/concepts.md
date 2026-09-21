# Concepts

## The Bayesian inverse problem

surrDAMH samples the posterior of

    f(u | y) ∝ f_noise(y − G(u)) · f_prior(u)

where `u` are unknown parameters, `G` an expensive forward model (a PDE solver, e.g.),
`y` observed data, and `f_noise` the observation-noise density (the `likelihood`
argument's `logpdf`, evaluated at `y − G(u)` or however the user's `Distribution`
subclass defines it — the library never subtracts `y` itself, that is the caller's
`likelihood`'s job). One MPI rank runs one Markov chain; several ranks run several
chains of the same posterior in parallel, independently (the only inter-chain coupling is the
end-of-stage pooling of an adaptive proposal's adaptation statistics, see `docs/stages.md`).

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

With `Stage.surrogate_model_updates=True` (DAMH-SMU; `None`, the default, means `True` for a
DAMH stage) the surrogate keeps retraining on
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

Since WS7 (2026-09-18) the same field also opts an **MH** stage with a gradient-based proposal
(`Hamiltonian`/`HamiltonianInfinite`, or a `block` proposal containing one) into refreshing its
surrogate: it then polls once per iteration and re-installs the proposal's gradient functions
when a newer evaluator arrives. There is no telescoping constraint to respect here, because an
MH stage's accept/reject test uses the exact model only — the surrogate enters through the
proposal's gradients alone, and the chain targets the exact posterior for any gradient field.
The default (`None`) keeps the pre-WS7 behaviour: one evaluator for the whole MH stage.

## Why the surrogate must not depend on the current state

The correction above cancels only because the sub-chain is reversible w.r.t. **one fixed**
surrogate posterior `pi~`: that is what gives `Q(y→x)/Q(x→y) = pi~(y)/pi~(x)`. A surrogate
re-centred on the outer chain's current state — e.g. the removed
`Configuration.state_dependent_approximation`, which used `observations_approx + G(x₀) − G~(x₀)`
instead of the surrogate values directly — makes `pi~` a different density at every outer step,
so that identity (and with it the DAMH acceptance ratio) no longer holds. This is true already at
`subchain_max_length = 1`, not only for `> 1` as this document previously claimed; the option was
therefore removed on 2026-09-18 rather than fixed (`library_notes/06_findings_consolidated.md`
finding 1.1).

## What "posterior-affecting" means in this documentation

Throughout `docs/configuration.md`/`docs/stages.md` and the docstrings, a field is
called **posterior-affecting** if changing it can change the stationary distribution
the chain targets, or the accept/reject sequence that produces samples from it (hence
also the acceptance rate and effective sample size) — as opposed to fields that only
change performance, buffering, or what gets written to disk. A field can be
posterior-affecting only for certain algorithm/stage combinations (e.g.
`min_snapshots_initial` only matters when a surrogate is trained at all).
