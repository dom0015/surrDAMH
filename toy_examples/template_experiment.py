#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Canonical starting point for a new surrDAMH experiment: copy this file, then edit
sections 2-6 for your own problem. Run from the toy_examples/ directory with:

    mpiexec -n 4 python3 -m mpi4py template_experiment.py

(here: 2 samplers + 1 solvers pool + 1 collector; more ranks = more sampler chains).
With use_collector=False and use_solvers_pool=False (section 4) an MH-only version of
this script (drop the two DAMH stages in section 6 -- DAMH needs either a collector rank
to retrain the surrogate, or a pre-trained fixed surrogate_evaluator, neither of which a
single process without a collector can provide) runs the same way with
mpiexec -n 1 python3 -m mpi4py template_experiment.py. For a quick check with no MPI at
all, run one chain via surrDAMH.runner_local.run_local(...).
"""

import solver_examples.solver_spec_examples as solver_spec_examples

import surrDAMH
from surrDAMH.stages import Stage

# --- 2. forward model -------------------------------------------------------
# G(parameters) -> observations; here a linear map with a closed-form posterior
# (see solver_examples/solver_examples.py::LinearGaussianSolver). Point this at
# your own solver module the same way (solver_module_path is resolved relative
# to this working directory).
solver_spec = solver_spec_examples.SolverSpecLinearGaussian()

# --- 3. prior and likelihood -------------------------------------------------
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
])
# artificial ("synthetic") observations: evaluate the true forward model once at a
# chosen "true" parameter vector, then add noise via the likelihood below:
true_observations = surrDAMH.solvers.calculate_artificial_observations(
    solver_spec=solver_spec, parameters=[0.7, -0.4])
likelihood = surrDAMH.distributions.Normal(mean=true_observations, sd=0.1)

# --- 4. configuration ---------------------------------------------------------
conf = surrDAMH.Configuration(
    output_dir="out_template_experiment",       # everything gets written under here
    no_parameters=2,                            # must match the solver/prior
    no_observations=2,                          # must match the solver/likelihood
    use_collector=True,                         # False = no surrogate model at all (plain MH only)
    use_solvers_pool=True,                       # False = solver runs in-process on each sampler (no spawn)
    no_solvers=2,                                # child solver processes spawned by the pool (only if use_solvers_pool)
    min_snapshots_initial=4,                    # snapshots needed before the initial surrogate is built
    min_snapshots_to_update=2,                  # further snapshots needed before each surrogate re-fit
    initial_sample_type="prior",                 # "prior"/"lhs"/"user_specified"/"continued"; "prior" is NOT
                                                 # reproducible run-to-run today (library_notes finding 1.9)
    state_dependent_approximation=False,        # True shifts the DAMH approximation by the surrogate error at
                                                 # the current sample; UNVERIFIED for subchain_max_length > 1
                                                 # (finding 1.1) -- leave at the default unless you know why.
)

# --- 5. surrogate model -------------------------------------------------------
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(
    no_parameters=conf.no_parameters, no_observations=conf.no_observations, max_degree=5)
# updater = surrDAMH.surrogates.RBFInterpolationUpdater(
#     no_parameters=conf.no_parameters, no_observations=conf.no_observations,
#     neighbors=20, kernel="thin_plate_spline")
# updater = surrDAMH.surrogates.KDTreeUpdater(
#     no_parameters=conf.no_parameters, no_observations=conf.no_observations, no_nearest_neighbors=5)
# updater = surrDAMH.surrogates.NeuralNetworkUpdaterMinibatches(
#     no_parameters=conf.no_parameters, no_observations=conf.no_observations,
#     hidden_layer_sizes=(32, 32), solver="adamw")

# --- 6. sampling stages --------------------------------------------------------
list_of_stages = [
    # MH: no surrogate involved yet; also produces the first snapshots for it.
    Stage(algorithm_type="MH", proposal_sd_or_cov=0.5,
          max_evaluations=80,           # stop after this many full-model evaluations (also: time_limit in seconds)
          adaptive=False),              # if True, proposal_sd_or_cov is adapted online (see Stage docstring)
    # DAMH-SMU: delayed-acceptance, surrogate is retrained while sampling.
    Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5,
          max_evaluations=80,
          surrogate_model_updates=True,  # only meaningful for DAMH: surrogate keeps learning during this stage
          subchain_max_length=5),        # length of the cheap surrogate-only sub-chain between full-model checks
    # DAMH with a frozen surrogate: cheapest stage, used for the final posterior.
    Stage(algorithm_type="DAMH", proposal_sd_or_cov=0.5,
          max_evaluations=80,
          surrogate_model_updates=False,
          subchain_max_length=5),
]

# --- 7. run --------------------------------------------------------------------
sam = surrDAMH.SamplingFramework(conf, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec,
                                 surrogate_updater=updater)
sam.run()

# --- 8. report ------------------------------------------------------------------
# called on every rank: rank 0 builds sampling_output/post_processing_output/report_extended.html
# and summary.csv, the other ranks just wait at the internal MPI barrier.
sam.write_report(observations=true_observations)
