"""
surrDAMH: surrogate-accelerated MCMC (delayed-acceptance Metropolis-Hastings) for
Bayesian inverse problems, parallelised with mpi4py.

A sampling script needs five things, all reachable from here::

    import surrDAMH
    from surrDAMH.proposals import RandomWalk          # optional: proposal settings

    prior = surrDAMH.distributions.Normal(mean=0, sd=1, dim=3)
    likelihood = surrDAMH.distributions.Normal(mean=observed_data, sd=0.1)
    problem = surrDAMH.Problem(prior, likelihood, solver=my_solver)   # sizes: from solver/prior/likelihood
    conf = surrDAMH.Configuration(output_dir="out_x", use_solvers_pool=False, use_collector=False)
    stages = [surrDAMH.Stage(max_evaluations=1000)]
    run = problem.run_sampling(conf, stages)      # launch with: mpiexec -n 4 python3 -m mpi4py script.py
    # run = problem.run_sampling_local(conf, stages)   # or one chain in one process, no MPI
    run.write_report(observations=observed_data)  # HTML report under <output_dir>/post_processing_output/

Where to look for what:

- ``surrDAMH.Configuration``      -- run-wide settings (MPI roles, output, surrogate timing)
- ``surrDAMH.distributions``      -- priors and likelihoods: ``Normal``,
  ``PriorIndependentComponents``, ``GaussianMixture``, ``FromScipy``
- ``surrDAMH.Stage``              -- one sampling stage (also ``surrDAMH.stages.Stage``)
- ``surrDAMH.proposals``          -- proposal settings of a stage: ``RandomWalk``, ``PCN``,
  ``Hamiltonian``, ``Block``
- ``surrDAMH.Solver``             -- base class for your forward model;
  ``surrDAMH.SolverSpec`` describes one that must be built in another process
- ``surrDAMH.surrogates``         -- surrogate updaters (``PolynomialSklearnUpdater``,
  ``RBFInterpolationUpdater``, ``KDTreeUpdater``, ``NeuralNetworkUpdater``)
  and the ``Updater``/``Evaluator`` base classes for writing your own
- ``surrDAMH.Problem``            -- prior + likelihood + solver; ``run_sampling`` (MPI) and
  ``run_sampling_local`` (one chain, one process) return a ``surrDAMH.SamplingRun``
  (``write_report``)
- ``surrDAMH.post_processing``    -- analysis of a finished run (``Samples``);
  ``surrDAMH.read_run`` loads its raw output

Documentation: ``docs/README.md``; runnable examples: ``toy_examples/``.
"""

from . import (distributions, post_processing, proposals, solver_specification,
               solvers, stages, surrogates)
from .configuration import Configuration
from .distributions import Distribution
from .solvers import Solver
from .solver_specification import SolverSpec
from .stages import Stage
from .core import Problem, SamplingRun
from . import runner_local
from .modules.manifest import RunFormatError
from .modules.run_data import RunData, read_run
from .modules.surrogate_restart import SurrogateRestart
from .modules.test_data import TestData

__all__ = [
    # building a sampling script
    "Configuration", "Stage", "Solver", "SolverSpec", "Distribution", "Problem", "SamplingRun",
    # optional inputs of Problem.run_sampling
    "TestData", "SurrogateRestart",
    # reading a finished run
    "read_run", "RunData", "RunFormatError",
    # subpackages
    "distributions", "proposals", "stages", "solvers", "solver_specification",
    "surrogates", "post_processing", "runner_local",
]
