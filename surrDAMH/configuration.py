#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:32:40 2021

@author: simona
"""

import sys
import warnings
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from surrDAMH.distributions.parent import Distribution


@dataclass
class Configuration:
    """
    Run-wide configuration, identical on every MPI rank (not broadcast or checked;
    an inconsistent copy across ranks is not detected, see
    ``library_notes/10_manual_review_notes.md`` §2.8/2.9). ``__post_init__`` derives
    the MPI role layout (``no_samplers``, ``rank_collector``, ``rank_solvers_pool``)
    from ``MPI.COMM_WORLD``'s size and the two topology flags below; see
    ``docs/running.md`` for the process-count table and ``docs/configuration.md`` for
    the full field reference (generated from this dataclass).

    Posterior-, acceptance-rate- or reproducibility-affecting fields: ``transform_before_surrogate``
    (which space the surrogate is trained on), ``state_dependent_approximation`` (see
    below), ``min_snapshots_initial``/``min_snapshots_to_update``/``max_collected_snapshots_per_loop``
    (control exactly when the surrogate is (re)trained, hence the DAMH-SMU accept/reject
    sequence for runs that use a collector), ``use_surrogate_gradients`` (may be silently
    disabled by ``SamplingFramework`` for an incompatible surrogate, see
    ``run_manifest.json``'s ``use_surrogate_gradients_requested``), ``initial_sample_type``
    (``"prior"`` draws from the unseeded global NumPy RNG and is therefore NOT
    reproducible, finding 1.9/G4), and everything under "invalid combinations" below.
    ``transform_before_saving`` and ``save_snapshots_to_file`` affect only what is
    written to disk, not the posterior itself.

    Unverified: ``state_dependent_approximation=True`` shifts the DAMH surrogate
    approximation by the model-vs-surrogate error at the current state; it is not
    validated for ``Stage.subchain_max_length > 1`` (finding 1.1, decision 1 in
    ``library_notes/09_improvement_plan.md`` §3) and emits a ``RuntimeWarning`` at
    construction. Do not use it unless you have checked the theory for your own
    ``subchain_max_length``.

    Currently ignored: ``paths_to_append`` is appended to ``sys.path`` in this
    process only -- it does NOT reach solver-pool children spawned by
    ``MPI.Comm.Spawn`` (finding M18); give ``SolverSpec`` an absolute module path
    instead (see ``docs/writing_a_solver.md``). ``pickled_observations=False`` selects
    a raw (non-pickled) MPI transport path that is scheduled for removal (decision 5,
    no replacement needed by user code -- just leave this at its default ``True``);
    it also has a known crash with ``solver_returns_tag=True`` and a negative tag
    (``library_notes/10_manual_review_notes.md`` §3).

    Invalid combinations that raise (at ``Stage``/proposal construction, i.e. at
    ``SamplingFramework.run()`` time, not eagerly at ``Configuration()`` time):
    an unknown ``Stage.algorithm_type``; a DAMH or Hamiltonian-family first stage with
    no evaluator available yet (no ``surrogate_updater``/preloaded snapshots when
    ``use_collector=True``, or no fixed ``surrogate_evaluator`` when
    ``use_collector=False`` -- see ``use_collector`` below); a pCN stage with a
    non-Gaussian internal prior (``FromScipy``, ``GaussianMixture``); a Hamiltonian
    proposal without ``use_surrogate_gradients=True`` in effect; ``Stage.adaptive=True``
    together with ``proposal_type="pCN"`` is not an error but is silently forced back to
    ``adaptive=False`` by ``Stage.__post_init__`` (a printed warning, not an exception).
    """

    no_parameters: int  # number of unknowns (i.e. parameters of the forward model)
    no_observations: int  # number of observed values (i.e. outputs of the forward model)
    output_dir: str  # directory where samples and other outputs will be saved
    use_solvers_pool: bool = True  # if False, the solver runs locally on each sampler process
    no_solvers: int = 2  # number of child solvers spawned by solvers pool
    solver_maxprocs: int = 1  # processes used by each spawned solver
    solver_returns_tag: bool = False  # if True, solver returns Tuple(observations, tag:int), negative tag indicates solver error
    use_collector: bool = True  # if False, no surrogate model is trained during the run: an Updater only ever runs on the collector rank, so DAMH/Hamiltonian stages then need a pre-trained, fixed surrogate_evaluator= passed to SamplingFramework (no in-run updates/DAMH-SMU); MH-only stages work with any combination of use_collector/use_solvers_pool
    save_snapshots_to_file: bool = False  # save all obtained snapshots to file
    transform_before_saving: bool = True  # if False, save samples based on internal distribution
    transform_before_surrogate: bool = False  # if False, construct surrogate on internal distribution; posterior-affecting: changes what the surrogate is trained/evaluated on
    initial_sample_type: Literal["lhs", "prior", "user_specified", "continued"] = "prior"  # specifies how to generate initial samples; "prior" draws from the unseeded global NumPy RNG (not reproducible, finding 1.9/G4)
    initial_samples_distribution: Distribution | None = None  # only if initial_sample_type == "user_specified"
    continued_from_dir: str | None = None  # experiment directory to continue from (only if initial_sample_type == "continued")
    lhs_scale: float | npt.NDArray = 1.0  # only if initial_sample_type == "lhs"
    state_dependent_approximation: bool = False  # shift posterior approximation by surrogate model error in current sample; NOT verified (incorrect for subchain_max_length > 1, see library_notes/06 item 1.1), do not use unless you know what you are doing
    min_snapshots_initial: int = 1  # minimal number of snapshots for the construction of initial surrogate model; posterior-affecting for DAMH-SMU (controls when the first surrogate is trained)
    min_snapshots_to_update: int = 1  # how many snapshots (at least) have to be added to update the surrogate model; posterior-affecting for DAMH-SMU (controls when later retrains happen)
    max_collected_snapshots_per_loop: int = 1000  # maximal number of snapshots to be collected in one loop (collector-side batching/performance knob)
    max_sampler_isend_requests: int = 100  # size of the buffer for isend requests (sending snapshots from samplers to collector; performance/buffering knob)
    use_surrogate_gradients: bool = True  # whether to allow autograd in pytorch surrogate; may be silently forced to False by SamplingFramework if the surrogate/settings are incompatible (see run_manifest.json)
    paths_to_append: list[str] | None = None  # appended to sys.path in this process only; does NOT reach spawned solver-pool children (M18) -- currently ignored for that purpose, prefer an absolute path in SolverSpec
    pickled_observations: bool = True  # if False, uses the raw (non-pickled) MPI transport path, scheduled for removal (decision 5); leave at the default
    max_buffer_size: int = 1 << 30  # size (bytes) of the pre-allocated irecv buffer used to receive a pickled Evaluator from the collector; performance/buffering knob, not posterior-affecting
    debug: bool = False  # collector-side: print extra diagnostics; not posterior-affecting

    def __post_init__(self) -> None:
        if self.state_dependent_approximation:
            warnings.warn(
                "state_dependent_approximation=True is NOT verified (incorrect for subchain_max_length > 1, "
                "see library_notes/06 item 1.1); do not use it unless you know what you are doing",
                RuntimeWarning,
                stacklevel=2,
            )
        if self.paths_to_append is None:
            self.paths_to_append = []
        else:
            self._append_path()

        size_world = MPI.COMM_WORLD.Get_size()

        # ranks of samplers, collector, solvers pool:
        if self.use_collector and self.use_solvers_pool:
            self.no_samplers = size_world - 2
            self.rank_solvers_pool = size_world - 2
            self.rank_collector = size_world - 1
        elif self.use_collector:  # without solvers pool, solver is local
            self.no_samplers = size_world - 1
            self.rank_solvers_pool = None
            self.rank_collector = size_world - 1
        elif self.use_solvers_pool:  # without collector
            self.no_samplers = size_world - 1
            self.rank_solvers_pool = size_world - 1
            self.rank_collector = None
        else:  # no collector, no solvers pool
            self.no_samplers = size_world
            self.rank_collector = None
            self.rank_solvers_pool = None
        assert self.no_samplers > 0, ("number of MPI processes is too low: use at least 3 MPI processes with collector and "
                                      "solvers pool (1 sampler + pool + collector); 4 recommended")
        self.sampler_ranks = np.arange(self.no_samplers)  # ranks 0, 1, ..., no_samplers-1

        self.continued_samples: npt.NDArray | None = None
        if self.initial_sample_type == "continued":
            if self.continued_from_dir is None:
                raise ValueError("continued_from_dir must be set when initial_sample_type == 'continued'")
            from surrDAMH.modules.continuation import load_last_samples
            self.continued_samples = load_last_samples(
                experiment_dir=self.continued_from_dir,
                no_parameters=self.no_parameters,
                no_chains=self.no_samplers,
            )

    def _append_path(self) -> None:
        assert self.paths_to_append is not None
        for path in self.paths_to_append:
            sys.path.append(path)
