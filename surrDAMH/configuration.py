#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb 28 12:32:40 2021

@author: simona
"""

import sys
from dataclasses import dataclass
from typing import Literal

import numpy as np
import numpy.typing as npt
from mpi4py import MPI

from surrDAMH.distributions.parent import Distribution


@dataclass
class Configuration:
    no_parameters: int  # number of unknowns (i.e. parameters of the forward model)
    no_observations: int  # number of observed values (i.e. outputs of the forward model)
    output_dir: str  # directory where samples and other outputs will be saved
    use_solvers_pool: bool = True  # if False, the solver runs locally on each sampler process
    no_solvers: int = 2  # number of child solvers spawned by solvers pool
    solver_maxprocs: int = 1  # processed used by spawned solvers
    solver_returns_tag: bool = False  # if True, solver returns Tuple(observations, tag:int), negative tag indicates solver error
    use_collector: bool = True  # if False, no surrogate model will be constructed
    save_snapshots_to_file: bool = False  # save all obtained snapshots to file
    transform_before_saving: bool = True  # if False, save samples based on internal distribution
    transform_before_surrogate: bool = False  # if False, construct surrogate on internal distribution
    initial_sample_type: Literal["lhs", "prior", "user_specified", "continued"] = "prior"  # specifies how to generate initial samples
    initial_samples_distribution: Distribution | None = None  # only if initial_sample_type == "user_specified"
    continued_from_dir: str | None = None  # experiment directory to continue from (only if initial_sample_type == "continued")
    lhs_scale: float | npt.NDArray = 1.0  # only if initial_sample_type == "lhs"
    state_dependent_approximation: bool = False  # shift posterior approximation by surrogate model error in current sample
    min_snapshots_initial: int = 1  # minimal number of snapshots for the construction of initial surrogate model
    min_snapshots_to_update: int = 1  # how many snapshots (at least) have to be added to update the surrogate model
    max_collected_snapshots_per_loop: int = 1000  # maximal number of snapshots to be collected in one loop
    max_sampler_isend_requests: int = 100  # size of the buffer for isend requests (sending snapshots from samplers to collector)
    use_surrogate_gradients: bool = True  # whether to allow autograd in pytorch surrogate
    paths_to_append: list[str] | None = None
    pickled_observations: bool = True
    max_buffer_size: int = 1 << 30
    debug: bool = False

    def __post_init__(self) -> None:
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
        assert self.no_samplers > 0, "number of MPI processes is too low, use at least 'mpirun -n 4'"
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
