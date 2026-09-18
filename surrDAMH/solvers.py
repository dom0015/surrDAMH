#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util as iu
import os
from typing import Any

import numpy as np
import numpy.typing as npt

from surrDAMH.solver_specification import SolverSpec


class Solver:
    """
    Parent class / contract for the forward model. Subclass and override
    ``set_parameters``/``get_observations`` (or ``__call__`` in full) to plug a model
    into the library; see ``docs/writing_a_solver.md``.

    Called from three places with different lifetimes: directly by ``run_local()``
    (one instance, one process); once per spawned child process by the solvers pool
    (``get_solver_from_spec``, one instance per child, reused for every request that
    child receives); once per sampler rank when ``use_solvers_pool=False`` (one
    instance per rank). A solver must therefore be safe to call repeatedly with
    different ``parameters`` and must not assume anything about which parameters were
    passed before.

    Optional duck-typed attributes, read via ``getattr``/``hasattr`` where used, not
    part of the required contract: ``par_names`` (list of parameter names, used by
    ``SamplingFramework.write_report``), ``field_builder``/``coords``/
    ``measurement_points`` (enable the posterior-field-statistics section of the HTML
    report, all three required together).
    """

    def __init__(self, solver_id: int = 0, output_dir: str | None = None) -> None:
        """
        Args:
            solver_id: identifies this solver instance (e.g. MPI rank of the spawned
                child or of the sampler running it locally); solvers that write scratch
                files should key them by ``solver_id`` to avoid collisions between
                concurrently running instances.
            output_dir: directory the solver may use for scratch/output files, already
                created by the caller when given (e.g. ``solver_output/rank<k>/``).
        """
        pass

    def set_parameters(self, parameters: npt.NDArray):
        """Stores ``parameters`` (physical space, i.e. after ``prior.transform``) for the next ``get_observations()`` call."""
        self.parameters = parameters

    def get_observations(self) -> npt.NDArray:
        """
        Runs the forward model on the last ``set_parameters`` call and returns the
        simulated observations.

        Returns:
            ``(no_observations,)`` array, matching ``Configuration.no_observations``. If
            ``conf.solver_returns_tag=True``, return ``(observations, tag)`` instead,
            where ``tag < 0`` signals a failed solve (the caller then treats
            ``observations`` as invalid and, in the solvers pool, sends zeros in its
            place); ``tag`` is otherwise unused. Not implemented by the base class.
        """
        raise NotImplementedError

    def set_parameters_and_get_observations(self, parameters: npt.NDArray):
        self.set_parameters(parameters)
        return self.get_observations()

    def visualize_solution(self, show: bool = False) -> list[tuple[Any, Any]]:
        """
        Optional hook, called by ``SamplingFramework.write_report`` on the best-fit
        sample. Return a list of ``(figure, axes)`` matplotlib pairs; the caller saves
        and closes each figure. Default: no visualizations.
        """
        return []

    def __call__(self, parameters: npt.ArrayLike) -> npt.NDArray:
        """Equivalent to ``set_parameters_and_get_observations(parameters)``; used by ``calculate_artificial_observations`` and wherever a plain callable is more convenient than the two-step contract."""
        self.set_parameters(parameters)
        return self.get_observations()


def get_solver_from_spec(solver_spec: SolverSpec, solver_id: int = 0, solver_output_dir: str | None = None) -> Solver:
    """
    Import ``solver_spec.solver_module_path`` and instantiate ``solver_class_name`` from it.

    The module path is made absolute here as well as in ``SolverSpec.__post_init__``, which
    covers a spec built without running ``__post_init__``. Note that this last resort
    resolves against the *importing* process's working directory: in a spawned solver child
    that is not guaranteed to be the launching one (finding M18), which is why the
    authoritative resolution happens on the launching rank.
    """
    module_path = os.path.abspath(solver_spec.solver_module_path)
    spec = iu.spec_from_file_location(solver_spec.solver_module_name, module_path)
    assert spec is not None
    module = iu.module_from_spec(spec=spec)
    assert spec.loader is not None
    spec.loader.exec_module(module=module)
    solver_init = getattr(module, solver_spec.solver_class_name)
    constructor_parameters = solver_spec.solver_parameters.copy()
    constructor_parameters["solver_id"] = solver_id
    constructor_parameters["output_dir"] = solver_output_dir
    solver_instance = solver_init(**constructor_parameters)
    return solver_instance


def calculate_artificial_observations(parameters: npt.ArrayLike,
                                      solver_instance: Solver | None = None,
                                      solver_spec: SolverSpec | None = None) -> npt.NDArray:
    if solver_instance is None:
        assert solver_spec is not None, "to calculate artificial observations, solver must be given"
        solver_instance = get_solver_from_spec(solver_spec)
    solver_instance.set_parameters(parameters)
    observations = solver_instance.get_observations()
    return np.array(observations.ravel())
