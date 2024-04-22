#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy.typing as npt
import importlib.util as iu
from surrDAMH.solver_specification import SolverSpec


class Solver:
    """Parent class for solvers."""

    def __init__(self, solver_id: int = 0, output_dir: str | None = None) -> None:
        pass

    def set_parameters(self, parameters: npt.ArrayLike) -> None:
        """Sets sample for which the observations will be computed later."""
        pass

    def get_observations(self) -> npt.NDArray:
        """Gets observations computed by the solver."""
        raise NotImplementedError


def get_solver_from_spec(solver_spec: SolverSpec, solver_id: int = 0, solver_output_dir: str | None = None) -> Solver:
    spec = iu.spec_from_file_location(solver_spec.solver_module_name, solver_spec.solver_module_path)
    assert spec is not None
    module = iu.module_from_spec(spec=spec)
    assert spec.loader is not None
    spec.loader.exec_module(module=module)
    solver_init = getattr(module, solver_spec.solver_class_name)
    constructor_parameters = solver_spec.solver_parameters.copy()  # TODO: copy?
    constructor_parameters["solver_id"] = solver_id
    constructor_parameters["output_dir"] = solver_output_dir
    solver_instance = solver_init(**constructor_parameters)
    return solver_instance


def calculate_artificial_observations(solver_spec: SolverSpec, parameters: npt.ArrayLike) -> npt.NDArray:
    solver_instance = get_solver_from_spec(solver_spec)
    # artificial observations and noise std
    solver_instance.set_parameters(parameters)
    observations = solver_instance.get_observations()
    return observations
