#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import importlib.util as iu
from typing import Any

import numpy as np
import numpy.typing as npt

from surrDAMH.solver_specification import SolverSpec


class Solver:
    """Parent class for solvers."""

    def __init__(self, solver_id: int = 0, output_dir: str | None = None) -> None:
        """
        Args:
        solver_id (int): ID of the solver. Defaults to 0.
        output_dir (str, optional): Output directory. Defaults to None.
        """
        pass

    def set_parameters(self, parameters: npt.NDArray):
        self.parameters = parameters

    def get_observations(self) -> npt.NDArray:
        """Gets observations computed by the solver."""
        raise NotImplementedError

    def set_parameters_and_get_observations(self, parameters: npt.NDArray):
        self.set_parameters(parameters)
        return self.get_observations()

    def visualize_solution(self, show: bool = False) -> list[tuple[Any, Any]]:
        """Return a list of (figure, axes) pairs for optional solver-specific visualizations."""
        return []
    
    def __call__(self, parameters: npt.ArrayLike) -> npt.NDArray:
        self.set_parameters(parameters)
        return self.get_observations()


def get_solver_from_spec(solver_spec: SolverSpec, solver_id: int = 0, solver_output_dir: str | None = None) -> Solver:
    spec = iu.spec_from_file_location(solver_spec.solver_module_name, solver_spec.solver_module_path)
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
