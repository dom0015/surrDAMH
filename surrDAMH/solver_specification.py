#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass
from typing import List


@dataclass
class SolverSpec:
    """
    Solvers are initiated by spawned processes.
    """
    solver_module_path: str
    solver_module_name: str
    solver_class_name: str
    solver_parameters: dict

    # def __post_init__(self) -> None:
    #     self.solver_parameters = {}
