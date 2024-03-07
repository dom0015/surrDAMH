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


class SolverSpecExample1(SolverSpec):
    """
    no_parameters = 2, no_observation = 1
    parameters = [x,y]
    observation = (x**2-y)*(log((x-y)**2+1))
    """

    def __init__(self) -> None:
        self.solver_module_path = "examples/solvers/solver_examples.py"
        self.solver_module_name = "solver_examples"
        self.solver_class_name = "Solver_illustrative_local"
        self.solver_parameters = {}


class SolverSpecExample2(SolverSpec):
    """
    1-D linear elasticity, no_parameters = 2, no_observation = 1
    x in (0,length), u = par[0] in (0,m), u = par[1] in (m,length)

    Simplified description:
    -(exp(u)p'(x))' = f
               p(0) = 0
         p'(length) = 0
    observed value is p(length)
    """

    def __init__(self, f: float = -0.1, length: float = 1.0, m: float = 0.5) -> None:
        self.solver_module_path = "examples/solvers/solver_examples.py"
        self.solver_module_name = "solver_examples"
        self.solver_class_name = "Solver_linela2exp_local"
        self.solver_parameters = {"f": f, "length": length, "m": m}


class SolverSpecExampleNonlinear(SolverSpec):
    """
    no_parameters = 4, no_observation = 1
    parameters = [x,y]
    coef = [a1,b1,c1,a2,b2,c2]
    observation = min(a1*x+b1*y+c1,a2*x+b2*y+c2)
    """

    def __init__(self, coef: List[float] = [1.0, -1.0, 0.0, -1.0, 1.0, 0.0]) -> None:
        self.solver_module_path = "examples/solvers/solver_examples.py"
        self.solver_module_name = "solver_examples"
        self.solver_class_name = "Nonlinear"
        self.solver_parameters = {"coef": coef}


class SolverSpecGeneric(SolverSpec):
    """
    Serves only for test purposes.
    Takes user-specified number of parameters (no_parameters), 
    returns their mean in the form of a constant vector of user-specified length (no_observations).
    """

    def __init__(self, no_parameters: int = 3, no_observations: int = 2) -> None:
        self.solver_module_path = "examples/solvers/solver_examples.py"
        self.solver_module_name = "solver_examples"
        self.solver_class_name = "Generic"
        self.solver_parameters = {"no_parameters": no_parameters, "no_observations": no_observations}


class SolverSpecNonlinearGeneric(SolverSpec):
    """
    Serves only for test purposes.
    Takes user-specified number of parameters (no_parameters), 
    returns their mean in the form of a constant vector of user-specified length (no_observations).
    """

    def __init__(self, no_parameters: int = 3, no_observations: int = 2, sleep: float = 0.0) -> None:
        self.solver_module_path = "examples/solvers/solver_examples.py"
        self.solver_module_name = "solver_examples"
        self.solver_class_name = "NonlinearGeneric"
        self.solver_parameters = {"no_parameters": no_parameters, "no_observations": no_observations, "sleep": sleep}
