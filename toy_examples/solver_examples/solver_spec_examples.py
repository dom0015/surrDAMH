#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from typing import List

from surrDAMH.solver_specification import SolverSpec

# Every spec below passes "solver_examples/solver_examples.py", i.e. a path relative to the
# working directory -- which is why the toy examples must be run from inside toy_examples/.
# super().__init__ (rather than assigning the four attributes directly) makes SolverSpec's
# __post_init__ run, so the path is stored absolute and the spawned solver children get an
# absolute path through the broadcast spec (finding M18, WS5).


class SolverSpecExample1(SolverSpec):
    """
    no_parameters = 2, no_observation = 1
    parameters = [x,y]
    observation = (x**2-y)*(log((x-y)**2+1))
    """

    def __init__(self, sleep_time: float = 0.0) -> None:
        super().__init__(solver_module_path="solver_examples/solver_examples.py",
                         solver_module_name="solver_examples",
                         solver_class_name="Solver_illustrative_local",
                         solver_parameters={"sleep_time": sleep_time})


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
        super().__init__(solver_module_path="solver_examples/solver_examples.py",
                         solver_module_name="solver_examples",
                         solver_class_name="Solver_linela2exp_local",
                         solver_parameters={"f": f, "length": length, "m": m})


class SolverSpecExampleNonlinear(SolverSpec):
    """
    no_parameters = 4, no_observation = 1
    parameters = [x,y]
    coef = [a1,b1,c1,a2,b2,c2]
    observation = min(a1*x+b1*y+c1,a2*x+b2*y+c2)
    """

    def __init__(self, coef: List[float] = [1.0, -1.0, 0.0, -1.0, 1.0, 0.0]) -> None:
        super().__init__(solver_module_path="solver_examples/solver_examples.py",
                         solver_module_name="solver_examples",
                         solver_class_name="Nonlinear",
                         solver_parameters={"coef": coef})


class SolverSpecGeneric(SolverSpec):
    """
    Serves only for test purposes.
    Takes user-specified number of parameters (no_parameters),
    returns their mean in the form of a constant vector of user-specified length (no_observations).
    """

    def __init__(self, no_parameters: int = 3, no_observations: int = 2) -> None:
        super().__init__(solver_module_path="solver_examples/solver_examples.py",
                         solver_module_name="solver_examples",
                         solver_class_name="Generic",
                         solver_parameters={"no_parameters": no_parameters, "no_observations": no_observations})


class SolverSpecSinProdGeneric(SolverSpec):
    """
    Serves only for test purposes.
    Takes user-specified number of parameters (no_parameters),
    returns constant vector of user-specified length (no_observations).
    """

    def __init__(self, no_parameters: int = 3, no_observations: int = 1, sleep: float = 0.0) -> None:
        super().__init__(solver_module_path="solver_examples/solver_examples.py",
                         solver_module_name="solver_examples",
                         solver_class_name="SinProdGeneric",
                         solver_parameters={"no_parameters": no_parameters, "no_observations": no_observations, "sleep": sleep})


class SolverSpecLinearGaussian(SolverSpec):
    """
    no_parameters = 2, no_observations = 2
    observation = A @ parameters, A = [[1, 0.5], [0, 1]] (fixed)

    Genuinely linear forward model; used by template_experiment.py as the canonical
    starting point (closed-form posterior for a Gaussian prior + Gaussian noise).
    """

    def __init__(self) -> None:
        super().__init__(solver_module_path="solver_examples/solver_examples.py",
                         solver_module_name="solver_examples",
                         solver_class_name="LinearGaussianSolver",
                         solver_parameters={})


class SolverSpecNonlinearGeneric(SolverSpec):
    """
    Serves only for test purposes.
    Takes user-specified number of parameters (no_parameters),
    returns constant vector of user-specified length (no_observations).
    """

    def __init__(self, no_parameters: int = 3, no_observations: int = 1, sleep: float = 0.0) -> None:
        super().__init__(solver_module_path="solver_examples/solver_examples.py",
                         solver_module_name="solver_examples",
                         solver_class_name="NonlinearGeneric",
                         solver_parameters={"no_parameters": no_parameters, "no_observations": no_observations, "sleep": sleep})
