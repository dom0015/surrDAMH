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
    Base class for the forward model ``G``: parameters in, observations out.

    To plug in your own model, subclass this and implement ``get_observations``; keep the
    default ``set_parameters`` (it stores the array in ``self.parameters`` and checks its
    size) unless you need to preprocess the parameters. Set ``no_parameters`` and
    ``no_observations``, e.g. via ``super().__init__``; ``Configuration``, the prior and the
    likelihood must use the same two numbers::

        class MySolver(surrDAMH.Solver):
            def __init__(self, solver_id=0, output_dir=None, my_option=1.0):
                super().__init__(solver_id, output_dir, no_parameters=3, no_observations=2)
                ...
            def get_observations(self):
                return G(self.parameters)          # array of shape (no_observations,)

    Keep the ``solver_id``/``output_dir`` keyword arguments in ``__init__``: the library
    passes them whenever it constructs the solver itself (from a ``SolverSpec``).

    Three ways to hand the model to ``surrDAMH.Problem(prior, likelihood, solver=...)``:

    - ``solver=MySolver(...)``: simplest; requires
      ``Configuration.use_solvers_pool=False`` (the solver then runs inside every sampler
      process). Every sampler rank runs the script and builds its own instance, so keep the
      constructor deterministic (seed any random data).
    - ``solver=SolverSpec(...)`` naming the ``.py`` file and class: required with
      ``use_solvers_pool=True`` (spawned solver processes import the file themselves), also
      fine without the pool. See ``surrDAMH.solver_specification.SolverSpec``.
    - a ready-made example from ``toy_examples/solver_examples/solver_spec_examples.py``.

    If the solver module imports code from other directories, add them to ``PYTHONPATH``;
    ``Configuration.paths_to_append`` reaches only the launching process, not spawned solver
    processes.

    Parameters arrive in physical space (after ``prior.transform``). An instance is called
    many times with different parameters and must not keep state between calls beyond what
    ``set_parameters`` sets. Optional extras picked up by ``SamplingRun.write_report``:
    ``visualize_solution()`` (figures of the best fit) and a ``par_names`` attribute (parameter
    names). Full contract: ``docs/writing_a_solver.md``.
    """

    #: Number of parameters the model takes. Declared, not assigned: a type checker therefore
    #: knows every solver has it as an ``int``, and a solver that forgets to set it fails with a
    #: plain ``AttributeError`` at the attribute instead of passing ``None`` on to the sampler.
    no_parameters: int
    #: Number of values ``get_observations`` returns. Declared, not assigned (see ``no_parameters``).
    no_observations: int

    def __init__(self, solver_id: int = 0, output_dir: str | None = None,
                 no_parameters: int | None = None, no_observations: int | None = None) -> None:
        """
        Args:
            solver_id: identifies this instance (MPI rank of the spawned child, or of the
                sampler running it locally); key any scratch files by it, several instances
                may run concurrently.
            output_dir: directory this instance may write scratch/output files to, e.g.
                ``<output_dir>/solver_output/rank<k>/``. Created the first time
                ``self.output_dir`` is read, so a solver that never uses it leaves nothing behind.
            no_parameters: number of parameters; sets the attribute when given. Leave it out
                only if the subclass assigns ``self.no_parameters`` itself.
            no_observations: number of observations; same.
        """
        self.solver_id = solver_id
        self.output_dir = output_dir  # stored via the property setter; created on first read
        if no_parameters is not None:
            self.no_parameters = no_parameters
        if no_observations is not None:
            self.no_observations = no_observations

    @property
    def output_dir(self) -> str | None:
        """
        Directory this solver may write to, or ``None`` if none was given. The directory is
        created on the first read (2026-09-22), not when the solver is constructed: the library
        hands every spawned or per-rank solver its own ``solver_output/rank<k>/`` path, and a
        solver that never writes should not leave an empty directory behind.
        """
        path = getattr(self, "_output_dir", None)
        if path is not None and not getattr(self, "_output_dir_created", False):
            os.makedirs(path, exist_ok=True)
            self._output_dir_created = True
        return path

    @output_dir.setter
    def output_dir(self, path: str | None) -> None:
        self._output_dir = path
        self._output_dir_created = False

    def set_parameters(self, parameters: npt.NDArray) -> None:
        """
        Store ``parameters`` (physical space, i.e. after ``prior.transform``) in
        ``self.parameters`` for the next ``get_observations()`` call. If ``no_parameters``
        is set, a size mismatch raises ``ValueError``. Override only if the model needs the
        parameters in another form.
        """
        parameters = np.asarray(parameters)
        # getattr, because no_parameters is declared but not assigned on the base class: a solver
        # that never set it simply gets no size check here (it fails later at the attribute itself)
        no_parameters = getattr(self, "no_parameters", None)
        if no_parameters is not None and parameters.size != no_parameters:
            raise ValueError(f"{type(self).__name__}.set_parameters: expected {no_parameters} "
                             f"parameters, got an array of shape {parameters.shape}")
        self.parameters = parameters

    def get_observations(self) -> npt.NDArray:
        """
        Run the model for the parameters given to the last ``set_parameters`` call.

        Returns:
            Array of shape ``(no_observations,)``. A model that can fail may return
            ``(observations, tag)`` instead (detected automatically); ``tag < 0`` marks a
            failed solve (the sample then gets zero likelihood), ``tag = -2`` is reserved.
        """
        raise NotImplementedError

    def set_parameters_and_get_observations(self, parameters: npt.NDArray):
        """``set_parameters(parameters)`` followed by ``get_observations()``."""
        self.set_parameters(parameters)
        return self.get_observations()

    def visualize_solution(self, show: bool = False) -> list[tuple[Any, Any]]:
        """
        Optional: figures of the current solution, shown in the HTML report for the
        best-fit sample. Return a list of matplotlib ``(figure, axes)`` pairs; the caller
        saves and closes them. Default: none.
        """
        return []

    def __call__(self, parameters: npt.NDArray) -> npt.NDArray:
        """Same as ``set_parameters_and_get_observations(parameters)``."""
        self.set_parameters(parameters)
        return self.get_observations()


def check_observations_shape(observations, no_observations: int, solver_name: str) -> None:
    """
    C3 (2026-10-08): raise ``ValueError`` unless ``observations`` has shape
    ``(no_observations,)``, the ``Solver.get_observations`` contract. Called on the first
    successful evaluation of a stage (sampler) or of a spawned solver child.
    """
    returned = np.shape(observations)
    if returned != (int(no_observations),):
        raise ValueError(f"{solver_name}.get_observations() returned shape {returned}, expected "
                         f"({int(no_observations)},) (no_observations={int(no_observations)}); return a flat "
                         "array with one value per observation")


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
    """
    Observations of the exact model at ``parameters``, to build a likelihood from (synthetic
    data)::

        observations = surrDAMH.solvers.calculate_artificial_observations(
            solver_instance=my_solver, parameters=[-2, 2])
        likelihood = surrDAMH.distributions.Normal(mean=observations, sd=1.0)

    Args:
        parameters: physical-space parameters; anything array-like, e.g. a list.
        solver_instance: a ready solver, or
        solver_spec: a spec to build one from (exactly one of the two).

    Returns:
        Flat ``(no_observations,)`` array, also for a solver returning a scalar.
    """
    if solver_instance is None:
        assert solver_spec is not None, "to calculate artificial observations, solver must be given"
        solver_instance = get_solver_from_spec(solver_spec)
    solver_instance.set_parameters(np.asarray(parameters))
    observations = solver_instance.get_observations()
    return np.asarray(observations).ravel()
