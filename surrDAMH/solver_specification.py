#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass


@dataclass
class SolverSpec:
    """
    Tells the library how to build your ``Solver`` in another process: the ``.py`` file,
    the class name and the constructor arguments. Required with
    ``Configuration.use_solvers_pool=True`` (spawned solver processes import the file
    themselves); with ``use_solvers_pool=False`` you may pass either this or a ready
    ``solver_instance`` to ``SamplingFramework``::

        spec = SolverSpec(solver_module_path="my_solver.py", solver_module_name="my_solver",
                          solver_class_name="MySolver", solver_parameters={"my_option": 1.0})

    The class must live in an importable ``.py`` file. That file may be the sampling script
    itself only if everything except the class definition sits under
    ``if __name__ == "__main__":`` (the file is imported afresh in every solver process).
    ``surrDAMH.solvers.get_solver_from_spec(spec)`` turns a spec into an instance; the
    reverse is not possible, a live object cannot be shipped to spawned processes.
    Ready-made specs for the example solvers: ``toy_examples/solver_examples/solver_spec_examples.py``.

    Args:
        solver_module_path: path to the ``.py`` file defining the class. A relative path is
            resolved against the working directory at construction time (build the spec
            before any ``os.chdir``).
        solver_module_name: name the module is imported under; any valid identifier.
        solver_class_name: name of the ``Solver`` subclass inside that file.
        solver_parameters: keyword arguments for the constructor. ``solver_id`` and
            ``output_dir`` are added by the library, do not include them.

    If the solver module imports code from other directories, put them on ``PYTHONPATH``;
    ``Configuration.paths_to_append`` does not reach spawned solver processes.
    """
    # Maintainer note (M18/WS5): the path is made absolute here, on the launching rank,
    # because process_SOLVER broadcasts this object to spawned children that inherit neither
    # sys.path nor (reliably) the working directory. resolve_module_path() is also called by
    # SamplingFramework.__init__ for subclasses that define their own __init__ and thereby
    # skip __post_init__ (see the toy_examples SolverSpec* classes).
    solver_module_path: str
    solver_module_name: str
    solver_class_name: str
    solver_parameters: dict

    def __post_init__(self) -> None:
        self.resolve_module_path()

    def resolve_module_path(self) -> str:
        """
        Make ``solver_module_path`` absolute (in place) and return it.

        Idempotent, and safe to call from any rank that still has the launching working
        directory. Subclasses that define their own ``__init__`` (the ``toy_examples``
        ``SolverSpec*`` classes used to, before WS5) never run ``__post_init__``, which is
        why ``SamplingFramework.__init__`` calls this explicitly as well.
        """
        self.solver_module_path = os.path.abspath(self.solver_module_path)
        return self.solver_module_path
