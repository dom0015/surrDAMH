#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
from dataclasses import dataclass


@dataclass
class SolverSpec:
    """
    Recipe for constructing a ``Solver`` out-of-process: loaded by file path rather than
    passed as a live object, because it must be picklable/broadcastable to spawned
    solver-pool children (``get_solver_from_spec`` uses
    ``importlib.util.spec_from_file_location`` to import the module fresh in each
    child, then instantiates ``solver_class_name(**solver_parameters, solver_id=...,
    output_dir=...)``).

    ``solver_module_path`` may be given relative to the launching process's working
    directory (as the ``toy_examples`` solver specs do, e.g.
    ``"solver_examples/solver_examples.py"`` with the documented ``cd toy_examples``): it is
    made **absolute at construction** (:meth:`resolve_module_path`, called from
    ``__post_init__`` and again by ``SamplingFramework.__init__`` for subclasses that define
    their own ``__init__``), so what is broadcast to the spawned children is an absolute
    path. Children do not inherit the parent's ``sys.path`` and are not guaranteed to
    inherit its working directory (finding M18; this is also why
    ``Configuration.paths_to_append`` does not help spawned children, see
    ``docs/writing_a_solver.md``) — hence the resolution has to happen on the launching
    rank, not in the child.

    Args:
        solver_module_path: path to the ``.py`` file defining the solver class; relative
            paths are resolved against the working directory of the process that
            constructs the spec.
        solver_module_name: name under which the module is registered (only used
            internally by ``importlib``, does not need to match any installed package).
        solver_class_name: name of the ``Solver`` subclass inside that module.
        solver_parameters: keyword arguments forwarded to the class constructor
            (``solver_id``/``output_dir`` are added automatically by the caller, do not
            include them here).
    """
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
