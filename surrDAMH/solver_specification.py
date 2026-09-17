#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from dataclasses import dataclass


@dataclass
class SolverSpec:
    """
    Recipe for constructing a ``Solver`` out-of-process: loaded by file path rather than
    passed as a live object, because it must be picklable/broadcastable to spawned
    solver-pool children (``get_solver_from_spec`` uses
    ``importlib.util.spec_from_file_location`` to import the module fresh in each
    child, then instantiates ``solver_class_name(**solver_parameters, solver_id=...,
    output_dir=...)``). Use an absolute ``solver_module_path`` — spawned children do not
    inherit the parent's working directory or ``sys.path`` (finding M18; this is also
    why the ``Configuration.paths_to_append`` mechanism does not help spawned children,
    see ``docs/writing_a_solver.md``).

    Args:
        solver_module_path: absolute path to the ``.py`` file defining the solver class.
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
