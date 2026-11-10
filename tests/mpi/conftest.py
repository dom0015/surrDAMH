#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Shared helpers for the ``tests/mpi`` integration tests (WS1, ``library_notes/07_testing_plan.md``
Phase 3, items I1-I15).

Each test writes a **self-contained driver script** into ``tmp_path`` and runs it as

    mpiexec -n <k> <python> -m mpi4py <driver.py> [args]

in a subprocess under a timeout, then asserts on the exit code, on stdout and on the
files produced in ``<output_dir>/sampling_output`` (layout: ``library_notes/00_overview.md`` §10).

Why driver files instead of in-process MPI: the solvers pool spawns children with
``MPI_Comm_spawn`` and those children import the solver class from
``SolverSpec.solver_module_path`` with ``importlib``, i.e. from a **real module file** and
with an **absolute path** (``paths_to_append`` does not reach spawned children, finding M18).
``write_solver_module()`` therefore drops a small module with the test solvers into
``tmp_path`` and the drivers point ``SolverSpec`` at it.

Process-tree cleanup: some of the pinned behaviours are deadlocks (I4), so ``run_mpi``
starts ``mpiexec`` in its own session (``start_new_session=True``) and kills the whole
process group on timeout; otherwise hydra proxies, samplers and spawned solver children
survive the test run.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import textwrap
from dataclasses import dataclass, field
from pathlib import Path

import pytest

MPIEXEC = "/usr/local/bin/mpiexec"  # MPICH/Hydra; /dolfinx-env/bin/mpiexec does not exist
HAS_MPIEXEC = os.path.exists(MPIEXEC)
requires_mpiexec = pytest.mark.skipif(not HAS_MPIEXEC, reason=f"{MPIEXEC} not found")


@dataclass
class MpiRunResult:
    """Outcome of one ``mpiexec`` subprocess (``CompletedProcess``-like, plus ``timed_out``)."""

    command: list[str]
    returncode: int | None
    stdout: str
    stderr: str
    timed_out: bool
    duration: float
    orphans_killed: bool = False
    _tail: int = field(default=4000, repr=False)

    def describe(self) -> str:
        """Compact description used in assertion messages."""
        return (f"command: {' '.join(self.command)}\n"
                f"returncode: {self.returncode}, timed_out: {self.timed_out}, duration: {self.duration:.1f}s\n"
                f"--- stdout (tail) ---\n{self.stdout[-self._tail:]}\n"
                f"--- stderr (tail) ---\n{self.stderr[-self._tail:]}")


def _run_mpi(script_path, n: int, timeout: float = 120.0, args: list[str] | None = None,
             cwd: str | os.PathLike | None = None, env: dict | None = None) -> MpiRunResult:
    """
    Run ``mpiexec -n <n> <python> -m mpi4py <script_path> [args]`` under a timeout.

    On timeout the whole process group is killed (SIGKILL), so that hung samplers,
    the hydra proxy and any ``MPI_Comm_spawn``-ed solver children do not outlive the test.
    """
    import time

    script_path = str(script_path)
    command = [MPIEXEC, "-n", str(n), sys.executable, "-m", "mpi4py", script_path] + list(args or [])
    run_env = dict(os.environ)
    if env:
        run_env.update(env)
    if cwd is None:
        cwd = os.path.dirname(script_path)

    started = time.monotonic()
    process = subprocess.Popen(command, cwd=str(cwd), env=run_env, text=True,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                               start_new_session=True)  # own process group -> killable as a tree
    timed_out = False
    orphans_killed = False
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        timed_out = True
        orphans_killed = _kill_process_group(process)
        try:
            stdout, stderr = process.communicate(timeout=30)
        except subprocess.TimeoutExpired:  # pragma: no cover - only if SIGKILL did not take
            process.kill()
            stdout, stderr = process.communicate()
    duration = time.monotonic() - started
    return MpiRunResult(command=command, returncode=process.returncode, stdout=stdout or "",
                        stderr=stderr or "", timed_out=timed_out, duration=duration,
                        orphans_killed=orphans_killed)


def _kill_process_group(process: subprocess.Popen) -> bool:
    """SIGKILL the process group of ``process``; returns True if a signal was delivered."""
    try:
        pgid = os.getpgid(process.pid)
    except ProcessLookupError:  # pragma: no cover - process already gone
        return False
    try:
        os.killpg(pgid, signal.SIGKILL)
        return True
    except ProcessLookupError:  # pragma: no cover
        return False


@pytest.fixture
def run_mpi():
    """Fixture exposing :func:`_run_mpi` (see its docstring for the signature)."""
    return _run_mpi


SOLVER_MODULE_SOURCE = '''\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Solvers and problem definition shared by the ``tests/mpi`` driver scripts.

Written into ``tmp_path`` by ``tests/mpi/conftest.write_solver_module``; it must be a real
file because the solvers pool spawns children that import it with ``importlib`` from
``SolverSpec.solver_module_path``.

The problem is the linear-Gaussian toy problem of ``tests/conftest.py``
(``G(x) = A @ x``, ``no_parameters = no_observations = 2``, standard normal prior, so
``prior.transform`` is the identity and internal space equals physical space).
"""

import numpy as np

import surrDAMH
from surrDAMH.solvers import Solver

A = np.array([[1.0, 0.5], [0.0, 1.0]])
TRUE_PARAMETERS = np.array([0.7, -0.4])
OBSERVED_DATA = A @ TRUE_PARAMETERS
OBS_SD = np.array([0.3, 0.3])

NO_PARAMETERS = 2
NO_OBSERVATIONS = 2


class LinearSolver(Solver):
    """G(x) = A @ x."""

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = NO_PARAMETERS
        self.no_observations = NO_OBSERVATIONS

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        return A @ self.parameters


class TaggedLinearSolver(LinearSolver):
    """As ``LinearSolver`` but returns ``(observations, tag)`` (detected by the child automatically)."""

    def get_observations(self):
        return A @ self.parameters, 0


class TagFailingSolver(Solver):
    """Returns ``(observations, tag)`` with ``tag = -1`` whenever the first parameter is positive.

    Used by I7 to drive a negative solver tag through the two observation transports.
    """

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = NO_PARAMETERS
        self.no_observations = NO_OBSERVATIONS

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        if self.parameters[0] > 0:
            return np.zeros(NO_OBSERVATIONS), -1
        return A @ self.parameters, 0


class LinearSolver1Obs(Solver):
    """G(x) = [x0 + 0.5*x1]: the same model with a single observation.

    Needed by the ``use_only_surrogate`` test: with ``no_observations > 1`` that code path
    dies in ``Normal.calculate_logpdf_uncorrelated`` (finding A20, ``library_notes/01`` -
    the evaluator is called with a 1-D parameter vector and returns a ``(1, no_obs)`` array).
    """

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = NO_PARAMETERS
        self.no_observations = 1

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        return np.array([A[0, :] @ self.parameters])


class FailingSolver(Solver):
    """Raises ``RuntimeError("deliberate solver failure")`` on its 5th ``get_observations`` call.

    Used by the WS8 fail-loud tests: before that change an exception on one rank (or in a
    spawned solver child) left every other rank blocked in a matching MPI call forever.
    """

    FAIL_ON_CALL = 5

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = NO_PARAMETERS
        self.no_observations = NO_OBSERVATIONS
        self.calls = 0

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        self.calls += 1
        if self.calls >= self.FAIL_ON_CALL:
            raise RuntimeError("deliberate solver failure")
        return A @ self.parameters


class EchoSolver(Solver):
    """G(x) = x: the observations are the parameters themselves (used by I8)."""

    def __init__(self, solver_id=0, output_dir=None):
        self.no_parameters = NO_PARAMETERS
        self.no_observations = NO_OBSERVATIONS

    def set_parameters(self, parameters):
        self.parameters = np.asarray(parameters, dtype=float).reshape(-1)

    def get_observations(self):
        return self.parameters.copy()


class FixedSample:
    """Degenerate 'distribution' returning a fixed initial sample (as in tests/test_runner_local.py)."""

    def __init__(self, parameters):
        self.parameters = np.array(parameters, dtype=float)

    def rvs(self):
        return self.parameters.copy()


def make_prior():
    return surrDAMH.distributions.PriorIndependentComponents([
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
        surrDAMH.distributions.NormalComponent(mu=0.0, sigma=1.0),
    ])


def make_likelihood():
    return surrDAMH.distributions.Normal(mean=OBSERVED_DATA, sd=OBS_SD)


def make_likelihood_1obs():
    return surrDAMH.distributions.Normal(mean=float(OBSERVED_DATA[0]), sd=float(OBS_SD[0]))
'''


def write_solver_module(tmp_path) -> Path:
    """Write the shared solver/problem module into ``tmp_path`` and return its absolute path."""
    path = Path(tmp_path) / "solver_module.py"
    path.write_text(SOLVER_MODULE_SOURCE)
    return path.resolve()


DRIVER_HEADER = '''\
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Driver script generated by tests/mpi (see the test that writes it)."""

import os
import sys

SUPPORT_DIR = {support_dir!r}
sys.path.insert(0, SUPPORT_DIR)
SOLVER_MODULE_PATH = os.path.join(SUPPORT_DIR, "solver_module.py")
DEFAULT_OUTPUT_DIR = os.path.join(SUPPORT_DIR, {output_name!r})
OUTPUT_DIR = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_OUTPUT_DIR

import numpy as np  # noqa: E402
import solver_module  # noqa: E402
import surrDAMH  # noqa: E402
from surrDAMH.proposals import Block, Hamiltonian, PCN, RandomWalk  # noqa: E402
from surrDAMH.solver_specification import SolverSpec  # noqa: E402
from surrDAMH.stages import Stage  # noqa: E402


def solver_spec(class_name="LinearSolver"):
    return SolverSpec(solver_module_path=SOLVER_MODULE_PATH, solver_module_name="solver_module",
                      solver_class_name=class_name, solver_parameters={{}})

'''


def write_driver(tmp_path, name: str, body: str) -> Path:
    """
    Write a driver script ``<name>.py`` into ``tmp_path``.

    The generated file starts with a header that puts ``tmp_path`` on ``sys.path``, imports
    ``solver_module`` and defines ``OUTPUT_DIR`` (absolute, inside ``tmp_path``; overridable
    by the first command line argument) and a ``solver_spec()`` helper with an **absolute**
    ``solver_module_path``. ``body`` is appended verbatim.
    """
    support_dir = str(Path(tmp_path).resolve())
    header = DRIVER_HEADER.format(support_dir=support_dir, output_name=f"out_{name}")
    path = Path(tmp_path) / f"{name}.py"
    path.write_text(header + textwrap.dedent(body))
    return path.resolve()


@pytest.fixture
def mpi_workspace(tmp_path):
    """``tmp_path`` with the shared solver module already written into it."""
    write_solver_module(tmp_path)
    return Path(tmp_path).resolve()


# ---------------------------------------------------------------------------
# small helpers for asserting on the produced output files
# ---------------------------------------------------------------------------

def sampling_output(output_dir) -> Path:
    return Path(output_dir) / "sampling_output"


def stage_sample_files(output_dir, stage_name: str) -> list[Path]:
    """Sorted ``samples/<stage_name>/rank%04d.csv`` files."""
    directory = sampling_output(output_dir) / "samples" / stage_name
    if not directory.is_dir():
        return []
    return sorted(directory.glob("rank*.csv"))
