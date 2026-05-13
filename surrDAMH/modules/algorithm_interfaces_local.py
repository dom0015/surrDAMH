#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Local / in-process implementations of the backend-neutral algorithm interfaces.

This module is intentionally standalone-friendly:
- no MPI imports,
- no process/rank assumptions,
- synchronous control flow.

It is meant as the next step after introducing
``surrDAMH.modules.algorithm_interfaces``.

The classes here are small building blocks that can later be used by a
standalone sampling runner without modifying the algorithmic core.

Provided implementations:
- ``LocalSolverAdapter``: wraps a local ``Solver`` as an ``ObservationProvider``
- ``NoOpSnapshotSink``: discards snapshots
- ``InMemorySnapshotSink``: stores snapshots in memory
- ``LocalEvaluatorProvider``: simple evaluator holder / swapper
- ``LocalSurrogateManager``: in-process snapshot sink + evaluator provider

TODO:
- Add a dedicated standalone runner that wires these services into MH / DAMH.
- Decide whether ``LocalSurrogateManager`` should stay synchronous or gain a
  buffered / batched update mode.
- Replace the loose snapshot payload with a shared snapshot dataclass in the
  wider codebase.
- If standalone multi-chain execution is added later, extract reduction /
  aggregation logic into a separate service instead of growing this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from surrDAMH.modules.algorithm_interfaces import (
    EvaluatorProvider,
    ObservationProvider,
    SnapshotCollector,
)
from surrDAMH.solvers import Solver
from surrDAMH.surrogates.parent import Evaluator, Updater


@dataclass
class LocalSnapshot:
    """
    Normalized local representation of one sampler snapshot.

    Current algorithms send snapshots as loose lists of the form:
    ``[parameters, observations, weight]``.

    This dataclass does not change that external contract; it only makes the
    local implementations easier to read and maintain.

    TODO:
    - Promote this concept into a shared snapshot type once algorithms and
      backends are refactored to use structured payloads directly.
    """

    parameters: npt.NDArray
    observations: npt.NDArray
    weight: float


def _to_1d_array(values: npt.ArrayLike) -> npt.NDArray:
    """Convert input into a copied 1D numpy array."""
    arr = np.asarray(values, dtype=float)
    return np.atleast_1d(arr).copy()


def _normalize_snapshot(data: list[Any]) -> LocalSnapshot:
    """
    Normalize the current loose snapshot payload into ``LocalSnapshot``.

    Expected payload shape: ``[parameters, observations, weight]``.
    """
    if len(data) != 3:
        raise ValueError(
            "Snapshot payload must have exactly 3 items: "
            "[parameters, observations, weight]"
        )
    parameters, observations, weight = data
    return LocalSnapshot(
        parameters=_to_1d_array(parameters),
        observations=_to_1d_array(observations),
        weight=float(weight),
    )


class LocalSolverAdapter(ObservationProvider):
    """
    Wrap a local ``Solver`` instance as an ``ObservationProvider``.

    This mirrors the minimal behaviour expected by the current algorithms:
    parameters are first supplied through ``set_parameters()``, then evaluated
    by ``get_observations()``.

    Notes:
    - The wrapped solver may return observations only, or
      ``(observations, solver_tag)``.
    - This adapter intentionally does not modify that behaviour.

    TODO:
    - If solver result handling is standardized later, normalize the return
      value here into a single result type.
    """

    def __init__(self, solver: Solver) -> None:
        self.solver = solver

    def set_parameters(self, parameters: npt.ArrayLike) -> None:
        self.solver.set_parameters(parameters)

    def get_observations(self):
        return self.solver.get_observations()


class NoOpSnapshotSink(SnapshotCollector):
    """Snapshot sink that ignores all incoming snapshots."""

    def send_to_collector(self, data: list[Any]) -> None:
        return None


class InMemorySnapshotSink(SnapshotCollector):
    """
    Store all snapshots in memory.

    This is useful for:
    - standalone experiments,
    - testing,
    - later in-process surrogate updates.
    """

    def __init__(self) -> None:
        self.snapshots: list[LocalSnapshot] = []

    def send_to_collector(self, data: list[Any]) -> None:
        self.snapshots.append(_normalize_snapshot(data))

    def clear(self) -> None:
        self.snapshots.clear()

    def __len__(self) -> int:
        return len(self.snapshots)

    def get_snapshot_arrays(self) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
        """
        Return stored snapshots stacked into arrays.

        Returns:
            parameters: shape ``(n_snapshots, no_parameters)``
            observations: shape ``(n_snapshots, no_observations)``
            weights: shape ``(n_snapshots, 1)``
        """
        if not self.snapshots:
            return (
                np.empty((0, 0)),
                np.empty((0, 0)),
                np.empty((0, 1)),
            )

        parameters = np.vstack([snapshot.parameters for snapshot in self.snapshots])
        observations = np.vstack([snapshot.observations for snapshot in self.snapshots])
        weights = np.array([[snapshot.weight] for snapshot in self.snapshots], dtype=float)
        return parameters, observations, weights


class LocalEvaluatorProvider(EvaluatorProvider):
    """
    Simplest local implementation of ``EvaluatorProvider``.

    It stores:
    - the current evaluator,
    - an optional pending evaluator that will replace the current one when
      requested via ``get_evaluator()``.

    This is intentionally simpler than the MPI version. There is no message
    passing; instead, orchestration code can call ``set_evaluator()`` whenever
    a new evaluator becomes available.

    TODO:
    - Decide whether the explicit request/poll semantics should be preserved in
      standalone mode, or whether a simpler "always get latest evaluator"
      approach would be clearer.
    """

    def __init__(self, evaluator: Evaluator | None = None) -> None:
        self.evaluator = evaluator
        self._pending_evaluator: Evaluator | None = None
        self._request_pending = False

    def set_evaluator(self, evaluator: Evaluator) -> None:
        """Queue a new evaluator to be exposed on the next ``get_evaluator()`` call."""
        self._pending_evaluator = evaluator

    def request_evaluator(self) -> None:
        self._request_pending = True

    def evaluator_is_available(self) -> bool:
        return self._pending_evaluator is not None or self.evaluator is not None

    def get_evaluator(self) -> Evaluator:
        if self._pending_evaluator is not None:
            self.evaluator = self._pending_evaluator
            self._pending_evaluator = None
        if self.evaluator is None:
            raise RuntimeError("No evaluator is currently available.")
        self._request_pending = False
        return self.evaluator

    def close(self) -> None:
        """Optional no-op lifecycle hook for symmetry with future backends."""
        self._pending_evaluator = None
        self._request_pending = False


class LocalSurrogateManager(SnapshotCollector, EvaluatorProvider):
    """
    In-process surrogate manager for standalone DAMH-style workflows.

    This class combines the two responsibilities that are currently split across
    MPI sampler/collector communication:
    - consume snapshots,
    - train/update an ``Updater``,
    - provide the resulting ``Evaluator`` to the algorithm.

    Behaviour:
    - snapshots are added synchronously,
    - surrogate retraining is triggered immediately once the configured
      thresholds are met,
    - the newest evaluator is exposed through the ``EvaluatorProvider`` API.

    This is deliberately simpler than the current collector loop in
    ``process_COLLECTOR.py``. It is not a drop-in replacement for MPI
    orchestration yet, but it captures the same *core local responsibility*.

    TODO:
    - Consider adding optional batching to better mimic collector behaviour.
    - Consider keeping quality-monitoring metrics here in the future.
    - If one wants exact parity with the collector, introduce a separate local
      runner that calls training only at specific synchronization points.
    """

    def __init__(
        self,
        updater: Updater,
        min_snapshots_initial: int = 1,
        min_snapshots_to_update: int = 1,
        delayed_init_data: Any | None = None,
        initial_snapshots: tuple[npt.NDArray, npt.NDArray, npt.NDArray] | None = None,
    ) -> None:
        self.updater = updater
        self.min_snapshots_initial = min_snapshots_initial
        self.min_snapshots_to_update = min_snapshots_to_update

        self.evaluator: Evaluator | None = None
        self._pending_evaluator: Evaluator | None = None
        self._request_pending = False

        self.no_snapshots_total = 0
        self.no_snapshots_used = 0
        self.snapshot_count_since_last_update = 0

        self.updater.delayed_init(delayed_init_data)

        if initial_snapshots is not None:
            parameters, observations, weights = initial_snapshots
            if len(parameters) > 0:
                self.updater.add_data(parameters, observations, weights)
                self.no_snapshots_total = int(len(parameters))
                self._maybe_update_evaluator()

    def request_evaluator(self) -> None:
        self._request_pending = True

    def evaluator_is_available(self) -> bool:
        return self._pending_evaluator is not None or self.evaluator is not None

    def get_evaluator(self) -> Evaluator:
        if self._pending_evaluator is not None:
            self.evaluator = self._pending_evaluator
            self._pending_evaluator = None
        if self.evaluator is None:
            raise RuntimeError(
                "No surrogate evaluator is available yet. "
                "Collect more snapshots or lower the initialization threshold."
            )
        self._request_pending = False
        return self.evaluator

    def send_to_collector(self, data: list[Any]) -> None:
        snapshot = _normalize_snapshot(data)
        parameters = snapshot.parameters.reshape((1, -1))
        observations = snapshot.observations.reshape((1, -1))
        weights = np.array([[snapshot.weight]], dtype=float)

        self.updater.add_data(parameters, observations, weights)
        self.no_snapshots_total += 1
        self.snapshot_count_since_last_update += 1
        self._maybe_update_evaluator()

    def _maybe_update_evaluator(self) -> None:
        cond_init = self.no_snapshots_used == 0 and self.no_snapshots_total >= self.min_snapshots_initial
        cond_update = (
            self.no_snapshots_used > 0
            and self.no_snapshots_total - self.no_snapshots_used >= self.min_snapshots_to_update
        )

        if cond_init or cond_update:
            self.updater.train()
            self.no_snapshots_used = self.no_snapshots_total
            self.snapshot_count_since_last_update = 0
            self._pending_evaluator = self.updater.get_evaluator()

    def close(self) -> None:
        """Optional no-op lifecycle hook for future standalone runners."""
        self._pending_evaluator = None
        self._request_pending = False


__all__ = [
    "LocalSnapshot",
    "LocalSolverAdapter",
    "NoOpSnapshotSink",
    "InMemorySnapshotSink",
    "LocalEvaluatorProvider",
    "LocalSurrogateManager",
]
