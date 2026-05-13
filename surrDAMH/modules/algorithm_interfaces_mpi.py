#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MPI-backed implementations of the backend-neutral algorithm interfaces.

This module wraps the existing sampler-side MPI communication classes from
``surrDAMH.modules.communication`` behind the newer interface layer defined in
``surrDAMH.modules.algorithm_interfaces``.

Purpose:
- preserve current behaviour,
- keep MPI-specific details out of future algorithm code,
- provide a migration path without modifying the original MPI classes.

These adapters are intentionally thin. They do not redesign the old MPI logic;
they only expose it through clearer, backend-neutral roles.

Implemented roles:
- ``MpiObservationProvider`` / ``MpiSolverPoolObservationProvider``
- ``MpiSnapshotSink``
- ``MpiEvaluatorProvider``

TODO:
- Once algorithms are refactored to depend on the new interfaces, consider
  moving MPI-specific lifecycle methods into a dedicated orchestration layer.
- If collector-side abstractions are introduced later, create matching MPI
  wrappers there as well.
- Standardize shutdown semantics across local and MPI implementations.
"""

from __future__ import annotations

from typing import Any

from surrDAMH.configuration import Configuration
from surrDAMH.modules.algorithm_interfaces import (
    EvaluatorProvider,
    ObservationProvider,
    ObservationResult,
    SnapshotCollector,
)
from surrDAMH.modules.communication import (
    CommEvaluator_sampler,
    CommSnapshot_sampler,
    Communicator,
    SolverMPI,
)
from surrDAMH.surrogates.parent import Evaluator


class MpiObservationProvider(ObservationProvider):
    """
    Adapter exposing an existing MPI-capable communicator as an
    ``ObservationProvider``.

    This is the thinnest possible bridge around the existing sampler-side
    ``Communicator`` / ``SolverMPI`` classes.

    Notes:
    - Behaviour is intentionally unchanged.
    - Result format is whatever the wrapped communicator already returns.

    TODO:
    - Replace use of the old ``Communicator`` base class with a cleaner, shared
      result object when the solver interface is modernized.
    """

    def __init__(self, communicator: Communicator) -> None:
        self.communicator = communicator

    def set_parameters(self, parameters) -> None:
        self.communicator.set_parameters(parameters)

    def get_observations(self) -> ObservationResult:
        return self.communicator.get_observations()

    def terminate(self) -> None:
        """Forward optional MPI shutdown to the wrapped communicator."""
        terminate = getattr(self.communicator, "terminate", None)
        if callable(terminate):
            terminate()


class MpiSolverPoolObservationProvider(MpiObservationProvider):
    """
    Convenience adapter that builds a ``SolverMPI`` from ``Configuration``.

    This corresponds to the current sampler-side behaviour in
    ``process_SAMPLER.py`` when ``conf.use_solvers_pool`` is true.
    """

    def __init__(self, conf: Configuration) -> None:
        super().__init__(communicator=SolverMPI(conf=conf))


class MpiSnapshotSink(SnapshotCollector):
    """
    Adapter exposing ``CommSnapshot_sampler`` as a backend-neutral snapshot sink.

    Functionality is identical to the current MPI sampler-side snapshot sender.
    """

    def __init__(self, communicator: CommSnapshot_sampler) -> None:
        self.communicator = communicator

    @classmethod
    def from_rank_collector(
        cls,
        rank_collector: int,
        max_sampler_isend_requests: int = 100,
    ) -> "MpiSnapshotSink":
        """Build the wrapped MPI communicator using the original constructor."""
        return cls(
            communicator=CommSnapshot_sampler(
                rank_collector=rank_collector,
                max_sampler_isend_requests=max_sampler_isend_requests,
            )
        )

    def send_to_collector(self, data: list[Any]) -> None:
        self.communicator.send_to_collector(data)

    def terminate(self) -> None:
        """Forward MPI finalization to the wrapped snapshot communicator."""
        self.communicator.terminate()


class MpiEvaluatorProvider(EvaluatorProvider):
    """
    Adapter exposing ``CommEvaluator_sampler`` as an ``EvaluatorProvider``.

    This preserves the current algorithm-facing semantics:
    - explicit ``request_evaluator()``
    - non-blocking ``evaluator_is_available()`` polling
    - blocking ``get_evaluator()`` retrieval

    The wrapped MPI communicator already keeps track of ``evaluator``; this
    adapter exposes the same state through a property to remain compatible with
    the new interface.

    TODO:
    - Revisit whether future code should still require an explicit request step,
      or whether a pull-based model would be simpler.
    """

    def __init__(self, communicator: CommEvaluator_sampler) -> None:
        self.communicator = communicator

    @classmethod
    def from_rank_collector(
        cls,
        rank_collector: int,
        max_buffer_size: int = 1 << 30,
        request_initial_evaluator: bool = False,
    ) -> "MpiEvaluatorProvider":
        """
        Build the wrapped MPI communicator using the original constructor.

        Args:
            rank_collector: MPI rank of the collector.
            max_buffer_size: same meaning as in ``CommEvaluator_sampler``.
            request_initial_evaluator: if true, perform the initial request
                immediately, mirroring the current startup pattern from
                ``process_SAMPLER.py``.
        """
        provider = cls(
            communicator=CommEvaluator_sampler(
                rank_collector=rank_collector,
                max_buffer_size=max_buffer_size,
            )
        )
        if request_initial_evaluator:
            provider.request_evaluator()
        return provider

    @property
    def evaluator(self) -> Evaluator | None:
        return self.communicator.evaluator

    @evaluator.setter
    def evaluator(self, value: Evaluator | None) -> None:
        self.communicator.evaluator = value

    def request_evaluator(self) -> None:
        self.communicator.request_evaluator()

    def evaluator_is_available(self) -> bool:
        return self.communicator.evaluator_is_available()

    def get_evaluator(self) -> Evaluator:
        return self.communicator.get_evaluator()

    def get_evaluator_and_terminate(self) -> Evaluator | None:
        """
        Preserve the original MPI shutdown method for orchestration code.

        This is intentionally not part of the backend-neutral protocol, because
        current algorithms do not require it directly.
        """
        return self.communicator.get_evaluator_and_terminate()

    def close(self) -> Evaluator | None:
        """
        Friendly lifecycle alias for orchestration code.

        Returns the last available evaluator, matching the wrapped MPI method.
        """
        return self.get_evaluator_and_terminate()


__all__ = [
    "MpiObservationProvider",
    "MpiSolverPoolObservationProvider",
    "MpiSnapshotSink",
    "MpiEvaluatorProvider",
]
