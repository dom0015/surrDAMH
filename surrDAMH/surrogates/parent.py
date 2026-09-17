#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import numpy.typing as npt

from surrDAMH.solvers import Solver


class SurrogateAsSolver(Solver):
    """Exposes an ``Evaluator`` through the ``Solver`` interface (``use_only_surrogate`` stages)."""

    def __init__(self, call_method) -> None:
        self.call_method = call_method

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.parameters = parameters

    def get_observations(self) -> npt.NDArray:
        """
        Returns observations of shape ``(no_observations,)``, i.e. the ``Solver`` contract.

        The evaluator is called on a ``(1, no_parameters)`` batch; its result is flattened,
        because implementations differ (batched evaluators return ``(1, no_observations)``,
        the torch ones return a flattened ``(no_observations,)``, see finding 3.5). Without
        this, a ``(1, no_observations)`` array reached ``Normal.logpdf`` and broke for
        ``no_observations > 1`` (finding A20).
        """
        result = self.call_method(np.atleast_2d(self.parameters))
        return np.asarray(result).reshape(-1)


class Evaluator:
    """
    Parent class for surrogate evaluators, produced by ``Updater.get_evaluator()`` and
    picklable so they can be shipped from the collector to sampler ranks over MPI.

    Today's contract (this base class only documents the interface; it does not enforce
    shapes, and concrete evaluators do not all agree — see the file:line-referenced
    conformance table in ``library_notes/12_evaluator_contract_spec.md`` §1, e.g. the two
    torch evaluators flatten ``__call__``'s output to ``(n*no_observations,)`` instead of
    ``(n, no_observations)``; ``SurrogateAsSolver.get_observations`` compensates for this
    so ``use_only_surrogate`` stages see a uniform ``(no_observations,)`` regardless).
    That table also lists the standardized contract WS6 will migrate to (batch shapes
    always 2-D, ``no_parameters``/``no_observations`` set on this base class) — not
    implemented yet, so code against today's per-class shapes, not the proposed ones.
    """

    def __init__(self) -> None:
        self.no_parameters: int

    def __call__(self, datapoints: npt.NDArray) -> npt.NDArray:
        """
        Evaluates the surrogate model in datapoints.

        datapoints shape: (number of datapoints, no_parameters)

        output NDArray shape: (number of datapoints, no_observations) for the classical
        (polynomial/RBF/kd-tree) evaluators; the two torch evaluators instead return a
        flattened ``(number of datapoints * no_observations,)`` array (see class
        docstring). Not implemented by the base class.
        """
        raise NotImplementedError

    def jacobian(self, datapoints: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Returns ``(J(x), evaluation)`` for ONE sample (convention used by all implementations
        and by ``vjp`` / ``Algorithm.*_compute_surrogate_log_likelihood_gradient``).

        datapoints shape: (no_parameters,)

        output Jacobian shape: (no_observations, no_parameters)
        output evaluation shape: (no_observations,)
        """
        raise NotImplementedError(f"Jacobian is not implemented for {type(self).__name__}")

    def vjp(self, datapoint: npt.NDArray, vector: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        """
        Returns ``(J(x)^T @ vector, evaluation)`` for one sample.

        datapoint shape: (no_parameters,)
        vector shape: (no_observations,)

        output gradient shape: (no_parameters,)
        output evaluation shape: (no_observations,)
        """
        jacobian, evaluation = self.jacobian(datapoint)
        return jacobian.T @ vector, evaluation

    def supports_gradients(self) -> bool:
        """Return whether this evaluator can provide input derivatives."""
        return type(self).jacobian is not Evaluator.jacobian or type(self).vjp is not Evaluator.vjp

    def set_use_gradients(self, enabled: bool) -> None:
        """Optional hook for evaluators with switchable gradient support."""
        return None

    def as_solver(self):
        return SurrogateAsSolver(self.__call__)

    
class Updater:
    """
    Parent class for surrogate model updaters: owns the training data and the
    trainable state (e.g. a fitted sklearn pipeline or a torch model + optimizer), runs
    on the collector rank (or in-process for ``run_local``'s ``LocalSurrogateManager``).
    ``process_COLLECTOR``/``LocalSurrogateManager`` decide *when* to call ``add_data``/
    ``train``/``get_evaluator`` (governed by ``Configuration.min_snapshots_initial``/
    ``min_snapshots_to_update``); the updater itself is passive.

    Which methods each of the five concrete updaters (polynomial/RBF/kd-tree, two torch
    variants) actually implement, including weight handling and persistence, is
    tabulated in ``library_notes/12_evaluator_contract_spec.md`` §1 — several
    ``supports_*`` methods below report ``False`` for functionality that is in fact
    implemented (documented mismatch, not fixed here).
    """

    def __init__(self, no_parameters: int, no_observations: int) -> None:
        self.no_snapshots: int
        pass

    def delayed_init(self, data):
        """
        Additional settings of the surrogate model.
        """
        pass

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray, weights: npt.NDArray) -> None:
        """
        Adds more snapshots to the surrogate model.
        parameters shape: (number of snapshots, no_parameters)
        observations shape: (number of snapshots, no_observations)
        weights shape: (number of snapshots, 1); a rejected proposal is sent with
        weight 0 (multiplicity weighting, ``algorithms.py``). Whether/how a concrete
        updater actually uses ``weights`` (some ignore them outright) is documented per
        class and in ``library_notes/12_evaluator_contract_spec.md`` §1 ("weight handling" row).
        """
        pass

    def train(self,):
        """
        Trains the surrogate model, e.g. neural network.
        Called periodically by collector, regardless of whether new data have been added.
        Several updaters instead (re)train inside ``get_evaluator()`` and leave this a
        no-op — see the "train() semantics" row of the spec referenced in the class docstring.
        """

    def get_evaluator(self) -> Evaluator:
        """
        Called by collector, when a new evaluator is requested by a sampler.
        Returns Evaluator instance.
        """
        raise NotImplementedError

    def supports_gradients(self) -> bool:
        """Return whether evaluators produced by this updater support derivatives."""
        return False

    def set_use_gradients(self, enabled: bool) -> None:
        """Optional hook for updaters with switchable gradient support."""
        return None

    def get_initial_snapshots(self) -> list[npt.NDArray] | None:
        """Snapshots to preload into the collector, if this updater already has training data."""
        return None

    def supports_training_data_persistence(self) -> bool:
        """Return whether this updater can save and load training data."""
        return False

    def supports_state_persistence(self) -> bool:
        """Return whether this updater can save and load the full state of the surrogate model."""
        return False

    def save_training_data(self, path: str) -> None:
        """
        Saves the surrogate model training data to files.
        """
        raise NotImplementedError(
            f"Training-data persistence is not implemented for {type(self).__name__}"
        )
    
    def load_training_data(self, path: str):
        """
        Loads the surrogate model training data from files.
        Returns a list of numpy arrays containing the loaded snapshots and weights.
        """
        raise NotImplementedError(f"Training data persistence is not implemented for {type(self).__name__}")

    def save_state(self, checkpoint_path: str, data_path: str | None = None,
                   save_optimizer: bool = True) -> None:
        """
        Saves the surrogate model state to files.
        """
        raise NotImplementedError(
            f"State persistence is not implemented for {type(self).__name__}"
        )

    def load_state(self, checkpoint_path: str, data_path: str | None = None,
                   load_optimizer: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """
        Loads the surrogate model state from files.
        Returns a list of numpy arrays containing the loaded snapshots and weights.
        """
        raise NotImplementedError(f"State persistence is not implemented for {type(self).__name__}")


def closest_point_distance(par, point):
    """
    Calculates distances of the given point to all points in self.par
    and returns the smallest distance.
    """
    distances = np.linalg.norm(par - point, axis=1)
    closest_index = np.argmin(distances)
    smallest_distance = distances[closest_index]
    return smallest_distance


def closest_point_distance_kdtree(par, kdtree, point):
    """
    Returns the distance to the closest point in kdtree.
    """
    closest_index = kdtree.query(point)[1]
    smallest_distance = np.linalg.norm(par[closest_index] - point)
    return smallest_distance
