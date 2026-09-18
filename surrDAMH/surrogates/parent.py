#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import warnings
from typing import Literal

import numpy as np
import numpy.typing as npt

from surrDAMH.solvers import Solver

WeightingPolicy = Literal["uniform", "multiplicity"]
WEIGHTING_POLICIES: tuple[str, ...] = ("uniform", "multiplicity")


class SurrogateAsSolver(Solver):
    """Exposes an ``Evaluator`` through the ``Solver`` interface (``use_only_surrogate`` stages)."""

    def __init__(self, call_method) -> None:
        self.call_method = call_method

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.parameters = parameters

    def get_observations(self) -> npt.NDArray:
        """
        Returns observations of shape ``(no_observations,)``, i.e. the ``Solver`` contract.

        The evaluator is called on a ``(1, no_parameters)`` batch and returns
        ``(1, no_observations)`` for every evaluator (WS6 contract, see ``Evaluator``), so
        the ``reshape(-1)`` below is an exact, information-preserving conversion to the
        ``Solver`` shape (before WS6 it also had to absorb the torch evaluators' flattened
        output, finding 3.5). Without it, a ``(1, no_observations)`` array reaches
        ``Normal.logpdf`` and breaks for ``no_observations > 1`` (finding A20).
        """
        result = self.call_method(np.atleast_2d(self.parameters))
        return np.asarray(result).reshape(-1)


class Evaluator:
    """
    Parent class for surrogate evaluators, produced by ``Updater.get_evaluator()`` and
    picklable so they can be shipped from the collector to sampler ranks over MPI.

    Contract (WS6, ``library_notes/12_evaluator_contract_spec.md`` §2 — now implemented by
    every evaluator shipped here):

    - ``__call__(X: (n, no_parameters)) -> (n, no_observations)``, **always 2-D, also for
      ``n == 1``**. There is no single-point convenience overload: callers pass
      ``x.reshape(1, -1)`` explicitly (as ``SurrogateAsSolver`` does).
    - ``jacobian(x: (no_parameters,)) -> (J: (no_observations, no_parameters),
      y: (no_observations,))``, single point only (batched input raises ``ValueError``).
    - ``vjp(x, v) -> (J(x).T @ v, y)``; the base-class default derives it from ``jacobian``.
    - ``no_parameters``/``no_observations`` are real attributes, set here; every subclass
      must call ``super().__init__(no_parameters, no_observations)``.

    Evaluators that cannot differentiate keep raising ``NotImplementedError`` from
    ``jacobian``/``vjp``; ``set_use_gradients`` silently no-ops for them.
    """

    def __init__(self, no_parameters: int, no_observations: int) -> None:
        self.no_parameters = int(no_parameters)
        self.no_observations = int(no_observations)

    def __call__(self, datapoints: npt.NDArray) -> npt.NDArray:
        """
        Evaluates the surrogate model in datapoints.

        datapoints shape: (number of datapoints, no_parameters)

        output NDArray shape: (number of datapoints, no_observations), for every number of
        datapoints including 1. Not implemented by the base class.
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

    Snapshot weighting (WS6 decision 3)
    -----------------------------------
    Every snapshot arrives with a ``multiplicity``: ``1 + rejections`` for a chain state
    that was just left, and ``0`` for a rejected proposal (``algorithms.py``). The
    ``weighting`` constructor option decides what the fit does with it:

    - ``"uniform"`` (**default for every updater, including the neural networks**): every
      snapshot counts exactly once, rejected proposals (multiplicity 0) included.
    - ``"multiplicity"``: rows with multiplicity ``0`` are excluded everywhere (they never
      enter the stored training data), and rows with multiplicity ``m > 0`` are weighted by
      ``m`` where the fit supports per-sample weights (``supports_sample_weights = True``)
      or included once where it does not (interpolants).

    The policy is implemented once here (``_rows_to_use`` / ``_training_weights``); a
    concrete updater only has to apply the mask and pass the weights on to its fit.

    Which methods each concrete updater implements, including persistence, is tabulated in
    ``library_notes/12_evaluator_contract_spec.md`` §1.
    """

    #: Whether this updater's fit can honour per-row weights. ``False`` means
    #: ``weighting="multiplicity"`` only *filters* zero-multiplicity rows for it.
    supports_sample_weights: bool = False

    #: Output-normalization mode; only updaters that normalize their targets override this
    #: (see ``set_output_normalization``).
    output_normalization: str = "identity"

    def __init__(self, no_parameters: int, no_observations: int,
                 weighting: WeightingPolicy = "uniform") -> None:
        self.no_parameters = int(no_parameters)
        self.no_observations = int(no_observations)
        self.weighting = validate_weighting(weighting)
        self.no_snapshots = 0

    def describe_weighting(self) -> str:
        """One-line description of the snapshot-weighting policy, logged at start-up.

        Tolerates a third-party updater that never called ``super().__init__`` (it is then
        reported as the default ``"uniform"``).
        """
        if getattr(self, "weighting", "uniform") == "multiplicity":
            detail = ("snapshots with multiplicity 0 (rejected proposals) are NOT used; "
                      + ("remaining rows are weighted by their multiplicity"
                         if self.supports_sample_weights
                         else f"{type(self).__name__} cannot weight rows, so the rest count once each"))
        else:
            detail = "every snapshot counts once, rejected proposals (multiplicity 0) included"
        return f"{type(self).__name__}: weighting={getattr(self, 'weighting', 'uniform')!r} ({detail})"

    def _rows_to_use(self, multiplicity: npt.NDArray | None, no_rows: int) -> npt.NDArray:
        """
        Boolean mask ``(no_rows,)`` of the snapshots that take part in the fit.

        ``"uniform"`` keeps everything; ``"multiplicity"`` drops rows with multiplicity 0.
        """
        if self.weighting == "multiplicity" and multiplicity is not None:
            values = np.asarray(multiplicity, dtype=float).reshape(-1)
            if values.shape[0] != no_rows:
                raise ValueError(f"multiplicity has {values.shape[0]} rows, expected {no_rows}")
            return values > 0.0
        return np.ones((no_rows,), dtype=bool)

    def _training_weights(self, multiplicity: npt.NDArray | None) -> npt.NDArray | None:
        """
        Per-row sample weights for rows that already passed ``_rows_to_use``, or ``None``
        when the fit should be unweighted (``"uniform"``, or an updater that cannot weight).
        """
        if self.weighting != "multiplicity" or multiplicity is None or not self.supports_sample_weights:
            return None
        return np.asarray(multiplicity, dtype=float).reshape(-1)

    def set_output_normalization(self, mean: npt.NDArray, scale: npt.NDArray) -> None:
        """
        Hook: supply per-observation normalization statistics derived from the likelihood.

        Called once by ``SamplingFramework``/``run_local`` (via
        ``apply_output_normalization_from_likelihood``) for updaters configured with
        ``output_normalization="likelihood"``. No-op in the base class and in every updater
        that does not normalize its targets.
        """
        return None

    def delayed_init(self, data):
        """
        Additional settings of the surrogate model.
        """
        pass

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray,
                 multiplicity: npt.NDArray | None = None) -> None:
        """
        Adds more snapshots to the surrogate model.
        parameters shape: (number of snapshots, no_parameters)
        observations shape: (number of snapshots, no_observations)
        multiplicity shape: (number of snapshots, 1); ``1 + rejections`` for a chain state
        that was left, ``0`` for a rejected proposal. How it is used is decided by the
        ``weighting`` option (see the class docstring), not by the concrete updater.
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
        Returns ``(parameters, observations, multiplicity)`` numpy arrays.
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
        Returns ``(parameters, observations, multiplicity)`` numpy arrays.
        """
        raise NotImplementedError(f"State persistence is not implemented for {type(self).__name__}")


def validate_weighting(weighting: str) -> str:
    """Validates the ``weighting`` option of an ``Updater`` (WS6 decision 3)."""
    if weighting not in WEIGHTING_POLICIES:
        raise ValueError(f"weighting must be one of {WEIGHTING_POLICIES}, got {weighting!r}")
    return weighting


def likelihood_output_statistics(likelihood, no_observations: int) -> tuple[npt.NDArray, npt.NDArray] | None:
    """
    Per-observation ``(mean, scale)`` derived from the likelihood, or ``None`` if it cannot
    provide them (WS6 decision: ``output_normalization="likelihood"``).

    ``mean`` is the observed data (``likelihood.mean``) and ``scale`` the per-observation
    noise standard deviation: ``likelihood.sd`` broadcast to ``no_observations`` for an
    uncorrelated ``Normal``, ``sqrt(diag(likelihood.cov))`` for a covariance one. Returns
    ``None`` for anything else, for a shape that does not match ``no_observations``, and
    for non-finite or non-positive scales - the caller then falls back to the identity.
    """
    mean = getattr(likelihood, "mean", None)
    if mean is None:
        return None
    mean_array = np.asarray(mean, dtype=float).reshape(-1)

    if getattr(likelihood, "cov", None) is not None:
        scale_array = np.sqrt(np.diag(np.asarray(likelihood.cov, dtype=float)))
    elif getattr(likelihood, "sd", None) is not None:
        scale_array = np.asarray(likelihood.sd, dtype=float).reshape(-1)
    else:
        return None

    if mean_array.shape[0] != no_observations:
        return None
    if scale_array.shape[0] == 1 and no_observations > 1:
        scale_array = np.full((no_observations,), scale_array[0], dtype=float)
    if scale_array.shape[0] != no_observations:
        return None
    if not np.all(np.isfinite(mean_array)) or not np.all(np.isfinite(scale_array)):
        return None
    if np.any(scale_array <= 0.0):
        return None
    return mean_array, scale_array


def apply_output_normalization_from_likelihood(updater, likelihood) -> None:
    """
    Feeds likelihood-derived normalization statistics into ``updater``, if it asked for them.

    Called once by ``SamplingFramework`` and ``run_local``. Does nothing unless the updater
    is configured with ``output_normalization="likelihood"``. If the likelihood cannot
    supply usable per-observation statistics, warns (``RuntimeWarning``, naming the
    likelihood class) and leaves the updater at the identity normalization.
    """
    if updater is None or likelihood is None:
        return
    if getattr(updater, "output_normalization", "identity") != "likelihood":
        return
    statistics = likelihood_output_statistics(likelihood, int(getattr(updater, "no_observations", 0)))
    if statistics is None:
        warnings.warn(
            f"output_normalization='likelihood' requested for {type(updater).__name__}, but "
            f"{type(likelihood).__name__} does not provide usable per-observation statistics "
            "(mean plus sd or cov matching no_observations); falling back to identity "
            "normalization.",
            RuntimeWarning,
            stacklevel=2,
        )
        return
    updater.set_output_normalization(*statistics)


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
