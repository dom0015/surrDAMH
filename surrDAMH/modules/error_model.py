#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Global Gaussian approximation-error model of the surrogate (S1, 2026-10-09; note 22 §4 P3, note 25 S1).

With observed data ``y``, the Gaussian noise model ``Normal(mean=y, sd or cov=Sigma_noise)``, a surrogate
``G~`` and the statistics of its residuals ``r = G(u) - G~(u)`` -- the mean ``eps`` and the covariance
``Sigma_surr`` -- the SURROGATE likelihood becomes

    L~(u) = N( y ; G~(u) + eps , Sigma_noise + Sigma_surr ).

The exact likelihood is never changed; the delayed-acceptance correction uses the same ``L~`` as the
sub-chain, so a DAMH chain still targets the exact posterior (the error model changes which proposals pass
the first stage, not the target).

Pure functions, no MPI: used by the collector (``process_COLLECTOR``), by the local runner's
``LocalSurrogateManager`` and by the sampler (``AlgorithmBase``):

- ``estimate_error_model(residuals) -> ErrorModel`` (Ledoit-Wolf shrinkage for up to
  ``ERROR_MODEL_FULL_COV_MAX_OBS`` observations, a diagonal above),
- ``corrected_likelihood(likelihood, error_model) -> Normal``,
- ``ErrorModel.as_state()`` / ``ErrorModel.from_state()`` (entrywise averaging, ``Stage.polyak_averaging``),
- ``error_model_for_evaluator(...)``: the residuals of one evaluator on the updater's snapshots and on the
  held-out set, pooled with uniform weights, and their model (or ``None``).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from surrDAMH.distributions.normal import Normal

#: Up to this many observations ``Sigma_surr`` is a full (Ledoit-Wolf shrunk) covariance matrix; above it,
#: the diagonal of the residual variances (stored as a vector, so the pickled evaluator stays small).
ERROR_MODEL_FULL_COV_MAX_OBS = 50
#: Names of the error-model entries in ``Evaluator.averageable_state()`` (S3 averaging).
ERROR_MODEL_STATE_KEYS: tuple[str, str] = ("error_model.bias", "error_model.cov")


@dataclass
class ErrorModel:
    """
    Residual statistics of one surrogate evaluator: ``bias`` = ``eps`` (shape ``(m,)``) and ``cov`` =
    ``Sigma_surr``, either a full ``(m, m)`` PSD matrix (``m <= ERROR_MODEL_FULL_COV_MAX_OBS``) or, above
    that size, the ``(m,)`` vector of its diagonal (residual variances). ``no_residuals`` is the number of
    residuals it was estimated from (0 when built by hand or averaged).
    """

    bias: npt.NDArray
    cov: npt.NDArray
    no_residuals: int = 0

    def __post_init__(self) -> None:
        self.bias = np.asarray(self.bias, dtype=float).reshape(-1)
        self.cov = np.asarray(self.cov, dtype=float)
        m = self.bias.shape[0]
        if self.cov.shape not in ((m, m), (m,)):
            raise ValueError(f"ErrorModel: cov has shape {self.cov.shape}, expected ({m}, {m}) or ({m},)")

    @property
    def is_diagonal(self) -> bool:
        """``cov`` holds the diagonal only (``(m,)`` vector of variances)."""
        return self.cov.ndim == 1

    def covariance_matrix(self) -> npt.NDArray:
        """``Sigma_surr`` as an ``(m, m)`` matrix."""
        return np.diag(self.cov) if self.is_diagonal else self.cov

    @property
    def bias_norm(self) -> float:
        """``||eps||`` (column ``error_bias_norm`` of ``surrogate_quality.csv``)."""
        return float(np.linalg.norm(self.bias))

    @property
    def cov_trace(self) -> float:
        """``tr Sigma_surr`` (column ``error_cov_trace`` of ``surrogate_quality.csv``)."""
        return float(np.sum(self.cov) if self.is_diagonal else np.trace(self.cov))

    def as_state(self) -> dict[str, npt.NDArray]:
        """float64 copies keyed by ``ERROR_MODEL_STATE_KEYS``; averaging them entrywise keeps ``cov`` PSD."""
        return {ERROR_MODEL_STATE_KEYS[0]: np.array(self.bias, dtype=np.float64),
                ERROR_MODEL_STATE_KEYS[1]: np.array(self.cov, dtype=np.float64)}

    @classmethod
    def from_state(cls, state: dict[str, npt.NDArray]) -> "ErrorModel":
        """Inverse of ``as_state`` (extra keys are ignored)."""
        return cls(bias=np.array(state[ERROR_MODEL_STATE_KEYS[0]], dtype=float),
                   cov=np.array(state[ERROR_MODEL_STATE_KEYS[1]], dtype=float))


def split_error_model_state(state: dict[str, npt.NDArray]) -> tuple[dict[str, npt.NDArray], dict[str, npt.NDArray]]:
    """``(model part, error-model part)`` of an ``averageable_state()`` dict."""
    model_part = {name: value for name, value in state.items() if name not in ERROR_MODEL_STATE_KEYS}
    error_part = {name: value for name, value in state.items() if name in ERROR_MODEL_STATE_KEYS}
    return model_part, error_part


def estimate_error_model(residuals: npt.ArrayLike) -> ErrorModel | None:
    """
    ``ErrorModel`` of the residuals ``r_i = G(u_i) - G~(u_i)`` (shape ``(n, m)``; non-finite rows dropped).

    ``bias = mean(r)``. ``cov``: for ``m <= ERROR_MODEL_FULL_COV_MAX_OBS`` the Ledoit-Wolf estimate
    (``sklearn.covariance.LedoitWolf(assume_centered=False)``: the maximum-likelihood covariance shrunk toward
    ``tr/m * I``, which keeps its trace and makes it positive definite unless all residuals are equal); for
    larger ``m`` the vector of the residual variances (``ddof=0``). With fewer than ``max(5, m + 1)``
    residuals the diagonal ``var(r)`` is used in both cases (as a diagonal matrix when ``m`` is small, so the
    layout does not change during a run); ``bias`` is the mean as it is. ``None`` when no finite residual
    exists or the statistics overflow.
    """
    r = np.asarray(residuals, dtype=float)
    if r.ndim == 1:
        r = r.reshape(-1, 1)
    r = r[np.all(np.isfinite(r), axis=1)]
    n, m = r.shape
    if n == 0:
        return None
    bias = r.mean(axis=0)
    variances = np.maximum(r.var(axis=0), 0.0)
    if not (np.all(np.isfinite(bias)) and np.all(np.isfinite(variances))):
        return None  # overflowing residuals: no usable model
    if m > ERROR_MODEL_FULL_COV_MAX_OBS:
        return ErrorModel(bias=bias, cov=variances, no_residuals=n)
    if n < max(5, m + 1):
        return ErrorModel(bias=bias, cov=np.diag(variances), no_residuals=n)
    from sklearn.covariance import LedoitWolf
    cov = LedoitWolf(assume_centered=False).fit(r).covariance_
    cov = 0.5 * (cov + cov.T)
    return ErrorModel(bias=bias, cov=cov, no_residuals=n)


class CorrectedNormal(Normal):
    """
    The Gaussian surrogate likelihood ``N(y - eps, Sigma_noise + Sigma_surr)`` built by
    ``corrected_likelihood``. A ``Normal`` (``mean``, ``cov`` or ``sd``, ``get_covariance``,
    ``grad_logpdf``) whose ``logpdf`` adds ``log_normalization`` = ``-1/2 log det(Sigma_tot) + 1/2 log
    det(Sigma_noise)``: ``Normal.logpdf`` drops the constant ``-1/2 log det(2 pi Sigma_noise)`` of the exact
    likelihood, and this keeps the SAME convention, so ``log L~`` and the exact ``log L`` are on one scale
    (``raw_data``, the S2 audit statistic ``log L - log L~``). A full covariance is factorised once.
    """

    def __init__(self, mean: npt.NDArray, log_normalization: float, cov: npt.NDArray | None = None,
                 sd: npt.NDArray | None = None) -> None:
        if cov is not None:
            super().__init__(mean=np.asarray(mean, dtype=float), cov=np.asarray(cov, dtype=float))
            from scipy.linalg import cho_factor
            self._cho = cho_factor(self.cov, lower=True)
        else:
            super().__init__(mean=np.asarray(mean, dtype=float), sd=np.asarray(sd, dtype=float))
            self._cho = None
        self.log_normalization = float(log_normalization)
        # Normal.__init__ binds logpdf/grad_logpdf per instance; the class methods below take over
        for name in ("logpdf", "grad_logpdf"):
            self.__dict__.pop(name, None)

    def _solve(self, v: npt.NDArray) -> npt.NDArray:
        if self._cho is None:
            return v / self.sd ** 2
        from scipy.linalg import cho_solve
        # no finiteness check: a divergent leapfrog evaluates the surrogate at inf/NaN, and the
        # result must propagate as non-finite (as Normal's np.linalg.solve does), not raise
        return cho_solve(self._cho, v, check_finite=False)

    def logpdf(self, sample: npt.NDArray) -> float:
        v = self.mean - np.asarray(sample, dtype=float).ravel()
        return -0.5 * float(np.dot(v, self._solve(v))) + self.log_normalization

    def grad_logpdf(self, sample: npt.NDArray) -> npt.NDArray:
        v = self.mean - np.asarray(sample, dtype=float).ravel()
        return self._solve(v)


def _logdet(matrix: npt.NDArray) -> float:
    sign, value = np.linalg.slogdet(matrix)
    if sign <= 0:
        raise np.linalg.LinAlgError("covariance is not positive definite")
    return float(value)


def corrected_likelihood(likelihood: Normal, error_model: ErrorModel) -> CorrectedNormal:
    """
    ``N(mean = y - eps, cov = Sigma_noise + Sigma_surr)`` for the ``Normal`` likelihood ``N(y, Sigma_noise)``
    (``likelihood.get_covariance()``: a matrix, or a vector of standard deviations meaning ``diag(sd^2)``).
    Evaluated at ``G~(u)`` it is ``N(y; G~(u) + eps, Sigma_tot)``. Diagonal noise with a diagonal ``Sigma_surr``
    stays an uncorrelated (``sd``) ``Normal``. ``logpdf`` includes the normalisation term described in
    ``CorrectedNormal``.

    Raises:
        TypeError: ``likelihood`` is not a ``Normal``.
        ValueError: sizes differ.
    """
    if not isinstance(likelihood, Normal):
        raise TypeError(f"the surrogate error model needs a Normal likelihood, got {type(likelihood).__name__}")
    y = np.asarray(likelihood.mean, dtype=float).reshape(-1)
    m = y.shape[0]
    if error_model.bias.shape[0] != m:
        raise ValueError(f"error model has {error_model.bias.shape[0]} observations, the likelihood {m}")
    noise = np.asarray(likelihood.get_covariance(), dtype=float)
    mean = y - error_model.bias
    if noise.ndim <= 1:
        noise_var = np.broadcast_to(noise.reshape(-1) ** 2, (m,)).astype(float)
        if error_model.is_diagonal:
            total_var = noise_var + error_model.cov
            log_normalization = -0.5 * float(np.sum(np.log(total_var)) - np.sum(np.log(noise_var)))
            return CorrectedNormal(mean, log_normalization, sd=np.sqrt(total_var))
        noise_cov = np.diag(noise_var)
        noise_logdet = float(np.sum(np.log(noise_var)))
    else:
        noise_cov = noise
        noise_logdet = _logdet(noise_cov)
    total = noise_cov + error_model.covariance_matrix()
    total = 0.5 * (total + total.T)
    log_normalization = -0.5 * (_logdet(total) - noise_logdet)
    return CorrectedNormal(mean, log_normalization, cov=total)


def _finite_pairs(parameters, observations) -> tuple[npt.NDArray, npt.NDArray] | None:
    if parameters is None or observations is None:
        return None
    parameters = np.asarray(parameters, dtype=float)
    observations = np.asarray(observations, dtype=float)
    if parameters.ndim != 2 or parameters.shape[0] == 0:
        return None
    observations = observations.reshape(parameters.shape[0], -1)
    keep = np.all(np.isfinite(parameters), axis=1) & np.all(np.isfinite(observations), axis=1)
    return parameters[keep], observations[keep]


def training_snapshots_of(updater) -> tuple[npt.NDArray, npt.NDArray] | None:
    """``(parameters, observations)`` the updater holds (``get_training_data_arrays()``), else ``None``."""
    getter = getattr(updater, "get_training_data_arrays", None)
    if getter is None:
        return None
    arrays = getter()
    if arrays is None:
        return None
    return arrays[0], arrays[1]


def error_model_for_evaluator(evaluator, training: tuple | None = None,
                              heldout: tuple | None = None) -> ErrorModel | None:
    """
    The ``ErrorModel`` of ``evaluator`` from its residuals on the ``training`` snapshots ``(parameters,
    observations)`` and on the ``heldout`` set, pooled with uniform weights (one batched call each; rows that
    are not finite are dropped). ``None`` when neither set has a finite residual.
    """
    blocks = []
    for pair in (training, heldout):
        finite = _finite_pairs(*pair) if pair is not None else None
        if finite is None or finite[0].shape[0] == 0:
            continue
        parameters, observations = finite
        predictions = np.asarray(evaluator(parameters), dtype=float).reshape(observations.shape)
        blocks.append(observations - predictions)
    if not blocks:
        return None
    return estimate_error_model(np.vstack(blocks))


def attach_error_model(evaluator, updater, heldout: tuple | None = None) -> ErrorModel | None:
    """
    Estimate the error model of a freshly built ``evaluator`` (the updater's snapshots + ``heldout``) and set
    ``evaluator.error_model`` (``None`` when unavailable). Returns it. Shared by the collector and the local
    runner's ``LocalSurrogateManager``.
    """
    model = error_model_for_evaluator(evaluator, training_snapshots_of(updater), heldout)
    evaluator.error_model = model
    return model


__all__ = ["ERROR_MODEL_FULL_COV_MAX_OBS", "ERROR_MODEL_STATE_KEYS", "ErrorModel", "CorrectedNormal",
           "estimate_error_model", "corrected_likelihood", "error_model_for_evaluator", "attach_error_model",
           "training_snapshots_of", "split_error_model_state"]
