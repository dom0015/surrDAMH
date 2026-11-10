#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Statistics helpers shared by the posterior-correctness tests.

Used by

* ``tests/validation/test_gaussian_toy.py`` -- single-chain, in-process (``run_local``), and
* ``tests/mpi/test_mpi_posterior.py`` -- several chains produced by a real ``mpiexec`` run.

This is a plain module (``tests/`` is not a package); pytest puts ``tests/`` on ``sys.path``
when it imports ``tests/conftest.py``, so both test packages can ``import helpers_statistics``.

What is in here
---------------
* chain I/O: :func:`read_weighted_samples` / :func:`read_all_chains` decompress the
  ``sampling_output/samples/<stage>/rank%04d.csv`` files (output format v2: a header row,
  then column 0 = ``multiplicity``, columns ``1..no_parameters`` = parameters, see
  ``docs/outputs.md``);
* Monte-Carlo error: :func:`batch_means_se` -- the standard error of a mean of a correlated
  sequence, estimated by batch means (no autocorrelation-time estimate needed, which keeps it
  usable for the strongly autocorrelated DAMH chains);
* :func:`chain_statistics` (one chain) and :func:`pooled_chain_statistics` (several chains
  pooled) return mean/covariance together with those standard errors;
* :func:`assert_close` / :func:`assert_matches_closed_form` compare against a reference within
  ``TOLERANCE_IN_SE`` standard errors;
* :func:`gelman_rubin` -- the classical R-hat across chains.

``TOLERANCE_IN_SE`` is a property of the test suite, not of a single test: keep it here and do
not loosen it locally.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from surrDAMH.surrogates.parent import Evaluator

BURN_IN_FRACTION = 0.1  # discarded fraction at the start of every (decompressed) chain
NO_BATCHES = 64  # batch-means blocks used for the standard-error estimate
TOLERANCE_IN_SE = 4.0  # accepted deviation, in estimated standard errors


class FixedInitialSample:
    """Degenerate 'distribution' returning a fixed initial sample (keeps runs deterministic)."""

    def __init__(self, parameters: npt.ArrayLike) -> None:
        self.parameters = np.asarray(parameters, dtype=float)

    def rvs(self) -> npt.NDArray:
        return self.parameters.copy()


class FixedLinearEvaluator(Evaluator):
    """A deterministic, never-updated surrogate: ``G~(x) = A' x``."""

    def __init__(self, A: npt.ArrayLike) -> None:
        self.A = np.asarray(A, dtype=float)
        self.no_parameters = self.A.shape[1]

    def __call__(self, datapoints: npt.NDArray) -> npt.NDArray:
        return np.atleast_2d(datapoints) @ self.A.T


@dataclass
class ChainStatistics:
    """Weighted mean / covariance of one or more chains together with batch-means standard errors."""

    mean: npt.NDArray  # (no_parameters,)
    cov: npt.NDArray  # (no_parameters, no_parameters)
    se_mean: npt.NDArray  # (no_parameters,)
    se_cov: npt.NDArray  # (no_parameters, no_parameters)
    no_samples: int  # length of the decompressed chain(s) after burn-in
    counter_accepted: int = 0
    counter_rejected: int = 0
    counter_prerejected: int = 0
    no_chains: int = 1


def batch_means_se(values: npt.NDArray, no_batches: int = NO_BATCHES) -> float:
    """Standard error of ``values.mean()`` for a correlated sequence, via batch means."""
    batch_size = len(values) // no_batches
    assert batch_size > 0, "chain is too short for the requested number of batches"
    batch_means = values[: batch_size * no_batches].reshape(no_batches, batch_size).mean(axis=1)
    return float(batch_means.std(ddof=1) / np.sqrt(no_batches))


def samples_path(output_dir: str | os.PathLike, stage_name: str, rank: int = 0) -> str:
    """Path of one chain's sample file inside ``output_dir``."""
    return os.path.join(str(output_dir), "sampling_output", "samples", stage_name, f"rank{rank:04d}.csv")


def read_weighted_samples(output_dir: str | os.PathLike, stage_name: str, no_parameters: int,
                          rank: int = 0) -> npt.NDArray:
    """Decompressed chain of one sampler rank (``multiplicity`` in column 0, parameters in 1..p).

    ``skiprows=1`` skips the format-v2 header row (``multiplicity,par_0,...,log_posterior``).
    """
    rows = np.loadtxt(samples_path(output_dir, stage_name, rank), delimiter=",", skiprows=1, ndmin=2)
    multiplicities = rows[:, 0].astype(int)
    parameters = rows[:, 1:1 + no_parameters]
    return np.repeat(parameters, multiplicities, axis=0)


def read_all_chains(output_dir: str | os.PathLike, stage_name: str, no_parameters: int,
                    no_chains: int) -> list[npt.NDArray]:
    """Decompressed chains of ranks ``0 .. no_chains-1`` of one stage."""
    return [read_weighted_samples(output_dir, stage_name, no_parameters, rank)
            for rank in range(no_chains)]


def _burn_in(samples: npt.NDArray, burn_in_fraction: float) -> npt.NDArray:
    return samples[int(burn_in_fraction * len(samples)):]


def chain_statistics(samples: npt.NDArray, counter_accepted: int = 0, counter_rejected: int = 0,
                     counter_prerejected: int = 0, burn_in_fraction: float = BURN_IN_FRACTION,
                     no_batches: int = NO_BATCHES) -> ChainStatistics:
    """Mean/covariance of one decompressed chain (after burn-in) with batch-means standard errors."""
    samples = _burn_in(samples, burn_in_fraction)
    mean = samples.mean(axis=0)
    centered = samples - mean
    cov = centered.T @ centered / len(samples)
    no_parameters = samples.shape[1]
    se_mean = np.array([batch_means_se(samples[:, i], no_batches) for i in range(no_parameters)])
    se_cov = np.array([[batch_means_se(centered[:, i] * centered[:, j], no_batches)
                        for j in range(no_parameters)] for i in range(no_parameters)])
    return ChainStatistics(mean=mean, cov=cov, se_mean=se_mean, se_cov=se_cov,
                           no_samples=len(samples), counter_accepted=counter_accepted,
                           counter_rejected=counter_rejected,
                           counter_prerejected=counter_prerejected, no_chains=1)


def pooled_chain_statistics(chains: list[npt.NDArray], counter_accepted: int = 0,
                            counter_rejected: int = 0, counter_prerejected: int = 0,
                            burn_in_fraction: float = BURN_IN_FRACTION,
                            no_batches: int = NO_BATCHES) -> ChainStatistics:
    """
    Mean/covariance of several chains pooled, with a standard error that respects chain boundaries.

    Each chain is burnt in separately and the pooled estimate is the length-weighted average of
    the per-chain estimates (which is exactly the estimate of the concatenated chains). Batches
    are therefore formed *within* one chain and never straddle a chain boundary; the per-chain
    standard errors are combined in quadrature with the same length weights::

        m_pooled = sum_c (n_c/N) m_c        se_pooled = sqrt( sum_c (n_c/N)^2 se_c^2 )

    The covariance is computed about the pooled mean (each chain is centred by the pooled mean
    before its products are batched), so the reported covariance is that of the pooled sample.
    """
    assert len(chains) >= 1
    chains = [_burn_in(chain, burn_in_fraction) for chain in chains]
    lengths = np.array([len(chain) for chain in chains], dtype=float)
    total = float(lengths.sum())
    weights = lengths / total
    no_parameters = chains[0].shape[1]

    mean = sum(w * chain.mean(axis=0) for w, chain in zip(weights, chains))
    centered = [chain - mean for chain in chains]
    cov = sum(w * (c.T @ c / len(c)) for w, c in zip(weights, centered))

    se_mean = np.zeros(no_parameters)
    se_cov = np.zeros((no_parameters, no_parameters))
    for w, chain, c in zip(weights, chains, centered):
        for i in range(no_parameters):
            se_mean[i] += (w * batch_means_se(chain[:, i], no_batches)) ** 2
            for j in range(no_parameters):
                se_cov[i, j] += (w * batch_means_se(c[:, i] * c[:, j], no_batches)) ** 2
    se_mean = np.sqrt(se_mean)
    se_cov = np.sqrt(se_cov)

    return ChainStatistics(mean=np.asarray(mean), cov=np.asarray(cov), se_mean=se_mean,
                           se_cov=se_cov, no_samples=int(total),
                           counter_accepted=counter_accepted, counter_rejected=counter_rejected,
                           counter_prerejected=counter_prerejected, no_chains=len(chains))


def gelman_rubin(chains: list[npt.NDArray], burn_in_fraction: float = BURN_IN_FRACTION) -> npt.NDArray:
    """
    Classical Gelman-Rubin potential scale reduction factor R-hat, one value per parameter.

    Same formula as ``surrDAMH.post_processing.Analysis.calculate_gelman_rubin`` (which needs a
    loaded ``Analysis`` object and is therefore not reusable here): chains are burnt in, truncated
    to their common length ``n``, and

        B = n/(m-1) * sum_c (mean_c - mean)^2,  W = mean_c var_c,
        var_hat = (n-1)/n * W + B/n,  R-hat = sqrt(var_hat / W).
    """
    burnt = [_burn_in(chain, burn_in_fraction) for chain in chains]
    assert len(burnt) >= 2, "R-hat needs at least two chains"
    n = min(len(chain) for chain in burnt)
    assert n >= 2, "chains are too short for R-hat"
    samples = np.stack([chain[:n, :] for chain in burnt], axis=0)  # (m, n, p)
    m = samples.shape[0]
    chain_means = samples.mean(axis=1)
    grand_mean = chain_means.mean(axis=0)
    B = (n / (m - 1)) * np.sum((chain_means - grand_mean) ** 2, axis=0)
    W = np.var(samples, axis=1, ddof=1).mean(axis=0)
    var_hat = ((n - 1) / n) * W + B / n
    return np.sqrt(var_hat / np.maximum(W, 1e-15))


def assert_close(label: str, value: npt.NDArray, reference: npt.NDArray,
                 standard_error: npt.NDArray, tolerance_in_se: float = TOLERANCE_IN_SE) -> None:
    """Assert ``value`` is within ``tolerance_in_se`` estimated standard errors of ``reference``."""
    deviation_in_se = (value - reference) / standard_error
    assert np.all(np.abs(deviation_in_se) <= tolerance_in_se), (
        f"{label}: deviation exceeds {tolerance_in_se} standard errors\n"
        f"  value     = {np.array2string(value, precision=6)}\n"
        f"  reference = {np.array2string(reference, precision=6)}\n"
        f"  std error = {np.array2string(standard_error, precision=6)}\n"
        f"  deviation/SE = {np.array2string(deviation_in_se, precision=2)}"
    )


def assert_matches_closed_form(statistics: ChainStatistics, posterior_mean: npt.NDArray,
                               posterior_cov: npt.NDArray, label: str,
                               tolerance_in_se: float = TOLERANCE_IN_SE) -> None:
    """Assert sampled mean and covariance match the closed-form posterior within the tolerance."""
    assert_close(f"{label} posterior mean vs closed form", statistics.mean, posterior_mean,
                 statistics.se_mean, tolerance_in_se)
    assert_close(f"{label} posterior covariance vs closed form", statistics.cov, posterior_cov,
                 statistics.se_cov, tolerance_in_se)


def max_deviation_in_se(value: npt.NDArray, reference: npt.NDArray,
                        standard_error: npt.NDArray) -> float:
    """Largest ``|value - reference| / standard_error`` (used for reporting)."""
    return float(np.max(np.abs((value - reference) / standard_error)))
