#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
One persistent estimate of the posterior mean and covariance per chain (C1, author decision D7,
2026-10-11; ``library_notes/25_robust_by_default_roadmap_2026-10-08.md`` §9).

``PosteriorStatistics`` holds the running Welford statistics ``(n, mean, M2)`` of the
post-decision chain states in the prior's INTERNAL space, plus ``n_seen`` (the number of states
ever added, never discounted). The algorithm updates it once per outer iteration of EVERY stage,
whatever the stage's proposal; the runners pool it across the sampler ranks at every stage end
and carry it to the next stage and, through ``carry_over/``, to continued runs. It is never
reset. Consumers: the adaptive random walk's shape (``GaussRandomWalk_adaptive``) and the
Hamiltonian mass ``Hamiltonian(mass="covariance")``.

Stage-end pooling (fixed 2026-10-11): every rank starts a stage with the SAME pooled
statistics ``S``. :meth:`PosteriorStatistics.begin_stage` remembers ``S`` and opens a per-stage
increment that :meth:`PosteriorStatistics.update` feeds alongside; at the stage end the runner
gathers the ranks' increments ``I_r`` (:meth:`PosteriorStatistics.stage_increment_vector`) and
:meth:`PosteriorStatistics.finish_stage` installs ``S (+) I_1 (+) ... (+) I_R`` (Chan). Before
the fix the ranks' FULL objects ``S (+) I_r`` were pooled, so ``S`` was counted ``R`` times at
every stage boundary.

MPI-free: the runner does the ``Allgather``.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

#: minimum number of pooled states before ``Hamiltonian(mass="covariance")`` uses the
#: statistics (the adaptive random walk's default ``warmup``); fewer -> identity mass
MASS_MIN_STATES = 100
#: C1b (2026-10-11) placeholder: after a ``burn_in=True`` stage the pooled statistics keep the
#: weight of ``burn_in_statistics_count(d) = max(MASS_MIN_STATES, BURN_IN_STATISTICS_PER_DIMENSION * d)``
#: states (mean and covariance unchanged, see :meth:`PosteriorStatistics.discount`)
BURN_IN_STATISTICS_PER_DIMENSION = 10


def burn_in_statistics_count(d: int) -> int:
    """Effective count the statistics keep after a burn-in stage (C1b): ``max(100, 10 d)``."""
    return max(MASS_MIN_STATES, BURN_IN_STATISTICS_PER_DIMENSION * int(d))


class PosteriorStatistics:
    """
    Running mean and scatter of chain states, ``(n, mean, M2)``.

    Welford update (one call of :meth:`update` per state ``x``)::

        n    += 1
        diff  = x - mean
        mean += diff / n
        M2   += outer(diff, x - mean)

    so that ``mean`` is the sample mean and ``M2 / (n - 1)`` the unbiased sample covariance of
    the states seen so far. The arithmetic is exactly the one ``GaussRandomWalk_adaptive`` used
    for its own statistics before C1, so an adaptive random-walk stage that reads a fresh object
    is bit-identical to the code before C1.
    """

    def __init__(self, d: int) -> None:
        self.d = int(d)
        self.n = 0
        #: states ever added (Welford updates, pooled increments); NOT reduced by :meth:`discount`.
        #: Used only for the shrinkage weight of :meth:`shrunk_covariance` (author decision 1,
        #: 2026-10-11)
        self.n_seen = 0
        self.mean = np.zeros(self.d)
        self.M2 = np.zeros((self.d, self.d))
        # stage-end pooling (see the module docstring): the statistics at the stage start and the
        # per-stage increment fed by update(); both None outside begin_stage()/finish_stage()
        self._stage_start: PosteriorStatistics | None = None
        self._increment: PosteriorStatistics | None = None

    # -- update -------------------------------------------------------------------------
    def update(self, x: npt.ArrayLike) -> None:
        """Add one chain state ``x`` (internal space, length ``d``)."""
        state = np.asarray(x, dtype=float)
        self.n += 1
        self.n_seen += 1
        difference = state - self.mean
        self.mean = self.mean + difference / self.n
        self.M2 += np.outer(difference, state - self.mean)
        if self._increment is not None:
            self._increment.update(state)

    # -- stage-end pooling --------------------------------------------------------------
    def begin_stage(self) -> None:
        """Remember the current statistics as the stage start and open an empty per-stage
        increment, which every following :meth:`update` also feeds (call on every rank at the
        start of every stage, before anything updates the statistics)."""
        start = PosteriorStatistics(self.d)
        start.install(self)
        self._stage_start = start
        self._increment = PosteriorStatistics(self.d)

    def stage_increment_vector(self) -> npt.NDArray:
        """:meth:`as_vector` of the states added since :meth:`begin_stage` (length ``1 + d + d^2``,
        the runner's ``Allgather`` buffer). An increment is never discounted (``n_seen == n``)."""
        if self._increment is None:
            raise RuntimeError("stage_increment_vector() called outside begin_stage()/finish_stage()")
        return self._increment.as_vector()

    def finish_stage(self, increment_rows: npt.ArrayLike) -> None:
        """
        Install IN PLACE ``start (+) I_1 (+) ... (+) I_R``: the statistics at :meth:`begin_stage`
        Chan-combined with the gathered :meth:`stage_increment_vector` rows of all ``R`` chains
        (one :meth:`pooled_rows` call over ``[start, I_1, ..., I_R]``; an empty start is left out,
        which makes a run's first stage the plain pooling of the ranks' rows, as before). ``n_seen``
        becomes ``start.n_seen + sum_r n_r``. Every rank gets the same result (same start, same
        rows). Ends the stage's increment tracking.

        With ONE row (local runner, one-sampler MPI run) this object already holds exactly
        ``start (+) I_1`` (it saw exactly those states); it is re-pooled as a single row, the
        formula used before the fix, so a one-chain run is bit-identical to before.
        """
        if self._stage_start is None:
            raise RuntimeError("finish_stage() called without begin_stage()")
        d = self.d
        rows = np.atleast_2d(np.asarray(increment_rows, dtype=float))
        if rows.shape[1] != 1 + d + d * d:
            raise ValueError(f"expected increment rows of length {1 + d + d * d}, got {rows.shape}")
        start = self._stage_start
        n_seen = int(start.n_seen) + int(round(float(rows[:, 0].sum())))
        if rows.shape[0] == 1:
            pooled = PosteriorStatistics.from_rows(self.as_vector()[None, :], d)
        else:
            stacked = rows if start.n == 0 else np.vstack([start.as_vector()[None, :], rows])
            pooled = PosteriorStatistics.from_rows(stacked, d)
        self._stage_start = None
        self._increment = None
        self.install(pooled)
        self.n_seen = n_seen

    # -- pooling ------------------------------------------------------------------------
    @staticmethod
    def pooled_rows(counts: npt.NDArray, means: npt.NDArray, scatters: npt.NDArray
                    ) -> tuple[int, npt.NDArray, npt.NDArray] | None:
        """
        Chan-Golub-LeVeque parallel combination of ``R`` sets of statistics::

            N     = sum_r n_r
            mean  = sum_r n_r mean_r / N
            M2    = sum_r M2_r + sum_r n_r (mean_r - mean)(mean_r - mean)^T

        which gives exactly the statistics of the concatenated states (the formula
        ``GaussRandomWalk_adaptive.set_pooled_state`` used before C1, operation for operation).
        ``None`` if ``N == 0``.
        """
        total = float(counts.sum())
        if total <= 0:
            return None
        pooled_mean = (counts[:, None] * means).sum(axis=0) / total
        pooled_scatter = scatters.sum(axis=0)
        for count, mean in zip(counts, means):
            shift = mean - pooled_mean
            pooled_scatter = pooled_scatter + count * np.outer(shift, shift)
        return int(round(total)), pooled_mean, pooled_scatter

    def merge(self, others: "list[PosteriorStatistics]") -> "PosteriorStatistics":
        """A new object with the pooled statistics of ``[self] + others`` (see :meth:`pooled_rows`),
        ``n_seen`` = the sum of their ``n_seen``. The objects must hold DISJOINT sets of states (for
        the stage-end pooling of chains that share a start, see :meth:`finish_stage`)."""
        result = PosteriorStatistics.from_rows(np.vstack([s.as_vector() for s in [self, *others]]), self.d)
        result.n_seen = sum(int(s.n_seen) for s in [self, *others])
        return result

    @classmethod
    def from_rows(cls, rows: npt.ArrayLike, d: int) -> "PosteriorStatistics":
        """Pool the :meth:`as_vector` rows of several chains (the runner's ``Allgather`` buffer)."""
        rows = np.atleast_2d(np.asarray(rows, dtype=float))
        d = int(d)
        if rows.shape[1] != 1 + d + d * d:
            raise ValueError(f"expected posterior-statistics rows of length {1 + d + d * d}, got {rows.shape}")
        result = cls(d)
        pooled = cls.pooled_rows(rows[:, 0], rows[:, 1:1 + d], rows[:, 1 + d:].reshape(-1, d, d))
        if pooled is not None:
            result.n, result.mean, result.M2 = pooled
            result.n_seen = result.n
        return result

    def install(self, other: "PosteriorStatistics") -> None:
        """Replace this object's statistics IN PLACE by ``other``'s, ``n_seen`` included (so every
        holder of a reference -- the algorithm, the adaptive random walk -- sees the pooled state).
        Does not touch the stage tracking of :meth:`begin_stage`."""
        if other.d != self.d:
            raise ValueError(f"dimension mismatch: {other.d} != {self.d}")
        self.n = int(other.n)
        self.n_seen = int(getattr(other, "n_seen", other.n))
        self.mean = np.array(other.mean, dtype=float)
        self.M2 = np.array(other.M2, dtype=float)

    def discount(self, n_eff: int) -> int | None:
        """
        Reduce the WEIGHT of the statistics to ``n_eff`` states, keeping the mean and the
        covariance (C1b burn-in discount)::

            M2 <- M2 * (n_eff - 1) / (n - 1),    n <- n_eff     (only if n > n_eff)

        so ``M2 / (n - 1)`` and ``mean`` are unchanged, and later states (Welford updates, pooling)
        outweigh the discounted ones as if only ``n_eff`` states had been seen. ``n_seen`` is NOT
        changed, so the shrinkage of :meth:`shrunk_covariance` (``delta = min(1, 2d/n_seen)``)
        does not grow after a burn-in (author decision 1, 2026-10-11). Returns the previous ``n``
        if a discount was applied, else ``None``.
        """
        n_eff = int(n_eff)
        if n_eff < 2:
            raise ValueError(f"n_eff must be at least 2, got {n_eff}")
        if self.n <= n_eff:
            return None
        previous = int(self.n)
        self.M2 = self.M2 * ((n_eff - 1) / (previous - 1))
        self.n = n_eff
        return previous

    def copy(self) -> "PosteriorStatistics":
        result = PosteriorStatistics(self.d)
        result.install(self)
        return result

    # -- serialisation ------------------------------------------------------------------
    def as_vector(self) -> npt.NDArray:
        """``[n, mean (d), M2 (d*d, row-major)]``, length ``1 + d + d^2``, float64."""
        return np.concatenate((np.array([float(self.n)]), self.mean.astype(np.float64).ravel(),
                               self.M2.astype(np.float64).ravel()))

    @classmethod
    def from_vector(cls, v: npt.ArrayLike, d: int, n_seen: int | None = None) -> "PosteriorStatistics":
        """Inverse of :meth:`as_vector` (no pooling arithmetic: the values are taken as they are).
        ``n_seen`` (not part of the vector): given -> ``max(n, n_seen)``; ``None`` -> ``n``."""
        v = np.asarray(v, dtype=float).ravel()
        d = int(d)
        if v.size != 1 + d + d * d:
            raise ValueError(f"expected a posterior-statistics vector of length {1 + d + d * d}, got {v.size}")
        result = cls(d)
        result.n = int(round(v[0]))
        result.mean = v[1:1 + d].copy()
        result.M2 = v[1 + d:].reshape(d, d).copy()
        result.n_seen = result.n if n_seen is None else max(result.n, int(n_seen))
        return result

    # -- estimates ----------------------------------------------------------------------
    def is_informative(self, warmup: int = MASS_MIN_STATES) -> bool:
        """``n >= warmup``."""
        return self.n >= int(warmup)

    def shrunk_covariance(self) -> tuple[npt.NDArray, float, float] | None:
        """
        ``(C_reg, tr C / d, delta)`` or ``None`` when no covariance can be formed (``n < 2``, a
        non-finite statistic, or a chain that never moved: ``tr C <= 0``). With
        ``C = M2 / (n - 1)`` and ``N = max(n, n_seen)`` (the UNDISCOUNTED count of states seen,
        author decision 1, 2026-10-11; ``N == n`` unless a burn-in discount was applied)::

            delta = min(1, 2d/N)
            C     = (1 - delta) C + delta (tr C / d) I       [Ledoit-Wolf-style shrinkage]
            C_reg = C + 1e-6 (tr C / d) I                    [relative ridge]

        This is the estimator ``GaussRandomWalk_adaptive`` installs as its shape (times
        ``2.38^2/d``), computed with the same floating-point operations.
        """
        d, n = self.d, self.n
        if n < 2:
            return None
        C = self.M2 / (n - 1)
        trace_over_d = float(np.trace(C)) / d
        if not np.all(np.isfinite(C)) or not np.isfinite(trace_over_d) or trace_over_d <= 0.0:
            return None
        delta = min(1.0, 2.0 * d / max(n, self.n_seen))
        C = (1.0 - delta) * C + delta * trace_over_d * np.eye(d)
        return C + 1e-6 * trace_over_d * np.eye(d), trace_over_d, delta

    def covariance(self, shrinkage: bool = True) -> npt.NDArray:
        """
        The posterior covariance estimate, ``(d, d)``.

        ``shrinkage=True`` (default): :meth:`shrunk_covariance`'s ``C_reg`` (symmetric positive
        definite). ``shrinkage=False``: the plain unbiased ``M2 / (n - 1)``.

        Raises:
            ValueError: ``n < 2``, or (with shrinkage) a degenerate / non-finite statistic.
        """
        if self.n < 2:
            raise ValueError(f"a covariance needs at least 2 states, got n={self.n}")
        if not shrinkage:
            return self.M2 / (self.n - 1)
        result = self.shrunk_covariance()
        if result is None:
            raise ValueError("degenerate statistics (non-finite, or the chain never moved): no covariance")
        return result[0]

    def precision(self) -> npt.NDArray:
        """Inverse of :meth:`covariance` (regularised), via its Cholesky factor ``C = L L^T``:
        ``P = L^-T L^-1``, symmetrised. Raises like :meth:`covariance`."""
        C = self.covariance()
        L = np.linalg.cholesky(0.5 * (C + C.T))
        L_inv = np.linalg.solve(L, np.eye(self.d))
        P = L_inv.T @ L_inv
        return 0.5 * (P + P.T)

    def summary(self) -> dict:
        """Numpy-savable snapshot for the carry-over report."""
        return {"n": int(self.n), "n_seen": int(self.n_seen), "mean": self.mean.copy(), "M2": self.M2.copy()}

    def __repr__(self) -> str:
        return f"PosteriorStatistics(d={self.d}, n={self.n}, n_seen={self.n_seen})"


__all__ = ["PosteriorStatistics", "MASS_MIN_STATES", "BURN_IN_STATISTICS_PER_DIMENSION", "burn_in_statistics_count"]
