"""
Bimodal 3-D test problem with a controllable valley (2026-10-12, for the C3 study).

Prior ``N(0, I_3)``; forward model ``G(u) = (u_1 + u_2 + u_3)^2 / 9 = t^2 / 3`` with ``t = u . e``,
``e = (1, 1, 1) / sqrt(3)``; likelihood ``N(y; G(u), sigma)``. The posterior factorises into a Gaussian
``N(0, I_2)`` in the two directions perpendicular to ``e`` and the 1-D density
``pi_t(t) ~ exp(-t^2/2 - (t^2/3 - y)^2 / (2 sigma^2))`` along ``e``, which is symmetric and bimodal.

With ``y = 1 + 1.5 sigma^2`` the two modes are EXACTLY ``u = (1, 1, 1)`` and ``u = (-1, -1, -1)``
(``t = +-sqrt(3)``), the origin is the saddle between them, and

    log pi(mode) - log pi(0) = 1 / (2 sigma^2) =: valley_depth   (nats, exact).

So the single knob ``valley_depth`` (``Delta``) sets ``sigma = (2 Delta)^(-1/2)`` and ``y = 1 + 0.75 / Delta``;
the density at the saddle is ``exp(-Delta)`` times the density at a mode. Each mode holds exactly half of
the mass. Reference moments are exact up to a 1-D quadrature (``reference``).

Any dimension ``d`` (2026-10-13, for note 32's dimension test): ``G(u) = (sum u_i)^2 / d^2 = t^2 / d`` with
``t = u . e``, ``e = 1 / sqrt(d)``, ``y = 1 + d sigma^2 / 2``: modes exactly at ``+-(1, ..., 1)`` (``t = +-sqrt(d)``),
the perpendicular part ``N(0, I_{d-1})``, and ``log pi(mode) - log pi(0) = 1 / (2 sigma^2) = Delta`` again (same
algebra). ``d = 3`` is the default everywhere, so the 3-D example and its numbers are unchanged.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt

import surrDAMH
from surrDAMH.solvers import Solver

E = np.ones(3) / np.sqrt(3.0)  # the 3-D direction; use ``unit_direction(d)`` in general


def unit_direction(no_parameters: int = 3) -> np.ndarray:
    return np.ones(no_parameters) / np.sqrt(float(no_parameters))


class Bimodal3D(Solver):
    """``G(u) = (sum u_i)^2 / d^2`` (``d = 3``: ``(u_1 + u_2 + u_3)^2 / 9``); ``d`` parameters, 1 observation."""

    def __init__(self, solver_id=0, output_dir=None, no_parameters: int = 3):
        self.no_parameters = int(no_parameters)
        self.no_observations = 1
        self.par_names = [f"u{i + 1}" for i in range(self.no_parameters)]

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.u = np.asarray(parameters, dtype=float)

    def get_observations(self) -> npt.NDArray:
        return np.array([float(np.sum(self.u)) ** 2 / float(self.no_parameters) ** 2])


def noise_and_data(valley_depth: float, no_parameters: int = 3) -> tuple[float, float]:
    """``(sigma, y)`` for the valley depth ``Delta`` in nats: ``sigma = (2 Delta)^(-1/2)``, ``y = 1 + d sigma^2 / 2``."""
    if not valley_depth > 0.0:
        raise ValueError(f"valley_depth must be positive (nats), got {valley_depth!r}")
    sigma = (2.0 * float(valley_depth)) ** -0.5
    return sigma, 1.0 + 0.5 * int(no_parameters) * sigma ** 2


def make_problem(valley_depth: float, prior_bound: float | None = 8.0, no_parameters: int = 3) -> "surrDAMH.Problem":
    """``surrDAMH.Problem`` with modes at ``+-(1, ..., 1)`` (``d = no_parameters``) and the given valley depth (nats)."""
    sigma, y = noise_and_data(valley_depth, no_parameters)
    return surrDAMH.Problem(surrDAMH.distributions.Normal(mean=0.0, sd=1.0),
                            surrDAMH.distributions.Normal(mean=y, sd=sigma),
                            solver=Bimodal3D(no_parameters=no_parameters), prior_bound=prior_bound)


def log_density_t(t: npt.NDArray, valley_depth: float, no_parameters: int = 3) -> npt.NDArray:
    """Unnormalised log density of ``t = u . e`` (the perpendicular part is ``N(0, I_{d-1})``)."""
    sigma, y = noise_and_data(valley_depth, no_parameters)
    t = np.asarray(t, dtype=float)
    return -0.5 * t ** 2 - (t ** 2 / float(no_parameters) - y) ** 2 / (2.0 * sigma ** 2)


def reference(valley_depth: float, half_width: float = 8.0, n: int = 200_001, no_parameters: int = 3) -> dict:
    """
    Exact posterior summaries (1-D trapezoid quadrature in ``t`` on ``[-half_width, half_width]``):
    ``mean`` (0 by symmetry), ``sd`` and ``m2`` per coordinate (``E u_i^2 = E t^2 / d + (d - 1) / d``),
    ``cov`` (``E t^2 / d - 1 / d`` off the diagonal), ``E_t2``, ``mode_mass`` (0.5 each),
    ``t_mode = sqrt(d)``, ``sd_t_within_mode`` (Laplace at the mode), ``saddle_ratio = exp(-Delta)``,
    ``sigma``, ``y``, and the quadrature change ``E_t2_change`` from halving the grid.
    """
    d = int(no_parameters)
    sigma, y = noise_and_data(valley_depth, d)
    half_width = max(float(half_width), np.sqrt(d) + 6.0)
    t = np.linspace(-half_width, half_width, n)
    w = np.exp(log_density_t(t, valley_depth, d) - log_density_t(np.sqrt(float(d)), valley_depth, d))
    z = np.trapezoid(w, t)
    e_t2 = np.trapezoid(w * t ** 2, t) / z
    t2 = t[::2]; w2 = w[::2]
    e_t2_half = np.trapezoid(w2 * t2 ** 2, t2) / np.trapezoid(w2, t2)
    var_i = e_t2 / d + (d - 1.0) / d
    cov = np.full((d, d), e_t2 / d - 1.0 / d); np.fill_diagonal(cov, var_i)
    # curvature of log pi_t at the mode t^2 = d: f'' = -1 - (4 t^2 / d^2) / sigma^2 - (2 / d)(t^2 / d - y) / sigma^2
    curvature = -1.0 - (4.0 / d) / sigma ** 2 - (2.0 / d) * (1.0 - y) / sigma ** 2
    return {"valley_depth": float(valley_depth), "no_parameters": d, "sigma": sigma, "y": y,
            "saddle_ratio": float(np.exp(-valley_depth)),
            "mean": np.zeros(d), "sd": np.sqrt(np.full(d, var_i)), "m2": np.full(d, var_i), "cov": cov,
            "E_t2": float(e_t2), "E_t2_change": float(e_t2 - e_t2_half), "mode_mass": 0.5,
            "t_mode": float(np.sqrt(float(d))), "sd_t_within_mode": float((-curvature) ** -0.5),
            "mass_outside": float(1.0 - np.trapezoid(w[np.abs(t) <= half_width - 1.0], t[np.abs(t) <= half_width - 1.0]) / z)}


def t_of(samples: npt.NDArray) -> npt.NDArray:
    """``t = u . e`` of an ``(n, d)`` array (``e = 1 / sqrt(d)`` from the array's width)."""
    samples = np.asarray(samples, dtype=float)
    return samples @ unit_direction(samples.shape[1])


def crossings(samples: npt.NDArray, lo: float = -0.5, hi: float = 0.5) -> int:
    """Number of mode-to-mode transitions of one chain (hysteresis on ``t``: left below ``lo``, right above ``hi``)."""
    t = t_of(samples)
    state = 1 if t[0] > 0 else 0
    count = 0
    for value in t:
        new = 1 if value > hi else (0 if value < lo else state)
        count += int(new != state)
        state = new
    return count
