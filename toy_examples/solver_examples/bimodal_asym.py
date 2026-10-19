"""
Asymmetric bimodal test problem with a LIKELIHOOD-made barrier, any dimension (2026-10-14; the problem review item
23 of ``library_notes/10_manual_review_notes.md`` asks for, to separate the flattened-likelihood and the
tempered-momentum components outside the point-symmetric ``bimodal3d`` family).

Prior ``N(0, I_d)``. Forward model ``G(u) = (t - c)^2`` with ``t = u . e`` and the fixed, non-axis unit direction
``e = (1, 2, ..., d) / |(1, 2, ..., d)|``; likelihood ``N(y; G(u), sigma)`` with ``y = r^2``. The likelihood fits
exactly at ``t = c - r`` and ``t = c + r``, so the posterior along ``e`` is bimodal with modes NEAR those two points
(the prior shifts them slightly towards 0; ``reference`` locates them numerically), while the ``d - 1`` directions
perpendicular to ``e`` stay ``N(0, I_{d-1})``. What makes the family different from ``bimodal3d``:

* **the barrier is made by the likelihood**: at the saddle ``t ~ c`` the model misfit is ``r^2``, a penalty of
  about ``r^4 / (2 sigma^2)`` nats (11.6 of the 11.8/10.0-nat depths at the suggested study point), while the prior
  is nearly flat between the saddle and the near mode (0.2 nats) and HIGHER at the saddle than at the far mode
  (1.6 nats) -- the prior alone would not separate the modes at all;
* **no symmetry connects the modes**: with ``c != 0`` the modes sit at different prior radii (``|c - r| != |c + r|``),
  the antipode ``-u`` of one mode is not a mode, and the prior rotation by ``pi`` (the ``tau = 0`` dynamics of a
  flattened Hamiltonian, note 33 §4) maps neither mode onto the other;
* **unequal masses**: the prior weight ratio of the modes is about ``exp(-2 c r)`` (times a width ratio), so the
  far mode holds a minority of the mass that a correct sampler must reproduce.

Knobs: ``c`` (asymmetry), ``r`` (mode separation ``2 r``), ``sigma`` (barrier depth; ``sigma_for_depth`` inverts
the numerically computed depth), ``no_parameters``. Suggested study point: ``c = 0.75``, ``r = 1.25`` (modes near
``t = -0.5`` and ``t = 2.0``, far-mode mass about 0.15) with ``sigma`` set for a 10-nat barrier.
"""
from __future__ import annotations

import numpy as np
import numpy.typing as npt
from scipy.optimize import brentq

import surrDAMH
from surrDAMH.solvers import Solver


def unit_direction(no_parameters: int) -> np.ndarray:
    v = np.arange(1, int(no_parameters) + 1, dtype=float)
    return v / np.linalg.norm(v)


class AsymBimodal(Solver):
    """``G(u) = (u . e - c)^2``; ``d`` parameters, 1 observation."""

    def __init__(self, solver_id=0, output_dir=None, no_parameters: int = 10, c: float = 0.75):
        self.no_parameters = int(no_parameters)
        self.no_observations = 1
        self.c = float(c)
        self.e = unit_direction(self.no_parameters)
        self.par_names = [f"u{i + 1}" for i in range(self.no_parameters)]

    def set_parameters(self, parameters: npt.NDArray) -> None:
        self.u = np.asarray(parameters, dtype=float)

    def get_observations(self) -> npt.NDArray:
        return np.array([(float(self.u @ self.e) - self.c) ** 2])


def make_problem(sigma: float, c: float = 0.75, r: float = 1.25, no_parameters: int = 10,
                 prior_bound: float | None = 8.0) -> "surrDAMH.Problem":
    """``surrDAMH.Problem`` of the family; data ``y = r^2``."""
    if not (sigma > 0.0 and r > 0.0):
        raise ValueError(f"sigma and r must be positive, got {sigma!r}, {r!r}")
    return surrDAMH.Problem(surrDAMH.distributions.Normal(mean=0.0, sd=1.0),
                            surrDAMH.distributions.Normal(mean=float(r) ** 2, sd=float(sigma)),
                            solver=AsymBimodal(no_parameters=no_parameters, c=c), prior_bound=prior_bound)


def log_density_t(t: npt.NDArray, sigma: float, c: float = 0.75, r: float = 1.25) -> npt.NDArray:
    """Unnormalised log density of ``t = u . e`` (the perpendicular part is ``N(0, I_{d-1})``)."""
    t = np.asarray(t, dtype=float)
    return -0.5 * t ** 2 - ((t - c) ** 2 - r ** 2) ** 2 / (2.0 * sigma ** 2)


def _landscape(sigma: float, c: float, r: float, n: int = 400_001, half_width: float = 10.0) -> dict:
    """Modes (two local maxima of ``log_density_t``), the saddle between them and the depths, numerically."""
    t = np.linspace(-half_width, half_width, n)
    f = log_density_t(t, sigma, c, r)
    left = (t < c)
    i_left = int(np.argmax(np.where(left, f, -np.inf)))
    i_right = int(np.argmax(np.where(~left, f, -np.inf)))
    i_saddle = i_left + int(np.argmin(f[i_left:i_right + 1]))
    if not (f[i_left] > f[i_saddle] and f[i_right] > f[i_saddle]):
        raise ValueError(f"not bimodal for sigma={sigma}, c={c}, r={r} (no saddle between the maxima)")
    return {"t": t, "f": f, "t_left": float(t[i_left]), "t_right": float(t[i_right]), "t_saddle": float(t[i_saddle]),
            "depth_left": float(f[i_left] - f[i_saddle]), "depth_right": float(f[i_right] - f[i_saddle]),
            "log_mode_ratio": float(f[i_right] - f[i_left])}


def sigma_for_depth(depth: float, c: float = 0.75, r: float = 1.25, which: str = "min") -> float:
    """``sigma`` such that the barrier (``which``: ``"min"`` of the two depths, ``"left"`` or ``"right"``) is ``depth`` nats."""
    key = {"min": None, "left": "depth_left", "right": "depth_right"}[which]

    def gap(log_sigma: float) -> float:
        try:
            land = _landscape(float(np.exp(log_sigma)), c, r, n=100_001)
        except ValueError:  # unimodal at this noise level: depth 0
            return -float(depth)
        value = min(land["depth_left"], land["depth_right"]) if key is None else land[key]
        return value - float(depth)

    return float(np.exp(brentq(gap, np.log(0.02), np.log(3.0), xtol=1e-6)))


def reference(sigma: float, c: float = 0.75, r: float = 1.25, no_parameters: int = 10) -> dict:
    """
    Exact posterior summaries by 1-D trapezoid quadrature in ``t``: the two modes ``t_left``/``t_right``, the saddle
    ``t_saddle``, the barrier depths from each mode (nats), ``mass_right`` = posterior mass of ``{t > t_saddle}``,
    ``E_t``, ``E_t2``, ``sd_t_left``/``sd_t_right`` (Laplace widths), per-coordinate ``mean = E_t e``,
    ``cov = I + (Var t - 1) e e^T``, ``sd``, ``m2``, and the quadrature change ``mass_right_change`` from halving the grid.
    """
    d = int(no_parameters)
    land = _landscape(sigma, c, r)
    t, f = land["t"], land["f"]
    w = np.exp(f - f.max())
    z = np.trapezoid(w, t)
    e_t = np.trapezoid(w * t, t) / z
    e_t2 = np.trapezoid(w * t ** 2, t) / z
    right = t > land["t_saddle"]
    mass_right = np.trapezoid(w[right], t[right]) / z
    t2, w2 = t[::2], w[::2]
    mass_right_half = np.trapezoid(w2[t2 > land["t_saddle"]], t2[t2 > land["t_saddle"]]) / np.trapezoid(w2, t2)
    e = unit_direction(d)
    var_t = e_t2 - e_t ** 2
    cov = np.eye(d) + (var_t - 1.0) * np.outer(e, e)

    def laplace_sd(t0: float) -> float:
        h = 1e-4
        curv = (log_density_t(t0 + h, sigma, c, r) - 2 * log_density_t(t0, sigma, c, r) + log_density_t(t0 - h, sigma, c, r)) / h ** 2
        return float((-curv) ** -0.5) if curv < 0 else float("nan")

    return {"sigma": float(sigma), "c": float(c), "r": float(r), "no_parameters": d, "y": float(r) ** 2,
            "e": e, "t_left": land["t_left"], "t_right": land["t_right"], "t_saddle": land["t_saddle"],
            "depth_left": land["depth_left"], "depth_right": land["depth_right"], "log_mode_ratio": land["log_mode_ratio"],
            "mass_right": float(mass_right), "mass_right_change": float(mass_right - mass_right_half),
            "E_t": float(e_t), "E_t2": float(e_t2), "sd_t_left": laplace_sd(land["t_left"]),
            "sd_t_right": laplace_sd(land["t_right"]), "mean": e_t * e, "cov": cov,
            "sd": np.sqrt(np.diag(cov)), "m2": np.diag(cov) + (e_t * e) ** 2}


def t_of(samples: npt.NDArray) -> npt.NDArray:
    """``t = u . e`` of an ``(n, d)`` array."""
    samples = np.asarray(samples, dtype=float)
    return samples @ unit_direction(samples.shape[1])


def crossings(samples: npt.NDArray, t_saddle: float, band: float = 0.3) -> int:
    """Mode-to-mode transitions of one chain: hysteresis ``t_saddle +- band`` (left below, right above)."""
    t = t_of(samples)
    state = 1 if t[0] > t_saddle else 0
    count = 0
    for value in t:
        new = 1 if value > t_saddle + band else (0 if value < t_saddle - band else state)
        count += int(new != state)
        state = new
    return count
