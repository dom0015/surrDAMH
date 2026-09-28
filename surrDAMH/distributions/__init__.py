"""
Distributions used as the prior over parameters and as the likelihood of the observations.

Prior (how parameters are distributed a priori)::

    prior = surrDAMH.distributions.Normal(mean=0, sd=1, dim=no_parameters)
    prior = surrDAMH.distributions.PriorIndependentComponents([        # one component per parameter
        surrDAMH.distributions.LognormalComponent(mu=-40, sigma=3),
        surrDAMH.distributions.UniformComponent(a=0, b=0.5),
    ])

Likelihood (noise model of the measured data: the data are the mean)::

    likelihood = surrDAMH.distributions.Normal(mean=observed_data, sd=noise_sd)

Classes:

- ``Normal``                    -- Gaussian; univariate, independent components, or with a
  full covariance. Usable as prior or likelihood. As a prior it is standardized
  automatically (``StandardizedNormal``): the chain runs in N(0, I), ``transform`` maps back.
- ``PriorIndependentComponents`` -- prior assembled from one univariate component per
  parameter: ``NormalComponent``, ``UniformComponent``, ``LognormalComponent``,
  ``BetaComponent``. Its internal space is standard normal, which is what pCN and the
  dimension-robust Hamiltonian proposal expect.
- ``GaussianMixture``           -- multimodal prior.
- ``FromScipy``                 -- wraps a frozen ``scipy.stats`` distribution.
- ``Distribution``              -- base class to subclass for your own; see
  ``docs/concepts.md`` for the internal-versus-physical space design.

Note the difference between ``Normal`` (a whole distribution) and ``NormalComponent`` (one
parameter's component inside ``PriorIndependentComponents``).
"""

from .independent_components import (
    PriorIndependentComponents,
    UniformComponent,
    LognormalComponent,
    BetaComponent,
    NormalComponent,
)
from .normal import Normal, StandardizedNormal, standardize_prior
from .gaussian_mixture import GaussianMixture
from .parent import Distribution, FromScipy

__all__ = [
    "Distribution",
    "FromScipy",
    "Normal",
    "StandardizedNormal",
    "standardize_prior",
    "GaussianMixture",
    "PriorIndependentComponents",
    "UniformComponent",
    "LognormalComponent",
    "BetaComponent",
    "NormalComponent",
]
