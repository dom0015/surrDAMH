"""
Surrogate models: cheap approximations of the forward model, trained during the run on the
collector rank and used by DAMH stages and gradient-based proposals.

Pass one as ``problem.run_sampling(conf, stages, surrogate_updater=...)``::

    updater = surrDAMH.surrogates.PolynomialSklearnUpdater(
        no_parameters=problem.no_parameters, no_observations=problem.no_observations, max_degree=3)

- ``PolynomialSklearnUpdater``      -- least-squares polynomial; cheap, good for smooth models
- ``RBFInterpolationUpdater``       -- radial basis function interpolation
- ``KDTreeUpdater``                 -- nearest-neighbour averaging
- ``NeuralNetworkUpdater`` -- torch MLP; the only one providing gradients
  (needed by ``proposals.Hamiltonian``)

``Updater`` and ``Evaluator`` are the base classes to subclass for your own surrogate
(``Updater`` trains and owns the data, ``Evaluator`` is the picklable object samplers call);
see ``docs/writing_a_surrogate.md``.
"""

from .nearest_kdtree import KDTreeUpdater
from .parent import Evaluator, Updater
from .polynomial_sklearn import PolynomialSklearnUpdater
from .rbf_scipy import RBFInterpolationUpdater
from .torch_perceptron_minibatches import NeuralNetworkUpdater

__all__ = ["PolynomialSklearnUpdater", "RBFInterpolationUpdater", "KDTreeUpdater",
           "NeuralNetworkUpdater", "Updater", "Evaluator"]
