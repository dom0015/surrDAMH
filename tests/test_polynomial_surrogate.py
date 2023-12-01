from surrDAMH.surrogates.polynomial_new import PolynomialProjectionUpdater as TrainerNew
from surrDAMH.surrogates.polynomial_sklearn import PolynomialSklearnUpdater as TrainerSklearn
import numpy as np
import time

# binomial - number of terms:
#           degree: 0 1 2  3  4  5  6  7  8  9
#   of that degree: 1 2 3  4  5  6  7  8  9 10
# all up to degree: 1 3 6 10 15 21 28 36 45 55


def observation_operator(x, y): return x*x*x*y - y*y


no_snapshots = 20  # to learn the surrogate model
no_test_points = 10000  # to test the surrogate model

# CREATE SNAPSHOTS
np.random.seed(5)
parameters = np.random.randn(no_snapshots, 2)
x = parameters[:, 0]
y = parameters[:, 1]
observations = observation_operator(x, y)

# CREATE TEST DATA
test_data = np.random.randn(no_test_points, 2)
x_test = test_data[:, 0]
y_test = test_data[:, 1]
z_test = observation_operator(x_test, y_test)
z_test = z_test.reshape((-1, 1))

# CREATE AND APPLY THE SURROGATE MODELS
trainer_new = TrainerNew(2, 1, 5, "jlkjl")  # own
trainer_sklearn = TrainerSklearn(2, 1, 5)  # sklearn
for trainer in [trainer_new, trainer_sklearn]:
    print("------")
    trainer.add_data(parameters, observations)
    t = time.time()
    evaluator = trainer.get_evaluator()
    t_training = time.time() - t
    t = time.time()
    evaluations = evaluator(test_data)
    t_evaluation = time.time() - t

    # COMPARE RESULTS
    rel_err = np.abs((z_test-evaluations)/z_test)
    print(trainer)
    print("relative error (min, max, norm):", min(rel_err), max(rel_err), np.linalg.norm(rel_err))
    print("training time:", t_training)
    print("evaluation time:", t_evaluation)
