#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with:
mpirun -n 6 python3 -m mpi4py run_model_problem.py
"""

import numpy as np
from mpi4py import MPI
import surrDAMH
# import surrDAMH.post_processing as post
import os
from surrDAMH.priors.independent_components import Uniform, Beta, Lognormal, Normal
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage
import matplotlib.pyplot as plt


conf = surrDAMH.Configuration(output_dir="output_nonlinear2_nn_relu20", no_parameters=2, no_observations=1,
                              use_collector=True, initial_sample_type="prior", transform_before_surrogate=True, save_raw_data=True)

# solver_spec = surrDAMH.solver_specification.SolverSpecExample1()
solver_spec = surrDAMH.solver_specification.SolverSpecNonlinearGeneric(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
# solver_spec = surrDAMH.solver_specification.SolverSpecExampleNonlinear(coef=[1.0, -1.0, 0.0, -1.0, 1.0, 0.0])
# solver_spec = surrDAMH.solver_specification.SolverSpecExampleNonlinear(coef=[1.0, 2.0, 0.0, 2.0, 1.0, 0.0])

# updater = surrDAMH.surrogates.PolynomialSklearnUpdater(conf.no_parameters, conf.no_observations, max_degree=15)
# updater = surrDAMH.surrogates.PolynomialSklearnSigmoidUpdater(conf.no_parameters, conf.no_observations, max_degree=15)
# updater = surrDAMH.surrogates.GaussianSklearnUpdater(conf.no_parameters, conf.no_observations)
# updater = surrDAMH.surrogates.PolynomialProjectionUpdater(conf.no_parameters, conf.no_observations, max_degree=5)
updater = surrDAMH.surrogates.RBFInterpolationUpdater(conf.no_parameters, conf.no_observations, kernel="thin_plate_spline")
# updater = surrDAMH.surrogates.NearestInterpolationUpdater(conf.no_parameters, conf.no_observations)
# updater = surrDAMH.surrogates.NNSklearnUpdater(conf.no_parameters, conf.no_observations, hidden_layer_sizes=(20, 20), activation='relu')
# updater = surrDAMH.surrogates.KDTreeUpdater(conf.no_parameters, conf.no_observations, no_nearest_neighbors=3) # TODO

# prior = surrDAMH.priors.PriorNormal(conf.no_parameters, mean=[5.0, 3.0], cov=[[4, -2], [-2, 4]])
list_of_components = [Normal(0, 4), Normal(0, 4), Uniform(-4, 8), Uniform(-4, 8), Beta(2, 2), Uniform(3, 5), Lognormal(0, 1), Normal(0, 2)]
list_of_components = list_of_components[0:conf.no_parameters]
prior = surrDAMH.priors.PriorIndependentComponents(list_of_components)
# prior = surrDAMH.priors.PriorNormal(conf.no_parameters, 0.0, 1.0)

observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec, [-2, 2])
noise_sd = 1.0  # np.abs(observations)*0.1  # TODO: cannot be zero
likelihood = surrDAMH.likelihoods.LikelihoodNormal(conf.no_observations, observations, sd=noise_sd)

list_of_stages = []
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd=0.5, max_evaluations=100))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.5, max_evaluations=200, surrogate_is_updated=True))
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd=0.5, max_evaluations=2000, surrogate_is_updated=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood, solver_spec=solver_spec, list_of_stages=list_of_stages)
data_for_analysis = sam.run()


comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)
    fig, axes = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[1, 2])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")


def create_image(x, y, Z, name):
    # plots matrix Z using imshow and saves to pdf file
    plt.figure()
    plt.imshow(Z, extent=(x.min(), x.max(), y.min(), y.max()), origin='lower', cmap='viridis', aspect="auto")
    plt.colorbar()  # (label='log scale')
    plt.title('Surrogate')
    plt.xlabel('X internal')
    plt.xlabel('Y internal')
    file_path = os.path.join(conf.output_dir, "post_processing_output", name + ".pdf")
    plt.savefig(file_path)  # , bbox_inches="tight")


ANALYZE = True
if ANALYZE and rank_world == conf.rank_collector:
    all_parameters = data_for_analysis.list_all_snapshots[0]

    # surrogate error of all surrogate models created during the sampling process
    error_min = []
    error_max = []
    error_mean = []
    for evaluator in data_for_analysis.list_all_evaluators:
        surrogate_observations = evaluator(all_parameters)
        surrogate_error = np.linalg.norm(surrogate_observations - data_for_analysis.list_all_snapshots[1], axis=1)
        error_min.append(min(surrogate_error))
        error_max.append(max(surrogate_error))
        error_mean.append(np.mean(surrogate_error))
    plt.yscale("log")
    plt.plot(error_max)
    plt.plot(error_mean)
    plt.plot(error_min)
    plt.legend(["max", "mean", "min"])
    file_path = os.path.join(conf.output_dir, "post_processing_output", "surrogate_error.pdf")
    plt.savefig(file_path, bbox_inches="tight")

    # preparation of a 2-D grid of parameters (first and second)
    N = 100
    xx = all_parameters[:, 0]
    x = np.linspace(min(xx), max(xx), N)
    yy = all_parameters[:, 1]
    y = np.linspace(min(yy), max(yy), N)
    X, Y = np.meshgrid(x, y)
    grid_points = np.column_stack((X.flatten(), Y.flatten()))
    if conf.no_parameters > 2:
        tmp = np.full((N*N, conf.no_parameters), prior.mean)
        tmp[:, :2] = grid_points
        grid_points = tmp

    # evaluation of the last surrogate model in all grid points
    evaluator = data_for_analysis.list_all_evaluators[-1]  # last evaluator
    Z_flat = evaluator(grid_points)
    print("///////////")
    print(evaluator(np.zeros((1, conf.no_parameters))))
    print(evaluator(np.ones((1, conf.no_parameters))))
    print(evaluator(2*np.ones((1, conf.no_parameters))))
    Z = Z_flat[:, 0].reshape(X.shape)

    # evaluation of the observation operator in all grid points
    from surrDAMH.solvers import get_solver_from_spec
    solver_instance = get_solver_from_spec(solver_spec)
    Z2 = Z_flat.copy()
    for i in range(N*N):
        if conf.transform_before_surrogate:
            solver_instance.set_parameters(grid_points[i, :])
        else:
            solver_instance.set_parameters(prior.transform(grid_points[i, :]))
        Z2[i] = solver_instance.get_observations()
    Z2 = Z2[:, 0].reshape(X.shape)

    # plot evaluations of surorgate model, observation operator and their difference
    create_image(x, y, Z, "last_surrogate")
    create_image(x, y, Z2, "exact")
    create_image(x, y, np.minimum(10, np.log(np.abs((Z2-Z)/Z2))), "error")


"""
Stages:
list of Stage
    type
    max_samples
    time_limit
    proposal_std --- prefferably Proposal instance
    surrogate_is_updated (bool)
    excluded (bool)
    adaptive (bool)
    use_only_surrogate (bool)
    target_rate (float)
    sample_limit (int)
(sampler)
"""

"""
prior po skupinách parametrů
OK initial_sample - pokud je to prior mean, nevolit tentýž snímek, ale s malým rozptylem náhodně
transform bere i více samplů najednou (np array) ? není třeba
sjednotit sample, parameters, atd. vs. snapshots
seřadit parametry funkcí - nutné, volitelné, =None, ...
check seed!!
imports within the package
readme, čím je knihovna specifická, vymezit škálu úloh
vyzkoušet knihovnu emcee, případně TinyDA
vytvořit modelové úlohy, kde se načte všechno včetně prioru, likelihooodu apod.
    - example.prior, example.likelihood apod. (třída example)
    - možná lepší exmply mít jen jako sadu skriptů
přidat uživatelsky bezpracné spuštění bez specifikování surrogatu a samplování
    - dopředný model, měření, prior a chyba (kdyžtak jen přibližně)
používat vizualizační knihovnu arviz (upravit data do požadovaného formátu pymc3)
surrogate se momentálně tvoří na datech z N(0,1), přidat moožnost tvoření na transformovaných datech
ujasnit si, co všechno se má ukládat, oddělit ukládání od zbytku kódu, možná použít iscream
zlepšit (nejen) názvosloví ohledně transformed, např. original/internal distribution
oddělit mpi od ostatních kódů

# TODO: prior, likelihood, etc. set_from_dict as another option?
# the dictionary can be also created on start and saved to yaml file
#  = everything needed to reproduce the sampling process, maybe also seed? (no)
#  - output dir and yaml path should not be in this file
# přidat možnost navázat na samplování
#  - může být na základě yaml file
#  - nutné uložit setting surrogate modelu
#  - taky může být možnost začít samplovat "od začátku", ale s existujícím (už naučeným) surrogate modelem
# in stage: if "use_only_surrogate", then "surrogate_is_updated" should be False (or implement this? no)
# in classes_SAMPLER: is G_initial_sample used?
# přidat adaptivní volbu směrodatné odchylky návrhového rozdělení vzhledem k target acceptance rate

Analýza kvality surrogatu 
- zaznamenávat všechny body, ve kterých byl počítán přesný model
- při každém updatu surrogatu vyhodnotit tento i všechny předchozí surrogaty ve všech bodech (pro analýzu)
- zjednodušit posuzování chyby tak aby mohlo zůstat součástí knihovny
- účelem je, aby nebyl používán surrogate, který je horší než ty předchozí
- případně diagnostikovat, že surrogate už je dobrý a není třeba dále vylepšovat
- sestrojit více surrogatů a použít ten lepší
- "How bad the surrogate model can be?"
- monitoring zlepsovani surrogatu v prubehu samplovani pomoci poctu zamitanych snimku (jako v DP/PANM)

Použít cizí multidimenzionální aproximaci/interpolaci
- např. scipy
- např. RBF, gauss, spline
- něco, čemu nevadí nelinearita - může jednuduše použít všechny známé body a nesnaží se vyhlazovat
- provést porovnání růzých surrogatů - analýza přesnosti, (výpočetní čas) 
- používají surrogaty i zámítnuté snímky?

Diagnostika stagnace řetězců
- takový řetězec odebrat, v lepším případě restartovat
- pracovat se spolehlivostí surrogatu
- pokud je snímek přijat, zjistit jaká je chyba surrogatu, pokud je moc velká, provést MH krok
"""
