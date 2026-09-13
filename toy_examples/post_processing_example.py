"""
Run with:
python3 post_processing_example.py

Solver is local (i.e. solvers pool is not used),
no surrogate model is used (i.e. no collector is used
and samples are generated using the basic MH algorithm).
"""

import os

from solver_examples.solver_spec_examples import SolverSpecNonlinearGeneric

import surrDAMH
from surrDAMH.modules.tools import ensure_dir

conf = surrDAMH.Configuration(output_dir="out_post_proc_example", no_parameters=3, no_observations=1,
                              use_collector=False, use_solvers_pool=False, save_snapshots_to_file=True)
solver_spec = SolverSpecNonlinearGeneric(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
prior_mean = [2.0] * conf.no_parameters
prior = surrDAMH.distributions.PriorIndependentComponents([
    surrDAMH.distributions.NormalComponent(mu=float(mu), sigma=1.0) for mu in prior_mean
])
obs_prior_mean = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=prior_mean)
print("Solver output for prior mean: ", obs_prior_mean)
observation = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[3.0, 1.0, 3.0])
print("Artificial observation: ", observation)
likelihood = surrDAMH.distributions.Normal(mean=observation, sd=0.2)

# sampling process stages:
list_of_stages = []
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=0.1, max_evaluations=1000))
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=1000))
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=1.0, max_evaluations=1000))
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=1.5, max_evaluations=1000))

sam = surrDAMH.SamplingFramework(conf=conf, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()

# Post processing:

samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

# print summary:
samples.get_summary()

# print mean and covariance matrix
posterior_mean, cov = samples.get_mean_and_cov(stages_to_disp=[1, 2, 3])
print("Posterior mean approximation: ", posterior_mean)
print("Covariance matrix: ")
print(cov)

path_figures = os.path.join(conf.output_dir, "post_processing_output")
ensure_dir(path_figures)

# save histograms grid to file:
fig, _ = samples.plot_hist_grid(bins1d=20, bins2d=20, stages_to_disp=[1, 2, 3])
file_path = os.path.join(path_figures, "histograms.pdf")
fig.savefig(file_path, bbox_inches="tight")

# save plot of chains and plot of cummulative averages to file, each stage separately:
for i in range(len(list_of_stages)):
    fig, _ = samples.plot_chains(stages_to_disp=[i])
    file_path = os.path.join(path_figures, "chains" + str(i) + ".pdf")
    fig.savefig(file_path, bbox_inches="tight")

    # save plot of chain cummulative averages to file:
    fig, _ = samples.plot_chains(average=True, stages_to_disp=[i])
    file_path = os.path.join(path_figures, "averages" + str(i) + ".pdf")
    fig.savefig(file_path, bbox_inches="tight")

fig = samples.hist_observations(no_observations=conf.no_observations, bins=[20], observations=observation)
file_path = os.path.join(path_figures, "hist_observations.pdf")
fig.savefig(file_path, bbox_inches="tight")

print("Figures saved to", path_figures)

# Reports:
samples.html_report(no_observations=conf.no_observations)