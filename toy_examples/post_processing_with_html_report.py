"""
Example: Using html_report_extended() with the post_processing_example.py

This script demonstrates how to use the new html_report_extended function
with the existing post_processing example.
"""

import os

from solver_examples.solver_spec_examples import SolverSpecNonlinearGeneric

import surrDAMH
from surrDAMH.modules.tools import ensure_dir

conf = surrDAMH.Configuration(output_dir="out_post_proc_example", no_parameters=3, no_observations=1,
                              use_collector=False, use_solvers_pool=False, save_snapshots_to_file=True)
solver_spec = SolverSpecNonlinearGeneric(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
prior_mean = [2.0] * conf.no_parameters
prior = surrDAMH.distributions.Normal(mean=prior_mean, sd=1.0)
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

# =============================================================================
# NEW: Generate Extended HTML Report
# =============================================================================
print("\n" + "="*80)
print("Generating Extended HTML Report...")
print("="*80 + "\n")

# Define custom parameter names (optional)
parameter_names = ["Young's Modulus", "Poisson's Ratio", "Density"]

# Generate the extended HTML report with all visualizations and statistics
output_html = os.path.join(path_figures, "report_extended.html")

samples.html_report_extended(
    no_observations=conf.no_observations,  # Number of observations (can be 0 if not using raw_data)
    chosen_observations=None,              # Indices of observations to display (None = all)
    grid=None,                             # Time/spatial grid for observations (optional)
    grid_interp=None,                      # Interpolation grid (optional)
    bins=[20],                             # Bins for observation histograms (optional)
    chains_to_disp=None,                   # Which chains to display (None = all)
    stages_to_disp=[0, 1, 2, 3],          # Which stages to display (list of indices)
    observations=observation,              # Actual observations (optional, for comparison)
    cmap="viridis_r",                      # Colormap for observation histograms
    output_file=output_html,               # Output HTML file path
    bins1d=20,                             # Number of bins for 1D histograms
    bins2d=20,                             # Number of bins for 2D histograms
    par_names=parameter_names              # Custom parameter names (optional)
)

print("\n" + "="*80)
print(f"Extended HTML Report Generated Successfully!")
print(f"Location: {output_html}")
print("="*80 + "\n")

print("The extended report includes:")
print("  ✓ Summary statistics table for all stages")
print("  ✓ Overall analysis (combined stages):")
print("     - Posterior mean and covariance matrix")
print("     - Parameter distribution histograms (1D and 2D)")
print("     - Chain traces")
print("     - Cumulative averages")
print("  ✓ Individual stage analysis with all of the above for each stage")
print("  ✓ Observation histograms (if available)")
print("\nOpen the HTML file in a web browser to view the full interactive report.")
