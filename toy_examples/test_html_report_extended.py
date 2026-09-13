"""
Test script for the new html_report_extended function.
This script demonstrates how to use the extended HTML report generation.

Run with:
python3 test_html_report_extended.py
"""

import os

from solver_examples.solver_spec_examples import SolverSpecNonlinearGeneric

import surrDAMH
from surrDAMH.modules.tools import ensure_dir

# Configuration
conf = surrDAMH.Configuration(output_dir="out_test_html_extended", no_parameters=3, no_observations=1,
                              use_collector=False, use_solvers_pool=False, save_snapshots_to_file=False)
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

# Sampling process stages with different proposal standard deviations
list_of_stages = []
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=0.1, max_evaluations=500))
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=500))
list_of_stages.append(surrDAMH.stages.Stage(algorithm_type="MH", proposal_sd_or_cov=1.0, max_evaluations=500))

# Run sampling
sam = surrDAMH.SamplingFramework(conf=conf, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()

# Post processing:
samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

# Generate extended HTML report
path_output = os.path.join(conf.output_dir, "post_processing_output")
ensure_dir(path_output)
output_html = os.path.join(path_output, "report_extended.html")

print("\nGenerating extended HTML report...")

# Create the extended HTML report with parameter names
parameter_names = ["param_0", "param_1", "param_2"]

samples.html_report_extended(
    no_observations=0,  # Set to 0 since we're not saving raw data
    stages_to_disp=[0, 1, 2],  # Display all stages
    output_file=output_html,
    bins1d=20,
    bins2d=20,
    par_names=parameter_names
)

print(f"\n{'='*60}")
print(f"Extended HTML report successfully generated!")
print(f"Report location: {output_html}")
print(f"{'='*60}\n")
print("You can open this file in a web browser to view the complete report.")
print("The report includes:")
print("  - Summary statistics for all stages")
print("  - Posterior mean and covariance (overall and per stage)")
print("  - Parameter distribution histograms (overall and per stage)")
print("  - Chain traces (overall and per stage)")
print("  - Cumulative averages (overall and per stage)")
