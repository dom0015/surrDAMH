#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Run with (replace 4 with required number of MPI processes):
mpiexec -n 4 python3 -m mpi4py typical_example.py

(Here, one of the processes will be used as collector,
one as solvers pool, remaining processes will be used as samplers.
Additional MPI processes will be spawned by solvers pool.)
"""

import os

import numpy as np
import scipy.stats
import solver_examples.solver_spec_examples
from mpi4py import MPI

import surrDAMH
from surrDAMH.modules.tools import ensure_dir
from surrDAMH.stages import Stage

# solver example, takes 2 parameters, returns 1 observation:
solver_spec = solver_examples.solver_spec_examples.SolverSpecNonlinearGeneric(no_parameters=3, no_observations=3)

# configuration (specification of basic settings of the sampling framework):
conf = surrDAMH.Configuration(output_dir="out_typical_example_generic", no_parameters=3, no_observations=3, min_snapshots_initial=4, 
                              save_snapshots_to_file=True)

# choice of surrogate model:
updater = surrDAMH.surrogates.PolynomialSklearnUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
# updater = surrDAMH.surrogates.RBFInterpolationUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations)
# updater = surrDAMH.surrogates.KDTreeUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations, no_nearest_neighbors=5)
# updater = surrDAMH.surrogates.PyTorchNNOngoingUpdater(no_parameters=conf.no_parameters, no_observations=conf.no_observations, hidden_layer_sizes=(20, 10))

# Gaussian prior distribution:
prior = surrDAMH.distributions.FromScipy(scipy.stats.multivariate_normal(mean=[-1.0, 1.0, 0.0], cov=np.eye(3)))

# likelihood (additive Gaussian noise):
observations = surrDAMH.solvers.calculate_artificial_observations(solver_spec=solver_spec, parameters=[-2, 2, 1])
print("Artificial observations: ", observations)
observations = observations.flatten()
likelihood = surrDAMH.distributions.Normal(mean=observations, sd=0.5)

# sampling process stages:
list_of_stages = []
# during MH stage, initial surrogate model is constructed:
list_of_stages.append(Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, max_evaluations=400))
# during DAMH-SMU stage, surrogate model is further updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=1.0, time_limit=20, surrogate_model_updates=True))
# during DAMH stage, surrogate model is used but not updated:
list_of_stages.append(Stage(algorithm_type="DAMH", proposal_sd_or_cov=1.0, time_limit=20, surrogate_model_updates=False))

sam = surrDAMH.SamplingFramework(conf, surrogate_updater=updater, prior=prior, likelihood=likelihood,
                                 list_of_stages=list_of_stages, solver_spec=solver_spec)
sam.run()

# post processing:
comm_world = MPI.COMM_WORLD
rank_world = comm_world.Get_rank()
if rank_world == 0:
    samples = surrDAMH.post_processing.Samples(conf.no_parameters, conf.output_dir)

    # print summary:
    samples.get_summary()

    # save histograms grid to file:
    fig, _ = samples.plot_hist_grid(bins1d=30, bins2d=30, stages_to_disp=[1, 2])
    ensure_dir(os.path.join(conf.output_dir, "post_processing_output"))
    file_path = os.path.join(conf.output_dir, "post_processing_output", "histograms.pdf")
    fig.savefig(file_path, bbox_inches="tight")

    # save plot of chains to file:
    fig, _ = samples.plot_chains(stages_to_disp=[2])
    file_path = os.path.join(conf.output_dir, "post_processing_output", "chains.pdf")
    fig.savefig(file_path, bbox_inches="tight")

    # save plot of chain cummulative averages to file:
    fig, _ = samples.plot_chains(average=True, stages_to_disp=[2])
    file_path = os.path.join(conf.output_dir, "post_processing_output", "averages.pdf")
    fig.savefig(file_path, bbox_inches="tight")


    # =============================================================================
    # NEW: Generate Extended HTML Report
    # =============================================================================
    print("\n" + "="*80)
    print("Generating Extended HTML Report...")
    print("="*80 + "\n")

    path_figures = os.path.join(conf.output_dir, "post_processing_output")
    ensure_dir(path_figures)

    # Define custom parameter names (optional)
    parameter_names = ["PAR1_name", "PAR2_name", "PAR3_name"]  # Example custom names for parameters

    # Generate the extended HTML report with all visualizations and statistics
    output_html = os.path.join(path_figures, "report_extended.html")

    samples.html_report_extended(
        no_observations=conf.no_observations,  # Number of observations (can be 0 if not using raw_data)
        chosen_observations=None,              # Indices of observations to display (None = all)
        grid=None,                             # Time/spatial grid for observations (optional)
        grid_interp=None,                      # Interpolation grid (optional)
        bins=None,                             # Bins for observation histograms (optional)
        chains_to_disp=None,                   # Which chains to display (None = all)
        stages_to_disp=[0, 1, 2],          # Which stages to display (list of indices)
        observations=observations,              # Actual observations (optional, for comparison)
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
