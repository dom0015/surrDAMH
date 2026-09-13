import surrDAMH

from grf_diffusion import Solver_diffusion_GRF


# TODO: now, only solver_instance is tested, test also with solver_spec
solver_instance = Solver_diffusion_GRF()
no_parameters =solver_instance.no_parameters
no_observations = solver_instance.no_observations
ref_observations = solver_instance.generate_artificial_observations()

conf = surrDAMH.Configuration( # TODO: comments not shown
    no_parameters=no_parameters,
    no_observations=no_observations,
    output_dir="out_grf_simplified",
    use_solvers_pool=False, # TODO: check both possibilities, consider better naming
    use_collector=False, # TODO: when surrogate in not specified, this must be set manually; at least prepare good error messages
)

prior = surrDAMH.distributions.Normal(mean=0, sd=1, d=no_parameters) # TODO: rename d, sd
likelihood = surrDAMH.distributions.Normal(mean=ref_observations, sd=0.1)

list_of_stages = []
list_of_stages.append(surrDAMH.stages.Stage(
    proposal_sd_or_cov=0.1, # TODO: rename proposal_sd_or_cov
    time_limit=10,
))

sf = surrDAMH.SamplingFramework(conf, prior, likelihood, list_of_stages, 
                                solver_instance=solver_instance, # TODO: either solver_instance or solver_spec must be provided
)
"""
                                surrogate_updater=surrogate_updater, # TODO: either surrogate_updater or surrogate_evaluator must be provided
                                initial_snapshots=initial_snapshots, 
                                surrogate_test_data=surrogate_test_data)
                                """

sf.run()

# TODO: pass arguments to html_report_extended, choose better names
sf.write_report(observations=ref_observations)