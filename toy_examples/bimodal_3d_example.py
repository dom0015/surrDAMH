#!/usr/bin/env python3
"""
Bimodal 3-D problem (modes at +-(1, 1, 1), saddle at the origin, valley depth in nats). Run with:
    mpiexec -n 4 python3 -m mpi4py bimodal_3d_example.py
(4 samplers, no collector; the solver runs inside the samplers.) Prints the mode masses, the mode-to-mode
crossings per chain and E[t^2] against the quadrature reference.
"""
import numpy as np
from mpi4py import MPI

import surrDAMH
from surrDAMH.proposals import RandomWalk
from surrDAMH.stages import Stage
from solver_examples.bimodal3d import make_problem, reference, t_of, crossings

VALLEY_DEPTH = 5.0                                                     # nats: density at the saddle = exp(-5) x mode
problem = make_problem(VALLEY_DEPTH)
conf = surrDAMH.Configuration(output_dir="out_bimodal_3d_example", use_solvers_pool=False, use_collector=False)
stages = [
    Stage(proposal=RandomWalk(), max_evaluations=2000, burn_in=True),  # adaptive random walk warm-up
    Stage(proposal=RandomWalk(), max_evaluations=20000),               # continues from the tuned scale
]
run = problem.run_sampling(conf, stages)

if MPI.COMM_WORLD.Get_rank() == 0:
    ref = reference(VALLEY_DEPTH)
    samples = surrDAMH.post_processing.Samples(3, conf.output_dir)
    chains = [np.asarray(c) for c in samples.list_of_stages[1].samples]
    t = np.concatenate([t_of(c) for c in chains])
    print(f"valley depth {VALLEY_DEPTH} nats: sigma = {ref['sigma']:.4f}, data y = {ref['y']:.4f}, "
          f"saddle/mode density ratio = {ref['saddle_ratio']:.2e}")
    print(f"mass of the right mode (t > 0): {np.mean(t > 0):.3f} (reference 0.5)")
    print(f"E[t^2]: {np.mean(t ** 2):.4f} (reference {ref['E_t2']:.4f})")
    print("crossings per chain:", [crossings(c) for c in chains])
