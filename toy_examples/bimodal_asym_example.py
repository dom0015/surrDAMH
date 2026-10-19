#!/usr/bin/env python3
"""
Asymmetric bimodal problem with a likelihood-made barrier (``solver_examples/bimodal_asym.py``), d = 10. Run with:
    mpiexec -n 4 python3 -m mpi4py bimodal_asym_example.py
(4 samplers, no collector.) Prints the modes, the barrier, the minority-mode mass and the crossings per chain
against the quadrature reference. The barrier is set to 3 nats here so that an adaptive random walk still crosses;
the C3/note-32 studies use 10 nats (``sigma_for_depth(10.0)``).
"""
import numpy as np
from mpi4py import MPI

import surrDAMH
from surrDAMH.proposals import RandomWalk
from surrDAMH.stages import Stage
from solver_examples.bimodal_asym import make_problem, reference, sigma_for_depth, t_of, crossings

D, C, R = 10, 0.75, 1.25
SIGMA = sigma_for_depth(3.0, C, R)                                     # 3-nat barrier (the study uses 10)
problem = make_problem(SIGMA, C, R, no_parameters=D)
conf = surrDAMH.Configuration(output_dir="out_bimodal_asym_example", use_solvers_pool=False, use_collector=False)
stages = [
    Stage(proposal=RandomWalk(), max_evaluations=4000, burn_in=True),
    Stage(proposal=RandomWalk(), max_evaluations=40000),
]
run = problem.run_sampling(conf, stages)

if MPI.COMM_WORLD.Get_rank() == 0:
    ref = reference(SIGMA, C, R, no_parameters=D)
    samples = surrDAMH.post_processing.Samples(D, conf.output_dir)
    chains = [np.asarray(c) for c in samples.list_of_stages[1].samples]
    t = np.concatenate([t_of(c) for c in chains])
    print(f"sigma = {SIGMA:.4f}: modes at t = {ref['t_left']:.3f} / {ref['t_right']:.3f}, saddle {ref['t_saddle']:.3f}, "
          f"barrier {ref['depth_left']:.2f} / {ref['depth_right']:.2f} nats, log mode ratio {ref['log_mode_ratio']:.2f}")
    print(f"mass of the right (far) mode: {np.mean(t > ref['t_saddle']):.3f} (reference {ref['mass_right']:.3f})")
    print(f"E[t]: {np.mean(t):.4f} (reference {ref['E_t']:.4f});  E[t^2]: {np.mean(t ** 2):.4f} (reference {ref['E_t2']:.4f})")
    print("crossings per chain:", [crossings(c, ref["t_saddle"]) for c in chains])
