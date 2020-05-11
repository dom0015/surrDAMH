# MCMC-Bayes-python
Python implementation of adaptive MCMC methods for Bayesian inversion

## test run (solver are spawned)
mpirun -n 3 --oversubscribe python3 process_SAMPLER.py : -n 1 python3 process_SOLVER.py : -n 1 python3 process_COLLECTOR.py
