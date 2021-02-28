# MCMC-Bayes-python
Python implementation of surrogate-accelerated MCMC methods for Bayesian inversion
 
### inputs:
 - **N** ... number of sampling processes
 - **problem_name** ... name of the problem
    - prepared toy examples:
        - simple
        - simple_MPI
        - Darcy
    - loads configuration file "conf/" + **problem_name** + ".json"
 
### run:
mpirun -n **N** python3 process_SAMPLER.py : -n 1 python3 process_SOLVER.py **problem_name** : -n 1 python3 process_COLLECTOR.py

#### if there are not enough slots available:
mpirun -n **N** \-\-oversubscribe python3 process_SAMPLER.py : -n 1 \-\-oversubscribe python3 process_SOLVER.py **problem_name** : -n 1 \-\-oversubscribe python3 process_COLLECTOR.py

#### for example:
mpirun -n 4 \-\-oversubscribe python3 process_SAMPLER.py : -n 1 \-\-oversubscribe python3 process_SOLVER.py simple : -n 1 \-\-oversubscribe python3 process_COLLECTOR.py

### MPI processes:
 - process_SAMPLER.py: **N** sampling processes based on the Metropolis-Hastings (MH) and the delayed-acceptance MH algorithm
 - process_SOLVER.py: solver parent process that spawns new MPI processes
 - process_COLLECTOR.py: collects snapshots and uses them for the construction and updates of the surrogate model ("poly" or "rbf" ... polynomial or radial basis functions based surrogate model)

### visualization of obtained chains:
 - python3 visualization_simple.py **N**
 - python3 visualization_simple_MPI.py **N**
 - python3 visualization_simple_Darcy.py **N**
 
