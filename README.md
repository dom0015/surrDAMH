# surrDAMH
Python implementation of surrogate-accelerated Markov chain Monte Carlo methods for Bayesian inversion

Provides samples from the posterior distribution π(u|y) ∝ f<sub>η</sub>(y -- G(u)) π<sub>0</sub>(u), where y is a given vector of observations, G is an observation operator, f<sub>η</sub> is probability density function (pdf) of Gaussian observational noise, π<sub>0</sub>(u) is Gaussian prior pdf.


## Requirements
- numpy
- scipy
- pandas
- json
- mpi4py
- petsc4py (for "Darcy" example)
- MyFEM (for "Darcy" example)
    - github: https://github.com/Beremi/Simple_Python_PETSc_FEM/tree/SD
- pcdeflation (for the use of an own deflation basis in "Darcy" example)
    - `make -C examples/solvers/pcdeflation clean`
    - `make -C examples/solvers/pcdeflation build`
- cython (for pcdeflation build)

## Run
- **problem_name**
    - prepared toy examples: "simple", "simple_MPI", "Darcy"
    - loads configuration file "conf/" + **problem_name** + ".json"
- **N** = number of sampling processes
 
### run sampling process:
> ### `python3 run.py problem_name N (oversubscribe)`

- toy examples:
    - `python3 run.py simple 4`
    - `python3 run.py simple_MPI 4`
    - `python3 run.py Darcy 4`
- use "oversubscribe" if there are not enough slots available, for example:
    - `python3 run.py simple 4 oversubscribe`

- `python3 run.py simple 4` &nbsp; **<=>** &nbsp; `mpirun -n 4 python3 -m mpi4py surrDAMH/process_SAMPLER.py : -n 1 python3 -m mpi4py surrDAMH/process_SOLVER.py simple : -n 1 python3 -m mpi4py surrDAMH/process_COLLECTOR.py`
- `python3 run.py simple 4 oversubscribe` &nbsp; **<=>** &nbsp; `mpirun -n 4 --oversubscribe python3 -m mpi4py surrDAMH/process_SAMPLER.py : -n 1 --oversubscribe python3 -m mpi4py surrDAMH/process_SOLVER.py simple : -n 1 --oversubscribe python3 -m mpi4py surrDAMH/process_COLLECTOR.py`

Obtained samples are saved into saved_samples/**problem_name**.

### run visualization of obtained samples:
> ### `python3 run.py problem_name N visualize`

- toy examples:
    - `python3 run.py simple 4 visualize`
    - `python3 run.py simple_MPI 4 visualize`
    - `python3 run.py Darcy 4 visualize`
    
- `python3 run.py simple 4 visualize` &nbsp; **<=>** &nbsp; `python3 examples/visualization/simple.py 4`

## MPI processes
 - process_SAMPLER.py: **N** sampling processes based on the Metropolis-Hastings (MH) and the delayed-acceptance MH algorithm
 - process_SOLVER.py: solvers parent process that spawns new MPI processes (number and type specified in the JSON configuration file)
 - process_COLLECTOR.py: collects snapshots and uses them for the construction and updates of the surrogate model ("poly" or "rbf" ... polynomial or radial basis functions based surrogate model)
 
### spawned solvers:
- provide evaluations of an observation operator
    > observations = G(parameters)
- e.g. wrapper of an external library (see the "Darcy" example)

Path and constructor arguments of the spawed solver (wrapper) class are specified in the JSON configuration file. The following methods are required (executed by all MPI processes in the spawned communicator):

- `set_parameters(self, parameters: list[no_parameters]) -> None`
- `get_observations(self) -> list[no_observations]`

![MPI processes](img.pdf "MPI processes")

## JSON configuration file example & comments
  `{`<br>
  `    "problem_name": "Darcy",`<br>
  `    "no_parameters": 4,`<br>
  `    "no_observations": 3, ` *length of the vector of the observations, repetitive observations are not supported*<br>
  `    "problem_parameters": {`<br>
  `        "prior_mean": 0.0, ` *scalar or vector of length no_parameters*<br>
  `        "prior_std": 1.0, ` *scalar or vector or covariance matrix*<br>
`        "observations": [9.62638828, -5.90755323, -3.71883564], ` *vector of observations*<br>
`        "noise_std": [0.2, 0.1, 0.1] ` *standard deviation of observational noise (independent components)*<br>
`    },`<br>
`    "paths_to_append": [ ` *optional*<br>
`        "/home/simona/GIT/Simple_Python_PETSc_FEM"`<br>
`    ],`<br>
`    "solver_module_path": "examples/solvers/FEM_interfaces.py",`<br>
`    "solver_module_name": "FEM_interfaces",`<br>
`    "solver_init_name": "FEM", ` *spawned solver class*<br>
`    "solver_parameters": { ` *spawned solver class constructor arguments (in the form of a dictionary)*<br>
`    },`<br>
`    "solver_parent_parameters": {` *optional*<br>
`        "maxprocs": 4` *number of MPI processes of each spawned solver*<br>
`    },` <br>
`    "no_solvers": 3, ` *number of separate solvers to be spawned*<br>
`    "samplers_list": [ ` *dictionary of sampling algoritms run by all sampling processes in consecutive order*<br>
`        {`<br>
`            "type": "MH", ` *MH (does not use surrogate model) or DAMH (uses surrogate model)*<br>
`            "max_samples": 200, ` *maximal length of the chain*<br>
`            "time_limit": 60, ` *maximal sampling time in seconds*<br>
`            "proposal_std": 0.2, ` *standard deviation of the proposal distribution*<br>
`            "surrogate_is_updated": true ` *update surrogate while sampling*<br>
`        },`<br>
`        {`<br>
`            "type": "DAMH",`<br>
`            "max_samples": 2000,`<br>
`            "time_limit": 60,`<br>
`            "proposal_std": 0.2,`<br>
`            "surrogate_is_updated": true`<br>
`        },`<br>
`        {`<br>
`            "type": "DAMH",`<br>
`            "max_samples": 10000,`<br>
`            "time_limit": 60,`<br>
`            "proposal_std": 0.2,`<br>
`            "surrogate_is_updated": false`<br>
`        }`<br>
`    ],`<br>
`    "surrogate_type": "rbf", ` *"poly" or "rbf" (polynomial or radial basis functions based surrogate model)*<br>
`    "surr_solver_parameters": { ` *optional - arguments of class that evaluates surrogate model*<br>
`        "kernel_type": 1`<br>
`    },`<br>
`    "surr_updater_parameters": { ` *optional - arguments of class that updates surrogate model*<br>
`        "kernel_type": 1,`<br>
`        "no_keep": 500,`<br>
`        "expensive": false`<br>
`    }`<br>
`}`<br>
