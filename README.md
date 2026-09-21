# surrDAMH
Python implementation of surrogate-accelerated Markov chain Monte Carlo methods for postrior sampling in Bayesian inversion. Based on the delayed acceptance Metropolis-Hastings algorithm.

## Requirements
- numpy
- scipy
- mpi4py
- matplotlib
- torch (for neural network surrogate model)
- scikit-learn (for polynomial surrogate model)
- pandas (for post-processing)
- emcee (for post-processing)

## Bayesian inverse problem setting
What is given:
 - forward model: $G:\mathbb{R}^{n}\rightarrow\mathbb{R}^{m}$
 - observations (noisy outputs of the forward model): $y\in\mathbb{R}^{m}$
 - prior probability density function (pdf): $f_{U}$
 - noise pdf: $f_{Z}$

Input parameters $u\in\mathbb{R}^{n}$ are unknown.

Additive noise is considered; therefore, the posterior pdf is given by the following formula:

$f_{U|Y}\left(u|y\right)\propto{f_{Z}\left(y-G\left(u\right)\right)}{f_{U}\left(u\right)}$

## Parallel processes:
 - several Markov chains generated in parallel
 - the chains share one surrogate model, which is refined during the sampling process using data from all chains
 - the chains share a pool of spawned solvers (processes that evaluate $G$)
    - assumed to be the computationally most demanding part
    - a solver is typically a linked numerical library
    - number of solvers is typically lower than number of chains

## Getting started:
 - open repository in the Docker container (e.g. using the Dev Containers extension of Visual Studio Code)
 - `pip install .`

Before running the sampling process, it is necessary to specify:
 - configuration (basic settings, e.g. number of solvers, initial samples, ...)
 - prior
 - likelihood
 - solver specification
 - surrogate model updater
 - list of stages

See examples in the **toy_examples** folder, e.g.:
 - `cd toy_examples/`
 - `mpiexec -n 2 python3 -m mpi4py minimal_example.py` (2 processes is the minimum for this example; more processes give more chains)

### The maintained examples

| Example | What it shows | How to run it (from `toy_examples/`) |
|---|---|---|
| `minimal_example.py` | smallest MH run, solvers pool, no surrogate | `mpiexec -n 2 python3 -m mpi4py minimal_example.py` |
| `one_process_only.py` | one process, local solver, no MPI orchestration | `python3 one_process_only.py` |
| `own_solver.py` | your own `Solver` subclass + polynomial surrogate, MH → DAMH-SMU → DAMH | `mpiexec -n 4 python3 -m mpi4py own_solver.py` |
| `with_simple_postprocessing.py` | the same via the solvers pool, plus basic plots | `mpiexec -n 6 python3 -m mpi4py with_simple_postprocessing.py` |
| `typical_example.py` | the "typical" 3-stage DAMH-SMU workflow | `mpiexec -n 4 python3 -m mpi4py typical_example.py` |
| **`template_experiment.py`** | **canonical starting point — copy this file** | `mpiexec -n 4 python3 -m mpi4py template_experiment.py` |
| `neural_network_surrogate.py` | MLP surrogate instead of the polynomial one | `mpiexec -n 4 python3 -m mpi4py neural_network_surrogate.py` |
| `post_processing_with_html_report.py` | sampling + `html_report_extended` in one script | `python3 post_processing_with_html_report.py` |
| `toy_example_hamilton.py` | advanced workflow (pCN → DAMH + Hamiltonian proposal, minibatch MLP, test data, surrogate restart) on a trivial solver | `mpiexec -n 4 python3 -m mpi4py toy_example_hamilton.py` |
| `sampling_diffusion_grf.py` | the same workflow on the GRF-diffusion forward model (**FEniCSx**) | `mpiexec -n 4 python3 -m mpi4py sampling_diffusion_grf.py` |
| `sampling_diffusion_grf_simplified.py` | the same solver, MH only, no surrogate (**FEniCSx**) | `python3 sampling_diffusion_grf_simplified.py` |
| `sampling_TSX.py` | real TSX inversion; needs `wrapper.py`, `tunnel_with_subdomains.py` and mesh files (**FEniCSx**) | see the file header |

`grf_diffusion.py`, `wrapper.py` and `tunnel_with_subdomains.py` are solver modules, not runnable
examples; `solver_examples/` holds the toy solvers and their `SolverSpec`s.

Notes on running the examples:
 - run them **from the `toy_examples/` directory**: the solver specifications use a relative `solver_module_path`, which is resolved against the working directory when the `SolverSpec` is constructed
 - `sampling_TSX.py` (with `wrapper.py`, `tunnel_with_subdomains.py`) and `sampling_diffusion_grf.py` / `sampling_diffusion_grf_simplified.py` (with `grf_diffusion.py`) require **FEniCSx / dolfinx** in addition to the requirements above
 - the layout of the output directory: `sampling_output/`, `solver_output/`, `post_processing_output/`
 - full documentation is in [`docs/`](docs/README.md)

## Surrogate models:
The following non-intrusive surrogate models are implemented. They are constructed from snapshots $\left(u^{\left(k\right)},G\left(u^{\left(k\right)}\right)\right)$ and they can be adaptively refined during the sampling process.
 - polynomial chaos approximation - complete polynomials, adaptive increase of maximum degree based on available data (using scikit-learn)
 - radial basis functions interpolation (RBF) - combined with polynomials up to chosen degree (using SciPy)
 - approximation using nearest points identified using kd-tree (using SciPy)
 - multilayer perceptron regressor (using PyTorch)

![MPI processes](img.png "MPI processes")