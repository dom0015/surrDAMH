# surrDAMH documentation

Surrogate-accelerated MCMC (delayed-acceptance Metropolis-Hastings, DAMH) for Bayesian
inverse problems, parallelised with mpi4py. Python 3.12.

This is the user-facing documentation (WS11 of `library_notes/09_improvement_plan.md`).
For the ongoing refactor's rationale, findings and open decisions see `library_notes/`
(start at `00_overview.md`); for the posterior-affecting changes so far see
`../CHANGELOG.md`.

## Pages

| Page | Covers |
|---|---|
| [`concepts.md`](concepts.md) | Bayesian setting, internal vs. physical space, MH, DAMH/DAMH-SMU derivation |
| [`configuration.md`](configuration.md) | Every `Configuration` field: type, default, meaning, posterior-affecting? |
| [`stages.md`](stages.md) | Every `Stage` field, invalid combinations |
| [`writing_a_solver.md`](writing_a_solver.md) | The `Solver` contract, `SolverSpec`, the solvers pool |
| [`writing_a_surrogate.md`](writing_a_surrogate.md) | The `Updater`/`Evaluator` contract (today's) |
| [`outputs.md`](outputs.md) | On-disk layout (format v2), the run manifest, `read_run` |
| [`running.md`](running.md) | Process counts per role combination, `mpiexec`, `run_local`, continuation, tests |

## Quick start

Install (`pip install -e .` or `./rebuild_surrDAMH.sh`), then from `toy_examples/`:

```bash
mpiexec -n 4 python3 -m mpi4py minimal_example.py       # plain MH, no surrogate
mpiexec -n 4 python3 -m mpi4py template_experiment.py    # MH -> DAMH-SMU -> DAMH, annotated
```

`minimal_example.py` is the smallest runnable script (prior, likelihood, one MH stage).
`template_experiment.py` is the canonical starting point for a new experiment: copy it
and edit its numbered sections (forward model, prior/likelihood, configuration,
surrogate, stages) for your own problem — see `writing_a_solver.md` and
`writing_a_surrogate.md` for what goes into sections 2 and 5. Both scripts, and every
process-count combination, are explained in `running.md`.

## The maintained examples

`toy_examples/` is the only maintained example set (`library_notes/05_toy_examples_and_docs.md`);
the `README.md` table lists every file with its process count. In tiers:

- **basic**: `minimal_example.py`, `one_process_only.py`, `own_solver.py`,
  `with_simple_postprocessing.py`, `typical_example.py`, `post_processing_with_html_report.py`
- **canonical template**: `template_experiment.py`
- **surrogate**: `neural_network_surrogate.py`
- **advanced** (pCN + Hamiltonian proposals, minibatch MLP, `TestData`, `SurrogateRestart`,
  extended report): `toy_example_hamilton.py` on a trivial solver, `sampling_diffusion_grf.py`
  as the reference version on a real forward model
- **FEniCSx tier** (needs dolfinx): `sampling_diffusion_grf.py`,
  `sampling_diffusion_grf_simplified.py`, `sampling_TSX.py` plus the solver modules
  `grf_diffusion.py`, `wrapper.py`, `tunnel_with_subdomains.py`

Archived on 2026-09-17 (duplicates of the above): `typical_example_generic.py`,
`post_processing_example.py`, `test_html_report_extended.py`, `neural_network_surrogate_copy.py`.
