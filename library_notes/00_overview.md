# surrDAMH — architecture overview

Notes produced by reading the working tree (incl. uncommitted changes) on 2026-09-12.
No code was executed. `out_*` directories were not opened.

## 1. What the library does

surrDAMH samples the posterior of a Bayesian inverse problem

    f(u | y) ∝ f_noise(y − G(u)) · f_prior(u)

with several Markov chains in parallel (one MPI rank per chain). The expensive forward
model `G` is evaluated either locally in each sampler process or by a shared pool of
spawned solver processes. A shared surrogate model of `G` is trained on the fly from
all `(u, G(u))` snapshots and used in the **delayed-acceptance Metropolis-Hastings
(DAMH)** algorithm, optionally with **surrogate model updates during sampling (DAMH-SMU)**.

Key idea: the chain state lives in an *internal* parameter space (for
`PriorIndependentComponents` this is N(0, I)); `prior.transform` maps to the *physical*
parameters that the solver receives. Prior log-pdf, proposals and saved samples (if
`transform_before_saving=False`) are in the internal space.

## 2. Package layout

| Path | Role |
|---|---|
| `surrDAMH/core.py` | `SamplingFramework`: role dispatch in `run()`, HTML report in `write_report()` |
| `surrDAMH/configuration.py` | `Configuration` dataclass; derives rank layout from `MPI.COMM_WORLD` size |
| `surrDAMH/stages.py` | `Stage` dataclass (one MCMC stage: algorithm, proposal, stopping rule) |
| `surrDAMH/process_SAMPLER.py` | per-chain driver: builds proposal + algorithm per stage, runs it |
| `surrDAMH/process_SOLVER.py` | solver-pool manager rank: spawns `no_solvers` child processes, queues requests |
| `surrDAMH/process_CHILD.py` | spawned solver process (script run via `MPI.COMM_SELF.Spawn`) |
| `surrDAMH/process_COLLECTOR.py` | collector rank: receives snapshots, trains surrogate, ships evaluators |
| `surrDAMH/modules/algorithms.py` | `Sample`, `AlgorithmBase`, `Algorithm_MH`, `Algorithm_DAMH` |
| `surrDAMH/modules/proposals.py` | `GaussRandomWalk`, `GaussRandomWalk_adaptive`, `PCN`, `Hamiltonian`, `HamiltonianInfinite`, `BlockProposal` |
| `surrDAMH/modules/algorithm_interfaces*.py` | backend-neutral Protocols + local and MPI adapters |
| `surrDAMH/modules/communication.py` | low-level MPI classes (sampler↔collector, sampler↔pool) |
| `surrDAMH/modules/monitoring.py` | CSV writers (`SamplingOutputMonitor`) |
| `surrDAMH/modules/continuation.py` | save/load last sample per chain (`.npz`) |
| `surrDAMH/modules/lhs_normal.py`, `Gaussian_process.py`, `test_data.py`, `tools.py` | helpers |
| `surrDAMH/surrogates/` | `parent.py` (Updater/Evaluator contract), polynomial (sklearn), RBF (scipy), kd-tree, torch MLP (two variants), `reuse.py` |
| `surrDAMH/distributions/` | `Distribution` base, `Normal`, `PriorIndependentComponents` (+ Uniform/Lognormal/Beta/Normal components), `GaussianMixture`, transformations |
| `surrDAMH/solvers.py`, `solver_specification.py` | `Solver` base class, `SolverSpec` (module path + class + kwargs), loader |
| `surrDAMH/post_processing.py` | `Samples`, `StageSamples`, `Autocorrelation`, HTML report (2450 lines) |
| `toy_examples/` | runnable examples (see `05_toy_examples_and_docs.md`) |
| `tests/` | one pytest file |

Detailed notes per subsystem: `01_mcmc_algorithms.md`, `02_mpi_processes_and_communication.md`,
`03_surrogates_and_distributions.md`, `04_post_processing.md`, `05_toy_examples_and_docs.md`.
Cross-cutting prioritized findings: `06_findings_consolidated.md`. Test plan: `07_testing_plan.md`.

## 3. Process topology

Derived in `Configuration.__post_init__` (`configuration.py:54-74`) from `size = COMM_WORLD.Get_size()`:

| `use_collector` | `use_solvers_pool` | samplers | solvers pool rank | collector rank |
|---|---|---|---|---|
| True | True | ranks 0 … size−3 | size−2 | size−1 |
| True | False | 0 … size−2 | – (solver local in each sampler) | size−1 |
| False | True | 0 … size−2 | size−1 | – |
| False | False | 0 … size−1 | – | – |

`SamplingFramework.run()` (`core.py:111-149`) dispatches on `rank_world`. Each role calls
`COMM_WORLD.Split` with colour 0/1/2 (samplers/solver pool/collector) — a collective that
every rank must hit. The solver-pool rank spawns `no_solvers` children with
`MPI.COMM_SELF.Spawn(sys.executable, [process_CHILD.py, id, out_dir], maxprocs=solver_maxprocs)`
(`process_SOLVER.py:26-28`) and broadcasts `[conf, solver_spec]` to each.

Minimum process count with both pool and collector is therefore 3 (1 sampler); the
assertion message suggests `mpirun -n 4`.

## 4. Sampler-side control flow (`process_SAMPLER.run_SAMPLER`)

1. Build observation provider: `MpiSolverPoolObservationProvider` (pool) or the local
   `solver_instance`.
2. If collector: `MpiEvaluatorProvider` (requests the first evaluator immediately) and
   `MpiSnapshotSink`. Otherwise: `LocalEvaluatorProvider(surrogate_evaluator)` or `None`.
3. Initial sample: `"lhs"` (shared seed 0, row = rank), `"prior"` (`prior.rvs()`, global
   NumPy RNG), `"user_specified"`, `"continued"` (loaded in `Configuration`).
4. For each `Stage i`: `seed0 = 10*(no_stages*rank + i)`; proposal seed `seed0+1`, algorithm
   RNG seed `seed0+2`. Proposal chosen by `stage.proposal_type` / `stage.adaptive`; the
   `Stage.proposal` field itself is never read. Algorithm: `Algorithm_MH` or `Algorithm_DAMH`.
   Stage names are set here: `alg%04d_MH`, `alg%04d_MH-adaptive`, `alg%04d_DAMH`, `alg%04d_DAMH-SMU`.
5. After the stage: adaptive proposals `Allreduce` their covariance and average it (used by
   later stages that leave `proposal_sd_or_cov=None`); the last sample is saved
   (`continuation.save_last_sample`); if no later stage needs the surrogate, the evaluator
   and snapshot channels to the collector are closed; `comm_sampler.Barrier()`.
6. At the end: terminate the solver channel, two `COMM_WORLD.Barrier()` (matched by the
   other roles + one more in `core.run`).

## 5. Algorithms (`modules/algorithms.py`)

`Sample` holds `parameters`, `observations` (exact), `observations_approx` (surrogate),
`log_likelihood`, `log_prior`, `log_likelihood_approx`, `solver_tag`.

**MH** (`Algorithm_MH.run`): `min(max_samples, max_evaluations)` iterations; propose,
evaluate exactly, `log α = likelihood_part + prior_part` from
`proposal.get_log_acceptance_probability`, accept/reject, `adapt()`.

**DAMH** (`Algorithm_DAMH.run`): each outer iteration runs a surrogate-only subchain of
`subchain_max_length` MH steps starting from the current state, accumulating
`correction_log_ratio = Σ accepted likelihood_parts` (the surrogate log-likelihood ratio,
telescoping to `log L*(y) − log L*(x)` if the surrogate is unchanged). If ≥1 subchain step
was accepted, the endpoint `y` is evaluated exactly and accepted with

    log α = [log L(y) − log L(x)] − correction_log_ratio            (algorithms.py:461-462)

otherwise the iteration counts as "pre-rejected". Before each subchain step, if
`surrogate_model_updates` and a new evaluator is available, it is swapped in and the
surrogate values of the current state are recomputed (`_evaluate_surrogate_transition`).
`state_dependent_approximation` shifts surrogate outputs by `G(x) − G*(x)` at the current state.

**Weights / output**: on acceptance the *current* state is written with `multiplicity`
`1 + counter_rejected_current` (the CSV column is called `multiplicity` since format v2) and
sent to the collector; rejected proposals
are sent to the collector with weight 0 (training data only). At the end the final current
state is written once more. Terminal counters go to `notes/`. Since A30 (2026-09-17) the
**first** written row of a stage whose initial state was carried over from the previous stage
drops the leading `+1` (weight `= counter_rejected_current`, possibly 0), so a boundary state
is counted exactly once across concatenated stages; the collector-side weight is unchanged.
See `docs/outputs.md`.

**Gradients**: `Hamiltonian*` proposals need `∇(−log L)` and `∇(−log prior)`; the likelihood
gradient comes from the *surrogate* (`Evaluator.vjp` or `jacobian`), the acceptance uses the
exact likelihood — a valid MH proposal as long as the integrator is reversible and volume
preserving.

## 6. Collector (`process_COLLECTOR.run_COLLECTOR`)

Busy loop while any sampler still `needs_evaluator`: drain available snapshots from every
sampler (cap `max_collected_snapshots_per_loop`), out-of-sample quality check of the new
batch with the current evaluator (→ `surrogate_quality.csv`), `updater.add_data`, retrain if
`min_snapshots_initial` (first) / `min_snapshots_to_update` (later) reached, evaluate on the
fixed test set (→ `surrogate_quality_test.csv`), ship the evaluator (pickled object) to any
sampler that has requested one, handle stop signals (only once a surrogate exists).

## 7. Solver pool (`process_SOLVER.run_SOLVER`) and child (`process_CHILD.py`)

Busy loop while any sampler active: `Iprobe` each sampler (random order) for a parameters
message (tag = request counter; tag 0 = sampler finished), queue it, hand queued requests to
free children (`Bcast` tag + parameters), poll children for results and forward them to the
requesting sampler. Child: loop `Bcast` tag → if 0 exit, else `Bcast` parameters, call
`solver.set_parameters/get_observations`, rank 0 sends `[obs, solver_tag]` (pickled) or raw
buffer with `tag=solver_tag`. Failed solves are signalled by `solver_tag < 0`
(`solver_returns_tag=True`), in which case the child sends zeros.

## 8. MPI tags (`modules/communication.py:15-21`)

| Tag | Direction | Meaning |
|---|---|---|
| 0 | sampler→collector, sampler→pool | terminate / no more messages |
| 3 (`TAG_UPDATE`) | sampler→collector | "ready for a new evaluator" (payload: request index) |
| 4 (`TAG_EVALUATOR_OBJECT`) | collector→sampler | pickled `Evaluator` or `None` |
| 5 (`TAG_STOP_UPDATING`) | sampler→collector | sampler will not need evaluators any more |
| 10, 11, … | sampler→collector | snapshot k (pickled `[parameters, observations, weight]`) |
| 1, 2, … | sampler→pool | k-th parameter vector (raw doubles) |
| request tag or `solver_tag` | pool→sampler | observations (pickled with request tag, or raw with `tag=solver_tag`) |

## 9. Surrogate contract (`surrogates/parent.py`)

`Updater`: `delayed_init`, `add_data(params, obs, weights)`, `train()`, `get_evaluator()`,
`supports_gradients()`, `set_use_gradients()`, `get_initial_snapshots()`, optional
persistence (`save/load_training_data`, `save/load_state`).
`Evaluator`: `__call__(X: (n, p)) -> (n, m)`, optional `jacobian`, `vjp`, `as_solver()`.
The evaluator object is pickled and sent through MPI each time it changes.

## 10. On-disk layout of one run (`conf.output_dir`)

```
sampling_output/
  samples/<stage.name>/rank%04d.csv        multiplicity, par_0..par_{p-1}, log_posterior   (header row; v2)
  raw_data/<stage.name>/rank%04d.csv       state_type, par_0.., solver_tag, obs_0..obs_{m-1}, obs_approx_0..obs_approx_{m-1}, log_likelihood, log_prior  (header row; v2; if save_snapshots_to_file)
  notes/<stage.name>/rank%04d.csv          accepted, rejected, pre-rejected, sum, seed
  subchain_stats/<stage.name>/rank%04d.csv (DAMH only) per-iteration subchain statistics
  last_sample/<stage.name>/rank%04d.npz    parameters (float32), no_parameters
  surrogate_quality.csv, surrogate_quality_test.csv   (collector)
solver_output/rank<k>/                     per-solver scratch dir
post_processing_output/summary.csv, report_extended.html, best_fit_solver_visualization_*.png
  run_manifest.json                        format_version = 2 (required to read the directory)
```

**Output format v2** since 2026-09-17 (WS9a): every CSV has a header, `raw_data` is
rectangular with separate exact (`obs_*`) and surrogate (`obs_approx_*`) blocks, and
`surrDAMH.read_run(output_dir)` is the reader. Pre-v2 directories are refused
(`RunFormatError`, no converter — decision 6). Full reference: `docs/outputs.md`.

## 11. Environment observed in this container (read from `site-packages`, not executed)

Active interpreter: `/dolfinx-env` (Python 3.12) with numpy 2.2.6, scipy 1.16.3,
pandas 3.0.3, matplotlib 3.10.7, mpi4py 4.1.1, torch 2.11.0+cu130, scikit-learn 1.8.0,
emcee 3.1.6; surrDAMH 0.2.0 installed editable. A second `.venv/` exists in the repo without
torch/mpi4py. pandas 3 (copy-on-write, string dtype by default) and mpi4py 4 are recent
major versions — see the test plan for what to check.
