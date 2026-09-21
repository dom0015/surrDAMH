# surrDAMH — architecture note

Short internal-maintainer reference: the parts of the architecture `docs/` does not cover
(protocol/tag detail, process-role call sequence, package map). For the user-facing contract
(fields, output format, how to run, how to write a solver/surrogate) see `docs/README.md`.
For findings/decisions/evidence see `06_findings_consolidated.md`, `09_improvement_plan.md`,
`10_manual_review_notes.md`.

Originally written 2026-09-12 from a read-only review of five subsystem notes (01–05); those
notes' ~100 findings were deduplicated into `06`, validated/fixed through the refactor
(commits `175059e`..`9efb54b`), and the notes themselves were superseded and deleted
2026-09-18 — their content is fully absorbed into `06`/`09`/`10`/`docs/`.

## What the library does

surrDAMH samples the posterior of a Bayesian inverse problem

    f(u | y) ∝ f_noise(y − G(u)) · f_prior(u)

with several Markov chains in parallel (one MPI rank per chain, or a single process via
`run_local`). The expensive forward model `G` is evaluated either locally in each sampler
process or by a shared pool of spawned solver processes. A shared surrogate model of `G` is
trained on the fly from all `(u, G(u))` snapshots and used in **delayed-acceptance
Metropolis-Hastings (DAMH)**, optionally with **surrogate model updates during sampling
(DAMH-SMU)**.

The chain state lives in an *internal* parameter space (for `PriorIndependentComponents` this
is N(0, I)); `prior.transform` maps to the *physical* parameters the solver receives. Prior
log-pdf, proposals and saved samples (unless `transform_before_saving=True`) are in the
internal space — see `docs/concepts.md`.

## Package layout

| Path | Role |
|---|---|
| `surrDAMH/core.py` | `SamplingFramework`: role dispatch in `run()`, HTML report in `write_report()`, fail-loud `_run_role` wrapper |
| `surrDAMH/configuration.py` | `Configuration` dataclass; derives MPI rank layout from `MPI.COMM_WORLD` size |
| `surrDAMH/stages.py` | `Stage` dataclass; `stage_name()` shared by all three run paths |
| `surrDAMH/runner_local.py` | `run_local()`, the MPI-free single-chain runner (WS2) |
| `surrDAMH/process_SAMPLER.py` / `SOLVER.py` / `CHILD.py` / `COLLECTOR.py` | the four MPI roles |
| `surrDAMH/modules/algorithms.py` | `Sample`, `AlgorithmBase`, `Algorithm_MH`, `Algorithm_DAMH` (backend-neutral, no `mpi4py` import) |
| `surrDAMH/modules/proposals.py`, `proposal_builder.py` | proposal kernels; shared `build_proposal()` used by both runners |
| `surrDAMH/modules/algorithm_interfaces*.py` | `ObservationProvider`/`SnapshotCollector`/`EvaluatorProvider` protocols + MPI and local adapters |
| `surrDAMH/modules/communication.py` | low-level MPI classes (sampler↔collector, sampler↔pool) |
| `surrDAMH/modules/seeds.py` | the per-(chain, stage, stream) seed formula, used everywhere a reproducible draw is needed |
| `surrDAMH/modules/manifest.py`, `run_data.py` | `run_manifest.json` writer; `read_run()`/`RunData` reader (format v2) |
| `surrDAMH/modules/monitoring.py`, `continuation.py`, `torch_threads.py`, `describe.py`, `surrogate_restart.py` | CSV writers; last-sample save/load; torch thread pinning; start-up config dump; collector-side restart |
| `surrDAMH/surrogates/` | `parent.py` (`Updater`/`Evaluator` contract); polynomial (sklearn), RBF (scipy), kd-tree, torch MLP (`torch_perceptron_minibatches.py`), `reuse.py` |
| `surrDAMH/distributions/` | `Distribution` base, `Normal`, `PriorIndependentComponents` (+ components), `GaussianMixture`, `transformations` |
| `surrDAMH/solvers.py`, `solver_specification.py` | `Solver` base class, `SolverSpec` (absolute module path + class + kwargs) |
| `surrDAMH/post_processing/` | package: `loading` (`Samples`/`StageSamples`), `statistics`, `plots`, `html_report`, facade `__init__` |
| `toy_examples/` | the only maintained examples — see `README.md`'s table and `docs/README.md` |
| `tests/` | `unit/`, `validation/`, `mpi/` — see `docs/running.md` |

## Process topology

Derived in `Configuration.__post_init__` from `size = COMM_WORLD.Get_size()`:

| `use_collector` | `use_solvers_pool` | samplers | solvers pool rank | collector rank |
|---|---|---|---|---|
| True | True | ranks 0 … size−3 | size−2 | size−1 |
| True | False | 0 … size−2 | – (solver local in each sampler) | size−1 |
| False | True | 0 … size−2 | size−1 | – |
| False | False | 0 … size−1 | – | – |

`SamplingFramework.run()` dispatches on `rank_world`; every role calls `COMM_WORLD.Split` with
its own colour, runs inside `_run_role` (any uncaught exception → traceback + `MPI.Abort`), and
the roles rendezvous at a fixed number of `COMM_WORLD.Barrier()` calls (checked, matched).
The solver pool spawns `no_solvers` children via `MPI.COMM_SELF.Spawn(...)`, one
intercommunicator each; `SolverSpec.solver_module_path` is resolved to an absolute path so it
reaches the spawned children regardless of their working directory.

## MPI tags (`modules/communication.py`)

| Tag | Direction | Meaning |
|---|---|---|
| 0 (`TAG_TERMINATE`) | sampler→collector, sampler→pool | terminate / no more messages |
| 3 (`TAG_UPDATE`) | sampler→collector | "ready for a new evaluator" (payload: request index) |
| 4 (`TAG_EVALUATOR_OBJECT`) | collector→sampler | pickled `Evaluator` or `None` |
| 5 (`TAG_STOP_UPDATING`) | sampler→collector | sampler will not need evaluators any more |
| 6 (`TAG_INITIAL_SURROGATE`) | collector→sampler, once at start-up | can an evaluator be provided without a new snapshot? (closes the DAMH/Hamiltonian-first-stage deadlock, finding 2.2) |
| 10, 11, … (`TAG_FIRST_SNAPSHOT`+) | sampler→collector | snapshot k (pickled `[parameters, observations, multiplicity]`) |
| 1, 2, … | sampler→pool | k-th parameter vector (raw doubles) |
| request tag | pool→sampler | pickled `[observations, solver_tag]` |

`TAG_READY_TO_RECEIVE=1`/`TAG_DATA=2` are defined but unused — kept, not deleted, per the
author's no-dead-code-deletion decision (`13_dead_code_report.md`).

## Algorithm and collector mechanics (one paragraph each — see `06`/`10` for the derivation and evidence)

**MH**: propose, evaluate exactly, accept/reject in log domain. **DAMH**: a surrogate-only
sub-chain of `subchain_max_length` steps is run first (the surrogate is frozen for its
duration, WS3); if any inner step was accepted, the endpoint is evaluated exactly and the
outer decision is `log α = [log L(y) − log L(x)] − [log L̃(y) − log L̃(x)]` — the correct
second-stage ratio for a π̃-reversible sub-chain kernel (validated V2/V3/V3b/V3c/V3d/B2/B3/B9,
`10_manual_review_notes.md` §5). **Collector**: busy loop draining snapshots from every
sampler, periodic retrain, ships the evaluator to samplers that requested one. **Solver pool**:
busy loop, `Iprobe`s samplers, `Bcast`s work to free spawned children.

## On-disk layout and output format v2

Fully documented in `docs/outputs.md`; not duplicated here. One-line summary: every CSV under
`sampling_output/` has a header; the sampler-side per-row count column is `multiplicity`
(1+rejections for an accepted state, 0 for a rejected proposal — never "weight"); `raw_data`
is rectangular (`obs_*` exact, `obs_approx_*` surrogate); `run_manifest.json` with
`format_version=2` is required to read a directory (`read_run()`; no converter for pre-v2
output, author decision).

## Environment

Editable install lives in `/dolfinx-env` (Python 3.12), not the repo's own `.venv/` (which
lacks mpi4py/torch/scipy). Run examples from `toy_examples/` — `SolverSpec.solver_module_path`
strings are relative to it for the non-FEniCSx examples. See `docs/running.md` for exact
commands and process counts, and `CLAUDE.md` for the project's working conventions.
