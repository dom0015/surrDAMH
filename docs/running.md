# Running

## `Problem` and runs

Build a `surrDAMH.Problem(prior, likelihood, solver)` once (`solver` is a `Solver` instance or
a `SolverSpec` — one argument, type-dispatched). `no_parameters`/`no_observations` are resolved
from whichever of `solver`/`prior`/`likelihood` supplies them (`problem.describe()` prints the
sizes and where each came from; see [`configuration.md`](configuration.md)). Then run it:

```python
problem = surrDAMH.Problem(prior, likelihood, solver)
run = problem.run_sampling(conf, stages, surrogate_updater=updater)   # MPI; call on every rank
# or, with no MPI roles at all:
run = problem.run_sampling_local(conf, stages, surrogate_updater=updater)
run.write_report(observations=...)
```

The prior is truncated to a box of `prior_bound=8.0` internal-prior scales around its centre by
default (proposals outside are rejected unevaluated, `Problem(..., prior_bound=None)` for the
unbounded prior; see [`concepts.md`](concepts.md#bounded-prior-prior_bound-2026-10-09)).

Both return a `SamplingRun` (`.problem`, `.conf`, `.stages`, `.output_dir`, `.write_report(...)`).
See `toy_examples/minimal_example.py` for the smallest `run_sampling` script.
`toy_examples/one_process_only.py` and `toy_examples/post_processing_with_html_report.py` also
call `run_sampling`, just under a single rank (`use_collector=False`, `use_solvers_pool=False`,
one process); see "Single process / no MPI at all" below for the no-MPI-at-all alternative,
`run_sampling_local`.

## Automatic mode (2026-10-08)

**Status (2026-10-09, evening): preliminary, two of three safeguards in place.** Every DAMH step targets the exact posterior for the surrogate installed at that moment, the prior is bounded by default (S0, `Problem(prior_bound=8.0)`), and every automatic DAMH chunk mixes in exact random-walk steps and audits pre-rejected proposals (S2, placeholder weights 0.05 / 0.05). What is still missing for the ergodicity argument of `library_notes/18_damh_smu_validity_proof_2026-09-22.md` is the diminishing adaptation of the on-the-fly network retraining (Polyak averaging of received weights, note 25 S3). Until then treat Auto as "exact per installed surrogate, with retraining whose theoretical justification is pending"; all its numbers are placeholders until the validation plan of note 25 §6 has run.

Instead of writing an explicit stage list by hand, `Problem.run_sampling_auto` /
`run_sampling_local_auto` pick a stage layout and a surrogate from a budget
(`surrDAMH/auto.py::plan_auto`; design in `library_notes/25_robust_by_default_roadmap_2026-10-08.md`
§5):

```python
run = problem.run_sampling_auto(conf, budget=80_000, mode="robust")        # MPI, collective call
run = problem.run_sampling_auto(conf, time_limit=3600.0, mode="fast")      # wall-clock budget instead
run = problem.run_sampling_local_auto(conf, budget=20_000, mode="robust")  # one chain, one process
```

Give exactly one of `budget=`/`time_limit=` (`ValueError` otherwise, before any MPI call).
`budget` is the TOTAL number of **exact** model evaluations over the whole run — every sampler
chain plus the held-out test set below, not per chain. `time_limit` is wall-clock seconds for the
whole run instead, used to size the stages' `Stage.time_limit` rather than
`Stage.max_evaluations`.

**Both modes sample the exact posterior** — the difference is only how fast the chain mixes on a
hard problem, never whether it is biased:

- `mode="robust"` (default): the DAMH chunks propose with an adaptive `RandomWalk()`.
- `mode="fast"`: the DAMH chunks propose with `Hamiltonian(num_steps=30,
  integrator="dimension_robust", mass=1.0)` on the surrogate's gradients — usually far more
  efficient, but it may mix slowly (or diverge more often) on a hard/non-smooth problem, where
  `"robust"` is the safer default. Needs a gradient-capable surrogate and
  `Configuration(use_surrogate_gradients=True, transform_before_surrogate=False)` (both the
  defaults) — `ValueError` otherwise, naming the fix.

**Every number below is a PLACEHOLDER** picked before validation (note 25 §6 will tune them); do
not rely on the exact constants staying as they are. Let `d` = `problem.no_parameters`, `C` = the
number of chains (`conf.no_samplers`, 1 for a local run), `B` = the per-chain evaluation budget
after subtracting the held-out set (below) and dividing by `C`:

- If `B < 50·d` evaluations per chain, or the run has no collector
  (`conf.use_collector=False`, so no surrogate can be trained): **one** adaptive-random-walk MH
  stage runs with the whole budget, with a note explaining why — a surrogate trained on fewer
  points than that does not pay.
- Otherwise: a `burn_in=True` MH warm-up stage (adaptive random walk) of
  `n0 = clip(20·d, 0.05·B, 0.25·B)` evaluations per chain — the chain **continues** into chunk 1
  from the warm-up's last state, only its samples are left out of the posterior by default
  (`docs/outputs.md`'s selection mask) — followed by **4 DAMH-SMU chunks** (`subchain_length=1`,
  the mode's proposal) sharing the remaining `B - n0` evaluations as evenly as possible (the
  remainder goes to the last chunk). There is **no frozen stage** — every chunk keeps retraining
  the surrogate. **S2 safety (2026-10-09)**: every chunk sets `exact_step_probability=0.05` (an exact
  random-walk MH step instead of the DA step 5 % of the time, so the chain can enter a region the
  surrogate wrongly rules out) and `audit_prerejected=0.05` (5 % of the pre-rejected proposals are
  evaluated exactly and sent to the collector, the chain unchanged); both spend evaluations inside
  the chunk budgets (the per-chain budget rule is unchanged), both are placeholders
  (`surrDAMH.auto.AUTO_EXACT_STEP_PROBABILITY`, `AUTO_AUDIT_PREREJECTED`; see `docs/stages.md`).

  **Fixed 2026-10-09**: before this date the warm-up used `is_excluded=True` instead, which made
  chunk 1 **restart** from the warm-up's starting (LHS/prior) point rather than continue from its
  last state — i.e. the warm-up's evaluations were wasted, the chain effectively began fresh. This
  was a defect, not an intended design choice; `Stage.burn_in` (`docs/stages.md`) was added so the
  warm-up could continue the chain while still defaulting out of the posterior.
- With `time_limit=` instead of `budget=`, the same shares apply to wall time: the warm-up gets
  `0.15 * time_limit` and the 4 chunks `0.85 * time_limit / 4` each (`Stage.time_limit=`, no
  evaluation cap); a note says the `20·d` warm-up floor cannot be enforced against a clock, so a
  too-short `time_limit` may leave the first surrogate undertrained.

**Surrogate automation**: unless `surrogate_updater=` overrides it, a surrogate-using plan gets a
`NeuralNetworkUpdater(hidden_layer_sizes=(64, 64), activation="silu", solver="adamw", seed=0)`
(`output_normalization="likelihood"` wired as usual). A **held-out test set** is generated only
when the plan uses a surrogate (not for the single-MH-stage case above), `problem.solver_instance`
is an in-process `Solver` (not a `SolverSpec`, where no rank but the solver-pool children hold a
solver — a note says so) and no `surrogate_test_data=` override is given; its size
is `min(256, max(2·d, 0.02·budget))` for an evaluation budget or `min(256, max(2·d, 64))` for a
time budget, **subtracted from `budget`** (an explicit `surrogate_test_data=` is used as given and
NOT subtracted). For an MPI run it is generated **on the collector rank only**, inside
`run_sampling`, so no sampler spends an evaluation on it; for `run_sampling_local_auto` it is
generated before the run. Either way it is saved to
`sampling_output/surrogate_test_data.npz`/`surrogate_quality_test.csv` as usual — except
`run_sampling_local_auto` does not monitor surrogate quality with it (the local runner has no
quality-monitoring loop), so there the set is only saved for a later run to reuse.
`use_surrogate_gradients` is set to `mode == "fast"`. `min_snapshots_initial` defaults to
`max(1, min(10·d, n0*C // 2))` for an evaluation budget (`max(1, 10*d)` for a time budget) and
`min_snapshots_to_update` to `max(20, 2*d)` — **but only if the user left them at `Configuration`'s
own defaults** (`1` and `0`); a value the user set explicitly is always kept, with a note in the
plan instead of being overwritten.

**What Auto sets on `conf`** (recorded under `conf_settings` below): `initial_sample_type="lhs"`,
but only if it was left at the default `"prior"` (a user's `"lhs"`/`"user_specified"`/`"continued"`
is kept); `use_surrogate_gradients`; `min_snapshots_initial`/`min_snapshots_to_update` as above.

**Expert overrides**: `surrogate_updater=` (an already-constructed `Updater`; `mode="fast"` then
needs one with `supports_gradients()`) and `surrogate_test_data=` (a `TestData` or the
`(parameters, observations)` tuple `run_sampling` already accepts — used as given, not
subtracted from the budget, not generated).

The resolved plan is printed once at start-up (rank 0 / local, `AutoPlan.describe()`), and the
explicit stage list it produced — `run.stages` — is what actually runs; an expert can read it off
and copy it into a plain `run_sampling(conf, stages, ...)` call. It is also recorded in
`run_manifest.json` under `"auto"` and as `run.auto` (keys: `mode`, `budget`, `time_limit`,
`no_samplers`, `per_chain_budget`, `test_data_size`, `warm_up`, `chunks`, `stage_names`,
`proposal`, `exact_step_probability`, `audit_prerejected`, `surrogate`, `conf_settings`, `notes` —
see `docs/outputs.md`).

**Not implemented yet**: `continue_sampling_auto` (an automatic continuation of a finished Auto
run) does not exist; `conf`'s lineage fields (`stage_index_offset`, `no_stages_lineage`,
`lineage_generation`) must stay at their defaults, or `plan_auto` raises `ValueError` (Auto is not
a continuation). See `toy_examples/auto_mode_example.py`.

## Process counts and roles

`Configuration.__post_init__` derives roles from `MPI.COMM_WORLD`'s size (`size`) and
the two topology flags:

| `use_collector` | `use_solvers_pool` | sampler ranks | solvers-pool rank | collector rank | minimum `size` |
|---|---|---|---|---|---|
| `True` | `True` | `0 … size-3` | `size-2` | `size-1` | 3 (1 sampler + pool + collector) |
| `True` | `False` | `0 … size-2` | – (solver runs locally per sampler) | `size-1` | 2 |
| `False` | `True` | `0 … size-2` | `size-1` | – (no surrogate) | 2 |
| `False` | `False` | `0 … size-1` | – | – | 1 |

Each sampler rank is one independent MCMC chain. The solvers-pool rank additionally
spawns `Configuration.no_solvers` child processes via `MPI.Comm.Spawn` — these are
*not* counted in `size` (they are extra OS processes, not extra `COMM_WORLD` ranks).
See `docs/configuration.md`'s "DAMH without a collector" table for which
`use_collector`/`use_solvers_pool` combinations support which stage types.

Observations always travel pickled together with their solver tag — the spawned child
sends `[observations, solver_tag]` to the pool and the pool forwards `[observations,
solver_tag]` to the requesting sampler, in both cases using the *request counter* as the
MPI tag. A negative `solver_tag` (failed solve, from a solver returning `(observations, tag)`) is therefore
just payload; the sampler rejects that proposal and does not hand it to the surrogate.
There is no alternative raw-buffer transport: `Configuration.pickled_observations` was
removed on 2026-09-17 (decision 5 / WS8).

## `mpiexec` vs `python -m mpi4py`

```bash
mpiexec -n 4 python3 -m mpi4py my_experiment.py
```

Prefer `-m mpi4py` over `-m mpi4py.run` / plain `python3`: it installs mpi4py's own
excepthook, so an uncaught exception on rank 0 aborts the whole job even without the
library's own guard. The library now also aborts the job itself, independent of how it
was launched: `Problem.run_sampling()` wraps every role body, and `process_CHILD.py`
wraps the spawned solver loop, so that ANY rank's uncaught exception (sampler,
collector, solvers pool, or a spawned solver child) prints a traceback and calls
`MPI.COMM_WORLD.Abort(1)` — no configuration is known to hang silently on an ordinary
Python exception any more (`KeyboardInterrupt`/`SystemExit` are re-raised instead, so
Ctrl-C behaves normally). See `library_notes/10_manual_review_notes.md` §2.10 for what
this replaced (a solver exception used to hang the job under plain `mpiexec python
driver.py`).

## Single process / no MPI at all

With `use_collector=False` and `use_solvers_pool=False`, an MH-only script still runs
under mpi4py with a single rank:

```bash
mpiexec -n 1 python3 -m mpi4py my_experiment.py
```

(`toy_examples/one_process_only.py` is exactly this.) DAMH stages need either a
collector rank to retrain the surrogate, or a pre-trained fixed `surrogate_evaluator=`
— neither of which a single sampler process without a collector can provide (see
`docs/configuration.md`).

For no MPI dependency at all, use
`problem.run_sampling_local(conf, stages, surrogate_updater=None, surrogate_evaluator=None)`:
one chain, in one Python process, reproducing rank 0 of an equivalent MPI run (same seed
formula). It supports `surrogate_updater=` for in-process DAMH-SMU (`LocalSurrogateManager`
trains the surrogate in-process instead of needing a collector), `surrogate_evaluator=` for a
fixed surrogate, and continuation via `Configuration.initial_sample_type="continued"`.
(`surrDAMH.runner_local.run_local` is the internal engine behind it — not part of the public
API — in case you need to read the implementation.) See `docs/writing_a_surrogate.md`.

## Continuation

This is the low-level mechanism; for continuing a finished `SamplingRun` with lineage stage
numbering, generation-shifted seeds, carried proposal state and automatic surrogate/test-data
reuse, use `SamplingRun.continue_sampling(_local)` instead — see "Continuing a run" below.

Every stage (MPI or local) writes `sampling_output/last_sample/<stage>/rank%04d.npz`
after it finishes. To continue from it:

```python
conf = surrDAMH.Configuration(..., initial_sample_type="continued",
                               continued_from_dir="out_previous_run")
```

`load_last_samples` reads the *last* stage directory of the source run by default,
requires at least as many saved chains as the new run's `no_samplers`, and warns (does
not error) if there are more than needed (only the first `no_samplers`, sorted by
filename, are used).

## Continuing a run

`SamplingRun.continue_sampling(_local)` continues a finished run (the `SamplingRun` returned by
`problem.run_sampling(_local)`, or rebuilt by `SamplingRun.load(output_dir, problem)`) under a
fresh `Configuration`/stage list, carrying over everything a hand-written restart would otherwise
have to wire up itself: chain states, stage numbering and seeds, the tuned proposal, and the
surrogate with its held-out test data.

```python
previous = problem.run_sampling(conf_a, stages_a, surrogate_updater=updater)   # or run_sampling_local,
                                                                               # or SamplingRun.load(dir, problem)
run_b = previous.continue_sampling(conf_b, stages_b,
                                   chains="continue",                # "continue" | "prior" | "lhs"
                                   problem=None,                     # a different Problem, same sizes
                                   surrogate_updater=None, surrogate_evaluator=None,
                                   surrogate_initial_training_data=None, surrogate_test_data=None,
                                   surrogate_restart=None)            # MPI, collective like run_sampling
run_b = previous.continue_sampling_local(conf_b, stages_b, chains="continue", problem=None,
                                         surrogate_updater=None, surrogate_evaluator=None)
```

`conf_b` is a fresh `Configuration` with its own `output_dir` (must differ from every output
directory already in the lineage, else `ValueError`). Its initial-sample fields
(`initial_sample_type`, `continued_from_dir`) and the three lineage fields
(`stage_index_offset`, `no_stages_lineage`, `lineage_generation`, see `docs/configuration.md`) are
set by `continue_sampling` itself — leave them at their defaults, or it raises `ValueError`.

**What is reused by default** (true for every `chains=` choice unless stated otherwise):

- **Chain states** — only with `chains="continue"` (the default): every chain of `run_b` starts
  from `previous`'s `sampling_output/last_sample/` (its last stage's state). `previous` must have
  at least as many saved chains as `conf_b.no_samplers` (the existing `load_last_samples` rule:
  fewer is an error, more drops the extra ones with a warning); no `last_sample/` at all raises
  `FileNotFoundError`.
- **Carried proposal state** — always, regardless of `chains=`: the proposal carry-over of every
  adaptive stage of the whole lineage so far (oldest first, later stages win) is merged and handed
  to `stages_b`'s first stage, so `RandomWalk(scale=None)` / `PCN(beta=None)` /
  `Hamiltonian(step_size=None)` pick up the tuned value instead of the prior-derived default. This
  is about the posterior, not the chain state, so it applies even when `chains="prior"`/`"lhs"`
  draws fresh starting points.
- **Surrogate, via `surrogates.reuse.SurrogateReused`** — if neither `surrogate_updater=` nor
  `surrogate_evaluator=` is given: a checkpoint saved in `previous`'s `sampling_output/`
  (`surrogate_checkpoint.pt` + `surrogate_training_data.npz`) is restored, so a DAMH/Hamiltonian
  first stage in `stages_b` needs no MH warm-up. With `conf_b.use_collector=False` the restored
  surrogate becomes a fixed evaluator instead. Only **`NeuralNetworkUpdater`** and
  **`PolynomialSklearnUpdater`** implement state persistence (`Updater.supports_state_persistence()`
  — `RBFInterpolationUpdater` and `KDTreeUpdater` do not): a continuation of one of those starts
  with **no surrogate at all**, so a DAMH/Hamiltonian first stage then fails through the usual
  start-up deadlock check (finding 2.2) instead of silently starting cold. If `previous` holds
  training data but no checkpoint, `continue_sampling` raises `ValueError` naming the fix
  (`surrogate_updater=<an updater>` together with
  `surrogate_restart=SurrogateRestart(..., mode="data")`).
- **Held-out test data** — `surrDAMH.TestData.reuse(previous.output_dir)` if a set was saved
  there and `surrogate_test_data=` is not given; the set used is saved into `run_b` as well, so the
  lineage stays self-contained.

**A plain run's surrogate state is NOT saved automatically** — only a run produced by
`continue_sampling(_local)` saves it at the end (`core._save_surrogate_state_of`, gated by the
private `_save_surrogate_state` flag that only `_prepare_continuation` sets). To make a *first*
run's surrogate state reusable by a later continuation, call, right after
`problem.run_sampling(...)` returns (on the collector rank of an MPI run):

```python
surrDAMH.SurrogateRestart(state_dir=os.path.join(conf_a.output_dir, "sampling_output")).save(updater)
```

See `toy_examples/continue_sampling_example.py` for this in context.

**`chains=`**:

| `chains` | Initial state of each chain of `run_b` |
|---|---|
| `"continue"` (default) | `previous`'s last state |
| `"prior"` | drawn from the prior, with the generation-shifted seeds below (not `previous`'s starting points) |
| `"lhs"` | a fresh Latin-hypercube design, same shift |

**`problem=`**: a different `Problem` for `run_b` — typically new observed data with the same
forward model. Its `no_parameters`/`no_observations` must equal `previous`'s (`ValueError` naming
both sets of sizes otherwise); the manifest then records `"same_problem": false`. Omit it to keep
`previous`'s `Problem` (`"same_problem": true`).

**Stage numbering**: `stages_b` is numbered after every earlier stage of the lineage, so e.g. a
two-stage `previous` continued by a one-stage `run_b` gets stage directory `alg0002_...`
(`stages.stage_name`), not `alg0000_...` again — stage names stay unique across the whole lineage,
which is what lets `post_processing.read_lineage` concatenate them unambiguously.

**Seeds**: every seed of `run_b` additionally shifts by `1_000_000 * generation`
(`modules/seeds.py`, `GENERATION_SEED_STRIDE`), where `generation` is `previous.generation + 1` (0
for a plain run) — so no random stream of an earlier run of the lineage is ever reused, including
the initial-sample draw that `chains="continue"` otherwise does not need. This requires
`10 * no_stages_lineage * no_samplers < 1_000_000` for the whole lineage (`SEED_STRIDE *
no_stages_lineage * no_samplers < GENERATION_SEED_STRIDE`) — i.e. the lineage's total stage count
times its sampler count must stay under 100,000 — or two generations' seed ranges would overlap
(not checked at run time).

`SamplingRun.load(output_dir, problem)` rebuilds a `SamplingRun` for a new Python session from
`<output_dir>/sampling_output/run_manifest.json`, including its `.lineage` (output directories,
oldest first) and `.generation`; the result supports both `continue_sampling(_local)` and
`write_report()`. `problem=` is optional for `write_report()` (sections that need the prior or a
solver are then marked "Not available") but required for `continue_sampling(_local)` if the
original `Problem` object is gone — `ValueError` otherwise, naming the fix.

**Error cases** (raised identically on every rank, before any MPI call): `chains` not one of
`"continue"`/`"prior"`/`"lhs"`; `conf_b` sets `initial_sample_type`/`continued_from_dir` or any of
`stage_index_offset`/`no_stages_lineage`/`lineage_generation` itself; `conf_b.output_dir` equal to
a directory already in the lineage; `problem=`'s sizes disagree with `previous`'s; previous run has
no manifest (`FileNotFoundError`); `chains="continue"` with no `last_sample/` in `previous`
(`FileNotFoundError`); surrogate training data without a checkpoint in `previous` (see above).

`continue_sampling_local` follows the same defaults/rules for one chain in this process
(`Problem.run_sampling_local`): a restored surrogate is trained further in process via the
`surrogate_updater` argument; there is no held-out test data and no `surrogate_restart=` keyword
here (load training data into an updater yourself with `updater.load_training_data(path)` first).

See `docs/outputs.md` for the manifest's `"lineage"` entry, `read_lineage`, and the report's
selection mask (`post_processing_output/selection.json`). Since 2026-10-09 every chain of every
stage is always shown in `report_extended.html` (with an "in posterior"/"not in posterior" badge);
to change which chains are pooled into the posterior sections instead, edit `posterior` (and,
rarely, `drop_first_rows`) in `selection.json` and re-run
`SamplingRun.load(output_dir).write_report()` — see `docs/outputs.md`'s
"Selection mask" section.

## Surrogate restart

A run can start from the surrogate a previous run ended with, instead of learning it again
from scratch. `Problem.run_sampling` takes a `surrogate_restart=` argument
(`surrDAMH.SurrogateRestart`, `surrDAMH/modules/surrogate_restart.py`); it is applied on the
**collector rank only**, immediately before the collector loop starts, since that is the
only rank that owns an `Updater`.

```python
restart = surrDAMH.SurrogateRestart(state_dir="out_previous_run/sampling_output", mode="state")
run = problem.run_sampling(conf, stages, surrogate_updater=updater, surrogate_restart=restart)
if rank_world == conf.rank_collector:
    restart.save(updater)     # write this run's state back, for the next restart
```

`state_dir` holds the two files `surrogate_checkpoint.pt` and `surrogate_training_data.npz`
(the layout `surrDAMH.surrogates.reuse.surrogate_state_paths` produces, i.e.
`<experiment_dir>/sampling_output`). `mode`:

| `mode` | What is restored | Effect at start-up |
|---|---|---|
| `"state"` (default) | `Updater.load_state`: network weights, optimizer state and the stored snapshots | the updater is `pretrained_ready`, so the collector's start-up handshake reports "evaluator available" and a **DAMH or Hamiltonian first stage** is possible without any full-model evaluation first |
| `"data"` | `Updater.load_training_data` plus one `train()` call | the old snapshots are reused, the network is refitted from scratch |
| `"none"` | nothing | keeps the argument in a script without deleting it |

Missing files are not an error: a message is printed and the run starts cold. An
*incompatible* checkpoint is an error (`ValueError` from the updater), as is an updater
without state persistence (`NotImplementedError`) — today only
`NeuralNetworkUpdater` implements it. The restored snapshots are handed to the
collector as `surrogate_initial_training_data` unless the updater reports them itself via
`get_initial_snapshots()`; either way they are counted once (finding 2.8).

Related but different: `surrDAMH.surrogates.reuse.SurrogateReused(experiment_folder)`
*constructs* an updater from a checkpoint (hyper-parameters taken from the checkpoint),
whereas `SurrogateRestart` restores state into an updater the script has already configured.

`toy_examples/toy_example_hamilton.py` and `toy_examples/sampling_diffusion_grf.py` both use
it (`SURROGATE_RESTART_MODE` at the top of each file).

## Torch CPU threads

`Configuration.torch_threads` (default `1`, author decision 2026-09-18) sets
`torch.set_num_threads()` deterministically on every rank that has torch loaded —
samplers evaluating the NN surrogate, and the collector when it trains on CPU (irrelevant
on GPU). Without it, torch's own default is *all visible cores per process*, which
oversubscribes the node as soon as more than one rank evaluates/trains the network
concurrently. Measured under this container's MPICH `mpiexec`, every rank already starts
at 1 thread regardless (a launcher artefact — OpenMP/MKL throttled by the launch
environment; a bare `python` process gets one thread per visible core instead), so
`torch_threads=1` is a no-op there and only becomes load-bearing under a launcher that
does not constrain OpenMP itself (Slurm `srun`, OpenMPI, `python -m mpi4py` without
`mpiexec`). Set `torch_threads=None` to leave torch's default alone (e.g. to let a
collector-only process use all cores for training); set it to a larger integer to give the
collector more threads while still capping the samplers by using a value larger than 1 (the
same value applies on every rank — there is currently no per-role override). The *effective*
count is recorded in `run_manifest.json` as `environment.torch_num_threads`; the
*requested* `Configuration` value is `configuration.torch_threads`. Performance-only: does
not affect the posterior, acceptance rate or surrogate accuracy. See
`surrDAMH/modules/torch_threads.py` for the two mechanisms used (direct
`torch.set_num_threads` if torch is already imported — always the case in this codebase,
since `import surrDAMH` itself imports torch — vs. `OMP_NUM_THREADS`/`MKL_NUM_THREADS`
env vars as a fallback for a torch import that has not happened yet).

## Start-up log

`Problem.run_sampling()` and `run_sampling_local()` print `Configuration.describe()` and one
`Stage.describe(i)` block per stage on rank 0 before anything else happens — the effective
settings, after every silent correction. See
[`configuration.md`](configuration.md#configurationdescribe--stagedescribe).

## `./run_tests.sh`

```bash
./run_tests.sh             # unit tests only (~10 s), no MPI beyond size-1 COMM_WORLD
./run_tests.sh validation  # statistical validation on the Gaussian toy (~2 min)
./run_tests.sh mpi         # mpiexec-driven integration/deadlock tests (~1.5 min)
./run_tests.sh all         # everything (~5 min)
```

`PYTHON` overrides the interpreter (defaults to `/dolfinx-env/bin/python3` if present).
`pytest.ini` markers: `unit`, `validation`, `mpi`.

## Surrogate stage before enough snapshots (2026-09-22)

A DAMH or Hamiltonian-proposal stage needs a surrogate. If it starts before the collector has
`Configuration.min_snapshots_initial` snapshots (short first stage, large threshold), the collector
used to wait forever. Now every sampler tells the collector when it blocks for its first
evaluator (`TAG_EVALUATOR_NEEDED`); once all of them have, the collector trains the initial
surrogate on the snapshots it has and prints
`collector: every sampler is waiting for the first surrogate but only N snapshots exist ...`.
Treat that line as a configuration warning: lower `min_snapshots_initial` or lengthen the
preceding stage. With no snapshot at all the run stops with a `RuntimeError` that names the fix.
`run_sampling_local` behaves the same way.
