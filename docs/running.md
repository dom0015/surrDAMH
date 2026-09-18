# Running

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
MPI tag. A negative `solver_tag` (failed solve, `solver_returns_tag=True`) is therefore
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
was launched: `SamplingFramework.run()` wraps every role body, and `process_CHILD.py`
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

For no MPI dependency at all, use `surrDAMH.runner_local.run_local(conf, prior,
likelihood, stages, solver, updater=None, evaluator=None)`: one chain, in one Python
process, reproducing rank 0 of an equivalent MPI run (same seed formula). It supports
`updater=` for in-process DAMH-SMU (`LocalSurrogateManager` trains the surrogate
in-process instead of needing a collector), `evaluator=` for a fixed surrogate, and
continuation via `Configuration.initial_sample_type="continued"`. See
`toy_examples/` for `run_local` usage and `docs/writing_a_surrogate.md`.

## Continuation

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

## Surrogate restart

A run can start from the surrogate a previous run ended with, instead of learning it again
from scratch. `SamplingFramework` takes a `surrogate_restart=` argument
(`surrDAMH.SurrogateRestart`, `surrDAMH/modules/surrogate_restart.py`); it is applied on the
**collector rank only**, immediately before the collector loop starts, since that is the
only rank that owns an `Updater`.

```python
restart = surrDAMH.SurrogateRestart(state_dir="out_previous_run/sampling_output", mode="state")
sam = surrDAMH.SamplingFramework(conf, ..., surrogate_updater=updater, surrogate_restart=restart)
sam.run()
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
`NeuralNetworkUpdaterMinibatches` implements it. The restored snapshots are handed to the
collector as `initial_snapshots` unless the updater reports them itself via
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

`SamplingFramework.run()` and `run_local()` print `Configuration.describe()` and one
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
