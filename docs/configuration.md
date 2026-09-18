# `Configuration` reference

`surrDAMH.Configuration` (`surrDAMH/configuration.py`), one instance constructed
identically on every MPI rank (not broadcast or cross-checked — an inconsistent copy
across ranks is not detected until it causes a mismatched collective call,
`library_notes/10_manual_review_notes.md` §2.8/2.9). `__post_init__` derives the MPI
role layout (`no_samplers`, `rank_collector`, `rank_solvers_pool`) from
`MPI.COMM_WORLD`'s size and `use_collector`/`use_solvers_pool` — see
[`running.md`](running.md) for the resulting process-count table.

| Field | Type | Default | Meaning | Posterior-affecting? |
|---|---|---|---|---|
| `no_parameters` | `int` | required | Dimension of the parameter space. | affects shapes only |
| `no_observations` | `int` | required | Dimension of the observation space. | affects shapes only |
| `output_dir` | `str` | required | Root directory for `sampling_output/`, `solver_output/`, `post_processing_output/`. | no (output location only) |
| `use_solvers_pool` | `bool` | `True` | If `False`, each sampler runs the solver in-process instead of using spawned children. | no (topology only) |
| `no_solvers` | `int` | `2` | Number of child solver processes spawned by the pool. | no (throughput only) |
| `solver_maxprocs` | `int` | `1` | MPI processes per spawned solver. | no |
| `solver_returns_tag` | `bool` | `False` | If `True`, `Solver.get_observations()` returns `(observations, tag)`; `tag < 0` marks a failed solve. | no (failure signalling) |
| `use_collector` | `bool` | `True` | If `False`, no surrogate is *trained* during the run — an `Updater` only ever runs on the collector rank. **See "DAMH without a collector" below.** | yes, for any DAMH/Hamiltonian stage |
| `save_snapshots_to_file` | `bool` | `False` | Write every proposal (accepted/rejected/prerejected) to `raw_data/*.csv`. | no (output only) |
| `transform_before_saving` | `bool` | `True` | If `False`, `samples/*.csv` stores internal-space samples instead of `prior.transform`-ed ones. | no (output only) |
| `transform_before_surrogate` | `bool` | `False` | If `False`, the surrogate is trained/evaluated on internal-space parameters; if `True`, on physical-space ones. | **yes** |
| `initial_sample_type` | `"lhs"\|"prior"\|"user_specified"\|"continued"` | `"prior"` | How the first sample of each chain is drawn. | **yes** — all four are reproducible: since G4 (2026-09-17) `"prior"`/`"user_specified"` draw from a per-chain `np.random.default_rng(10*no_stages*rank_world + 3)` (`surrDAMH.modules.seeds`) instead of the unseeded global NumPy RNG. A *user-supplied* `initial_samples_distribution` is only reproducible if its `rvs()` accepts the `generator` argument |
| `initial_samples_distribution` | `Distribution\|None` | `None` | Source for `initial_sample_type="user_specified"`. | yes, when used |
| `continued_from_dir` | `str\|None` | `None` | Source run directory for `initial_sample_type="continued"`. | yes, when used |
| `lhs_scale` | `float\|ndarray` | `1.0` | Spread of the LHS initial-sample design (`initial_sample_type="lhs"`). | yes, when used |
| `state_dependent_approximation` | `bool` | `False` | Shift the DAMH surrogate approximation by the model/surrogate error at the current state. | **unverified** for `Stage.subchain_max_length > 1` (finding 1.1); warns at construction if `True` |
| `min_snapshots_initial` | `int` | `1` | Snapshots needed before the first surrogate is trained. | **yes**, for DAMH-SMU (changes retrain timing → accept/reject sequence) |
| `min_snapshots_to_update` | `int` | `1` | Further snapshots needed before each retrain. | **yes**, for DAMH-SMU |
| `max_collected_snapshots_per_loop` | `int` | `1000` | Collector-side batching cap per poll loop. | no (performance) |
| `max_sampler_isend_requests` | `int` | `100` | Size of the sampler→collector snapshot `isend` buffer. | no (performance) |
| `use_surrogate_gradients` | `bool` | `True` | Whether Hamiltonian-family proposals may use surrogate autograd. | **yes** — may be silently forced to `False` by `SamplingFramework` if the surrogate/settings are incompatible; the *effective* value is recorded in `run_manifest.json` |
| `paths_to_append` | `list[str]\|None` | `None` | Appended to `sys.path` in this process only. | **ineffective for spawned children** (finding M18): they unpickle `conf` without running `__post_init__`. No longer needed to locate the solver module — `SolverSpec` stores an absolute `solver_module_path` since WS5 — but a solver module whose own *imports* need these directories still fails in the child; use `PYTHONPATH` for those |
| `max_buffer_size` | `int` | `1 << 30` | Bytes pre-allocated for the collector→sampler evaluator `irecv` buffer. | no (performance/buffering) |
| `debug` | `bool` | `False` | Collector-side: print extra diagnostics. | no |
| `torch_threads` | `int\|None` | `1` | Torch intra-op CPU thread count, set on every rank that has torch loaded (samplers evaluating the NN surrogate; the collector when it trains on CPU — irrelevant on GPU). `None` leaves torch's own default (all visible cores per process), which oversubscribes the node once more than one rank evaluates/trains the NN. Author decision 2026-09-18; see [`running.md`](running.md#torch-cpu-threads). | no (performance only) |

Computed by `__post_init__`, not settable directly: `no_samplers`, `rank_collector`,
`rank_solvers_pool`, `sampler_ranks`, `continued_samples` (loaded eagerly when
`initial_sample_type="continued"`).

### `Configuration.describe()` / `Stage.describe()`

Both dataclasses have a `describe()` returning a short multi-line string of their
**effective** settings — every field as `name=value`, with a trailing `*` on the ones
marked "yes" in the table above (`surrDAMH.configuration.POSTERIOR_AFFECTING_FIELDS` and
`surrDAMH.stages.POSTERIOR_AFFECTING_FIELDS`). `SamplingFramework.run()` prints the
configuration plus one block per stage once on rank 0 before dispatching to the roles,
and `run_local()` does the same; a three-stage run takes about 34 lines.

Values are shown *after* every silent correction, which is the point: `Stage.__post_init__`
forcing `surrogate_model_updates=False` in an MH stage or `adaptive=False` for pCN, the
default `max_evaluations=10` inserted when no stopping condition was given, and the
`[effective] use_surrogate_gradients: requested=… , in effect=…` line that
`Configuration.describe(use_surrogate_gradients_requested=…)` adds when
`SamplingFramework` had to disable gradients for an incompatible surrogate. `Stage.describe()`
also appends `[note]` lines naming fields that do not apply to the stage as configured
(`adaptive_target_rate` etc. when `adaptive=False` — the ignored-setting class of bug that
motivated this, finding G1).

`tests/unit/test_describe.py` asserts that *every* dataclass field appears in the output, so
a newly added field cannot stay invisible.

### Removed fields

| Field | Removed | What to do |
|---|---|---|
| `pickled_observations` | 2026-09-17 (decision 5 / WS8) | Delete the argument; there is no replacement. Observations now always travel pickled together with their solver tag (`[observations, solver_tag]`, MPI tag = request counter) from the spawned solver child to the pool and on to the sampler, so a negative `solver_tag` is never used as an MPI tag. Passing the field raises `TypeError: Configuration.__init__() got an unexpected keyword argument 'pickled_observations'`. |

## DAMH without a collector

`use_collector=False` does not just mean "no surrogate model" — it forecloses DAMH-SMU
entirely, because an `Updater` (surrogate training) only ever runs on the collector
rank. Valid combinations:

| Stages in the run | Valid `Configuration` | Notes |
|---|---|---|
| MH only | any `use_collector`/`use_solvers_pool` | no surrogate needed |
| DAMH or a Hamiltonian-family proposal | `use_collector=True` + `surrogate_updater=` passed to `SamplingFramework` | surrogate retrains on the collector; DAMH-SMU available |
| DAMH or a Hamiltonian-family proposal | `use_collector=False` + `surrogate_evaluator=` (a different `SamplingFramework` argument than `surrogate_updater`) | a fixed, pre-trained surrogate; **no retraining, no DAMH-SMU** |

A DAMH/Hamiltonian stage with `use_collector=False` and no `surrogate_evaluator=` hits a
bare `assert commEvaluator is not None` in `process_SAMPLER.py` with no diagnostic
message — a common trap when flipping `use_collector`/`use_solvers_pool` to `False` "for
a quick single-process run" while keeping DAMH stages. See `toy_examples/one_process_only.py`
(MH-only, both flags `False`) and `surrDAMH.runner_local.run_local` (supports
`updater=` for in-process DAMH-SMU) for the two ways to actually run without a collector.
