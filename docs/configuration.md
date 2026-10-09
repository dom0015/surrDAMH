# `Configuration` reference

`surrDAMH.Configuration` (`surrDAMH/configuration.py`), one instance constructed
identically on every MPI rank — and checked: `Problem.run_sampling()` broadcasts rank 0's
requested values of the fields in `POSTERIOR_AFFECTING_FIELDS` and every rank asserts
equality, so a script that builds a different `Configuration` per rank aborts the job at
start-up instead of sampling a different posterior per chain (finding 2.9;
`modules/communication.py::check_configuration_consistency`). Fields that are not
posterior-affecting are not compared. `__post_init__` derives the MPI
role layout (`no_samplers`, `rank_collector`, `rank_solvers_pool`) from
`MPI.COMM_WORLD`'s size and `use_collector`/`use_solvers_pool` — see
[`running.md`](running.md) for the resulting process-count table.

## Problem and runs

`output_dir` is `Configuration`'s first field. `no_parameters`/`no_observations` are no
longer required here: leave them `None` (the default) and `surrDAMH.Problem` resolves them
from the solver/prior/likelihood instead (`Problem.describe()` prints the sizes and which
source supplied each; see "Sizes" below). Give them explicitly only to pin a value or to
catch a mismatch early — `Problem.run_sampling`/`run_sampling_local` call
`Configuration.resolve_problem_sizes(...)` first thing, which raises `ValueError` if an
explicit `Configuration` value disagrees with what `Problem` resolved.

## Sizes: precedence and checks

`no_parameters`, first source that supplies a value wins, and every source that is present
must agree (else `ValueError` naming each source and its value):

1. explicit `Problem(no_parameters=...)`;
2. the solver instance's declared `no_parameters` (only if a `Solver` instance, not a
   `SolverSpec`, was passed — a spec is not imported until the run starts);
3. the prior's dimension (`PriorIndependentComponents.no_parameters`, `Normal`/
   `StandardizedNormal.n`, `GaussianMixture.means.shape[1]`, `FromScipy`'s wrapped
   `scipy_rv.dim`, or `dist.no_parameters`/`dist.dim` for anything else);
4. otherwise `ValueError` naming the fix (`dim=` on the prior, a `Solver` instance, or
   `no_parameters=`).

`no_observations` follows the same order with the solver instance's `no_observations` and
the likelihood's dimension in place of the prior's.

A `Normal(mean=<scalar>, sd=...)` with no `dim=` is **dimension-free**: used alone it is
1-D, but if `Problem` resolves a larger dimension from another source (typically a `Solver`
instance) it broadcasts the prior/likelihood to that size instead of raising. A `SolverSpec`
cannot supply this broadcast target (point 2 above), so with a `SolverSpec` give the prior/
likelihood an explicit size instead — `dim=d`, or an array-valued `mean` whose length says
it (`Normal(mean=[5.0], sd=1.0)`), or `Problem(no_observations=1)`.

Two further checks guard the sizes at run time, not just at `Problem` construction: C2 (a
solver-pool child's declared sizes must match `conf`, checked once at start-up, naming the
`solver_id`) and C3 (every solver/evaluator return value must have shape
`(no_observations,)`, checked once per stage).

| Field | Type | Default | Meaning | Posterior-affecting? |
|---|---|---|---|---|
| `output_dir` | `str` | required | Root directory for `sampling_output/`, `solver_output/`, `post_processing_output/`. First field. | no (output location only) |
| `no_parameters` | `int\|None` | `None` | Dimension of the parameter space. `None` (recommended): resolved by `Problem` from the solver/prior — see "Sizes" above. Set explicitly only to pin or sanity-check a value. | affects shapes only |
| `no_observations` | `int\|None` | `None` | Dimension of the observation space. Same resolution rule as `no_parameters`, from the solver/likelihood. | affects shapes only |
| `use_solvers_pool` | `bool` | `True` | If `False`, each sampler runs the solver in-process instead of using spawned children. | no (topology only) |
| `no_solvers` | `int` | `2` | Number of child solver processes spawned by the pool. | no (throughput only) |
| `solver_maxprocs` | `int` | `1` | MPI processes per spawned solver. | no |
| `use_collector` | `bool` | `True` | If `False`, no surrogate is *trained* during the run — an `Updater` only ever runs on the collector rank. **See "DAMH without a collector" below.** | yes, for any DAMH/Hamiltonian stage |
| `save_snapshots_to_file` | `bool` | `False` | Write every proposal (accepted/rejected/prerejected) to `raw_data/*.csv`. | no (output only) |
| `transform_before_saving` | `bool` | `True` | If `False`, `samples/*.csv` stores internal-space samples instead of `prior.transform`-ed ones. | no (output only) |
| `transform_before_surrogate` | `bool` | `False` | If `False`, the surrogate is trained/evaluated on the internal (standardized, N(0, I)) parameters the chain works with; if `True`, on the physical parameters the solver receives. | **yes** |
| `initial_sample_type` | `"lhs"\|"prior"\|"user_specified"\|"continued"` | `"prior"` | How the first sample of each chain is drawn. | **yes** — all four are reproducible: since G4 (2026-09-17) `"prior"`/`"user_specified"` draw from a per-chain `np.random.default_rng(10*no_stages*rank_world + 3)` (`surrDAMH.modules.seeds`) instead of the unseeded global NumPy RNG. A *user-supplied* `initial_samples_distribution` is only reproducible if its `rvs()` accepts the `generator` argument |
| `initial_samples_distribution` | `Distribution\|None` | `None` | Source for `initial_sample_type="user_specified"`; drawn in the prior's INTERNAL (standardized) coordinates (2026-09-22). | yes, when used |
| `continued_from_dir` | `str\|None` | `None` | Source run directory for `initial_sample_type="continued"`. | yes, when used |
| `lhs_scale` | `float\|ndarray` | `1.0` | Spread of the LHS initial-sample design (`initial_sample_type="lhs"`), in the prior's INTERNAL (standardized) coordinates, i.e. prior standard deviations (2026-09-22). | yes, when used |
| `stage_index_offset` | `int` | `0` | Set by `SamplingRun.continue_sampling`; leave at the default. Global index of this run's first stage in a lineage of continued runs (stage count of every earlier run); used in the stage directory names (`alg%04d_...`) and in the seeds. | **yes** |
| `no_stages_lineage` | `int\|None` | `None` | Set by `SamplingRun.continue_sampling`; leave at the default. Total stage count of the lineage up to and including this run, used as `no_stages` in the seed formula (`surrDAMH.modules.seeds`); `None` = this run's own stage count. | **yes** |
| `lineage_generation` | `int` | `0` | Set by `SamplingRun.continue_sampling`; leave at the default. Generation of this run in its lineage (0 = not a continuation); shifts every seed by `1_000_000 * lineage_generation` (`surrDAMH.modules.seeds`). | **yes** |
| `min_snapshots_initial` | `int` | `1` | Snapshots needed before the first surrogate is trained. | **yes**, for DAMH-SMU (changes retrain timing → accept/reject sequence) |
| `min_snapshots_to_update` | `int` | `0` | Further snapshots needed before each retrain. `0` (default since 2026-09-23) = retrain whenever `Updater.needs_retraining(new)` says the surrogate would change: continuously for `NeuralNetworkUpdater`, only after new snapshots for the interpolating/least-squares updaters (they never refit identical data). | **yes**, for DAMH-SMU |
| `max_collected_snapshots_per_loop` | `int` | `1000` | Collector-side batching cap per poll loop. | **yes** — controls exactly when the surrogate is (re)trained (`surrDAMH.configuration.POSTERIOR_AFFECTING_FIELDS`), hence the DAMH-SMU accept/reject sequence |
| `max_sampler_isend_requests` | `int` | `100` | Size of the sampler→collector snapshot `isend` buffer. | no (performance) |
| `use_surrogate_gradients` | `bool` | `True` | Whether Hamiltonian-family proposals may use surrogate autograd. | **yes** — may be silently forced to `False` by `Problem.run_sampling` if the surrogate/settings are incompatible; the *effective* value is recorded in `run_manifest.json` |
| `paths_to_append` | `list[str]\|None` | `None` | Appended to `sys.path` in this process only. | **ineffective for spawned children** (finding M18): they unpickle `conf` without running `__post_init__`. No longer needed to locate the solver module — `SolverSpec` stores an absolute `solver_module_path` since WS5 — but a solver module whose own *imports* need these directories still fails in the child; use `PYTHONPATH` for those |
| `max_buffer_size` | `int` | `1 << 30` | Bytes pre-allocated for the collector→sampler evaluator `irecv` buffer. Lower it for a smaller memory footprint per sampler, or raise it for a large surrogate (e.g. a big NN); the collector checks the pickled evaluator's actual size against this every time it builds one and raises `RuntimeError` before sending if it would not fit (warns above half, 2026-09-18, finding 4.1). | no (performance/buffering) |
| `debug` | `bool` | `False` | Collector-side: print extra diagnostics. | no |
| `torch_threads` | `int\|None` | `1` | Torch intra-op CPU thread count, set on every rank that has torch loaded (samplers evaluating the NN surrogate; the collector when it trains on CPU — irrelevant on GPU). `None` leaves torch's own default (all visible cores per process), which oversubscribes the node once more than one rank evaluates/trains the NN. Author decision 2026-09-18; see [`running.md`](running.md#torch-cpu-threads). | no (performance only) |

Computed by `__post_init__`, not settable directly: `no_samplers`, `rank_collector`,
`rank_solvers_pool`, `sampler_ranks`, `continued_samples` (loaded eagerly when
`initial_sample_type="continued"`).

See [`running.md`](running.md#continuing-a-run) for `SamplingRun.continue_sampling`, which sets
`stage_index_offset`/`no_stages_lineage`/`lineage_generation` (and the initial-sample fields
above) for you — a script calls it instead of setting any of these six fields by hand.

### Fields set by `run_sampling_auto`/`run_sampling_local_auto`

See [`running.md`](running.md#automatic-mode-2026-10-08) for the automatic mode
(`surrDAMH.auto.plan_auto`, `Configuration._set_by_auto`). It may set, on the `conf` it is given:

| Field | Rule |
|---|---|
| `initial_sample_type` | set to `"lhs"` **only if** left at the default `"prior"`; `"lhs"`/`"user_specified"`/`"continued"` given by the user are kept unchanged |
| `use_surrogate_gradients` | set to `mode == "fast"`, whenever the plan uses a surrogate |
| `min_snapshots_initial` | set to a size-dependent value (`max(1, min(10·d, n0·C // 2))` for an evaluation budget, `max(1, 10·d)` for a time budget) **only if** still at `Configuration`'s own default `1`; a value the user set explicitly is always kept (a note is added to the plan instead) |
| `min_snapshots_to_update` | set to `max(20, 2·d)` **only if** still at the default `0`; otherwise kept, same rule as above |

The rule is the same for all three size-dependent fields: **the user's own value always wins** —
Auto only fills in a field the user left untouched, and the fields it actually changed are
recorded in the manifest's `"auto"` entry (`conf_settings`, see `docs/outputs.md`) alongside a
note for every field it left alone because the user had set it. `initial_sample_type` is the one
exception with its own rule above (not "default vs. non-default" in general, since `"prior"` is
itself the default it replaces).

### `Configuration.describe()` / `Stage.describe()`

Both dataclasses have a `describe()` returning a short multi-line string of their
**effective** settings — every field as `name=value`, with a trailing `*` on the ones
marked "yes" in the table above (`surrDAMH.configuration.POSTERIOR_AFFECTING_FIELDS` and
`surrDAMH.stages.POSTERIOR_AFFECTING_FIELDS`). `Problem.run_sampling()` prints the
configuration plus one block per stage once on rank 0 before dispatching to the roles,
and `run_sampling_local()` does the same; a three-stage run takes about 34 lines.

Values are shown *after* every silent correction, which is the point: `Stage.__post_init__`
resolving `surrogate_model_updates` (`None` -> the stage type's default; `True` refused on an
MH stage whose proposal needs no gradients), the
default `max_evaluations=10` inserted when no stopping condition was given, and the
`[effective] use_surrogate_gradients: requested=… , in effect=…` line that
`Configuration.describe(use_surrogate_gradients_requested=…)` adds when
`Problem.run_sampling` had to disable gradients for an incompatible surrogate. `Stage.describe()`
also appends `[note]` lines naming fields that do not apply to the stage as configured
(a proposal's `target_rate` etc. when the proposal is not adaptive — the ignored-setting class
of bug that motivated this, finding G1).

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
| DAMH or a Hamiltonian-family proposal | `use_collector=True` + `surrogate_updater=` passed to `run_sampling`/`run_sampling_local` | surrogate retrains on the collector; DAMH-SMU available |
| DAMH or a Hamiltonian-family proposal | `use_collector=False` + `surrogate_evaluator=` (a different `run_sampling`/`run_sampling_local` argument than `surrogate_updater`) | a fixed, pre-trained surrogate; **no retraining, no DAMH-SMU** |

A DAMH/Hamiltonian stage with `use_collector=False` and no `surrogate_evaluator=` hits a
bare `assert commEvaluator is not None` in `process_SAMPLER.py` with no diagnostic
message — a common trap when flipping `use_collector`/`use_solvers_pool` to `False` "for
a quick single-process run" while keeping DAMH stages. See `toy_examples/one_process_only.py`
(MH-only, both flags `False`) and `problem.run_sampling_local` (supports
`surrogate_updater=` for in-process DAMH-SMU) for the two ways to actually run without a
collector.
