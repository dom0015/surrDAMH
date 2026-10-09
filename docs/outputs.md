# Outputs

Everything is written under `Configuration.output_dir`. This describes **format v2**
(`manifest.FORMAT_VERSION == 2`), the current on-disk layout, verified against the writers.

Format v2 is a **breaking** change and there is **no converter** (author decision 6,
2026-09-17): every CSV now carries a header row, `raw_data` is rectangular with separate
exact/surrogate observation blocks, the multiplicity column is called `multiplicity`, and
`read_run`/`Samples` **refuse** to open a directory whose `run_manifest.json` is missing or
declares another `format_version`. Re-run the sampling to get v2 output.

## Layout

```
sampling_output/
  samples/<stage.name>/rank%04d.csv        multiplicity, par_0..par_{p-1}, log_posterior                (header row)
  raw_data/<stage.name>/rank%04d.csv       state_type, par_0..par_{p-1}, solver_tag,
                                           obs_0..obs_{m-1}, obs_approx_0..obs_approx_{m-1},
                                           log_likelihood, log_prior                                    (header row; only if save_snapshots_to_file)
  notes/<stage.name>/rank%04d.csv          accepted, rejected, pre-rejected, sum, seed, out_of_bounds,
                                           exact_steps, exact_accepted, audited, audit_hidden            (header row)
  subchain_stats/<stage.name>/rank%04d.csv iteration, subchain_max_length, subchain_accepted, subchain_acceptance_rate, correction_log_ratio, outer_proposed_changed, outer_accepted, rank_world, kernel   (header row; DAMH stages only)
  adaptive_stats/<stage.name>/rank%04d.csv  one row per adaptation period                                (header row; adaptive=True stages only)
  carry_over/<stage.name>.npz              carry_over__*, summary__*                                    (one file per adaptive stage, no rank suffix; written once by sampler rank 0 / run_sampling_local)
  last_sample/<stage.name>/rank%04d.npz    parameters (float64), no_parameters                          (one file per chain, written after each stage)
  surrogate_quality.csv                    snapshots_total, batch_size, rmse, max_abs_error             (header row; collector only)
  surrogate_quality_test.csv               update_index, snapshots_total, n_test, max_log_posterior, rmse, max_abs_error, weighted_rmse, weighted_mean_abs_error, weighted_ess   (header row; collector only; weighted_ess added 2026-09-18, finding S10 -- effective sample size of the posterior weights, Kish's formula; equals n_test when no weights are supplied)
  run_manifest.json                        see "Run manifest" below
solver_output/rank<k>/                     solver-defined scratch directory
post_processing_output/
  summary.csv, report_extended.html, best_fit_solver_visualization_*.png    (written by SamplingRun.write_report)
  selection.json                                                          (the (stage, chain) mask and burn-in, see "Selection mask" below)
```

`<stage.name>` is `alg%04d_<MH|MH-adaptive|DAMH|DAMH-SMU>` (`stages.stage_name`), e.g.
`alg0000_MH`, `alg0001_DAMH-SMU`. The **leading `alg%04d` index is authoritative**: readers
parse it instead of relying on `sorted(os.listdir(...))`, so a renamed or missing stage
directory is reported rather than silently shifting every later stage.

`p` = `Configuration.no_parameters`, `m` = `Configuration.no_observations`; both are fixed
for the whole run, so every row of a file has the same width.

Header rows are written when the file is created, and files are still created lazily: a
stage with `save_to_file=False` writes no `samples`/`notes` file at all (not even a header).

### `samples/<stage>/rank%04d.csv`

| column | meaning |
|---|---|
| `multiplicity` | how many chain iterations this state accounts for (the state itself plus the proposals rejected from it) — see the `+1`/`+0` rule below |
| `par_0 .. par_{p-1}` | the sample, transformed to physical space if `Configuration.transform_before_saving=True` |
| `log_posterior` | `log_likelihood + log_prior` of that state |

One row per acceptance, plus one final row per chain at stage end. Rejected proposals are
*not* written here (they only reach `raw_data`/the surrogate's training data, at weight 0).

**Stage boundaries — the `+1`/`+0` rule** (A30, decided 2026-09-17). The final state of a
stage is written by that stage *and* carried over as the initial state of the next stage,
which writes it again when it leaves it. To count it exactly once, the **first row of a
stage whose initial state was carried over** gets `multiplicity = counter_rejected_current`,
**without** the leading `+1` — the state itself was already counted by the previous stage.
"Carried over" means every stage after the first whose predecessor wrote a samples file
(`save_to_file=True`), including a stage that follows an `is_excluded=True` stage (that
stage wrote the same restart state as its own first row). Consequences:

- per stage: `sum(multiplicity) == accepted + rejected + prerejected + 1` for the first
  stage of a run, and `== accepted + rejected + prerejected` (no `+1`) for every
  carried-over stage;
- concatenating all stages of a chain: `sum(multiplicity) == total iterations + 1` (one
  state per iteration, plus the state the run started from), with no state counted twice;
- a stage's first row may have **multiplicity 0** (its very first proposal was accepted
  immediately). Such a row contributes nothing to `length`, to weighted means/covariances,
  or to the decompressed chain (`post_processing.decompress` drops it), and is not counted
  by `Samples.no_unique_samples` (which uses `np.count_nonzero`, not `len`).

The weight sent to the *collector* for surrogate training is unchanged
(`1 + counter_rejected_current`): the collector never receives a stage's final state, so
there is nothing to double-count there.

### `raw_data/<stage>/rank%04d.csv` (opt-in via `save_snapshots_to_file`)

One row per proposal regardless of outcome, **rectangular**: the same columns for every
stage type, NaN where a quantity does not exist.

| column | meaning |
|---|---|
| `state_type` | `accepted`, `rejected`, `prerejected` or `audited` (S2, 2026-10-09: a pre-rejected proposal evaluated with the exact model by `Stage.audit_prerejected`; it follows the `prerejected` row of its iteration, which repeats the chain state, while the `audited` row holds the rejected proposal itself; never a chain state) |
| `par_0 .. par_{p-1}` | the proposed parameters (physical space if `transform_before_saving=True`) |
| `solver_tag` | the solver's status for this proposal (`< 0` = solver failure, observations invalid); `-2` = the proposal itself was not finite, so the solver was **never called** for it (2026-09-20, see `docs/writing_a_solver.md`); `-3` = the proposal lies outside the prior bound (`Problem(prior_bound=...)`, 2026-10-09), so **nothing** (solver or surrogate) was evaluated: an MH `rejected` row, or a DAMH sub-chain step logged as `prerejected`, with NaN observation blocks and `log_likelihood = -inf`; otherwise always `0` for `prerejected` |
| `obs_0 .. obs_{m-1}` | **exact** model observations `G(x)`. **All NaN for `prerejected` rows** — the exact model was never called for them; filled for `audited` rows. |
| `obs_approx_0 .. obs_approx_{m-1}` | **surrogate** observations `G~(x)`. All NaN where no surrogate value exists (any plain MH stage); filled for every DAMH row. |
| `log_likelihood` | the log-likelihood **the acceptance decision for this row used**: the exact one for `accepted`/`rejected` rows, the **surrogate** one for `prerejected` rows (the sub-chain never left the current state, so only the surrogate was ever evaluated); for an `audited` row the **exact** one (no decision used it; the surrogate value follows from `obs_approx_*`). |
| `log_prior` | `log prior(x)` (model-independent, so the same in either case) |

The single `log_likelihood` column is deliberate: a row is scored either exactly or by the
surrogate, never both, and the non-NaN observation block says which. Consumers that need
exact-model quantities (`find_best_fits`, `hist_observations`,
`Samples.load_snapshots`) therefore **exclude `prerejected` rows**; they keep `audited` rows, which
are exactly evaluated. The histogram of accepted observations (`hist_G`) drops `audited` rows, which
are not chain states.

**S2 rows (2026-10-09).** With `Stage.exact_step_probability > 0` an exact-step iteration writes an
ordinary `accepted`/`rejected` row with BOTH observation blocks filled (the proposal is scored with
the installed surrogate before the exact model), so `log L(y) − log L~(y)` can be read off at points
chosen independently of the surrogate; it is identified by `kernel = 1` in `subchain_stats`, which
is 1:1 with the non-`audited` rows of `raw_data` per rank (plus out-of-box sub-chain rows, tag -3).
With `Stage.audit_prerejected > 0` an `audited` row follows some `prerejected` rows (see `notes`).

This replaces v1's ambiguity, where `prerejected` rows smuggled *surrogate* values into the
same observation block as the exact ones with no marker (finding 3.5 / P10).

One caveat: in a stage with `Stage.use_only_surrogate=True` the surrogate *is* the forward
model (`SurrogateAsSolver`), so its values land in the `obs_*` block and `obs_approx_*` stays
NaN. The block says "what the observation provider returned", and in such a stage that
provider is the surrogate.

### `notes/<stage>/rank%04d.csv`

One row per chain, written once at stage end: `accepted, rejected, pre-rejected, sum,
seed, out_of_bounds`. `out_of_bounds` (appended 2026-10-09) counts proposals rejected unevaluated
because they lie outside the prior bound: outer MH proposals (also counted in `rejected`) and
DAMH sub-chain steps (not counted in the outer columns; each also has a `raw_data` row with tag
`-3`). S2 (2026-10-09) appends `exact_steps, exact_accepted, audited, audit_hidden` (0 on an MH
stage and on a DAMH stage with both S2 fields at 0): exact steps of the mixture kernel (all /
accepted; their outcomes are also counted in `accepted`/`rejected`), audited pre-rejected proposals
(exact evaluations NOT counted in the outer columns; `max_evaluations` stops on `accepted + rejected
+ audited`), and the **audit statistic** `audit_hidden` = the audits with `log L(y) − log L~(y) >
AUDIT_HIDDEN_NATS = 5` nats — points where the surrogate made a region of high exact posterior look
improbable (note 22's P6 statistic; recomputable from the `audited` rows of `raw_data`). Readers
select the columns by name, so files without them (older runs) still load; the post-processing
summary (and the HTML summary table) sums every column except `seed`, and the cost ratio of CpUS
counts `audited` as exact evaluations.

### `subchain_stats/<stage>/rank%04d.csv` (DAMH stages only)

One row per outer DAMH iteration: `iteration, subchain_max_length, subchain_accepted,
subchain_acceptance_rate, correction_log_ratio, outer_proposed_changed, outer_accepted,
rank_world, kernel`. `kernel` (S2, 2026-10-09, appended last, present on every DAMH stage) is `0` for a
delayed-acceptance iteration and `1` for an exact step (`Stage.exact_step_probability`), whose row has
`subchain_accepted = 0`, `subchain_acceptance_rate = NaN`, `correction_log_ratio = 0`,
`outer_proposed_changed = 1` and `outer_accepted` the exact decision. (This CSV column is still literally named `subchain_max_length` in
`surrDAMH/modules/algorithms.py`, unlike the `Stage` field itself, which was renamed to
`subchain_length` — apparent oversight in the library, not fixed here.)

### `adaptive_stats/<stage>/rank%04d.csv` (`adaptive=True` stages only)

One row per completed adaptation period (10 `adapt()` calls by default), written at stage end
(new 2026-09-20). The columns depend on the proposal class, and every row ends with
`rank_world`:

| proposal | columns |
|---|---|
| `GaussRandomWalk_adaptive` | `n, mean_acceptance_probability, log_sigma, trace_C_over_d, shrinkage_delta, rank_world` |
| `PCN_adaptive` | `n, mean_acceptance_probability, beta, rank_world` |
| `Hamiltonian_adaptive`, `HamiltonianInfinite_adaptive` | `m, mean_acceptance_probability, log_step_size, log_step_size_bar, rank_world` |

`mean_acceptance_probability` is the mean over that period of the acceptance probability the
proposal was fed (which in a DAMH stage is the *overall* outer probability for `RandomWalk`/`PCN`
and the *sub-chain* probability for the `Hamiltonian` proposal, see `docs/stages.md`). For the random walk,
`trace_C_over_d` and `shrinkage_delta` are `NaN` in the periods before the `warmup=100` count is
reached, where the covariance is not yet re-estimated (the Robbins–Monro `log_sigma` runs from
the first call). The Hamiltonian's `m` counts *sub-chain* steps in a DAMH stage, so it advances
by `subchain_length` per outer iteration.

No file is written for a stage whose proposal does not adapt, and none for a stage with
`save_to_file=False`; `read_run` then reports `adaptive_stats[stage] is None`
(`Samples.adaptive_stats[stage]` an empty `DataFrame`). A run produced before this file existed
reads back the same way, so the reader is backwards compatible here.

### `carry_over/<stage>.npz` (`adaptive=True` stages only)

Written once (new 2026-09-21, no `rank%04d` suffix -- every rank already pooled to the
identical state, see `GaussRandomWalk_adaptive.set_pooled_state`) at the end of an adaptive
stage, by sampler rank 0 in the MPI runner and unconditionally by `run_sampling_local`: the same
cross-rank hand-over that is printed to stdout as `Stage ... carry-over ...:`, so a report can
show the proposal the *next* stage actually started from. Unlike `save_last_sample`, this is
not guarded by `Stage.save_to_file` -- there is no separate switch for it.

Two groups of keys, each value stored as a numpy array of its own dtype (a scalar as a 0-d array, read back as a Python `int`/`float`):

- `carry_over__<key>`: `Proposal.carry_over()`, i.e. exactly what `build_proposal` consumes
  for the next stage (`scale`, `beta`, or `step_size`; the proposal spec's step field, see
  `docs/stages.md`).
- `summary__<key>`: `Proposal.adapted_summary()`, diagnostic-only (never consumed by
  `build_proposal`) -- for `GaussRandomWalk_adaptive`: `base_cov`, `log_sigma`, `n_pooled`,
  `mean`; for `PCN_adaptive`: `beta`; for the Hamiltonian family: `step_size`.

No file is written for a non-adaptive stage; `read_run` then reports `carry_over[stage] is
None`, same as for a run produced before this file existed.

### `last_sample/<stage>/rank%04d.npz`

Written by both the MPI sampler and `run_sampling_local` after every stage (regardless of
`save_to_file`), so any later run can continue from it
(`Configuration.initial_sample_type="continued"`, `continued_from_dir=...`).

## Run manifest (`run_manifest.json`)

Written by rank 0 (or by `run_sampling_local`) before role dispatch and finalized after the run:
`manifest_version`, `format_version` (`2`), `surrdamh_version`, `runner`
(`"mpi"`/`"local"`), timestamps, `hostname`, package versions, `git` (commit/dirty/branch),
the full `configuration`/`stages`/`prior`/`likelihood` summaries (each stage entry is every
`Stage` dataclass field by name, including `is_excluded` and `burn_in` — see `docs/stages.md` and
the selection mask below), `surrogate`
(updater/evaluator class + scalar hyperparameters), `solver`, `seeds` (`modules.seeds.SEED_FORMULA`:
`seed0 = 10*(no_stages*rank+i) + 1_000_000*generation` and every per-rank/per-stage seed —
`proposal_seed = seed0+1`, `algorithm_seed = seed0+2`, and since S2 (2026-10-09) `exact_step_seed =
seed0+4` (exact-step choice and audit draws) and `exact_proposal_seed = seed0+5` (the exact step's
random walk), listed for every stage though drawn from only by a DAMH stage that uses them — plus
whether `initial_sample_type` makes the run reproducible — `generation` is 0 and `no_stages` this
run's own stage count for a plain run, see "`lineage`" below for a continuation), `mpi` layout,
`environment` (thread-count env vars), `unverified_options` (e.g.
`use_surrogate_gradients was disabled by Problem.run_sampling`), `continued_from` (the OLD manual
continuation only — `Configuration(initial_sample_type="continued", continued_from_dir=...)` —
with the source run's own manifest embedded), and `lineage` (a continuation made by
`SamplingRun.continue_sampling`/`continue_sampling_local`, see below). Manifest writing/finalizing
never aborts a run — any failure there is printed as a `WARNING`, not raised.

The manifest is also what makes a directory readable: `read_run` requires it and requires
`format_version == 2`.

### `lineage` (`SamplingRun.continue_sampling`, 2026-10-08)

Present only on a run produced by `SamplingRun.continue_sampling`/`continue_sampling_local`
(`modules.manifest.lineage_entry`); absent on a plain run and on one continued the old manual way
(`initial_sample_type="continued"` set by hand) — both count as generation 0 of their own lineage
when read back (`core._lineage_of`). Keys:

| key | meaning |
|---|---|
| `continued_from` | absolute output directory of the run this one continues, or `null` |
| `generation` | 0, 1, 2, ... — how many times the lineage has been continued |
| `stage_index_offset` | stage count of every earlier run of the lineage; added to this run's local stage index for both the stage directory names and the seeds |
| `no_stages_lineage` | total stage count of the lineage up to and including this run; the seed formula's `no_stages` |
| `same_problem` | `true` iff this run sampled the previous run's `Problem` unchanged (`false` after `continue_sampling(problem=...)`) |
| `same_problem_lineage` | `same_problem` held for *every* continuation of the lineage so far |
| `chains` | `"continue"`, `"prior"` or `"lhs"` (the `chains=` argument) |
| `initial_sample_is_carried_over` | A30 flag of this run's first stage: `true` iff the previous run's last stage had already written the state these chains start from |
| `dirs` | absolute output directories of the whole lineage, oldest first, this run last |

`modules.run_data.read_lineage(output_dir)` follows `continued_from` recursively and returns one
`RunData` with every per-stage list concatenated in lineage order (new `RunData` fields:
`output_dirs`, `stage_output_dir`, `same_problem`, `generation`, `manifests`, `chain_ranks`; stage
names must be unique across the lineage — a lineage made with the old manual continuation is not,
and raises `RunFormatError` naming `include_previous=False` as the fix).
`post_processing.Samples(..., include_previous=...)` is the read side most scripts use — see
"`include_previous` and the lineage" below.

### `auto` (`Problem.run_sampling_auto`/`run_sampling_local_auto`, 2026-10-08)

Present only on a run started with the automatic mode (`surrDAMH.auto.AutoPlan.manifest_entry()`;
see `docs/running.md#automatic-mode-2026-10-08`); absent otherwise. Also exposed as `run.auto`
(`SamplingRun.auto`). Keys:

| key | meaning |
|---|---|
| `mode` | `"robust"` or `"fast"` |
| `budget` | the `budget=` argument (total exact evaluations, including the held-out set), or `null` if `time_limit=` was given instead |
| `time_limit` | the `time_limit=` argument (seconds), or `null` if `budget=` was given |
| `no_samplers` | number of chains the plan was built for (1 for `run_sampling_local_auto`) |
| `per_chain_budget` | exact evaluations per chain after subtracting the held-out set, or `null` for a time budget |
| `test_data_size` | held-out prior-draw points generated (0 if none) |
| `warm_up` | `{"evaluations": n0}` or `{"time_limit": t0}` for the excluded MH warm-up stage, or `null` for the single-MH-stage layout |
| `chunks` | number of DAMH-SMU chunks after the warm-up (0 for the single-MH-stage layout) |
| `stage_names` | the resolved stage directory names, in order (same as `[s.name for s in run.stages]`) |
| `proposal` | `repr()` of the chunks' proposal spec (`RandomWalk(...)`/`Hamiltonian(...)`), or of the single MH stage's, or `null` |
| `exact_step_probability`, `audit_prerejected` | the S2 settings of the DAMH chunks (`AUTO_EXACT_STEP_PROBABILITY`, `AUTO_AUDIT_PREREJECTED`, 0.05 each), `null` for the single-MH-stage layout |
| `surrogate` | `"none"`, or `{"class", "default", "hparams"}` for the updater actually used (`default=True` for the built-in `NeuralNetworkUpdater`) |
| `conf_settings` | the `Configuration` fields Auto set (see `docs/configuration.md`) |
| `notes` | human-readable reasons for every decision that is not the plain rule (e.g. "use_collector=False: ...", "min_snapshots_initial=7 set by the user is kept") |

JSON-safe (`json.dumps` round-trips it). The plan is also printed once at start-up (rank 0 /
local, `AutoPlan.describe()`) before the stage list itself.

## Selection mask (`post_processing_output/selection.json`, 2026-10-08; layout simplified 2026-10-09)

`SamplingRun.write_report()` loads every chain of every stage, then applies a per-(stage, chain)
posterior mask (`post_processing.Samples`/`post_processing/selection.py`,
`Samples.posterior_view()`) to decide which chains form the posterior. The file lives in the
NEWEST run's own `post_processing_output/`, even when the report covers a whole lineage
(`include_previous`, below):

```json
{"posterior": {"alg0000_MH-adaptive": [0, 0, 0, 0], "alg0001_DAMH-SMU": [1, 1, 1, 1]},
 "runs": ["<dir0>", "<dir1>"],
 "note": "..."}
```

- `posterior[stage][c]` is indexed by the chain's **original** rank number (the position of its
  `rank%04d` file), not its position after an earlier exclusion: `1` = this chain's samples of
  that stage are part of the **posterior**, `0` = they are not (but the chain is still loaded and
  shown, see below). One key per stage of the loaded run(s), in stage order.
- `runs` is written only when the lineage spans more than one run (informational: the absolute
  output directory of each one, oldest first); the loader does not read it back.
- An optional expert entry, `"drop_first_rows": {"alg0001_DAMH-SMU": [5, 0]}`, drops each listed
  chain's leading **compressed** rows (not raw iterations) from the posterior (`raw_data`
  snapshots get no rows dropped). It is **never written by default** — a stage/chain absent from
  it drops 0 rows — and only appears in the file if hand-added.
- There is no `format`, `lineage` (the old key holding what `runs` holds now), `stages`,
  `is_excluded`, `burn_in_stage` or `recommended` key any more; a file still in that pre-2026-10-09
  layout is rejected with `ValueError` naming the file and asking for it to be deleted so it is
  re-created with the current defaults.

**Since 2026-10-09 the mask selects the posterior only — it no longer hides anything.** Every
chain of every stage is always loaded (`Samples` loads every chain regardless of `selection`) and
shown in the report (traces, acceptance/pre-rejection, adaptation, carry-over, counters, per-stage
histograms, `summary.csv`); `posterior`/`drop_first_rows` decide only which chains are pooled into
the *posterior* sections (overall mean & covariance, combined 1D/2D histograms, ESS, R-hat, CpUS
over stage groups, best fits, combined observation histograms, posterior field statistics) and the
default `field_statistics`. `Samples.posterior_view()` returns that posterior subset as a shallow
view (filtered `chain_indices`, per-chain dropped rows sliced off the arrays it shares with the
full `Samples`, and `notes`/`subchain_stats`/`adaptive_stats` filtered to the included chains by
`rank_world` — nothing is re-read or re-decompressed); `write_report`'s return value (below) is
the full, unfiltered `Samples`, and `.posterior_view()` on it is what the report's pooled sections
actually used. Before 2026-10-09 the default selection (`is_excluded` stages left out) dropped
those stages from the report and `summary.csv` entirely; that is no longer true — they still
appear, just outside the posterior. Internally, the validated mask
(`post_processing/loading.py`) still has the pre-2026-10-09 shape
`{stage_name: {"include": [...], "burn_in": [...]}}` — only this file's own layout changed.

`write_report(selection=None)` (the default) creates the file, if missing, with every chain of
every stage marked 1 (posterior) **except** stages with `Stage.is_excluded=True` or
`Stage.burn_in=True` (`post_processing.selection.default_selection`), which get 0 for every chain
by default. The file is then read and applied on this and every later call, so editing it by hand
and calling `SamplingRun.load(output_dir).write_report()` again re-runs the post-processing with
the edit. `write_report(selection=<path>)` uses that file instead (must exist);
`write_report(selection=False)` ignores any file and puts every chain in the posterior with no
rows dropped (the pre-selection behaviour). If *every* stage ends up with no chain in the
posterior, `write_report` raises `ValueError` naming the file to edit. The run manifest is never
modified by any of this — `selection.json` is the only record of what was excluded from the
posterior.

A **user note**, with the file's absolute path and the re-run command verbatim, appears twice in
the report: at the top of the pooled "Overall analysis" section and again in the final "Selection
and re-run" section:

> Which chains form the posterior is decided by `<output_dir>/post_processing_output/selection.json`:
> set each stage's chain list in `posterior` to 1 (used) or 0 (not), then re-run
> `python -c 'import surrDAMH; surrDAMH.SamplingRun.load("<output_dir>").write_report()'` (pass
> `problem=` to `load` for the solver-dependent sections). The note mentions `drop_first_rows` only
> when the file actually has that (optional) entry.

Each chain's label/legend in the per-stage sections also gets a badge, `"in posterior"` or
`"not in posterior (burn-in stage | excluded stage | excluded by user)"`, from the mask and the
stage flags (`Samples.chain_status`); the per-stage summary rows and the pooled "Overall analysis"
section header ("Aggregated over the chains marked 1 in selection.json's "posterior": `<k>` stages,
`<m>` chain-stage pairs.") say which stages/chains actually contributed.

The report's last section, **"Selection and re-run"**, repeats the user note, shows a stage x
chain table (each cell: the 0/1 value, the badge reason, and `drop_first_rows=<n>` when non-zero)
next to the two `Stage` flags (`burn_in stage`, `is_excluded`, read from the run manifest(s), not
from the selection file), lists the excluded chains in prose ("excluded by the user: ...", "not in
the posterior as burn-in stages (burn_in=True): ...", "not in the posterior as excluded stages
(is_excluded=True): ..."), the absolute path of `selection.json`, and the re-run snippet:

```python
import surrDAMH
surrDAMH.SamplingRun.load("<output_dir>").write_report()
```

plus a note that `problem=` can be passed to `load` to restore the sections that need the prior or
a solver (prior overlay, parameter names, posterior field statistics, best-fit solver
visualization).

### `summary.csv` and `write_report`'s return value

`summary.csv` has one row per **displayed** stage of the lineage, with every chain's counters —
unaffected by the selection mask — and gains a final column, `chains_in_posterior`
(`"<chains marked 1>/<all chains>"`, e.g. `"2/4"`); it is appended last so a run with no
exclusion is otherwise byte-identical to before 2026-10-09. Its `autocorr`/`CpUS` columns
(`Samples.calculate_CpUS`, called on the full `Samples` by `core._write_report_rank0`) are
computed over **all** chains of each stage, unlike the HTML report's own CpUS figures — sections
**4.4 "Cost per Uncorrelated Sample (CpUS)"** and **4.8 "Diagnostics Summary"** — which call
`calculate_CpUS` again on `posterior_view()` and so are computed over the posterior chains only;
the two numbers for the same stage can therefore differ.

`SamplingRun.write_report()` returns the full `surrDAMH.post_processing.Samples` (every chain
loaded) on rank 0 / in a local run, not just the posterior subset; call `.posterior_view()` on it
to get the same filtered view the report's pooled sections used.

## `include_previous` and the lineage

`post_processing.Samples(no_parameters, samples_dir, include_previous=None, ...)` and
`SamplingRun.write_report(include_previous=None, ...)` decide how much of a lineage is loaded:

| `include_previous` | What is loaded |
|---|---|
| `None` (default) | the whole lineage (`read_lineage`) iff `same_problem` holds for every link of it; otherwise this run alone (`read_run`), with a note in the report explaining why |
| `True` | the whole lineage regardless — the report then notes that the problem changed, if it did |
| `False` | this run alone |

`summary.csv` then has one row per loaded stage (every stage of the lineage, or just this run's).

## Reading outputs

```python
from surrDAMH import read_run

run = read_run("out_my_experiment")          # the directory that CONTAINS sampling_output/
run.stage_names                               # ['alg0000_MH', 'alg0001_DAMH-SMU'] (index order)
run.samples_columns                           # ['multiplicity', 'par_0', 'par_1', 'log_posterior']
rows = run.samples[0][0]                      # stage 0, chain 0: (n, 2+p) float64
multiplicity, parameters = rows[:, 0], rows[:, 1:1 + run.no_parameters]

snapshots = run.raw_data[1][0]                # a pandas DataFrame with the v2 column names
exact_only = snapshots[snapshots["state_type"] != "prerejected"]

run.notes[0]                                  # DataFrame, one row per chain
run.subchain_stats[1]                         # DataFrame for a DAMH stage, None for MH
run.adaptive_stats[0]                         # DataFrame for an adaptive stage, None otherwise
run.carry_over[0]                             # (carry_over, summary) dict pair, None otherwise
run.last_sample["alg0001_DAMH-SMU"]           # (no_chains, p) float64
run.surrogate_quality, run.surrogate_quality_test   # DataFrames or None
```

`read_run(output_dir, load_raw_data=False)` skips `raw_data` (the largest output of a run);
that is what `post_processing.Samples` uses, since its snapshot consumers stream those
files instead.

Per-stage lists are always `len(stage_names)` long and in stage-index order. A stage that
wrote nothing for a category carries an empty entry: `samples[i] == []` for
`Stage.save_to_file=False`, `raw_data[i] is None` when snapshots were not saved,
`subchain_stats[i] is None` for a non-DAMH stage, `adaptive_stats[i] is None` for a stage whose
proposal did not adapt, `carry_over[i] is None` for a stage whose proposal did not adapt or
whose `carry_over/<stage>.npz` is missing (a run that predates it).

`surrDAMH.post_processing.Samples(no_parameters, output_dir)` is a facade on top of
`read_run` and keeps the plotting/statistics API used by
`SamplingRun.write_report()`; see its docstrings. Note that the posterior *field statistics*
section evaluates the solver once per posterior state; pass `field_statistics_max_samples=<N>` to
sub-sample N states (fixed seed) when chains are long and the solver is fast — otherwise the report
can take longer than the sampling (see `library_notes/14_grf_validation_2026-09-17.md`, F1).
`surrDAMH.modules.continuation.load_last_samples` reads `last_sample/`.

`surrDAMH.post_processing` is a package (WS9b): `loading.py` builds `Samples`/`StageSamples`
from `read_run`, `statistics.py` holds the moments/ESS/R-hat/autocorrelation/best-fit code,
`plots.py` the matplotlib figures and `html_report.py` the report writers. Import paths are
unchanged — everything is re-exported from `surrDAMH.post_processing` itself:

```python
from surrDAMH.post_processing import (Samples, StageSamples, read_run, RunData,
                                      RunFormatError, rank_best_fit_candidates,
                                      decompress, Autocorrelation, autocorr_FM,
                                      auto_window, add_normal_dist_grid)
```

`find_best_fits`, `calculate_gelman_rubin`, `calculate_effective_sample_size` and
`calculate_CpUS` are methods of `Samples`, not module-level functions.

`report_extended.html` opens with a **Run Configuration** section: the effective
configuration (from `run_manifest.json`, unless a configuration object is passed to
`html_report_extended`), the run provenance, and the run's `unverified_options` — the
options the manifest flags as not covered by the library's verification. It is followed by
**Sampling Stages** (2026-09-21): one column per stage of the run, one row per `Stage` field
(`*` = posterior-/acceptance-rate-affecting, as in `Stage.describe()`; `unbounded` = a
stopping condition that was not set), from the `stages=` argument of `html_report_extended`
or, by default, the manifest's `stages` list, with a last row saying which stages the report
analyses.

Every section and every per-stage block of the report is a collapsible `<details>` element,
**collapsed when the file is opened** (2026-09-21); use the "Expand all" / "Collapse all"
buttons under the title, or the table-of-contents links, which unfold the section they point
into. Section **5. Proposal Adaptation** plots, for every displayed stage with an adaptive
proposal, the `adaptive_stats/<stage>/rank%04d.csv` trace described above
(`Samples.plot_adaptation(stage, chains_to_disp=None, target_rate=None)`): the per-period
`mean_acceptance_probability` with the target rate as a dashed line, then every adapted
parameter column as logged (`log_sigma` stays on the log scale), one line per chain, plus a
table of the last logged period of every chain. Surrogate quality and observation histograms
are sections 6 and 7 since then.

### Refusal of pre-v2 directories

Opening a directory that is not format v2 raises
`surrDAMH.RunFormatError` (also `surrDAMH.modules.manifest.RunFormatError`), for example:

```
<dir>/sampling_output/run_manifest.json not found: this is not a surrDAMH output format v2
directory (a run older than the manifest, or not an output directory at all). Pre-v2 runs
cannot be read -- there is no converter (author decision 6, 2026-09-17); re-run the
sampling to obtain v2 output.
```

and, for a directory that declares another version:

```
<dir>/sampling_output/run_manifest.json: output format_version 1, expected 2. Runs written
in another format cannot be read -- there is no converter (author decision 6, 2026-09-17);
re-run the sampling to obtain v2 output.
```

The same error is raised when a CSV header does not match the v2 column layout.
