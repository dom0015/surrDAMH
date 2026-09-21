# Output format v2 — specification (WS9, decision 6: no converter)

**Dated record.** Written as a pre-implementation design draft (2026-09-16); §1 was already
the pre-change layout and §2 the proposed one. Implemented as proposed, decisions applied
inline. Kept for the writer↔reader design rationale; the current contract is `docs/outputs.md`
and the current behaviour is verified in `10_manual_review_notes.md` §2.14. Inline references
below to `03_surrogates_and_distributions.md`/`04_post_processing.md` point to per-subsystem
review notes deleted 2026-09-18 (superseded by `06_findings_consolidated.md`) — left as-is,
they're historical citations of what those notes said at the time, not live links.

Status: **implemented 2026-09-17 (WS9a)**, with WS9b applied the same day. The writers,
`manifest.FORMAT_VERSION = 2`, `read_run`, the `post_processing` reader migration, the tests
and `docs/outputs.md` are in the working tree; §1 below still describes the *old* (v1) layout
for reference. WS9b then did what WS9a had left open — splitting `post_processing.py` into the
package `surrDAMH/post_processing/`, the P6 `except BaseException` cleanup, the
`find_best_fits` rewrite (P3 + P10), removing `load_posterior_surrogate` (P1) and honouring
`chains_to_disp` (P5) — without changing any output file. See
`10_manual_review_notes.md` §2.15.

**Deviations from this spec as written, decided while implementing:**

1. **The multiplicity column is called `multiplicity`**, not `weight` (author naming decision,
   2026-09-17): CSV header, `read_run`/`RunData` field names and docs. `StageSamples.weights`
   and `AlgorithmBase._current_state_row_weight` keep their historical *internal* names.
2. **`RunFormatError` lives in `modules/manifest.py`** (open question 5), which also owns
   `FORMAT_VERSION` and the two column-name builders `samples_columns`/`raw_data_columns`
   shared by the writer and the reader. It is re-exported as `surrDAMH.RunFormatError`.
3. **No `allow_unversioned` escape hatch** (open question 4): "no converter, no reader".
4. **The `obs_approx_*` block exists for every stage type** (open question 2), all-NaN where
   there is no surrogate — one layout to learn, no format detection at read time.
5. `RunData.raw_data[stage][chain]` is a **`pandas.DataFrame`**, not an `np.ndarray` as §3
   sketched: the rows are mixed-dtype (`state_type` is text).
6. `read_run` takes **`load_raw_data: bool = True`**. `post_processing.Samples` passes `False`:
   snapshot files are the largest output of a run and its consumers (`find_best_fits` in
   particular) stream them chunk by chunk, so loading them eagerly into `RunData` would be a
   memory regression for big runs.
7. `read_run` lives in **`surrDAMH/modules/run_data.py`** (so `Samples` can stay a facade) and
   is re-exported as `surrDAMH.read_run`.
8. `stage_names` is the union of the stage directories under *all* per-stage data names, not
   just `samples/` — otherwise a `save_to_file=False` stage would be missing from the index.
9. For a `prerejected` row, `log_likelihood` holds the **surrogate** log-likelihood the
   sub-chain decision used (v1 wrote the carried-over exact value of the current state, which
   was inconsistent with the surrogate observations in the same row). One column, with the
   non-NaN observation block saying which quantity it is; documented in `docs/outputs.md`.

Every "today" row is backed by a file:line actually read
2026-09-16; every "v2" row cites the finding it fixes. Where this contradicts `04_post_processing.md`
or `03_surrogates_and_distributions.md`, it is flagged **[NOTES DISAGREE]** — those notes predate
WS0–WS4 fixes already in the working tree.

## 1. Today (v1) — every file under `<output_dir>/`

| Path | Writer (file:function) | Header | Columns / shape | Delimiter | When written | Reader | By position/name |
|---|---|---|---|---|---|---|---|
| `sampling_output/samples/<stage.name>/rank%04d.csv` | `algorithms.py:246 _record_current_sample`, called from `_emit_current_state` (218) on every acceptance and once from `_finalize_run` (267) | none | `weight, u_1..u_p, log_posterior` (2+p cols) | `,` | one row per acceptance + 1 final row per chain | `post_processing.py:59 StageSamples.__init__` (78-91) | **by position** (`iloc`) |
| `sampling_output/raw_data/<stage.name>/rank%04d.csv` (only if `conf.save_snapshots_to_file`) | `algorithms.py:187 _record_proposed_snapshot`, called from `_handle_acceptance`(220)/`_handle_rejection`(226)/`Algorithm_DAMH.run`(534, prerejected) | none | `state_type, u_1..u_p, solver_tag, obs_1..obs_m, log_likelihood, log_prior` (3+p+m cols) | `,` | one row per proposal (accepted/rejected/prerejected) | `load_snapshots`(265), `_load_...`(1190), `find_best_fits`(1245), `hist_observations` | **by position** |
| `sampling_output/notes/<stage.name>/rank%04d.csv` | `algorithms.py:266 _finalize_run` (268 header row, 271 values row) | **yes**, written as an ordinary first CSV row | row1: `accepted,rejected,pre-rejected,sum,seed` (names); row2: the 5 values | `,` | once per chain, at stage end | `Samples.load_notes`(147) | by **name** (`pd.read_csv` default header) |
| `sampling_output/subchain_stats/<stage.name>/rank%04d.csv` (DAMH stages only) | header: `algorithms.py:161 AlgorithmBase._prepare_run`(162-174, conditioned on `algorithm_type=="DAMH"`); rows: `Algorithm_DAMH.run`(535-548) | **yes** | `iteration, subchain_max_length, subchain_accepted, subchain_acceptance_rate, correction_log_ratio, outer_proposed_changed, outer_accepted, rank_world` | `,` | one row per outer DAMH iteration | `Samples.load_subchain_stats`(161), `summarize`(195-201) | by **name** |
| `sampling_output/last_sample/<stage.name>/rank%04d.npz` | `continuation.py:14 save_last_sample` | n/a (npz) | `parameters` float64 `(p,)`, `no_parameters` int | n/a | once per chain per stage, after the stage | `continuation.py:24 load_last_samples` | by key |
| `sampling_output/surrogate_quality.csv` | `process_COLLECTOR.py:251-254` | **yes** | `snapshots_total, batch_size, rmse, max_abs_error` | `,` | once, all rows written together at collector shutdown | `Samples.plot_surrogate_quality` (not read here) | by name |
| `sampling_output/surrogate_quality_test.csv` | `process_COLLECTOR.py:259-270` | **yes** | `update_index, snapshots_total, n_test, max_log_posterior, rmse, max_abs_error, weighted_rmse, weighted_mean_abs_error` | `,` | once, at collector shutdown | `plot_surrogate_quality_test[_weighted]` | by name |
| `sampling_output/run_manifest.json` | `modules/manifest.py` (`build_run_manifest`/`write_run_manifest`/`finalize_run_manifest`) | n/a (JSON) | see `manifest.py:245-271`; `format_version=1` (`manifest.py:36`) | n/a | written before role dispatch, finalized after (`core.py`, `runner_local.py`) | none yet (WS9 adds `read_run`) | n/a |
| `solver_output/rank<k>/` | solver-defined (`Solver` scratch dir) | n/a | solver-specific | n/a | per solver call | not read by `post_processing.py` | — |
| `post_processing_output/summary.csv` | `core.write_report` → `Samples.get_summary(csv_filepath=...)` (`post_processing.py:232-236`) | **yes** | `accepted,rejected,pre-rejected,sum,seed?,subchain_acc_rate,subchain_move_rate,outer_acc_given_move[,ratio_eval,autocorr,CpUS]` | `,` | once, at report time | external scripts only (not re-read by the library) | by name |
| snapshots-to-file | not a separate path — snapshots that reach the collector are only ever persisted via `raw_data/` (above) or the surrogate's own checkpoint (`surrogate_training_data.npz`, `surrogates/reuse.py:12`); there is no separate "snapshots" file | — | — | — | — | — | — |

**Notes-vs-code disagreements found** (fixed in the tree since `03`/`04` were written, both dated
2026-09-12/13; not yet reflected there):
- `04`'s **P2** ("`no_unique_samples` never updated, stays 0") is **fixed**: `StageSamples.__init__`
  sets it inside the per-chain loop (since 2026-09-17 as `np.count_nonzero(self.weights[i])`, so an
  A30 weight-0 boundary row is not counted).
- `03`'s **S3** ("kd-tree `1/distances` has no zero guard, NaN on exact hit") is **fixed**:
  `nearest_kdtree.py:33-39` uses `np.maximum(distances, 1e-300)` plus an explicit `exact_hit` branch
  that returns the exact training observation; pinned by `tests/unit/test_surrogates.py::TestKDTreeExactHit`.
- `03`'s **S0/S1** (unimportable package; `TestData.as_surrogate_test_data` called with wrong args)
  are fixed: `core.py:158` now calls `td.as_surrogate_test_data()` with no arguments; `temptemptemp`
  no longer exists in `core.py`.

## 2. Proposed v2

| Item | v2 change | Finding |
|---|---|---|
| Header row | Every CSV gets a header row, including `samples/*.csv` (`weight,par_0..par_{p-1},log_posterior`) and `raw_data/*.csv` (`state_type,par_0..par_{p-1},solver_tag,obs_0..obs_{m-1},log_likelihood,log_prior`). `pd.read_csv`/`np.loadtxt` calls that assume no header must pass `header=0`/`skiprows=1`. | P-format, WS9 bullet 1 |
| `raw_data` rectangularity | Fixed-width observation block, NaN-filled when a row has no real observations (e.g. `state_type="prerejected"` rows currently store *surrogate* observations in the same block as exact ones with no marker — v2 splits this: `obs_0..obs_{m-1}` is the exact-model block, always NaN for `prerejected` rows; a new `obs_approx_0..obs_approx_{m-1}` block holds the surrogate value where available (every DAMH row) and NaN elsewhere). One `state_type` column (`accepted\|rejected\|prerejected`) is already the first column — v2 just gives it a header and forbids adding new row shapes without adding NaN-filled columns for the others. | P10, 5.x, finding 3.5 (surrogate-vs-exact ambiguity in raw_data) |
| `format_version` | `manifest.py` `FORMAT_VERSION` bumps 1→2. `read_run`/`Samples` **refuse** to open a `run_manifest.json` with `format_version != 2` (missing manifest = pre-manifest run = also refused) with a message naming the found/expected version and pointing at decision 6 (no converter). | WS9 bullet 1, decision 6 |
| Stage directories | Keep `alg%04d_<type>` (`stages.py:60 stage_name`) — index-prefixed **and** type-suffixed, unchanged in v2. `Samples`/`read_run` must stop relying on `sorted(os.listdir(...))` happening to equal index order (true only because of zero-padding) and instead parse the leading `alg%04d` index explicitly, so a manually renamed/reordered directory is caught rather than silently mis-indexed. | WS9 bullet "keyed by index and name" |
| A30 stage-boundary state | **DECIDED 2026-09-17 and already IMPLEMENTED in v1** (this row supersedes the earlier "adopt option (C)" paragraph, which proposed an extra zero-weight endpoint row). The decided rule: a stage's final row keeps its **full** weight (its real dwell time), and the **first row of a stage whose initial state was carried over** is written with `weight = counter_rejected_current`, i.e. **without** the leading `+1`, because that state was already counted by the previous stage. "Carried over" = every stage after the first whose predecessor had `save_to_file=True`, including a stage following `is_excluded=True` (that stage already wrote the same restart state as its own first row). Per-stage invariant: `sum(weight) == accepted+rejected+prerejected+1` for the first stage, `== accepted+rejected+prerejected` for a carried-over stage; concatenated over a chain: `total iterations + 1`. No new rows and no format change — the only new value in the existing column is a possible `weight = 0` on a stage's first row. `decompress()` (`post_processing.py`) needs no change (`np.cumsum` of the weights already drops it); `no_unique_samples` now uses `np.count_nonzero`, not `len` (applied). The collector's surrogate-training weight is deliberately unchanged, since `_finalize_run` never forwards a stage's final state. Implemented in `AlgorithmBase._current_state_row_weight` + the `initial_sample_is_carried_over` flag both runners set. | A30 / finding 5.7 |
| A30 — why not (B) or (C) | (B) (never re-write a carried-over endpoint) silently drops the dwell time between the last acceptance and the stage boundary and leaves a stage file without its endpoint. (C) (extra zero-weight endpoint row) keeps that information but adds a row shape and needs every reader to treat weight-0 rows as markers. The decided rule keeps the endpoint **with its real weight** in the stage that produced it and removes the duplicate where the duplication actually happens — the *next* stage's first row — so every per-stage report is self-contained and concatenation is exact, with no new rows. | decided 2026-09-17 |
| `notes/*.csv` | Keep as a proper CSV (it already has a header row, `algorithms.py:268`) — no structural change beyond documenting it as one data row per chain (today's two-row-per-file form is already "header then one values row", i.e. already what a plain CSV reader expects; the note in `04` describing it as unusual is just describing that exact plain-CSV shape). No change needed here beyond the general header requirement (already met). | WS9 bullet ("what `notes` becomes") — answer: **unchanged**, already a proper single-row-per-file CSV |
| `subchain_stats` columns | Keep the current 8 columns verbatim (already named, already read by name in `summarize()`); v2 adds no new columns. | WS9 bullet ("subchain_stats columns named") — already satisfied |
| `last_sample` | Unchanged (`.npz`, float64, `parameters`+`no_parameters`) — no posterior-affecting or shape reason to touch it. | — |
| `snapshots_total` counts | Stay in `surrogate_quality.csv`/`surrogate_quality_test.csv` (collector-owned, unaffected by the sampler-side format change); the known double-count at `initial_snapshots` (finding 2.8, `process_COLLECTOR.py:130` vs `186`) is a WS8 collector bug, not a v2 format change — v2 does not fix it, only notes it is orthogonal. | WS9 bullet ("where do snapshots_total counts live") |
| `summary.csv` | Keep the current v1-working-tree names (`subchain_acc_rate`, `outer_acc_given_move`, `ratio_eval`, per `post_processing.py:203-205,245`); this is external-consumer-facing only, not read back by the library. | P4 (documents the rename, does not reverse it) |

## 3. Reader contract: `read_run(output_dir) -> RunData`

```python
def read_run(output_dir: str) -> RunData: ...

@dataclass
class RunData:
    manifest: dict                          # parsed run_manifest.json, format_version == 2 enforced
    no_parameters: int
    no_observations: int
    stage_names: list[str]                  # index order, parsed from "alg%04d_..."
    samples: list[list[np.ndarray]]         # [stage][chain] -> (n_i, 2+p) float64, incl. header-verified columns
    notes: list[pd.DataFrame]               # [stage] -> one row per chain
    subchain_stats: list[pd.DataFrame | None]  # [stage] -> None for MH stages
    raw_data: list[list[np.ndarray] | None] # [stage][chain] -> rectangular (n_i, 3+2p... ) or None if save_to_file was False for raw_data
    last_sample: dict[str, np.ndarray]      # stage_name -> (no_chains, p)
    surrogate_quality: pd.DataFrame | None
    surrogate_quality_test: pd.DataFrame | None
```

- Raises `RunFormatError` (new, specific — not `BaseException`, cf. P6) if `run_manifest.json` is
  missing or has `format_version != 2`; message states found/expected version and that no
  converter exists (decision 6), pointing at the archived-run caveat.
- `save_to_file=False` stages (`Stage.save_to_file`, `stages.py`): `samples[stage]` is `[]` for
  those stages, not a list of empty arrays — callers must check length, not assume one entry per
  configured stage. Same for `raw_data` when `conf.save_snapshots_to_file=False`.
- Pool vs local mode (finding 2.7/I10): `read_run` does not care which collector/pool topology
  produced the files — the on-disk layout is topology-independent already (each mode writes the
  same tree). `read_run` only needs the manifest's `mpi.layout`/`runner` fields to label the report,
  not to change parsing.
- `Samples` (existing class) becomes a thin facade: it keeps its public plotting/statistics methods
  but its `__init__` calls `read_run` internally and stores the `RunData`, instead of re-walking
  directories itself. `StageSamples` survives as a per-stage view assembled from `RunData.samples[i]`.

## 4. Migration (decision 6: no converter)

Breaks at the format switch — concrete grep-verified call sites:

| File | Assumption that breaks | Fix needed |
|---|---|---|
| `tests/unit/test_post_processing.py` (writes synthetic `header=None`-style rows via `pp.Samples(...)`, e.g. around lines 93-301) | Constructs fixture CSVs matching v1's no-header, positional layout | Fixtures must gain header rows and the new `state_type`/observation-block layout |
| `tests/validation/test_gaussian_toy.py::_read_weighted_samples` (line 113-119) | `np.loadtxt(path, delimiter=",", ndmin=2)` with no `skiprows` — a header row makes every field a string and `loadtxt` raises `ValueError: could not convert string to float` | Add `skiprows=1` (or switch to `pd.read_csv`) |
| `tests/mpi/test_mpi_basic.py:71,113,132,190` | `np.loadtxt(f, delimiter=",", ndmin=2)` on `samples/*.csv`, asserting column count `1+2+1` and reading columns positionally | Same `skiprows=1` fix; column-count assertion still valid (header adds a row, not a column) |
| `tests/mpi/test_mpi_surrogate.py:214` | Already uses `skiprows=1` for `surrogate_quality*.csv` (those already have headers today) — unaffected | none |
| `tests/mpi/conftest.py:319` `stage_sample_files` | Lists files by name only (`rank%04d.csv`), does not parse contents — unaffected by header/format changes, only by the A30/zero-weight row addition if it inspects row counts anywhere (it does not, per grep) | none |
| `tests/unit/test_algorithms_local.py::test_mh_weight_column_bookkeeping` | Asserts `weight_sum == accepted+rejected+prerejected+1`. It runs a single, FIRST stage, so the `+1` survives the decided A30 rule unchanged — **DONE 2026-09-17**: the test keeps its assertion (with a comment explaining why) and `test_mh_weight_column_drops_the_plus_one_for_a_carried_over_initial_state` plus `tests/test_runner_local.py::test_two_stage_weights_count_each_state_once` / `::test_stage_after_is_excluded_stage_also_drops_the_plus_one` cover the `+0` case. | done |
| `surrDAMH/post_processing.py` itself | Every `pd.read_csv(path, header=None)` call on `samples/`/`raw_data/` (7 call sites found: `post_processing.py:78,293,651,1219,1313` and the two `StageSamples`/`find_best_fits` chunked variants) | Drop `header=None`, use column names instead of `iloc[:, k]` positional indexing throughout — this is the bulk of the WS9 "mechanical" work the improvement plan refers to |
| `surrDAMH/modules/continuation.py`, `surrDAMH/surrogates/reuse.py` | Read `.npz`/`.pt`, not CSV — unaffected by this change | none |

WS9 checklist (from the above, plus the plan's own acceptance criteria):
1. Add headers to `samples/`, `raw_data/` writers (`algorithms.py`).
2. ~~Implement the A30 (C) zero-weight boundary row in `_finalize_run`/stage hand-over.~~ **DONE 2026-09-17** in v1, with the decided rule (drop the `+1` on a carried-over stage's first row) instead of option (C); no v2 work left.
3. Rectangularize `raw_data` (two observation blocks, NaN-filled).
4. Bump `manifest.py:FORMAT_VERSION` to 2; add the "no manifest / wrong version" refusal to `Samples`/`read_run`.
5. Rewrite every `post_processing.py` reader to use headers/names, not `header=None`/`iloc[:,k]`.
6. Fix the 3 test files above (`_read_weighted_samples`, `test_mpi_basic.py`, `test_post_processing.py` fixtures); add the multi-stage A30 test.
7. ~~`no_unique_samples` → count `weight>0` rows only.~~ **DONE 2026-09-17** (`np.count_nonzero` in `StageSamples.__init__`).

## 5. Open questions for the author

1. ~~A30: option (C) or (B)?~~ **ANSWERED/decided 2026-09-17**: neither — the stage-final row keeps its full weight and the *next* stage's first row drops the `+1`. Implemented in v1 (see the A30 rows in §4 above and `docs/outputs.md`).
2. Should the `raw_data` `obs_approx_*` block (new in v2) be added even for non-DAMH stages, where it would always be all-NaN — or gated by `stage.algorithm_type=="DAMH"` (variable column count, format-detection at read time)?
3. Is losing the "`state_type` currently smuggles surrogate observations into the exact-observation block for `prerejected` rows" behavior an acceptable breaking change for anyone parsing `raw_data` by hand outside `post_processing.py`?
4. `read_run`'s refusal on missing/old manifest: should there be an explicit escape hatch (`allow_unversioned=True`) for a user who wants to inspect an old `out_*` directory read-only, or is "no converter, no reader" the intended hard line?
5. Should `RunFormatError` live in `post_processing.py`, a new `surrDAMH/exceptions.py`, or `modules/manifest.py` (which already owns `format_version`)?
