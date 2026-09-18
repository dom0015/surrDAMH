# Dead code report (WS10 follow-up: ruff + vulture, grep-verified)

Read-only analysis, 2026-09-16. No source changed. Snapshot is **after** the uncommitted
WS0–WS8 work described in the auto-memory note "ws0-safe-changes-implemented" (commit
`175059e` + further uncommitted WS1–WS8 changes) — several items `09_improvement_plan.md`
§WS10 lists as "delete" (`core.temptemptemp`, `Algorithm_PARENT`, commented debug blocks,
`if False and …`) are **already gone**; only the still-present items are listed below.

## (a) Tools run

- `/dolfinx-env/bin/python3 -m pip install --quiet ruff vulture` — **succeeded**, installed
  `ruff 0.16.7`, `vulture 2.16` into `/dolfinx-env`.
- `cd /workspaces/surrDAMH && /dolfinx-env/bin/python3 -m ruff check --select F401,F841,F811,F632,E711,E712 --output-format concise surrDAMH toy_examples`
- `cd /workspaces/surrDAMH && /dolfinx-env/bin/python3 -m vulture surrDAMH --min-confidence 60`
- `cd /workspaces/surrDAMH && /dolfinx-env/bin/python3 -m vulture surrDAMH --min-confidence 80`
- No `--fix` applied anywhere. `out_*/` and `TSX_experiments_archived/` were never passed to
  either tool and neither tool touched them (ruff respects `.gitignore`; vulture was only
  pointed at `surrDAMH/`).
- Every hit below was then checked with `grep -rn <symbol> surrDAMH/ tests/ toy_examples/
  README.md library_notes/` (exact commands per row where non-trivial); vulture only scans
  `surrDAMH/`, so several "unused" hits are toy_examples/tests-only callers — a scope
  blind spot of the tool, not real dead code.

**Counts**: ruff 10 hits (all `F401`/`F811`, all in `toy_examples/`, 0 in `surrDAMH/`); vulture
`--min-confidence 60` → 73 hits, `--min-confidence 80` → 4 hits.

## (b) Safe to delete — grep-verified zero uses anywhere

| Symbol | Definition | Grep command | Result |
|---|---|---|---|
| `TAG_READY_TO_RECEIVE`, `TAG_DATA` | `surrDAMH/modules/communication.py:16-17` | `grep -rn "TAG_READY_TO_RECEIVE\|TAG_DATA" surrDAMH tests toy_examples` | only the two `=` definitions |
| `Stage.proposal` field | `surrDAMH/stages.py:18` | `grep -rn "stage\.proposal\b\|\.proposal\b" surrDAMH` | only `self.proposal` (a different attribute, set from `proposal_type`) is read; the `Stage.proposal` field itself never appears on the RHS anywhere |
| `Proposal.subchain_length` (class default + `__init__` copy) | `surrDAMH/modules/proposals.py:16,23` | `grep -rn "subchain_length" surrDAMH tests toy_examples` | only the two assignments |
| `PriorIndependentComponents.sd_approximation` | `surrDAMH/distributions/independent_components.py:38` | `grep -rn "sd_approximation" surrDAMH tests toy_examples` | only the assignment |
| `closest_point_distance`, `closest_point_distance_kdtree` | `surrDAMH/surrogates/parent.py:174,185` | `grep -rn "closest_point_distance" surrDAMH tests toy_examples` | only the two `def`s |
| `run_COLLECTOR(surrogate_delayed_init_data=…)` param | `surrDAMH/process_COLLECTOR.py:96` | `grep -rn "run_COLLECTOR(" surrDAMH` | single caller `core.py:204` does not pass it; always `None` |
| `evaluate_on_a_grid` | `surrDAMH/modules/tools.py:15` | `grep -rn "evaluate_on_a_grid" surrDAMH tests toy_examples` | only the `def` and its own error message |
| `algorithm_interfaces_local.get_snapshot_arrays` | `surrDAMH/modules/algorithm_interfaces_local.py:154` | `grep -rn "get_snapshot_arrays" surrDAMH tests toy_examples` | only the `def` |
| `TestData.reduce_size` | `surrDAMH/modules/test_data.py:81` | `grep -rn "reduce_size" surrDAMH tests toy_examples` | only the `def` |
| `torch_perceptron.py` / `torch_perceptron_minibatches.py::get_loss_on_data` | `torch_perceptron.py:270`, `torch_perceptron_minibatches.py:532` | `grep -rn "get_loss_on_data" surrDAMH tests toy_examples` | only the two `def`s |
| `NeuralNetworkUpdaterMinibatches.save_snapshots` | `torch_perceptron_minibatches.py:645` | `grep -rn "\.save_snapshots(" surrDAMH tests toy_examples` | only `sampling_TSX.py:80` calls `save_snapshots()`, and that updater is `NeuralNetworkUpdaterBasic` (torch_perceptron.py's version, which **is** called) |
| `post_processing.py::add_normal_dist_grid` | `surrDAMH/post_processing.py:2385` | `grep -rn "add_normal_dist_grid" surrDAMH tests toy_examples` | only the `def` |
| `post_processing.py::calculate_autocorr_time` (not `_mean`) | `surrDAMH/post_processing.py:2363` | `grep -n "calculate_autocorr_time\b" surrDAMH/post_processing.py` | only the `def`; call sites (`:253,940`) all use the sibling `calculate_autocorr_time_mean` |
| `independent_components.Beta` alias | `surrDAMH/distributions/independent_components.py:136` | `grep -rn "\bBeta(" surrDAMH tests toy_examples` | 0 calls (sibling aliases `Uniform`/`Lognormal` **are** used, see §c) |

## (c) Unused in `surrDAMH/` but keep / decide (protocol, dynamic, test-only, or already-decided)

| Symbol | Definition | Why not dead / status |
|---|---|---|
| `Updater.supports_training_data_persistence`, `supports_state_persistence` | `surrogates/parent.py:133,137` | Documented `Updater` contract; grep shows **0 callers anywhere**, not even in `SurrogateReused` (`surrogates/reuse.py`), which calls `load_state` directly without checking. `09_improvement_plan.md` §WS10 lists this as "decide per item" (not decided) — flagging as verified-dead-but-undecided rather than moving to (b). |
| `_request_pending` (`algorithm_interfaces_local.py:198,205,223,229,273,289,311,341`) | local runner evaluator adapter | Set/reset in 8 places, **never read** anywhere in the file or elsewhere — looks like an intended pending-request guard that was never wired to a conditional. Possible logic gap, not obviously safe to delete without checking intent (WS2 author area). |
| `snapshot_count_since_last_update` (`algorithm_interfaces_local.py:277,322,335`) | local runner evaluator adapter | Incremented/reset, **never read** — same pattern as above, possibly an abandoned throttle. |
| `communication.py::PendingRequest.active` (`:133,161`) / `max_requests` (`:381`) | MPI comm helpers | Set, never read — same "write-only flag" pattern; worth a second look before deleting since it may be a half-finished robustness feature (WS8 area). |
| `TestData.generate`, `.reuse` | `modules/test_data.py:34,57` | Test-only today: called from `tests/unit/test_helpers.py`, `tests/mpi/test_mpi_surrogate.py` (via a `SurrogateTestData` alias). Not yet used by `toy_examples/` (WS5 has not moved `sampling_diffusion_grf.py`/`toy_example_hamilton.py` off the `tools.py` helpers — see next row). |
| `tools.generate_surrogate_test_data`, `compute_test_log_posterior`, `normalized_weights_from_log_posterior`, `surrogate_restart_state_has_snapshots`, `load_surrogate_restart_state_if_available` | `modules/tools.py:41,63,71,80,119` | **Not dead** — all five are imported and called by `toy_examples/sampling_diffusion_grf.py` and `toy_examples/toy_example_hamilton.py` today. `09`'s WS10 row ("delete after WS5 moves the two toy examples to `TestData`/`SurrogateRestart`") is a future-conditional, not yet true. Vulture missed the toy_examples usage (scope blindness). |
| `SamplingFramework.write_report`, `Samples.html_report`, `rank_best_fit_candidates` | `core.py:241`, `post_processing.py:871`, `post_processing.py:17` | **Not dead** — `write_report` called from `toy_examples/sampling_diffusion_grf_simplified.py:41`; `html_report` from `toy_examples/post_processing_example.py:80`; `rank_best_fit_candidates` from `tests/test_best_fit_ranking.py`. All three missed by vulture (scope blindness: toy_examples/tests not scanned). `rank_best_fit_candidates` is still the known WS10/P3 issue: it is tested but **not** on the production `find_best_fits` path — keep as-is, tracked in `06_findings_consolidated.md` 5.3. |
| `Samples.pdf_report`, `Samples._load_snapshot_parameters_and_observations`, `Samples.load_snapshots` | `post_processing.py:828,1187,265` | `load_snapshots`/`_load_snapshot_parameters_and_observations` have no current caller anywhere (incl. toy_examples/tests) but are the documented raw-`raw_data/` reader pair referenced in `04_post_processing.md`; `pdf_report` likewise has 0 callers. Genuinely unused today, but part of the documented legacy (non-extended) report API — listed here rather than (b) pending an author call on whether the legacy report path is still wanted. |
| `distributions/transformations.py::lognormal_to_normal`, `beta_to_normal` | `:19,54` | Test-only: exercised by `tests/unit/test_distributions.py:200,207` (round-trip checks for the forward transforms). Not called in production code — matches the WS10 "decide" entry for `*_to_normal`/`beta_to_uniform` inverses; `beta_to_uniform` itself has an internal caller (`beta_to_normal`) so vulture did not flag it separately. |
| `independent_components.Uniform`, `Lognormal` aliases | `independent_components.py:134-135` | **Not dead** — both used in `toy_examples/sampling_TSX.py:47-55`. Only the sibling `Beta` alias (§b) is actually unused. |
| `torch_perceptron.py` (`NeuralNetworkUpdaterBasic`, whole module) | `surrogates/torch_perceptron.py` | **Decided, WS6**: `09_improvement_plan.md` decision 4 says delete/merge into minibatches. Not yet applied — still exported in `surrogates/__init__.py`, still used by `toy_examples/neural_network_surrogate.py`, `neural_network_surrogate_copy.py`, `sampling_TSX.py`, and unit-tested in `tests/unit/test_surrogates.py`. Do not delete until those callers are migrated. |
| ~~`Configuration.pickled_observations` + raw-observation MPI path~~ | ~~`configuration.py:45`, `communication.py:380-401`, `process_SOLVER.py`, `process_CHILD.py:72`~~ | **REMOVED 2026-09-17** (decision 5 / WS8), together with the rewrite of `tests/mpi/test_mpi_transport.py`. See `CHANGELOG.md` and `10_manual_review_notes.md` §2.17. |
| `Configuration.paths_to_append` / `_append_path` | `configuration.py:44,57-60,96-98` | Wired and called from `__post_init__`, so not "dead" by vulture's definition, but no toy example currently sets it and finding M18 (`06_findings_consolidated.md` 2.12) says it doesn't reach spawned children anyway — `09`'s WS10 plan is to replace it with an absolute path on `SolverSpec`, not delete outright. |
| `Configuration.debug` | `configuration.py:47` | **Not dead** — read at `process_COLLECTOR.py:197,236`. Listed only because the task brief asked to check it explicitly. |

## (d) Linter/manual findings that look like real bugs

| Finding | Location | Note |
|---|---|---|
| Unreachable code after `raise` | `surrDAMH/modules/algorithms.py:317` | `_compute_surrogate_log_likelihood_gradient` does `raise NotImplementedError(...)` then a line computing `argument = self.prior.transform(...)` that can never execute (vulture 100% confidence). Low practical impact since the guarded feature (`transform_before_surrogate=True` gradients) is unimplemented anyway, but the dead line should either be removed or the `raise` turned into the intended fallback. |
| Duplicate imports | `toy_examples/grf_diffusion.py:11/25` (`numpy as np`) and `:14/26` (`matplotlib.pyplot as plt`) | ruff `F811`: same module imported twice, second shadows the first. Harmless but sloppy; one-line fix. |
| `core.py::identity` fallback is effectively dead | `surrDAMH/core.py:30,155-156` | `if not hasattr(self.prior, "transform"): self.prior.transform = identity` — but `Distribution.transform` (`distributions/parent.py:17`) is a **concrete base-class method**, so every `Distribution` subclass already has it; the branch only fires for a non-`Distribution` duck-typed prior. Not caught by either linter (the `identity` function *is* referenced, just from unreachable-in-practice code); found via the task's explicit check. |
| Unused tuple-unpack element `axes_sq` | `surrDAMH/post_processing.py:2223` | `fig_sq, axes_sq = self.plot_surrogate_quality()` — `axes_sq` is never used (the very next line correctly uses `fig_sq_test, _ = ...`). Vulture 60%-confidence hit; ruff's `F841` does not flag unused tuple-unpack targets by default, which is why it didn't show up in the ruff pass. Cosmetic (rename to `_`). |
| Write-only flags (possible unfinished logic) | `algorithm_interfaces_local.py::_request_pending`, `::snapshot_count_since_last_update`; `communication.py::PendingRequest.active`, `::max_requests` | See §c — set but never read anywhere in the tree. Worth a second look by whoever wrote the local-runner/communication code (WS2/WS8) in case a guard was meant to consult them and doesn't. |

## (e) toy_examples duplication (line counts + `diff --stat`, no recommendation beyond `05_toy_examples_and_docs.md`)

| Files | Lines | `git diff --no-index --stat` | What differs (one line) |
|---|---|---|---|
| `neural_network_surrogate.py` / `neural_network_surrogate_copy.py` | 70 / 146 | 96 insertions, 20 deletions | `_copy` is a disabled scratch script (`run_sampling = False` hardcoded at line 26), 5-D `SinProdGeneric` solver, 10 DAMH stages instead of the 3-stage MH→DAMH-SMU→DAMH pattern |
| `post_processing_example.py` / `post_processing_with_html_report.py` | 79 / 119 | 47 insertions, 8 deletions | same problem/solver; `_example` calls legacy `samples.html_report(...)`, `_with_html_report` adds `samples.html_report_extended(...)` on the same run |
| `post_processing_with_html_report.py` / `test_html_report_extended.py` | 119 / 73 | 36 insertions, 82 deletions | `test_html_report_extended.py` is a trimmed repro of the same `html_report_extended` demo in its own output dir |
| `typical_example.py` / `typical_example_generic.py` | 127 / 130 | 11 insertions, 8 deletions | `typical_example.py` uses the special-cased 1-D linear-elasticity `SolverSpecExample2` (2 params/1 obs); `_generic` uses the generic nonlinear solver (3 params/3 obs) |

## (f) Proposed order of deletion (smallest / least controversial first)

| Step | Delete | Proof-of-safety command |
|---|---|---|
| 1 | `TAG_READY_TO_RECEIVE`, `TAG_DATA` (communication.py:16-17) | `./run_tests.sh unit && ./run_tests.sh mpi` (no protocol touches these constants) |
| 2 | `Stage.proposal` field, `Proposal.subchain_length` | `./run_tests.sh unit` (covers `Stage`/`Proposal` construction) then a full `mpiexec -n 4 python3 -m mpi4py minimal_example.py` smoke run |
| 3 | `PriorIndependentComponents.sd_approximation`, `closest_point_distance*`, `evaluate_on_a_grid`, `TestData.reduce_size`, `get_loss_on_data` (both torch files), `NeuralNetworkUpdaterMinibatches.save_snapshots`, `add_normal_dist_grid`, `calculate_autocorr_time` (non-`_mean`) | `./run_tests.sh unit` (all have unit coverage of the surrounding classes) |
| 4 | `run_COLLECTOR(surrogate_delayed_init_data=...)` parameter | `./run_tests.sh unit && ./run_tests.sh mpi` (collector start-up tests) |
| 5 | `independent_components.Beta` alias | `./run_tests.sh unit`; also `grep -rn "independent_components.Beta" toy_examples` must stay empty |
| 6 | `Samples.pdf_report`, `load_snapshots`, `_load_snapshot_parameters_and_observations` | needs an explicit author decision first (still-documented legacy report path, §c) — no test currently exercises them either way |
| 7 | `Updater.supports_*_persistence`, `_request_pending`, `snapshot_count_since_last_update`, `communication.active`/`max_requests` | needs author review of intent (possible unfinished logic, §c/d) before any test can prove safety |
| 8 | `torch_perceptron.py` (decision 4) / ~~raw-observation path + `pickled_observations` (decision 5)~~ | already decided but blocked on migrating callers (`toy_examples/neural_network_surrogate*.py`, `sampling_TSX.py`); do last, together with WS6. The decision-5 half is **done 2026-09-17** (no caller had to migrate — no `toy_examples` script set the field; `tests/mpi/test_mpi_transport.py` rewritten in the same change). |
