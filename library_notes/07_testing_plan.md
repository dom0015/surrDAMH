# Testing plan — status

Originally a from-scratch test plan (2026-09-12, when `tests/` held one file exercising an
off-path function). Now a status table: is each planned item actually implemented, and where.
Re-verified 2026-09-18 by reading `tests/` against this list (not by re-running — see the
session's test-suite timings in the reconciliation report for a fresh run).

## Phase 0 — smoke

| Check | Status | Location |
|---|---|---|
| Import smoke test | done | `tests/unit/test_imports.py` |
| Sampler import without torch | done, skipped in this env | `test_imports.py::test_import_process_sampler_without_torch` (needs a torch-free venv) |
| Minimal example runs | covered by MPI I1 | `tests/mpi/test_mpi_basic.py` |

## Phase 1 — unit tests (items 1–40)

All 40 items have a matching, content-verified test (not just a similarly-named stub) —
`tests/unit/test_config_stages.py`, `test_proposals.py`, `test_algorithms_local.py`,
`test_distributions.py`, `test_surrogates.py`, `test_helpers.py`, `test_post_processing.py`.
Two deliberate exceptions:

- **Item 12** (`state_dependent_approximation=True` telescoping invariant): moot — the feature was
  removed on 2026-09-18 (finding 1.1, decision 26), so there is nothing to test.
- **Item 15** (`artificial_acceptance_multiplicator` warning): moot — the field was removed
  entirely rather than warned on; `test_artificial_acceptance_multiplicator_field_removed`
  documents the removal instead.

Two items' predicted-failure text in the original plan turned out to be wrong once written as
a real test, and are corrected here rather than in the test file: item 8 predicted
`matrix_rank(sd_or_cov) ≤ 10` for `period=10` — wrong, `self.samples` accumulates across
periods so the rank eventually reaches full; `test_adaptive_random_walk_reaches_full_rank_covariance`
asserts the real behaviour, and `Stage.adaptive_sample_limit` (G1) is what actually bounds the
history, orthogonally to rank.

## Phase 2 — statistical validation (V1–V8)

| # | Setup | Status |
|---|---|---|
| V1 | plain MH vs. closed form | done, passes |
| V2 | DAMH, wrong fixed surrogate, K∈{1,5,20} | done, passes |
| V3, V3b, V3c, V3d | DAMH-SMU retrained (poly / kd-tree / NN / 3-stage), K∈{1,5,20} | done, all pass — see `10_manual_review_notes.md` §5/§5b/§5d |
| V4 | `state_dependent_approximation=True` | **dropped** — feature removed 2026-09-18 (finding 1.1, decision 26); nothing left to validate |
| V5 | pCN vs RWMH | **not written** — no test found under this or an equivalent name |
| V6 | Hamiltonian with surrogate gradients | partially covered by MPI B8 (Hamiltonian-in-MH), not by a dedicated validation test |
| V7 | `initial_sample_type="prior"` reproducibility | covered, different test names: `tests/test_runner_local.py::test_prior_initial_sample_is_reproducible_across_runs` / `::test_prior_initial_sample_differs_between_ranks` |
| V8 | adaptive stage converges to the requested target rate | **not written** |

`09_improvement_plan.md` and `10_manual_review_notes.md` previously claimed "V1–V3, V5–V8" or
"V5/V6 validations pass" — both wrong; corrected here and in those files. Only V1–V3d exist.

## Phase 3 — MPI integration and deadlock regressions (I1–I15)

| # | Status | Location |
|---|---|---|
| I1–I9, I11 | done | `tests/mpi/test_mpi_basic.py`, `test_mpi_hangs.py`, `test_mpi_surrogate.py`, `test_mpi_transport.py` |
| I10 (report: pool vs local mode, same sections) | **not written** | ties to finding 2.7's open half |
| I12 (2-rank protocol tests for `communication.py`) | **not written** | `09` WS8's own acceptance criterion, never done — ties to finding 2.10's open half |
| I13 (`MPI.TAG_UB` check) | **not written** | ties to finding 2.11 |
| I14/I15 (RSS / busy-wait measurement) | **not written** | diagnostics, not correctness tests; ties to findings 4.1/4.2 |

## Phase 4 — example smoke matrix

Not a pytest suite by design. Covered manually each time the maintained-examples set changes;
see the README.md table for the current list and `docs/running.md` for exact commands.

## Phase 5 — environment compatibility

`numpy.matlib` removal (`test_lhs_normal_no_longer_imports_numpy_matlib`) and
`torch.load(weights_only=True)` (exercised by the checkpoint round-trip tests) are covered.
pandas-3 copy-on-write/dtype warnings and mpi4py-4 `irecv`/`waitany`/`Cancel` semantics have no
dedicated test — only exercised implicitly by the rest of the suite passing.
