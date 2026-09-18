# library_notes — structured review of surrDAMH

Read-only study of the `surrDAMH` package and `toy_examples/` as of **2026-09-12**
(branch `working_Kuba`, working tree with uncommitted changes). Produced by reading the source;
**no code was run** (one exception disclosed below). `out_*` directories and the real-experiment
folders (`TSX_complete_experiment_*`, `hpcse26_nn_training`, `sampling_test_Hamilton`) were not
opened. No existing file in the repository was modified; everything is in this folder.

## Files

| File | Contents |
|---|---|
| `00_overview.md` | Architecture: process topology, sampler control flow, MH/DAMH mechanics, collector and solver pool, MPI tags, surrogate contract, on-disk layout, environment facts |
| `01_mcmc_algorithms.md` | Algorithms, proposals, stages, seeds, bookkeeping — 30 findings (A1–A30), 18 improvements, 12 tests, 10 questions |
| `02_mpi_processes_and_communication.md` | Roles, full message-protocol table, lifecycle/termination, 25 findings (M1–M25) |
| `03_surrogates_and_distributions.md` | Updater/Evaluator conformance matrix, per-surrogate notes, distributions, helpers — 23 findings (S0–S22) |
| `04_post_processing.md` | `Samples` structure map, writer↔reader format table, 10 findings (P1–P10), test ideas |
| `05_toy_examples_and_docs.md` | Example inventory, API conformance, packaging, hygiene, canonical example set, smoke matrix |
| `06_findings_consolidated.md` | **Start here.** Deduplicated, prioritized (Tier 0–6), with a manager-verified column, list of things confirmed correct, uncommitted changes, order of attack |
| `07_testing_plan.md` | Phased test/validation plan: smoke → unit → statistical validation → MPI/deadlock → examples → environment |
| `08_safe_changes_plan.md` | Changes that can be made now without further tests (crash fixes, fail-fast guards, latent bugs, dead code, packaging), grouped into commits, with what is deliberately deferred |
| `09_improvement_plan.md` | Medium-refactor roadmap (~5–6 weeks): 12 workstreams with deliverables and acceptance criteria, week-by-week sequencing, **decisions recorded 2026-09-13 and 2026-09-16/17**, future work, risks, definition of done — **updated 2026-09-17 with a DONE/PARTIAL/NOT STARTED status and ✔/✘ per bullet for every workstream** |
| `10_manual_review_notes.md` | Running review log kept during the refactor: which uncommitted files belong to which workstream (§1), per-change behaviour evidence (§2.1–§2.18), pre-existing bugs the new tests pin but do not fix (§3), validation results (§5/§5b/§5c), MPI checks (§6), spec drafts and decisions (§7a/§7b), and **§8, the single authoritative "what is still open" list**, added 2026-09-17 |
| `11_output_format_v2_spec.md` | Output format v2 specification — **implemented 2026-09-17** (WS9a/WS9b); documents the v1→v2 diff, the `read_run()`/`RunData` contract, and the deviations decided while implementing |
| `12_evaluator_contract_spec.md` | `Evaluator`/`Updater` contract specification (WS6) — **implemented 2026-09-17**; conformance table of the pre-WS6 state plus the contract now in force and the decisions applied instead of the draft |
| `13_dead_code_report.md` | `ruff`+`vulture` dead-code inventory (2026-09-16), grep-verified per symbol; the author's 2026-09-17 decision was **not to delete any of it** — the report stays a reference of what is deliberately unused/superseded, not a to-do list |

## Status changes after the review (2026-09-13)

- `TSX_complete_experiment_1..4`, `sampling_test_Hamilton` and all 64 `out_TSX*` directories were
  moved to `TSX_experiments_archived/` (added to `.gitignore`); they and the hpcse26 results are
  obsolete. Only `toy_examples/` is maintained.
- Author decisions on the eight open questions are in `09_improvement_plan.md` §3 (two deferred
  to §4 Future work). Notes 06–08 were annotated accordingly; findings that referenced the
  archived scripts are kept for the record.

## Status changes (2026-09-17)

- **`08_safe_changes_plan.md` §G1–G6 and A30 are decided and implemented.** The adaptive `Stage`
  fields are wired now (G1), the adaptive `Allreduce` shape mismatch is gone (G2), `lhs_normal`
  returns the maximin-best design (G3), initial samples are per-rank seeded and reproducible
  (G4), block sub-proposals are re-seeded per rank and per stage (G5), CSV writers are
  line-buffered (G6), and a state carried across a stage boundary is counted once in the
  `multiplicity`/weight column (A30). Evidence per item: `10_manual_review_notes.md` §2.13; one
  CHANGELOG entry per item. Only §G8 (`raw_data` ragged rows) is superseded rather than deferred —
  output format v2 (below) rectangularized `raw_data` outright.
- **WS6 (surrogates), WS8 (MPI robustness/raw-path removal) and WS9 (output format v2 +
  post-processing split) were applied 2026-09-17**, all **breaking**: `NeuralNetworkUpdaterBasic`/
  `torch_perceptron.py` deleted; `Configuration.pickled_observations` deleted; every sampling CSV
  gained a header and the multiplicity/weight column was renamed; `run_manifest.json`'s
  `format_version` is now 2 and enforced (no converter for old `out_*` directories, author
  decision 6). See `CHANGELOG.md` (reorganized 2026-09-17 into Breaking/Behaviour/Bug
  fixes/Additions/Removed/Documentation) for the full, deduplicated list.
- **WS5 (sampling-script structure) and WS11 (documentation) are done**: `docs/` now has 7 pages
  plus an index; `toy_examples/template_experiment.py` is the canonical script; the four
  duplicate example scripts were archived (`neural_network_surrogate_copy.py`,
  `typical_example_generic.py`, `post_processing_example.py`, `test_html_report_extended.py`).
- **WS10 (dead code): the author decided *not* to delete dead code** (2026-09-17,
  `10_manual_review_notes.md` §7b) — `13_dead_code_report.md`'s "safe to delete" table is a
  reference only. What *was* removable was archived instead: `hpcse26_nn_training/`, `talk/`, the
  root-level TSX analysis scripts and scratch files (`summarize_tsx2_results.py`,
  `analyze_tsx2_*.py`, `KL_mode_selection_report.md`, etc.) all now live under
  `TSX_experiments_archived/` (confirmed absent from the repo root by listing).
- **09_improvement_plan.md, 06_findings_consolidated.md and 10_manual_review_notes.md were
  re-checked against the code and the test suite on 2026-09-17** (this pass): every workstream
  has a DONE/PARTIAL/NOT STARTED line, every finding in 06 has a status, and 10's §8 is now the
  single authoritative "still open" list.

## Headline results (re-verified 2026-09-17, superseding the review-time claims below)

- **The package imports and the maintained examples run.** `python -c "import surrDAMH"` and
  `mpiexec -n 4 python3 -m mpi4py minimal_example.py` both succeed; the three Tier-0 start-up
  crashes described below (predicted from reading, 2026-09-12) were fixed in WS0 (commit
  `175059e`) and are gone from the current tree.
- **Test suite, freshly run**: `./run_tests.sh unit` → **303 passed, 1 skipped** (10.2 s);
  `./run_tests.sh validation` → **10 passed** (73.0 s); `./run_tests.sh mpi` → **32 passed, 0
  xfailed** (157.9 s). Coverage is no longer "effectively zero": ~345 tests across unit,
  statistical-validation and MPI-integration/deadlock suites, plus 8 statistical validations
  (V1–V3, V3b, plus the B1–B8 MPI posterior-vs-closed-form checks in
  `10_manual_review_notes.md` §5c).
- **DAMH/DAMH-SMU correctness**: validated against the closed-form posterior on the Gaussian toy
  for `subchain_max_length ∈ {1, 5, 20}`, with and without SMU
  (`state_dependent_approximation=False`) — see `10_manual_review_notes.md` §5/§5b. The one
  remaining posterior-correctness gap, `state_dependent_approximation=True` with
  `subchain_max_length > 1` (finding 1.1), is unfixed by design: default is `False`, `True` warns
  and is excluded from tests, theory check deferred to future work.
- **No known hang remains untreated**: the collector-poll hang (2.1), the DAMH/Hamiltonian
  first-stage deadlock (2.2), and the raw-mode negative-solver-tag MPI error (2.4, path since
  removed) are all fixed and covered by `tests/mpi/test_mpi_hangs.py`/`test_mpi_transport.py`;
  any rank's uncaught exception now aborts the whole job within seconds (WS8 fail-loud).
- **Confirmed correct** (unchanged since the original review): the DAMH acceptance formula, pCN,
  HMC integrators, the evaluator handshake, barrier accounting; all now also protected by
  regression tests rather than only by reading.
- **What is still open**: see `10_manual_review_notes.md` §8 for the authoritative list —
  broadly, a handful of MPI/surrogate performance items (1 GiB `irecv` buffer, busy-wait sleeps,
  `TAG_UB` check, RBF/polynomial hardening, surrogate-quality ESS) and three items deliberately
  deferred to future work by author decision (`state_dependent_approximation=True` theory,
  adaptive-target-in-DAMH semantics, `HamiltonianInfinite` parametrisation).

**Original review-time headline (2026-09-12, kept as a record — see above for the current
state):**

- **The working tree is predicted not to import**: `core.py` uses `Iterable`/`Any` in annotations
  without importing them (Tier 0). Two more start-up crashes: a stray `torch.mtia` import in the
  sampler and a wrong `TestData` call in `core.py`.
- **Posterior-affecting issues**: `state_dependent_approximation` and DAMH-SMU are only correct for
  `subchain_max_length == 1`; the adaptive `Stage` fields are never wired to the proposal; the
  default of `state_dependent_approximation` was flipped in the uncommitted diff.
- **Hangs**: collector never checks stop signals before the first surrogate exists; a DAMH or
  Hamiltonian first stage waits for a surrogate that nobody can train; float32 continuation samples
  hit a float64 MPI buffer; negative solver tags used as MPI tags in raw mode.
- **Confirmed correct**: the DAMH acceptance formula (the code comment calling it a "hot fix" is
  misleading), pCN, HMC integrators, the evaluator handshake, barrier accounting, output formats.
- **Test coverage** is effectively zero; the one existing test covers an unused function.

Counts: 5 subsystem notes, ~100 distinct findings after deduplication, 40 unit-test ideas,
8 statistical validations, 15 MPI integration checks.

## Method and limits

- Manager (this session) read `core.py`, `configuration.py`, `stages.py`, `process_*.py`,
  `modules/algorithms.py`, `proposals.py`, `communication.py`, `algorithm_interfaces*.py`,
  `monitoring.py`, `continuation.py`, `tools.py`, `test_data.py`, `solvers.py`,
  `surrogates/parent.py`, `nearest_kdtree.py`, parts of `torch_perceptron_minibatches.py`,
  `distributions/{parent,normal,independent_components}.py`, `Gaussian_process.py` (part), and all
  package `__init__` files, and verified the high-severity claims of the five reviewers against
  those lines (the **Checked** column in `06_findings_consolidated.md`).
- Five reviewers (three on algorithms / MPI / surrogates, two on post-processing / examples) wrote
  notes 01–05 under the same read-only rules.
- **Disclosure**: the surrogate reviewer ran one Python one-liner (`numpy.linalg.LinAlgError.__mro__`)
  to confirm that `LinAlgError` subclasses `ValueError`. Nothing from `surrDAMH` was executed. That
  note also states "scipy is not installed"; this refers to whichever `python3` was on the
  reviewer's PATH — the `/dolfinx-env` site-packages, where surrDAMH is installed editable, lists
  scipy 1.16.3 (see `00_overview.md` §11).
- Findings marked *suspected* depend on runtime behaviour (MPI implementation, mpi4py buffer
  semantics, scipy exception types) and need the checks in `07_testing_plan.md`.
- Line numbers refer to the working tree at review time and will drift after edits.
