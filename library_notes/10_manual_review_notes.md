# Manual review notes (maintained by Claude during the refactor)

Running list of things the author should check by hand, with what was verified automatically
so it does not need re-checking. Newest entries at the bottom of each section. Items are
removed or struck through once the author confirms them.

Conventions: **[verify]** = run/inspect something on your side; **[decide]** = needs an author
decision; **[read]** = just be aware of it; **[bug]** = pre-existing defect surfaced by tests,
not fixed. Finding IDs refer to `06_findings_consolidated.md`; G-items to `08_safe_changes_plan.md` §G.

---

## 1. Uncommitted work waiting for your review (state on 2026-09-16, git status refreshed 2026-09-17)

`git log --oneline -3` now shows **two** commits since the review started:
`1ded39e` "refactor: internal changes 2" (2026-09-17) on top of `175059e` "refactor: internal
changes, no functional changes" (2026-09-16, WS0 §A–§F). **Everything below `1ded39e` is still
uncommitted** — confirmed by `git status --short` run fresh in this session. The per-item
narrative table below (written 2026-09-16, before WS6/WS8's raw-path removal/WS9/G1–G6+A30/WS5
landed) is kept for its verification notes; it is **not** a complete list of what is uncommitted
today. Use the table immediately below instead to commit in order — it is grouped by workstream
from the actual `git status --short` output of 2026-09-17.

### Files to commit, grouped by workstream (`git status --short`, 2026-09-17)

| Workstream | Modified (`M`) | Deleted (`D`, tracked) | New (`??`, untracked) |
|---|---|---|---|
| WS0 (fail-fast guards, G1–G6, A30 — spread across many files, see §2.13) | `surrDAMH/modules/lhs_normal.py`, `surrDAMH/modules/proposals.py`, `surrDAMH/stages.py`, `surrDAMH/distributions/normal.py`, `surrDAMH/distributions/independent_components.py`, `surrDAMH/distributions/parent.py` | — | `surrDAMH/modules/seeds.py` |
| WS2 (local runner) | `surrDAMH/process_SAMPLER.py`, `surrDAMH/modules/algorithm_interfaces.py`, `surrDAMH/modules/algorithm_interfaces_local.py`, `surrDAMH/runner_local.py` | — | `surrDAMH/modules/proposal_builder.py`\* | 
| WS3 (DAMH kernel) | `surrDAMH/modules/algorithms.py`, `surrDAMH/surrogates/parent.py` | — | — |
| WS4 (reproducibility/manifest) | `surrDAMH/core.py`, `surrDAMH/modules/manifest.py`, `surrDAMH/configuration.py` (also WS0/WS8) | — | — |
| WS5 (script structure) | `surrDAMH/solver_specification.py`, `surrDAMH/solvers.py`, `toy_examples/toy_example_hamilton.py`, `toy_examples/sampling_diffusion_grf.py`, `toy_examples/solver_examples/solver_spec_examples.py`, `toy_examples/grf_diffusion.py`, `toy_examples/tunnel_with_subdomains.py`, `toy_examples/wrapper.py`, `README.md` | — | `surrDAMH/modules/describe.py`, `surrDAMH/modules/surrogate_restart.py`, `toy_examples/template_experiment.py`\*\* |
| WS6 (surrogates) | `surrDAMH/surrogates/__init__.py`, `surrDAMH/surrogates/nearest_kdtree.py`, `surrDAMH/surrogates/polynomial_sklearn.py`, `surrDAMH/surrogates/rbf_scipy.py`, `surrDAMH/surrogates/reuse.py`, `surrDAMH/surrogates/torch_perceptron_minibatches.py`, `toy_examples/neural_network_surrogate.py`, `toy_examples/sampling_TSX.py` | `surrDAMH/surrogates/torch_perceptron.py` | — |
| WS8 (MPI robustness / raw-path removal) | `surrDAMH/modules/communication.py`, `surrDAMH/process_CHILD.py`, `surrDAMH/process_COLLECTOR.py`, `surrDAMH/process_SOLVER.py` | — | — |
| WS9 (output format v2 + post-processing split) | `surrDAMH/modules/monitoring.py`, `surrDAMH/modules/manifest.py` (shared with WS4), `surrDAMH/__init__.py`, `toy_examples/post_processing_with_html_report.py` | `surrDAMH/post_processing.py` (became the package below) | `surrDAMH/modules/run_data.py`, `surrDAMH/post_processing/` (package) |
| WS10 (dead code / duplicate examples / archiving) | — | `toy_examples/neural_network_surrogate_copy.py`, `toy_examples/post_processing_example.py`, `toy_examples/test_html_report_extended.py`, `toy_examples/typical_example_generic.py`, `TSX_complete_experiment_1/*` (11 files), `hpcse26_nn_training/*` (6 files), `sampling_test_Hamilton/sampling_hamilton.py` | — |
| WS11 (docs / changelog) | `README.md` (shared with WS5) | — | `CHANGELOG.md`, `docs/` (7 pages + index) |
| WS1 (tests) | — | — | `pytest.ini`, `run_tests.sh`, `tests/` (whole tree: `conftest.py`, `helpers_statistics.py`, `test_best_fit_ranking.py`, `test_runner_local.py`, `unit/`, `mpi/`, `validation/`) |
| Not a workstream (environment / project meta) | `.devcontainer/devcontainer.json` | — | `CLAUDE.md`, `library_notes/` (this review) |

\* `surrDAMH/modules/proposal_builder.py` also carries the WS7 fail-fast guards (§2.9) — shared
between WS2 and WS7. \*\* `toy_examples/template_experiment.py` is WS5's canonical script but was
introduced together with `toy_examples/solver_examples/` additions.

Suggested commit order (dependency-driven, smallest first): **(1)** WS1 test infrastructure
(`pytest.ini`, `run_tests.sh`, `tests/`) so every subsequent commit can be verified against it;
**(2)** WS0 (lhs/proposals/stages/distributions + `seeds.py`); **(3)** WS2+WS7
(`proposal_builder.py`, `process_SAMPLER.py`, `algorithm_interfaces*.py`, `runner_local.py`);
**(4)** WS3 (`algorithms.py`, `surrogates/parent.py`); **(5)** WS4 (`core.py`, `manifest.py`,
`configuration.py`); **(6)** WS8 (`communication.py`, `process_CHILD/COLLECTOR/SOLVER.py`);
**(7)** WS6 (surrogates + the two migrated toy examples + `torch_perceptron.py` deletion);
**(8)** WS9 (`post_processing/` package, `run_data.py`, `monitoring.py`, `__init__.py`); **(9)**
WS5 (`describe.py`, `surrogate_restart.py`, `solver_specification.py`, the two advanced examples,
`template_experiment.py`); **(10)** WS10 (the four duplicate-example deletions + the archived
directories, which need no `git add` since they are now git-ignored — `git status` will simply
stop listing them as deleted once the deletions themselves are committed); **(11)** WS11
(`CHANGELOG.md`, `docs/`, `README.md`); **(12)** `CLAUDE.md`/`library_notes/` if you want the
review versioned. This supersedes the WS0/WS1/WS2 split suggested in the `[verify]` note further
down this section, which predates WS6/WS8's raw-path removal and WS9/WS5/WS10.

### Narrative table (written 2026-09-16, before the 2026-09-17 work — kept for its verification notes)

WS0 was committed by you as `175059e`. Everything below is still unstaged/untracked:

| Path | What it is | Status |
|---|---|---|
| `surrDAMH/process_SAMPLER.py` (modified) | 53-line inline proposal block replaced by one `build_proposal(...)` call | verified byte-identical sampler output before/after (deterministic MH script, `mpiexec -n 1`, md5 of `rank0000.csv` equal) |
| `surrDAMH/modules/proposal_builder.py` (new) | MPI-free `build_proposal(stage, conf, prior, seed, prev_cov=None, stage_index=None)` | all six proposal branches compared against the old inline code: identical types, `sd_or_cov`, and 10 draws each |
| `surrDAMH/runner_local.py` (new) | `run_local(conf, prior, likelihood, stages, solver, updater=None, evaluator=None) -> SamplingResult` | reproduces chain 0 of an MPI run bit-for-bit (automated test + manual `mpiexec -n 1` vs plain `python`, same md5) |
| `surrDAMH/modules/algorithm_interfaces_local.py` (modified) | `evaluator_is_available()` now means "a *new* evaluator is pending" (see §2.1) | unit-tested; one MPI configuration unverified, see §2.1 |
| `surrDAMH/modules/algorithms.py` (modified, WS3) | sub-chain surrogate freeze (§2.5); `sample_carried_to_next_stage()` helper (§2.6); docstrings on the DAMH sub-chain methods; `solver_tag` documented on `Sample` | K=1 SMU output byte-identical before/after; V2/V3/V3b pass at K=1,5,20 after the change (§5) |
| `surrDAMH/process_SAMPLER.py`, `surrDAMH/runner_local.py` (WS3 hunk) | stage hand-over goes through `sample_carried_to_next_stage(alg_instance.current, stage)` in both runners | `tests/test_runner_local.py::test_stage_after_use_only_surrogate_starts_from_exact_observations`; MPI I5 with 4 stages |
| `surrDAMH/surrogates/parent.py` (modified, WS3) | `SurrogateAsSolver.get_observations` returns `(no_observations,)` (A20 fix, §2.7) | 6 parametrised shape tests; MPI I5 now runs with `no_observations=2` |
| `surrDAMH/stages.py` (modified, WS3) | `adaptive_target_rate` comment: DAMH targets the *outer* rate; field currently not wired (G1) | comment only |
| `surrDAMH/modules/manifest.py` (new, WS4) | `build_run_manifest` / `write_run_manifest` / `finalize_run_manifest` → `<output_dir>/sampling_output/run_manifest.json` (§2.8) | `tests/unit/test_manifest.py` (7 tests) incl. two identical `run_local` runs → identical samples + manifests |
| `surrDAMH/core.py` (modified, WS4) | rank 0 writes the manifest before role dispatch and finalizes it after `run_SAMPLER` returns; captures the *requested* `use_surrogate_gradients` before `_configure_surrogate_gradients` may flip it | `mpiexec -n 2 minimal_example.py` → manifest with `runner="mpi"`, correct MPI layout |
| `surrDAMH/runner_local.py` (WS4 hunk) | writes + finalizes the manifest after the stage loop (with per-stage counters) | unit tests |
| `surrDAMH/modules/continuation.py` (modified, WS4) | `load_last_samples` warns (`RuntimeWarning`) when more chains are saved than requested; parameter-count `ValueError` unchanged | `tests/unit/test_continuation.py` (4 tests) |
| `surrDAMH/modules/proposals.py` (modified, WS7) | `GaussRandomWalk_adaptive.adapt()`: degenerate acceptance weights/covariance → skip the period + `RuntimeWarning` instead of `ZeroDivisionError` (§2.9) | non-degenerate 30-period sequence bit-identical before/after (hard-coded 2×2 regression test) |
| `surrDAMH/modules/proposal_builder.py` (WS7 hunk) | pCN requires `Normal`/`PriorIndependentComponents`; single `needs_gradients ⇒ use_surrogate_gradients` check for all proposal types (§2.9) | 8 new `build_proposal` tests |
| `surrDAMH/stages.py::stage_name()` + `core.py`, `process_SAMPLER.py`, `runner_local.py` (manager, after WS4) | one stage-naming function; `SamplingFramework.run()` assigns names on all ranks before the manifest is written → MPI manifest now has `stages[i].name` | `mpiexec -n 2 minimal_example.py` → manifest `stages: ["alg0000_MH"]`; all suites unchanged |

| `surrDAMH/core.py` `_run_role` + `process_CHILD.py` (WS8) | any uncaught exception on any rank / spawned child → traceback + `MPI.COMM_WORLD.Abort(1)` (§2.10) | new MPI tests: solver raising in the spawned child → exit 9 in 3.4 s (was: hang); on the sampler rank under plain `mpiexec python` → exit 1 in 2 s (was: hang) |
| `surrDAMH/modules/communication.py` `TAG_INITIAL_SURROGATE=6` + `process_COLLECTOR.py` / `process_SAMPLER.py` (WS8) | start-up handshake: collector tells each sampler whether it can provide an evaluator; sampler raises if its first stage needs one (§2.11, replaces the B2 print) | I4 flipped: fast abort with the new message; I3 (MH first stage) unaffected; new positive test DAMH-first + `initial_snapshots` |
| `surrDAMH/process_COLLECTOR.py` I9 fix (WS8) | `initial_snapshots` counted once (was 2N) (§2.12) | I9 xfail flipped to pass: N=8 → `snapshots_total` 8 |

| `toy_examples/template_experiment.py` (new, WS5) + `LinearGaussianSolver`/`SolverSpecLinearGaussian` in `toy_examples/solver_examples/` | 97-line canonical script: problem → prior/likelihood → solver → surrogate → stages → run → report, one comment per knob | `mpiexec -n 4` → exit 0, 3 stages, manifest + `report_extended.html` + `summary.csv`; MH-only variant runs with `-n 1` |
| `docs/*.md` (8 pages, new, WS11), `CHANGELOG.md` (new), docstrings in 16 `surrDAMH/` files | user-facing API documented as it is today; planned WS6/WS9 changes referenced via specs 11/12; CHANGELOG lists every behaviour-affecting change with its evidence pointer | AST comparison vs `HEAD` with docstrings stripped: the 12 files no other workstream touched are code-identical; unit gate unchanged (149) |
| `tests/mpi/test_mpi_posterior.py` + `tests/helpers_statistics.py` (new) | MPI transport bit-exactness (A1/A2) + posterior-vs-closed-form under `mpiexec` for 8 configurations (B1–B8), see §5c | full MPI suite 30 passed / 3 xfailed; `tests/validation` still 10 passed on the shared helpers |
| `toy_examples/neural_network_surrogate_copy.py` **deleted** (2026-09-17, your decision "maintain only `neural_network_surrogate.py`") | disabled scratch duplicate (`run_sampling=False`, finding E7); referenced only from `library_notes/` | unstaged deletion; `tests/unit/test_imports.py` passes |
| `surrDAMH/modules/tools.py` + `modules/manifest.py` (manager, 2026-09-17) | root-cause fix of the circular import (`from __future__ import annotations` in `tools.py`; `manifest.py` imports `ensure_dir` normally again instead of lazily) | fresh-interpreter imports from 5 entry points OK; unit gate unchanged |
| `run_tests.sh` (new, WS1 leftover) | `./run_tests.sh [unit\|validation\|mpi\|all]`, uses `/dolfinx-env/bin/python3` if present (`PYTHON=` overrides) | run locally, see below |

Suite totals after the WS8 slice (2026-09-16, manager re-run after moving `ABORT_GRACE_SECONDS`
from `core.py` to `modules/communication.py` so spawned children don't import the whole
framework): `pytest tests/mpi -q -m mpi` → **20 passed, 3 xfailed** (77 s, 0 orphans);
`pytest tests -q -m "not mpi"` → **160 passed, 2 skipped, 1 xfailed** (77 s). The I7 raw-path
"Invalid tag" case now ends with a fast nonzero exit instead of a hang (fail-loud side effect).
**[decide]** WS1 also asked for a CI stub (GitHub Actions or Makefile target: `unit` on push,
`validation`+`mpi` nightly). Not written — I can't verify a CI runner here. The remote is GitHub
(`dom0015/surrDAMH`, no `.github/` yet), so a workflow calling `./run_tests.sh unit` on push and
`./run_tests.sh all` nightly is the natural stub; it would need `apt install mpich` + `pip install
-e .` in the runner and would run without FEniCSx (fine: no test needs it). Say so and I'll add it.
**Decided 2026-09-17, §7b (decision 20): no CI** — no `.github/` workflow; `run_tests.sh` stays
the entry point.

Suite totals after WS4 + WS7 (2026-09-16, manager re-run): `pytest tests -q -m "not mpi"` →
**160 passed, 2 skipped, 1 xfailed** (77 s); `pytest tests/mpi -q -m mpi` → **16 passed, 4 xfailed**
(97 s), 0 orphans.
| `pytest.ini`, `tests/conftest.py`, `tests/unit/*`, `tests/validation/*`, `tests/test_runner_local.py` | WS1 test suite | `pytest tests -q -m "not mpi"` → 131 passed, 2 skipped, 1 xfailed (~70 s); `-m unit` alone ~8 s |
| `tests/mpi/*` (5 files) | WS1 MPI integration/deadlock tests (I1–I9, I11), subprocess `mpiexec` under timeout | `pytest tests/mpi -q -m mpi` → 16 passed, 4 xfailed, ~2 min, run 3× (agent 2×, manager 1×), stable, 0 orphaned processes |

**[verify]** Review and commit when convenient. Suggested split: (1) `proposal_builder.py` +
`process_SAMPLER.py`; (2) `runner_local.py` + `algorithm_interfaces_local.py` +
`tests/test_runner_local.py`; (3) `pytest.ini` + `tests/`.

**[decide]** `pytest tests` with no `-m` now takes ~3–4 min (unit + validation + mpi). Options:
leave as is (explicit, nothing silently skipped — current choice), or add
`addopts = -m "not mpi"` (or `"not mpi and not validation"`) to `pytest.ini` so the default
run is the fast gate and the slow markers are opt-in, matching the 09 plan ("unit on every
push, validation + mpi nightly"). Not changed by me.
**Decided 2026-09-17, §7b (decision 21): `addopts = -m "not mpi"` added** — bare `pytest` now runs
unit+validation only; `./run_tests.sh mpi` / `pytest -m mpi` for the MPI suite.

## 2. Behaviour changes to check against your own configurations

### 2.1 `evaluator_is_available()` semantics (algorithm_interfaces_local.py) — **[verify]**
Old: `True` whenever *any* evaluator exists. New: `True` only when a *new* evaluator is pending
(matches the MPI class `CommEvaluator_sampler`, which reports the `irecv` status). Effect in
`Algorithm_DAMH._refresh_surrogate_evaluator_if_needed`: with a fixed evaluator the sub-chain now
takes the 1-surrogate-call branch of `_evaluate_surrogate_transition` instead of the 3-call
branch on every step. Acceptance decisions should be identical (same evaluator → recomputed and
retained values are bit-identical), only the surrogate cost drops ~3×.

**Resolved 2026-09-16 — no manual check needed.** Which configurations are affected at all:
`evaluator_is_available()` of the *local* providers is only consulted by
`Algorithm_DAMH._refresh_surrogate_evaluator_if_needed`, and only when
`stage.surrogate_model_updates=True`. The MPI provider (`MpiEvaluatorProvider`, every
`use_collector=True` run) is untouched. So the only pre-existing configuration affected is
`use_collector=False` + a fixed `surrogate_evaluator=` passed to `SamplingFramework` + a DAMH stage
with `surrogate_model_updates=True` (the default for DAMH). There, exactly one surrogate version
ever exists, so "the same version at each time" holds trivially; the question was only whether
re-scoring with it changed any number. It does not: `_get_surrogate_observations` evaluates each
point as its own 1-row call (`algorithms.py:290-291`), so the old "changed" branch recomputed the
very same single-row evaluations that were already stored.

Empirical proof (scratch script, `run_SAMPLER` in-process at world size 1 = chain 0, fixed initial
sample, DAMH-SMU stage, 300 evaluations, old semantics monkeypatched back in vs new code):

| fixed evaluator | K | evaluator calls old → new | `samples`/`notes`/`subchain_stats` |
|---|---|---|---|
| degree-1 polynomial (1.3·A) | 1 / 5 / 20 | 3439→1147 / 2682→1916 / 6645→6041 | byte-identical |
| torch MLP (`NeuralNetworkUpdaterMinibatches`) | 1 / 5 / 20 | 3016→1006 / 2633→1881 / 6623→6021 | byte-identical |

`run_local()` and `LocalSurrogateManager` are new code (no "before"). Conclusion: identical
sampling process and identical surrogate versions for every configuration; the only effect is
~3× fewer surrogate calls per outer step at K=1.

### 2.2 `run_local()` continuation — **done 2026-09-16** (you asked for it)
`run_local` now calls `save_last_sample(conf, stage.name, 0, ...)` after every stage exactly like
`run_SAMPLER`, so `sampling_output/last_sample/<stage>/rank0000.npz` exists and a later run (local
or MPI chain 0) can use `initial_sample_type="continued", continued_from_dir=...`. Test:
`tests/test_runner_local.py::test_run_local_writes_last_sample_and_can_continue_from_it` — run A
writes the file (float64, equals `final_sample.parameters`), run B starts bit-for-bit from it.
Unit gate: 149 passed.

### 2.3 `build_proposal` takes an extra `stage_index` — **[read]**
Only used in the assertion message `"proposal sd/cov not specified for stage {i}"`; both
callers pass `i`. Message text unchanged.

### 2.5 Sub-chain surrogate freeze (WS3, finding 1.2) — **[verify]** / **[decide]**
`Algorithm_DAMH._propose_new_sample_using_subchain` now calls
`_refresh_surrogate_evaluator_if_needed()` **once before** the sub-chain loop (re-scoring
`self.current` and the sub-chain start only if the evaluator really changed), instead of at the
top of every inner step. `correction_log_ratio` therefore telescopes under a single surrogate,
as the derivation in `run()` requires.

What changes: the **sample stream of every DAMH stage with `surrogate_model_updates=True` and
`subchain_max_length > 1`** (different proposals are scored by a different surrogate than
before → different accept/reject sequence, different per-run counts). What does not change,
verified: K = 1 SMU runs are byte-identical (`[MH, DAMH-SMU K=1]`, kd-tree updater swapped 200+
times, all five output files `cmp`-identical); V2 (fixed surrogate) and all MH stages are
untouched by construction. Statistically neutral by theory and by V3/V3b (§5).
MPI protocol unchanged: one poll per outer step, at most one outstanding evaluator request.

**[decide]** `09_improvement_plan.md` §5 suggested keeping a `legacy_subchain_refresh=True`
switch for one release to compare against old K>1 SMU runs. Not added (CLAUDE.md: no feature
flags; V3b is the before/after comparison). Say so if you want it — it is a ~5-line addition.
**Not revisited by 2026-09-17 — still open**, listed in §4 future work item 5 of `09_improvement_plan.md`;
no author decision recorded against it specifically (unlike the items resolved in §7b).

### 2.6 Exact re-evaluation after a `use_only_surrogate` stage (WS3, finding A11) — **[verify]**
Both runners now hand the carried sample through `algorithms.sample_carried_to_next_stage()`,
which drops `observations`/`log_likelihood`/`*_approx` when the finished stage had
`use_only_surrogate=True`, so `_prepare_run` re-evaluates it with the full model (one extra
solver call per chain per such boundary). Before, the next exact stage compared exact `log L(y)`
against a *surrogate* `log L(x)` in its first ratio and wrote a surrogate log-posterior as its
first row. Regression test fails when the helper is disabled.
**New failure mode:** if the full model fails (`solver_tag < 0`) exactly at that carried state,
the next stage now raises `RuntimeError` (B4) instead of silently continuing on surrogate
values. Intended fail-loud behaviour, but a previously "working" configuration can now abort.

### 2.7 `SurrogateAsSolver.get_observations` shape (WS3, finding A20) — **[read]**
Now returns `(no_observations,)` like every real `Solver`, regardless of whether the evaluator
returns `(1, q)` or a flattened `(q,)`. Fixes the `ValueError: shapes (1,2) and (1,2) not
aligned` crash of `use_only_surrogate` with `no_observations > 1`. No `Evaluator.__call__`
contract changed (torch evaluators still return flattened output — finding 3.5, WS6).

### 2.8 Run manifest (WS4) — **[read]** / **[decide]**
Every run now writes `sampling_output/run_manifest.json` (rank 0 / local runner): `manifest_version`,
`format_version` (1 = today's layout; WS9 bumps to 2), `surrdamh_version`, `runner`, timestamps,
`hostname`, Python + numpy/scipy/mpi4py/torch/sklearn versions, `git` {commit, dirty, branch},
`configuration` (incl. *requested* vs *effective* `use_surrogate_gradients`), `stages`,
`prior`/`likelihood` summaries, `surrogate` (updater/evaluator class + scalar hyper-parameters),
`solver`, `seeds` (the literal formula + per-rank/per-stage `seed0`, proposal and algorithm seeds;
`initial_sample_type="prior"` flagged non-reproducible, finding 1.9/G4 — **superseded 2026-09-17**:
since G4 (§2.13) it is seeded and the manifest now reports `initial_sample_reproducible: true`
for every `initial_sample_type`), `mpi` layout,
`environment` (`OMP/MKL/OPENBLAS_NUM_THREADS`, `torch_num_threads` if torch is loaded),
`unverified_options`, `continued_from` (with the source run's manifest embedded if it has one).
Pure addition — no existing file or reader changes; failures only print a warning.
- **[read]** Stage names are now assigned by `stages.stage_name(stage, i)` in
  `SamplingFramework.run()` on every rank *before* role dispatch (and re-derived identically in
  the sampler), so the MPI manifest records them. Side effect, intentional: an unknown
  `algorithm_type` in *any* stage now raises on all ranks at start-up, before solvers are spawned
  or earlier stages run (previously it raised only when the sampler reached that stage).

### 2.9 WS7 fail-fast guards (proposal_builder.py / proposals.py) — **[verify]** / **[decide]**
None fires for a configuration that runs today with a Gaussian prior and consistent
`use_surrogate_gradients`; checked against every `toy_examples/*.py` (pCN is used only in
`toy_example_hamilton.py` and `sampling_diffusion_grf.py`, both with `PriorIndependentComponents`;
`typical_example*.py` use `FromScipy` but with RWMH).
- **pCN needs a Gaussian internal prior** (finding 1.10): `build_proposal` raises `ValueError` unless
  the prior is `Normal` or `PriorIndependentComponents`. `FromScipy` is rejected even when it wraps
  `multivariate_normal` — no non-fragile way to detect the family, and `FromScipy` has no
  `get_covariance()` anyway. **[decide]** if you want `FromScipy` pCN-eligible, it needs
  `get_covariance()` + an explicit `is_gaussian` flag on that class.
  **Resolved 2026-09-17, §7b (decision 10): stays rejected**, no change.
- **`needs_gradients ⇒ use_surrogate_gradients`** for every proposal type, incl. a `BlockProposal`
  containing a Hamiltonian sub-proposal (before: only bare Hamiltonian/HamiltonianInfinite, B3).
  Same message/exception as B3. **[read]** the message still says "Hamiltonian proposals" when it
  fires from inside a block proposal.
- **Adaptive RW, all-zero/NaN acceptance weights in a period** (A15c/d): before → `ZeroDivisionError`
  crash; now → that adaptation is skipped, previous covariance kept, `RuntimeWarning`. Only the
  crashing case changed; a non-degenerate 30-period sequence is bit-identical (pinned).
- Explicitly **not** done (would change working runs' sample streams; several are G-items):
  covariance shrinkage for `no_parameters > period`, bounded adaptive history / wiring of
  `Stage.adaptive_*` (G1), block sub-proposal re-seeding (G5), block-group gradients at the full
  state (`proposals.py` TODO — efficiency, not correctness: any gradient field gives a valid
  reversible HMC proposal), surrogate refresh for Hamiltonian proposals inside MH stages (same:
  efficiency), `HamiltonianInfinite` parametrisation (decision 8).
- **[decide]** `torch.set_num_threads(conf.torch_threads or 1)` (WS4) was **not** added: forcing
  one thread would silently slow collector training. Only recorded. Say if you want a
  `torch_threads` field (default `None` = leave torch alone).
  **Resolved 2026-09-17, §7b (decision 22): add nothing** — measured under this container's
  launcher, every rank already runs single-threaded; re-measure under a different launcher.
  **Reopened and decided 2026-09-18: `torch_threads: int | None = 1` field added,
  implemented.** The 2026-09-17 measurement only showed this container's MPICH `mpiexec`
  already throttles every rank to 1 thread (a launcher artefact) — the library itself never
  set it, so a different launcher (Slurm `srun`, OpenMPI, a bare `python` process) would still
  oversubscribe the node. `SamplingFramework.run()`/`run_local()` now call
  `apply_torch_threads(conf)` (`surrDAMH/modules/torch_threads.py`) before anything
  evaluates/trains a network and before the run manifest is built; `torch_threads=None`
  restores the old "leave torch alone" behaviour. See `docs/running.md#torch-cpu-threads`
  and `CHANGELOG.md`.
- **[read]** `load_last_samples` with more saved chains than requested silently used the first
  `no_chains` files (sorted by filename) — unchanged, but it now warns and says so.
- Seed *architecture* (SeedSequence per chain/stage/stream, `Distribution.rvs(generator)`) not
  started — it is G4/G5. **Done 2026-09-17** (§2.13): `Distribution.rvs(generator=...)` shipped
  and wired for the initial sample (G4) and block sub-proposals (G5), via per-object
  `np.random.RandomState(seed)` streams from `surrDAMH/modules/seeds.py` rather than a literal
  `SeedSequence.spawn()` tree — see `09_improvement_plan.md` §4 future work item 11 for what is
  still not unified (torch's global RNG, `tools.py`/`test_data.py`'s save/restore pattern).

### 2.10 Fail loud on every rank (WS8) — **[read]** / **[decide]**
`SamplingFramework.run()` wraps each role body (`_run_role`) and `process_CHILD.py` wraps the
spawned solver loop: uncaught `Exception` → `FATAL: unhandled exception on MPI rank r (ROLE role)`
+ traceback on stderr → `MPI.COMM_WORLD.Abort(1)`. `KeyboardInterrupt`/`SystemExit` re-raised
untouched. `run_local()` not wrapped. Measured before: a solver raising inside the spawned child
hung the job forever (pool polling `Iprobe`); on a sampler rank under plain `mpiexec python
driver.py` also hung (under `python -m mpi4py` mpi4py's own excepthook already aborted). After:
exit 9 in ~3 s / exit 1 in ~2 s with the traceback.
**[decide]** `ABORT_GRACE_SECONDS = 0.5` — a `time.sleep` before `Abort`, because Hydra discards
pending I/O forwarding on abort and the traceback was lost (0/3 runs) when another rank was inside
`MPI_Comm_spawn`; with 0.5 s it survived 5/5. Tuned on this container's MPICH 4.3.1 only. Alternative
if you dislike a sleep in library code: also write the traceback to a file under `output_dir`.
**Resolved 2026-09-17, §7b (decision 9): keep the 0.5 s sleep.**

### 2.11 Start-up handshake collector → samplers (WS8, finding 2.2) — **[verify]**
New `TAG_INITIAL_SURROGATE = 6`. Collector, after set-up and before its loop: `Send(1 int)` to each
sampler = "I can provide an evaluator without new snapshots" (`evaluator_instance is not None` or
preloaded snapshots ≥ `min_snapshots_initial`, which covers `min_snapshots_initial == 0`). Sampler:
blocking `Recv` where the B2 warning used to be printed (every sampler consumes exactly one
message; `use_collector=False` → no message). If the first stage is DAMH or uses a Hamiltonian
proposal and the flag is 0 → `RuntimeError` ("... the run would deadlock. Start with an MH stage,
pass initial_snapshots, use a pretrained/restored surrogate updater, or lower
min_snapshots_initial.") → job-wide abort via §2.10. No cycle: everything the sampler does before
its `Recv` is collective or non-blocking, and the collector's `Send` waits on nothing.
Gap (unchanged from before): a *first* stage with `use_only_surrogate=True` is not covered by this
condition; it hits the existing `assert commEvaluator.evaluator is not None`, now a clean abort.
**[verify]** on your collector-based scripts that the extra start-up message changes nothing
(it cannot affect sampling; `typical_example.py -n 4` runs to completion).

### 2.12 `initial_snapshots` counted once (WS8, finding 2.8) — **[verify]**
`process_COLLECTOR.py`: preloaded rows are counted where they are consumed (up front only when the
updater reports `training_data_loaded`, otherwise once at `add_data`). `surrogate_quality.csv`
first `snapshots_total` for N=8 is 8 (was 16). **Behaviour change:** runs that pass
`initial_snapshots` (or get them from a restored updater) reach `min_snapshots_initial` /
`min_snapshots_to_update` at different iterations → surrogate (re)trained at different moments →
**different DAMH-SMU sample stream** for such runs. Runs without `initial_snapshots` unaffected.
Unverified by running: the restored-state (`training_data_loaded` / `pretrained_ready`) branch of
the torch updaters — read, not executed (no MPI test restores a torch checkpoint).

### 2.13 G1-G6 + A30 (decided and applied 2026-09-17) - **[verify]**

Author decision: "do G1-G6 as recommended" (`08_safe_changes_plan.md` §G) plus A30 (§7,
finding 5.7). All seven applied together. CHANGELOG has one entry per item; this section is
the evidence. Suite results for the whole batch:

| Suite | Before (2026-09-17, same tree) | After |
|---|---|---|
| `./run_tests.sh unit` | 149 passed, 2 skipped, 1 xfailed, 5.6 s | **172 passed, 2 skipped, 0 xfailed**, 5.8 s (+22 new tests; the G5 xfail now passes) |
| `./run_tests.sh validation` | 10 passed, 72.1 s | **10 passed**, 72.7 s |
| `./run_tests.sh mpi` | 30 passed, 2 xfailed, 1 **xpassed**, 167.9 s | **32 passed, 1 xfailed**, 166.6 s |

`./run_tests.sh all` at the end: **184 passed, 2 skipped, 33 deselected** (77.8 s, the
`not mpi` half) then **32 passed, 1 xfailed** (166.3 s, the mpi half), exit 0.

The MPI delta is exactly the two I6 tests: `test_i6_adaptive_block_proposal_allreduce`
(strict xfail -> passes, now asserting the fast `ValueError` and that no `samples/` directory
is produced) and `test_i6_damh_adaptive_allreduce_shape_mismatch` (non-strict xfail that used
to XPASS at random -> plain passing test asserting exit 0 and no "Message truncated"). The one
remaining xfail is unrelated (I7 raw-buffer transport with a negative solver tag, finding 2.4).

Manual end-to-end check: `cd toy_examples && mpiexec -n 4 python3 -m mpi4py
template_experiment.py` -> exit 0, full HTML report written (artefact deleted afterwards).

Reference runs saved *before* editing and re-run afterwards (scratchpad, not committed):
a fixed-seed `adapt()` sequence built through `build_proposal`; three `run_local` MH runs with
a `user_specified` start (single stage / two stages / two stages with `is_excluded`); a
2-sampler adaptive MPI run (`mpiexec -n 3`, adaptive MH 25 evaluations + plain MH 15,
`user_specified` start) in which **both** ranks adapted twice. The MPI reference was run twice
before editing and was byte-identical, so the comparison is meaningful.

- **G1 (`Stage.adaptive_*` wired, behaviour change for stages that set them).**
  `build_proposal` forwards each field only when it is not `None`; `GaussRandomWalk_adaptive`
  gained `sample_limit` (bounded history).
  *Bit-identical with all three fields `None`* - the reference sequence (3 parameters, seed 1,
  80 `adapt()` calls driven by `RandomState(0)`, one `propose_sample` each) reproduced the same
  `sd_or_cov`
  `[[3.49478905, -0.44379604, 0.25222071], [-0.44379604, 2.4095101, 0.11361171], [0.25222071, 0.11361171, 3.13552088]]`,
  the same 80 proposals (checksum 5.458453546712782), `len(samples)=80`, `target_rate=0.25`,
  `corr_limit=0.3`, `period=10` - `np.testing.assert_array_equal` on all six arrays.
  The pre-existing `test_adaptive_random_walk_nondegenerate_sequence_is_bit_identical_to_pre_guard_behaviour`
  (hard-coded 2x2 covariance) also still passes.
  *Changed*: any stage that sets a field. No maintained script in the repo does
  (`grep` over `toy_examples/` and `surrDAMH/`: zero hits outside `stages.py`/
  `proposal_builder.py`/tests), so nothing in-tree changes; the affected scripts are the
  author's own (`08 §G1` names `TSX_complete_experiment_2/sampling_.py`, target 0.1, archived).
  New tests: `adaptive_target_rate=0.1` gives a
  different adapted covariance than the 0.25 default; `adaptive_corr_limit=0.05` caps the
  adapted correlation at 0.05; `adaptive_sample_limit=50` gives `len(samples)==50` after 80
  proposals (vs 80 unbounded) and a different covariance.

- **G2 (2-D covariance before the adaptive `Allreduce`).**
  *Bit-identical where every rank adapted*: the 2-sampler reference run's
  `sampling_output/samples` and `.../notes` trees are byte-identical before/after (`diff -r`,
  no output). Both ranks ended stage 0 with a 2-D matrix (rank 1 `prop_cov`
  `[[0.7355054, 0.01620385], [0.01620385, 0.73258118]]`), which is the case that already
  worked.
  *Changed*: a rank that never adapted now hands over `diag(sd**2)` instead of the 1-D `sd`
  vector - e.g. `sd_or_cov = [0.5, 0.5]` becomes `[[0.25, 0], [0, 0.25]]`, so the next stage
  draws with `multivariate_normal` instead of independent `normal` calls (same distribution,
  different RNG stream). Not exercised by the reference run by construction; covered by the
  unit test `test_as_covariance_matrix_agrees_before_and_after_a_single_adaptation`.
  *Fail-fast*: `proposal_type="block"` + `adaptive=True` now raises `ValueError` in
  `build_proposal` before any sampling (unit test + the flipped MPI I6 test).

- **G3 (`lhs_normal` maximin selection).** `maxmin = quality` is now assigned inside the `if`.
  Changed for every `initial_sample_type="lhs"` run with more than ~2 chains. Measured
  (`loc=0`, `scale=1`, `seed=0`): for `n=2` the last candidate *happens* to be the
  maximin-best, so 2-chain starts are unchanged; for `n=4` the design changes from
  `[[1.36126493, 0.10730728], [-0.41372423, -0.15687537], [-1.02807333, -1.39663415], [0.02408189, 1.33199153]]`
  to
  `[[-0.29964565, 0.36383969], [0.6170501, -0.61960476], [-2.01825558, -2.57207158], [1.72966538, 1.59406434]]`
  (min squared pairwise distance 0.0347 -> 0.1268 for the seed-0 candidate set with `n=8, d=3`).
  That is why `tests/mpi/test_mpi_posterior.py` and `test_mpi_basic.py` A1 still pass: they use
  2 samplers, and A1 compares two runs of the *new* code anyway.
  `tests/unit/test_helpers.py::TestLHSNormal` now asserts the returned design is the argmax of
  the 5 candidates' qualities over seeds 0-11, and (seed 2, where the argmax is candidate 2)
  that it is *not* the last candidate. The two `numpy.matlib.repmat` calls were replaced by
  `np.tile`, verified bit-identical on 20 random vectors and pinned by
  `test_lhs_normal_no_longer_imports_numpy_matlib`; the suite no longer emits the
  `PendingDeprecationWarning`.

- **G4 (per-rank seeded initial sample).** New seed stream `initial_sample_seed(no_stages,
  rank) = 10*no_stages*rank + 3`, next to the proposal's `+1` and the algorithm's `+2`; the
  whole family now lives in the new `surrDAMH/modules/seeds.py` (also used by
  `manifest.py::_seeds_dict`, so the formula is stated once).
  Measured (`PriorIndependentComponents` of 2 standard-normal components, `no_stages=1`):
  rank 0 seed 3 -> `[2.04091912, -2.55566503]`, rank 1 seed 13 -> `[1.82675656, -3.07833191]`,
  rank 2 seed 23 -> `[0.55326059, 0.21760061]`, rank 3 seed 33 -> `[0.39836997, -0.56282334]`;
  rank 0 repeated gives the identical vector. The 4-rank `template_experiment.py` run above
  indeed starts chain 0 at `2.0409191213851825, -2.5556650313141818`.
  *Changed*: every `initial_sample_type="prior"` run - but those were never reproducible
  before (unseeded global RNG), so there is no previous value to be identical to; that is
  exactly finding 1.9/A14.
  *Unchanged*: `rvs()` with no `generator` (external callers), `lhs` and `continued` starts,
  and duck-typed `initial_samples_distribution` objects whose `rvs()` takes no arguments -
  `rvs_with_generator` falls back to the no-argument call (the `FixedSample` doubles in
  `tests/test_runner_local.py`, `tests/unit/test_manifest.py`, `tests/mpi/conftest.py`,
  `tests/helpers_statistics.py` all still work; the `user_specified` reference run is
  byte-identical).
  Manifest: `seeds.per_rank[].initial_sample_seed` added, `initial_sample_reproducible` is now
  `true` for every type and `unreproducible_reason` is gone; `Configuration`/
  `docs/configuration.md` updated.

- **G5 (`BlockProposal` sub-proposal re-seeding).** New `Proposal.reseed(seed)` (default
  implementation re-creates `self._generator`, warns if a custom proposal has none) and
  `proposals.subproposal_seed(block_seed, index) = 2**31 + 1000*block_seed + index`.
  Deviation from the plan's suggested `seed + 100*(k+1)`: that form collides for realistic
  runs - with 2 groups and 11+ samplers, rank 10's group-0 stream equals rank 0's group-1
  stream (`10*A+1+100*(k+1)` collides whenever `A-B = 10*(j-k)`). The form used is injective in
  `(block_seed, index)` for up to 1000 sub-proposals and sits above `2**31`, so it also cannot
  collide with another chain's `seed0`-family seeds.
  Measured: block seed 1 (rank 0, stage 0) -> sub seeds `[2147484648, 2147484649]`; block
  seed 11 (rank 1, stage 0) -> `[2147494648, 2147494649]`. Before G5 both ranks' sub-proposals
  kept the user's `seed=0` and produced the *identical* increment `0.88202617`; now rank 0
  proposes `-0.04261991` and rank 1 `-0.10524088` on group 0.
  Per-stage re-seeding (A13) follows from doing this in `BlockProposal.__init__`, which
  `build_proposal` calls once per stage with that stage's seed: a sub-proposal *object* shared
  by two stages no longer continues the previous stage's stream, and re-running a stage gives
  the same stream (`test_block_proposal_reseeds_subproposals_per_stage`).
  `test_block_proposal_subproposals_reseeded_per_rank` flipped from strict xfail to passing;
  `test_gauss_random_walk_adaptive_inside_block_proposal_constructs` kept.
  *Changed*: every block-proposal run (intended).

- **G6 (line-buffered CSV writers).** `open(path, "w", buffering=1)` in
  `modules/monitoring.py`. No content change; `test_csv_writers_are_line_buffered` reads the
  file back through `csv.reader` after each `writerow` without closing the writer and finds
  the rows already there. Cost: one `write()` syscall per row (no measurable change in the
  suite timings above).

- **A30 (stage-boundary state counted once).** Rule implemented, one sentence:
  *the first row of a stage whose initial state was already written by the preceding stage
  (i.e. any stage after the first whose predecessor had `save_to_file=True`, including a stage
  following `is_excluded=True`) carries `weight = counter_rejected_current`, without the
  leading `+1`.* `AlgorithmBase._current_state_row_weight` + the
  `initial_sample_is_carried_over` constructor flag, set identically by both runners as
  `initial_sample_is_carried_over = stage.save_to_file` after each stage.
  Measured on the `run_local` references (`user_specified` start, `ToySolver`):
  | run | stage | before | after | `acc+rej+prerej` |
  |---|---|---|---|---|
  | two stages 30+25 | `alg0000_MH` | sum 31, first row weight 1 | unchanged | 12+18+0 |
  | two stages 30+25 | `alg0001_MH` | sum 26, first row weight 3 | **sum 25, first row weight 2** | 14+11+0 |
  | 30 (`is_excluded`) + 25 | `alg0000_MH` | sum 31 | unchanged | 12+18+0 |
  | 30 (`is_excluded`) + 25 | `alg0001_MH` | sum 26, first row weight 1 | **sum 25, first row weight 0** | 19+6+0 |
  | single stage 50 | `alg0000_MH` | sum 51 | unchanged | 15+35+0 |
  Concatenated: 31+25 = 56 = 55 iterations + 1 in both two-stage runs (was 57).
  The 4-rank `template_experiment.py` run shows the same on a real DAMH-SMU run:
  `alg0000_MH` sum 81 = 80+1 (first row weight 10), `alg0001_DAMH-SMU` sum 262 = 262+0 (first
  row weight **0**), `alg0002_DAMH` sum 238 = 238+0 (first row weight 0); 81+262+238 = 581 =
  580 iterations + 1, matching the 581-row decompressed chain the report prints.
  *Weight-0 rows*: grepped every use of the `weight` column. `decompress` is fine
  (`np.cumsum`, pinned by a new test); `np.average(weights=)`/`np.cov(fweights=)` in
  `get_mean_and_cov` accept a zero (checked); the `raw_data` weight arithmetic at
  `post_processing.py:~299`/`~657` is a different column and unaffected;
  `StageSamples.no_unique_samples` was the only thing that would misbehave (it counted a
  weight-0 row as a sample, inflating the *default* histogram bin count - its only consumer),
  fixed as the one-liner `np.count_nonzero(self.weights[i])` recommended by
  `11_output_format_v2_spec.md`.
  *Deliberately unchanged*: the weight handed to the collector for surrogate training.
  `_finalize_run` never forwards a stage's final state, so nothing is double-counted there -
  dropping the `+1` would *under*-count instead.
  Tests: `test_mh_weight_column_bookkeeping` keeps its `+1` (it is a first stage) with a
  comment saying why; new `test_mh_weight_column_drops_the_plus_one_for_a_carried_over_initial_state`,
  `test_two_stage_weights_count_each_state_once`,
  `test_stage_after_is_excluded_stage_also_drops_the_plus_one`,
  `test_zero_weight_row_survives_decompress`,
  `test_zero_weight_boundary_row_is_not_counted_and_not_decompressed`.
  Docs: rule added to `docs/outputs.md`, `library_notes/00_overview.md` §weights, and
  `11_output_format_v2_spec.md` (whose option-C paragraph is replaced and marked decided).
  `tests/validation` re-run unchanged (its multi-stage V3/V3b statistics use `stage_index=1`
  with 10 % burn-in on the *decompressed* chain, where a one-off weight difference of 1 in the
  first row is statistically inert): 10 passed.

**[verify]** on your own scripts: (a) any script that sets `Stage.adaptive_target_rate` /
`adaptive_corr_limit` / `adaptive_sample_limit` now honours it - past runs that set them ran
with 0.25 / 0.3 / unbounded and should be re-labelled; (b) any `initial_sample_type="lhs"` run
with 3+ chains starts from a different (better-spread) design; (c) any block-proposal run
changes; (d) the `weight` column of every multi-stage run changes in its per-stage first row,
so any external script that assumed `sum(weight) == acc+rej+prerej+1` per stage, or `weight >= 1`,
must be updated.

### 2.14 Output format v2 (WS9a, applied 2026-09-17) — **[verify]** / **BREAKING**

Implements `11_output_format_v2_spec.md` under author decision 6 ("no compatibility with old
saved experiments; a clear user interface is more important; no converter"). Format reference:
`docs/outputs.md`. CHANGELOG: "WS9a — output format v2".

**What changed on disk**

| File | v1 | v2 |
|---|---|---|
| `samples/<stage>/rank%04d.csv` | no header; `weight, u_1..u_p, log_posterior` | header `multiplicity,par_0..par_{p-1},log_posterior`; values unchanged |
| `raw_data/<stage>/rank%04d.csv` | no header; ragged: `state_type, u.., solver_tag, obs.., log_likelihood, log_prior`, where a `prerejected` row silently put SURROGATE values in the observation block | header + rectangular: `state_type,par_0..,solver_tag,obs_0..obs_{m-1},obs_approx_0..obs_approx_{m-1},log_likelihood,log_prior`. `obs_*` = exact model (all-NaN for `prerejected`), `obs_approx_*` = surrogate (all-NaN where none exists). For `prerejected` rows `log_likelihood` is now the SURROGATE log-likelihood the sub-chain used (v1 wrote the carried-over exact value of the current state) |
| `notes`, `subchain_stats`, `surrogate_quality*`, `last_sample/*.npz` | — | unchanged |
| `run_manifest.json` | `format_version: 1` | `format_version: 2`, and it is now *required* to read a directory |

**What changed in the library**

- `surrDAMH/modules/run_data.py` (new): `read_run(output_dir, load_raw_data=True) -> RunData`,
  the single reader of the on-disk layout. Re-exported as `surrDAMH.read_run`.
- `surrDAMH/modules/manifest.py`: `FORMAT_VERSION = 2`, new `RunFormatError`, and the shared
  column-name builders `samples_columns`/`raw_data_columns` used by BOTH the writer and the
  reader (so a rename cannot drift). `modules/tools` is now imported lazily inside
  `write_run_manifest` (importing `manifest` from `post_processing` would otherwise be a
  circular import through `surrDAMH/__init__`).
- `surrDAMH/modules/monitoring.py`: `CsvWriter(..., header=...)` and
  `SamplingOutputMonitor.set_header(data_name, header)`; the header is written when the file is
  created, and creation stays lazy (a `save_to_file=False` stage still writes nothing at all).
- `surrDAMH/modules/algorithms.py`: registers the two headers; `_record_proposed_snapshot` now
  takes the exact and the surrogate observation block plus the log terms explicitly.
- `surrDAMH/post_processing.py`: `Samples.__init__` calls `read_run(..., load_raw_data=False)`
  once and builds `StageSamples` from it; `load_notes`/`load_subchain_stats` read from it too.
  No `pd.read_csv(header=None)` / positional `iloc[:, k]` on `samples`/`raw_data` remains
  (`load_snapshots`, `hist_observations`, `_load_snapshot_parameters_and_observations`,
  `find_best_fits` select by column NAME; `find_best_fits` keeps its chunked streaming).
  The plotting/statistics API is unchanged.
- `Samples` now raises `ValueError` if the `no_parameters` you pass disagrees with the manifest.

**Refusal of old directories** (`RunFormatError`, verified by hand on a fabricated v1 dir):

```
<dir>/sampling_output/run_manifest.json not found: this is not a surrDAMH output format v2
directory ... Pre-v2 runs cannot be read -- there is no converter (author decision 6,
2026-09-17); re-run the sampling to obtain v2 output.
<dir>/sampling_output/run_manifest.json: output format_version 1, expected 2. ...
```

**Tests** (`./run_tests.sh all`)

| Suite | Before (2026-09-17, same tree) | After |
|---|---|---|
| `pytest tests -q -m "not mpi"` | 184 passed, 2 skipped, 77.8 s | **199 passed, 1 skipped**, 79.0 s |
| `./run_tests.sh mpi` | 32 passed, 1 xfailed, ~166 s | see below |

Count changes: `+14` new `tests/unit/test_read_run.py` tests; `+1` because the previously
skipped golden-file test (testing-plan item 40) is now implemented and passes, hence one
skip fewer. Test-side v1 assumptions fixed: `tests/helpers_statistics.py::read_weighted_samples`
(`skiprows=1`, "weight" → "multiplicity" in its wording), `tests/test_runner_local.py::_read_samples`,
four `np.loadtxt` sites in `tests/mpi/test_mpi_basic.py`, two in `tests/mpi/test_mpi_posterior.py`,
the two CSV readers in `tests/unit/test_algorithms_local.py` (now assert the header),
`tests/unit/test_manifest.py` (`format_version == 2`), and the
`tests/unit/test_post_processing.py` fixtures (now written through the production `CsvWriter`
and the production column lists, with a minimal v2 `run_manifest.json`).

**[verify]** on your own scripts: any external analysis that reads `samples`/`raw_data` with
`np.loadtxt(...)` or `pd.read_csv(..., header=None)` needs `skiprows=1`/`header=0`; anything
that indexed the observation block of a `prerejected` row now finds NaN there (use
`obs_approx_*`, or filter `state_type != "prerejected"`); and old `out_*` directories can no
longer be opened with `Samples`/`read_run` at all.

### 2.15 Post-processing structure and cleanliness (WS9b, applied 2026-09-17) — **[verify]**

Implements `09_improvement_plan.md` WS9 bullets 2-5 on top of WS9a. **No output file format
changed and no plot was redesigned.** CHANGELOG: "WS9b — post-processing structure and
cleanliness".

**New module layout** — `surrDAMH/post_processing.py` (2466 lines) became the package
`surrDAMH/post_processing/`, import graph bottom-up:

| Module | Lines | Contents |
|---|---|---|
| `base.py` | 158 | `SamplesBase` (WS-later: attribute annotations for what `loading.py` assigns, plus the `_resolve_stages`/`_resolve_chains`/`_burn_in_for`/`_decompressed_samples`/`_get_stage_names`/`_raw_data_available` validators, moved verbatim out of `loading.py` so every layer above can see them) |
| `statistics.py` | 684 | `SamplesStatistics(SamplesBase)` (`get_mean_and_cov`, `calculate_CpUS`, `calculate_effective_sample_size`, `_collect_samples_matrix`, `calculate_gelman_rubin`, `_load_snapshot_parameters_and_observations`, `find_best_fits`, `compute_posterior_field_statistics`), `Autocorrelation`, `rank_best_fit_candidates`, `decompress`, `autocorr_FM`, `auto_window`, and the private `raw_data` column helpers |
| `plots.py` | 944 | `SamplesPlots(SamplesStatistics)` (all `plot_*`/`_plot_hist_*`/`hist_observations`/surrogate-quality figures), `add_normal_dist_grid` |
| `html_report.py` | 843 | `SamplesReports(SamplesPlots)` (`html_report_extended`, legacy `pdf_report`/`html_report`) |
| `loading.py` | 218 | `StageSamples`, `Samples(SamplesReports)`, `summarize`/`load_notes`/`load_subchain_stats`/`load_snapshots`/`get_summary` |
| `__init__.py` | 50 | facade + `matplotlib.use("Agg")` |

(WS-later, 2026-09-18: the four `Samples*` classes were turned from independent mixins,
each reading attributes/methods defined only on a sibling, into the linear chain
`SamplesStatistics(SamplesBase)` -> `SamplesPlots` -> `SamplesReports` -> `Samples` shown
above, so a type checker can resolve `self.` accesses across layers. Purely structural:
every method still has exactly one definition, and `Samples.__mro__` still ends the same
way a type checker (and Python) would resolve any of the old mixin combinations.)

Public surface re-exported from `surrDAMH.post_processing` (unchanged import paths):
`Samples`, `StageSamples`, `read_run`, `RunData`, `RunFormatError`, `raw_data_columns`,
`sampling_output_dir`, `Autocorrelation`, `rank_best_fit_candidates`, `decompress`,
`autocorr_FM`, `auto_window`, `add_normal_dist_grid`, plus `SamplesBase` and the three
mixin-chain classes above it.
(`find_best_fits`, `calculate_gelman_rubin`, `calculate_effective_sample_size`,
`calculate_CpUS` are **methods of `Samples`**, not module-level functions — the improvement
plan lists them as if they were.)

**Numerical identity of the split.** `toy_examples/typical_example.py` was run once with the
pre-WS9b tree; that output directory was kept and the post-processing was then re-run on it
with (a) the unmodified tree, (b) the tree right after the verbatim move, (c) the final tree.
`typical_example.py` itself writes no `summary.csv` and its two DAMH stages are `time_limit`
bound, so re-running the sampling could not give a comparable file; the check was therefore
done on the *same* run directory, which is stricter. Compared: `summary.csv`
(`calculate_CpUS` over all three stages, i.e. the columns `accepted, rejected, pre-rejected,
sum, subchain_acc_rate, subchain_move_rate, outer_acc_given_move, ratio_eval, autocorr, CpUS`)
plus a dump of overall and per-stage mean/cov, `no_unique_samples`, `length`, ESS, R-hat and
`_collect_samples_matrix`. **All three are byte-identical** (`diff` clean).

**P6 — every `except BaseException` / bare `except` in the post-processing code** (line
numbers in the pre-split `surrDAMH/post_processing.py`):

| Site (old file:line) | Guarded | Old | New |
|---|---|---|---|
| `get_mean_and_cov:184` | `samples_compressed[s][burn_in[idj][s]:]` | `except BaseException` → print `CHAIN … NOT AVAILABLE` | removed; `Samples._burn_in_for` validates the table once and raises `ValueError` naming the shape |
| `plot_chains:389` | `stage.samples[s][…]` | `except BaseException` → print `HISTOGRAM 1D: CHAIN … NOT AVAILABLE` (wrong message, too) | removed; `_decompressed_samples` raises `AttributeError` naming `decompress_samples=False` |
| `_plot_hist_1d:426` | same | `except BaseException` → print | removed, same fix |
| `_plot_hist_1d:448` | `prior_component.pdf(x_grid)` | `except Exception: pass` | `except (AttributeError, TypeError, ValueError)` → prints which component failed and why, then skips the overlay |
| `_plot_hist_2d:466` | `stage.samples[i][…]` | `except BaseException` → print | removed, same fix as `plot_chains` |
| `calculate_effective_sample_size:983` | `len(stage.samples[s][…])` | bare `except: pass` (silently under-counted `total_samples`) | removed, same fix |
| `_collect_samples_matrix:1115` | `stage.samples[s][…]` | `except BaseException: continue` (silently dropped chains) | removed, same fix |
| `calculate_gelman_rubin:1144` | `stage.samples[chain_idx][…]` | `except BaseException: continue` | removed; an explicit `chain_idx >= stage.no_chains` test keeps the "stage written by fewer chains" case |
| `html_report_extended:1831` | `plot_hist_grid` | bare `except:` → fixed "not enough samples or an error occurred" text | `except Exception as e` → the real message, escaped, in the report |
| `html_report_extended:1855/1863/1874` | — | two more bare `except:` inside a commented-out `"""…"""` block (dead "2.3 Chain Traces / 2.4 Cumulative Averages"; both plots exist per stage in section 3) | block deleted |
| `plot_posterior_field_statistics:1607` | `mtri.Triangulation(...)` | `except Exception` → silent scatter fallback | `except (RuntimeError, ValueError)` → prints why it fell back |

The remaining `except Exception as e` blocks in `html_report_extended` (best fits, field
statistics, CpUS, ACF, ESS, 1D marginals) were left: each already renders the real message
into its own report section, and the two genuinely optional sections (no raw data saved / no
field statistics supplied) are chosen by an explicit `if`, not by catching.

**P3 chains_to_disp per method** — see the CHANGELOG entry; `plot_acceptance_rates` and the
section-1 summary table are the only chain-insensitive parts left, and the report now states
that in both places.

**Tests** (`./run_tests.sh`)

| Suite | Before (WS9a tree) | After |
|---|---|---|
| `pytest tests -q -m "not mpi"` | 199 passed, 1 skipped, 82.9 s | **225 passed, 1 skipped**, 78.8 s |
| `./run_tests.sh mpi` | 32 passed, 1 xfailed, 167.5 s | unchanged |

Count change `+26` = `−2` pinned-bug tests replaced (`TestFindBestFits::test_missing_log_columns_silently_degrades_known_unfixed_bug`,
`TestLoadPosteriorSurrogateKnownUnfixedBug::test_raises_indexerror_on_current_on_disk_format`)
`+28` new: 6 in `TestFindBestFits` (missing log columns raise, per mode; l2 unaffected;
missing `obs_*` raise; order matches `rank_best_fit_candidates`; `chains_to_disp`), 3
`TestLoadPosteriorSurrogateRemoved`, 7 `TestChainsToDisp`, 3 `TestExplicitExceptions`, 2
`TestAutocorrelationChainCount`, 4 `TestReportConfigurationSection`, 3
`TestFiguresAreNotLeaked`.

**Manual checks**

| Command | Result |
|---|---|
| `cd toy_examples && mpiexec -n 4 python3 -m mpi4py typical_example.py` | exit 0; `out_typical_example/post_processing_output/{report_extended.html,histograms.pdf,chains.pdf,averages.pdf}` written; the report's new "Run Configuration" section shows the manifest configuration, `runner=mpi`, `surrDAMH version=0.2.0` and "no option flagged as unverified" |
| `cd toy_examples && mpiexec -n 4 python3 -m mpi4py template_experiment.py` | exit 0; `write_report` wrote `summary.csv` + `report_extended.html` |

**[verify]** on your own scripts: `Samples(..., load_posterior_surrogate=True)` now raises
`TypeError`; `find_best_fits(..., ranking_mode="posterior"/"likelihood")` on a `raw_data` file
without the log columns now raises `RunFormatError` instead of returning an arbitrary table;
`Samples(..., decompress_samples=False)` followed by a trace/histogram/ESS/R-hat call now
raises instead of printing "NOT AVAILABLE"; a mis-shaped `burn_in` now raises.

### 2.16 Evaluator/Updater contract, snapshot weighting, NN normalization (WS6, applied 2026-09-17) — **BREAKING** / **[verify]**

Implements the decisions recorded in `library_notes/12_evaluator_contract_spec.md` (now marked
implemented). Five user-visible changes; the first two change numbers, the rest are interface.

**(a) NN snapshot weighting: `"multiplicity"` → `"uniform"` (new default).** Every updater now
takes `weighting: Literal["uniform", "multiplicity"] = "uniform"`. Before WS6 the NN updater
always weighted its loss by the multiplicity, i.e. rejected proposals (multiplicity 0) never
influenced training; now they do, each counting once. `"multiplicity"` restores the old
*semantics* (zero-multiplicity rows dropped, the rest weighted by `m`).
**[verify]** any NN run you want to compare with an older one: pass `weighting="multiplicity"`.

Measured (`tests/unit/test_surrogates.py` dataset, 24 snapshots of which 8 have multiplicity 0
and 8 have multiplicity 2; 8-unit MLP, AdamW, 50 steps, seed 0, identity normalization),
evaluator output at 3 query points:

| | pre-WS6 | `weighting="multiplicity"` | `weighting="uniform"` (new default) |
|---|---|---|---|
| first output value | `0.09266213` | `0.09266209` | `0.09365773` |
| training loss | `12.034626` | `12.034626` | `12.208886` |

`"multiplicity"` is *mathematically* the old behaviour but not bit-identical: the old code kept
the zero-weight rows in the batch and divided by `sum(w)`, the new one drops them first, so the
float32 summation order differs (observed deviation ~5e-8, and the reported loss is identical).
On a dataset whose multiplicities are all 1, `"uniform"` and `"multiplicity"` are **bit-identical**
(checked: `max |Δ| = 0.0`), so runs with no rejected proposals are unaffected.

**Classical surrogates are bit-identical under the default.** Poly/RBF/kd-tree previously ignored
the weights outright, which is exactly what `"uniform"` does. Verified by hard-coded pre-WS6
reference values in `tests/unit/test_surrogates.py::test_classical_surrogates_unchanged_under_default_weighting`
(`np.testing.assert_array_equal`, all three updaters, dataset with multiplicity 0/1/2 rows).
New: `PolynomialSklearnUpdater` now *can* weight (`supports_sample_weights = True`,
`sample_weight=` into `LinearRegression.fit`) when you ask for `"multiplicity"`; RBF and kd-tree
are interpolants and declare `supports_sample_weights = False`.

**(b) NN output normalization: identity → likelihood (new default).**
`NeuralNetworkUpdaterMinibatches(output_normalization="identity" | "likelihood" | "manual")`,
default `"likelihood"`: the targets are centred on `likelihood.mean` and scaled by the
per-observation noise sd (`likelihood.sd` broadcast, or `sqrt(diag(likelihood.cov))`).
`SamplingFramework.__init__` and `run_local` call the new `Updater.set_output_normalization`
hook once. A likelihood without usable statistics → `RuntimeWarning` naming its class + identity.
`"manual"` is the old explicit `output_mean`/`output_scale`; `"identity"` is the old default.
**[verify]** every existing NN script: surrogate accuracy changes. Measured on a deliberately
badly scaled toy (300 train / 100 test points, observations on scales ~500 and ~1, noise sd
`[50, 1]`, 32-32 MLP, AdamW, 2000 steps, seed 0):

| `output_normalization` | test RMSE (total) | per observation |
|---|---|---|
| `"identity"` (old default) | 54.95 | `[77.71, 0.759]` |
| `"likelihood"` (new default) | 7.65 | `[10.82, 0.0724]` |

Numbers from one toy with one seed — they show the setting matters, not that it is always better.

**(c) `Evaluator.__call__` always returns `(n, no_observations)`.** The two torch evaluators used
to return a flattened `(n*no_observations,)`. The five call sites that compensated for this are
gone (`algorithms.py` `_get_surrogate_observations` ×2 and the gradient path, `process_COLLECTOR.py`
quality metrics ×2); `_get_surrogate_observations` now indexes `[0]` into the one-row batch.
`no_parameters`/`no_observations` are real attributes on `Evaluator`, set via `super().__init__`.
**Value-neutral**: the V3-style DAMH-SMU chain with a polynomial surrogate (linear-Gaussian toy,
K=5, 200 MH + 4000 DAMH evaluations, seed-fixed `run_local`) produced **byte-identical**
`samples/alg0001_DAMH-SMU/rank0000.csv` (4002 lines), `samples/alg0000_MH`, `notes` and
`subchain_stats` files before and after. `SurrogateAsSolver.get_observations` keeps its
`reshape(-1)`, now provably exact rather than right by accident (§2.7).

**(d) `NeuralNetworkUpdaterBasic` deleted** (`surrDAMH/surrogates/torch_perceptron.py` removed).
`import surrDAMH.surrogates.torch_perceptron` raises `ImportError`;
`surrDAMH.surrogates.NeuralNetworkUpdaterBasic` raises `AttributeError`. Replacement, used by
`toy_examples/neural_network_surrogate.py` and `toy_examples/sampling_TSX.py`:
`NeuralNetworkUpdaterMinibatches(..., solver="lbfgs", batch_size=None, replay_ratio=0.0,
train_on_added_data=False)`. All four remaining updaters are registered for
`surrogates.reuse.SurrogateReused`, and checkpoints are read with
`torch.load(..., weights_only=True)` (S22).

**(e) "weight" renamed to "multiplicity"** in the sampler→collector payload and everything
downstream: `AlgorithmBase._send_to_collector(sample, multiplicity)`, `LocalSnapshot.multiplicity`,
`InMemorySnapshotSink.get_snapshot_arrays()`, `Updater.add_data(parameters, observations,
multiplicity)`, `get_training_data_arrays()`, the `surrogate_training_data.npz` key
(`weights` → `multiplicity`), and the third array of `initial_snapshots`. `TestData`'s posterior
weights are a different concept and keep the name "weights". **BREAKING** for a custom `Updater`
that declared `add_data(..., weights=...)` as a keyword, and for old `.npz` training-data files
(no compatibility shim by decision).

**Bug fixed (finding 3.4):** `NeuralNetworkUpdaterMinibatches.initial_training` no longer
persists its synthetic `randn` rows — `get_training_data_arrays()`/`get_initial_snapshots()`/the
checkpoint now report only real snapshots, and real snapshots added earlier survive the call.
`tests/unit/test_surrogates.py::TestInitialTrainingSyntheticRows` was flipped from pinning the
bug to pinning the fix.

**Manifest:** the `surrogate` block gained `weighting`, `supports_sample_weights`,
`output_normalization` and `output_normalization_provenance`. The collector prints one start-up
line naming the weighting policy and the normalization provenance.

### 2.17 Raw-observation MPI path removed (WS8, decision 5, applied 2026-09-17) — **BREAKING** / **[verify]**

`Configuration.pickled_observations` is **deleted**. It selected between two solver-pool
transports; only the pickled one survives. A script that still passes the argument now fails at
construction with `TypeError: Configuration.__init__() got an unexpected keyword argument
'pickled_observations'` — no compatibility shim, per the project's "no compatibility" rule. Fix:
delete the argument. Nothing else changes for anyone who used the default (`True`), which is
every maintained example: `grep -rn pickled_observations toy_examples/` finds **nothing**, so no
toy example needed editing.

**Message sequence — before (pickled mode, the default) and after: identical.**

| step | before (`pickled_observations=True`) | after |
|---|---|---|
| sampler → pool | `Send(parameters, dest=pool, tag=k)`, `k = 1,2,3,…` | unchanged |
| pool → child | `Bcast([k,'i'])` then `Bcast([parameters, MPI.DOUBLE])` | unchanged |
| child(0) → pool | `send([observations, solver_tag], dest=0, tag=int(k))`; pool polls `Iprobe(source=0, tag=k)` and `recv(source=0, tag=k)` | unchanged |
| pool → sampler | `send([observations, solver_tag], dest=sampler, tag=k)`; sampler `recv(source=pool)` | unchanged |
| sampler → pool (shutdown) | `Send(np.zeros(1), tag=0)` | unchanged |

**Removed (the raw mode, `pickled_observations=False`):** child(0) → pool
`Send(sent_data, dest=0, tag=solver_tag)` + pool `Iprobe/Recv(tag=ANY_TAG)`; pool → sampler
`Send(obs, dest=sampler, tag=solver_tag)` + sampler `Recv(tag=ANY_TAG)` with the tag read from
`MPI.Status`. These were rows A4 and B5 of the protocol table in
`library_notes/02_mpi_processes_and_communication.md`, now marked REMOVED there.

**What this fixes:** finding 2.4 / M6 — in raw mode the solver's status code *was* the MPI tag, so
a failed solve (`solver_returns_tag=True`, `tag=-1`) raised `mpi4py.MPI.Exception: Invalid tag` in
the spawned child and (since the WS8 fail-loud change) aborted the whole job. The tag now always
travels inside the payload, so a negative tag is ordinary data: `_compute_log_posterior_terms`
returns `-inf` for it, the proposal is rejected, and `_handle_rejection` does not forward it to
the collector. The raw path's fixed-size/dtype buffer hazards of finding 2.3 go with it.

**Tests flipped** (`tests/mpi/test_mpi_transport.py`, rewritten):

| before | after |
|---|---|
| `test_i7_observation_transport_matrix`, 5 parametrized cells (`pickled × returns_tag`) | `test_i7_observation_transport`, 2 cells (`solver_returns_tag ∈ {False, True}`) |
| `test_i7_raw_buffer_transport_with_negative_solver_tag` — **xfail(strict)** | `test_i7_negative_solver_tag_completes_and_rejects` — **plain passing test**: exit 0, `DRIVER FINISHED`, and (with `save_snapshots_to_file=True`) every `raw_data` row whose `solver_tag < 0` has `state_type == "rejected"`, plus no accepted state with `par_0 > 0` |

**[verify]** if you have a private script that sets `pickled_observations` (nothing in this repo
does): drop the argument, the behaviour you get is the one you already had at the default.

### 2.18 WS5 finished: `TestData`/`SurrogateRestart`/`describe()`/absolute solver path (applied 2026-09-17) — **[verify]** / **[read]**

Five changes, none of which alters a posterior for an unchanged script.

**(a) The two advanced examples now use the library helpers.**
`toy_examples/toy_example_hamilton.py` (131 lines, was 257) and
`toy_examples/sampling_diffusion_grf.py` (139 lines, was 292 — under the plan's 150-line
acceptance limit) no longer import anything from `surrDAMH.modules.tools` except nothing at all:
test data comes from `TestData.reuse(conf.output_dir)` falling back to
`TestData.generate(...).save(...)`, surrogate restart from `SurrogateRestart`, and the HTML report
from `sam.write_report(...)` instead of a locally defined `generate_html_report`. The five
superseded `tools.py` helpers (`generate_surrogate_test_data`, `compute_test_log_posterior`,
`normalized_weights_from_log_posterior`, `surrogate_restart_state_has_snapshots`,
`load_surrogate_restart_state_if_available`) are **kept**, per the no-delete rule, each with a
one-line "Superseded by …" docstring; they now have zero callers in the tree.

**[verify]** two visible consequences for anyone re-running these two scripts:
- the test-data file is now written by `TestData.save()`, i.e. with the array names
  `parameters`/`surrogate_parameters`/`observations` instead of the old
  `test_parameters`/`test_observations`/`test_log_posterior`/`test_weights`. An existing
  `sampling_output/surrogate_test_data.npz` from an older run is **not readable** by
  `TestData.reuse` (it raises `KeyError`); delete it and the script regenerates one. The test
  points themselves are the same 128 prior draws from the same seed, and `log_posterior`/`weights`
  are recomputed by `SamplingFramework` rather than persisted.
- the surrogate state is saved to `surrogate_checkpoint.pt` / `surrogate_training_data.npz` (the
  names `SurrogateRestart` reads), not to the `*_after.pt` / `*_after.npz` names the old scripts
  used — which no restart path ever read, so a restart in those scripts could only ever have
  worked after a manual rename. A script with `SAVE_SURROGATE_STATE=True` and
  `SURROGATE_RESTART_MODE="state"` now overwrites its own input state, which is what "continue
  where the last run stopped" means; point `state_dir` elsewhere to keep the input intact.

**(b) `SurrogateRestart` (`surrDAMH/modules/surrogate_restart.py`), new optional
`SamplingFramework(..., surrogate_restart=...)`.** Applied on the collector rank only, just
before `run_COLLECTOR`; `mode="state"` restores weights+optimizer+snapshots (updater becomes
`pretrained_ready`), `mode="data"` restores snapshots and retrains, `mode="none"` does nothing.
Restored snapshots become `initial_snapshots` only if the updater does not report them itself via
`get_initial_snapshots()`, so the finding-2.8 single counting is preserved. Missing files print a
message and start cold; an incompatible checkpoint still raises. New tests:
`tests/unit/test_surrogate_restart.py` (7 unit tests, incl. the restored evaluator reproducing the
saved one exactly) and `tests/mpi/test_mpi_surrogate.py::test_surrogate_restart_lets_a_damh_first_stage_run`
— run B sets `min_snapshots_initial=1000` so the collector's start-up handshake can only say
"evaluator available" through the `pretrained_ready` branch, and a **DAMH first stage** then runs
to completion (this is otherwise exactly the I4 deadlock configuration).

**(c) `Configuration.describe()` / `Stage.describe()`**, printed once on rank 0 at the start of
`SamplingFramework.run()` and by `run_local()` (~34 lines for a three-stage run). Every dataclass
field appears as `name=value` with `*` on the posterior-affecting ones, after `__post_init__`'s
silent corrections; `Configuration.describe(use_surrogate_gradients_requested=…)` adds a
requested-vs-effective line when `SamplingFramework` disabled gradients. This is the diagnostic
that would have caught the ignored `adaptive_target_rate` (G1). `tests/unit/test_describe.py`
asserts every field appears, so a new field cannot be added silently. **[read]** only — new
stdout at the start of every run; nothing else changes.

**(d) `SolverSpec` stores an absolute `solver_module_path`** (`__post_init__` →
`resolve_module_path()`, re-applied by `SamplingFramework.__init__` and, as a last resort, by
`get_solver_from_spec`). `paths_to_append` is **kept** (no-delete rule) and re-documented as
ineffective for spawned children. The seven `toy_examples/solver_examples/solver_spec_examples.py`
classes now call `super().__init__(...)` instead of assigning the four attributes, so
`__post_init__` actually runs; their relative paths and the "run from `toy_examples/`" contract
are unchanged.

**Today's behaviour, measured before the change (2026-09-17, MPICH/Hydra, `mpiexec -n 2`):** a
relative `solver_module_path` **already worked** — `MPI_Comm_spawn` gave the spawned child the
launcher's working directory (verified by printing `os.getcwd()` inside the child's solver
constructor). So this is a latent portability fix, not a repair of an observed failure. The new
test `tests/mpi/test_mpi_basic.py::test_relative_solver_module_path_reaches_spawned_children`
makes it observable by `os.chdir("/")` between building the spec and `sam.run()`, which is what a
launcher that gives children a different `wdir` looks like to the spec; with the change disabled
that test fails (checked).

**(e) toy_examples import hygiene**: the ruff `F401`/`F811` hits of
`13_dead_code_report.md` (a)/(d) are fixed — imports only, no logic. `ruff check --select
F401,F811 toy_examples` is clean. Note the report says 10 hits, the current tree had 9 (the two
`grf_diffusion.py` duplicates plus 7 unused imports); one had already been removed by the WS6/WS9
work.

**Counts**: `pytest tests -q` 300 → **315 passed, 1 skipped** (+15 unit tests: 7 in
`tests/unit/test_surrogate_restart.py`, 8 in `tests/unit/test_describe.py`; no existing test
changed). `./run_tests.sh mpi` 30 → **32 passed** (+2: the restart test and the relative-path test).
Every maintained example was re-run (`mpiexec` at the `-n` its own docstring gives), all exit 0.

### 2.4 Commit message of `175059e` says "no functional changes" — **[read]**
True for §A/§C/§D/§E of `08_safe_changes_plan.md`; §B is deliberately behavioural for
configurations that previously hung, crashed later, or ran a dead chain (fail-fast asserts,
`artificial_acceptance_multiplicator` removed, `state_dependent_approximation=True` warns).
Posterior/acceptance rate of any configuration that completed before is unchanged.

## 3. Pre-existing bugs the new tests pin but do not fix — **[bug]** / **[decide]**

Each test asserts *today's* behaviour with a docstring citing the finding; flip the assertion
when you fix the bug. None of these are regressions.

| Where | What | Test | Blocked on |
|---|---|---|---|
| `distributions/gaussian_mixture.py` | `grad_logpdf` = 0 and `logpdf` = −inf far from all components (1.11); `rvs()` returns `(1, d)` not `(d,)` | `tests/unit/test_distributions.py` | WS7/WS11 |
| ~~`modules/lhs_normal.py:47`~~ | ~~`maxmin = quality` never assigned → the *last* of 5 candidates is returned (3.8)~~ **fixed, G3 decided 2026-09-17** (§2.13) | `test_helpers.py::TestLHSNormal::test_maximin_best_of_five_candidates_is_returned` (now asserts the argmax) | — |
| ~~`modules/proposals.py` `BlockProposal`~~ | ~~sub-proposals not re-seeded per rank → identical increments on all chains (1.9/A13)~~ **fixed, G5 decided 2026-09-17** (§2.13) | `test_proposals.py::test_block_proposal_subproposals_reseeded_per_rank` (xfail → **passes**) + 3 new reseed tests | — |
| ~~`modules/proposals.py` `GaussRandomWalk_adaptive`~~ | ~~all-zero acceptance weights → `ZeroDivisionError`~~ **fixed in WS7** (§2.9: skip + warn) | `test_proposals.py::test_adaptive_random_walk_all_zero_weights_skips_adaptation_and_warns` | — |
| `modules/proposals.py` `PCN` with non-Gaussian internal prior (1.10) | targets the wrong measure silently → **now rejected at construction (WS7, §2.9)** | `test_proposals.py` pCN guard tests | `FromScipy` eligibility = your decision |
| `surrogates/torch_perceptron*.py` | `PyTorchNNEvaluator.__call__` returns a flattened 1-D array instead of `(n, no_observations)` (3.5) | `tests/unit/test_surrogates.py` | WS6 |
| `surrogates/torch_perceptron_minibatches.py` | `initial_training` persists synthetic rows into `get_training_data_arrays()` (3.4) | `test_surrogates.py::TestInitialTrainingSyntheticRows` | WS6 |
| `post_processing.py` `find_best_fits` | missing log-likelihood/log-prior columns → silent `-inf` score instead of an error (P10) | `tests/unit/test_post_processing.py::TestFindBestFits` | WS9 |
| `post_processing.py` `Samples(load_posterior_surrogate=True)` | `IndexError` on the current on-disk format (5.1) | `test_post_processing.py::TestLoadPosteriorSurrogateKnownUnfixedBug` | WS9 (remove or implement) |
| ~~`process_SAMPLER.py` adaptive `Allreduce` + `proposals.BlockProposal`~~ | ~~`proposal_type="block"` with `adaptive=True` → `AttributeError: 'BlockProposal' object has no attribute 'sd_or_cov'`…~~ **fixed, G2 decided 2026-09-17**: `build_proposal` now rejects the combination up front with a `ValueError` naming the alternative (§2.13) | `test_mpi_hangs.py::test_i6_adaptive_block_proposal_allreduce` (xfail → **passes**, now asserting the fast, diagnosable failure) | — |
| ~~`process_SAMPLER.py` adaptive `Allreduce` with **DAMH**~~ | ~~ranks' adapt counts straddle the period → 1-D vs 2-D `sd_or_cov` → `Message truncated`, exit 9, nondeterministic~~ **fixed, G2 decided 2026-09-17**: both runners normalise to a 2-D covariance on every rank before the reduction (§2.13) | `test_mpi_hangs.py::test_i6_damh_adaptive_allreduce_shape_mismatch` (non-strict xfail → **plain passing test**, asserts exit 0 and no "Message truncated") | — |
| `process_COLLECTOR.py:130,186` | `initial_snapshots` with N rows counted twice: first `snapshots_total` in `surrogate_quality.csv` is **16 for N=8** (2.8) | `tests/mpi/test_mpi_surrogate.py::test_i9_initial_snapshots_are_counted_once` (xfail strict) + passing companion | WS8 |
| ~~raw-buffer path `pickled_observations=False`~~ | ~~with `solver_returns_tag=True` and a solver returning `tag=-1`, the tag is used as an MPI tag → `mpi4py.MPI.Exception: Invalid tag` in the spawned child~~ **fixed, decision 5 applied 2026-09-17**: the raw path and the field are gone (§2.17) | `test_mpi_transport.py::test_i7_negative_solver_tag_completes_and_rejects` (xfail strict → **plain passing test**, asserts exit 0 and that every `solver_tag < 0` row is `rejected`) | — |
| ~~`SurrogateAsSolver` via `use_only_surrogate`~~ | ~~`no_observations=2` → `ValueError: shapes (1,2) and (1,2) not aligned` (A20)~~ **fixed in WS3** (§2.7) | `tests/unit/test_surrogates.py` shape tests; MPI I5 at `no_observations=2` | — |
| `AlgorithmBase._prepare_run` after `use_only_surrogate` | ~~next exact stage starts from surrogate observations (A11)~~ **fixed in WS3** (§2.6) | `tests/test_runner_local.py::test_stage_after_use_only_surrogate_starts_from_exact_observations` | — |

**[read]** The plan's I6 prediction ("adaptive MH stage with `max_evaluations < period` → shape
mismatch") does **not** reproduce: `Algorithm_MH.run` iterates exactly `min(max_samples,
max_evaluations)` times on every rank, so all ranks call `adapt()` equally often. Pinned as a
passing test (`test_i6_adaptive_mh_shorter_than_adaptation_period_does_not_break_allreduce`).
The two rows above are the real reproductions of 2.5.

~~**[decide]** Keep or delete the nondeterministic test `test_i6_damh_adaptive_allreduce_shape_mismatch`?~~
**Resolved 2026-09-17 with G2**: kept, and made a plain passing test (exit 0, no "Message truncated"),
as the note's own default said. The suite no longer has a nondeterministic outcome.

## 4. Documentation corrections — **[read]**

- `07_testing_plan.md` Phase 1 item 8 predicts `matrix_rank(sd_or_cov) ≤ 10` for
  `GaussRandomWalk_adaptive(no_parameters=20, period=10)`. **Wrong today**: `self.samples`
  accumulates across periods (never reset), so the covariance reaches full rank after enough
  periods. `tests/unit/test_proposals.py::test_adaptive_random_walk_reaches_full_rank_covariance`
  asserts the actual behaviour. Consider correcting item 8. The "unbounded history" half of G1
  is settled (2026-09-17): the real issue was memory growth / never forgetting the early chain,
  not rank deficiency, and `Stage.adaptive_sample_limit` now bounds the history when set
  (default `None` = unbounded, i.e. today's behaviour). See §2.13.

- **[read] API friction found while writing the template (WS5):** `use_collector=False` is documented
  as "no surrogate model", but it also forecloses DAMH-SMU entirely — an `Updater` can only be
  driven by the collector rank, so a no-collector DAMH stage needs a pre-trained fixed
  `surrogate_evaluator=` (a different constructor argument), and a newcomer who flips the two
  flags "for a quick single-process run" hits `process_SAMPLER.py assert commEvaluator is not
  None` with no pointer to the fix. Passed to the WS11 docs; a proper fail-fast message would be a
  small follow-up (`build_proposal`/`run_SAMPLER`: "DAMH stage but neither collector nor
  surrogate_evaluator").
- The spec drafts (11/12, below) found four items in the older notes that are **already fixed** in
  the tree and should be annotated there: `04` P2 (`no_unique_samples`, fixed by C11), `03` S3
  (kd-tree exact hit, fixed by C4), `03` S4 (`Evaluator.jacobian` docstring, fixed by D10),
  `03` S0/S1 (import crash / `TestData` call, fixed by A1/A3).

## 5. Validation results and their limits (tests/validation/test_gaussian_toy.py)

Verified 2026-09-16, `algorithms.py` byte-identical to HEAD (nothing was changed to make this
pass), tolerance 4 batch-means standard errors fixed *before* the runs:

| run | evaluations | max |deviation|/SE vs closed form | vs V1 (MH) |
|---|---|---|---|
| V1 MH | 2·10⁵ | 1.12 | — |
| V2 DAMH, wrong fixed surrogate `A'=1.3A`, K=1 / 5 / 20 | 10⁵ / 10⁵ / 6·10⁴ | 0.59 / 1.14 / 2.49 | 1.00 / 1.14 / 1.66 |
| V3 DAMH-SMU, retrained degree-1 polynomial, K=1 / 5 / 20 | 2·10⁴ / 2·10⁴ / 10⁴ | 1.82 / 0.87 / 0.68 | 2.04 / 1.18 / 0.78 |
| V3b DAMH-SMU, retrained 3-NN `KDTreeUpdater` (wrong *and* changing), K=1 / 5 / 20 | 3·10⁴ / 3·10⁴ / 1.5·10⁴ | 0.76 / 2.21 / 1.47 | 0.88 / 1.84 / 0.97 |

V3b outer rejections: 477 / 472 / 351 (so the correction term really is exercised, unlike V3);
prerejected 59456 / 4953 / 12. Runtime 5–8 s per K.

**Conclusion: the DAMH correction is correct for a fixed surrogate at K = 1, 5, 20, and
DAMH-SMU with a surrogate that is refreshed *inside* running sub-chains also reproduces the
posterior at K = 1, 5, 20** (`state_dependent_approximation=False`).

- **[read] V3 alone is weak evidence.** A degree-1 polynomial fits the linear toy *exactly*, so
  after ~20 snapshots the surrogate *is* the model and outer rejections are 0 for all K. That is
  why V3b (kd-tree, added 2026-09-16 by the manager, in the suite) exists: it is the variant that
  stresses finding 1.2. **Implication for WS3:** on this toy the mid-sub-chain refresh does *not*
  produce a detectable bias at 4 SE with 1.5–3·10⁴ evaluations. Finding 1.2 remains a theoretical
  concern (the sub-chain kernel is not reversible w.r.t. a single surrogate posterior when the
  surrogate changes mid-way), but WS3's "freeze the surrogate per sub-chain" change is now a
  *cleanliness/derivation* fix with a regression test in place, not a fix for an observed bias.
  Re-run V3b after WS3; if it still passes, both variants are fine to keep.
- **[read]** Validation chains start from the closed-form posterior mean via
  `initial_sample_type="user_specified"` (deterministic, no burn-in bias). `"prior"` is unseeded
  (1.9, G4) and would make the tests non-reproducible; `"lhs"` starts ~3.7 posterior sd out and
  visibly biased the K=20 chain in a first sweep. Not a code issue, just the reason for the choice.
- **[read]** `tests/test_runner_local.py::test_damh_with_exact_surrogate_never_rejects` is a
  0.2 s canary: with `G~ = G` the outer log ratio is identically 0, so any outer rejection means
  the correction term is wrong. Mutation-tested (sign flip in `algorithms.py` → fails for
  K=1,5,20; source restored and verified identical).
- **[read]** The V1 reference chain is cached module-wide (`_MH_REFERENCE_CACHE`) because the
  `linear_gaussian_problem` fixture is function-scoped. Cosmetic.

## 6. MPI checks — manual ones done, now automated in `tests/mpi/`

WS0 fixes confirmed by automated MPI tests (2026-09-16, `pytest tests/mpi -q -m mpi`):

| Item | Fix | What the test shows |
|---|---|---|
| I1 | — | `-n 2/3/4`: exit 0, exactly `n-1` non-empty `rank%04d.csv`, child prints its evaluation count |
| I2 | — | `-n 1`, no pool/collector: exit 0; chain is **bit-identical** to `run_local()` with the same fixed initial sample |
| I3 | B1 | MH only, collector, `min_snapshots_initial=1e9`: exit 0 in ~5 s (hung forever before). No diagnostic message yet (WS8) |
| I4 | B2 | DAMH as first stage, no `initial_snapshots`: **still deadlocks by design**, killed at 15 s; B2 warning present on stdout. Test asserts the hang + warning; **invert it when WS8's fail-fast lands** |
| I5 | C1 | `[MH, DAMH-SMU, MH(use_only_surrogate=True)]`: exit 0, all three stages' files non-empty (with `no_observations=1`, see A20 in §3) |
| I8 | C9/C10 | continuation: A's `last_sample/<stage>/rank0000.npz` (float64) equals the first row of B's `samples/<stage>/rank0000.csv` **bit-for-bit** (`transform_before_saving=False` in both runs) |
| I11 | A3 | `TestData` object as `surrogate_test_data`: exit 0, `surrogate_quality.csv` + `surrogate_quality_test.csv` written, `n_test=8` in every row |
| I7 | — | since 2026-09-17 (§2.17) the matrix is only `solver_returns_tag ∈ {False, True}`; all cells exit 0, including the negative-tag solver that used to fail on the removed raw path |

Not done: I12 (2-rank protocol unit tests of `CommSnapshot_*`/`CommEvaluator_*`, WS8 acceptance
criterion), I10 (report in pool vs local mode, WS9), I13–I15 (diagnostics, not tests).

Manual checks that preceded the automated ones (same tree, same results):

| Check | Command (from `toy_examples/`) | Result 2026-09-16 |
|---|---|---|
| MH only, solver pool | `timeout 120 mpiexec -n 2 python3 -m mpi4py minimal_example.py` | exit 0, 500 evaluations, CSVs written |
| MH → DAMH-SMU → DAMH with collector, polynomial surrogate, HTML report | `timeout 180 mpiexec -n 4 python3 -m mpi4py typical_example.py` | exit 0, all stages + report |
| B1 hang regression: MH only, `use_collector=True`, `min_snapshots_initial=10**9` | temp script, `timeout 60 mpiexec -n 3` | exit 0 in seconds (would hang before WS0) |

(`mpiexec` is `/usr/local/bin/mpiexec`; `/dolfinx-env/bin/mpiexec` does not exist.
Python: `/dolfinx-env/bin/python3`.)

## 5c. MPI communication + posterior correctness across configurations (2026-09-16, requested by you)

`tests/mpi/test_mpi_posterior.py` (10 tests, `-m mpi`, ~85 s) + shared `tests/helpers_statistics.py`
(also used by `tests/validation`). Tolerance 4 batch-means SE fixed before the first run and never
changed; 2 sampler chains from the deterministic `lhs` design, 10 % burn-in per chain, batches never
straddle a chain boundary; Gelman–Rubin R̂ per parameter. Manager re-ran the full MPI suite on the
final tree: **30 passed, 3 xfailed** (161 s), 0 orphans.

**A — transport is bit-exact.** A1: MH with the solvers pool (`-n 3`) vs the in-process solver
(`-n 2`), same `lhs` starts, 300 evaluations → both chains' `samples/*.csv` byte-identical (float64
parameters + pickled observations survive the pool round trip exactly). A2: DAMH (K=5, frozen) with
a collector-trained polynomial on 40 preloaded snapshots of a *wrong* 1.3·A map vs `run_local()`
with an identically fitted evaluator → `rank0000.csv` byte-identical, with 86 real outer rejections
(so the transported surrogate values are load-bearing). Manifest `runner="mpi"` + layout asserted.

**B — posterior vs closed form** (mean `[0.6154, −0.3416]`):

| case | -n | evals/chain | acc/rej/prerej | pooled mean | max dev/SE mean · cov | R̂ | s |
|---|---|---|---|---|---|---|---|
| B1 MH + pool | 3 | 25 000 | 16688/33312/0 | [0.6200, −0.3469] | 1.57 · 1.89 | 1.000 | 6 |
| B2 MH→DAMH-SMU K=5→DAMH, poly (SMU stage) | 4 | 12 000 | 24000/0/3890 | [0.6177, −0.3430] | 0.76 · 1.05 | 1.000 | 16 |
| B2 … (frozen stage) | 4 | 12 000 | 24000/0/3946 | [0.6128, −0.3412] | 0.93 · 1.63 | 1.000 | " |
| B3 … kd-tree (SMU, wrong & changing; rejections > 0 asserted) | 4 | 12 000 | 23726/274/3836 | [0.6125, −0.3381] | 1.35 · 0.71 | 1.000 | 10 |
| B3 … (frozen stage) | 4 | 12 000 | 23812/188/4022 | [0.6140, −0.3432] | 0.63 · 0.33 | 1.000 | " |
| B4 MH pCN β=0.5 | 3 | 30 000 | 21417/38583/0 | [0.6177, −0.3429] | 0.60 · 0.91 | 1.000 | 6 |
| B5 adaptive MH → MH with `Allreduce`d cov (2nd stage) | 3 | 25 000 | 11718/38282/0 | [0.6182, −0.3366] | 1.15 · 1.18 | 1.000 | 6 |
| B6 in-process solver + collector, DAMH-SMU K=1 | 3 | 15 000 | 30000/0/59751 | [0.6168, −0.3405] | 0.46 · 0.39 | 1.000 | 6 |
| B7 DAMH-SMU K=20 as *first* stage, `initial_snapshots` | 4 | 10 000 | 19478/522/13 | [0.6082, −0.3430] | 2.66 · 0.95 | 1.000 | 10 |
| B8 MH → Hamiltonian on NN-surrogate gradients | 4 | 8 000 | 6092/9908/0 | [0.6331, −0.3485] | 2.42 · 1.63 | 1.000 | 14 |

B7/B8's 2.4–2.7 SE deviations were re-run at 4× / 4× the evaluations: 0.86 and 1.32 SE — they shrink
with sample size, i.e. Monte-Carlo noise, not bias (recorded in the file). B5 also asserts both
samplers reached the `Allreduce` and that the reduced covariance was used (2nd-stage acceptance 23 %
vs target 0.25). Deterministic cases (B1, B4, B5, B6) reproduce identical numbers run to run; the
collector-driven ones (B2, B3, B7, B8) vary slightly with MPI arrival order, all inside tolerance
(5 runs by the agent + 1 by me, all green). **No communication or correctness problem found.**
Not covered: ~~`pickled_observations=False` raw path beyond I7~~ (removed 2026-09-17, §2.17), block
proposals (G2/G5 pending), `state_dependent_approximation=True` (unverified by design).

## 5d. Multi-stage DAMH with a **neural-network surrogate built on the fly** (2026-09-17, your question)

Your question was: "did you check a whole simulation with several sampling stages and an NN
surrogate constructed on the fly?" — before today, no. B2/B3 ran the full `MH → DAMH-SMU(K=5) →
DAMH(K=5, frozen)` chain but with a polynomial / kd-tree surrogate; B8 trained a torch MLP on the
fly but only used it for Hamiltonian *proposal gradients* inside an MH stage, where the chain is
exact for any gradient field whatsoever; V3/V3b stopped after the SMU stage and never used an NN.
Three tests now close that gap (same tolerance as everything else, 4 batch-means SE, fixed before
the first run and never touched):

* `tests/mpi/test_mpi_posterior.py::test_b9_mh_then_damh_smu_then_damh_with_nn_surrogate` — the
  real `mpiexec -n 4` run (2 samplers + solvers pool + collector), the collector training the MLP;
* `tests/validation/test_gaussian_toy.py::test_V3c_damh_smu_with_nn_surrogate_matches_mh` — the
  in-process twin of the SMU stage, K ∈ {1, 5, 20};
* `tests/validation/test_gaussian_toy.py::test_V3d_mh_then_damh_smu_then_frozen_damh_with_nn_surrogate`
  — the in-process **three-stage** chain, including the hand-over into the frozen stage.

Surrogate in all three: `NeuralNetworkUpdaterMinibatches`, hidden layers `(16, 16)`, SiLU, AdamW,
**50 optimizer steps per `train()`**, `weighting="uniform"` and `output_normalization="likelihood"`
at their WS6 defaults (B9 asserts both, from the manifest and from the collector's start-up line).
That budget was chosen so the network stays clearly short of the exact linear model: a surrogate
that becomes exact (as B2's degree-1 polynomial does) collapses the outer acceptance ratio to 1 and
tests nothing. It worked as intended — outer rejections are 482–501 (B9 SMU) and 566–707 (V3c), so
the correction term is load-bearing; the budget did **not** have to be reduced further.

| case | -n | evals/chain | acc/rej/prerej | retrainings | pooled mean | max dev/SE mean · cov | R̂ | s |
|---|---|---|---|---|---|---|---|---|
| B9 MH→DAMH-SMU K=5→DAMH, NN (SMU stage) | 4 | 12 000 | 23505/495/4138 | 104 | [0.6122, −0.3384] | 1.19 · 0.73 | 1.000 | 12 |
| B9 … (frozen stage) | 4 | 12 000 | 23831/169/3986 | " | [0.6111, −0.3429] | 1.53 · 0.46 | 1.000 | " |
| V3c DAMH-SMU, NN, K=1 | 1 | 20 000 | 19434/566/39104 | 101\* | [0.6116, −0.3353] | 1.75 · 0.71 | n/a | 5 |
| V3c DAMH-SMU, NN, K=5 | 1 | 20 000 | 19354/646/3177 | 101 | [0.6172, −0.3406] | 0.55 · 1.30 | n/a | 6 |
| V3c DAMH-SMU, NN, K=20 | 1 | 20 000 | 19293/707/15 | 101\* | [0.6137, −0.3398] | 0.81 · 1.19 | n/a | 13 |
| V3d … (SMU stage, K=5) | 1 | 20 000 | 19354/646/3177 | 101 | [0.6172, −0.3406] | 0.55 · 1.30 | n/a | 10 |
| V3d … (frozen stage, K=5) | 1 | 20 000 | 19951/49/3285 | 100 more, none used | [0.6121, −0.3423] | 1.11 · 0.60 | n/a | " |

Closed-form posterior mean `[0.6154, −0.3416]`. V3c/V3d are single-chain, so no R̂; both also match
the V1 MH reference chain within tolerance. B9 numbers are from run 1 of 3.

\* retrainings are instrumented only in V3d (a test-local `NeuralNetworkUpdaterMinibatches`
subclass counting `train()` calls): 101 up to the end of the SMU stage, 201 in total. V3c has the
identical snapshot budget (20 200 snapshots, `min_snapshots_to_update=200`), so the same 101
applies; for K=5 it is exact, because V3d's SMU stage is seed-identical to V3c's. In B9 the count
is reconstructed from `sampling_output/surrogate_quality.csv` by replaying the collector's
retraining rule (`_collector_retrainings`, documented in the test): 104 = 2 samplers × 26 000
snapshots ÷ 500.

**All three runs each, as requested.** B9: 3/3 pass, max deviation 1.19/1.35/1.12 SE (mean) and
0.73/1.56/1.65 SE (covariance) for the SMU stage, 1.53/0.23/0.81 and 0.46/1.45/1.33 for the frozen
stage; wall time 11.3–12.0 s. A fourth, earlier B9 run (while sizing the test) gave the largest
deviation seen, 2.27 SE (mean) / 2.87 SE (covariance) on the frozen stage — still inside the
tolerance, and it does not repeat. V3c/V3d: 3/3 pass with **byte-identical numbers** each time
(single chain, fixed seeds, no MPI arrival order involved) — the table's V3 rows are exact, not a
sample of a distribution of outcomes.

**What V3d additionally proves.** The frozen stage really keeps the surrogate the SMU stage ended
with. This is *not* "training stops": `send_snapshots_to_collector` defaults to `True`, so the
surrogate manager keeps receiving snapshots and keeps retraining during the frozen stage as well
(100 more trainings) — what stops is the evaluator swap inside the algorithm. The test therefore
counts calls on each evaluator object: evaluator #100 (the last one built during the SMU stage) got
117 283 calls, and every one of the 100 evaluators built after it got **zero**.

**Limits.** The forward model is the 2-parameter linear-Gaussian toy, so "the NN is a poor
surrogate" here means a small MLP under-fitting a linear map — it says nothing about NN surrogate
*accuracy* on a real problem, only that a poor, constantly-rebuilt NN surrogate does not bias the
posterior. Gradient-using NN stages are still covered only by B8 (Hamiltonian proposal inside MH),
not by a DAMH stage with `use_surrogate_gradients=True`.

## 5e. GRF diffusion solver — long-run validation (2026-09-18, requested by you)

Full report: `library_notes/14_grf_validation_2026-09-17.md`; outputs kept in
`toy_examples/out_grf_validation_2026-09-17/` (**14 GB** — delete when reviewed). Finished at
T+1h12 of the 3 h budget. Solver cost 0.19 ms/eval (20×20 dolfinx mesh).

| check | result |
|---|---|
| mechanics (6 runs) | exit 0, 0 orphans, manifests finalized, HTML reports with field statistics |
| R-A reference (pCN, 6 chains × 10⁶ states) | R̂ max 1.00017, pooled ESS ≈ 40 000 |
| **DAMH ≡ MH** (R-B Hamiltonian on NN; R-C pCN K=5 on NN; R-C frozen) | max |Δmean|/SE 2.65 / 2.43 / 1.64, max |Δvar|/SE 2.06 / 1.61 / 2.01, **0 of 120 beyond 3 SE**; posterior predictive agrees within 0.006 noise sd |
| efficiency | ESS per full-model evaluation: MH 0.0066, DAMH-Hamiltonian 0.254 (38×), DAMH-pCN×5 0.031 (4.7×) |
| surrogate held-out RMSE | 0.085 → 0.0012 (R-B), 0.083 → 0.0016 (R-C); 4–5 % of noise sd |
| reproducibility / continuation | two identical runs byte-identical incl. `raw_data`; continuation from R-A bit-exact on all 6 ranks |

Findings: **F1** `write_report` recomputes the field statistics over *every* decompressed state
(6 M solver calls: 36.6 min report for 6.5 min sampling) — `compute_posterior_field_statistics`
has `n_max_samples` but `write_report` never exposed it → **fixed 2026-09-18** (new
`write_report(field_statistics_max_samples=...)`, default `None` = old behaviour; the GRF example
sets 20 000). Measured on a 200 000-state pCN run of the same problem: report 49.6 s with all
states → 11.0 s with `field_statistics_max_samples=1000` (the remaining 11 s are the other report
sections); both reports complete. **[verify]** the field-statistics section of the report is now
a Monte-Carlo estimate when the cap is set (relative error ~1/√N of the posterior spread). **F2** surrogate pre-rejects only 0.3–0.5 % (RMSE ≈ noise/25): DAMH's gain here is the
proposal, not saved solver calls. **F3 [read]** the example is effectively rank 2 in 20 unknowns
(2×2 sensors duplicate across y; one direction above noise): no KL mode is individually informed
(max shrinkage 7.5 %) — any future "mode k is informed" assertion on this example would fail; more
sensors / a source term / longer `length_scale` would be needed to demonstrate inversion.
**F4 (flagged, weak)** R-C SMU variance in the identifiable direction 3.1 SE below R-A, not
reproduced by the other two stages and consistent with R-A's own estimation error (all three share
one reference); a second reference run would settle it.

## 5b. Validation re-run after the WS3 sub-chain freeze (2026-09-16)

`pytest tests/validation -q` → 10 passed. Max |deviation|/SE (tolerance 4):

| | K=1 | K=5 | K=20 |
|---|---|---|---|
| V2 vs V1 / closed form | 1.00 / 0.59 | 1.14 / 1.14 | 1.66 / 2.49 |
| V3 vs V1 / closed form | 2.04 / 1.82 | 1.18 / 0.87 | 0.78 / 0.68 |
| V3b vs V1 / closed form | 0.88 / 0.76 | 1.84 / 2.21 | 0.97 / 1.47 |

V2 and K=1 rows are identical to §5 by construction (unchanged code path); V3/V3b at K=5,20
run on the new, frozen-sub-chain kernel and still match. V3b outer rejections 477/472/351.

## 7a. Spec drafts awaiting your decisions (written 2026-09-16, no code) — **[decide]**

`09_improvement_plan.md` §5 asks for both to be written before WS6/WS9 coding starts.

- **`11_output_format_v2_spec.md`** — today's file table (writer, header, columns, reader, by
  position/name, all file:line), the v2 changes each tied to a finding, the `read_run() -> RunData`
  reader contract, and a grep-verified list of what breaks at the switch (3 test files + 7
  `pd.read_csv(header=None)` sites in `post_processing.py`). **It recommends A30 option (C)**
  (zero-weight endpoint row, `sum(weight) == acc+rej+prerej`, `no_unique_samples` counts
  `weight>0` rows) — i.e. it reverses the (B) default I noted in §7 below; the argument is that
  (B) silently drops the dwell time between the last acceptance and the stage end. Five numbered
  questions at the end; the consequential ones: A30 (C) vs (B); whether the new `obs_approx_*`
  block in `raw_data` exists for MH stages (all-NaN) or only for DAMH; whether `read_run` gets an
  `allow_unversioned=True` escape hatch for read-only inspection of old `out_*` runs (decision 6
  says no converter — this is a weaker ask).
- **`12_evaluator_contract_spec.md`** — conformance table of all five updater/evaluator pairs
  (shapes actually produced, weight handling, train semantics, persistence, picklability, what
  WS1 already pins), the proposed contract (`__call__` always `(n,q)`, single-point `jacobian`/`vjp`,
  `no_parameters`/`no_observations` on the base class, `snapshot_weighting` policy inside the
  updater, opt-in normalisation, `initial_training` fix), and the decision-4 deletion list with
  every reference to `NeuralNetworkUpdaterBasic` (incl. `toy_examples/neural_network_surrogate*.py`,
  `sampling_TSX.py`) plus the Minibatches preset that reproduces full-batch L-BFGS. Five questions;
  the consequential ones: ship `snapshot_weighting="uniform"` as default with no legacy switch
  (it changes NN surrogate training relative to every past run — decision 3 says yes, this asks
  you to confirm "no escape hatch"); classical surrogates keep ignoring `weights` (document only)
  or get a weighted refit; how `output_stats="auto"` picks its snapshot count.

- **`13_dead_code_report.md`** (WS10 input, read-only, `ruff 0.16.7` + `vulture 2.16` installed into
  `/dolfinx-env`): zero ruff findings inside `surrDAMH/`, 10 in `toy_examples/` (unused/duplicate
  imports); 16 symbols grep-verified as unused anywhere (incl. tests/examples/docs) → table (b);
  ~17 "unused but keep/decide" (protocol methods, decided WS6/WS8 items, dynamic use) → table (c);
  5 real-bug-shaped hits → table (d), the notable ones: unreachable code after `raise` in
  `algorithms.py:~317` (`_compute_surrogate_log_likelihood_gradient`), the dead
  `hasattr(prior, "transform")` fallback in `core.py`, and three write-only flags
  (`algorithm_interfaces_local.py::_request_pending`, `::snapshot_count_since_last_update`,
  `communication.py::PendingRequest.active/max_requests`) that look like unfinished logic; 4
  duplicate toy-example pairs → table (e); an 8-step deletion order with the test command per step
  → table (f). **[decide]** which batches to delete; nothing was removed.
  **Resolved 2026-09-17, §7b (decision 19): do not delete dead code** — the report's "safe to
  delete" table is a reference only, not to be acted on.

## 7b. Decisions applied 2026-09-17 (round 2)

- Weighting: `"uniform"` default for **all** surrogates incl. NN; `"multiplicity"` only where supported → WS6 (running).
- Output normalisation from the likelihood (centre = data, scale = noise sd), default on for NN → WS6.
- **Keep** `toy_examples/sampling_TSX.py`, `wrapper.py`, `tunnel_with_subdomains.py`, `out_tsx/` as the FEniCSx example.
- Archived into `TSX_experiments_archived/`: `talk/`, `hpcse26_nn_training/` (6 tracked files → unstaged
  deletions in `git status`, commit them when convenient), `toy_examples/test_grf_diffusion.ipynb`;
  earlier the 10 root scratch files (`root_scratch_archived_2026-09-17/`).
- **Do not delete dead code** — some of it is implemented for planned use; `13_dead_code_report.md`
  stays a reference. (Its "safe to delete" table is therefore *not* to be acted on.)
- Keep the 0.5 s `ABORT_GRACE_SECONDS`.
- Raw-observation MPI path removal (decision 5) → WS8b (running).
- pCN with `FromScipy`: **stays rejected** (already the WS7 behaviour; no change).
- CI: **none** (no workflow file; `run_tests.sh` is the entry point).
- `pytest` default scope: `addopts = -m "not mpi"` added to `pytest.ini` — bare `pytest` = unit +
  validation (301 tests, ~1.5 min); `./run_tests.sh mpi` / `pytest -m mpi` runs the 30 MPI tests
  (an explicit `-m` overrides the default; checked with `--collect-only`).
- Torch threads — **measured 2026-09-17, recommendation: add nothing.** Premise was wrong for this
  launcher: under `/usr/local/bin/mpiexec` (MPICH/Hydra 4.3.1) every rank already starts with
  `torch.get_num_threads() == 1` (OpenMP/MKL max threads throttled to 1 by the launch environment;
  a bare `python` gets 16). 24 runs (HMC-on-NN-gradients and DAMH-SMU-on-NN, 4 thread arms × 3
  interleaved repeats, `-n 4`): all arms within the ±0.2–0.8 s run-to-run spread (wall ≈ 26 s
  both configs); pinning samplers to 1 = no-op, giving the collector 4 threads did not speed up
  training of the small MLP (0.47 vs 0.44 s). The manifest already records `torch_num_threads`
  per run. **[read]** re-measure if the production launcher is different (Slurm `srun`, OpenMPI,
  or `python -m mpi4py` without a launcher may not constrain OpenMP). Scripts kept in the session
  scratchpad `thread_bench/` (not in the repo).
  **Decided 2026-09-18 (§7b): make it explicit anyway** — this measurement only shows the
  behaviour is a no-op *under this specific launcher*; `Configuration.torch_threads` (default
  `1`) now sets it deterministically so a different launcher can't silently oversubscribe.

## 7. Still open from before this work — **[decide]**

- ~~**A30 — stage-boundary state written twice** (finding 5.7)~~ **DECIDED AND IMPLEMENTED
  2026-09-17** — neither option (A)/(B)/(C) of the earlier analysis: the stage-final row keeps
  its full weight and the *next* stage's first row drops the `+1`. Per-stage invariant is now
  `sum(weights) == acc+rej+prerej` for a carried-over stage (`+1` only for a run's first
  stage); concatenation is exact; `no_unique_samples` counts `weight>0` rows. See §2.13,
  `docs/outputs.md` and `11_output_format_v2_spec.md`.
- ~~G1–G6 (`08_safe_changes_plan.md` §G) remain undecided.~~ **ALL SIX DECIDED AND
  IMPLEMENTED 2026-09-17** ("do G1–G6 as recommended"). Evidence per item in §2.13; the
  tests that were xfail'd or pinned to the old behaviour are flipped (§3). G2's two real
  triggers — `adaptive=True` with a block proposal, and adaptive DAMH — are both closed:
  the first now fails fast with a `ValueError`, the second can no longer produce a shape
  mismatch.
- ~~Root-level scratch files (`TSX2_*.md`, `KL_mode_selection_report.md`,
  `MLP_jacobian_stability_report.md`, `test_cuda.ipynb`, `test_toy_shrinkable.html`,
  `loss_during_incremental_training.png`, `wait_for_experiment25_then_run_26.sh`, `talk/`) and
  `hpcse26_nn_training/`: archive or delete? (WS10, needs your go-ahead.)~~ **RESOLVED 2026-09-17**
  (§7b): all archived into `TSX_experiments_archived/` (`root_scratch_archived_2026-09-17/`,
  `talk/`, `hpcse26_nn_training/`) — confirmed by listing, none of these remain at the repo root.
  `library_notes/` itself is still untracked — commit it if you want the notes versioned (this is
  the one item here still open, and it is your call, not a code decision).

## 8. Still open (2026-09-17) — the single authoritative list

Everything from `09_improvement_plan.md`'s workstreams, §4 future work and §6 definition of done
that is **not** done, as of this review. Cross-checked against the code and the fresh
`./run_tests.sh unit|validation|mpi` run (303 passed/1 skipped, 10 passed, 32 passed) — see
`06_findings_consolidated.md` for the finding-by-finding version and `09_improvement_plan.md` §1
status table / §4 / §6 for the workstream-by-workstream version. Grouped as requested:

### (a) Needs the author's decision

- **Commit the uncommitted tree.** Two commits (`175059e`, `1ded39e`) exist; everything else —
  every workstream from WS0 onward — is still uncommitted. See §1's grouped table and suggested
  order. Not a code decision, but nothing else in this list can be "done" in a durable sense
  until it is committed.
- **`library_notes/` itself is untracked.** Commit it (or not) if you want the review versioned
  alongside the code it describes.
- **Finding 1.1** (`state_dependent_approximation=True` with `subchain_max_length > 1`, decision
  1/§4 item 1): needs a derivation of the correct shifted sub-chain kernel and a V4 validation
  before the "unverified" label can be removed. Nobody has proposed a fix; this needs you (or
  whoever picks it up) to decide whether it is worth doing at all, given `False` is the safe
  default already.
- **Finding 5.8 / decision 2** (adaptive proposal target in DAMH — second-stage rate vs overall
  rate, §4 item 2): a real design choice with a re-tuning consequence for `adaptive_target_rate`
  defaults; still open because nobody has been asked to pick one.
- **`HamiltonianInfinite` parametrisation** (decision 8, §4 item 3): mass matrix vs prior
  covariance — deferred, needs a decision if the class is to be used with a non-identity prior.
- **`transformations.*_to_normal`/`beta_to_uniform`** (`13_dead_code_report.md` §c): still an open
  "keep as documented public helper or delete" question from `09` WS10 that the blanket
  "do not delete dead code" decision (19) did not explicitly resolve (that decision was about not
  *deleting*; whether to keep documenting these as public API vs. quietly-unused is still an
  open call).
- **`Samples.pdf_report`/`load_snapshots`/`_load_snapshot_parameters_and_observations`**
  (`13_dead_code_report.md` §c, §f step 6): whether the legacy (non-extended) report path is
  still wanted — explicitly flagged as needing your call before any test could prove deleting it
  safe.
- **`Updater.supports_*_persistence`, `_request_pending`, `snapshot_count_since_last_update`,
  `communication.PendingRequest.active`/`max_requests`** (`13_dead_code_report.md` §c/§d, §f step
  7): possible unfinished logic (write-only flags) — needs someone who knows the original intent
  to say whether a guard was meant to consult them, before touching them.

### (b) Decision-free but not done (left out of scope, or blocked on something other than a decision)

- **WS4 seed architecture, the last mile**: `torch.manual_seed` in
  `torch_perceptron_minibatches.py` and the save/restore-global-`np.random`-state pattern in
  `modules/tools.py`/`modules/test_data.py` still touch global RNG state, and the per-object
  seeding uses `np.random.RandomState(seed)` rather than a literal `SeedSequence.spawn()` tree.
  Left out because WS4's tested acceptance criterion (two identical runs → identical files) is
  already met through the formula in `modules/seeds.py`; finishing the "one architecture, zero
  global RNG use" ideal is cleanup, not a bug fix.
- **WS6 classical-surrogate hardening**: RBF de-duplication/`neighbors` cap/shift-fallback
  removal (finding 3.3) and polynomial `StandardScaler → PolynomialFeatures → Ridge` +
  minimum-snapshot rule (finding 3.10) were out of WS6's "mechanical" scope as actually executed
  (only the weighting contract was mechanical; numerical hardening is a bigger, separate change).
- **WS6 surrogate-quality ESS metric** (finding S10): not implemented; no blocker other than not
  having been scheduled.
- **WS7 efficiency items**: block-group gradients at the full current state instead of zeros
  (`proposals.py` TODO), and surrogate refresh for Hamiltonian proposals inside MH stages —
  both explicitly left because they are efficiency, not correctness, issues (any gradient field
  gives a valid reversible HMC proposal either way).
- **WS7 adaptive-covariance regularisation** for `no_parameters > period` (finding 3.7): the
  crash-causing edge cases (NaN/zero weights) are fixed; singularity for high-dimensional
  problems with a short adaptation period is not, and needs numerical design (shrinkage target,
  regularisation strength), not just a decision.
- **WS8 performance items**: two-step evaluator transfer replacing the 1 GiB `irecv` buffer
  (4.1), batched snapshot `vstack` (4.3), collector/solver busy-wait sleep or blocking `Waitany`
  (4.2), `MPI.TAG_UB` start-up check (2.11), cross-rank `Configuration` consistency
  broadcast+assert (2.9). All performance/hardening, not scheduled in this refactor's scope
  (which prioritised correctness and hangs over throughput, per `09` §0 guiding principle 1).
- **WS8 acceptance criterion I12** (2-rank protocol unit tests for `communication.py`): listed in
  `09`'s own WS8 acceptance criteria but never written; not blocked on a decision, just not done.
- **WS9 bullet 5, remainder**: `write_report` stage selection by name (still takes indices), and
  a report section listing warnings raised during the run (only `unverified_options` from the
  manifest is shown today). Small, mechanical, not scheduled.
- **Finding 2.7 remainder**: pool-mode `solver_instance is None` skip of `par_names`/field
  statistics/solver visualisation was not re-verified fixed or broken.
- **Finding 3.9 remainder**: no input normalisation (`normalize_inputs`) for the NN surrogates,
  no validation split, early stop still on one minibatch's loss.
- **Finding 4.5**: no cached Cholesky for a correlated `Normal.logpdf`; torch evaluator still
  casts the whole module to float64 and back on every gradient call.
- **Finding 4.6 remainder**: `pd.concat` inside loops and the Python-loop `decompress` are
  unchanged (the `chains_to_disp` filtering half of this finding is fixed, WS9b).
- **Finding 4.7**: the `print(..., end="\r")` progress print was not confirmed removed; the five
  superseded `tools.py` helpers (and whatever they import) are deliberately kept, so any
  matplotlib-in-the-sampler import they carry is still there by design (no-delete rule).
- **`algorithm_interfaces*.py` TODO banners** (WS10 table): not shortened; WS2 is functionally
  done but the banners themselves were not revisited.

### (c) Future work by design (deliberately deferred, not "not done")

- **Output-format converter** for archived `out_*` directories (§4 item 4) — decision 6 says no
  converter unless actually needed.
- **`refresh_within_subchain=True`** as a studied DAMH-SMU variant (§4 item 5) — deliberately not
  added (no feature flags by default).
- **Thread pinning under other MPI launchers** (§4 item 6, decision 22) — "add nothing" was a
  measurement under one specific launcher; by design, re-measure before generalising.
- **`state_dependent_approximation=True`, `adaptive_target_rate` semantics, and
  `HamiltonianInfinite` parametrisation** also appear here as "future work" in `09_improvement_plan.md`
  §4 even though they are listed under (a) above as needing a decision — the two are not
  contradictory: the *decision* of whether/how to resolve them is open, but *if* resolved, the
  work itself was always scoped as future work, not part of this refactor.
