# surrDAMH improvement plan (medium refactor, ~5–6 weeks)

Based on notes 00–08 and the author's answers (2026-09-13, see §3 for the recorded decisions):

- **Priority**: correctness and reproducibility of results; then structure and understandability of
  the user's sampling scripts; identification of dead/duplicated code; good documentation of the
  functions users call.
- **Compatibility**: no hard constraints. Breaking changes are acceptable if documented;
  experiments will be re-run. All TSX and Hamilton experiment folders and their `out_TSX*`
  outputs were moved to `TSX_experiments_archived/` on 2026-09-13 and are ignored from now on;
  the hpcse26 results are obsolete as well. **Only `toy_examples/` is maintained.**
- **Scope kept first-class**: torch MLP (minibatches), Hamiltonian/pCN proposals with surrogate
  gradients, classical surrogates (poly/RBF/kd-tree), and a standalone non-MPI runner.
- **Size**: medium refactor — fix Tier 0–3, real test suite, cleaner examples and docs; no
  re-architecture of the MPI roles.

Effort figures are rough working-day estimates for one person; workstreams marked ∥ can run in
parallel. Finding IDs refer to `06_findings_consolidated.md` (tier numbers) and notes 01–05.

---

## 0. Guiding principles

1. **A result is only as good as its test.** Every posterior-affecting change lands together with
   the validation that pins it (Gaussian toy: DAMH with a wrong fixed surrogate must equal MH).
2. **One way to do each thing.** Where the code offers two paths (pickled vs raw observations,
   Basic vs Minibatches NN, three post-processing demos, duplicated helpers in scripts), keep one
   and delete the other.
3. **The user's script is the product.** A sampling script should read top-down as: problem →
   prior/likelihood → solver → surrogate → stages → run → report, in ≈100 lines, with every
   knob documented at its definition.
4. **Everything that affects the posterior is explicit, recorded and versioned**: seeds, defaults,
   surrogate policy, output format version, git commit.
5. **Fail loudly.** No silent hangs, no silently ignored settings, no swallowed exceptions.
6. **Unverified options are labelled as such** in code, warnings and docs, and excluded from
   tests until their theory has been checked — or removed once the theory says they cannot be
   made correct (as `state_dependent_approximation` was, 2026-09-18; no unverified option is
   currently labelled).

## 1. Workstreams

**Status summary (2026-09-17, re-checked against the current tree and a fresh
`./run_tests.sh unit|validation|mpi` run — 303 passed/1 skipped, 10 passed, 32 passed):**

| WS | Status | Remaining (if PARTIAL) |
|---|---|---|
| WS0 Stabilise | **DONE** | — |
| WS1 Test infrastructure | **DONE** (CI stub explicitly declined, §7b); ⚠ correction 2026-09-18: only V1–V3d exist, not "V1–V3, V5–V8" as this table previously said — V4/V5/V6/V8 were never written, see `07_testing_plan.md` | — |
| WS2 Standalone non-MPI runner | **PARTIAL** — ⚠ downgraded 2026-09-18, was marked DONE | `Configuration` is still not an MPI-free dataclass: `configuration.py` imports `mpi4py.MPI` and calls it in `__post_init__`, so `import surrDAMH.modules.algorithms` still loads `mpi4py` transitively (verified with `python -X importtime`). Only the "runs without `mpiexec`" half of the acceptance criterion holds; the "no `mpi4py` import" half does not. Not fixed because nothing in the plan depended on the literal import being absent — `run_local()` and the local adapters all work regardless. |
| WS3 DAMH kernel correctness | **DONE** (decisions 1 and 2 deliberately deferred to §4, not omissions) | — |
| WS4 Reproducibility | **PARTIAL** | eliminate the remaining global-RNG touches (`torch.manual_seed` in `torch_perceptron_minibatches.py:26`; the save/restore-global-state pattern in `modules/tools.py`/`modules/test_data.py`) |
| WS5 Sampling-script structure | **DONE** (2026-09-17, stated in-file below) | — |
| WS6 Surrogates | **PARTIAL** | Classical hardening done 2026-09-18 (RBF de-duplication/`neighbors` cap/shift-fallback removal, 3.3; polynomial `StandardScaler`+`Ridge`+minimum-snapshot rule, 3.10); ESS metric (S10) done 2026-09-18. Left: NN input normalisation (`normalize_inputs`) and the float64 module round-trip per gradient call (S11) |
| WS7 Proposals and gradients | **PARTIAL** | adaptive-covariance shrinkage/regularisation for `no_parameters > period` (3.7) — the only item left; block-group gradients at the full current state and the surrogate refresh for Hamiltonian proposals inside MH stages are **done 2026-09-18** |
| WS8 MPI robustness | **PARTIAL** | two-step evaluator transfer replacing the 1 GiB `irecv` buffer (4.1); batched snapshot `vstack` (4.3). Done 2026-09-18: `MPI.TAG_UB` start-up check (2.11), cross-rank `Configuration` consistency broadcast+assert (2.9), collector busy-wait sleep (4.2) |
| WS9 Output format and post-processing | **DONE** (bullets 1–6 all complete — see the corrected status note in WS9 below) | — |
| WS10 Dead code and duplication removal | **DONE** (author decided "do not delete dead code", §7b; everything deletable was archived) | — |
| WS11 Documentation | **DONE** | — |

### WS0 — Stabilise (2–3 days, first)

**Status: DONE** (commit `175059e` for §A–§F; G1–G6+A30 decided and applied 2026-09-17, `10_manual_review_notes.md` §2.13).

Execute `08_safe_changes_plan.md` in full, **including** its §G items: wire the adaptive `Stage`
fields (G1) ✔, normalise the `Allreduce` shape (G2) ✔, fix LHS selection (G3) ✔, seed `rvs()` (G4) ✔ and
block sub-proposals (G5) ✔, line-buffer CSVs (G6) ✔. G7 is decided: default
`state_dependent_approximation=False`; setting it to `True` emits a warning that the option is
not verified and must not be used unless the user knows what they are doing (08 §B8) ✔.
(G7 superseded 2026-09-18, decision 26: the field and its warning were removed.)

Deliverable: `import surrDAMH` works ✔ (re-verified 2026-09-17); `minimal_example.py` runs ✔; no known start-up crash or
silent-hang configuration remains without a diagnostic ✔ (fail-loud abort, WS8).

### WS1 — Test infrastructure (3–4 days, ∥ with WS2)

**Status: DONE**, except the CI stub, which the author explicitly declined (§7b: "CI: none (no
workflow file; `run_tests.sh` is the entry point)") — a decision, not an omission.

- `tests/unit/` (no MPI): proposals, distributions, transformations, surrogates, post-processing
  pure functions, dataclasses — Phase 1 of `07_testing_plan.md` (items 1–40, excluding the
  `state_dependent_approximation=True` cases). ✔ (`tests/unit/`, 15 files; `./run_tests.sh unit` →
  303 passed, 1 skipped, 10.2 s, re-run 2026-09-17)
- `tests/validation/`: Gaussian-toy statistical checks V1–V3, V5–V8, run through the local runner
  (WS2), marked `slow`. V4 (state-dependent) is deferred, see §4. ✔ (`tests/validation/test_gaussian_toy.py`;
  `./run_tests.sh validation` → 10 passed, 73.0 s, re-run 2026-09-17)
- `tests/mpi/`: subprocess-driven `mpiexec -n k python -m mpi4py …` with `timeout`, asserting on
  produced files; deadlock regressions I3–I11. ✔ (`tests/mpi/`, 5 files; `./run_tests.sh mpi` →
  32 passed, 0 xfailed, 157.9 s, re-run 2026-09-17 — the one long-standing xfail, I7, was closed by
  WS8's raw-path removal, §2.17)
- Shared fixtures: linear Gaussian problem with closed-form posterior; tiny synthetic
  `sampling_output/` tree; wrong-but-fixed surrogate. ✔ (`tests/conftest.py`, `tests/helpers_statistics.py`)
- `pytest.ini` markers (`unit`, `validation`, `mpi`), one `run_tests.sh` ✔, and a CI stub (GitHub
  Actions or a Makefile target) running `unit` on every push and `validation`+`mpi` nightly. ✘
  **decided against** (§7b, 2026-09-17): no `.github/` workflow exists; `pytest.ini`'s
  `addopts = -m "not mpi"` makes bare `pytest` the fast unit+validation gate instead.

Acceptance: `pytest -m unit` < 1 min ✔ (10.2 s); import smoke test present ✔
(`tests/unit/test_imports.py`); the DAMH-equals-MH validation exists even before WS3 makes it
pass for `subchain_max_length > 1` ✔ (V2/V3/V3b, since WS3 landed the same day these pass for
K=1/5/20).

### WS2 — Standalone non-MPI runner (3–5 days)

**Status: PARTIAL** (⚠ corrected 2026-09-18, was DONE — see the status-table note above).

The adapters already exist (`algorithm_interfaces_local.py`); what is missing is a runner and an
MPI-free configuration.

- `surrDAMH/runner_local.py`: `run_local(conf, prior, likelihood, stages, solver, updater=None,
  evaluator=None) -> SamplingResult`, wiring `LocalSolverAdapter`, `LocalSurrogateManager`,
  per-stage proposal construction (shared with `process_SAMPLER` — extract a
  `build_proposal(stage, conf, prior, seed, prev_cov)` function so both runners use one code path).
- Split `Configuration` into an MPI-free dataclass (everything the algorithms need; the existing
  `AlgorithmConfiguration` protocol lists it) plus an MPI layout computed in `SamplingFramework`,
  so `import surrDAMH.modules.algorithms` never imports `mpi4py`.
- Fix `LocalEvaluatorProvider.evaluator_is_available()` to mean "a *new* evaluator is pending"
  (A27/M25), so local DAMH-SMU does not re-evaluate three times per step.
- Single-chain semantics documented: seeds identical to rank 0 of an MPI run, so a local run
  reproduces chain 0 of an MPI run with the same configuration (test this).

Three of four bullets ✔ — `surrDAMH/runner_local.py::run_local`, shared `build_proposal()`
(`modules/proposal_builder.py`), `evaluator_is_available()` fixed (`10_manual_review_notes.md`
§2.1), MPI I2 test asserts bit-identity with `run_local()`. ✘ The "`Configuration` split into
an MPI-free dataclass ... never imports `mpi4py`" bullet was **not done**: `Configuration`
still imports and calls `mpi4py.MPI` in `__post_init__`. It happens not to matter in practice
(`mpi4py` auto-inits a size-1 `COMM_WORLD` under plain `python`, so `Configuration` already
works standalone with `use_collector=False, use_solvers_pool=False` — this is why the runner
and every local-mode test still work), which is presumably why nobody noticed it wasn't done.

Acceptance: `toy_examples/one_process_only.py` runs with `python` (no `mpiexec`, no mpi4py
import) ✔; validations run in pytest in well under a minute each ✔ (73.0 s total for 10 tests).

### WS3 — DAMH kernel correctness (3 days, after WS2)

**Status: DONE** (both "decision deferred" bullets are by-design deferrals to §4, not gaps).

- Freeze the surrogate for the duration of a sub-chain: move the evaluator refresh out of the
  inner loop (1.2). Optionally keep a `refresh_within_subchain=False` flag for experiments, with
  the correction recomputed at the end as `log L̃_final(y) − log L̃_final(x)`. ✔ refresh moved
  before the loop; ✘ no `refresh_within_subchain` flag added (CLAUDE.md: no feature flags by
  default; author did not ask for one, `10_manual_review_notes.md` §2.5 — tracked as §4 future work item 5).
- `state_dependent_approximation`: **no algorithmic change in this plan** (decision 1; superseded
  2026-09-18 by decision 26 — the field was removed outright). Keep the
  current code, default `False`, warn on `True`, document in `Configuration` and `docs/concepts.md`
  that the `True` path is unverified for `subchain_max_length > 1` (finding 1.1). Theory check and
  fix go to §4 Future work. ✔ (deferred as planned)
- Replace the "weird hot fix" comment with the derivation; docstrings on
  `_propose_new_sample_using_subchain` and `_evaluate_surrogate_transition`. ✔
- Solver failures: finite prior term, fail-fast on a non-finite initial state, do not forward
  failed proposals to the surrogate (1.7). Document `solver_tag` (`<0` = failed, observations
  invalid). ✔
- Remove `artificial_acceptance_multiplicator`, or restrict it to the outer decision and record it
  in `notes` (1.5). ✔ removed entirely (commit `175059e`)
- Adaptive proposal target in DAMH: **decision deferred** (decision 2). Keep the current behaviour
  (adaptation sees only sub-chain endpoints, i.e. the second-stage acceptance rate) and state
  this explicitly in the `Stage.adaptive_target_rate` comment and in `docs/stages.md`, so the
  user knows what the number means until the decision is made (§4). ✔ (deferred as planned)
- Stage boundary: stop double-writing the boundary state (A30) — write the final state only in the
  stage that produced it, and let the next stage write it when it is left. ✔ decided and
  implemented 2026-09-17 with a rule different from the one sketched here (the stage-final row
  keeps its full weight; the *next* stage's first row drops the `+1` — see finding 5.7).

Acceptance: V2 and V3 pass for `subchain_max_length ∈ {1, 5, 20}` with and without SMU
(`state_dependent_approximation=False`) ✔ (`10_manual_review_notes.md` §5b); telescoping
invariant unit test passes ✔ (`tests/test_runner_local.py::test_damh_with_exact_surrogate_never_rejects`,
mutation-tested).

### WS4 — Reproducibility (2–3 days, ∥ with WS3)

**Status: PARTIAL.** Manifest, continuation and the DAMH/MH-relevant seed streams (initial
sample, proposal, algorithm, block sub-proposals) are done and tested; the "one seed
architecture, no global RNG use" ideal is not fully realised — see the ✘ items below, none of
which is known to affect any tested configuration's reproducibility today.

- One seed architecture: `np.random.SeedSequence(conf.seed).spawn(...)` → per (chain, stage,
  stream) `Generator`s for proposal, acceptance, initial sample, block-group choice, LHS; torch
  seeded per updater from the same tree; no use of the global `np.random` or `torch.manual_seed`
  inside the library. `Distribution.rvs(generator)`. **PARTIAL**: ✔ a per-(chain, stage, stream)
  seed formula exists (`surrDAMH/modules/seeds.py`) and every posterior-relevant draw (initial
  sample G4, block sub-proposals G5, `Distribution.rvs(generator=...)`) is seeded from it,
  recorded in the manifest; the mechanism is `np.random.RandomState(seed)` per object, not a
  `SeedSequence.spawn()` tree, but is equally deterministic and tested. ✘ `torch.manual_seed(seed)`
  (`torch_perceptron_minibatches.py:27`, ⚠ corrected 2026-09-18, was cited as `:26`) still
  touches torch's *global* RNG (seeded from the tree, but not scoped to the updater); ✘
  `modules/tools.py` and `modules/test_data.py` still save/restore the global `np.random` state
  (`np.random.get_state/seed/set_state`) around a synthetic draw instead of using an owned
  generator; ✘ **not previously listed** (added 2026-09-18): `initial_training`
  (`torch_perceptron_minibatches.py:661`) also draws its synthetic rows via bare
  `np.random.randn(...)`, the global RNG. None of these three is known to be exercised
  concurrently with anything that would make this non-reproducible in practice.
- Record a machine-readable run manifest `sampling_output/run_manifest.json`: full `Configuration`
  and `Stage` list, seeds, surrogate class and hyper-parameters, `output_format_version`, git
  commit + dirty flag, package versions, MPI size/roles, start/end time, and all warnings raised
  at start-up (e.g. the unverified-option warning). ✔ (`surrDAMH/modules/manifest.py`;
  `tests/unit/test_manifest.py`, 7 tests)
- `torch.set_num_threads(conf.torch_threads or 1)` on every rank that imports torch (S18);
  document `OMP_NUM_THREADS`. ✔ **2026-09-18** (§7b, reopening the 2026-09-17 "decided
  against" call): `Configuration.torch_threads: int | None = 1`, applied by
  `apply_torch_threads()` (`surrDAMH/modules/torch_threads.py`) in both runners before any
  network evaluation/training and before the run manifest is built. The 2026-09-17 measurement
  (24 benchmark runs, no effect either way) only held because this container's MPICH/Hydra
  launcher already throttles every rank to 1 torch thread on its own; a launcher that does not
  (Slurm `srun`, OpenMPI, a bare `python` process) would oversubscribe the node without this.
  `torch_num_threads` (effective) and `configuration.torch_threads` (requested) are both in the
  manifest.
- Continuation stores float64 and the manifest of the source run; `load_last_samples` checks
  parameter count and warns on chain-count mismatch. ✔ (WS0 §C9/C10; `tests/unit/test_continuation.py`)

Acceptance: two runs with the same manifest produce identical `samples/` files ✔ (unit test on the
local runner: `tests/unit/test_manifest.py`; MPI test with `-n 4`: I8 continuation bit-for-bit).

### WS5 — Sampling-script structure (3–4 days, after WS2 and WS9's format decision) — **DONE 2026-09-17**

Status: complete. `TestData` is the only test-data path, `SurrogateRestart` is the only restart
path, `SamplingFramework.write_report` the only reporting path, `template_experiment.py` exists,
`Configuration.describe()`/`Stage.describe()` print the effective settings on rank 0, and
`SolverSpec` stores an absolute module path. Acceptance met: `sampling_diffusion_grf.py` is 139
lines with no locally defined helpers, and the template runs end-to-end
(`mpiexec -n 4 python3 -m mpi4py template_experiment.py`, exit 0). Two deviations from the text
below, both from the author's no-delete rule of 2026-09-17: the five superseded `tools.py` helpers
and the `paths_to_append` field are **kept** (documented as superseded/ineffective) instead of
being deleted. Evidence: `CHANGELOG.md` WS5, `10_manual_review_notes.md` §2.18.


Evidence (before archiving): the seven experiment driver scripts (`TSX_complete_experiment_1..4/
sampling*.py`, `sampling_test_Hamilton/sampling_hamilton.py`) were 415–624 lines each and
re-defined the same six helpers (`generate_html_report`, two surrogate-restart helpers, three
test-data helpers) that already exist in the library (`TestData`,
`tools.load_surrogate_restart_state_if_available`, `SamplingFramework.write_report`). Two toy
examples (`toy_example_hamilton.py`, `sampling_diffusion_grf.py`) still import the obsolete
`tools.py` versions.

- Absorb the remaining boilerplate into the library:
  - `TestData.generate/reuse/save` as the only test-data path; delete the obsolete `tools.py`
    functions once the two toy examples use `TestData`.
  - Surrogate restart: one `SurrogateRestart(state_dir, mode="state"|"data"|"none")` helper used by
    `SamplingFramework` (collector rank only), replacing the `tools.py` versions.
  - Reporting: `SamplingFramework.write_report(...)` is the single entry point; rank-0-only with an
    explicit barrier in the caller and a `try/except → MPI.Abort`.
- Provide `toy_examples/template_experiment.py` (≈100 lines) with the canonical section order and a
  one-line comment per knob. The **reference "advanced" example** is
  `toy_examples/sampling_diffusion_grf.py` (pCN + HamiltonianInfinite, minibatch MLP, test data,
  restart, field statistics), re-expressed against the template; `toy_example_hamilton.py` becomes
  the same workflow on the trivial 2-parameter solver.
- `Configuration.describe()` / `Stage.describe()` print the effective settings on rank 0 at
  start-up (this is what would have caught the ignored `adaptive_target_rate`).
- Remove the `paths_to_append` mechanism in favour of `SolverSpec` carrying an absolute module
  path (it does not reach spawned children anyway, M18).

Acceptance: `sampling_diffusion_grf.py` ≤ 150 lines with no locally defined helpers; the template
runs end-to-end on the Gaussian toy.

### WS6 — Surrogates (5–6 days, ∥ with WS5)

**Status: PARTIAL.** The evaluator contract, torch merge/deletion, weighting/normalisation
policy, the surrogate-quality ESS metric and (both 2026-09-18) the classical-surrogate
hardening are done and tested (`10_manual_review_notes.md` §2.16). **Remaining**: the two
torch-side items only — input-side normalisation (`normalize_inputs`, never added) and the
float64 round-trip of the shared module on every gradient call (S11). Nothing classical is
left open.

- **Torch**: merge `torch_perceptron.py` into `torch_perceptron_minibatches.py`
  (single module; `PyTorchMLP`, `PyTorchNNEvaluator`, checkpoint mix-in). **Delete
  `NeuralNetworkUpdaterBasic`** (decision 4); if full-batch L-BFGS is wanted later it is a preset
  of the minibatches updater. Register every updater for `SurrogateReused`;
  `torch.load(weights_only=True)`. ✔ all sub-items done.
- **Evaluator contract** (S4, S12): `__call__` returns `(n, no_observations)` everywhere;
  `jacobian(x) -> (J (q,p), y)` and `vjp(x, v) -> (g, y)` for a single point, documented in
  `parent.py`; add `no_observations` to the base class; remove the `.ravel()`/`.reshape()`
  compensation in `algorithms.py` and `process_COLLECTOR.py`; `SurrogateAsSolver` reshapes
  explicitly (A20). ✔ all sub-items done.
- **Normalisation**: opt-in `normalize_inputs=True` and `output_stats="auto"` (estimate from the
  first N snapshots, then freeze and persist in the checkpoint); today's behaviour (identity)
  stays available. **PARTIAL, superseded design**: ✔ output normalisation shipped, but as
  `output_normalization: Literal["identity","likelihood","manual"]` defaulting to `"likelihood"`
  (statistics from the likelihood, not from the first N snapshots) — `12_evaluator_contract_spec.md`
  deviation 2. ✘ `normalize_inputs` (input-side normalisation) was **not** added; still open.
- **Training-weight policy** (decision 3): parameter `snapshot_weighting` with
  **`"uniform"` as default** (every snapshot, accepted or rejected, has weight 1 in the loss) and
  `"multiplicity"` as the second option, whose docstring and start-up log state clearly that
  rejected proposals (weight 0) are then **not used** for training. The collector keeps sending
  multiplicity weights; the updater applies the policy. ✔ shipped as `weighting=` (renamed from
  the drafted `snapshot_weighting`) on the base `Updater` class, applied to all four surrogates
  (⚠ corrected 2026-09-18: "five" was stale — `NeuralNetworkUpdaterBasic` was deleted by this
  same workstream, leaving polynomial/RBF/kd-tree/minibatches-NN).
- `initial_training` (3.4): train on synthetic rows without storing them, or drop the feature;
  never persist synthetic rows as snapshots. ✔
- **Classical**: kd-tree zero-distance guard ✔; RBF de-duplication + `neighbors` cap + drop the
  shift-fallback ✔ **done 2026-09-18** (finding 3.3: `deduplicate_snapshots()` before every fit,
  `max_neighbors=50` → local fit above 50 snapshots, shifted-copy fallback deleted in favour of a
  smoothing ladder); polynomial `StandardScaler → PolynomialFeatures → Ridge` and a
  minimum-snapshot rule ✔ **done 2026-09-18** (finding 3.10: `alpha=1e-6`, degree `d` only once
  `num_terms(d) < num_snapshots`, so a single snapshot gives a constant fit; the weighting half
  had landed earlier and is preserved, now as `ridge__sample_weight`);
  all three honour or explicitly ignore `weights` (document) ✔ (RBF/kd-tree declare
  `supports_sample_weights = False` and document it; polynomial documents + implements it).
  Both changes alter the exact numbers of every RBF/polynomial run (CHANGELOG "Behaviour
  changes"); DAMH's correction is exact whatever the surrogate returns, so they change
  efficiency and the sample stream, not the sampled posterior.
- **Gradients for HMC**: avoid mutating the shared module on every call (S11) — hold a float64
  copy or compute in float32 with a documented accuracy note ✘ **confirmed still open**
  (⚠ corrected 2026-09-18, was "not verified changed"): `torch_perceptron_minibatches.py`'s
  `jacobian`/`vjp` still call `self.model.double()`/`.float()` on every gradient evaluation
  (lines 128, 142, 162, 178); remove
  the dead `NotImplementedError` fallback in `algorithms.py:287-290` ✘ **still present**
  (`algorithms.py:404-405`, tracked in `13_dead_code_report.md` (d), kept under the no-delete
  rule).
- Surrogate quality monitoring: report ESS of the posterior weights in `surrogate_quality_test.csv`
  (S10). ✔ **done 2026-09-18**: `_compute_surrogate_quality_metrics` now also returns
  `weighted_ess` (Kish's formula, `1/sum(w_i^2)` on the normalized weights; equals `n_test`
  when no weights are supplied), written as a 9th CSV column. Additive-only: existing columns
  and their values are unchanged; readers select columns by name
  (`plots.py::plot_surrogate_quality_test*`), so nothing else needed updating except one
  hardcoded header list in `tests/mpi/test_mpi_surrogate.py` and `docs/outputs.md`.
  `./run_tests.sh unit` unchanged (317 passed, 1 skipped).
- `modules/Gaussian_process.py`: keep as a low-priority optional component (decision 7); fix the
  vector-`std` bug (08 §C5), add a docstring and a symmetry unit test, no further investment.
  ✔ all sub-items done (finding 1.6).

Acceptance: parametrised surrogate test suite (07 §1 items 22–30) green for the four remaining
updaters ✔; gradient finite-difference check passes ✔ (`tests/unit/test_surrogates.py::TestNNGradients`);
checkpoint round-trip exact ✔; MLP on `y = x²` with auto normalisation reaches the same accuracy
as with hand-supplied statistics — **not literally tested as worded** (the shipped design uses
likelihood statistics, not an auto-estimated-from-snapshots `output_stats="auto"`, so this exact
acceptance check does not apply; `TestNNOutputNormalization` covers the shipped design instead);
a test documents the difference between `uniform` and `multiplicity` ✔
(`TestSnapshotWeightingAffectsTraining`).

### WS7 — Proposals and gradients (3 days, ∥)

**Status: PARTIAL.** The fail-fast guards, `BlockProposal` seeding/adaptive-rejection, the two
gradient-efficiency items (done 2026-09-18) and the pCN/`HamiltonianInfinite` decisions are
done; only the adaptive-covariance regularisation is left, see the ✘ row.

- Enforce `proposal.needs_gradients ⇒ conf.use_surrogate_gradients` ✔ (WS7, for every proposal
  type including inside a `BlockProposal`), and refresh the surrogate
  for Hamiltonian proposals inside **MH** stages the same way DAMH does ✔ **done 2026-09-18**
  (author-approved): `Stage.surrogate_model_updates` is tri-state (`None` = the stage type's
  historical default) and may be set `True` on an MH stage whose proposal needs gradients;
  `Algorithm_MH.run` then polls once per iteration via the now-shared
  `AlgorithmBase._refresh_surrogate_evaluator_if_needed` and re-installs the gradient functions.
  Opt-in only — MH stages that do not set it keep one evaluator for the whole stage. Tests:
  `tests/unit/test_config_stages.py` (field semantics), `tests/unit/test_algorithms_local.py`
  (refresh / no refresh), `tests/mpi/test_mpi_posterior.py::test_b10_…` (both stages within 4 SE
  of the closed form, ≥ 2 refreshes vs exactly 0 in the same run).
- `BlockProposal`: per-rank seeding (WS4) ✔ (delivered as G5, `Proposal.reseed()`), and evaluate
  group gradients at the **full current state** instead of zeros for the other groups ✔ **done
  2026-09-18** (author-approved): the block tracks `current_sample` in `propose_sample` and the
  wrapped gradient functions fill the inactive groups from it; bit-identical for a separable
  model, different (and better) for a coupled one, both pinned in `tests/unit/test_proposals.py`;
  `sd_or_cov` for adaptive averaging or forbid `adaptive=True` ✔ **forbidden**
  instead — `build_proposal` raises `ValueError` for `proposal_type="block"` + `adaptive=True` (G2).
- Adaptive RW: bounded history (`adaptive_sample_limit`) ✔ (G1, default `None` = unbounded, i.e.
  unchanged unless the stage opts in), covariance shrinkage/regularisation so
  `no_parameters > period` is not singular (3.7) ✘ **explicitly not done**, NaN-safe weights ✔
  (WS7, degenerate periods skip adaptation + warn instead of crashing), `period` configurable —
  already was (`GaussRandomWalk_adaptive.__init__`'s `period` argument; not newly added).
- `HamiltonianInfinite`: **decided 2026-09-18, keep the mass-matrix contract permanently, no
  behaviour change** (an analysis re-derived the rotation math and confirmed it is exactly what
  was documented, and found a stronger true statement than either the docstring or the test
  previously claimed: energy is preserved exactly for ANY positive-definite mass, not only
  `sd_or_cov==1`). Applied: the class docstring (`proposals.py`) now states the general
  positive-definite-mass result and the permanent mass-vs-prior-covariance contract; the
  `Hamiltonian` class's dangling `docs/concepts.md` pointer is fixed to `docs/stages.md`;
  `tests/unit/test_proposals.py`'s energy-preservation test is parametrized over a scalar,
  diagonal-vector and full-SPD mass instead of only `1.0` (all three pass).
  ✔ (documented, no code change, as decided)
- pCN: assert Gaussian internal prior and `0 < β ≤ 1`. ✔ both (WS7 `build_proposal` guard; WS0
  §B7 `PCN.__init__` assertion).

Acceptance: proposal unit tests 4–10 in `07` pass ✔; ⚠ corrected 2026-09-18: "V5/V6 validations
pass" was **false** — `tests/validation/` has only V1–V3d, no V5 or V6 test exists (see
`07_testing_plan.md`). The 10-test `tests/validation` count this line cited is V1/V2/V3/V3b/V3c/V3d
plus a few from other files, not V5/V6.

### WS8 — MPI robustness (4–5 days, ∥ with WS6)

**Status: PARTIAL.** The fail-fast/hang-elimination half (which is what most of this workstream's
weight is) is done; the pure-performance items (buffer size, batching, tag exhaustion, config
broadcast) are open.

- Fail fast: collector polls stop signals unconditionally (2.1) ✔; before the stage loop the
  collector broadcasts "evaluator available / pretrained: yes/no" so a DAMH/Hamiltonian first
  stage without a surrogate raises on all ranks instead of hanging (2.2) ✔; every role body wrapped
  in `try/except → print traceback → MPI.COMM_WORLD.Abort(1)` ✔ (`SamplingFramework._run_role`,
  `process_CHILD.py`, WS8 fail-loud change).
- ~~**Remove the raw-observation path**~~ (`pickled_observations=False`; decision 5) — **DONE
  2026-09-17** (`CHANGELOG.md` WS8, `10_manual_review_notes.md` §2.17): the configuration field and
  the raw branches in `communication.py`, `process_SOLVER.py`, `process_CHILD.py` are deleted.
  One protocol; `solver_tag` always in the payload (fixes M6 and the dtype hazards by
  construction). `max_buffer_size` stays — it is only the evaluator `irecv` buffer.
- ~~Configuration consistency: rank 0 broadcasts the posterior-relevant fields; every rank asserts
  equality (2.9)~~ — **DONE 2026-09-18**: `communication.check_configuration_consistency()`, first
  call in `SamplingFramework.run()` (inside `_run_role`, so a mismatch aborts the job with a
  traceback). Compares the **requested** values of `POSTERIOR_AFFECTING_FIELDS`
  (snapshotted in `Configuration.__post_init__`), which sidesteps the legitimately per-rank
  *effective* `use_surrogate_gradients`; MPI-layout fields excluded (identical by construction).
  Tests: `tests/mpi/test_mpi_config_consistency.py`, `tests/unit/test_communication.py`.
  No-op for `run_local()` (single process, no MPI call added).
- Evaluator transfer: two-step (size message, then exact-size receive) instead of a 1 GiB
  `irecv` buffer (4.1) ✘ **not done**, `max_buffer_size` (1 GiB default) is unchanged by explicit
  choice (see the WS8 raw-path note above); consider sending `state_dict` + hyper-parameters
  instead of pickling the whole object (M22) ✘ not done.
- Service loops: `Iprobe` + short adaptive sleep, or a blocking `Waitany` when nothing can
  progress (4.2) ✘ **not done** (the dead `if False and …` disabled-alternative branch was
  deleted, WS0 §D6, but the busy-wait itself is unchanged); batch snapshots per loop instead of
  per-snapshot `vstack` (4.3) ✘ not done.
- Tags: sequence numbers in the payload or modulo a safe range (2.11) — already effectively true
  (the request counter is the tag, not a status code, since the raw-path removal); check
  `MPI.TAG_UB` at start-up ✘ **not done** (`grep -rn TAG_UB surrDAMH/` → no hits).
- Fix `initial_snapshots` double count (2.8) ✔ (`10_manual_review_notes.md` §2.12); complete the
  last `Isend`/`Cancel` handling (2.10) — **partial, ⚠ corrected 2026-09-18** (was "likely
  fixed"): the collector-side `TAG_EVALUATOR_OBJECT` send (`send_evaluator`/`terminate`) is now
  waited-on before shutdown, but the sampler-side `TAG_UPDATE` `Isend`
  (`get_evaluator_and_terminate`) is still abandoned without `.Wait()`
  (`communication.py:109-111`). No dedicated protocol test pins either half (I12).

Acceptance: deadlock regression tests I3–I11 pass under `timeout` ✔ (`./run_tests.sh mpi` → 32
passed, re-run 2026-09-17); a deliberately raised exception on any rank terminates the job within
seconds ✔ (WS8 fail-loud: exit 9 in ~3 s / exit 1 in ~2 s, measured in `10_manual_review_notes.md`
§2.10); 2-rank protocol unit tests for `communication.py` ✘ **not done** — `09`'s own I12
acceptance criterion is listed as "not done" in `10_manual_review_notes.md` §6.

### WS9 — Output format and post-processing (4–5 days)

**Status: DONE**, with one narrow item left inside bullet 5 (re-checked 2026-09-17 against the
current tree, correcting the note below written earlier the same day). Bullets 1–5 **done**
(WS9a = output format v2, WS9b = the split and P1/P3/P5/P6/P7/P8/P10; see
`10_manual_review_notes.md` §2.14 and §2.15). Bullet 5's `write_report` stage selection **by
name** is now **done** (2026-09-18): `Samples._resolve_stages` (and therefore every method
built on it, incl. `write_report`/`html_report_extended`) accepts stage names as produced by
`stages.stage_name()` mixed in with the existing positional indices; unit tests
`TestStagesToDispByName` in `tests/unit/test_post_processing.py`. Still open in bullet 5:
listing the *warnings raised during the run* (the manifest records `unverified_options`, which
the report now shows, but no warning log is collected yet) — genuinely open, tracked in the WS9
status table above and in §8. **Bullet 6 is
also done**, contrary to the "needs author go-ahead" note this paragraph carried earlier the
same day: `summarize_tsx2_results.py` and the three `analyze_tsx2_*.py` scripts are confirmed
**not** at the repo root any more (`ls` 2026-09-17) — they now live under
`TSX_experiments_archived/`, alongside the rest of the WS10 archiving.

**Two follow-on additions after this workstream closed (2026-09-18, not separate workstreams):**
`write_report(field_statistics_max_samples=...)` (default `None` = old behaviour; caps the
report's field-statistics section to a Monte-Carlo estimate over a sample instead of every
decompressed state — fixed a 37-minute report on a 6.5-minute GRF run, see
`14_grf_validation_2026-09-17.md` finding F1); and the `post_processing/` package's four
`Samples*` classes were turned from independent mixins into the linear chain
`SamplesStatistics(SamplesBase) → SamplesPlots → SamplesReports → Samples` (purely structural,
fixed spurious `reportAttributeAccessIssue` pyright warnings, no behaviour change).

- **Output format v2** (decision 6, approved, **no converter**): header row in every CSV;
  rectangular `raw_data` (fixed observation block, NaN-filled) with an explicit `state_type`
  column; `format_version` in the manifest; stage directories keyed by stage index **and** name;
  a single `read_run(output_dir)` loader that returns typed arrays. `Samples` refuses (with a
  clear message) to read directories without a manifest / with an older format version.
- Split `post_processing.py` (2450 lines) into `loading.py`, `statistics.py` (means, ESS, R-hat,
  autocorrelation, best fits), `plots.py`, `html_report.py`; `Samples` keeps a thin facade.
- Fix P1–P8: remove or implement `load_posterior_surrogate`; `no_unique_samples`; honour
  `chains_to_disp` everywhere or rename; replace `except BaseException` with specific exceptions
  and real messages; `Autocorrelation` chain count from the stages analysed; close figures in
  `finally`.
- `find_best_fits`: use `rank_best_fit_candidates` (the tested function) or delete it; tests with
  synthetic CSVs for all ranking modes.
- `write_report`: stage selection by name (**done** 2026-09-18); a report section listing the
  effective configuration (from the manifest, done) and warnings raised during the run (still
  open).
- `summary.csv` column names: keep the new short names; delete `summarize_tsx2_results.py` and the
  `analyze_tsx2_*.py` scripts at the root (they belong to the archived experiments) or move them
  into `TSX_experiments_archived/`.

Acceptance: golden-file tests of every output file on a short toy run; post-processing unit tests
34–40 green; report generation on a run with `save_to_file=False` stages and in pool mode
produces the same sections as in local mode.

### WS10 — Dead code and duplication removal (2 days, spread over the others)

**Status: DONE** (every row of the table below has been either archived/deleted or explicitly
decided "keep" by the author — `10_manual_review_notes.md` §7b, 2026-09-17: "Do not delete dead
code — some of it is implemented for planned use; `13_dead_code_report.md` stays a reference.
(Its 'safe to delete' table is therefore *not* to be acted on.)"; the table below is kept
verbatim as the historical per-item record, see the per-row notes added after it).

Done on 2026-09-13: `TSX_complete_experiment_1..4`, `sampling_test_Hamilton`, and all 64
`out_TSX*` directories moved to `TSX_experiments_archived/` (ignored). Remaining items:

| Item | Action |
|---|---|
| `core.temptemptemp`, `Algorithm_PARENT`, commented debug blocks + `evaluate_on_a_grid` import in `algorithms.py`, commented `HamiltonianInfinite.__init__`, `if False and …` in the solver pool, `TAG_READY_TO_RECEIVE`/`TAG_DATA` | delete (08 §A/§D) |
| `Stage.proposal`, `Proposal.subchain_length`, `PriorIndependentComponents.sd_approximation`, `Updater.supports_*_persistence` (no callers), `closest_point_distance*`, duplicated imports in `tools.py`, `run_COLLECTOR(surrogate_delayed_init_data=…)` (never passed) | delete or wire (decide per item) |
| `tools.generate_surrogate_test_data`, `compute_test_log_posterior`, `normalized_weights_from_log_posterior`, `surrogate_restart_state_has_snapshots`, `load_surrogate_restart_state_if_available` | delete after WS5 moves the two toy examples to `TestData` / `SurrogateRestart` |
| `torch_perceptron.py` (`NeuralNetworkUpdaterBasic`, ~250 duplicated lines) | delete (decision 4, WS6) |
| raw-observation MPI path | delete (decision 5, WS8) |
| `neural_network_surrogate_copy.py`; `post_processing_example.py` / `post_processing_with_html_report.py` / `test_html_report_extended.py`; `typical_example.py` vs `typical_example_generic.py` | keep one of each (canonical set in note 05) |
| root-level `analyze_tsx2_*.py`, `summarize_tsx2_results.py`, `wait_for_experiment25_then_run_26.sh`, `TSX2_*.md`, `KL_mode_selection_report.md`, `MLP_jacobian_stability_report.md`, `torch_perceptron_*.csv`, `test_toy_shrinkable.html`, `loss_during_incremental_training.png`, `test_cuda.ipynb` | belong to the archived experiments or are scratch: move into `TSX_experiments_archived/` or delete — **needs the author's go-ahead** (not moved yet) |
| `hpcse26_nn_training/` | results declared obsolete; archive alongside the TSX folders — **needs the author's go-ahead** |
| `toy_examples/out_tsx/` (contains a tracked `sampling_TSX.py`) and `toy_examples/sampling_TSX.py`, `wrapper.py`, `tunnel_with_subdomains.py` | TSX-specific and FEniCSx-dependent; decide whether they stay as the "real solver" example or join the archive |
| `transformations.*_to_normal`, `beta_to_uniform` (unused inverses) | keep as documented public helpers **or** delete; decide |
| `modules/Gaussian_process.py` | keep, low priority (decision 7) |
| `algorithm_interfaces*.py` TODO banners | shorten once WS2 makes the design real |

**Row-by-row resolution (2026-09-17, corrected 2026-09-18)**: row 1 (`temptemptemp`, commented
debug blocks, `evaluate_on_a_grid` import, `Algorithm_PARENT`, `if False and …`) — deleted, WS0.
⚠ **`TAG_READY_TO_RECEIVE`/`TAG_DATA` were NOT deleted** as this row originally claimed —
`grep -n "TAG_READY_TO_RECEIVE\|TAG_DATA" surrDAMH/modules/communication.py:16-17` still finds
both constants, unreferenced. `CHANGELOG.md` correctly lists them under "kept and documented as
unused" (the no-delete decision, 19), so this row contradicted the project's own changelog;
corrected here. Row 2
(`Stage.proposal` etc.) — **kept**, documented as unused/not-read (decided, no-delete rule); the
`run_COLLECTOR(surrogate_delayed_init_data=…)` param in that row is also still present, same
decision. Row 3 (`tools.*` test-data/restart helpers) — **kept**, marked "Superseded by …"
(WS5, §2.18a); duplicate imports in `tools.py` itself were removed (WS0 §D8). Row 4
(`torch_perceptron.py`) — **deleted** (WS6, decision 4). Row 5 (raw-observation path) —
**deleted** (WS8, decision 5). Row 6 (`neural_network_surrogate_copy.py`,
`post_processing_example.py`/`post_processing_with_html_report.py`/`test_html_report_extended.py`,
`typical_example.py` vs `typical_example_generic.py`) — one of each **kept**, the rest
**archived** (confirmed via `git status`: all four show as deleted from `toy_examples/`; listed
in `docs/README.md` as "Archived on 2026-09-17"). Row 7 (root-level TSX scripts/scratch files) —
**archived**, `TSX_experiments_archived/` (confirmed present there, not at root). Row 8
(`hpcse26_nn_training/`) — **archived** into `TSX_experiments_archived/hpcse26_nn_training/`
(confirmed). Row 9 (`toy_examples/out_tsx/`, `sampling_TSX.py`, `wrapper.py`,
`tunnel_with_subdomains.py`) — **decided: keep** as the FEniCSx real-solver example (§7b). Row
10 (`transformations.*_to_normal`, `beta_to_uniform`) — **kept**, test-only, unresolved decision
(still just "keep or delete" per `13_dead_code_report.md` §c). Row 11 (`Gaussian_process.py`) —
**kept**, low priority (decision 7). Row 12 (`algorithm_interfaces*.py` TODO banners) — not
revisited; WS2 is done but the banners were not specifically checked.

`ruff`/`vulture` were run once (`13_dead_code_report.md`, 2026-09-16): 10 `ruff` hits, all in
`toy_examples/` (fixed, WS5 §2.18e — `ruff check --select F401,F811 toy_examples` is clean now),
0 in `surrDAMH/`; `vulture --min-confidence 60` → 73 hits, `--min-confidence 80` → 4 hits,
triaged into the report's tables (b)/(c)/(d), none acted on beyond what is listed above (author
decision: do not delete dead code).

### WS11 — Documentation of the user-facing API (3–4 days, last, ∥ with WS10)

**Status: DONE.** `docs/` has all 7 pages listed below (`concepts.md`, `configuration.md`,
`stages.md`, `writing_a_solver.md`, `writing_a_surrogate.md`, `outputs.md`, `running.md`) plus a
`docs/README.md` index; `CHANGELOG.md` exists and records the posterior-affecting changes.

User-facing surface to document with full docstrings (Args / Returns / Raises / Notes / Example):

- `SamplingFramework.__init__`, `.run()`, `.write_report()`; `Configuration` (every field, which
  are posterior-affecting, which are unverified); `Stage` (every field, invalid combinations,
  the meaning of `adaptive_target_rate` in DAMH); `run_local` (WS2).
- `Solver` (contract: `set_parameters`, `get_observations`, `solver_tag`, `par_names`,
  `visualize_solution`, optional `field_builder/coords/measurement_points`), `SolverSpec`.
- `Distribution`, `Normal`, `PriorIndependentComponents` + components, `GaussianMixture`,
  `FromScipy` — including the internal-vs-physical space design and the "logpdf up to a constant"
  convention (make it uniform).
- `Updater` / `Evaluator` contract; each remaining updater's constructor incl. `snapshot_weighting`.
- `TestData`; `Samples` facade and `read_run` (WS9).
- Proposal classes (parameters, assumptions: pCN Gaussian prior, HamiltonianInfinite `C = I`).

Docs layout: `README.md` (what, install, 10-line quick start, link to docs); `docs/concepts.md`
(Bayesian setting, internal/physical space, MH, DAMH derivation, DAMH-SMU caveats, the unverified
state-dependent option); `docs/configuration.md` and `docs/stages.md` (generated from the
dataclasses); `docs/writing_a_solver.md`; `docs/writing_a_surrogate.md`; `docs/outputs.md`
(format v2); `docs/running.md` (process counts, cluster, continuation, restart); `CHANGELOG.md`
recording the posterior-affecting changes of this refactor (default of
`state_dependent_approximation`, sub-chain surrogate freeze, seeds, weighting default, format v2,
removed raw path and Basic updater).

## 3. Decisions recorded (2026-09-13)

| # | Topic | Decision | Where applied |
|---|---|---|---|
| 1 | `state_dependent_approximation` | ~~Default `False`, `True` marked non-verified, theory check → §4.~~ **Superseded 2026-09-18: the field is removed** (see §3 round 4 decision 26). | WS0 (08 §B8/G7), WS3, WS11 |
| 2 | Adaptive target in DAMH | **Deferred.** Keep current behaviour, document it as "second-stage acceptance rate". | WS3, WS11, §4 |
| 3 | Snapshot weighting for surrogate training | `"uniform"` default; `"multiplicity"` as option with an explicit note that rejected samples are then not utilised. | WS6 |
| 4 | `NeuralNetworkUpdaterBasic` | Delete. | WS6, WS10 |
| 5 | Raw-observation MPI path | Remove. | WS8, WS10 |
| 6 | Output format v2 | Approved; **no converter** for old `out_*` directories (can be added later if really necessary). | WS9 |
| 7 | Experiment scripts / `Gaussian_process.py` | All TSX experiments (incl. `_4_for_paper`) and `sampling_test_Hamilton` archived; only `toy_examples/` maintained. `Gaussian_process.py` kept as a low-priority optional component. | done (archive), WS5, WS6, WS10 |
| 8 | `HamiltonianInfinite` mass vs prior covariance | **Deferred.** Document current semantics only. | WS7, WS11, §4 |

### Decisions recorded 2026-09-16/17 (round 2, from `10_manual_review_notes.md` §7/§7a/§7b)

| # | Date | Topic | Decision | Where applied |
|---|---|---|---|---|
| 9 | 2026-09-16 | `ABORT_GRACE_SECONDS` | Keep the 0.5 s sleep before `MPI.Abort` (needed so Hydra does not discard the traceback). | WS8 §2.10 |
| 10 | 2026-09-16 | pCN with `FromScipy` prior | Stays rejected (no `get_covariance()`/`is_gaussian` flag added). | WS7 §2.9 |
| 11 | 2026-09-17 | A30 stage-boundary weight (finding 5.7) | Neither option (B) nor (C) from the earlier analysis: the stage-final row keeps its full weight; the *next* stage's first row drops the leading `+1`. | WS0/WS9, §2.13/§2.14, `11_output_format_v2_spec.md` |
| 12 | 2026-09-17 | `08_safe_changes_plan.md` §G1–G6 | "Do them as recommended", applied together with decision 11. Two deviations from the recommendation text, both documented in §2.13: G1's `adaptive_sample_limit` default is `None` (unbounded), not 10; G5's sub-proposal seed formula is `2**31 + 1000*block_seed + index`, not `seed + 100*(k+1)` (the latter collides for 11+ samplers). | WS0 |
| 13 | 2026-09-17 | Weighting policy scope | `"uniform"` default for **all** surrogates including the two NN updaters; `"multiplicity"` only where a surrogate actually supports weighting (declared via `supports_sample_weights`). No legacy escape hatch. | WS6, `12_evaluator_contract_spec.md` open question 1 |
| 14 | 2026-09-17 | NN output normalisation | Estimated from the likelihood (centre = observed data, scale = per-observation noise sd), default **on** for the NN updaters (`output_normalization="likelihood"`), not the drafted "auto from the first N snapshots". | WS6 |
| 15 | 2026-09-17 | Classical surrogates and `weights` | Document-only for RBF/kd-tree (declare `supports_sample_weights=False`); `PolynomialSklearnUpdater` gets a real weighted fit instead (a small enough change to include). | WS6, `12_evaluator_contract_spec.md` open question 2 |
| 16 | 2026-09-17 | `neural_network_surrogate.py` / `sampling_TSX.py` migration | Both migrated to the `NeuralNetworkUpdaterMinibatches` full-batch-L-BFGS preset now (author decided to keep the TSX example rather than defer it to WS10). | WS6, `12_evaluator_contract_spec.md` open question 5 |
| 17 | 2026-09-17 | Duplicate toy examples | Keep one of each, archive the rest: `neural_network_surrogate_copy.py`, `typical_example_generic.py`, `post_processing_example.py`, `test_html_report_extended.py` removed from `toy_examples/`. `toy_examples/sampling_TSX.py`, `wrapper.py`, `tunnel_with_subdomains.py`, `out_tsx/` kept as the FEniCSx example. | WS10 |
| 18 | 2026-09-17 | Root-level scratch files and `hpcse26_nn_training/`, `talk/` | Archived into `TSX_experiments_archived/` (`root_scratch_archived_2026-09-17/`, `hpcse26_nn_training/`, `talk/`). | WS10 |
| 19 | 2026-09-17 | Dead code (`13_dead_code_report.md`) | **Do not delete** — some of it is implemented for planned use; the report's "safe to delete" table is a reference only, not to be acted on. | WS10, §6.6 |
| 20 | 2026-09-17 | CI | None — no `.github/` workflow; `run_tests.sh` (+ `pytest.ini`'s `addopts = -m "not mpi"` default) is the entry point. | WS1 |
| 21 | 2026-09-17 | `pytest` default scope | `addopts = -m "not mpi"` added to `pytest.ini`: bare `pytest` runs unit+validation; `./run_tests.sh mpi` / `pytest -m mpi` for the MPI suite. | WS1 |
| 22 | 2026-09-17 | `torch.set_num_threads` | Add nothing — measured under this container's launcher (MPICH/Hydra) every rank already runs with 1 torch thread; re-measure if the production launcher differs. | WS4 §2.9 |

### Decisions recorded 2026-09-18 (round 3)

| # | Topic | Decision | Where applied |
|---|---|---|---|
| 23 | `transformations.*_to_normal`, `beta_to_uniform` (dead-code item, `13_dead_code_report.md` §c) | **Keep**, as documented public helpers. No code change. | — |
| 24 | Legacy (non-extended) report path — `Samples.pdf_report` | **Delete.** Zero callers (confirmed: the toy example that used to call the sibling `html_report` was archived in WS10, and nothing ever called `pdf_report`). Removed from `surrDAMH/post_processing/html_report.py`; `./run_tests.sh unit` unchanged (315 passed, 1 skipped). `html_report`, `load_snapshots` and `_load_snapshot_parameters_and_observations` are **not** part of this decision — asked separately, see `10_manual_review_notes.md` §8. | `surrDAMH/post_processing/html_report.py` |
| 25 | Write-only flags (`Updater.supports_*_persistence`, `_request_pending`, `snapshot_count_since_last_update`, `communication.PendingRequest.active`/`max_requests`) | **Track as a low-priority backlog item, not a decision to make now.** No investigation of original intent, no code change. | tracked in `10_manual_review_notes.md` §8 |

### Decisions recorded 2026-09-18 (round 4)

| # | Topic | Decision | Where applied |
|---|---|---|---|
| 26 | `state_dependent_approximation` (finding 1.1, supersedes decision 1) | **Remove the feature**, do not fix or keep deferred. The theory check asked for in §4 item 1 was done and came out negative twice over: (a) for `subchain_max_length > 1` the code also has a plain arithmetic bug (the sub-chain's "current" likelihood keeps using `G(x₀)` instead of being re-derived at the new position, so the correction does not telescope); (b) more fundamentally, delayed acceptance needs the cheap density to be **fixed and state-independent**, so re-centring the surrogate on the current outer state breaks `Q(y→x)/Q(x→y) = pi~(y)/pi~(x)` at *every* `subchain_max_length`, `1` included — the earlier "K=1 is fine" claim in the docs was itself wrong. No sound repair within this design (re-shifting per intermediate step, or shifting only the first/last step, either collapse to the same bug or need quantities the algorithm never computes). Breaking removal, no shim (no-compat rule). Behaviour-neutral: default was `False`, no maintained example/test/validation run used `True`. | `configuration.py`, `modules/algorithms.py`, `modules/algorithm_interfaces.py`, `modules/manifest.py`, `stages.py`, `docs/`, `CHANGELOG.md`, `tests/unit/`, `toy_examples/{template_experiment,sampling_TSX}.py` |

## 4. Future work (after this plan; each needs a careful theory check or a decision)

1. ~~**`state_dependent_approximation=True` with `subchain_max_length > 1`**: derive the correct
   shifted sub-chain kernel.~~ **Closed 2026-09-18 by removal** (decision 26): the theory check was
   done, the shifted kernel is unsound at every `subchain_max_length`, and the field, its warning
   and the shift branch are gone. V4 is dropped — there is nothing left to validate.
2. **Adaptive proposal target in DAMH** (decision 2): second-stage rate vs overall rate; pick one,
   implement, document, and re-tune `adaptive_target_rate` defaults accordingly. **Untouched** —
   `Stage.adaptive_target_rate`'s docstring documents today's "second-stage rate" meaning.
3. **`HamiltonianInfinite` parametrisation** (decision 8): keep `sd_or_cov` as mass matrix or make
   it the prior covariance and derive the rotation from it. **Untouched** — documented only, per
   decision 8.
4. **Output-format converter** for archived `out_*` directories, only if a past result must be
   re-analysed with the new post-processing. **Not built** — decision 6 explicitly says no
   converter unless it becomes necessary; no request for one has been made.
5. Optional: `refresh_within_subchain=True` as a studied variant of DAMH-SMU with the correction
   recomputed at the sub-chain end. **Not added** — WS3 landed the surrogate-freeze fix without
   this flag (CLAUDE.md: no feature flags by default; `10_manual_review_notes.md` §2.5).
6. **Thread pinning under other MPI launchers** (new, from decision 22): the "add nothing" call on
   `torch.set_num_threads` was measured only under this container's MPICH/Hydra launcher, where
   every rank already runs with 1 torch thread. Re-measure under Slurm `srun`, OpenMPI, or a
   launcher-less `python -m mpi4py` before assuming the same holds there; `torch_num_threads` is
   already recorded per run in the manifest to make this checkable.
7. ~~**Classical-surrogate hardening left out of WS6's mechanical scope**~~ — **done 2026-09-18**:
   RBF de-duplication/`neighbors` cap/shift-fallback removal (finding 3.3) and polynomial
   `StandardScaler → PolynomialFeatures → Ridge` + minimum-snapshot rule (finding 3.10) both
   implemented and tested; see the WS6 section and the CHANGELOG behaviour-change entries.
8. ~~**Surrogate-quality ESS metric**~~ (finding S10) — **done 2026-09-18**: `weighted_ess`
   column added to `surrogate_quality_test.csv`.
9. **MPI performance items left open by WS8** (new): two-step evaluator transfer instead of the
   1 GiB `irecv` buffer (finding 4.1); batched snapshot `vstack` (4.3); collector/solver busy-wait
   sleep or blocking `Waitany` (4.2); `MPI.TAG_UB` start-up check (2.11); cross-rank
   `Configuration` consistency broadcast+assert (2.9); a 2-rank protocol unit test suite for
   `communication.py` (I12, `09`'s own WS8 acceptance criterion, never written).
10. ~~**`write_report` stage selection by name**~~ — **done 2026-09-18** (`Samples._resolve_stages`
    accepts stage names, see WS9 above). Still open: a report section listing warnings raised
    during the run (WS9 bullet 5's remaining item — the manifest's `unverified_options` is shown,
    but no general warning log is collected yet).
11. **WS4's "one seed architecture" ideal** (new): replace the remaining global-RNG touches
    (`torch.manual_seed` in `torch_perceptron_minibatches.py`; the save/restore-global-state
    pattern in `modules/tools.py`/`modules/test_data.py`) with owned generators, and consider
    migrating the per-object `np.random.RandomState(seed)` instances to a
    `np.random.SeedSequence(...).spawn(...)` tree as originally drafted. Not known to affect any
    tested configuration's reproducibility today.

## 6. Definition of done for the refactor

Re-checked 2026-09-17 against the current tree and a fresh test run.

- `pytest -m unit` and `-m validation` green; `-m mpi` green nightly. **MET** (no nightly CI, per
  decision 20, but all three markers pass on demand): `./run_tests.sh unit` → 303 passed,
  1 skipped, 10.2 s; `./run_tests.sh validation` → 10 passed, 73.0 s; `./run_tests.sh mpi` →
  32 passed, 0 xfailed, 157.9 s.
- DAMH and DAMH-SMU reproduce the MH posterior on the Gaussian toy for
  `subchain_max_length ∈ {1, 5, 20}` (with `state_dependent_approximation=False`). **MET.**
  Evidence: `10_manual_review_notes.md` §5/§5b (V2/V3/V3b tables, max |deviation|/SE ≤ 2.5 across
  all cells) plus §5c (B1–B8 MPI posterior-vs-closed-form checks).
- Two identical runs produce identical output files; every run writes a manifest. **MET.**
  `sampling_output/run_manifest.json` on every run (WS4); `tests/unit/test_manifest.py` and MPI
  I8 assert bit-identical re-runs/continuation.
- No configuration known to hang; every error aborts the job with a traceback. **MET** for every
  configuration this refactor's tests exercise (WS0/WS8 fail-loud + start-up handshake). Not a
  universal proof — e.g. the still-open `2.9` (no cross-rank config-consistency assert) and
  `2.11` (no `TAG_UB` check) are latent-hang *candidates* that were never observed to hang and
  were not specifically re-tested here.
- One MPI protocol ✔ (raw path removed, WS8), one torch surrogate module ✔ (`torch_perceptron.py`
  deleted, WS6), one copy of every helper — **partial**: the five superseded `tools.py` helpers
  and `Configuration.paths_to_append` are deliberately *kept* as documented-superseded/ineffective
  rather than deleted (author's no-delete rule, decision 19) — one canonical example per feature
  ✔ (duplicates archived, decision 17), and a ≤150-line reference experiment script in
  `toy_examples/` ✔ (`template_experiment.py`, 97 lines; `sampling_diffusion_grf.py`, 139 lines).
- Every user-facing class and function documented; `docs/` covers concepts, configuration,
  solver/surrogate authoring, outputs, running; unverified options are labelled everywhere.
  **MET** (WS11): 7 `docs/*.md` pages + `docs/README.md`; `Configuration.describe()`/
  `Stage.describe()` print the effective, labelled configuration at start-up.

**Not part of the definition of done but worth naming**: several Tier 3/4 performance and
robustness items (RBF/polynomial hardening, MPI buffer/batching, surrogate-quality ESS, the
"one seed architecture" ideal) remain open — see §4 items 6–11 and the per-workstream ✘ rows
above. None of them was in this section's acceptance criteria, and none is known to affect a
tested configuration's correctness.
