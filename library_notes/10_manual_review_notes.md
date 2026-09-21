# Manual review notes — evidence log and still-open list

Compacted 2026-09-18 from the running log kept during the refactor (was 1354 lines). Deleted:
the uncommitted-file tracking table (§1, now moot — the refactor is committed by the author)
and the session-narrative framing around each entry. Kept: every behaviour change with its
evidence, every validation number, and the authoritative open list (§8). Section numbers are
preserved as-is (not renumbered) because `06`, `08`, `09` and `CHANGELOG.md` cite them by
number (`§2.13`, `§5c`, `§7b`, `§8`, …).

Conventions: **[verify]** = something the author may want to check by hand; **[decide]** =
needed an author decision (now resolved unless still marked open); **[bug]** = pre-existing
defect a test pins without fixing. Finding IDs refer to `06_findings_consolidated.md`; G-items
to `08_safe_changes_plan.md` §G.

---

## 2. Behaviour changes, with evidence

Each entry: what changed, what is proven identical, what is proven different, and the test(s)
that pin it. "Configuration" below means "combination of settings", not the `Configuration`
class.

**2.1 `evaluator_is_available()` semantics.** Old: `True` whenever any evaluator exists. New:
`True` only when a *new* one is pending. Only affects `use_collector=False` DAMH-SMU with a
fixed `surrogate_evaluator=`; there only one surrogate version ever exists, so re-scoring with
it changes nothing (`_get_surrogate_observations` evaluates each point independently). Measured:
evaluator calls per run drop ~3× (e.g. K=1 polynomial: 3439→1147); `samples`/`notes`/
`subchain_stats` byte-identical.

**2.2 `run_local()` continuation.** `run_local` now writes `last_sample/<stage>/rank0000.npz`
after every stage, matching `run_SAMPLER`, so a later local or MPI-chain-0 run can use
`initial_sample_type="continued"`. Test: `tests/test_runner_local.py::test_run_local_writes_last_sample_and_can_continue_from_it`.

**2.5 Sub-chain surrogate freeze (finding 1.2).** `_refresh_surrogate_evaluator_if_needed()`
moved to run once before the sub-chain loop instead of at the top of every inner step, so
`correction_log_ratio` telescopes under one surrogate as the DAMH derivation requires. Changes
the sample stream of every DAMH stage with `surrogate_model_updates=True` and
`subchain_max_length>1`. Identical (verified byte-for-byte): `subchain_max_length==1`, all MH
stages, `surrogate_model_updates=False`. Statistically neutral by theory and by V3/V3b (§5). No
`legacy_subchain_refresh` flag was added (decided against, `09` §4 item 5).

**2.6 Exact re-evaluation after `use_only_surrogate` (finding A11).** Both runners route the
carried-over sample through `algorithms.sample_carried_to_next_stage()`, which drops the
surrogate-derived fields so `_prepare_run` re-evaluates with the full model. New failure mode:
a solver failure exactly at that boundary now raises `RuntimeError` (fail-loud) instead of
silently continuing on surrogate values. Test: `test_stage_after_use_only_surrogate_starts_from_exact_observations`.

**2.7 `SurrogateAsSolver.get_observations` shape (finding A20).** Now always returns
`(no_observations,)` regardless of the evaluator's return shape, fixing the `use_only_surrogate`
+ `no_observations>1` crash. Tests: 6 parametrised shape tests + MPI I5 at `no_observations=2`.

**2.8 Run manifest.** Every run writes `sampling_output/run_manifest.json`: format version,
package/git/environment info, full `Configuration`/`Stage` list, seeds, surrogate class +
hyperparameters, MPI layout, `unverified_options`, `continued_from`. Pure addition. Stage names
are now assigned by `stages.stage_name()` on every rank *before* role dispatch — side effect:
an unknown `algorithm_type` now raises on all ranks at start-up rather than only when a sampler
reaches that stage.

**2.9 WS7 fail-fast guards.** None fires for any configuration that ran successfully before
(checked against every `toy_examples/*.py`). pCN now requires a Gaussian internal prior
(`ValueError` otherwise — `FromScipy` stays rejected by decision, §7b #10);
`needs_gradients ⇒ use_surrogate_gradients` enforced for every proposal type including inside a
`BlockProposal`; adaptive RW with all-zero/NaN acceptance weights in a period now skips the
adaptation + warns instead of `ZeroDivisionError` (non-degenerate sequences bit-identical,
pinned). Explicitly not done (would change working sample streams, several are G-items):
covariance shrinkage for `no_parameters>period`, block sub-proposal reseeding (done separately
as G5), block-group gradients at the full state, Hamiltonian refresh inside MH stages.

**2.10 Fail loud on every rank.** `SamplingFramework.run()`'s `_run_role` wrapper and
`process_CHILD.py` turn any uncaught exception into a traceback + `MPI.COMM_WORLD.Abort(1)`
(measured: exit 9 in ~3s from a spawned child, exit 1 in ~2s from a sampler rank; both hung
forever before). `ABORT_GRACE_SECONDS=0.5` (a `time.sleep` before `Abort`) is kept — without it
Hydra discarded the traceback in 3/3 tries; with it, 5/5 survived (decision 9, §7b).

**2.11 Start-up handshake (finding 2.2).** New `TAG_INITIAL_SURROGATE`: the collector tells
every sampler, before its main loop, whether it can already provide an evaluator. A first
DAMH/Hamiltonian stage with none now raises `RuntimeError` on all ranks instead of deadlocking.
Gap unchanged: a first stage with `use_only_surrogate=True` hits a plain `assert`, now a clean
abort via 2.10 but without the diagnostic message. Tests: `test_i4_damh_as_first_stage_fails_fast`
/ `::test_i4_damh_as_first_stage_with_initial_snapshots_runs`.

**2.12 `initial_snapshots` counted once (finding 2.8).** Preloaded rows counted where consumed,
not up-front-and-again. First `snapshots_total` for N=8 is 8, was 16. Behaviour change: runs
that pass `initial_snapshots` reach the retrain thresholds at different iterations → different
DAMH-SMU sample stream for those runs only.

**2.13 G1–G6 + A30 (decided and applied 2026-09-17).** Author: "do G1–G6 as recommended" plus
A30. Suite deltas for the whole batch: unit 149→172 passed (+22, 2 skipped, 0 xfailed), mpi
30→32 passed (1 xfailed remains, unrelated). Per item:

- **G1** (`Stage.adaptive_*` wired): bit-identical with all three fields unset (checked: same
  `sd_or_cov`, same 80 proposals, same checksum). Changes only stages that set the fields — none
  do in-tree. `adaptive_sample_limit` default `None` (unbounded), not the plan's suggested 10.
- **G2** (2-D covariance before the adaptive `Allreduce`): bit-identical where every rank had
  already adapted (checked via `diff -r` on a reference run). A rank that never adapted now
  sends `diag(sd**2)` instead of a 1-D vector — same distribution, different RNG stream, not
  exercised by any in-tree script. `block`+`adaptive` now raises `ValueError` up front.
- **G3** (`lhs_normal` maximin): `maxmin=quality` now assigned. Changes every
  `initial_sample_type="lhs"` run with 3+ chains (2-chain case happened to be unaffected by
  the bug for seed 0). Measured: min pairwise squared distance for an 8-chain, 3-D design went
  0.0347→0.1268.
- **G4** (per-rank seeded initial sample): new formula in `modules/seeds.py`,
  `initial_sample_seed(no_stages, rank) = 10*no_stages*rank + 3`. Every `"prior"` run changes
  (they were never reproducible before, so there's no "before" value to match) — this *is*
  finding 1.9/A14 being fixed. `"lhs"`/`"continued"`/duck-typed distributions unaffected.
- **G5** (`BlockProposal` per-rank/per-stage reseeding): `subproposal_seed(block_seed, index) =
  2**31 + 1000*block_seed + index` — not the plan's `seed + 100*(k+1)`, which collides for 11+
  samplers with 2 groups. Every block-proposal run changes (was a correctness defect: two ranks
  used to draw the identical increment).
- **G6** (line-buffered CSVs): `open(path, "w", buffering=1)`. No content change, no measurable
  timing change.
- **A30** (stage-boundary state counted once): the first row of a stage whose initial state was
  carried over from the preceding stage drops the leading `+1` (weight = rejections in this
  stage). Measured on a two-stage `run_local` reference: stage 2's row-sum went 26→25 (matching
  `acc+rej+prerej` exactly instead of one too many); a 4-rank `template_experiment.py` run's
  three stages sum to 581 = 580 iterations + 1 (was 582). `no_unique_samples` (which was already
  broken, finding 5.2) is fixed in the same change as `np.count_nonzero(weights)`, so a weight-0
  boundary row isn't counted. The weight sent to the collector for surrogate training is
  unaffected (a different bookkeeping path). 5 new/updated tests in `test_algorithms_local.py`.

**[verify]** on external scripts: (a) a script setting `Stage.adaptive_*` now honours it — past
runs that set it ran at 0.25/0.3/unbounded; (b) any `"lhs"` run with 3+ chains starts from a
different design; (c) any block-proposal run changes; (d) the `multiplicity`/weight column of
every multi-stage run's per-stage first row changes — a script assuming
`sum(weight)==acc+rej+prerej+1` per stage must update.

**2.14 Output format v2 (finding 5.6, `BREAKING`).** No compatibility, no converter (decision
6). `samples`/`raw_data` gain headers; `raw_data` becomes rectangular (`obs_*` exact,
`obs_approx_*` surrogate, both NaN where not applicable); `run_manifest.json` `format_version=2`
is required to read a directory (`RunFormatError`, verified by hand on a fabricated v1
directory). `surrDAMH.read_run()` is the single reader; no positional CSV access remains
anywhere in `post_processing`. Full column reference: `docs/outputs.md`. Test delta: 184→199
passed (+14 `test_read_run.py`, +1 golden-file test un-skipped).

**2.15 Post-processing split and cleanliness (WS9b).** `post_processing.py` (2466 lines) →
package `base.py`/`statistics.py`/`plots.py`/`html_report.py`/`loading.py`/`__init__.py`. No
output format changed, no plot redesigned — verified byte-identical `summary.csv` and
mean/cov/ESS/R-hat dumps on the same run directory across three points in the refactor. Nine
`except BaseException`/bare-`except` sites replaced with specific exceptions and real messages
(finding P6; table of old/new behaviour per site was here, now just: none of them silently
mask a configuration error as "data not available" any more). `chains_to_disp` now honoured
everywhere except `plot_acceptance_rates` and the summary table (chain-insensitive by nature,
now documented as such). Test delta: 199→225 passed (+26: `TestFindBestFits`,
`TestLoadPosteriorSurrogateRemoved`, `TestChainsToDisp`, `TestExplicitExceptions`,
`TestAutocorrelationChainCount`, `TestReportConfigurationSection`, `TestFiguresAreNotLeaked`).

*2026-09-18 follow-on*: the four `Samples*` classes were turned from independent mixins into
the linear chain `SamplesStatistics(SamplesBase) → SamplesPlots → SamplesReports → Samples`
(purely structural — fixed 132→2 spurious pyright `reportAttributeAccessIssue` hits, report
byte-identical, unit gate unchanged).

**2.16 Evaluator/Updater contract, weighting, NN normalisation (WS6, `BREAKING`).** Implements
`12_evaluator_contract_spec.md`. Five changes:

- **(a) weighting default `"multiplicity"` → `"uniform"`.** Rejected proposals (multiplicity 0)
  now count once in NN training instead of contributing nothing. Measured on a 24-snapshot
  toy: training loss 12.03 (old / `"multiplicity"`) vs 12.21 (`"uniform"`, new default) —
  `"multiplicity"` is mathematically the old behaviour but not bit-identical (drops zero-weight
  rows before summing instead of after, float32 order differs by ~5e-8). Classical surrogates
  (poly/RBF/kd-tree) are bit-identical under the new default — they always ignored weights
  outright, which is exactly what `"uniform"` does (pinned by
  `test_classical_surrogates_unchanged_under_default_weighting`). `PolynomialSklearnUpdater`
  gained real `sample_weight=` support for `"multiplicity"`; RBF/kd-tree declare
  `supports_sample_weights=False`.
- **(b) output normalisation `"identity"` → `"likelihood"`.** NN targets centred/scaled by the
  likelihood's mean/noise-sd instead of left raw. Measured on a badly-scaled toy: test RMSE
  54.95 (identity) → 7.65 (likelihood) — shows the setting matters, not that it's always better.
- **(c) `Evaluator.__call__` always `(n, no_observations)`.** The two torch evaluators used to
  return a flattened `(n*q,)`; five compensating `.ravel()`/`.reshape()` call sites removed.
  Value-neutral: a fixed reference DAMH-SMU chain (K=5, poly surrogate) produced byte-identical
  output files before/after.
- **(d) `NeuralNetworkUpdaterBasic` deleted** (decision 4). Replacement preset:
  `NeuralNetworkUpdaterMinibatches(solver="lbfgs", batch_size=None, replay_ratio=0.0,
  train_on_added_data=False)`, used by the two migrated toy examples (decision 16). All four
  remaining updaters registered for `SurrogateReused`; checkpoints load with `weights_only=True`.
- **(e) "weight" renamed "multiplicity"** everywhere in the sampler→collector payload and
  downstream (`add_data`, `.npz` key, `initial_snapshots`'s third array). `TestData`'s posterior
  weights are a different, unrenamed concept. `BREAKING` for a custom `Updater` using the old
  keyword, or old `.npz` files.

Also fixed in this pass: `initial_training` no longer persists its synthetic rows as if they
were real snapshots (finding 3.4).

**2.17 Raw-observation MPI path removed (decision 5, `BREAKING`).** `Configuration.pickled_observations`
deleted; only the pickled sampler↔pool transport survives (message sequence unchanged for
everyone who used the default `True` — `grep` confirms no toy example ever set the field).
Fixes finding 2.4/M6 (a negative solver-error tag used directly as an MPI tag). Tests rewritten:
`test_i7_negative_solver_tag_completes_and_rejects` now a plain passing test (was `xfail(strict)`).

**2.18 WS5: `TestData`/`SurrogateRestart`/`describe()`/absolute solver path.** Five changes,
none altering the posterior for an unchanged script:
- (a) `toy_example_hamilton.py` (131 lines, was 257) and `sampling_diffusion_grf.py` (139 lines,
  was 292) now use `TestData`/`SurrogateRestart`/`write_report` instead of local helpers. The
  five superseded `tools.py` helpers are kept (no-delete rule) with "Superseded by …" docstrings.
  **[verify]**: the test-data `.npz` array names changed (`TestData.save()`'s names, not the old
  `test_parameters`/`test_observations`/…) — an existing file from an older run is not readable
  by `TestData.reuse`, delete and regenerate.
- (b) `SurrogateRestart(state_dir, mode="state"|"data"|"none")`, collector-rank only. 7 unit
  tests + 1 MPI test proving a restart lets a DAMH-first-stage run that would otherwise deadlock.
- (c) `Configuration.describe()`/`Stage.describe()` print every field (with posterior-affecting
  ones marked) on rank 0 at start-up — the diagnostic that would have caught the ignored
  `adaptive_target_rate` before G1.
- (d) `SolverSpec` stores an absolute `solver_module_path` (measured: a relative path already
  worked under this container's MPICH/Hydra by accident — a latent portability fix, not a repair
  of an observed failure; made observable by a new test that `os.chdir("/")` between building
  the spec and running).
- (e) `toy_examples/` made `ruff --select F401,F811` clean.

Suite delta: unit 300→315 (+15: `test_surrogate_restart.py`, `test_describe.py`); mpi 30→32.

**2.19 MPI hygiene, tag diagnostic, idle throttle (2026-09-18, findings 2.10/2.11/4.2).** Three
independent changes, none altering the posterior, the sample stream or any output file:
- (a) `CommEvaluator_sampler.get_evaluator_and_terminate()` `Wait()`s its pending `TAG_UPDATE`
  `Isend` instead of dropping the reference (2.10). Same wire protocol (same messages, same tags,
  same order), so no matching change on the collector side. Pinned by
  `tests/mpi/test_mpi_protocol.py` (I12), verified to fail with the fix reverted.
- (b) `communication.check_tag_upper_bound()`, called on rank 0 in `SamplingFramework.run()`
  before role dispatch (2.11): `RuntimeWarning` if the projected tag count
  (`sum(min(max_evaluations, max_samples))` + `TAG_FIRST_SNAPSHOT`) exceeds `MPI.TAG_UB`, or if a
  stage has no evaluation/sample bound and `TAG_UB < 2^20`. Diagnostic only — it swallows every
  exception and returns the message it warned about, and it never changes control flow. Silent on
  every maintained example (this container's MPICH 4.3.1 reports `TAG_UB = 1073741823 = 2^30 − 1`).
- (c) `ServiceLoopThrottle` in the solvers-pool and collector loops (4.2): an iteration that did
  no work counts as idle; 100 consecutive idle iterations spin exactly as before, each further one
  sleeps 1 ms. **[verify]** the only user-visible effect is CPU use and, in principle, latency.
  Measured (4 ranks, MPICH, 32-core container, load ~24 throughout, 3 runs each):
  `typical_example.py` user CPU 4m17–4m27 → 3m36–3m41, wall 55.9–58.2 s → 55.7–57.9 s;
  `minimal_example.py` wall 9.4–11.6 s → 8.6–9.4 s. The spin budget is load-bearing: an
  unconditional 1 ms sleep per idle iteration (no spin) put `minimal_example.py` at 13.6–14.6 s,
  because its analytic solver turns a request around in far less than 1 ms.
- (d) `communication.check_configuration_consistency()`, first call in `SamplingFramework.run()`
  (2.9, 2026-09-18): rank 0 `bcast`s its **requested** values of
  `Configuration.POSTERIOR_AFFECTING_FIELDS` (snapshotted in `__post_init__`, before
  `_configure_surrogate_gradients()` can mutate anything) and every rank compares field by field;
  an `allreduce` makes all ranks raise, `_run_role` turns that into traceback + `MPI.Abort(1)`.
  Requested values are compared precisely because *effective* `use_surrogate_gradients` may
  legitimately differ per rank; MPI-layout fields are excluded (derived from `COMM_WORLD` size).
  Behaviour change **only** for a script whose `Configuration` genuinely differs per rank: that
  used to run (divergent posteriors), hang, or die later somewhere unrelated — verified: with the
  call removed, the deliberate-mismatch driver reaches the sampler and dies in
  `numpy.random` with `shape mismatch: objects cannot be broadcast to a single shape`. No
  maintained script is affected: `./run_tests.sh mpi` (37 passed) and
  `mpiexec -n 4 … typical_example.py` are unchanged. `run_local()` makes no such call.

**2.4 Commit `175059e`'s "no functional changes".** True for §A/§C/§D/§E of
`08_safe_changes_plan.md`; §B is deliberately behavioural only for configurations that
previously hung, crashed later, or ran a dead chain. Posterior/acceptance rate of any
configuration that completed before is unchanged.

**2.20 WS7 efficiency items (2026-09-18, author-approved).** Two independent changes, both
"valid kernel either way, better trajectory now"; neither touches an acceptance formula.
- (a) **Block-group gradients at the full current state.** `BlockProposal` tracks the full
  current sample (`current_sample`, set in `propose_sample`) and its wrapped gradient functions
  fill the inactive groups' coordinates from it instead of zeros. Proven identical for a
  separable (block-diagonal) model and proven different for a coupled one:
  `tests/unit/test_proposals.py::test_block_proposal_gradient_change_is_a_no_op_for_a_separable_model`
  / `…_moves_the_trajectory_for_a_coupled_model`. Affects only stages whose proposal is a
  `BlockProposal` containing a Hamiltonian-family sub-proposal; no maintained example or test
  configuration uses one, so nothing in the tree changes.
- (b) **Surrogate refresh for Hamiltonian proposals inside MH stages.** `Stage.surrogate_model_updates`
  became tri-state (`None` = each stage type's historical default) and may now be `True` on an
  MH stage whose proposal needs gradients; `Algorithm_MH.run` then polls once per iteration with
  `AlgorithmBase._refresh_surrogate_evaluator_if_needed` (moved up from `Algorithm_DAMH`,
  behaviour there unchanged) and re-installs the gradient functions when the evaluator really
  changed. MH stages that do not opt in are untouched — proven in the same MPI run by B10's
  second stage (0 refreshes on both ranks, against ≥ 2 in the opted-in first stage). Posterior of
  both stages within the suite's 4 SE.

**2.21 Adaptive proposals in DAMH stages see pre-rejected iterations (2026-09-20, author-requested;
finding 7.3, `16` §5 item 1).** One statement added in the pre-rejected branch of
`Algorithm_DAMH.run`: `self.proposal.adapt(proposed_sample=<current state>, log_acceptance_probability=-inf)`.
`Proposal.adapt` is `pass` for non-adaptive proposals, so only `DAMH` stages with `adaptive=True`
can change. Pinned by `tests/unit/test_algorithms_local.py::test_damh_adaptive_proposal_sees_prerejected_iterations`
(on the old code: 9 `adapt()` calls for 400 outer iterations). Suites after the change: unit 388 /
validation 14 / mpi 38.
Old-vs-new comparison, same seeds, 4 ranks, old code run from a pre-change snapshot via
`PYTHONPATH` (module path recorded per run): `toy_examples/out_damh_adapt_fix_2026-09-20/RESULTS_COMPARISON.md`
(+ `results_comparison.csv`, `identity_W.csv`, `figs/`, 18 runs, all exited 0, no time limit hit).
- **Author's workflow W** (adaptive MH → DAMH-SMU with the carried covariance, `adaptive=False` →
  fixed-surrogate DAMH): toy — all 27 posterior-bearing files (`samples`, `notes`, `last_sample`,
  3 stages × 3 ranks) **bit-identical**; the 7 files that differ (`subchain_stats` ×6,
  `surrogate_quality.csv`) differ only in `correction_log_ratio` by ≤ 3e-11, and the same files differ
  by the same amount between two runs of *one* version (asynchronous collector retraining), so not
  the change. GRF — bit-identity **could not be checked**: the NN-collector run is not reproducible
  by itself (stage-1 identical in all four runs, stage 2 onwards differs within a version as much as
  across); statistically indistinguishable (stage-3 ESS 438/458 old vs 471/503 new; means within
  1.1–1.8 SE of the pooled reference in all 20 parameters).
- **Adaptive DAMH stage (A1: K=1, A5: K=5)**: old code reproduces the `15` §3.3 divergence on both
  problems — proposal sd inflated ≈70× (toy K=1), ≈160× (toy K=5), ≈5× (GRF), 99.93 % pre-rejection,
  390/9000 (toy) and 400/6000 (GRF) exact evaluations before the `max_samples` net; new code: full
  budget, pre-rejection 63–69 %, outer acceptance 0.31 vs target 0.25, sd 0.49 (toy; posterior sd
  0.3) / 0.36–0.54 (GRF; stage-1 scale 0.43–0.71). A1-long (GRF, 8000 evals/chain): same picture
  (409 vs 24 000 evaluations).
- **Stage 3 with the covariance carried from the adaptive DAMH stage**: old — 8–36 exact evaluations
  of 12 000–60 000, 2–11 distinct states per chain, posterior sd 22–61 % wrong (8.3–22.3 SE off);
  new — like the hand-tuned W stage 3 (toy ESS 19 735 vs 26 362, closed form within 0.37 SE / 0.17 %
  sd; GRF ESS 427 vs 471, 2.1 SE of the pooled reference). K=5 is still over-scaled ≈ 3.3× relative
  to K=1 (valid posterior, 0.71 SE / 1.2 % sd on the toy) and its carried covariance costs the next
  stage 94 % pre-rejection (`docs/stages.md` and the proposal docstring now say so).
- **Cost**: non-adaptive stages unchanged (W wall clocks equal within noise). Adaptive DAMH stages:
  one `adapt()` call per outer iteration instead of per moved iteration, each appending a zero-weight
  row to the unbounded history (`np.cov` re-scan every 10 iterations → quadratic); isolated
  microbenchmark +0.9 s (p=2, 30 k iterations) / +2.1 s (p=20, 17.5 k) — a few per cent of the
  measured 3 s stages here, more for long stages; `adaptive_sample_limit` bounds it.
- **New, expected warnings**: 1–2 per adaptive DAMH stage, at its start, while the history holds ≤ 1
  moved iteration — `np.cov` with a single non-zero weight gives `Degrees of freedom <= 0` /
  `divide by zero` / `invalid value`, caught by the WS7 degenerate-weights guard, which keeps the
  previous covariance and emits its `RuntimeWarning`. (The guard sums the whole history, so this
  cannot recur later in the stage.) Not a failure; documented in the proposal docstring.
  Listed in §8(b) as a candidate for silencing. (Moot since §2.22: the guard and its warning were removed
  with the rewrite.)

**2.22 Adaptation reachable through `Proposal.adapt()`: random walk rewritten, pCN and Hamiltonian step
size new (2026-09-20, author-requested; `16` §5 items 2, 4 (step size only), 6).** Implemented by an
opus agent from the manager's specification, reviewed line by line against the pre-change snapshot;
one correction by the manager (below). Suites after the change: unit 397 / validation 14 / mpi 40.
- **Hook.** `Proposal.adapt(proposed_sample, log_acceptance_probability, current_sample,
  subchain_log_acceptance_probabilities=None)`, called AFTER the accept/reject decision in
  `Algorithm_MH.run` and in both DAMH branches (`current_sample` = post-decision state; DAMH also
  passes the sub-chain's per-step surrogate acceptance probabilities, returned as a fourth value of
  `_propose_new_sample_using_subchain`). Moving the call was proven stream-neutral with the OLD adapter
  still in place: identical sha256 of all `samples/*.csv` for an `MH-adaptive → MH(None)` and an
  adaptive DAMH `run_local` chain before/after the move.
- **`GaussRandomWalk_adaptive`.** Welford `(n, mean, M2)` of post-decision states; every 10 calls once
  `n ≥ 100`: `C = M2/(n−1)`, `δ = min(1, 2d/n)`, `C ← (1−δ)C + δ(tr C/d)I`,
  `base = (2.38²/d)(C + 1e-6 (tr C/d) I)` (relative ridge — the prototype's absolute `1e-6` assumed
  whitened parameters); `log σ += n^-0.7 (α − 0.234)` every call; draw `x + e^{log σ} L z`.
  `sd_or_cov` kept equal to `e^{2 log σ} base` (2-D) for every existing reader. Removed: `corr_limit`
  clip, `sample_limit` window, `coef` feedback, sample/weight history, degenerate-weights guard.
  Unit measurement (d10 κ10 Gaussian, 20 000 steps from `sd = 1`): acceptance 0.228, shape error 0.078,
  scale 1.12× the `2.38²/d·C` optimum.
- **`PCN_adaptive`.** `logit β += n^-0.7 (α − 0.234)`; unit: acceptance 0.234, β 0.128 on a κ = 100 toy.
- **`Hamiltonian_adaptive` / `HamiltonianInfinite_adaptive`** (mixin `_DualAveragingStepSize`).
  Hoffman–Gelman Alg. 5, δ = 0.8, γ = 0.05, t₀ = 10, κ = 0.75, `μ = log(10 ε₀)`; one update per
  sub-chain step in DAMH (surrogate acceptance), one per iteration in MH; frozen value `ε̄`; `|log ε − μ|`
  clipped at 50 (overflow guard, never fires when converged). `HamiltonianInfinite` recomputes its
  rotation angles from `step_size` per call, so nothing is stale. Unit (d10 Gaussian, exact gradients,
  mass I, 4000 steps): acceptance 0.80, `ε̄` 0.280 from `ε₀ = 0.01` vs 0.277 from `ε₀ = 1.0`, stability
  limit 0.447. Mass and `num_steps` untouched.
- **Hand-over.** `adapted_state()` (fixed-length float64) → one `Allgather` on the sampler communicator
  → `set_pooled_state()` (RW: Chan et al. combination of `(n, mean, M2)` = statistics of the concatenated
  chains, mean `log σ`; pCN: mean logit β; Hamiltonian: mean `log ε̄`) → `carry_over()` dict keyed by
  Stage field name, merged into `carried` and consumed by `build_proposal` for every field left `None`
  (`proposal_sd_or_cov`, `pcn_beta` (else 0.5), `hamiltonian_step_size` (else 0.1)). Replaces the
  `Allreduce` average of covariances (`15` §2.4: 6–17× rank disagreement). Single-row pooling keeps the
  chain's own state exactly (`run_local` ≡ MPI chain 0). Buffer shapes are fixed per class, so the I6
  shape-mismatch hazard cannot recur (the three I6 tests pass unchanged).
- **`Stage`.** `adaptive_corr_limit`/`adaptive_sample_limit` removed (`TypeError` if passed);
  `pcn_beta`/`hamiltonian_step_size` tri-state with the old defaults as fallback (no existing
  configuration changes value); the pCN `adaptive=True` force-off removed; `adaptive_target_rate=None`
  → class default 0.234/0.234/0.8.
- **Output.** `adaptive_stats/<stage>/rank%04d.csv`, one row per period (columns per class + `rank_world`),
  loaded by `read_run`/`Samples` (`None`/empty for non-adaptive stages); `docs/outputs.md`.
- **Manager's correction.** The agent had routed a Hamiltonian stage's `proposal_sd_or_cov=None` through
  `carried` too, i.e. a random-walk covariance became the mass; the old code fell back to `1.0` and
  `16` §4.3 measured `M = Σ̂` at 0.06–1.2 of oracle. Reverted to the unconditional `1.0`
  (`proposal_builder.py`, `docs/stages.md`, CHANGELOG); carrying `Σ̂⁻¹` remains `16` §5 item 5's decision.
- **Agent's deviations accepted:** `rank_world` column in `adaptive_stats`; RW skips installing a
  covariance when `tr C/d ≤ 0` (chain never moved) silently; B12 checks only the frozen stage's
  posterior (the adaptive stage's own covariance is 5.8 SE high during warm-up — a transient, shrinking
  along the chain, and unbiased with exact gradients; documented in the test).
- **Tests.** `test_proposals.py` +9 / −4 (target-rate-scale-shape, pooling identities, all-`-inf` period,
  pCN target, dual averaging ×2, builder resolution, `TypeError` on the removed fields);
  `test_algorithms_local.py` +2 `run_local` hand-over runs (adaptive RW → fixed, adaptive pCN → fixed:
  frozen acceptance in band, posterior within 4 SE) and 2 extended; MPI B5 rewritten to the pooled
  hand-over, B11 (adaptive pCN → `pcn_beta=None`) and B12 (adaptive Hamiltonian → `step_size=None`) new.
- **Efficiency** (`toy_examples/out_adaptivity_impl_2026-09-20/RESULTS_EFFICIENCY.md`; 63 runs, 5 chains
  + collector, serial, module path recorded per run; closed-form toys d2 and d10 κ10, GRF 20-D;
  references = fixed-parameter grids on the baseline package, best cell marked; frozen stage after a
  3000-evaluation adaptive stage with the carried parameter; ESS = min-over-parameters batch means):
  | | adaptive RW old rule | adaptive RW new | adaptive pCN | adaptive HMC step (after §2.23) | best grid cell |
  |---|---|---|---|---|---|
  | d2 ESS/eval | 0.097 | 0.122 | 0.121 | 0.576 | RW oracle 0.108 · pCN β 0.5 0.105 · HMC ε 0.01 0.088 |
  | d10 ESS/eval | 0.018 | 0.024 | 0.016 | 0.268 | RW oracle 0.028 · pCN β 0.3 0.014 · HMC ε 0.5 0.015 |
  | GRF ESS/eval | 0.016 | 0.020 | 0.056 | 1.233 | RW sd 0.5 0.016 · pCN β 1.0 0.091 · HMC ε 0.02 0.113 |
  | carried acceptance | 0.23 / 0.24 / 0.17 | 0.24 / 0.23 / 0.20 | 0.24 / 0.24 / 0.24 | 0.83 / 0.81 / 0.87 | |
  RW: 1.13 / 0.88 / 1.25 × the best grid cell (old rule 0.90 / 0.65 / 1.08, and it missed the acceptance
  window on GRF); toy posteriors within 1.7 SE, sd within 3 %; the old rule's 5 non-PSD warnings on GRF
  gone. pCN: 1.16 / 1.08 / 0.61 × — the GRF miss is the target, not the rule: the ESS optimum there is the
  independence sampler β = 1.0 (acceptance 0.20), which a 0.234 target cannot express. HMC step: ≥ 6.6 ×
  the best fixed step everywhere, tuning-stage acceptance 0.80 on all three against the 0.8 target,
  carried ε 0.294 / 0.279 / 0.209 — **confounded upward**: the H frozen stage's surrogate saw twice the
  training data of a grid cell's; the direction is not in doubt, the factor is. **Intended GRF workflow**
  (pCN → DAMH-SMU + HamiltonianInfinite + NN updates → DAMH frozen), example as shipped vs every step
  size adapted: shipped β 0.2 / ε 0.05 accepts 97 % and pre-rejects 0.6 % (the screening does nothing);
  adapted β 0.85 / ε 0.27 accepts 0.82–0.85, pre-rejects 20 %, **ESS/eval 0.106 → 0.92–1.21 (8.8–11.4 ×),
  ESS/s 29.5 → 221–284** (two runs; the GRF NN run is not reproducible, §2.21). GRF posterior: self-
  consistency only (the two workflow runs agree with each other, sds within 3–5 %; their SE-distance to
  the pool is large because the pool is dominated by far poorer runs). Cost: RW/pCN stages unchanged
  in wall clock; ESS/s comparable only within a problem. Before-fix Hamiltonian traces kept in
  `runs_before_fix/`.
- **Not done / open:** `T` parametrisation and mass carry-over (`16` §5 items 4b, 5), pCN per-mode
  variances, ESS in `summary.csv` (item 3), K > 1 over-scaling of the RW in DAMH (structural).

**2.23 Non-finite proposals and non-finite surrogates are now guarded (2026-09-20, found by the §2.22
efficiency study; finding 3.12 in `06`).** The adaptive Hamiltonian collapsed on two of three problems
(step size → 1e-22, 0 accepted; `toy_examples/out_adaptivity_impl_2026-09-20/RESULTS_EFFICIENCY.md`
§2.3, before-fix traces kept in `runs_before_fix/`). Causal chain, each link verified from the run
outputs: (1) dual averaging's first iterations overshoot upward by design (`μ = log 10ε₀`), and with
the example's 100 leapfrog steps a few early trajectories ran off to astronomically large or
non-finite positions; (2) such a proposal is rejected with probability 1, but the library still
evaluated it exactly (GRF: `exp` overflow in the solver) and forwarded it to the collector as a
training snapshot; (3) the NN updater's weights went NaN on it (`surrogate_quality.csv`: rmse = nan
from snapshot 19 874 (d2) / 10 124 (GRF, ≈ 25 iterations into the stage) onward — none of the 18
fixed-step grid runs sharing the same bootstrap ever did); (4) the collector published the NaN
evaluator, the stage (`surrogate_model_updates=True`) installed it, every gradient became NaN, every
proposal non-finite, acceptance probability 0 at ANY ε, so the recursion drove ε to its clamp; the
frozen stage inherited the NaN model. A pre-existing robustness gap (nothing rejected a NaN
evaluator anywhere), exposed by the new adaptation. Fixed by an opus agent, manager-specified,
without touching the adaptation rule:
- `AlgorithmBase`: a proposal with non-finite parameters is rejected WITHOUT calling the solver and
  without a snapshot, through the existing failed-solver path (`solver_tag = -2`,
  `SOLVER_TAG_NONFINITE_PROPOSAL`, `log_likelihood = -inf`); `Algorithm_MH` sets `log α = -inf`
  explicitly (a Hamiltonian's momentum term would be NaN), so `adapt()` gets the Stan-style
  "divergent transition" signal α = 0; the DAMH sub-chain skips the surrogate for such a step and
  appends `-inf` (RNG consumption unchanged). Counter `counter_nonfinite_proposals`, one
  `RuntimeWarning` per stage, stage-end line extended (`notes` CSV untouched).
- `Hamiltonian._leapfrog` / `HamiltonianInfinite._leapfrog` abort once `q` or `p` is non-finite
  (saves the remaining surrogate-gradient calls); bit-identical on finite trajectories — pinned by
  20 unit cases against the verbatim old loop body.
- Collector: a retrained evaluator that is non-finite on the newest snapshot batch is NOT published
  (previous evaluator kept, one print per failure streak and one on recovery). NN updater: `add_data`
  drops non-finite rows (after the float32 cast); `train()` snapshots the weights and rolls back —
  weights AND optimizer (manager's addition: Adam moments from NaN gradients would re-poison) — when
  the pass produced non-finite weights.
- Sampler: `_refresh_surrogate_evaluator_if_needed` probes the arriving evaluator at the current
  state (one call + one `vjp` when gradients are used) and refuses a non-finite one unless it is the
  first; `counter_nonfinite_evaluators`, one warning per stage.
- Tests: +32 unit (non-finite MH proposal: no solver call, no snapshot, `-inf` to `adapt`; DAMH
  sub-chain; leapfrog identity/abort; updater skip + rollback; refresh guard; collector probe).
  Suites: unit 429 / validation 14 / mpi 40. Unverified: the collector's keep-previous branch through
  a full MPI collector loop (helper-level test only). The optimizer-rebuild line was added after the
  agent's mpi run; it lives in the rollback branch only and is covered by the unit rollback test.
- Remaining limitation (documented, not fixed): a proposal that is astronomically large but FINITE
  is still evaluated exactly and still reaches the NN as a snapshot; the NN has no input
  normalisation or outlier guard (Tier 3). The dual-averaging overshoot that produces such
  proposals is bounded in practice by the α = 0 feedback (d10 recovered by itself).
- Efficiency after the fix: see the §2.22 efficiency bullet / `RESULTS_EFFICIENCY.md` (rerun of the
  four Hamiltonian runs).

## 3. Pre-existing bugs the new tests pin but do not fix — [bug] / [decide]

Each test asserts *today's* behaviour with a docstring citing the finding; flip the assertion
when the bug is fixed. None are regressions; strikethrough items below have since been fixed
and their test flipped from pinning the bug to pinning the fix.

| Where | What | Test | Status |
|---|---|---|---|
| `distributions/gaussian_mixture.py` | `grad_logpdf`=0, `logpdf`=−inf far from components (1.11); `rvs()` shape `(1,d)` | `test_distributions.py` | still open |
| `surrogates/polynomial_sklearn.py` | weights discarded before WS6 | — | fixed, G-independent (WS6) |
| `modules/lhs_normal.py` | `maxmin` never assigned (3.8) | `TestLHSNormal` | fixed (G3) |
| `modules/proposals.py::BlockProposal` | sub-proposals not reseeded (1.9/A13) | `test_block_proposal_subproposals_reseeded_per_rank` | fixed (G5) |
| `modules/proposals.py::GaussRandomWalk_adaptive` | all-zero weights → `ZeroDivisionError` | `test_adaptive_random_walk_all_zero_weights_skips_adaptation_and_warns` | fixed (WS7) |
| `modules/proposals.py::PCN` + non-Gaussian prior (1.10) | targeted wrong measure silently | pCN guard tests | fixed (WS7), `FromScipy` stays rejected by decision |
| `surrogates/torch_perceptron*.py` | flattened evaluator output (3.5) | `test_surrogates.py` | fixed (WS6) |
| `torch_perceptron_minibatches.py` | `initial_training` persisted synthetic rows (3.4) | `TestInitialTrainingSyntheticRows` | fixed (WS6) |
| `post_processing` `find_best_fits` | missing log columns → silent `-inf` (P10) | `TestFindBestFits` | fixed (WS9) |
| `post_processing` `Samples(load_posterior_surrogate=True)` | `IndexError` | — | fixed: option removed (WS9) |
| `process_SAMPLER.py` adaptive `Allreduce` + `BlockProposal` | `AttributeError` | `test_i6_adaptive_block_proposal_allreduce` | fixed (G2) |
| `process_SAMPLER.py` adaptive `Allreduce` + DAMH | shape mismatch, nondeterministic | `test_i6_damh_adaptive_allreduce_shape_mismatch` | fixed (G2) |
| `process_COLLECTOR.py` | `initial_snapshots` double-counted (2.8) | `test_i9_initial_snapshots_are_counted_once` | fixed (WS8) |
| raw-buffer path + negative solver tag | `Invalid tag` MPI exception | `test_i7_negative_solver_tag_completes_and_rejects` | fixed: path removed (decision 5) |
| `SurrogateAsSolver` + `use_only_surrogate` | shape mismatch (A20) | shape tests + MPI I5 | fixed (WS3) |
| `AlgorithmBase._prepare_run` after `use_only_surrogate` | surrogate obs leaked into next stage (A11) | `test_stage_after_use_only_surrogate_starts_from_exact_observations` | fixed (WS3) |

**[read]** the testing plan's item 8 prediction (`matrix_rank(sd_or_cov) ≤ 10` for `period=10`)
is wrong: `self.samples` accumulates across periods, so rank eventually reaches full.
`test_adaptive_random_walk_reaches_full_rank_covariance` asserts the real behaviour; corrected
in `07_testing_plan.md`.

## 4. Documentation corrections — [read]

- **API friction (WS5):** `use_collector=False` forecloses DAMH-SMU entirely (an `Updater` can
  only be driven by the collector rank); a no-collector DAMH stage needs a pre-trained fixed
  `surrogate_evaluator=`. Hitting `assert commEvaluator is not None` with no pointer to the fix
  is a known rough edge, documented in WS11's docs, not otherwise fixed.
- Four items from the deleted per-subsystem notes (01–05) were confirmed already fixed and
  annotated accordingly before deletion: `no_unique_samples` (C11), kd-tree exact hit (C4),
  `Evaluator.jacobian` docstring (D10), the import crash / `TestData` call (A1/A3).

## 5. Validation results and their limits (`tests/validation/test_gaussian_toy.py`)

Verified 2026-09-16, `algorithms.py` byte-identical to the pre-change tree (nothing was changed
to make this pass); tolerance 4 batch-means SE fixed *before* the runs.

| run | evaluations | max \|deviation\|/SE vs closed form | vs V1 (MH) |
|---|---|---|---|
| V1 MH | 2·10⁵ | 1.12 | — |
| V2 DAMH, wrong fixed surrogate `A'=1.3A`, K=1/5/20 | 10⁵/10⁵/6·10⁴ | 0.59/1.14/2.49 | 1.00/1.14/1.66 |
| V3 DAMH-SMU, retrained degree-1 polynomial, K=1/5/20 | 2·10⁴/2·10⁴/10⁴ | 1.82/0.87/0.68 | 2.04/1.18/0.78 |
| V3b DAMH-SMU, retrained 3-NN kd-tree (wrong *and* changing), K=1/5/20 | 3·10⁴/3·10⁴/1.5·10⁴ | 0.76/2.21/1.47 | 0.88/1.84/0.97 |

**Conclusion: the DAMH correction is correct for a fixed surrogate at K=1,5,20, and DAMH-SMU
with a surrogate refreshed inside running sub-chains also reproduces the posterior at K=1,5,20**
(`state_dependent_approximation=False`). V3 alone is weak evidence (a degree-1 fit becomes exact
for this linear toy, so outer rejections → 0); V3b (kd-tree) is the variant that stresses
finding 1.2, with 351–477 real outer rejections, and it passes too — so on this toy the
mid-sub-chain refresh produces no detectable bias; WS3's fix is a derivation-cleanliness fix
with a regression test, not a fix for an observed bias. Validation chains start from the
closed-form posterior mean (`initial_sample_type="user_specified"`) to avoid burn-in bias;
`"prior"` was unseeded before G4 and `"lhs"` visibly biased K=20 in a first sweep.

**5b. Re-run after the WS3 sub-chain freeze.** 10 passed. V2/K=1 rows identical to §5 by
construction; V3/V3b at K=5,20 run on the new frozen-sub-chain kernel and still match (V3b outer
rejections 477/472/351).

**5c. MPI communication + posterior correctness across configurations (10 tests, ~85s).**
Tolerance 4 batch-means SE, 2 sampler chains, `lhs` starts, 10% burn-in, Gelman–Rubin R̂.
**Transport is bit-exact** (pool vs in-process solver, A1; collector-shipped evaluator vs
`run_local()`, A2). **Posterior vs closed form** (mean `[0.6154, −0.3416]`):

| case | evals/chain | max dev/SE mean·cov | R̂ |
|---|---|---|---|
| B1 MH + pool | 25 000 | 1.57·1.89 | 1.000 |
| B2 MH→DAMH-SMU K=5→DAMH, poly (SMU / frozen) | 12 000 | 0.76·1.05 / 0.93·1.63 | 1.000 |
| B3 … kd-tree, wrong & changing (SMU / frozen) | 12 000 | 1.35·0.71 / 0.63·0.33 | 1.000 |
| B4 MH pCN β=0.5 | 30 000 | 0.60·0.91 | 1.000 |
| B5 adaptive MH → MH with `Allreduce`d cov | 25 000 | 1.15·1.18 | 1.000 |
| B6 in-process solver + collector, DAMH-SMU K=1 | 15 000 | 0.46·0.39 | 1.000 |
| B7 DAMH-SMU K=20 as *first* stage, `initial_snapshots` | 10 000 | 2.66·0.95 | 1.000 |
| B8 MH → Hamiltonian on NN-surrogate gradients | 8 000 | 2.42·1.63 | 1.000 |

B7/B8's 2.4–2.7 SE deviations shrank to 0.86/1.32 SE at 4× the evaluations (Monte-Carlo noise,
not bias). **No communication or correctness problem found.** Not covered here: block proposals
(`state_dependent_approximation=True` was the other gap; the feature was removed 2026-09-18).

**5d. Multi-stage DAMH with an NN surrogate built on the fly (author question, 2026-09-17).**
Closes a gap: earlier B2/B3 used poly/kd-tree, B8 only used NN gradients inside MH. Three tests
(`test_b9_mh_then_damh_smu_then_damh_with_nn_surrogate`, `test_V3c_damh_smu_with_nn_surrogate_matches_mh`,
`test_V3d_mh_then_damh_smu_then_frozen_damh_with_nn_surrogate`), NN budget deliberately capped
(50 optimizer steps/train) so it stays short of exact — outer rejections 482–707, so the
correction term is load-bearing.

| case | max dev/SE mean·cov | R̂ |
|---|---|---|
| B9 SMU / frozen stage (3 runs) | 1.19·0.73 / 1.53·0.46 (worst of 3) | 1.000 |
| V3c K=1/5/20 | 1.75·0.71 / 0.55·1.30 / 0.81·1.19 | n/a (single chain) |
| V3d SMU / frozen | 0.55·1.30 / 1.11·0.60 | n/a |

V3d additionally proves the frozen stage keeps using the SMU stage's last evaluator even though
training continues in the background: evaluator #100 got 117 283 calls, every evaluator built
after it got zero. Limits: the forward model is the 2-parameter linear toy, so this shows "a
poor, constantly-rebuilt NN surrogate does not bias the posterior," not NN accuracy on a real
problem; DAMH-with-NN-*gradients* (as opposed to NN-in-MH-only) is still untested.

**5e. GRF diffusion solver — long-run validation (2026-09-18).** Full report:
`14_grf_validation_2026-09-17.md`; outputs in `toy_examples/out_grf_validation_2026-09-17/`
(**14 GB, off-limits, delete only when the author reviews it**).

| check | result |
|---|---|
| mechanics (6 runs) | exit 0, 0 orphans, manifests finalized |
| R-A reference (pCN, 6×10⁶ states) | R̂ max 1.00017, pooled ESS ≈ 40 000 |
| **DAMH ≡ MH** (R-B Hamiltonian/NN; R-C pCN K=5/NN; R-C frozen) | max \|Δmean\|/SE ≤ 2.65, max \|Δvar\|/SE ≤ 2.06, **0/120 beyond 3 SE** |
| efficiency (ESS/eval) | MH 0.0066, DAMH-Hamiltonian 0.254 (38×), DAMH-pCN×5 0.031 (4.7×) |
| reproducibility | two identical runs byte-identical; continuation bit-exact |

Findings: **F1** `write_report`'s field-statistics loop over every state made a 37-min report
for 6.5 min of sampling → fixed with `write_report(field_statistics_max_samples=...)` (default
`None`=old behaviour). **F2** surrogate pre-rejects only 0.3–0.5% (the DAMH gain here is
proposal quality, not saved solver calls). **F3** the example is effectively rank-2 in 20
unknowns (no individual KL mode informed above 7.5% shrinkage) — any future "mode k is
informed" assertion on it would fail. **F4** (weak, flagged) one variance estimate 3.1 SE off,
not reproduced elsewhere, consistent with reference-run noise.

## 6. MPI checks (automated in `tests/mpi/`)

| Item | Fix | What the test shows |
|---|---|---|
| I1 | — | `-n 2/3/4`: exit 0, `n-1` non-empty output files |
| I2 | — | `-n 1`, no pool/collector: bit-identical to `run_local()` |
| I3 | B1 | collector never trains, `min_snapshots_initial=1e9`: exit 0 in ~5s (hung before) |
| I4 | WS8 handshake | DAMH first stage, no `initial_snapshots`: fails fast with a message (used to hang) |
| I5 | C1 | `[MH, DAMH-SMU, MH(use_only_surrogate=True)]`: surrogate available in the last stage |
| I6 | G2 | adaptive `Allreduce` + block / + DAMH: fails fast / completes, no shape mismatch |
| I7 | decision 5 | `solver_returns_tag ∈ {False,True}` incl. a `tag=-1` solver: all exit 0, rejected correctly |
| I8 | C9/C10 | continuation bit-for-bit across a pool-mode run |
| I9 | WS8 | `initial_snapshots` counted once, not twice |
| I11 | A3 | `TestData` object as `surrogate_test_data`: exit 0 |

**Not done**: I12 (2-rank protocol tests for `communication.py` — ties to finding 2.10's open
half), I10 (an MPI-integration test asserting pool vs local mode report parity — finding 2.7's
content gap itself is fixed 2026-09-18, but no automated test compares the two reports;
manual check only, see finding 2.7), I13–I15 (`TAG_UB`/RSS/busy-wait diagnostics, not
correctness tests).

## 7. Decisions — pointer

The full decision log (what was decided, when, and why) lives in `09_improvement_plan.md` §3
(round 1, 2026-09-13) and its "round 2" subsection (2026-09-16/17, decisions 9–22) plus the
2026-09-18 additions (torch threads, `field_statistics_max_samples`). Not duplicated here.
Two items from the spec drafts (`11_output_format_v2_spec.md`, `12_evaluator_contract_spec.md`)
that shaped a decision but aren't in 09's table: A30 was decided as "neither (B) nor (C)" from
the two options `11`'s draft weighed (see finding 5.7); `12`'s draft asked whether classical
surrogates should get a weighted refit or stay document-only — decided: `PolynomialSklearnUpdater`
gets a real weighted fit, RBF/kd-tree stay document-only (`supports_sample_weights=False`).

The `13_dead_code_report.md` "safe to delete" table was **not acted on** — decision 19: do not
delete dead code, some of it is implemented for planned use; that report stays a reference only.

## 8. Still open — the single authoritative list

Cross-checked against the current code on 2026-09-18 (not just against the 2026-09-17 text).
Corrections from that check are marked **⚠**. See `06_findings_consolidated.md` for the
finding-by-finding version and `09_improvement_plan.md` §1/§4/§6 for the workstream version.

### (a) Needs the author's decision

- ~~**Finding 1.1** (`state_dependent_approximation=True`): needs a derivation of the correct
  shifted sub-chain kernel and a V4 validation.~~ **Closed 2026-09-18 — feature removed**
  (decision 26, `09` §3 round 4). The derivation was attempted and fails: the shift is also
  arithmetically wrong for `subchain_max_length > 1`, and a state-dependent cheap density breaks
  delayed acceptance at `subchain_max_length = 1` too, so the previous "only K>1 is in question"
  framing (here and in `docs/`) was wrong. Field, warning and shift branch deleted; V4 dropped.
- **Finding 5.8 / decision 2** (adaptive proposal target in DAMH: second-stage rate vs overall
  rate): a real design choice with a re-tuning consequence for `adaptive_target_rate` defaults.
- **`Samples.html_report` / `load_snapshots` / `_load_snapshot_parameters_and_observations`**:
  the remaining pieces of "is the legacy (non-extended) report path still wanted?" —
  `pdf_report` (its sibling) was deleted 2026-09-18 (decision 24, `09` §3), but these three were
  explicitly **not** included in that decision and still need an author call. Newly relevant:
  `html_report` (the method) now also has **zero callers** in the tree — the one script that
  used it, `toy_examples/post_processing_example.py`, was archived as a duplicate in WS10, after
  this note was last written. `load_snapshots`/`_load_snapshot_parameters_and_observations`
  were already caller-less. If the answer for all three is "delete", it's the same shape of
  change as `pdf_report` (`grep` confirmed 0 callers, `./run_tests.sh unit` as the safety check).

### (b) Decision-free, not done

- ~~**Finding 2.7's silent report-content gap**~~ — **fixed 2026-09-18**: `core.py` prints a
  `RuntimeWarning:`-prefixed stdout message when pool mode leaves rank 0 without a live
  `Solver`, and the report itself now says "Not available: pool mode (`use_solvers_pool=True`)
  has no live Solver object on the reporting rank." in place of `par_names`, posterior field
  statistics, and the (newly added) best-fit solver visualisation section. See
  `06_findings_consolidated.md` finding 2.7.
- ~~**Finding 2.10's sampler-side half**~~ **Closed 2026-09-18**: `get_evaluator_and_terminate()`
  now `Wait()`s the pending `TAG_UPDATE` `Isend` before dropping it. Wire protocol unchanged, so no
  matching change anywhere else. Pinned by the first I12 test
  (`tests/mpi/test_mpi_protocol.py`, 2 ranks, asserts the handle is `MPI.REQUEST_NULL` afterwards;
  checked to fail with the fix reverted).
- **Finding 4.7** — ⚠ **confirmed still present, not "unverified"** (2026-09-18): the unguarded
  `print(..., end="\r")` progress line is still on every rank (`algorithms.py:450,646`);
  matplotlib still reaches every sampler rank via a module-level import in `core.py` (not via
  the kept-by-decision `tools.py` helpers, as previously guessed).
- **WS4 seed architecture, the last mile**: `torch.manual_seed`, `initial_training`'s
  `np.random.randn`, and the save/restore-global-state pattern in `tools.py`/`test_data.py` still
  touch global RNG state. Cleanup, not a bug fix — WS4's tested acceptance criterion (two
  identical runs → identical files) is already met via `modules/seeds.py`.
- **WS2's "no `mpi4py` import" half** — ⚠ **newly listed** (2026-09-18): `Configuration` still
  imports and calls `mpi4py.MPI`; harmless in practice (auto-inits a size-1 world under plain
  `python`) but the acceptance criterion as written isn't met.
- ~~**WS6 classical-surrogate hardening**~~ — **done 2026-09-18** (author-approved): RBF
  de-duplication + `max_neighbors=50` cap + shift-fallback removal (3.3); polynomial
  `StandardScaler → PolynomialFeatures → Ridge(alpha=1e-6)` + minimum-snapshot rule (3.10).
  **Changes the exact numbers of every RBF/polynomial run** — see the two CHANGELOG
  behaviour-change entries for the evidence (condition numbers, RMSE, fit times).
- ~~**WS7 efficiency items**: block-group gradients at the full state (not zeros); Hamiltonian
  refresh inside MH stages.~~ **Both done 2026-09-18** (author-approved): `BlockProposal` fills
  the inactive groups from the tracked current sample, and an MH stage with a gradient-based
  proposal may set `surrogate_model_updates=True` to keep re-polling the collector. Sample
  stream changes only for block+Hamiltonian proposals on a non-separable model and for MH
  stages that opt in; see §2.20 and the two CHANGELOG entries.
- **WS7 adaptive-covariance regularisation** for `no_parameters>period` (3.7): the crash cases
  are fixed; singularity in high dimensions needs numerical design (shrinkage target/strength).
- **WS8 performance items**: two-step evaluator transfer instead of the 1 GiB `irecv` buffer
  (4.1), batched snapshot `vstack` (4.3).
  Performance/hardening, out of this refactor's correctness-first scope. ~~busy-wait sleep
  (4.2)~~, ~~`MPI.TAG_UB` check (2.11)~~ and ~~cross-rank `Configuration` consistency
  broadcast+assert (2.9)~~ **done 2026-09-18**, see §2 for the evidence; a
  blocking `Waitany` instead of the spin-then-sleep throttle was **not** done (it would change the
  loop structure, not just its timing).
- **WS8's I12** (2-rank protocol tests): **started 2026-09-18** —
  `tests/mpi/test_mpi_protocol.py` has the first one (the evaluator-communicator shutdown
  handshake, which closes 2.10). The rest of the protocol surface (snapshot tags, the
  `TAG_INITIAL_SURROGATE` handshake) still has no dedicated 2-rank test.
- **WS9 bullet 5 remainder**: ~~`write_report` stage selection still by index, not name~~ —
  **done 2026-09-18** (`Samples._resolve_stages` also accepts stage names). Still open: no
  general warnings-raised-during-the-run report section (only `unverified_options` shown).
- **Finding 3.9 remainder**: no input normalisation, no validation split, early stop still on
  one minibatch's loss.
- **Finding 4.5**: no cached Cholesky for correlated `Normal.logpdf`; torch evaluator still
  casts the whole module to float64 and back on every gradient call.
- **Finding 4.6 remainder**: `pd.concat` in loops, Python-loop `decompress` (the `chains_to_disp`
  half is fixed).
- **`algorithm_interfaces*.py` TODO banners**: not shortened.
- **Write-only flags** (`Updater.supports_*_persistence`, `_request_pending`,
  `snapshot_count_since_last_update`, `communication.PendingRequest.active`/`max_requests`,
  possible unfinished logic — `13_dead_code_report.md` §c/§d): **decided 2026-09-18 (decision
  25) to track as a low-priority backlog item rather than investigate now.** Not a decision to
  revisit unless it becomes a priority.

### (b2) Added 2026-09-18 by the adaptivity study (`15_adaptivity_study_2026-09-18.md` §4.3, `06` Tier 7) — decision-free unless marked

- **Ungate the covariance install** in `GaussRandomWalk_adaptive.adapt()` (7.1), **replace
  the correlation clip by shrinkage + indefinite-covariance warning** (7.2), **make
  `adaptive=True` in a DAMH stage safe or impossible** (7.3), **guard `adaptive_sample_limit`
  / bound the adaptation cost with a running covariance** (7.4), **pre-rejection / no-move
  guard** (7.5), **persist the adapted covariance + per-period `adaptive_stats.csv`, ESS per
  evaluation and per second in `summary.csv`** (7.6), recent-window acceptance for the
  hand-over and pooling ranks instead of averaging covariances.
- **Needs a decision:** carry-over of the adapted covariance into a Hamiltonian mass (7.7 —
  the right transform depends on proposal class and target; toy and GRF disagree for
  `HamiltonianInfinite`), and any step-size adaptation for the Hamiltonian family (the
  acceptance rate is uninformative there; it would have to maximise expected squared jump
  distance per evaluation).
- **Not worth doing** on this evidence: exposing `period`; a windowed acceptance estimate on
  its own; textbook adaptive Metropolis in place of the `coef` loop.
- The 4 GB `toy_examples/out_adaptivity_study_2026-09-18/` is the author's to keep or delete.

### (b3) Added 2026-09-20 by the adaptivity literature research (`16_adaptivity_options_research_2026-09-20.md` §5) — items 1, 2, 4 (step size), 6 implemented the same day (§2.21, §2.22); (b2) above is superseded where §2.22 says so (7.1, 7.2, 7.4 gone; 7.6 partly)

- **Decision-free, with literature + prototype evidence:** ~~(1) in `Algorithm_DAMH` feed `adapt()` every
  outer iteration, scoring a pre-rejected one as α = 0~~ **done 2026-09-20 (author-requested)** — one
  call added in the pre-rejected branch of `Algorithm_DAMH.run`; unit test
  `test_damh_adaptive_proposal_sees_prerejected_iterations` (fails on the old code); unit 388 /
  validation 14 / mpi 38 green; example comparison old vs new in §2.21 (prototype: 17/18 → 0/18
  divergences for every adaptation rule; theory: `α₂|₁` → 1 has no optimum, Sherlock–Thiery–Golightly
  2021); (3) report
  ESS/evaluation, ESS/second, surrogate-call count, `η`, `α₁`, `α₂|₁` per stage + a pre-rejection
  guard; ~~(6) Robbins–Monro on `pcn_beta` to 0.234~~ **done 2026-09-20** (`PCN_adaptive`, §2.22); (7) warn when the
  DAMH exact-stage acceptance leaves [0.2, 0.4] (Lykkegaard et al. 2021).
- **Decided and done 2026-09-20 (author chose after the manager's recommendation):** (2) the combination
  (shrinkage-regularised covariance of post-decision states + Robbins–Monro scale, `2.38²/d` kept,
  relative ridge, target field kept at 0.234, pooled hand-over) — not RAM; RAM stays the documented
  fallback if a real problem misbehaves (§2.22 and the efficiency comparison). (4) the step-size half:
  dual averaging at fixed `num_steps`, δ = 0.8, on the sub-chain acceptance in DAMH.
- **Still needs the author's decision:** (4b) parametrise the Hamiltonian by integration time `T`
  (user-facing: `T` instead of `num_steps`; the prototype's δ = 0.65 vs 0.8 question is open at fixed `T`); (5) mass carry-over
  as `Σ̂⁻¹` (or the surrogate's Gauss–Newton `F̃ + I`) for both Hamiltonian classes, only together
  with (4) — this **closes the 7.7 "evidence is split" question**: the GRF `Σ̂` win was an
  effective-step artefact at ε = 0.05 (16 §3.3(c), §4.3); (8) adaptive Gaussian error model on the
  surrogate likelihood; (9) ChEES trajectory-length adaptation across ranks.
- **Retracted from `15` §4.3 on prototype evidence:** items 1–2 "ungate + vanishing shrinkage" (0.03
  of oracle at d20κ100, worst hand-over; the `coef` feedback is the defect) and the §2.6 suspicion that
  textbook AM collapsed because of proposed-sample estimation (it is the missing ridge: 16–19×).
- ~~Small follow-ups from the §2.21 comparison (degenerate-weights warning; quadratic history cost)~~ —
  both gone with the §2.22 rewrite (no weight history, running statistics).
- **From §2.23 (decision-free):** `torch_perceptron_minibatches.py:168` `overflow encountered in cast`
  (float32 narrowing of the vjp vector) — 5–6 per GRF Hamiltonian run, benign so far, should be looked
  at; `surrogate_quality.csv` records NaN rmse rows for batches whose non-finite observations the
  updater then drops (cosmetic; 332 rows in `new_grf_H`); the leapfrog abort (guard b) has no counter or
  message, so whether it fired cannot be read from a log; astronomically large but FINITE proposals
  still reach the exact solver and the NN (input normalisation / outlier guard, 3.9).
- **New small follow-ups from §2.22 (decision-free):** put the carried values (covariance / β / `ε̄`) into
  the run manifest so a `None`-field run is reproducible from it (7.6's remaining half); the pCN
  `adaptive=True` initial `pcn_beta` must be strictly inside (0, 1) — `Stage.__post_init__` could say so
  before the run starts instead of `PCN_adaptive` raising at stage start.
- Design caveat recorded by the manager (not from a paper): parameters adapted for a K > 1 sub-chain
  must be frozen for the duration of one sub-chain (composition of differently-parametrised reversible
  kernels is not `π_a`-reversible, and the surrogate-transition acceptance needs reversibility).
- Raw material: `toy_examples/out_adaptivity_research_2026-09-20/` (three reviews with per-reference
  verification, prototype scripts, `results_proto.csv` 1605 runs, 13 figures) — an `out_*` directory,
  the author's to keep or delete.

### (c) Future work by design (deliberately deferred, not "not done")

- **Output-format converter** for pre-v2 `out_*` directories — decision 6 says no converter
  unless actually needed.
- **`refresh_within_subchain=True`** as a studied DAMH-SMU variant — deliberately not added (no
  feature flags by default).
- **Thread pinning under other MPI launchers** — superseded: the 2026-09-18 decision made
  `torch_threads` explicit regardless of launcher instead of re-measuring per launcher.
- Adaptive-target-rate semantics and `HamiltonianInfinite` parametrisation also appear in (a) —
  not contradictory: *whether* to resolve them needs a decision; *if* resolved, the work itself is
  future work either way. (`state_dependent_approximation` was in this list too until its removal
  on 2026-09-18.)

### Not code, the author's own call

`library_notes/`, `tests/`, `pytest.ini`, `run_tests.sh`, `CHANGELOG.md` and `CLAUDE.md` were
untracked as of 2026-09-17; per CLAUDE.md this session makes no git commits — committing (or
not) any of this remains the author's call, as it always was.
