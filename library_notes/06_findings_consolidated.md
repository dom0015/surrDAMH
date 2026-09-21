# Findings — status table

Deduplicated findings from the original five-note review (2026-09-12; notes 01–05, since
deleted — see `00_overview.md`). IDs: A = algorithms, M = MPI, S = surrogates/distributions,
P = post-processing, E = examples/docs. Re-verified against the current code on 2026-09-18
(this pass corrects five items where the 2026-09-17 status column was wrong or stale — marked
**⚠ corrected** below; everything else re-confirmed as stated).

Legend: **fixed** (code no longer has the defect, test cited) · **partial** (some sub-items
fixed, rest named) · **open** (untouched; deferred to `09_improvement_plan.md` §4) ·
**decided: keep** (author chose not to change it) · **superseded** (premise no longer
applies).

## Tier 0 — start-up crashes

| # | IDs | Finding | Status | Evidence |
|---|---|---|---|---|
| 0.1 | A1 M1 S0 E1 | Dead method with unimported annotations crashed `import surrDAMH` | **fixed** (WS0) | `grep temptemptemp surrDAMH/` → none |
| 0.2 | A2 M7 P9 | Stray `from torch.mtia import snapshot`, unused | **fixed** (WS0) | `grep "torch.mtia" surrDAMH/` → none |
| 0.3 | E2 M2 S1 | `as_surrogate_test_data()` called with args it doesn't take | **fixed** (WS0) | `tests/mpi/test_mpi_surrogate.py::test_i11_*` |

## Tier 1 — posterior correctness

| # | IDs | Finding | Status | Evidence |
|---|---|---|---|---|
| 1.1 | A5 | `state_dependent_approximation=True`: shift not applied to the sub-chain's current state (arithmetic bug for `subchain_max_length>1`) **and** a state-dependent surrogate breaks the delayed-acceptance argument at *every* `subchain_max_length`, `1` included — the earlier "K=1 is fine" claim was wrong | **fixed: field removed** (2026-09-18, analysis + author decision) | field, `__post_init__` warning and the shift branch of `Algorithm_DAMH._evaluate_surrogate_transition` deleted (`configuration.py`, `modules/algorithms.py`); `CHANGELOG.md` breaking changes; `docs/concepts.md` "Why the surrogate must not depend on the current state" |
| 1.2 | A6 | DAMH-SMU refreshed the evaluator inside the sub-chain loop, breaking telescoping | **fixed** (WS3) | refresh moved before the loop; V2/V3/V3b pass K=1/5/20 |
| 1.3 | A3 A4 | `Stage.adaptive_target_rate/corr_limit/sample_limit` never reached the proposal | **fixed** (G1) | `modules/proposal_builder.py`; `tests/unit/test_proposals.py` |
| 1.4 | M24 | Default `state_dependent_approximation` flipped True→False | **moot: field removed** (2026-09-18, finding 1.1) — the removed behaviour equals the old `False` default | `configuration.py` |
| 1.5 | A7 | `artificial_acceptance_multiplicator` could bias acceptance silently | **fixed: field removed** | `test_algorithms_local.py::test_artificial_acceptance_multiplicator_field_removed` |
| 1.6 | S8 | `Gaussian_process.assemble_covariance_matrix` asymmetric for vector `std` | **fixed** (C5) | `np.outer(std,std)*corr`; `test_helpers.py` |
| 1.7 | A16 A8 | Solver failure → NaN ratios; failed proposals forwarded as training data | **fixed** (B4/C2/C3) | non-finite initial → `RuntimeError`; gate checks `proposed.solver_tag` |
| 1.8 | A11 | Stage after `use_only_surrogate` mixed exact/surrogate log-likelihoods | **fixed** (WS3) | `algorithms.sample_carried_to_next_stage()`; `test_runner_local.py` |
| 1.9 | A13 A14 | Block sub-proposals identical across ranks; `prior.rvs()` unseeded | **fixed** (G4/G5) | `modules/seeds.py`; `Proposal.reseed()`; `test_block_proposal_*` |
| 1.10 | A19 | pCN drops prior ratio unconditionally; `beta` unchecked | **fixed** (WS7/B7) | `build_proposal` rejects non-Gaussian priors; `PCN.__init__` asserts `0<beta<=1` |
| 1.11 | S9 | `GaussianMixture`: no log-sum-exp, `rvs()` shape `(1,d)`, no `mean`/`get_covariance` | **open, pinned not fixed** | `tests/unit/test_distributions.py` pins as known bug |

## Tier 2 — hangs, deadlocks, MPI hazards

| # | IDs | Finding | Status | Evidence |
|---|---|---|---|---|
| 2.1 | M3 | Collector only polled stop signal after first surrogate → hang if none ever trained | **fixed** (B1) | `process_COLLECTOR.py` polls unconditionally; `tests/mpi/test_mpi_hangs.py` I3 |
| 2.2 | M4 | DAMH/Hamiltonian first stage with no surrogate deadlocked | **fixed** (WS8) | `TAG_INITIAL_SURROGATE` handshake → `RuntimeError` on all ranks; I4 |
| 2.3 | M5 A22 | Continuation float32 into a float64 MPI buffer | **fixed** (C9/C10) | float64 end-to-end; MPI I8 bit-for-bit |
| 2.4 | M6 | Raw-mode solver error tag used as an MPI tag → `MPI_ERR_TAG` | **fixed: raw path removed** (decision 5) | `test_mpi_transport.py::test_i7_negative_solver_tag_completes_and_rejects` |
| 2.5 | A9 | Adaptive `Allreduce` shape mismatch (1-D vs 2-D `sd_or_cov`); block+adaptive `AttributeError` | **fixed** (G2) | normalised to 2-D before reduce; block+adaptive → `ValueError` |
| 2.6 | A10 M9 | `following_DAMH or following_onlySurr` (list `or`) closed the collector channel early | **fixed** (C1) | `any(...) or any(...) or any(following_hamiltonian)` |
| 2.7 | M14 | `write_report` hang on rank-0 exception; pool-mode `solver_instance is None` skips report sections | **fixed** (2026-09-18): hang fixed (`_run_role`, earlier); the pool-mode skip is now announced instead of silent | `core.py::_write_report_rank0` prints a `RuntimeWarning:`-prefixed stdout message when `use_solvers_pool=True` leaves rank 0 without a live `Solver`; `html_report_extended`'s new `pool_mode_note` fills the `par_names`/field-statistics gaps in the HTML with "Not available: pool mode (\`use_solvers_pool=True\`) has no live Solver object on the reporting rank.", and `core._insert_best_fit_visualization_note` adds the same note (or the embedded images) to a new "Best-fit Solver Visualization" section that did not exist before. Manual check: `mpiexec -n 4 /dolfinx-env/bin/python3 -m mpi4py template_experiment.py` (pool mode) → note printed on stdout and present in all three sections of `out_template_experiment/post_processing_output/report_extended.html`; a non-pool run with a solver exposing `field_builder`/`coords`/`measurement_points`/`visualize_solution` shows the real content in all three sections instead, with no note. |
| 2.8 | M8 | `initial_snapshots` counted twice (2N) | **fixed** (WS8) | `test_mpi_surrogate.py::test_i9_initial_snapshots_are_counted_once` |
| 2.9 | M12 | `use_surrogate_gradients` mutated per-rank with no cross-rank check | **fixed** (2026-09-18) | `needs_gradients` guard fixed everywhere (earlier); `communication.check_configuration_consistency()`, called from `SamplingFramework.run()` before role dispatch, `bcast`s rank 0's **requested** values of `Configuration.POSTERIOR_AFFECTING_FIELDS` and every rank compares field by field. Requested, not effective: the snapshot is taken in `Configuration.__post_init__` (`_requested_posterior_fields`, generalising the old `core._use_surrogate_gradients_requested`), so the one field that may legitimately differ per rank — `use_surrogate_gradients`, disabled locally by `_configure_surrogate_gradients()` on a rank whose updater/evaluator cannot do gradients — is compared *before* that mutation; no exclusion list needed. MPI-layout fields (`no_samplers`, `rank_collector`, `rank_solvers_pool`, `sampler_ranks`) are deliberately excluded: derived from `COMM_WORLD.Get_size()`, identical by construction. Values are normalised (`configuration.normalize_for_comparison`) so a numpy `lhs_scale` and an object-valued `initial_samples_distribution` compare structurally instead of raising/always differing. Fails on **every** rank (`bcast` + `allreduce`), through `_run_role` → traceback + `MPI.Abort`. Tests: `tests/mpi/test_mpi_config_consistency.py` (2; the mismatch test fails with the call removed — the run then dies later in `numpy` with `shape mismatch: objects cannot be broadcast`), `tests/unit/test_communication.py::TestConfigurationConsistencyCheck` (10) |
| 2.10 | M13 | Sampler's last `TAG_UPDATE` `Isend` abandoned; collector double-cancels the matching `Irecv` | **fixed** (2026-09-18; the collector-side `TAG_EVALUATOR_OBJECT` half was already done): `get_evaluator_and_terminate()` now `Wait()`s the pending `request_Isend` before dropping it, like `request_evaluator()` does. Wire protocol unchanged. | `tests/mpi/test_mpi_protocol.py::test_i12_sampler_side_update_isend_is_completed` (item I12, 2 ranks, asserts the handle is `MPI.REQUEST_NULL` afterwards; verified to fail with the fix reverted) |
| 2.11 | M15 | MPI tags grow unbounded; no `MPI_TAG_UB` check | **fixed: start-up diagnostic added** (2026-09-18). `communication.check_tag_upper_bound()` projects `sum(min(max_evaluations, max_samples))` + `TAG_FIRST_SNAPSHOT` against `MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)` and warns (`RuntimeWarning`) if it does not fit, or if a stage is unbounded (time limit only) and `TAG_UB < 2^20`. Diagnostic only: swallows every exception, never changes control flow. Tags themselves still grow without bound. | `core.py` `run()` (rank 0, before role dispatch); `tests/unit/test_communication.py::TestTagUpperBoundCheck` |
| 2.12 | M18 | `paths_to_append` doesn't reach spawned children | **fixed via replacement** (WS5) | `SolverSpec.resolve_module_path()` gives an absolute path instead |

## Tier 3 — surrogate quality and numerical robustness

| # | IDs | Finding | Status | Evidence |
|---|---|---|---|---|
| 3.1 | S2 | NN loss ignored rejected (weight-0) snapshots entirely | **decided** (WS6, weighting policy) — ⚠ cited test name was wrong, corrected here | `weighting: Literal["uniform","multiplicity"]`, default `"uniform"`; actual tests: `TestWeightingPolicy`, `TestNNWeightingModesDiffer`, `test_classical_surrogates_unchanged_under_default_weighting` (`tests/unit/test_surrogates.py`) |
| 3.2 | S3 | kd-tree `1/distances` → NaN at an exact hit | **fixed** (C4) | zero-distance guard + exact-hit branch; `TestKDTreeExactHit` |
| 3.3 | S7 S15 | RBF: O(N³) refit, no pruning, bad duplicate-handling fallback | **fixed** (2026-09-18) | `deduplicate_snapshots()` collapses repeated centres before the fit (cond. of the thin-plate system on a 19-row set with 4 duplicates: 9.1e18 → 9.1e2); `max_neighbors=50` cap → local fit above 50 snapshots (N=4000 fit 0.44 s → 0.002 s, rmse 8e-5 → 4e-3); shifted-copy fallback (`f(x)=f(x+k·1)`) deleted, replaced by a smoothing ladder that only relaxes exact interpolation; `TestRBFDuplicatedSnapshot`, `TestRBFNeighborCap` |
| 3.4 | S6 | `initial_training` persisted synthetic rows as if they were snapshots | **fixed** (WS6) | `TestInitialTrainingSyntheticRows` |
| 3.5 | S4 S12 | `Evaluator.jacobian`/`__call__` contract vs. actual flattened-shape implementations | **fixed** (WS6) | `__call__` returns `(n, no_observations)` everywhere; `.ravel()`/`.reshape()` compensation removed |
| 3.6 | S5 | `NeuralNetworkUpdaterBasic` half-wired persistence, no registration | **superseded: module deleted** (decision 4) | `ls surrDAMH/surrogates/torch_perceptron*.py` → only `_minibatches.py` |
| 3.7 | A15 | Adaptive covariance singular for `no_parameters>period`; NaN on degenerate weights | **gone 2026-09-20** | the `GaussRandomWalk_adaptive` rewrite (`10` §2.22) has no weight history to degenerate, installs nothing before `warmup=100` states, and shrinks toward `(tr C/d) I` with a relative ridge — PSD by construction in any dimension |
| 3.8 | S10 A21 | `lhs_normal`: `maxmin` never updated, deprecated `numpy.matlib` | **fixed** (G3) | `maxmin=quality` assigned; `np.tile` replaces `numpy.matlib.repmat` |
| 3.9 | S-general | No input normalisation; identity output normalisation by default; no validation split; `torch.set_num_threads` never called | **partial — ⚠ corrected** (previously said "decided not to add" for threads; that was reopened): output normalisation fixed (`"likelihood"` default, WS6); **torch thread pinning now done** (2026-09-18, `Configuration.torch_threads=1` default, `modules/torch_threads.py`); still open: input normalisation, validation split, early-stop-on-one-minibatch | `tests/unit/test_torch_threads.py` |
| 3.10 | S14 | Polynomial surrogate: weights discarded, unscaled/unregularised fit | **fixed** (2026-09-18) | `sample_weight=` honoured (WS6) + pipeline now `StandardScaler→PolynomialFeatures→Ridge(alpha=1e-6)` (design-matrix cond. at degree 5, badly scaled parameters: 9.2e20 → 1.1e2) + minimum-snapshot rule (degree `d` only once `num_terms(d) < num_snapshots`; a single snapshot gives a constant fit, not a hyperplane); `TestPolynomialExactQuadratic` (rmse 1.5e-7 on an exact quadratic), `TestPolynomialMinimumSnapshots` |
| 3.11 | A20 S12 | `SurrogateAsSolver` shape mismatch in `use_only_surrogate` | **fixed** (WS3) | returns `(no_observations,)` unconditionally; 6 parametrised tests + MPI I5 |
| 3.12 | — (found 2026-09-20) | Nothing rejected a non-finite proposal or a non-finite surrogate: a divergent leapfrog proposal was evaluated exactly and forwarded as a training snapshot, the NN's weights went NaN, the collector published the NaN evaluator, the sampler installed it, and every later proposal scored α = 0 (adaptive Hamiltonian: step size → 1e-22, 0 accepted, on 2 of 3 problems) | **fixed 2026-09-20** (`10` §2.23) | non-finite proposals rejected without evaluation/snapshot (`solver_tag = -2`); leapfrog aborts on divergence (20 bit-identity cases); collector never publishes a non-finite evaluator; NN `add_data` drops non-finite rows, `train()` rolls back weights + optimizer; sampler refuses a non-finite evaluator on refresh; +32 unit tests. Left open: astronomically large but finite proposals still reach the NN (no input normalisation, 3.9) |

## Tier 4 — performance and resource use (deferred, not correctness bugs; 4.2 fixed 2026-09-18)

| # | IDs | Finding | Status |
|---|---|---|---|
| 4.1 | M10 | `irecv(buf=1<<30)` — 1 GiB receive buffer per sampler | **partial** — ⚠ the override already existed (`Configuration.max_buffer_size`, documented, wired since WS8) but wasn't being checked against; 2026-09-18 added `process_COLLECTOR._check_evaluator_pickle_size`: raises before sending if a fresh evaluator's actual pickled size exceeds it, warns above half. The two-step size-then-payload transfer (removing the need for any fixed pre-allocation) remains open, deferred as riskier protocol work. |
| 4.2 | M11 M17 | Busy-wait service loops (dead disabled-alternative branch removed) | **fixed** (2026-09-18): `ServiceLoopThrottle` (`communication.py`) in both `process_SOLVER.py` and `process_COLLECTOR.py` — an iteration that did no work counts as idle; after 100 consecutive idle iterations each further one sleeps 1 ms. Timing only, no message change. Measured (4 ranks, MPICH, 32-core container, load ~24 throughout, 3 runs each): `typical_example.py` user CPU 4m17–4m27 → 3m36–3m41 at unchanged wall time (55.9–58.2 s → 55.7–57.9 s); `minimal_example.py` (analytic solver) 9.4–11.6 s → 8.6–9.4 s wall. The spin budget is what keeps the latter neutral: an unconditional 1 ms sleep per idle iteration made it 13.6–14.6 s. |
| 4.3 | M21 M22 | Per-snapshot `vstack`; whole-`Evaluator` pickled per update | **open, deferred** |
| 4.4 | A24 M20 | CSVs now line-buffered (G6); surrogate-quality CSVs still flushed only at shutdown | **partial** |
| 4.5 | S20 S11 | No cached Cholesky for correlated `Normal.logpdf`; torch module double↔float cast per gradient call | **open** |
| 4.6 | P | `pd.concat` in loops, Python-loop `decompress`; `chains_to_disp` filtering | **partial**: `chains_to_disp` fixed (WS9b); the two performance items unchanged |
| 4.7 | A25 S21 | Unguarded `print(..., end="\r")` on every rank; matplotlib reaches the sampler | **open — ⚠ corrected** (previously "not verified"): both confirmed still present. `algorithms.py:450,646` (MH/DAMH progress print); `core.py:12` (`import matplotlib.pyplot`, reaches every rank via `surrDAMH/__init__`) |

## Tier 5 — post-processing, output formats, reporting

| # | IDs | Finding | Status | Evidence |
|---|---|---|---|---|
| 5.1 | P1 | `load_posterior_surrogate=True` read a column the writer never wrote | **fixed: option removed** (WS9b) | `TestLoadPosteriorSurrogateRemoved` |
| 5.2 | P2 | `no_unique_samples` never updated (always 0) | **fixed** (C11, refined by A30) | `np.count_nonzero(self.weights[i])` |
| 5.3 | P3 | `rank_best_fit_candidates` untested-path duplicate | **fixed** (WS9b) | `find_best_fits` now ranks through it |
| 5.4 | P4 | Summary column renames broke an external analysis script | **superseded**: script archived | `TSX_experiments_archived/` |
| 5.5 | P5–P8 | `chains_to_disp` partial; blanket `except BaseException`; wrong `Autocorrelation` chain count; figure leaks | **fixed** (WS9b) | `TestChainsToDisp`, `TestExplicitExceptions`, `TestAutocorrelationChainCount`, `TestFiguresAreNotLeaked` |
| 5.6 | A23 M23 | `raw_data` ragged, no header | **fixed: output format v2** (WS9a) | `docs/outputs.md`; `test_read_run.py` |
| 5.7 | A30 | Stage-boundary state double-written across concatenated stages | **fixed, decided 2026-09-17**: carried-over stage's first row drops the leading `+1` | 4 named tests in `tests/unit/test_algorithms_local.py` |
| 5.8 | A17 | DAMH's `adapt()` sees only sub-chain endpoints | **decided: keep the mechanism, documentation corrected** (2026-09-18) — the "second-stage/outer rate" description was itself wrong (the outer correction term is never seen either); `adapt()` actually targets the exact-posterior ratio between the outer state and the endpoint, which under-scales the proposal for `subchain_max_length>1`. The design question (decision 2) of whether to change the mechanism remains deferred. | `modules/proposals.py` (`GaussRandomWalk_adaptive` docstring), `stages.py`, `docs/stages.md` |

## Tier 6 — packaging, docs, hygiene, dead code

| # | IDs | Finding | Status |
|---|---|---|---|
| 6.1 | E6 | `setup.py` missing torch/scikit-learn/emcee; `python_requires>=3'` too permissive | **fixed** (E1/E2) |
| 6.2 | E | `.gitignore` 10k-line `.venv` dump, no `out_*`/`.npz`/`.pt` patterns | **fixed** (E3) |
| 6.3 | E9 E8 | Stray root files; tracked file inside an `out_*`-named dir | **partial/decided**: stray files archived; `toy_examples/out_tsx/sampling_TSX.py` kept by decision (FEniCSx example) |
| 6.4 | E4 E5 E7 | Copy-pasted run commands; dead class reference; duplicate examples | **fixed** (D12/D13, WS5, WS10) |
| 6.5 | E | No `Configuration`/`Stage`/solver/surrogate/output docs | **fixed** (WS11) — `docs/` |
| 6.6 | various | Dead/fragile code inventory | **mostly fixed; remainder kept by explicit no-delete decision** — see `13_dead_code_report.md` |

## Tier 7 — adaptive-proposal findings from the 2026-09-18 adaptivity study (`15_adaptivity_study_2026-09-18.md`)

All **open**, decision-free unless marked; none biases the posterior (verified at adequate ESS), all cost efficiency or fail silently.

| # | Finding | Evidence | Status |
|---|---|---|---|
| 7.1 | `GaussRandomWalk_adaptive.adapt()` installs the new covariance only when the acceptance rate leaves the ±20 % band — the *shape* estimate is discarded whenever the *scale* is fine (no `else` branch) | toy: 8 of 500 periods installed anything on d2 κ10; ungating alone 0.18 → 0.75 of oracle on d20 κ1 (`15` §2.2, §2.6) | **gone 2026-09-20** with the rewrite of `GaussRandomWalk_adaptive` (`16` §5 item 2, author-requested): the shape is re-estimated every period unconditionally, the scale is a separate Robbins–Monro recursion; evidence `10` §2.22 |
| 7.2 | Entrywise clipping of correlations to `adaptive_corr_limit=0.3` breaks positive semi-definiteness in high d; numpy warns and samples via SVD | toy: 491–495/500 periods indefinite at d 20; manager: 200/200 rank-2 20-D matrices; GRF: 3 warnings per run, a −0.325 eigenvalue; the clip binds everywhere on GRF although true max \|ρ\| = 0.17 (`15` §2.2, §3.1) | **gone 2026-09-20**: the clip and `adaptive_corr_limit` were removed; shrinkage `δ = min(1, 2d/n)` toward `(tr C/d) I` plus a relative ridge keeps the matrix PSD (unit test on d10 κ10: shape error 0.078, no warnings); `10` §2.22 |
| 7.3 | `adaptive=True` inside a DAMH stage diverges: `adapt()` only sees iterations whose sub-chain moved, a selection-biased sample it over-scales to; positive feedback with a good surrogate | GRF: 2 775 431 iterations → 229 exact evaluations, sd 0.3 → 3.9, R-hat 1.72, reproduced K 1/5 and `corr_limit` 1.0; toy: 2.4–9.3× over-scaling at K 1/5 (`15` §3.3, §2.5) | **fixed 2026-09-20** (author-approved, `16` §5 item 1): `Algorithm_DAMH.run` now calls `adapt()` on every outer iteration, pre-rejected ones with acceptance probability 0; pinned by `tests/unit/test_algorithms_local.py::test_damh_adaptive_proposal_sees_prerejected_iterations` (fails on the old code: 9 calls for 400 iterations); example comparison in `toy_examples/out_damh_adapt_fix_2026-09-20/`, evidence `10` §2.21 |
| 7.4 | `adaptive_sample_limit` is unstable: bounded window → `P ← coef·P` unanchored recursion | toy: scale 27 500× oracle, all 3 seeds; GRF: 22× ESS/eval loss, sds differing 17× across ranks (`15` §2.3, §3.1) | **gone 2026-09-20**: `adaptive_sample_limit` and the `coef` recursion were removed with the rewrite (running Welford statistics over the whole stage, constant memory); `10` §2.22 |
| 7.5 | A Hamiltonian stage with a too-large step (`0.5`, 10× the shipped GRF default) pre-rejects 100 %: zero exact evaluations, constant chain, **no warning** | GRF `b2_step0.5_ns100` (`15` §3.2) | **open**; pre-rejection / no-move guard |
| 7.6 | The adapted covariance is persisted nowhere; a `proposal_sd_or_cov=None` run is not reproducible from its manifest; no ESS in `summary.csv` | toy §2.4, GRF §3.1 | **partly done 2026-09-20**: `adaptive_stats/<stage>/rank%04d.csv` (per-period trace of the adaptation) and the per-rank `carry-over` print exist; the carried value is still not in the manifest and `summary.csv` still has no ESS (`16` §5 item 3 remains open) |
| 7.7 | Carry-over into a Hamiltonian mass: 3.3–3.6× available on GRF (`mass = Σ̂`, `HamiltonianInfinite`), but the same transform froze the fully-informed toy, and `Σ̂⁻¹` is right for `Hamiltonian` — no blanket rule | `15` §3.2 reconciliation | **needs a decision + its own study** |
| 7.8 | On the cheap GRF solver the shipped DAMH-Hamiltonian stage is ~14× slower per second than DAMH-pCN K 5 and half the speed of plain pCN (per evaluation it is the best) | GRF §3.3 | informational — regime of the example, not a bug |

**Update 2026-09-20** (`16_adaptivity_options_research_2026-09-20.md`, literature + numpy prototypes,
nothing implemented): 7.1/7.2's suggested fix "ungate + shrinkage on top of the `coef` loop" is
**retracted** (0.03 of oracle at d20 κ100, worst hand-over) — the `coef` feedback itself is the defect;
the replacement is a shrinkage-regularised covariance of accepted states + Robbins–Monro scale (best
prototype) or RAM (Vihola 2012). 7.3's fix is **confirmed** as "score pre-rejected iterations as
α = 0" (17/18 → 0/18 divergences for every adaptation rule); theory: `α₂|₁` → 1 has no optimum
(Sherlock–Thiery–Golightly 2021). 7.7 is **resolved in principle**: mass = `Σ̂⁻¹` for both classes,
but only with step-size (and trajectory-length) re-adaptation — the GRF `Σ̂` win was an effective-step
artefact at ε = 0.05 (16 §3.3(c), §4.3). 7.5/7.6 confirmed. Decisions still needed: `10` §8(b3).

## Confirmed correct (protected by tests, not bugs)

DAMH second-stage acceptance formula; the DAMH-SMU stale-surrogate-at-current-state handling;
pCN prior cancellation for the internal N(0,I) prior; HMC momentum/kinetic-term/leapfrog
correctness; `HamiltonianInfinite`'s rotation (volume-preserving/reversible for any prior,
prior-preserving only for `C=I`, documented); the collector's TAG_UPDATE/TAG_EVALUATOR_OBJECT
handshake invariant (both halves of 2.10 now closed, pinned by I12); barrier accounting; writer↔reader
format agreement for every CSV.

See `09_improvement_plan.md` §3/§4 for the decision log and `10_manual_review_notes.md` for
behaviour-change evidence and the authoritative still-open list.
