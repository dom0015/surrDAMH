# What should be tested or validated

Nothing in this plan was executed. It consolidates the per-note test suggestions into phases,
cheapest and most decisive first. Current coverage: one pytest file
(`tests/test_best_fit_ranking.py`) exercising a function that is not on the production path.

Conventions: "expect" = what a correct implementation should show; "today" = what the reading
predicts the current code does. Run MPI tests under `timeout` so a hang is a failure, not a stall.

---

## Phase 0 — smoke (minutes)

| Check | Command | Expect | Today (predicted) |
|---|---|---|---|
| Import | `python -c "import surrDAMH"` | no output | `NameError: Iterable` (0.1) |
| Sampler import without torch | in an env without torch: `python -c "import surrDAMH.process_SAMPLER"` | ok | `ModuleNotFoundError` (0.2) |
| Existing tests | `python -m pytest tests -q` | 2 passed | passes (but see 5.3) |
| Minimal example | `cd toy_examples && timeout 300 mpiexec -n 2 python3 -m mpi4py minimal_example.py` | clean exit, `out_minimal_example/sampling_output/samples/alg0000_MH/rank0000.csv` | blocked by 0.1 |

Add both import checks as pytest cases so stray imports cannot land again.

## Phase 1 — pure-Python unit tests (no MPI, seconds each)

### Dataclasses / configuration
1. `Stage(adaptive_corr_limit=0.4)` constructs (today: `TypeError`, 1.3).
2. `Stage(algorithm_type="damh")` raises a clear error at construction (today: `NameError` later, 6.6).
3. `Configuration` can be constructed under `mpiexec -n 1`; assertion message matches the condition.

### Proposals (`modules/proposals.py`)
4. **pCN invariance**: from a prior draw, apply `PCN.propose_sample` 10⁵ times with no likelihood;
   empirical mean/cov ≈ `prior_mean`, `C₀`.
5. **Leapfrog energy**: quadratic `U`; `|ΔH| = O(ε²)`; halving `ε` divides `|ΔH|` by ≈4.
6. **Reversibility**: `leapfrog(leapfrog(q,p))` returns `(q,p)` to 1e-10 for both `Hamiltonian`
   and `HamiltonianInfinite` (after the momentum flips).
7. **Rotation invariant**: `HamiltonianInfinite._apply_prior_kinetic_flow` preserves
   `½q·q + ½p·M⁻¹p` exactly when `sd_or_cov == 1`; document the `C = I` assumption.
8. **Adaptive rank**: `GaussRandomWalk_adaptive(no_parameters=20, period=10)`, 10 `adapt()` calls →
   `matrix_rank(sd_or_cov) == 20` (today: ≤ 10, 3.7). Also NaN/zero acceptance weights → finite cov.
9. **Block seeding**: two `BlockProposal`s built the same way on "two ranks" must be reseeded
   differently (today identical, 1.9).
10. `GaussRandomWalk_adaptive` inside a `BlockProposal` constructs (today `AttributeError`, A26).

### Algorithms via local adapters (`algorithm_interfaces_local.py`)
11. **Telescoping invariant**: build `Algorithm_DAMH` with `LocalSolverAdapter` +
    `LocalEvaluatorProvider`, run `_propose_new_sample_using_subchain()` once with a fixed surrogate;
    assert `correction_log_ratio == subchain_current.log_likelihood_approx − initial.log_likelihood_approx`
    to machine precision, for `subchain_max_length ∈ {1,5,20}`.
12. Same with `state_dependent_approximation=True`: the invariant must hold with the *shifted*
    surrogate (today fails for length > 1, 1.1). **Deferred** together with V4 (decision 1); keep
    the test skipped with a reason string pointing to 09 §4.
13. **Solver failure**: initial sample with `tag = −1` → chain must still move (today: stuck, 1.7);
    a failed *proposed* sample must not reach the snapshot sink (today it does, 1.7).
14. **Weights bookkeeping**: sum of `weight` column per stage equals
    `accepted + rejected + prerejected` (+1 for the final row); flag the duplicated boundary
    state (5.7).
15. `_draw_acceptance_decision` with `artificial_acceptance_multiplicator ≠ 1` emits a warning.

### Distributions and transformations
16. `Normal.logpdf` differences between two points vs `scipy.stats.multivariate_normal` (sd and
    cov branches), 1e-10.
17. `grad_logpdf` vs central finite differences for `Normal`, `PriorIndependentComponents`,
    `GaussianMixture` (including a far-away point; today gradient is 0 and logpdf −inf, 1.11).
18. `rvs()` returns shape `(d,)` for every distribution (today `(1,d)` for `GaussianMixture`).
19. Transform round-trips `x → component → inverse ≈ x` on `[-5, 5]` for uniform/lognormal/beta.
20. **Marginal KS test**: 2·10⁵ draws of `transform(rvs())` vs each component's `pdf`/cdf, p > 0.01.
    Most valuable prior test: a wrong transform biases every posterior.
21. Pin the design: `PriorIndependentComponents.logpdf` equals the standard-normal log-density
    without a Jacobian term.

### Surrogates
22. Parametrised over all five updaters: `add_data` 20 snapshots, `train()`, `get_evaluator()`,
    output finite, **shape exactly `(n, no_observations)`** (today flattened for torch, 3.5), and
    `pickle.loads(pickle.dumps(ev))(X) == ev(X)` bit-exact (this is what MPI does).
23. kd-tree `k=5`: query a training point → finite (today NaN, 3.2).
24. RBF with an exactly duplicated snapshot → still reproduces training observations (probes 3.3).
25. Polynomial on an exact quadratic, 200 points → RMSE < 1e-8; degree escalation at documented
    counts.
26. NN: `normalize_outputs(denormalize_outputs(y)) ≈ y`; with non-trivial `output_mean/scale` the
    de-normalised evaluator matches raw targets on `y = x²`.
27. NN gradients: `jacobian(x)[0]` and `vjp(x, v)[0]` vs central differences (rel. err < 1e-4);
    `vjp ≈ Jᵀv`; `jacobian` raises on batched input; shape `(q, p)`.
28. Checkpoint round-trip for both NN updaters: outputs identical after `save_state`/`load_state`;
    `get_initial_snapshots()` returns the saved snapshots (today `None` for Basic, 3.6);
    `SurrogateReused` restores the minibatches updater.
29. `initial_training` must not persist synthetic rows into `get_training_data_arrays()` (today it
    does for minibatches, 3.4).
30. Weighted loss: identical datasets with all weights 1 vs half weights 0 → models differ
    (documents 3.1 so the policy is decided explicitly).

### Helpers
31. `assemble_covariance_matrix(std=[1,2,3])` symmetric and PSD (today asymmetric, 1.6).
32. `lhs_normal`: one point per stratum per dimension; returned design is the maximin-best of 5
    (today: last candidate, 3.8); `import numpy.matlib` still works under numpy 2.2.6.
33. `TestData.generate` leaves the global RNG state unchanged; `save → reuse →
    compute_log_posterior_and_weights → as_surrogate_test_data` works; the call as written in
    `core.py:124` is exercised (today `TypeError`, 0.3).

### Post-processing (synthetic CSV fixtures in a temp dir, real directory layout)
34. `decompress`: shape `(sum(weights), p)`, each row repeated `weights[i]` times.
35. `StageSamples`/`get_mean_and_cov` against hand-computed weighted mean/cov; assert
    `no_unique_samples` equals the compressed row count (today zeros, 5.2).
36. `find_best_fits` vs a naive full sort for all three `ranking_mode`s, with a small `chunk_size`
    to hit the merge path; `prerejected` rows excluded; missing log columns must raise, not
    degrade (P10).
37. `calculate_gelman_rubin`: i.i.d. chains → R̂ ≈ 1; offset chains → R̂ > 1.
38. `autocorr_FM`/`auto_window` on `ρ(k) = φᵏ` → τ ≈ (1+φ)/(1−φ).
39. `Samples(..., load_posterior_surrogate=True)` on a fixture → either works or is removed (5.1).
40. Golden-file test of the column layout of `samples`, `raw_data`, `notes`, `subchain_stats`,
    `summary.csv` (catches 5.4 and the uncommitted `raw_data` change).

## Phase 2 — statistical validation on a Gaussian toy (minutes to an hour, `-n 4`)

Linear solver `G(x) = A x`, Gaussian likelihood, standard-normal prior ⇒ closed-form posterior.
Compare weighted sample mean/cov from `sampling_output/samples/<stage>/rank*.csv` (weights in
column 0) with the closed form, within Monte-Carlo error.

| # | Setup | Expect |
|---|---|---|
| V1 | one `MH` stage, 2·10⁵ samples | matches closed form |
| V2 | `DAMH`, **wrong but fixed** surrogate (`A' = 1.3 A`), `surrogate_model_updates=False`, `subchain_max_length ∈ {1,5,20}` | identical posterior to V1 for every K — the single most decisive test (1.1/1.2, sign of the correction) |
| V3 | as V2 with `surrogate_model_updates=True` and frequent retraining | same posterior; drift ⇒ mid-sub-chain refresh (1.2) |
| V4 | as V2 with `state_dependent_approximation=True`, K=1 and K=5 | **Deferred (decision 1, 2026-09-13)**: the `True` option is marked non-verified and excluded from tests until its theory is checked (09 §4). When run: K=1 matches; K=5 predicted to fail today (1.1) |
| V5 | pCN vs RWMH on the same problem | same posterior; pCN acceptance independent of dimension for fixed β |
| V6 | `Hamiltonian`/`HamiltonianInfinite` with surrogate gradients + exact acceptance | same posterior; check `use_surrogate_gradients=False` fails fast instead of mid-run (2.9) |
| V7 | `initial_sample_type="prior"` run twice | today: different initial samples (1.9) |
| V8 | adaptive stage on 20 parameters | acceptance rate converges to the *requested* target (today always 0.25, 1.3) |

## Phase 3 — MPI integration and deadlock regressions (`timeout 120 mpiexec …`)

| # | Setup | Expect / today |
|---|---|---|
| I1 | `minimal_example.py` with `-n 3 / 4 / 8` | clean exit; `no_samplers = n−1`; one child prints its evaluation count |
| I2 | `one_process_only.py` (`-n 1`, no pool, no collector) | no point-to-point MPI calls touched |
| I3 | `use_collector=True`, one MH stage, `min_snapshots_initial` > producible snapshots | expect a clear error; today hangs (2.1) |
| I4 | DAMH or Hamiltonian as **first** stage, no `initial_snapshots` | expect fail-fast; today hangs (2.2) |
| I5 | `[MH, DAMH, MH(use_only_surrogate=True)]` and `[DAMH, MH(Hamiltonian)]` | surrogate still available in the last stage (2.6) |
| I6 | adaptive stage with `max_evaluations < 10` on ≥ 2 samplers | today: `Allreduce` shape mismatch (2.5) |
| I7 | ~~raw-buffer matrix `pickled_observations × solver_returns_tag`~~ → since 2026-09-17: `solver_returns_tag ∈ {False, True}` plus a solver returning `tag=−1` | all cells exit 0; the negative-tag run completes and its failed proposals are rejected (2.4 fixed by removing the raw path, decision 5) |
| I8 | run A, then B with `initial_sample_type="continued"` in pool mode, solver echoing its input | B's first evaluated parameters equal A's last sample bit-for-bit (today float32 / type mismatch, 2.3) |
| I9 | `initial_snapshots` with N rows | first `snapshots_total` in `surrogate_quality.csv` is N, not 2N (2.8) |
| I10 | `write_report` in pool mode vs local mode | same sections; today field statistics silently absent in pool mode (2.7); a stage with `save_to_file=False` must not shift report indices |
| I11 | `TestData` object passed as `surrogate_test_data` | today `TypeError` + hang (0.3) |
| I12 | 2-rank unit tests of `CommSnapshot_*` (terminate overtaking the last snapshot) and `CommEvaluator_*` shutdown in both `terminate()` branches | no hang, no unmatched messages |
| I13 | print `MPI.COMM_WORLD.Get_attr(MPI.TAG_UB)` at start-up; compare with planned evaluation/snapshot counts (2.11) | documented bound |
| I14 | RSS of a sampler rank right after start-up | predicted large virtual allocation from the 1 GiB `irecv` buffer (4.1) |
| I15 | `top` during a run | two cores at 100 % from busy-wait (4.2) |

## Phase 4 — example smoke matrix

See the table at the end of `05_toy_examples_and_docs.md`. Priority: `minimal_example.py` →
`own_solver.py` (polynomial + DAMH-SMU) → `neural_network_surrogate.py` (torch path) →
FEniCSx-dependent examples only inside the dolfinx devcontainer. Mark
`neural_network_surrogate_copy.py` as not runnable.

## Phase 5 — environment compatibility (this container: numpy 2.2, pandas 3.0, mpi4py 4.1, torch 2.11)

- `import numpy.matlib` (used by `lhs_normal.py`) under numpy 2.2.6.
- pandas 3 copy-on-write and default string dtype: run one full `write_report` and check for
  `SettingWithCopy`/dtype warnings in `post_processing.py`.
- mpi4py 4: `irecv(buf=<int>)` semantics, `Request.waitany`, `Cancel` on a matched request.
- torch 2.11: `torch.load` without `weights_only=True` (S22) — verify it still loads own checkpoints.

## Questions against existing experiment outputs — closed

The TSX and Hamilton experiments and their `out_TSX*` outputs were archived on 2026-09-13
(`TSX_experiments_archived/`, ignored) and declared obsolete, as were the hpcse26 results. The
questions that depended on them (which runs used `subchain_max_length > 1` with state-dependent
approximation or SMU, which believed they used `adaptive_target_rate ≠ 0.25`, zero-weight
fraction in saved training data, non-constant `std` in likelihood covariances) no longer need
answering. The surrogate-weighting policy was decided directly (uniform default, 09 §3).
