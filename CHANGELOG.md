# Changelog

Posterior-, sample-stream-, or otherwise user-visible changes made during the current refactor
(`library_notes/09_improvement_plan.md`). Each entry names its workstream, the finding/decision
it implements, and the evidence in `library_notes/10_manual_review_notes.md` §2 (behaviour
changes, cited as "§2.N") / §3 (pre-existing bugs pinned but not yet fixed). No release has been
cut yet — everything below is "Unreleased". Entries are grouped by kind; within a group, roughly
chronological (earliest change first). This file was reorganized on 2026-09-17 from an
earlier per-workstream layout; no entry's substance changed, only its grouping.

## Unreleased

### Breaking changes

- **`SamplingFramework` -> `Problem` + `run_sampling`/`run_sampling_local` + `SamplingRun`**
  (2026-10-08, author decision, API spec "problem_split"). Hard rename, no compatibility
  alias: `surrDAMH.SamplingFramework` is gone, and `surrDAMH.run_local` is no longer exported
  from the package (`surrDAMH.runner_local.run_local` stays as the internal engine behind
  `run_sampling_local`, not part of the public surface). The keyword arguments
  `solver_instance=`/`solver_spec=` are replaced by a single, type-dispatched `solver=` (a
  `Solver` instance or a `SolverSpec`). New usage:
  ```python
  problem = surrDAMH.Problem(prior, likelihood, solver, no_parameters=None, no_observations=None)
  run = problem.run_sampling(conf, stages, surrogate_updater=None, surrogate_evaluator=None,
                             surrogate_initial_training_data=None, surrogate_test_data=None,
                             surrogate_restart=None)                       # MPI, collective call
  run = problem.run_sampling_local(conf, stages, surrogate_updater=None, surrogate_evaluator=None)
  run.write_report(...)        # same body/arguments as the old SamplingFramework.write_report
  ```
  `SamplingRun.load(output_dir, problem)` rebuilds a report-only run object from a finished
  run's `run_manifest.json`.

  **Size resolution** (`no_parameters`/`no_observations`, same rule for both): the first
  source below that supplies a value wins, and every source that is present must agree
  (`ValueError` naming each source and its value otherwise): (1) explicit
  `Problem(no_parameters=..., no_observations=...)`; (2) the `Solver` instance's declared
  attribute (not available for a `SolverSpec` — it is not imported until the run starts);
  (3) the prior's (resp. likelihood's) dimension (`PriorIndependentComponents.no_parameters`,
  `Normal`/`StandardizedNormal.n`, `GaussianMixture.means.shape[1]`, `FromScipy`'s wrapped
  `scipy_rv.dim`, else `dist.no_parameters`/`dist.dim`); (4) otherwise `ValueError` naming the
  fix. `Configuration.no_parameters`/`no_observations` are now optional (`None` by default,
  and `output_dir` is the first field); when given they are checked, not just stored, by the
  new `Configuration.resolve_problem_sizes`, called by `run_sampling`/`run_sampling_local`
  before `check_configuration_consistency`.

  **Dimension-free `Normal`**: `Normal(mean=<scalar>, sd=...)` with no `dim=` is 1-D when
  used alone, but is now broadcast to a larger dimension resolved from another source
  (typically a `Solver` instance) instead of silently staying 1-D. This only works when that
  other source is available: a script passing a `SolverSpec` (which cannot supply it) needs
  an explicit size instead — `dim=`, an array-valued `mean`, or `Problem(no_observations=...)`.

  **Two new checks**: C2 (solver-pool start-up) — every spawned child's declared
  `(no_parameters, no_observations)` is gathered to the pool rank and compared against
  `Configuration`; a mismatch aborts the job naming the `solver_id` and both sets of values,
  a solver declaring neither size only warns. C3 (first evaluation) — the first observations
  returned each stage (in-process solver, spawned child, or `Stage.use_only_surrogate=True`)
  are checked against shape `(no_observations,)`, raising `ValueError` naming the solver class
  and the two shapes on a mismatch.

  **`TestData.generate` signature change**: old `(prior, likelihood, solver, conf, size, ...)`
  read `conf.no_observations`/`conf.transform_before_surrogate`; new
  `TestData.generate(problem, size, transform_before_surrogate=False, ...)` takes a `Problem`
  and the option explicitly instead (`TestData.reuse` unchanged). Updated in
  `toy_examples/toy_example_hamilton.py` and `toy_examples/sampling_diffusion_grf.py`.

  Behaviour is unchanged for every maintained example (same seeds, stages, `output_dir`,
  process counts) — only the API spelling changes. `toy_examples/minimal_example.py`,
  `neural_network_surrogate.py`, `typical_example.py`, `with_simple_postprocessing.py` and
  `post_processing_with_html_report.py` pass a `SolverSpec` (not a `Solver` instance) and
  dropped the old explicit `Configuration(no_parameters=..., no_observations=...)`; each now
  carries the same numbers as a plain local variable (fed to the solver spec/surrogate
  updater) instead, resolved for `Problem` from the prior/likelihood as before.
  `minimal_example.py`'s likelihood additionally changed from the dimension-free
  `Normal(mean=5.0, sd=1.0)` to `Normal(mean=[5.0], sd=1.0)`, since a `SolverSpec` cannot
  supply the broadcast target; `one_process_only.py` (a `Solver` instance) keeps the
  dimension-free form.
- **`Configuration.state_dependent_approximation` removed** (finding 1.1, author decision
  2026-09-18, supersedes the earlier "keep `False`, warn on `True`" decision 1). The field and the
  state-dependent branch of `Algorithm_DAMH._evaluate_surrogate_transition` (which shifted the
  surrogate by `G(x₀) − G~(x₀)`, the exact-vs-surrogate gap at the sub-chain start) are gone;
  passing the argument now raises `TypeError`, with no replacement and no compatibility shim.
  Reason: the option was not merely unverified for `subchain_max_length > 1` (where it also had a
  plain arithmetic bug — the sub-chain's "current" likelihood kept using `G(x₀)` at every step, so
  the correction term did not telescope), it is **unsound at every `subchain_max_length`,
  `1` included**: the delayed-acceptance correction requires the cheap density to be fixed and
  state-independent, and re-centring it on the current chain state breaks the
  `Q(y→x)/Q(x→y) = pi~(y)/pi~(x)` identity the derivation rests on. No sound repair exists within
  this design (re-shifting per step, or shifting only the first/last step, either reproduce the
  same bug or need quantities the algorithm never computes). **No run is affected numerically**:
  the default was `False`, `True` emitted a `RuntimeWarning`, and no maintained example, test or
  validation run used `True`. `toy_examples/template_experiment.py` and
  `toy_examples/sampling_TSX.py` passed an explicit `state_dependent_approximation=False`; that
  now-invalid argument was deleted from both (behaviour unchanged).
- **`Configuration.pickled_observations` removed** (WS8, author decision 5, findings 2.3/2.4/M6,
  §2.17). It selected between two solver-pool transports; only the pickled one survives. Passing
  the argument now raises `TypeError`; no replacement, no compatibility shim. No maintained
  example set it, so nothing in this repository needed updating.
- **`artificial_acceptance_multiplicator` removed** from `Stage`/`AlgorithmBase` (WS0 §B5, author
  decision 2026-09-13, finding 1.5, commit `175059e`, §2.4). Any script that set it to a value
  other than `1.0` now gets a `TypeError` instead of a silent acceptance-rate distortion; scripts
  that never set it are unaffected.
- **`NeuralNetworkUpdaterBasic` and `surrDAMH/surrogates/torch_perceptron.py` removed** (WS6,
  author decision 4, §2.16d). Replacement preset:
  `NeuralNetworkUpdaterMinibatches(..., solver="lbfgs", batch_size=None, replay_ratio=0.0,
  train_on_added_data=False)`, now used by `toy_examples/neural_network_surrogate.py` and
  `toy_examples/sampling_TSX.py`.
- **Snapshot "weight" renamed to "multiplicity"** throughout the sampler→collector payload and
  everywhere downstream (WS6, §2.16e): `Updater.add_data(parameters, observations, multiplicity)`,
  `get_training_data_arrays()`, `LocalSnapshot.multiplicity`, `InMemorySnapshotSink`, the
  `surrogate_training_data.npz` key (`weights` → `multiplicity`), the third array of
  `initial_snapshots`, and the `multiplicity` column header in `samples/*.csv` (see the output
  format v2 entry below). No compatibility shim: old `.npz` training-data files and custom
  updaters that used `weights=` as a keyword must be updated. (`TestData`'s posterior weights are
  a different concept and keep the name "weights".)
- **`Evaluator.__call__` shape contract standardized to `(n, no_observations)` for every `n`,
  including `n=1`** (WS6, findings S4/S12, §2.16c). The two torch evaluators no longer flatten
  their output; the five call sites that compensated for the old flattened shape are gone.
  Breaking for any custom `Evaluator` that relied on the old per-evaluator inconsistency; the four
  built-in evaluators are value-neutral (a DAMH-SMU chain with a polynomial surrogate produced a
  byte-identical `samples` CSV before and after).
- **Output format v2** (WS9a, author decision 6 "no converter", §2.14). Every sampling CSV now
  has a header row (`samples/<stage>/rank%04d.csv`: `multiplicity,par_0..par_{p-1},log_posterior`;
  `raw_data/<stage>/rank%04d.csv`: `state_type,par_0..par_{p-1},solver_tag,obs_0..obs_{m-1},
  obs_approx_0..obs_approx_{m-1},log_likelihood,log_prior`); `raw_data` is now rectangular with
  the exact (`obs_*`, NaN for `prerejected` rows) and surrogate (`obs_approx_*`, NaN where none
  exists) observations split into separate blocks, instead of v1 silently writing surrogate
  values into the exact block for `prerejected` rows. `run_manifest.json`'s `format_version` is
  now 2 and enforced: `surrDAMH.read_run`/`post_processing.Samples` raise `RunFormatError` on a
  missing manifest or any other version. **Output directories written before this change cannot
  be read any more** — any external script doing `np.loadtxt(...)`/`pd.read_csv(...,
  header=None)` on `samples`/`raw_data` needs `skiprows=1`/`header=0`, and code that indexed the
  observation block of a `prerejected` row must switch to `obs_approx_*` or filter it out.
- **`Samples(..., load_posterior_surrogate=True)` removed** (WS9b, finding P1, §2.15). The
  `samples` CSV never had a surrogate-posterior column, so the option could only raise
  `IndexError`; passing it now raises `TypeError`, and `StageSamples.posterior_surrogate` is gone.
- **`Stage.adaptive_corr_limit` and `Stage.adaptive_sample_limit` removed** (2026-09-20,
  `library_notes/16_adaptivity_options_research_2026-09-20.md` §5 item 2). Both belonged to
  `GaussRandomWalk_adaptive`'s old internals, which were replaced (see "Behaviour changes"):
  the entrywise correlation clip made the covariance indefinite so numpy drew ~every proposal
  through an SVD fallback (19 750 warnings in one 20 000-evaluation prototype run, `16` §4.1),
  and the bounded sample window blew the scale up by four orders of magnitude (`15` §2.3).
  There is no replacement and no shim: a script that still passes either field gets the
  dataclass `TypeError` (author's no-compat rule). `adaptive_target_rate` is unaffected.
- **`build_proposal(..., prev_cov=...)` replaced by `carried=...`** (2026-09-20). The
  hand-over between stages is now a `dict` keyed by `Stage` field name
  (`{"proposal_sd_or_cov": ..., "pcn_beta": ..., "hamiltonian_step_size": ...}`), merged over
  every adaptive stage of the run, because an adaptive pCN / Hamiltonian stage hands over
  something that is not a covariance. Only callers of `build_proposal` are affected (both
  runners and the tests in this repository); `Stage` itself is unchanged by this item.
- **`Proposal.adapt()` signature extended** (2026-09-20) to
  `adapt(proposed_sample, log_acceptance_probability, current_sample,
  subchain_log_acceptance_probabilities=None)`. `current_sample` is the chain state AFTER the
  accept/reject decision (what a Haario-style covariance estimator needs) and the last argument
  carries the DAMH sub-chain's per-step log acceptance probabilities (`None` in MH). A custom
  `Proposal` that overrides `adapt` must take the two new parameters; one that does not
  override it is unaffected. `Algorithm_DAMH._propose_new_sample_using_subchain` now returns a
  4-tuple (the sub-chain probabilities are the new element).

- **`Stage` redesigned around proposal objects; `Configuration.solver_returns_tag` removed**
  (2026-09-21, author decisions; design in `library_notes/17_stage_proposal_objects_design_2026-09-21.md`,
  evidence in `10_manual_review_notes.md` §2.28). No compatibility layer (decision "no compat").
  - Proposal settings moved out of `Stage` into spec classes in the new module `surrDAMH.proposals`:
    `RandomWalk(scale, adaptive, target_rate)`, `PCN(beta, adaptive, target_rate)`,
    `Hamiltonian(step_size, num_steps, mass, integrator="leapfrog"|"split", adaptive, target_rate)`,
    `Block(groups, proposals)`; `Stage(proposal=RandomWalk())` is the default. Renamed fields:
    `algorithm_type` -> `algorithm`, `subchain_max_length` -> `subchain_length`. Removed fields (now
    `TypeError`): `proposal_type`, `proposal`(old dead field, now the spec), `proposal_sd_or_cov`, `pcn_beta`,
    `hamiltonian_step_size`, `hamiltonian_num_steps`, `block_proposal_groups`, `block_proposal_list`,
    `adaptive` (now a read-only property = `proposal.adaptive`), `adaptive_target_rate`. Translation:
    `Stage(algorithm_type="MH", proposal_sd_or_cov=0.5, ...)` -> `Stage(proposal=RandomWalk(scale=0.5), ...)`;
    `proposal_type="HamiltonianInfinite"` -> `Hamiltonian(integrator="split")`; a `Block` takes specs, not
    runtime `Proposal` objects, and an adaptive sub-proposal inside a `Block` raises in `build_proposal`
    (a block never adapts its members; before, only `adaptive=True` on the block stage raised and an
    adaptive runtime member silently stayed at its start value).
  - Carry-over keys (`Proposal.carry_over()`, `sampling_output/carry_over/*.npz`, `RunData.carry_over`):
    `proposal_sd_or_cov`/`pcn_beta`/`hamiltonian_step_size` -> `scale`/`beta`/`step_size`;
    `load_carry_over` maps the old keys of existing files on read. `run_manifest.json` stage entries carry
    `"algorithm"`, `"subchain_length"` and a nested `"proposal": {"type": ..., <fields>}`; old manifests
    still load (`read_run` uses only the stage names), and the HTML report reads both layouts.
  - Not posterior-affecting for a one-to-one translation: the runtime proposal classes, their seeds and
    defaults are unchanged, and every maintained example was translated one-to-one. Stage directory
    names are unchanged.
  - `Configuration.solver_returns_tag` is gone: a `Solver.get_observations()` may return the observations
    or an `(observations, tag)` tuple, and the spawned child now detects the tuple with `isinstance`
    exactly as the sampler's algorithm always did for a local solver. Passing the argument raises
    `TypeError`; delete it.

- **Short component aliases removed from `distributions/independent_components.py`**
  (2026-09-21, author decision after the `__init__` audit §2.29). `Uniform`, `Lognormal`, `Beta`
  and `Normal` were aliases of `UniformComponent`, `LognormalComponent`, `BetaComponent` and
  `NormalComponent`; only the `*Component` names remain. Reason: `independent_components.Normal`
  was `NormalComponent` (one parameter's component) while `surrDAMH.distributions.Normal` is the
  full Gaussian distribution -- same name, different class, different constructor. Replace
  `surrDAMH.distributions.independent_components.Lognormal(...)` with the exported
  `surrDAMH.distributions.LognormalComponent(...)`. The only in-repo caller,
  `toy_examples/sampling_TSX.py`, was switched to the exported names beforehand; no test, example
  or library module referenced the aliases. Not posterior-affecting (pure renaming of an import
  path; the classes themselves are untouched).

- **`NeuralNetworkUpdaterMinibatches` renamed to `NeuralNetworkUpdater`** (2026-09-22, author
  decision: it is the only network updater shipped, next to `PolynomialSklearnUpdater`,
  `RBFInterpolationUpdater`, `KDTreeUpdater`). No alias in code (no-compat rule). Checkpoints are
  DATA, so they keep loading: `surrogate_checkpoint.pt` files that recorded
  `surrogate_type = "NeuralNetworkUpdaterMinibatches"` are accepted by `load_state` and by
  `SurrogateReused` through `surrogates.reuse.LEGACY_SURROGATE_TYPE_NAMES` (verified with a forged
  old-name checkpoint: loads, evaluator output identical). Module file
  `surrogates/torch_perceptron_minibatches.py` keeps its name (internal).

- **Every prior is sampled in the standard-normal internal space; a `Normal` prior is standardized
  automatically** (2026-09-22, author decision 5 of the template review; `10_manual_review_notes.md`
  §2.36). `SamplingFramework.__init__`, `run_local` and `TestData.generate`/
  `compute_log_posterior_and_weights` wrap a `Normal` prior in the new
  `distributions.StandardizedNormal` (`distributions.standardize_prior`): the chain samples
  `z ~ N(0, I)` and `transform(z) = mean + L z` (`L L^T = cov`, or `L = diag(sd)`) gives the physical
  parameters. `Normal` used as a *likelihood* is untouched. Posterior unchanged (verified against the
  closed-form Gaussian posterior for a N(8, 5) x N(-1, 0.5) prior: chain mean/sd equal the closed form
  to two decimals; `tests/validation` passes), but for every run with a `Normal` prior: (1) the sample
  STREAM changes (proposals now act on standardized coordinates); (2) `Configuration.lhs_scale` and
  `initial_samples_distribution` are now in standardized units; (3) with
  `transform_before_surrogate=False` (default) the surrogate is trained on standardized inputs; (4)
  `initial_sample_type="continued"` from a run made BEFORE this change is invalid for a `Normal` prior
  (its saved internal samples were physical) -- re-run instead; (5) `run_manifest.json` records
  `prior: {class: "Normal", standardized: true, ...}`. pCN is now exact for any `Normal` prior, and
  the adaptive random walk's default start is `2.38^2/d * I` for every prior.

- **Renamed: `Normal(d=...)` -> `Normal(dim=...)`; `SamplingFramework(initial_snapshots=...)` ->
  `SamplingFramework(surrogate_initial_training_data=...)`** (2026-09-22, author decisions 2 and 3 of
  the template review; no compatibility aliases). `dim` was chosen over `no_parameters`, which the
  author reserves for block distributions. All examples, tests and docs updated; the internal
  collector argument `initial_snapshots` and `Updater.get_initial_snapshots()` keep their names.

### Behaviour changes (sample stream may differ)

- **An adaptive sub-proposal inside a `Block` no longer raises (C2, 2026-10-12; the G2 guard in
  `build_proposal` is removed).** A configuration such as `Block([[0],[1]], [RandomWalk(), RandomWalk()])`
  (or any sub-proposal with a step left `None`/`adaptive=True`) raised `ValueError` before and now runs:
  the sub-proposals adapt in group dimension, start from their defaults (never from carry-over), the
  stage is `MH-adaptive`/`DAMH` with `adaptive=True`, and `adaptive_stats` is written (first adaptive
  sub-proposal). **Affected: only configurations that raised before** (no earlier configuration changes;
  fixed Blocks byte-identical, see the C2 entry under "Additions"). `tests/mpi/test_mpi_hangs.py::test_i6_adaptive_block_proposal_allreduce`, which asserted the
  G2 refusal, is inverted into `test_i6_adaptive_block_proposal_pools_and_finishes` (the adaptive sub-proposal is
  pooled across two sampler ranks, exit code 0, no carry-over, `adaptive_stats_component == 0`).

- **A non-updating stage that fetches the chain's first evaluator now starts with the collector's
  newest one** (2026-09-30, bug in §3 "Frozen DAMH stage directly after an MH stage uses the stale
  first evaluator", `19_efficiency_study_2026-09-28.md` §4.4). `DAMH` with
  `surrogate_model_updates=False`, or an MH stage with a Hamiltonian proposal and no updates, placed
  after an MH warm-up used to keep the collector's *first* surrogate (trained on
  `min_snapshots_initial` snapshots) for the whole stage. `AlgorithmBase._install_newest_evaluator`
  now asks for the current one (`CommEvaluator_sampler.get_newest_evaluator`: re-sends the existing
  `TAG_EVALUATOR_NEEDED` int with the outstanding request's idx and blocks for the answer; the
  collector answers it at once, with the evaluator the sampler already has if nothing newer exists).
  No new tag or message format; sampler (`communication.py`, `algorithms.py`,
  `algorithm_interfaces_mpi.py`) and collector (`process_COLLECTOR.py`) changed together.
  **Affected: only those layouts** (sample stream changes; toy check: frozen DAMH 0 → ~290/300
  accepted per chain, surrogate RMSE 7.55 → 0.02). DAMH-SMU stages, stages after an SMU stage,
  MH-only runs and `run_local` (whose provider already returns the newest evaluator) are
  unchanged — bit-identical in deterministic before/after runs. Pinned by
  `tests/mpi/test_mpi_surrogate.py::test_frozen_damh_after_mh_uses_the_newest_evaluator`.

- **Adaptive proposals in DAMH stages now see pre-rejected iterations** (2026-09-20,
  `Algorithm_DAMH.run` in `modules/algorithms.py`, recommendation 1 of
  `library_notes/16_adaptivity_options_research_2026-09-20.md`, §2.21). On an outer iteration
  whose surrogate sub-chain never moved (pre-rejected, no exact evaluation) `Proposal.adapt` is
  now called with `log_acceptance_probability=-inf` (acceptance probability 0) and the current
  state as `proposed_sample`; before, such iterations were skipped. `GaussRandomWalk_adaptive`'s
  acceptance-rate estimate therefore measures the overall acceptance of an outer iteration instead
  of the conditional second-stage rate, which tends to 1 as the surrogate improves and drove the
  proposal scale up without bound (GRF: 2.8 M iterations → 229 exact evaluations, sd 0.3 → 3.9,
  `15_adaptivity_study_2026-09-18.md` §3.3; prototype: 17/18 → 0/18 divergences for every
  adaptation rule, `16` §4.2). **Affected configurations: stages with `algorithm_type="DAMH"` and
  `adaptive=True` only** — their sample stream changes. MH stages, non-adaptive DAMH stages and the
  covariance carried into later stages from an *MH* adaptive stage are bit-identical (no code path
  touched). Cost: one `adapt()` call per outer iteration instead of per moved iteration (each appends
  a zero-weight row to the proposal's history). Pinned by
  `tests/unit/test_algorithms_local.py::test_damh_adaptive_proposal_sees_prerejected_iterations`;
  example comparison in `toy_examples/out_damh_adapt_fix_2026-09-20/RESULTS_COMPARISON.md`.
- **Block-proposal gradients are evaluated at the full current sample** (WS7, `modules/proposals.py`,
  §2.20a). When a `BlockProposal` updates one parameter group, the gradient functions it hands to
  that group's Hamiltonian-family sub-proposal used to build the full argument vector by
  **zero-filling** the other groups' coordinates; they now fill them from the chain's current
  sample, which `BlockProposal.propose_sample` tracks in `current_sample`. No acceptance
  probability changes — any gradient field gives a valid reversible HMC proposal, and the
  accept/reject test still uses the exact model — only the leapfrog trajectory does.
  **Affected configurations: stages whose proposal is a `block` proposal containing a
  `Hamiltonian`/`HamiltonianInfinite` sub-proposal, and among those only the ones whose model is
  not separable across the groups.** For a separable (block-diagonal) model old and new behaviour
  are bit-identical, pinned by
  `tests/unit/test_proposals.py::test_block_proposal_gradient_change_is_a_no_op_for_a_separable_model`;
  the coupled case is pinned as a deliberate difference by `…_moves_the_trajectory_for_a_coupled_model`.
  Everything else — every non-block proposal, every block proposal without a gradient-based
  sub-proposal, which is every configuration in this repository's examples and tests — is untouched.
- **MH stages with a gradient-based proposal may refresh their surrogate** (WS7, `stages.py`,
  `modules/algorithms.py`, §2.20b). `Stage.surrogate_model_updates` is now tri-state: the new
  default `None` means "what this stage type has always done" (`True` for DAMH, `False` for MH),
  and an **MH** stage whose proposal needs surrogate gradients (`Hamiltonian`,
  `HamiltonianInfinite`, or a `block` proposal containing one) may set it to `True`. Such a stage
  polls the collector once per iteration — `AlgorithmBase._refresh_surrogate_evaluator_if_needed`,
  moved up from `Algorithm_DAMH` and shared, same request/poll protocol — and re-installs the
  proposal's gradient functions when a newer evaluator actually arrived, instead of using the one
  fetched at stage start for the whole stage. The stage still samples the exact posterior: its
  accept/reject test uses the exact model only, so any gradient field gives a valid MH kernel;
  only mixing changes. On any other MH stage `True` is refused with a printed warning and
  resolved to `False`, as it was silently before. **Affected configurations: MH stages that
  explicitly opt in — which is none that existed before, since the field was unconditionally
  forced off for MH.** Every existing MH stage (including `MH` + `Hamiltonian`) keeps exactly
  today's behaviour, proven in one MPI run by `tests/mpi/test_mpi_posterior.py::test_b10_…`: the
  opted-in stage refreshes ≥ 2 times per rank, the otherwise identical stage next to it refreshes
  0 times, and both match the closed-form posterior within 4 SE. Side effect of the same change:
  a `block` proposal containing a Hamiltonian sub-proposal is now recognised as needing a
  surrogate by both runners (`Stage.proposal_needs_gradients()`), so such a stage is handed an
  evaluator instead of failing with a missing gradient function.
- **RBF surrogate: duplicate snapshots collapsed, neighbor count capped, shift-fallback removed**
  (WS6, finding 3.3, `surrogates/rbf_scipy.py`). Before every fit, repeated parameter rows are
  collapsed into one row carrying the mean of their observations (`deduplicate_snapshots()`;
  exact duplicates by default, `dedup_tolerance=` for a coarser grid) — DAMH re-sends the
  current chain state whenever the chain stays put, so duplicates were the normal case and made
  the interpolation matrix exactly singular. Above `max_neighbors` (new argument, default 50)
  de-duplicated snapshots, `scipy` is asked for a local fit (`neighbors=50`) instead of one dense
  global solve; at or below the cap the fit is the global one, bit-identical to before. The old
  singular-matrix fallback — refit on `par` stacked with `par+1, par+2, …` carrying the *same*
  observations and `kernel="linear"`, which imposes the false constraint `f(x) = f(x+k·1)` — is
  **deleted**; what remains is a smoothing ladder (1e-10 … 1e-2, with a `RuntimeWarning`) that
  only relaxes exact interpolation and is unreachable for exactly duplicated data.
  **Every RBF-surrogate run's exact numbers, and hence its accept/reject sequence, change.**
  Nothing about the posterior does: DAMH's delayed-acceptance correction is exact whatever the
  surrogate returns (`10_manual_review_notes.md` §5), so this is a robustness/efficiency change,
  not a correctness one. Evidence: condition number of the thin-plate system on a 15-snapshot set
  with 4 duplicated rows **9.1e18 → 9.1e2**, and the duplicated-snapshot fit now reproduces all 15
  training observations to 7e-15 (max abs) on the first attempt, with no fallback; fit cost at
  N = 1000/2000/4000 snapshots **0.014/0.073/0.444 s → 0.001/0.001/0.002 s** (the dense O(N³)
  solve is replaced by per-query 50×50 ones: evaluating 50 points costs 2.2 ms instead of 1.4 ms),
  at the price of interpolation accuracy on a smooth test function (RMSE 8.1e-5 → 4.0e-3 at
  N = 4000; raise `max_neighbors` to trade it back).
- **Polynomial surrogate: `StandardScaler → PolynomialFeatures → Ridge` and a minimum-snapshot
  rule** (WS6, finding 3.10, `surrogates/polynomial_sklearn.py`). The parameters are standardized
  before the polynomial features are formed and the regression is ridge-regularized
  (`alpha=1e-6`, new constructor argument) instead of an unregularized `LinearRegression` on raw
  parameters. The degree escalation now starts from 0 instead of a hard-coded 1 and uses the same
  rule at every step, so the model never has more terms than there are snapshots: a single
  snapshot gives a constant fit, not a degree-1 hyperplane through one point. `weighting=
  "multiplicity"` still feeds the multiplicities to the fit (now as `ridge__sample_weight`).
  **Every polynomial-surrogate run's exact numbers, and hence its accept/reject sequence,
  change** — the pre-existing "unchanged under default weighting" unit test had its polynomial
  reference row re-baselined for exactly this reason (the difference is ≤1.5e-6 absolute,
  7.7e-6 relative; the note in `tests/unit/test_surrogates.py` states why). Again a numerical
  change only, for the same reason as above. Evidence: design-matrix condition number at degree 5
  with badly scaled parameters (×1e2 / ×1e-2) **9.2e20 → 1.1e2** (degree 3: 2.7e12 → 11.2);
  RMSE on an exact quadratic (200 snapshots) 1.5e-14 → **1.5e-7**, i.e. the deliberate ridge bias,
  which scales linearly with `alpha` (1.5e-9 at `alpha=1e-8`) and is ~1e-8 relative to the O(10)
  observations. One latent bug found on the way: `Ridge.predict` returns `(n,)` for a
  single-observation target where `LinearRegression` returned `(n, 1)`, so
  `PolynomialSklearnEvaluator.__call__` now reshapes explicitly to keep the
  `(n, no_observations)` evaluator contract.
- **Sub-chain surrogate freeze for DAMH-SMU** (WS3, finding 1.2, §2.5). The surrogate evaluator
  used inside one DAMH sub-chain is now refreshed at most once, *before* the sub-chain starts,
  instead of at the top of every inner step, so the delayed-acceptance correction telescopes
  under a single, well-defined surrogate. **Changes the accept/reject sequence of every DAMH
  stage with `surrogate_model_updates=True` and `subchain_max_length > 1`**; verified
  byte-identical for `subchain_max_length=1` and for any fixed (non-SMU) surrogate; V3/V3b
  re-recover the posterior at K=1/5/20 afterwards (§5b).
- **`evaluator_is_available()` semantics changed** from "any evaluator exists" to "a *new*
  evaluator is pending" (WS2, finding A27/M25, §2.1), matching the MPI provider. `Algorithm_DAMH`
  now re-scores a fixed surrogate roughly 3× less often per sub-chain step; proved
  byte-for-byte stream-identical for two surrogate types × three `subchain_max_length` values,
  since re-scoring an unchanged surrogate recomputes the same numbers it already had. Only
  `use_collector=False` runs with a fixed `surrogate_evaluator=` and a DAMH stage are affected at
  all — every `use_collector=True` run is untouched.
- **Exact re-evaluation after a `use_only_surrogate` stage** (WS3, finding A11, §2.6). The sample
  carried into the next stage after a `use_only_surrogate=True` stage now drops its surrogate-only
  observations/likelihood, forcing that next stage to re-evaluate it with the exact model.
  Previously the next exact stage's first acceptance ratio compared an exact log-likelihood
  against a stale *surrogate* one. New failure mode: if the exact model fails exactly at that
  carried-over state, the run now raises `RuntimeError` (intended fail-loud behaviour) instead of
  silently continuing on surrogate values.
- **`state_dependent_approximation` default kept at `False`, `True` now warns** (WS0 §B8/G7,
  author decision 1, finding 1.4) — **superseded within this unreleased cycle**: the field itself
  was removed on 2026-09-18, see the breaking-changes entry above. The default was `False`
  throughout, so runs are unaffected either way.
- **pCN requires a Gaussian internal prior** (WS7, finding 1.10, §2.9). `build_proposal` raises
  `ValueError` for `proposal_type="pCN"` unless the prior is `Normal` or
  `PriorIndependentComponents`; `FromScipy`/`GaussianMixture` are rejected even if they happen to
  be Gaussian (no reliable way to detect it — stays rejected by author decision, §7b). Previously
  a non-Gaussian pCN prior silently targeted the wrong measure. No effect on any script that
  already used a Gaussian prior with pCN.
- **`needs_gradients ⇒ use_surrogate_gradients` enforced for every proposal type** (WS7, §2.9),
  including a `BlockProposal` containing a Hamiltonian sub-proposal (previously only bare
  Hamiltonian/HamiltonianInfinite were checked, finding B3). Configurations that already ran with
  the flag enabled are unaffected.
- **Adaptive random walk: degenerate acceptance weights no longer crash** (WS7, finding A15c/d,
  §2.9). If every acceptance weight in an adaptation period is zero or non-finite, that period's
  adaptation is now skipped (with a `RuntimeWarning`) and the previous covariance is kept, instead
  of `ZeroDivisionError`. Only the previously-crashing case is affected; a non-degenerate period
  is bit-identical.
- **`Stage.adaptive_target_rate`/`adaptive_corr_limit`/`adaptive_sample_limit` wired to the
  adaptive proposal** (WS0 §G1, finding 1.3, author decision "do G1–G6 as recommended",
  2026-09-17, §2.13). These fields were previously accepted but silently ignored (defaults 0.25 /
  0.3 / unbounded always applied); `build_proposal` now forwards each field when the stage sets
  it. **Changes any run that sets one of the fields**; no in-tree script does. Runs that leave all
  three `None` are bit-identical (verified on a fixed-seed 80-step `adapt()` sequence).
- **Adaptive covariance reduction always exchanges a full covariance matrix** (WS0 §G2, finding
  2.5, §2.13). Both runners normalise `proposal.sd_or_cov` to `(p, p)` before the per-stage
  `Allreduce`, removing the intermittent `Message truncated` abort of adaptive DAMH stages, and
  `proposal_type="block"` + `adaptive=True` now raises a clear `ValueError` instead of dying with
  `AttributeError` at the end of the stage. Changes the sample stream only where no rank ever
  adapted (next stage draws from `diag(sd**2)` instead of a 1-D `sd` vector — same distribution,
  different RNG stream); runs where every rank adapted at least once are bit-identical.
- **`lhs_normal` returns the maximin-best of its 5 candidate designs** (WS0 §G3, finding 3.8,
  §2.13), instead of always the last one (`maxmin` was never updated). Changes the initial samples
  of every `initial_sample_type="lhs"` run with more than ~2 chains (better-spread starts, same
  distribution family).
- **The initial sample is drawn from a per-chain seeded generator** (WS0 §G4, finding 1.9/A14,
  §2.13). `Distribution.rvs()` takes an optional `generator`; both runners now pass a seeded
  generator (`10*no_stages*rank_world + 3`, alongside the proposal's `+1` and the algorithm's
  `+2`, formula in `surrDAMH/modules/seeds.py`) for `initial_sample_type="prior"`/`"user_specified"`.
  **`initial_sample_type="prior"` runs become reproducible and their initial samples change** —
  they were drawn from the unseeded global NumPy RNG before, so there is no previous value to
  match. External callers of `rvs()` that pass no generator are unaffected.
- **`BlockProposal` re-seeds its sub-proposals** (WS0 §G5, finding 1.9/A13, §2.13). Every
  sub-proposal's RNG is now re-seeded per rank and per stage from the block's own seed
  (`Proposal.reseed()`, `proposals.subproposal_seed`). Before this, every MPI chain ran an
  identical sub-proposal stream, and a sub-proposal object reused across stages continued the
  previous stage's stream. **Changes every block-proposal run** (intended: the old behaviour was
  a correctness defect).
- **A state carried across a stage boundary is counted once** (WS0, finding 5.7, author decision
  2026-09-17, §2.13). The first row of a stage whose initial state was carried over from the
  previous stage now drops the leading `+1` from its `multiplicity`/weight (it was already
  counted by the previous stage's final row, which keeps its full weight). Per-stage invariant:
  `sum(weight) == accepted+rejected+prerejected+1` for a run's first stage, `== ...+0` for every
  later one. **Changes the multiplicity column of every multi-stage run**; single-stage runs are
  bit-identical. A stage's first row may now have `weight = 0`; `decompress()`/`no_unique_samples`
  handle it (see the matching bug-fix entry below).
- **NN surrogate training weighting flipped to `"uniform"`** (WS6, author decision 3, §2.16a).
  Every updater now takes `weighting: Literal["uniform", "multiplicity"] = "uniform"`. Before, the
  NN updater always weighted its loss by the snapshot multiplicity, so rejected proposals
  (multiplicity 0) were never trained on; now they are, counting once each. Pass
  `weighting="multiplicity"` for the old semantics. Poly/RBF/kd-tree ignored weights entirely
  before, which is what `"uniform"` does, so they are bit-identical under the default (pinned by
  hard-coded pre-WS6 reference values); `PolynomialSklearnUpdater` gained real weighted least
  squares for `weighting="multiplicity"`.
- **NN output normalization default flipped identity → likelihood** (WS6, §2.16b).
  `NeuralNetworkUpdaterMinibatches(output_normalization=...)` defaults to `"likelihood"`: targets
  centred on `likelihood.mean`, scaled by the per-observation noise sd, instead of no
  normalisation at all. **Surrogate accuracy changes for every existing NN run**; `"identity"`
  restores the old default, `"manual"` the old explicit `output_mean`/`output_scale`.
- **`initial_snapshots` counted once** (WS8, finding 2.8, §2.12). Preloaded snapshots are now
  counted where they are actually consumed instead of twice; `surrogate_quality.csv`'s first
  `snapshots_total` for N preloaded snapshots is now N (previously 2N). **Changes when the
  surrogate is (re)trained, hence the DAMH-SMU sample stream, for any run that passes
  `initial_snapshots` or restores a pretrained updater.** Runs without `initial_snapshots` are
  unaffected.
- **Fail-loud abort on any rank's exception** (WS8, §2.10). `SamplingFramework.run()` wraps every
  MPI role body, and `process_CHILD.py` wraps the spawned solver loop: an uncaught exception
  anywhere now prints a traceback and calls `MPI.COMM_WORLD.Abort(1)` for the whole job
  (`KeyboardInterrupt`/`SystemExit` are re-raised, not converted). Configurations that previously
  ran to completion are unaffected; configurations that previously hung on a rank's exception now
  terminate within seconds instead.
- **Start-up handshake, collector → samplers** (WS8, finding 2.2, §2.11). The collector now tells
  every sampler, before the stage loop, whether it can already provide an evaluator; a first
  DAMH/Hamiltonian stage with no evaluator available then raises `RuntimeError` on all ranks
  (job-wide abort) instead of deadlocking. A first stage that provides `initial_snapshots` or a
  pretrained/restored updater is unaffected (still runs).
- **Unknown `algorithm_type` now raises on every rank at start-up** (WS4 side effect, §2.8). Stage
  names are assigned by `stages.stage_name` on every rank before role dispatch, so an invalid
  `Stage.algorithm_type` is caught before solvers are spawned or earlier stages run, instead of
  only when the sampler reached that stage.
- **`load_last_samples` mismatch now warns** (WS4, §2.9). Continuing with more saved chains in the
  source run than the new run's chain count now emits an explicit `RuntimeWarning` (previously
  silently used the first N sorted by filename; still does, but now says so).
- **`GaussRandomWalk_adaptive`'s internals replaced; every `adaptive=True` stage's sample stream
  changes** (2026-09-20, `16` §5 item 2, prototype formulas in that study's
  `proto/RESULTS_PROTO.md` §1 "AMShrinkRM"). The class now keeps a Welford `(n, mean, M2)` of
  the **post-decision chain states** and, every `period=10` calls once `n >= warmup=100`,
  installs `base_cov = (2.38²/d)·(C + 1e-6·(tr C/d)·I)` with
  `C ← (1−δ)C + δ·(tr C/d)·I`, `δ = min(1, 2d/n)` (Haario et al. 2001 estimator +
  Ledoit–Wolf-style vanishing shrinkage); the scale is a Robbins–Monro recursion
  `log σ ← log σ + n^(−0.7)(α − 0.234)` (Andrieu–Thoms 2008 Alg. 4) and the proposal is
  `x + exp(log σ)·L z`. Gone with it: the correlation clip, the bounded history, the `coef`
  feedback, the stored sample/weight arrays and the degenerate-weights `RuntimeWarning`. The
  ridge is **relative** to `tr C/d`, not the prototype's absolute `1e-6`, so it does not depend
  on the units of the parameters. `sd_or_cov` is now always a 2-D covariance
  (`exp(2 log σ)·base_cov`), so every existing reader of it keeps working.
  **Affected configurations: every stage with `adaptive=True`, and every later stage that takes
  its covariance from one.** Measured on the d=10, κ=10 Gaussian of
  `tests/unit/test_proposals.py::test_adaptive_random_walk_reaches_target_rate_scale_and_shape`:
  acceptance 0.228 (target 0.234), shape error 0.078, scale 1.12× the `2.38²/d·C` optimum.
  **What is bit-identical, and how it was checked:** `GaussRandomWalk`, `PCN`, `Hamiltonian`,
  `HamiltonianInfinite` and `BlockProposal` are byte-for-byte unchanged (`diff` against a
  pre-change snapshot of the package), so every non-adaptive stage is untouched; the separate
  move of the `adapt()` call to *after* the accept/reject decision was verified not to change
  the stream at all by running an `MH-adaptive → MH(None)` and an adaptive `DAMH` chain through
  `run_local` before and after the move with the **old** adapter still in place — identical
  sha256 of all three `samples/*.csv`. The unit (397), validation (14) and MPI suites pass.
- **The cross-rank hand-over of an adaptive stage pools statistics instead of averaging
  covariances** (2026-09-20, `16` §5 item 2; Craiu et al. 2009, Solonen et al. 2012). Every
  sampler `Allgather`s `Proposal.adapted_state()` and combines the same pooled state, so all
  ranks carry an identical proposal into the next stage; before, the ranks `Allreduce`d their
  own covariances and the study measured them disagreeing by 6–17× (`15` §2.4). **Affects every
  MPI run with `no_samplers > 1` and an adaptive stage**; a single-chain `run_local` run keeps
  its own adapted state exactly, as before.
- **Adaptive random walk: effective covariance continuous across a covariance install;
  `proposal_sd_or_cov` no longer required on an adaptive stage** (2026-09-21, reported by the
  author on a `proposal_sd_or_cov=0.001` stage that "did not adapt to reasonable values").
  Cause: the Robbins–Monro scale `log σ` is learned relative to `base_cov`; at `n = warmup` the
  empirical estimate replaced `base_cov` while `log σ` was kept, so the effective covariance
  `exp(2 log σ)·base_cov` jumped by the ratio of the two traces. Measured on a d=4 unit Gaussian
  (optimal sd ≈ 1.2): from sd 0.001 the effective sd went 0.81 → **776** at n=100 and was still
  10 at n=2000 with the chain frozen; from sd 0.01 it went 1.03 → 121. The mirror case (start
  1.0, posterior sd 1e-3) collapsed to 4e-6 at n=1000. `_install_covariance_from_statistics` now
  shifts `log σ` by `½·log(tr base_old / tr base_new)` so that `tr(exp(2 log σ)·base_cov)` is
  unchanged by the install; the shape update changes only the shape. After the fix every start
  from 0.001 to 100 reaches the same effective sd 1.4 ± 0.05 and acceptance 0.20–0.22 at n=5000.
  **Affected configurations: every stage with `adaptive=True` and a random-walk proposal, and every
  later stage that takes its covariance from one** — the warm-up trajectory differs (a start far
  from the posterior scale now recovers; a start at the right scale, as in the maintained
  examples, sees only a small shift), the long-run target does not. Both call sites are covered:
  the per-period install and the cross-rank pooling in `set_pooled_state`; for the latter,
  `adapted_state()` now carries the effective log-scale `log σ + ½ log tr(base_cov)` instead of
  `log σ` (same vector length, so the `Allgather` buffers are unchanged), the ranks average that
  and convert it back against the pooled base — averaging the bare `log σ` would have given each
  rank a different pooled proposal once `log σ` became base-relative in this way. Non-adaptive
  proposals are untouched. Because the start now only shapes the warm-up, `build_proposal` no longer
  requires it: an adaptive RWMH stage with `proposal_sd_or_cov=None` and nothing carried over
  starts from the prior covariance × `2.38²/d` (`Normal`, `PriorIndependentComponents`), or from
  sd `1.0` with a printed warning for a prior without `get_covariance()`; a **non-adaptive** RWMH
  stage in that situation raises `ValueError` (it was an `AssertionError` before). Tests:
  `tests/unit/test_proposals.py::test_adaptive_random_walk_effective_scale_is_continuous_at_install`,
  `::test_build_proposal_adaptive_rwmh_default_start_*`.
- **Not changed on purpose: a Hamiltonian stage with `proposal_sd_or_cov=None` still gets the mass
  `1.0`**, never the carried random-walk covariance, although every other `None` field now resolves
  through the carry-over dict (2026-09-20). `16` §5 item 5 measured `M = Σ̂` at 0.06–1.2 of the
  random-walk oracle against 19–29× for `M = Σ̂⁻¹`, and carrying the inverse needs the step size
  re-adapted at fixed integration time — an open decision. Set the mass explicitly.

- **`Stage.adaptive` defaults to "adapt unless the step is pinned"; a non-adaptive random walk
  without a step no longer raises** (2026-09-21, author decisions; §2.27). `adaptive` is now
  `bool | None = None`: `None` resolves in `__post_init__` to `True` when the step field of the
  stage's `proposal_type` was left `None` (`proposal_sd_or_cov` / `pcn_beta` /
  `hamiltonian_step_size`) and to `False` when it was given; always `False` for `"block"`. An
  explicit value wins. **Every maintained example and every configuration that gives its step
  keeps its fixed proposal** (verified: 13/13 `toy_examples` scripts give the step). What changes:
  a stage that leaves the step `None` -- previously either the carry-over consumer of an adaptive
  stage or (first stage) an error -- is now adaptive and keeps tuning; seven tests that used such a
  stage as a frozen consumer were pinned with `adaptive=False`. Second change, same day: an explicit
  `adaptive=False` random-walk stage with no `proposal_sd_or_cov` and nothing to carry over gets the
  prior covariance scaled by `2.38^2/d` for the whole stage plus a printed warning, instead of the
  `ValueError` introduced earlier that day. Net effect for a newcomer: `Stage(max_evaluations=200)`
  runs (adaptive random walk from the prior-scaled start). Unit + validation: 481 passed.

- **`Configuration.min_snapshots_to_update` defaults to 0; `Updater.needs_retraining(new_snapshots)`
  gates every retraining** (2026-09-23, author decision after `library_notes/18_*` §3.1: the network
  surrogate had been starved by the retraining cadence, 67 vs 646 retrainings = test RMSE 0.0039 vs
  0.0012 and 8.3x vs 10.4x MH efficiency). The hook's default is `new_snapshots > 0`, so the
  polynomial, RBF and k-d-tree updaters behave exactly as with the old default 1 -- one refit per
  new batch, never a refit (nor a re-send) of the identical model, which before only the polynomial
  half-guarded (it skipped the fit but not the publish; RBF rebuilt its interpolant and the k-d tree
  its tree on every call). `NeuralNetworkUpdater.needs_retraining` returns `True` once it has data,
  so with the default it now trains on every collector loop (posterior-affecting for DAMH-SMU runs
  with a network that did not set the field: many more `iterations_batch` steps, a better surrogate,
  a different sample stream). `run_local`'s `LocalSurrogateManager` applies the same rule (it has no
  idle loop, so a network there is still retrained only when snapshots arrive). Docs:
  `docs/configuration.md`, `docs/writing_a_surrogate.md` contract table. Tests:
  `tests/unit/test_surrogates.py::test_needs_retraining_default_and_network_override`,
  `::test_local_manager_with_min_snapshots_to_update_zero_never_refits_identical_data`.

- **`toy_examples/grf_diffusion.py`: observations are now evaluated in the cells that contain the
  measurement points** (2026-09-28, `library_notes/19_efficiency_study_2026-09-28.md` §0,
  `10_manual_review_notes.md` §3). `Solver_diffusion_GRF.get_observations` passed dummy cell
  indices (all 0) to `Function.eval`, so every observation was the bilinear extrapolation of mesh
  cell 0 and the forward map had rank 2 whatever the number of observation points (invisible for a
  constant field, where the extrapolation is exact). The colliding cells are now found once in
  `__init__`. **Every GRF example's artificial observations and posterior change**; the numbers in
  notes 14 and 18 describe the rank-2 problem. Library code is untouched.

- **Fixed (2026-10-09): `run_sampling_auto`/`run_sampling_local_auto`'s warm-up stage silently
  restarted the chain instead of continuing it** (`surrDAMH/auto.py`, found while specifying
  `Stage.burn_in`, API spec `burn_in_report`). The warm-up was built with `Stage(...,
  is_excluded=True)`; `is_excluded` makes the *next* stage restart from the sample the excluded
  stage itself started at (`docs/stages.md`), so chunk 1 of every Auto run began at the plan's
  initial LHS/prior point rather than at the warm-up's last (tuned) state — every warm-up
  evaluation past the first contributed nothing to chain continuity (the tuned proposal scale was
  still carried over correctly; only the chain *state* restarted). Fix: the warm-up now uses the
  new `Stage(burn_in=True)` (below) instead, which continues the chain like a normal stage while
  still defaulting it out of the posterior. **Changes the sample stream of every
  `run_sampling_auto`/`run_sampling_local_auto` run that gets a warm-up stage** (`B >= 50·d` and a
  collector available, see `docs/running.md#automatic-mode-2026-10-08`; the single-MH-stage layout
  has no warm-up and is unaffected). No `run_sampling`/`run_sampling_local` script that builds its
  own stage list is affected. Test:
  `tests/unit/test_burn_in_report.py::test_auto_local_chunk_one_continues_from_the_warm_up`.

- **Proposals outside the prior bound are no longer evaluated** (2026-10-09, S0, `Problem(prior_bound=8.0)`
  is the default). A huge-but-finite proposal (|u_i| > 8 internal-prior scales) no longer reaches the
  solver or the surrogate (MH outer step, every DAMH sub-chain step, Hamiltonian trajectories): it is
  rejected with tag `-3` and never sent to the collector. The sampled target is the prior truncated to
  the box (removed mass about d * 1.24e-15 at R = 8, 1.2e-13 for d = 100). A run whose chains never leave the box is
  byte-identical in `samples`/`raw_data`/`subchain_stats`/`last_sample`; `notes` gains the
  `out_of_bounds` column (see Additions). Evidence: md5 of plain local/MPI reference runs and
  `tests/unit/test_prior_bound.py::test_runs_inside_the_box_are_byte_identical_to_the_code_before_s0`.
  Priors other than `Normal`/`PriorIndependentComponents`/`GaussianMixture`/`FromScipy(multivariate_normal)`
  now need `prior_bound=None` (`ValueError` otherwise). Also fixed on the way: a DAMH sub-chain none of
  whose steps was evaluated (all non-finite or out of bounds) right after a new evaluator arrived left
  the chain state scored with the old surrogate for the next sub-chain; the re-scoring is now carried
  over (cannot occur in a run without such proposals).

- **Adaptive `RandomWalk`/`PCN` in a DAMH stage adapt their step on the sub-chain's own
  acceptance** (2026-10-10, S6; note 21 §6 item 4, note 16 §2 items 1-2, note 19 §4.2,
  `10_manual_review_notes.md` item 14 of the "2026-10-08" section; `docs/stages.md`, `docs/outputs.md`).
  `GaussRandomWalk_adaptive.adapt` / `PCN_adaptive.adapt` now run their Robbins-Monro scale
  recursion (`log_sigma` / `logit_beta`, target 0.234) once per SUB-CHAIN step, on
  `min(1, exp(la_j))` of each of the `subchain_length = K` steps the algorithm already passes as
  `subchain_log_acceptance_probabilities` (pre-rejected iterations included, `-inf` = 0); the
  outer DAMH acceptance no longer drives the step (it measures the surrogate error). New
  counter `n_scale` (scale steps: `= n` in MH, K per outer iteration in DAMH); `n` stays the
  number of `adapt()` calls and keys the Welford shape update, the covariance install cadence,
  the `adaptive_stats` rows and the pooled state (`adapted_state`/`set_pooled_state`/
  `carry_over` layout unchanged). `adaptive_stats` column `mean_acceptance_probability` of a DAMH
  stage is now the mean SUB-CHAIN per-step acceptance probability of the period (was: the outer
  one). `Algorithm_DAMH`, seeds, MPI messages and output formats unchanged; Hamiltonian dual
  averaging untouched (it already worked this way).
  **Affected: exactly the DAMH stages whose stage proposal is an adaptive `RandomWalk` or `PCN`
  (any K, K = 1 included), and every stage downstream of them that consumes their carry-over.**
  At K = 1 with an exact surrogate the only difference is the pre-rejected iteration (scored 0
  before, the sub-chain probability of the rejected proposal now); with an inexact surrogate the
  moved iterations differ too (exact vs surrogate probability). **Byte-identical** (md5 of
  samples/raw_data/notes/subchain_stats/adaptive_stats/last_sample/carry_over, references made
  before the change): MH stages with adaptive RW and pCN, fixed-step RW/pCN DAMH stages, a
  fixed-step DAMH stage with `exact_step_probability=0.3` (its exact random walk adapts on the MH
  path), an adaptive Hamiltonian DAMH stage, the plain MPI/one-process/local-runner reference
  scripts; the MH stages before an adaptive DAMH stage. Evidence (2-D linear-Gaussian problem,
  exact surrogate so the surrogate posterior is the posterior, `run_sampling_local`, MH warm-up
  1000 evaluations, then DAMH 20 000 exact evaluations; 3 seeds, mean ± sd; fixed-step grids of
  10 RW sd-equivalent steps / 9 pCN betas, 2 seeds; "ESS" = min over parameters, decompressed
  chain, 10 % burn-in):

  | proposal | K | adapted BEFORE | adapted AFTER | best fixed step (ESS per sub-chain step) | ESS/eval BEFORE → AFTER (best fixed) | ESS per sub-chain step BEFORE → AFTER (best fixed) |
  |---|---|---|---|---|---|---|
  | RW sd | 1 | 0.587 ± 0.003 | 0.724 ± 0.003 | 0.45 | 0.376 → 0.438 (0.556) | 0.114 → 0.103 (0.124) |
  | RW sd | 5 | 1.50 ± 0.007 | 0.715 ± 0.006 | 0.45 | 0.622 → 0.654 (0.663) | 0.039 → 0.094 (0.101) |
  | RW sd | 20 | 3.03 ± 0.017 | 0.716 ± 0.005 | 0.45 | 0.665 → 0.855 (0.887) | 0.010 → 0.042 (0.044) |
  | pCN β | 1 | 0.579 ± 0.002 | 0.708 ± 0.001 | 0.45 | 0.376 → 0.470 (0.560) | 0.113 → 0.110 (0.120) |
  | pCN β | 5 | 1.000 (saturated) | 0.705 ± 0.004 | 0.6 | 0.700 → 0.653 (0.719) | 0.063 → 0.093 (0.100) |
  | pCN β | 20 | 1.000 (saturated) | 0.705 ± 0.003 | 0.45 | 0.878 → 0.876 (0.892) | 0.038 → 0.043 (0.045) |

  After S6 the adapted step is the same for every K (≈ the MH stage's: sd 0.69-0.71 after the
  1000-evaluation warm-up) instead of growing with K (RW 1.5× / 6.7× the per-step optimum at
  K = 5 / 20 before; pCN driven to β = 1, the independence sampler). The residual ≈ 1.6× above
  the per-step optimum is the 0.234 target in d = 2 (the MH stage has it too), not S6. ESS per
  sub-chain step (the surrogate cost) rises 2.4× / 4.1× (RW K = 5 / 20) and 1.5× (pCN K = 5);
  ESS per exact evaluation rises 1.29× at RW K = 20 and 1.05-1.25× elsewhere, except **pCN K = 5:
  −7 % (0.700 → 0.653)** — with an exact surrogate pre-rejections cost no exact evaluation, so
  ESS/eval rewards over-large steps (its K = 1 "optimum" is at the grid edge, sd 3.3 / β 0.95);
  on a real surrogate that saturated β = 1 has no such free lunch. Posterior: max |mean − exact|
  over 3 seeds 0.6-2.6 SE before and after (K = 1: 2.1 → 2.6 RW, 0.8 → 1.1 pCN). Pinned by
  `tests/unit/test_subchain_adaptation.py` (14 tests: recursion vs hand computation and vs the
  pre-S6 code; K = 20 adapted step within 1.25× of the MH-adapted step and < 2× the fixed
  optimum; K = 1 closed-form posterior; MH byte identity); the 14 `damh_adaptive` keys of
  `tests/unit/test_safety_kernel.py::REFERENCE_MD5` (its two adaptive DAMH stages) were
  re-recorded. Not changed: `Auto` keeps sub-chain length 1 in robust mode (the longer sub-chain
  this enables is an author decision, item 13 (a) of the same section).

- **One persistent posterior-covariance estimate per chain; Hamiltonian mass from it;
  integration time; dual-averaging state carried across stages** (2026-10-11, C1, author
  decision D7, `library_notes/25_robust_by_default_roadmap_2026-10-08.md` §9, `16` §5 items 4-5;
  `docs/stages.md` "Posterior statistics", `docs/outputs.md`). New MPI-free
  `surrDAMH.modules.posterior_statistics.PosteriorStatistics` (Welford `(n, mean, M2)` of the
  post-decision chain states, internal space; `update`, `merge`/`from_rows` (Chan-Golub-LeVeque,
  the formula the random walk used), `covariance()` (the random walk's former shrinkage + relative
  ridge estimator without `2.38²/d`), `precision()`, `as_vector`/`from_vector`,
  `is_informative`). Every algorithm updates the chain's one object once per outer iteration of
  every stage (MH, DAMH incl. pre-rejected iterations and S2 exact steps), after the decision and
  before `adapt()`; both runners pool it at EVERY stage end (one extra `Allgather` of
  `1 + d + d²` floats on the sampler communicator, no new message type), write it to
  `carry_over/<stage>.npz` for every stage (`carry_over__posterior_statistics`,
  `summary__posterior_statistics_n`; a non-adaptive stage now also gets a file) and hand it to the
  next stage and, via `continue_sampling*` (also `chains="prior"/"lhs"`), to continued runs;
  `problem=` drops it with a note. `GaussRandomWalk_adaptive` reads the shared object
  (`statistics=`; `None` = private, as before — the S2 exact step's own random walk keeps a
  private one); its `adapted_state()`/`set_pooled_state()` now carry only the effective log-scale
  (length 1). `Hamiltonian(mass="covariance")` resolves `M = Σ̂⁻¹` at stage start (identity + note
  below `MASS_MIN_STATES = 100` states), fixed within the stage; `Hamiltonian(integration_time=T)`
  (exclusive with `num_steps`, whose default is now `None` = 10) sets `num_steps =
  max(1, round(T / ε))` on every step change, **capped at 1000** (`INTEGRATION_TIME_MAX_STEPS`,
  a safety bound not in the spec), and appends a `num_steps` column to the adaptation trace of
  such a stage only; the dual averaging carries `dual_averaging = [m, H̄, log ε̄, μ]` and
  `hamiltonian_mass` and a following adaptive Hamiltonian stage with `step_size=None` CONTINUES
  the recursion when its resolved mass equals the carried one (rtol 1e-12), else restarts from the
  carried step ("mass changed: step size re-adapted"). `run_manifest.json` gains
  `proposal_resolution` (per stage: statistics count before/after, mass source, `T`, initial
  `num_steps`/step, dual averaging fresh/continued). Auto `mode="fast"`:
  `Hamiltonian(integrator="dimension_robust", mass="covariance", integration_time=3.0)`
  (`AUTO_FAST_INTEGRATION_TIME`, placeholder; was `num_steps=30, mass=1.0`). Report: the
  carry-over block shows the statistics' `n` and `tr C / d`.
  **Affected (sample stream):** (a) every adaptive random-walk stage AFTER the first stage of a
  run (or in a continued run) — its shape starts from the pooled statistics of everything before
  and installs at its first period boundary (n = 10 instead of 100), and its estimate keeps all
  earlier states (burn-in included: never discarded); (b) every adaptive Hamiltonian stage with
  `step_size=None` following an adaptive Hamiltonian stage with the same mass (continued dual
  averaging instead of a restart at `10 ε̄`); (c) Auto fast mode; (d) everything downstream of
  these. **Byte-identical** (md5 of samples/raw_data/subchain_stats/notes/adaptive_stats/
  last_sample and of every pre-existing carry_over key, references made before the change,
  scratchpad `c1/`): single-stage adaptive RW in MH and DAMH (K = 5), single-stage adaptive
  Hamiltonian (MH leapfrog, DAMH dimension-robust), fixed-matrix / fixed-vector-mass Hamiltonian
  stages, fixed RW → fixed pCN, adaptive pCN → adaptive pCN, adaptive RW → fixed-RW DAMH with
  `exact_step_probability = 0.2`, adaptive RW → adaptive Hamiltonian DAMH (fresh), the FIRST
  stage of every case; MPI 2-sampler: single-stage adaptive RW and stage 1 of every multi-stage
  case; the plain MPI / one-process / local-runner reference scripts (samples, notes,
  last_sample; a `carry_over/` directory is new). **Differs (intended):** stage 2 of adaptive
  RW → adaptive RW (MH → MH, MH → DAMH), stage 3 of RW → adaptive HMC → adaptive HMC (continued
  dual averaging), local and MPI. `tests/unit/test_safety_kernel.py::REFERENCE_MD5`: the 14
  `damh_adaptive` stage-1/2 keys re-based (adaptive RW after adaptive RW), C1-only carry-over keys
  stripped. Evidence 1 — random-walk shape, stage 2 of a two-stage adaptive-RW run
  (`run_sampling_local`, exact linear surrogate for DAMH (K = 5), MH warm-up W then stage 2 of S
  evaluations, 3 seeds via `lineage_generation`, mean ± sd; "lg2" = the 2-D linear-Gaussian
  problem, W/S = 1000/4000; "aniso10" = 10-D linear-Gaussian with rotated posterior sds
  log-spaced 0.5 → 0.05, W/S = 2000/8000, "short" S = 500; shape error = relative Frobenius error
  of `C/tr C` vs `Σ/tr Σ` of the stage-2 adapted covariance; cond = condition number of
  `C^-1/2 Σ C^-1/2` (1 = perfect shape); ESS = Geyer initial-monotone, min over parameters,
  decompressed chain, 10 % burn-in):

  | problem | stage 1 → 2 | first install n | shape error before → after | cond before → after | ESS/eval before → after |
  |---|---|---|---|---|---|
  | lg2 | MH → MH | 100 → 10 | 0.060 ± 0.033 → 0.042 ± 0.026 | 1.17 → 1.11 | 0.112 ± 0.007 → 0.109 ± 0.014 |
  | lg2 | MH → DAMH | 100 → 10 | 0.015 ± 0.005 → 0.013 ± 0.006 | 1.04 → 1.03 | 0.637 ± 0.055 → 0.617 ± 0.068 |
  | aniso10 | MH → MH | 100 → 10 | 0.119 ± 0.018 → 0.097 ± 0.017 | 1.76 → 1.56 | 0.025 ± 0.004 → 0.022 ± 0.001 |
  | aniso10 | MH → DAMH | 100 → 10 | 0.039 ± 0.003 → 0.046 ± 0.005 | 1.23 → 1.27 | 0.173 ± 0.014 → 0.166 ± 0.004 |
  | aniso10 | MH → MH, S = 500 | 100 → 10 | 0.443 ± 0.019 → 0.213 ± 0.018 | 11.8 ± 4.4 → 3.19 ± 0.10 | 0.015 ± 0.007 → 0.024 ± 0.009 |
  | aniso10 | MH → DAMH, S = 500 | 100 → 10 | 0.138 ± 0.036 → 0.181 ± 0.016 | 2.82 → 2.78 | 0.088 ± 0.007 → 0.131 ± 0.012 |

  The shape is installed at once in every case. For long second stages the end-of-stage shape and
  ESS/eval are unchanged within the seed spread (before, the stage re-learned the shape from its
  own 4000-8000 states anyway); for a short (Auto-chunk-like) second stage ESS/eval rises 1.5×.
  The slightly worse long-DAMH shape on aniso10 (0.039 → 0.046) is consistent with the estimate
  keeping the warm-up's initial transient (statistics are never discarded) — flagged, not fixed.
  Evidence 2 — mass (aniso10, exact linear surrogate with Jacobian, MH adaptive-RW warm-up 2000,
  then DAMH K = 1 `Hamiltonian(integration_time=3.0)`, adaptive step, 6000 evaluations, 3 seeds;
  outer acceptance 1.0 everywhere (exact surrogate), "sub-chain acc." = the dual-averaged rate):

  | integrator | mass | final ε / num_steps | sub-chain acc. | ESS/eval (x) | ESS/eval (x²) | max mean dev (ESS-SE) | cov rel. err |
  |---|---|---|---|---|---|---|---|
  | dimension_robust | 1.0 | 0.070 / 51 ± 9 | 0.798 | 0.024 ± 0.003 | 0.068 ± 0.006 | 1.9 ± 0.9 | 0.044 ± 0.010 |
  | dimension_robust | "covariance" (n = 2000) | 1.03 / 3 | 0.800 | 1.60 ± 0.43 | 0.109 ± 0.016 | 1.9 ± 0.3 | 0.053 ± 0.017 |
  | leapfrog | 1.0 | 0.070 / 43 ± 7 | 0.798 | 0.022 ± 0.003 | 0.056 ± 0.013 | 1.5 ± 0.9 | 0.051 ± 0.008 |
  | leapfrog | "covariance" (n = 2000) | 1.02 / 2.7 | 0.800 | 1.58 ± 0.48 | 0.120 ± 0.027 | 1.7 ± 0.3 | 0.072 ± 0.020 |

  Covariance mass ≫ identity for the mean (67-72× ESS/eval, ESS > N: antithetic) at ~17× fewer
  gradient steps per proposal; for second moments only 1.6-2.2× — with a near-perfect mass and
  `T ≈ π` the trajectory maps x → −x in whitened coordinates (x² barely moves). Jittering `T`
  (note 16 §5 item 4 "jittered") would address that; not in the C1 spec, not implemented.
  Both posteriors match the closed form (mean within 1.5-1.9 ESS-SE max over 10 parameters,
  covariance within 4-7 %). Integrator check: dimension-robust with a full / diagonal mass `M`
  follows the leapfrog trajectory with the same `M` to 1.8e-9 / 5.1e-9 (step 1e-4, T = 1, d = 4);
  identity mass bit-identical to the pre-C1 rotation (the existing code already rotated each
  eigen-coordinate of `M` with frequency `1/√λ`; no integrator change was needed). Pinned by
  `tests/unit/test_posterior_statistics.py` (21 tests) and `tests/mpi/test_mpi_posterior.py::
  test_b13_fast_mode_like_chunks_with_the_covariance_mass`.

- **C1b: jittered integration time, dual-averaging continuation across a small mass change, burn-in
  discount of the posterior statistics, S2 exact step on the shared statistics** (2026-10-11, author
  decisions on the C1 follow-ups; `docs/stages.md` "Posterior statistics" / Hamiltonian fields,
  `docs/outputs.md`, `docs/running.md`). Evidence (references made with the code BEFORE C1b,
  scratchpad `c1b/` of that session: `before*.json` / `after*.json` md5 tables, `results.jsonl`,
  `table.md`); means ± sd over 3 seeds; "10-D" = the C1 anisotropic Gaussian (posterior sds 0.5 → 0.05,
  rotated, exact linear evaluator).
  **Byte-identical** (md5 of samples/raw_data/subchain_stats/notes/adaptive_stats/last_sample and every
  carry_over key): every single-stage run, the plain scripts (MPI/one-process/local runner), fixed-step
  stages, adaptive RW → RW/DAMH/HMC without `burn_in`, `is_excluded` stages, MH → HMC → HMC with an
  explicit mass (r = 0: continued as before), `Hamiltonian(integration_time=T)` with
  `integration_time_jitter=0.0`, the MPI cases (2 samplers) without burn-in. **Differs (intended)**:
  (B) stages after a `burn_in=True` stage (the burn-in stage itself is identical except the carried
  statistics vector/count and the new summary key); (D) an adaptive Hamiltonian stage carrying a
  dual-averaging state whose mass changed (any change: continued if r ≤ 0.2, else the new rescaled
  restart); (S) every DAMH stage with `exact_step_probability > 0` (also as the first stage);
  (J) Auto fast mode (new T and jitter). Manifest: `proposal_resolution` gains
  `posterior_statistics_discounted_from_n`, `integration_time_jitter`, `mass_change_r`; `dual_averaging`
  value `"fresh (mass changed)"` is now `"restarted (mass changed)"`; the Hamiltonian spec repr gains
  `integration_time_jitter=0.0`. Re-based pinned md5s: the 14 `damh_adaptive` stage-1/2 entries of
  `tests/unit/test_safety_kernel.py::REFERENCE_MD5` (B; the new carry_over summary key is stripped like
  the C1 keys) and the 10 `damh_s2`/`poly_s2` stage-1 entries of `tests/unit/test_error_model.py` (S).

  *J — `Hamiltonian(integration_time_jitter=j)`*, `j ∈ [0, 1)`, only with `integration_time`: per
  trajectory `u ~ Uniform(1 − j, 1 + j)` from the proposal's own generator (drawn before the momentum,
  only when `j > 0`), `num_steps = max(1, round(T u / ε))` (cap 1000); valid because the length is
  independent of the state. Auto fast: `AUTO_FAST_INTEGRATION_TIME = 3.0` (π/2 was tried and reverted the same day: with jitter 0.5, T = 3 was better on means and second moments in the 10-D test),
  `AUTO_FAST_INTEGRATION_TIME_JITTER = 0.5` (placeholders). 10-D, MH warm-up 2000 (not burn-in, so B
  does not enter), DAMH covariance-mass dimension-robust HMC, 6000 evaluations; j = 0 rows identical
  before/after:

  | T | j | ESS/eval x | ESS/eval x² | final ε / nominal L | max mean dev (ESS-SE) | cov rel. err |
  |---|---|---|---|---|---|---|
  | 3 | 0 | 1.60 ± 0.43 | 0.109 ± 0.016 | 1.03 / 3 | 1.91 ± 0.34 | 0.053 ± 0.017 |
  | 3 | 0.5 | 1.27 ± 0.19 | 0.318 ± 0.008 | 0.92 / 3 | 1.63 ± 0.14 | 0.043 ± 0.011 |
  | π/2 | 0 | 0.31 ± 0.08 | 0.585 ± 0.111 | 0.82 / 2 | 1.87 ± 0.57 | 0.037 ± 0.009 |
  | π/2 | 0.5 | 0.35 ± 0.07 | 0.493 ± 0.031 | 0.86 / 2 | 1.86 ± 0.23 | 0.035 ± 0.004 |

  With T = 3 the jitter removes most of the antithetic second-moment deficit (x² ×2.9) at −20 % on the
  mean. T = π/2 is not antithetic for x² but loses the "super-efficient" mean; with jitter it is the
  best of the four by min(x, x²) (0.35 vs 0.32), with one leapfrog step fewer per proposal. Posterior
  matches the closed form in all rows.

  *D — continuation rule* (`proposal_builder`): `r = ‖M_new − M_old‖₂ / ‖M_old‖₂` (resolved masses,
  identity = I); `r ≤ DA_CONTINUE_MASS_TOLERANCE = 0.2`: continue `(m, H̄, log ε̄, μ)`; else restart
  from `ε₀ = ε_carried · (det M_new / det M_old)^(1/(2d))` with `μ = log(DA_RESTART_FACTOR · ε₀)`,
  `DA_RESTART_FACTOR = 2` (placeholders). Derivation for `H = U + ½ pᵀM⁻¹p`: with `M⁻¹ = LLᵀ`, `q = Lz`
  the integrator is the identity-mass one on `U(Lz)`, whose accuracy depends on `ε ω`, `ω²` = eig(`M⁻¹ ∇²U`);
  keeping the geometric-mean `ε ω` fixed gives `ε ∝ det(M Σ)^(1/(2d))`, the target cancels in the ratio.
  For `M = Σ̂⁻¹` this is `(det Σ̂_old / det Σ̂_new)^(1/(2d))` — the INVERSE of the spec's tentative
  `sqrt(det Σ̂_new / det Σ̂_old)^(1/d)` (a larger covariance = lighter mass = faster velocities needs a
  smaller step). Auto fast, budget 20 000 (before = C1 code, after = C1b incl. J/B/S):

  | problem | version | dual averaging, chunks 1-4 (r) | out-of-box per chunk | non-finite | ESS/eval x / x² (10-D) or min-ESS sum (2-D) | cov rel. err / mean |
  |---|---|---|---|---|---|---|
  | 10-D local, 1 chain | before | fresh; restart ×3 (`μ = log 10ε`) | 0 | 0 | 0.67 ± 0.13 / 0.081 ± 0.007 | 0.055 ± 0.005 |
  | 10-D local, 1 chain | after | fresh; restarted (r = 2.2-2.4); continued (0.04-0.05); continued (0.02-0.04) | 0 | 0 | 0.22 ± 0.02 / 0.29 ± 0.06 | 0.025 ± 0.005 |
  | 2-D illustrative, 2 chains + collector, network | before | fresh; restart ×3 | 0.011-0.015; 0.047-0.064 (chunks 2-4) | 0 | 13-21 | mean (−0.06…−0.69, −0.30…−0.61) |
  | 2-D illustrative | after | fresh; restarted (0.82-0.85); continued (0.06-0.10); continued (0.01-0.07) | 0.001-0.002; 0.017-0.033 | 0 | 33-223 | mean (−0.11…0.26, −0.58…−0.76) |

  Restarts per run 3 → 1; the step trace of a continued chunk starts at its final value (10-D chunks
  3-4: first trace row 0.71-0.87 vs final 0.82-0.85; before: 0.30-0.91 vs 1.16-1.21 after a 10× push);
  out-of-box proposals per chunk 2.5× fewer in 2-D. Attribution runs on the 10-D case (after code):
  T = 3 instead of π/2 gives x / x² = 0.85 / 0.25; without the burn-in discount 0.25 / 0.31: the mean-
  ESS drop is the choice T = π/2 (antithetic mean lost), the x² gain (3.6×) is the jitter; B is within
  noise there. The 2-D posterior is bimodal (note 28), the min-ESS numbers are noisy.

  *B — burn-in discount* (`PosteriorStatistics.discount`, `continuation.discount_burn_in_statistics`,
  both runners): after a `burn_in=True` stage, after pooling and after the adaptive hand-over, `M2 ←
  M2 (n_eff−1)/(n−1)`, `n ← n_eff = max(100, 10 d)` (`BURN_IN_STATISTICS_PER_DIMENSION = 10`,
  placeholder); mean and covariance preserved; `is_excluded` stages untouched. The weight-discount
  variant was kept (it removes the measured skew; the second-half rebuild was not needed for that).
  10-D, MH burn-in 2000 then DAMH, start at 0 or "far" (3 prior sd in every coordinate):

  | stage 2 | start | version | statistics cov err (end) | RW shape err / cond(C^-½ Σ C^-½) | ESS/eval x | ESS/eval x² | cov rel. err |
  |---|---|---|---|---|---|---|---|
  | RW K=5, 8000 | 0 | before | 0.086 ± 0.027 | 0.046 ± 0.005 / 1.27 | 0.166 ± 0.004 | 0.279 ± 0.016 | 0.056 ± 0.021 |
  | RW K=5, 8000 | 0 | after | 0.038 ± 0.009 | 0.036 ± 0.008 / 1.20 | 0.181 ± 0.005 | 0.286 ± 0.016 | 0.040 ± 0.006 |
  | RW K=5, 8000 | far | before | 3.19 ± 0.65 | 0.927 ± 0.072 / 15.4 | 0.059 ± 0.019 | 0.118 ± 0.026 | 0.074 ± 0.024 |
  | RW K=5, 8000 | far | after | 0.194 ± 0.042 | 0.154 ± 0.021 / 1.89 | 0.149 ± 0.005 | 0.233 ± 0.018 | 0.050 ± 0.012 |
  | cov-mass HMC T=3, 6000 | 0 | before | 0.080 ± 0.017 | mass cond 4.4 ± 1.1 | 1.60 ± 0.43 | 0.109 ± 0.016 | 0.053 ± 0.017 |
  | cov-mass HMC T=3, 6000 | 0 | after | 0.051 ± 0.017 | mass cond 10.6 ± 0.8 | 1.34 ± 0.07 | 0.099 ± 0.055 | 0.053 ± 0.013 |
  | cov-mass HMC T=3, 6000 | far | before | 4.29 ± 0.89 | mass cond 259 ± 210 | 0.18 ± 0.07 | 0.25 ± 0.19 | 0.050 ± 0.030 |
  | cov-mass HMC T=3, 6000 | far | after | 0.286 ± 0.041 | mass cond 64 ± 17 | 0.057 ± 0.017 | 0.076 ± 0.055 | 0.15 ± 0.12 |

  The C1 regression of the RW shape (0.039 → 0.046 there) is gone (0.046 → 0.036) and a far start no
  longer poisons the statistics (RW: shape error 0.93 → 0.15, ESS/eval ×2.5). **Side effect, flagged**:
  the shrinkage `δ = min(1, 2d/n)` of the covariance estimate uses the discounted `n` (δ = 0.2 at
  `n = 10 d`), so the covariance MASS of the stage right after the burn-in is more isotropic (cond of the
  preconditioned target 4.4 → 10.6 from 0; mean-ESS −16 %, x² unchanged); from the far start both
  versions are poor (before: transient-dominated mass; after: shrunk mass, mean ESS lower, one seed's
  covariance off by 29 % (rel. Frobenius error 0.287) — "result poor"). In Auto fast (10-D) the discount is within noise.

  *S — S2 exact step on the shared statistics*: `build_exact_step_proposal(..., statistics=<shared>)`
  (both runners); the exact random walk installs its shape from the chain's statistics at construction
  when they hold ≥ 100 states (scale = trace of the carried/default covariance), then every 10th of its
  own steps; its scale adapts on its own exact acceptances; it never updates the statistics (the
  algorithm does once per outer iteration, exact steps included: pinned, `statistics.n` = number of
  outer iterations). S2 test setups (2-D linear Gaussian, 3 seeds):

  | setup | version | exact steps | exact-step acc. | ESS/eval x | ESS/eval x² | max mean dev (ESS-SE) | cov rel. err |
  |---|---|---|---|---|---|---|---|
  | HMC stage, p = 0.3 | before | 4712 | 0.232 ± 0.005 | 0.39 ± 0.03 | 0.093 ± 0.016 | 1.2 ± 0.6 | 0.026 ± 0.015 |
  | HMC stage, p = 0.3 | after | 4708 | 0.231 ± 0.008 | 0.36 ± 0.06 | 0.088 ± 0.023 | 1.7 ± 0.7 | 0.042 ± 0.030 |
  | RW stage + poly updater, p = 0.5 | before | 16 230 | 0.234 ± 0.001 | 0.166 ± 0.005 | 0.173 ± 0.011 | 0.9 ± 0.2 | 0.027 ± 0.017 |
  | RW stage + poly updater, p = 0.5 | after | 16 190 | 0.235 ± 0.002 | 0.166 ± 0.007 | 0.160 ± 0.013 | 1.7 ± 0.7 | 0.026 ± 0.006 |

  Neutral on these 2-D setups (mild correlation; the differences are within the seed spread); the
  posterior matches the closed form in all. Pinned by `tests/unit/test_c1b.py` (20 tests).

### Bug fixes

- **`SamplingRun.write_report(observations=<number>)` failed for a single observation (2026-10-14).** With
  `no_observations == 1` and the reference observation given as a plain float, the observation histogram indexed it
  (`'float' object is not subscriptable`, `post_processing/plots.py::hist_observations`) whenever raw snapshot data
  existed. `write_report` now converts `observations` to a 1-D float vector once at its entry
  (`core._as_observation_vector`) and raises `ValueError` up front if its length differs from `no_observations`;
  `hist_observations` also accepts a scalar for direct callers. Reports for vector observations are unchanged.
  Test: `tests/unit/test_problem.py::test_write_report_accepts_a_plain_number_for_a_single_observation`. (no effect on any configuration that completed successfully before)

- **Fixed (2026-10-11): the pooled posterior statistics counted the shared history once per sampler at every
  stage end (MPI runs with ≥ 2 samplers and ≥ 2 stages, or a continued run).** Every rank started a stage with
  the pooled statistics `S` of all previous stages, added its own states, and the stage end pooled the ranks'
  FULL objects `S ⊕ I_r`, so `S` entered `R` times — geometrically over stages (2 samplers, 3 MH stages of 600
  evaluations: pooled `n` 1200 / 3600 / 8400 instead of 1200 / 2400 / 3600). The mean and covariance were
  biased towards the older stages (effective weights `R^k`), and the inflated `n` lowered the shrinkage.
  Now `PosteriorStatistics.begin_stage()` remembers `S` and opens a per-stage increment fed by `update()`
  (no numerical subtraction); the `Allgather` buffer (still `1 + d + d²`) carries the increment, and
  `finish_stage(rows)` installs `S ⊕ I_1 ⊕ … ⊕ I_R` (one Chan combination). Byte-identical: local runs,
  one-sampler MPI runs and every single-stage MPI run (md5 of samples/raw_data/notes/subchain_stats/
  adaptive_stats/last_sample and of every pre-existing carry_over key; 18 local + 5 MPI-1 + plain cases).
  Differs: multi-stage MPI runs with ≥ 2 samplers from the second stage on (statistics, and through them the
  adaptive random walk's shape and `mass="covariance"`). Also in this entry, **author decision 1
  (2026-10-11)**: `PosteriorStatistics.n_seen` = states ever added, never reduced by the C1b burn-in discount;
  `shrunk_covariance()` uses `δ = min(1, 2d/max(n, n_seen))` (`n` everywhere else). Carried as the new
  carry-over key `posterior_statistics_n_seen` (a file without it: `n_seen = n`; `problem=` drops it with
  the statistics). This changes stages after a `burn_in=True` stage whose proposal reads the statistics
  (adaptive random walk, `mass="covariance"`, S2 exact step): there δ was 0.2 for `d ≥ 10`, now `2d/N_seen`.
  Evidence (scratchpad `c1c/` of the session):

  | 2 samplers, 3 adaptive-RW MH stages × 600 evals (`mpi_rw3`) | stage 0 | stage 1 | stage 2 |
  |---|---|---|---|
  | true outer iterations (both chains, cumulative) | 1200 | 2400 | 3600 |
  | pooled `n` before | 1200 | 3600 | 8400 |
  | pooled `n` after | 1200 | 2400 | 3600 |
  | stats covariance diag before | 0.128, 0.0992 | 0.121, 0.0980 | 0.1162, 0.0945 |
  | stats covariance diag after | 0.128, 0.0992 | 0.1175, 0.0974 | 0.1073, 0.0879 |

  2 samplers, MH + 2 DAMH-HMC stages (`mpi_hmc_hmc`): before 600 / 1953 / 4664, after = true 600 / 1353 / 2111.
  10-D anisotropic linear Gaussian, MH burn-in 2000 → DAMH `Hamiltonian(mass="covariance", T=3,
  dimension_robust)` 6000 evals, local, 3 seeds (`c1b_study.py burn hmc zero`):

  | version | δ of the stage-2 mass | cond of preconditioned target | ESS/eval x (mean ± sd) | per seed |
  |---|---|---|---|---|
  | before C1b (no discount, CHANGELOG C1b) | 0.01 | 4.35 ± 1.11 | 1.60 ± 0.43 | 1.96, 1.13, 1.72 |
  | C1b (δ from discounted n) | 0.2 | 10.6 ± 0.79 | 1.34 ± 0.07 | 1.28, 1.31, 1.41 |
  | now (δ from n_seen) | 0.01 | 4.35 ± 1.11 (same masses as before C1b to 1e-14) | 1.43 ± 0.44 | 1.52, 0.95, 1.81 |

  The mass is restored exactly; the ESS difference to "before C1b" comes from the other C1b changes
  (dual-averaging restart) and is within the seed spread. Tests: `tests/unit/test_posterior_statistics.py`
  (pooling over 2 and 3 fake ranks × 3 stages == numpy on the concatenation to 1e-12; one row / empty start
  bit-identical; `n_seen` through update, discount, pooling, shrinkage, carry-over and continuation),
  `tests/mpi/test_mpi_posterior.py::test_b13*` (pooled `n` == the notes' cumulative outer iterations).
  Re-based: 14 `damh_adaptive` entries of `tests/unit/test_safety_kernel.py::REFERENCE_MD5` (burn-in followed
  by adaptive RW: decision 1); the new key is stripped in the safety-kernel and S6 byte-identity tests.

- **Fixed (2026-10-09): a failing report in a single process aborted the interpreter.** `SamplingRun.write_report` wrapped the report body in the job-wide abort (`_run_role`) whenever the run was an MPI run, including a report re-generated from `SamplingRun.load(...)` in a plain Python session, so an invalid `selection.json` ended in `MPI_Abort`. With a world of size 1 (or a local run) the failure is now an ordinary exception.

- **Fixed (2026-10-08): `write_report` failed with `IndexError: positional indexers are out-of-bounds` whenever the
  first stage was not displayed** -- an explicit `stages_to_disp=[1, ...]`, or (since the selection mask) an
  `is_excluded` warm-up stage dropped by default, i.e. every automatic-mode run with the full report sections.
  `_write_report_rank0` replaced `samples.summary` by the displayed subset, while the acceptance-rate plot and the
  CpUS section index it by the original stage position. The in-memory summary now keeps every stage; `summary.csv`
  still holds the displayed stages only. Regression tests: `tests/unit/test_report_stage_subset.py`.

- **A per-rank `Configuration` mismatch now fails fast at start-up** (WS8, finding 2.9,
  2026-09-18, §2.3d). `SamplingFramework.run()` begins with
  `communication.check_configuration_consistency()`: rank 0 broadcasts the **requested** values
  (the literal constructor arguments, snapshotted in `Configuration.__post_init__`) of every
  field in `Configuration.POSTERIOR_AFFECTING_FIELDS` and every rank asserts equality, raising a
  `RuntimeError` on *all* ranks — naming the field and both values — which `_run_role` turns into
  a traceback plus a job-wide `MPI.Abort(1)`. A script that (by bug) builds a different
  `Configuration` per rank used to run anyway, each chain with its own configuration, or hang, or
  crash later somewhere unrelated. **No effect on any correctly-written script** (every rank runs
  the same constructor call, so the requested values always match): every `toy_examples/*.py` and
  the whole MPI suite are unchanged. The *requested* values are compared deliberately — the
  *effective* `use_surrogate_gradients` may legitimately differ per rank, since
  `_configure_surrogate_gradients()` disables it locally on a rank whose surrogate object cannot
  do gradients. MPI-layout fields (`no_samplers`, `rank_collector`, `rank_solvers_pool`,
  `sampler_ranks`) are not compared: they are derived from `MPI.COMM_WORLD.Get_size()` and are
  identical by construction. `run_local()` is single-process and makes no such call.
- **Collector checks a fresh evaluator's pickled size against `Configuration.max_buffer_size`
  before sending it** (finding 4.1/M10, 2026-09-18). Previously an oversized evaluator (e.g. a
  large NN) would truncate the fixed-size MPI receive buffer with no clear diagnostic; now
  `process_COLLECTOR._check_evaluator_pickle_size` raises `RuntimeError` immediately, right
  where the evaluator is built, naming the actual size and the configured limit — and warns
  above half the buffer, since it may grow past it as more snapshots are added. No effect on any
  run whose evaluator already fit (every maintained example). `Configuration.max_buffer_size`
  itself is not new — it already existed, documented, as the override for this (WS8) — only the
  fail-loud check on top of it is new.
- **Start-up crashes fixed** (WS0 §A1–§A3, commit `175059e`, findings 0.1–0.3): the dead
  `core.temptemptemp` method (undefined `Iterable`/`Any` annotations made `import surrDAMH` raise
  `NameError`), a stray unused `from torch.mtia import snapshot` in `process_SAMPLER.py`, and a
  `TestData.as_surrogate_test_data(prior=..., likelihood=...)` call with two keyword arguments the
  method does not take, are all removed/fixed.
- **Collector polls the stop signal unconditionally** (WS0 §B1, finding 2.1). Previously polled
  only once a surrogate had been trained, so a run whose samplers never needed one hung forever at
  shutdown; now polled every loop iteration regardless.
- **`stages_will_use_surrogate` list-`or` bug fixed** (WS0 §C1, finding 2.6). `any(following_DAMH)
  or any(following_onlySurr) or any(following_hamiltonian)` replaces a bare `or` on two lists
  (which ignored the second list whenever the first was non-empty), so later
  `use_only_surrogate`/Hamiltonian-MH stages no longer close the collector channel early.
- **Solver-failure bookkeeping fixed** (WS0 §B4/C2/C3, finding 1.7): a non-finite initial
  log-likelihood now raises `RuntimeError` instead of causing every proposal to reject silently
  for the rest of the stage; a negative `solver_tag` gates on the *proposed* sample (not the
  current one) for whether it enters surrogate training data; the log-prior term returned on
  failure is finite instead of `-inf`.
- **kd-tree zero-distance guard** (WS0 §C4, `nearest_kdtree.py`, finding S3): an exact hit on a
  training point no longer produces `NaN` via `1/0` in the inverse-distance average; it now
  returns that point's stored observation directly.
- **`Gaussian_process.py` vector-`std` covariance bug fixed** (WS0 §C5, finding 1.6): the block is
  now built as `np.outer(std, std) * corr` instead of scaling columns only, which was asymmetric
  for non-constant `std`. The module is not called inside the package and is kept as a low-priority
  optional component.
- **float64 parameter transport and continuation** (WS0 §C9/C10): sampler→solver parameter
  messages and `last_sample/*.npz` are now explicitly `float64` end-to-end (previously float32
  could sneak into the continuation path). Files written before this fix still load correctly
  (cast on read); a previously-continued run would have resumed from a float32-rounded state, now
  resumes from the exact last sample.
- **`SurrogateAsSolver.get_observations` shape fixed** (WS3, finding A20, §2.7). Now always
  returns `(no_observations,)` regardless of whether the wrapped evaluator returns `(1, q)` or a
  flattened `(q,)`, fixing a `ValueError` crash of `Stage.use_only_surrogate=True` for
  `no_observations > 1`.
- **Raw-observation MPI path removed** (WS8, author decision 5, findings 2.3/2.4/M6, §2.17),
  together with `Configuration.pickled_observations` (see Removed). Fixes finding 2.4/M6: a failed
  solve (`solver_returns_tag=True`, `tag=-1`) used to raise `mpi4py.MPI.Exception: Invalid tag` on
  the deleted raw path because the solver's status code was used directly as the MPI tag; such a
  proposal is now simply rejected and never reaches the collector.
- **`NeuralNetworkUpdaterMinibatches.initial_training` no longer persists synthetic rows**
  (WS6, finding 3.4, §2.16). `get_training_data_arrays()`, `get_initial_snapshots()` and the
  checkpoint now report only real snapshots.
- **`find_best_fits` raises instead of silently degrading** (WS9b, finding P10, §2.15) when the
  `raw_data` file lacks the columns its ranking mode needs (`RunFormatError` naming them, instead
  of NaN scores producing an arbitrary but complete-looking table); it now also ranks through
  `rank_best_fit_candidates`, the unit-tested function, instead of a second untested
  implementation (finding P3).
- **`except BaseException`/bare `except` removed from the post-processing code** (WS9b, finding
  P6, §2.15): 9 sites replaced with explicit validation and a real message — notably
  `decompress_samples=False` now raises `AttributeError` naming the constructor argument instead
  of printing "CHAIN ... NOT AVAILABLE" per chain, and a mis-shaped `burn_in` raises once instead
  of silently skipping chains.
- **`Autocorrelation` takes its chain count from the stages actually analysed** (WS9b, finding P7),
  not from global stage index 0.
- **Figures are closed on the failure path** (WS9b, finding P8): every plotting method closes its
  figure if it raises after creating it.
- **`chains_to_disp` honoured everywhere it is accepted** (WS9b, finding P5): previously honoured
  only by the best-fit and observation-histogram sections; now every relevant report section
  restricts to the requested chains (`plot_acceptance_rates` and the section-1 summary table are
  documented exceptions — their counters are whole-stage totals).
- **Sampler no longer abandons its last `TAG_UPDATE` `Isend`** (2026-09-18, finding 2.10,
  `10_manual_review_notes.md` §2.19a). `CommEvaluator_sampler.get_evaluator_and_terminate()` set
  `request_Isend = None` without `Wait()`ing it, i.e. discarded a still-pending non-blocking send
  (undefined by the MPI standard; tolerated by this container's MPICH). It now waits, exactly as
  `request_evaluator()` and the collector-side `terminate()` already did. The wire protocol is
  unchanged — same messages, same tags, same order — so nothing changed in
  `CommEvaluator_collector` or anywhere else, and no run's results differ. Pinned by the first
  2-rank protocol test, `tests/mpi/test_mpi_protocol.py` (item I12).
- **`no_unique_samples` now updated** (WS0 §C11, finding P2; refined by the A30 change above to
  `np.count_nonzero(weights)` so a weight-0 boundary row is not counted): previously stayed `0`
  unconditionally, collapsing the default histogram bin count to the 10-bin floor whenever
  `bins=None`.
- **A non-finite proposal can no longer poison the surrogate and kill a run** (2026-09-20).
  Observed causal chain, from the efficiency study's own outputs: the dual averaging's first
  iterations overshoot the step size upward by design (`μ = log 10ε₀`), so with
  `hamiltonian_num_steps=100` an early leapfrog trajectory ran off to non-finite positions → the
  proposal was rejected but the EXACT solver was still called there (`exp` overflowed in the GRF
  example) and the outlier was sent to the collector as a training snapshot → the neural-network
  updater trained on it and its weights became NaN (`surrogate_quality.csv`: `rmse = nan` from
  then on) → the collector published the NaN evaluator, the sampler installed it, every leapfrog
  gradient became NaN, every acceptance probability 0 at any step size, so the dual averaging
  drove ε to its clamp and the following stage inherited the dead model and produced 0 accepted
  samples. Four independent guards, none of which changes the adaptation rule (no constant, no
  target, no `μ`) or any arithmetic on a finite path:
  (a) `AlgorithmBase._reject_nonfinite_proposal` (`modules/algorithms.py`) rejects a proposal
  whose parameters are not all finite **without** calling the observation provider — it gets the
  new, reserved `solver_tag = -2` (`SOLVER_TAG_NONFINITE_PROPOSAL`), hence `log_likelihood = -inf`
  through the existing `< 0` path and no snapshot; `Algorithm_MH.run` skips
  `get_log_acceptance_probability` for it (a Hamiltonian's momentum term would be NaN) and feeds
  `adapt()` `-inf`, i.e. α = 0 — the correct divergent-transition signal;
  `Algorithm_DAMH._propose_new_sample_using_subchain` does the same per sub-chain step without
  calling the surrogate (the sub-chain end state, and therefore the outer proposal, stays finite).
  Counted in the new `counter_nonfinite_proposals`, printed in the stage-end summary line of
  `process_SAMPLER`/`runner_local`, one `RuntimeWarning` per stage (the `notes` CSV columns are
  unchanged).
  (b) `Hamiltonian._leapfrog` / `HamiltonianInfinite._leapfrog` (`modules/proposals.py`) abandon a
  trajectory as soon as `q` or `p` stops being finite and return the non-finite `q`, saving the
  remaining surrogate-gradient calls. The checks only read the state, so a finite trajectory is
  **bit-identical** to the previous loop body (pinned by
  `test_leapfrog_is_bit_identical_to_the_pre_divergence_check_version`, which runs the old body
  verbatim for 5 seeds × {mass I, full SPD mass}).
  (c) `process_COLLECTOR` evaluates a freshly retrained evaluator on the newest snapshot batch
  (`_evaluator_is_finite_on`) and, if any prediction is non-finite, keeps the previous evaluator
  and the previous monitoring evaluator, printing `collector: retrained surrogate is non-finite
  on the newest N snapshots; keeping the previous evaluator` once per streak (plus one line when
  it recovers). `NeuralNetworkUpdaterMinibatches.add_data` drops snapshots with non-finite
  parameters/observations (counted in `no_nonfinite_snapshots_skipped`, one warning; the check is
  after the float32 cast, so a value that overflows float32 is caught too), and `train()`
  snapshots the weights and restores them if the pass produces non-finite ones, so a later update
  with more data can still succeed instead of the model being dead forever.
  (d) `AlgorithmBase._refresh_surrogate_evaluator_if_needed` probes an arriving evaluator at the
  current chain state (one call, plus one `vjp` when the proposal needs gradients) and refuses a
  non-finite one, keeping the previous evaluator while still posting the next request
  (`counter_nonfinite_evaluators`, one warning per stage). The very first evaluator is still
  installed — nothing better exists — and the return value still means "an evaluator was really
  replaced".
  **Effect on runs that were healthy:** none. No fixed-step run of the study ever produced a
  non-finite proposal, and on a finite trajectory / finite surrogate every code path above is a
  read-only check. The one behavioural difference in an *already broken* run is that a non-finite
  proposal no longer consumes a full-model solve and no longer appears in the surrogate training
  data; a DAMH sub-chain step still draws exactly one uniform whether or not its proposal was
  finite, so the random stream is unchanged.

- **`surrDAMH.runner_local.run_local(...)` works again** (2026-09-21). `runner_local` was never
  imported by `surrDAMH/__init__.py`, so the call spelled out in `docs/running.md` and in the two
  `template_experiment*` docstrings raised `AttributeError: module 'surrDAMH' has no attribute
  'runner_local'`; only `from surrDAMH.runner_local import run_local` worked. Both spellings work
  now, plus the new short `surrDAMH.run_local`.

- **`calculate_artificial_observations` accepts any array-like and tolerates a scalar-returning
  solver** (2026-09-22). Its `parameters` argument is typed `npt.ArrayLike` and converted with
  `np.asarray` inside, so the natural `parameters=[-2, 2]` of `toy_examples/own_solver.py`
  type-checks; the return is built with `np.asarray(observations).ravel()` instead of
  `observations.ravel()`, which raised `AttributeError` for a solver returning a plain Python
  float. Both spellings return the same `(no_observations,)` array as before.

- **Updater constructors visible to Pylance again** (2026-09-22). `surrogates.reuse.register_updater`
  was annotated `type[Updater] -> type[Updater]`, so every decorated updater was typed as the bare
  `Updater`: `NeuralNetworkUpdater(hidden_layer_sizes=...)` was reported as `No parameter named
  "hidden_layer_sizes"` and completion offered only the base-class arguments. The decorator is now
  generic (`TypeVar` bound to `type[Updater]`); pyright reports `type[NeuralNetworkUpdater]` etc.

- **A proposal appended in place of a `Stage` is refused at construction** (2026-09-22).
  `SamplingFramework(...)` and `run_local(...)` now call `stages.check_stage_list`, which raises
  `TypeError: list_of_stages[i] is a RandomWalk, not a surrDAMH.Stage; a proposal is not a stage,
  wrap it: Stage(proposal=..., max_evaluations=...)` (and a hint for a single `Stage` passed without a
  list). Before, the mistake surfaced inside `run()` as `AttributeError: 'RandomWalk' object has no
  attribute 'algorithm'`. The mistake was invited by an import leak: `surrDAMH.stages.RandomWalk`
  resolved; `stages.py` now imports the specs under a private alias, so `surrDAMH.stages` offers
  `Stage`, `stage_name`, `check_stage_list` and `POSTERIOR_AFFECTING_FIELDS` only.

- **Deadlock when a surrogate stage starts before `min_snapshots_initial` snapshots exist** (2026-09-22,
  found by the GRF scheme study's smoke run). The collector answered a sampler's initial evaluator
  request only after the first training, and trained only once `min_snapshots_initial` was reached; a
  DAMH (or Hamiltonian-proposal) stage that began earlier therefore blocked every sampler in
  `get_evaluator()` while the collector waited for snapshots that only those samplers could produce.
  New protocol message `TAG_EVALUATOR_NEEDED` (sampler -> collector, `CommEvaluator_sampler.
  announce_waiting()`, sent by the algorithm right before its blocking wait for the FIRST evaluator;
  `EvaluatorProvider.announce_waiting`, a no-op for the local providers). Once every active sampler
  has announced and no new snapshot arrived, the collector trains the initial surrogate on the
  snapshots it has and prints why; with zero snapshots it raises a `RuntimeError` naming the fix.
  `LocalSurrogateManager.get_evaluator` (run_local) does the same instead of raising. Behaviour of
  runs that never hit the condition is unchanged (the check is entered only while no evaluator exists
  and `min_snapshots_initial` is not yet met). Regression test
  `tests/mpi/test_mpi_hangs.py::test_i3b_damh_stage_before_min_snapshots_initial_trains_instead_of_deadlocking`.

### Additions

- **`subchain_stats` column `component` for `Mixture` DAMH stages (2026-10-13; `library_notes/32_*` D-32-3;
  `surrDAMH/modules/algorithms.py` `COMPONENT_SUBCHAIN_STATS_COLUMNS`/`_component_column`; test
  `tests/unit/test_subchain_component_column.py`; `docs/outputs.md`).** The `subchain_stats` file of a
  DAMH stage whose proposal is a `Mixture` gains a column `component`, appended LAST (after `kernel`): on
  a delayed-acceptance row (`kernel = 0`) the index of the component drawn for that outer iteration (it
  proposed the whole sub-chain), on an S2 exact-step row (`kernel = 1`) `-1` (no component proposed).
  **Output format (additive):** the column exists ONLY for a `Mixture` stage; the files of `Block` and
  plain-proposal stages are unchanged (readers select columns by name). `Mixture` itself dates from C2
  (2026-10-12). No seed, default or MPI message changed. Tests as reported by the implementing agent on
  2026-10-13 (not re-run for this entry): see the flattened-likelihood entry below.

- **Flattened-likelihood `Hamiltonian` component (2026-10-13; `library_notes/32_*` §1;
  `surrDAMH/proposals.py`, `modules/proposals.py`, `modules/proposal_builder.py`, `stages.py`; tests
  `tests/unit/test_flattened_hamiltonian.py`, `tests/mpi/test_mpi_flattened.py`; `docs/stages.md`
  "Flattened likelihood", `docs/outputs.md`).** `Hamiltonian` gained `likelihood_weight` `tau` in
  `[0, 1]` (default `1.0` = off): every momentum kick of the trajectory uses `tau * grad log L` instead
  of `grad log L`; the prior part of the dynamics, the position updates, the half kicks, the finiteness
  checks and the momentum flip are unchanged. The map stays volume-preserving and reversible and the
  acceptance test uses the exact energy, so the posterior is preserved (note 32 §1.1, the argument of
  note 30 §2). `tau = 0` is the prior dynamics: with `integrator="dimension_robust"`, `mass=1`,
  `integration_time=pi/2` and no jitter it is an independent prior draw (choose the step so that
  `round(T/step) * step = pi/2`). **Validation (`ValueError` at spec construction):** `tau` outside
  `[0, 1]`; `tau < 1` together with `adaptive=True` after resolution (a flattened-likelihood component
  does not adapt its step: pin `step_size` or set `adaptive=False`). Combinable with the C3 fields.
  Intended use: a pinned `Mixture` component (`adaptive=False`, takes the carried step). **Output
  format (additive):** (i) `run_manifest.json` `proposal_resolution[i]` gains `likelihood_weight` only
  when `tau < 1` (inside a mixture: `component<k>.likelihood_weight`); (ii) the `proposal` entry of every
  `stages` item in `run_manifest.json` and the spec `repr` (hence the component lines of
  `Stage.describe()`) now show `likelihood_weight=1.0` for EVERY `Hamiltonian` spec, flattened or not,
  because the manifest writes every dataclass field (as for the C3 fields) - this also changes manifests
  of existing configurations, by one additive key; (iii) `Stage.describe()` adds one note line for a
  flattened `Hamiltonian` (prefixed `component k:` inside a `Block`/`Mixture`). Not changed: `tau = 1`
  (default) runs are byte-identical to before (at `tau = 1` `_likelihood_force` returns the gradient array
  itself, no multiplication); the pinned C2/C3 byte-identity tests (`tests/unit/test_mixture.py`,
  `tests/unit/test_tempered_hamiltonian.py`) are unchanged and pass; no seed, default or MPI message
  changed. Tests as reported by the implementing agent on 2026-10-13 (not re-run for this entry): unit +
  validation 805 passed, 1 skipped; the MPI files `test_mpi_flattened`, `test_mpi_tempered`,
  `test_mpi_mixture`, `test_mpi_s2` 6 passed.

- **C3: tempered-trajectory `Hamiltonian` component (2026-10-12; `library_notes/30_*` §1-§4;
  `surrDAMH/proposals.py`, `modules/proposals.py`, `modules/proposal_builder.py`, `modules/algorithms.py`,
  `stages.py`, `process_SAMPLER.py`, `runner_local.py`; tests `tests/unit/test_tempered_hamiltonian.py`,
  `tests/mpi/test_mpi_tempered.py`; `docs/stages.md` "Tempered trajectories", `docs/outputs.md`).**
  `Hamiltonian` gained `tempered_fraction` `f` in `[0, 0.5]` (default `0`) and `tempered_beta` `beta` in
  `(0, 1]` (default `1`): Neal's tempered trajectory (Neal 2011 §5.5.2), a way to cross barriers that a
  random walk or an untempered trajectory cannot. A trajectory of `L` steps (with
  `integration_time_jitter`, that trajectory's own `L`) tempers `n_t = min(floor(f L), L // 2)` steps at
  each end: the momentum is multiplied by `sqrt(alpha)` before each of the first `n_t` position/flow
  updates and divided by `sqrt(alpha)` after each of the last `n_t`, with `alpha = beta^(-1/n_t)`, so the
  kinetic temperature reached in the middle is `1/beta` whatever `L` is (hence independent of the adapted
  step). The placement is palindromic, so the map (with the momentum flip) is a volume-preserving
  involution and the acceptance test is unchanged; both integrators (`leapfrog`, `dimension_robust`).
  **Validation (`ValueError` at spec construction):** values out of range; `tempered_fraction > 0` iff
  `tempered_beta < 1` (the message names both); `tempered_fraction > 0` together with `adaptive=True` after
  resolution (a tempered component does not adapt its step, note 30 §4: pin `step_size` or set
  `adaptive=False`). A trajectory with `n_t = 0` (`L < 2` or `f L < 1`) is not tempered and is counted per
  stage as `tempered_skipped`. Recommended use: a pinned component of a `Mixture`, e.g.
  `Mixture([Hamiltonian(integrator="dimension_robust", mass="covariance", integration_time=3,
  integration_time_jitter=0.5), Hamiltonian(integrator="dimension_robust", mass="covariance",
  integration_time=9, integration_time_jitter=0.5, adaptive=False, tempered_fraction=0.3,
  tempered_beta=0.1)], probabilities=[0.8, 0.2])` (the pinned component takes the carried step of the
  previous adaptive Hamiltonian stage/chunk); where the adapted step gives `L = 1-3` (study 29 §2.3) a
  component tempers nothing unless its own `T` is long enough. `PCN(beta=1.0)` (an independent prior draw,
  acceptance = likelihood ratio) was already supported; its docstring now says `(0, 1]`. **Output format
  (additive):** (i) the `notes` file gains a column `tempered_skipped`, appended LAST and ONLY for a stage
  whose proposal (or a `Block`/`Mixture` component) tempers (readers select columns by name; every other
  `notes` file is unchanged); (ii) the stage-end print gains ` - tempered skipped: N` under the same
  condition; (iii) `run_manifest.json` `proposal_resolution[i]` gains `tempered_fraction` and
  `tempered_beta` only when tempering is on (inside a mixture: `component<k>.tempered_fraction`/
  `component<k>.tempered_beta`); (iv) the `proposal` entry of every `stages` item in `run_manifest.json`
  now lists `tempered_fraction: 0.0` and `tempered_beta: 1.0` for EVERY `Hamiltonian` spec, tempered or
  not, because the manifest writes every dataclass field (as it did for the C1 `integration_time*` fields)
  — this also changes manifests of existing configurations, by two additive keys. `Stage.describe()` adds
  one note line for a tempered `Hamiltonian`, also for a tempered `Block`/`Mixture` component (prefixed
  `component k:`). Not changed: `f = 0` (default) runs are byte-identical to before C3
  (`tests/unit/test_tempered_hamiltonian.py::test_untempered_hamiltonian_runs_are_byte_identical_to_before_c3`,
  pinned md5s, environment-specific; `::test_explicit_zero_fraction_equals_the_default`); no seed, default
  or MPI message changed. Tests as reported by the implementing agent on 2026-10-12 (not re-run for this
  entry): unit + validation 788 passed, 1 skipped; full MPI suite 60 passed.

- **C2: `Mixture` proposal and adaptive `Block` (2026-10-12; `library_notes/23_*` §4.1/§4.3,
  `25_*` §9 D8; `surrDAMH/proposals.py`, `modules/proposals.py`, `modules/proposal_builder.py`,
  `stages.py`; tests `tests/unit/test_mixture.py`, `tests/mpi/test_mpi_mixture.py`; `docs/stages.md`
  "Mixtures and adaptive blocks", `docs/outputs.md`; example `toy_examples/mixture_proposal_example.py`).**
  `Mixture(proposals, probabilities=None)`: two or more `RandomWalk`/`PCN`/`Hamiltonian` components (no
  `Block`/`Mixture` inside, refused in both directions); every outer iteration draws one component
  (uniform, or with the given positive weights, normalised at build) and it proposes all parameters (in a
  DAMH stage it drives that iteration's whole sub-chain). Each component is built like a bare stage
  proposal (full dimension, consumes the carried steps, shares the posterior statistics, same guards).
  `Block` gained `probabilities` (per-iteration group weights, `None` = uniform as before) and an
  `adaptive` property (True iff any sub-proposal adapts). `adapt()` reaches only the component that
  proposed; adaptive components are pooled at stage end and a `Mixture` hands their tuned values on under
  the bare-proposal keys (`scale`, `beta`, `step_size`, `dual_averaging`, `hamiltonian_mass`); a `Block`
  hands nothing over. **V1 restriction:** at most one ADAPTIVE component per spec type in a `Mixture`
  (`ValueError` suggesting to pin a step; fixed duplicates are allowed). Decision D8's positional carry-over
  keying was deliberately **not implemented**. **Output format (additive, only for mixture/adaptive-block
  stages):** `adaptive_stats/<stage>/rank%04d.csv` holds the trace of the FIRST adaptive component only
  (one `RuntimeWarning` per process when several adapt); `run_manifest.json` `proposal_resolution[i]`
  gains `adaptive_stats_component` (and, for a `Mixture`, the components' build records as
  `component<k>.<key>`); the carry-over summary keys (manifest and `carry_over/<stage>.npz`) become
  `component<k>.<key>`; the HTML report's carry-over block does not display them yet (no crash).
  `Stage.describe()` prints a `Block(...)`/`Mixture(...)` note and one line per component (console only,
  also for existing fixed Blocks). Not changed: every configuration without a `Mixture` or an adaptive
  `Block`; a `Block` of fixed sub-proposals is byte-identical to before
  (`tests/unit/test_mixture.py::test_fixed_block_runs_are_byte_identical_to_before_c2`).
  Run check (not a test): `toy_examples/mixture_proposal_example.py`, output
  `toy_examples/out_mixture_example/`.

- **`Configuration.seed_offset` (2026-10-12, study-29 prerequisite P1).** A non-negative integer added
  to every seed of the run on top of the lineage-generation shift (`modules/seeds.py`:
  `run_seed_offset`, `conf_seed_offset`): every per-stage stream of every chain (proposal,
  acceptance, exact-step choice, exact random walk), the initial-sample seed and the LHS seed.
  Recorded as `seeds.seed_offset` in `run_manifest.json` (the `seeds.formula` string now mentions
  it) and in `describe()` (`*`). Validated (`int >= 0`). Default `0`: byte-identical `samples`,
  `raw_data`, `subchain_stats`, `notes`, `adaptive_stats`, `last_sample` and `carry_over` in the
  18 + 379 + 197 reference files of scratchpad `p29/` (plain scripts, 18 local cases, 5 MPI cases);
  `seed_offset=1_000_000` is byte-identical to `lineage_generation=1` (unit tests, local, prior and
  LHS starts; manual MPI check, 197/197 files). Surrogate-updater seeds are not shifted. Tests:
  `tests/unit/test_seed_offset_exact_step_scale.py`.
- **`Stage.exact_step_scale` (2026-10-12, study-29 prerequisite P2).** `None` (default) = the S2
  exact step's adaptive random walk (byte-identical, same reference files); a positive number `s` =
  a NON-adaptive `GaussRandomWalk` with covariance `s * 2.38^2/d * Sigma_hat`, `Sigma_hat` = the
  shared posterior statistics' covariance at stage start (identity with a printed note while fewer
  than 100 states). `build_exact_step_proposal(..., scale=None)`; `proposal_resolution` in the
  manifest gets `exact_step_scale`, `exact_step_covariance_source`, `exact_step_covariance_trace`
  for such a stage; `describe()` marks the field `*` and notes it; reset to `None` with a warning on
  an MH stage. Acceptance-affecting when set. Tests: same file.

- **Adaptive pCN: `PCN_adaptive`** (2026-09-20, `16` §5 item 6). `Stage(proposal_type="pCN",
  adaptive=True)` is now supported — `__post_init__` no longer forces `adaptive=False` for pCN —
  and adapts `beta` by a Robbins–Monro recursion on its logit,
  `logit β ← logit β + n^(−0.7)(α − 0.234)`, which keeps `β` in `(0, 1)` by construction. Target
  0.234, not 0.5: the prototype measured it recovering 0.77–0.99 of the ESS of the best fixed
  `β` and beating target 0.5 by 1.03–2.14× on every problem, with the optimal `β` spanning
  0.1–0.8 (`proto/RESULTS_PROTO.md` §7).
- **Adaptive Hamiltonian step size: `Hamiltonian_adaptive` / `HamiltonianInfinite_adaptive`**
  (2026-09-20, `16` §5 item 4). Dual averaging of the leapfrog step size (Hoffman & Gelman 2014,
  Algorithm 5: `γ=0.05, t₀=10, κ=0.75, μ=log 10ε₀`) with target acceptance **0.8** (0.65
  overshot by ~2× into the `εL ≈ 2π` resonance in the prototype, `RESULTS_PROTO.md` §5.2). In a
  DAMH stage it updates once per **sub-chain** step on the sub-chain's own acceptance against the
  surrogate — that is the Hamiltonian the leapfrog integrates, and the outer DAMH rate is flat
  over a 20× step range (`15` §2.7). The mass (`proposal_sd_or_cov`) and `hamiltonian_num_steps`
  are **not** adapted. Measured on a d=10, κ=10 Gaussian with exact gradients: acceptance 0.80,
  `ε̄ = 0.280` from `ε₀ = 0.01` and `0.277` from `ε₀ = 1.0` (a factor 1.01 apart), both below the
  leapfrog stability limit `2√λ_min(C) = 0.447`
  (`tests/unit/test_proposals.py::test_dual_averaging_step_size_hits_its_target_and_stays_stable`).
  Caveat measured while adding the MPI case B12: in an **MH** stage the only acceptance
  statistic is the exact-model one, which also pays for the *surrogate gradient field's* error,
  so an under-trained surrogate pushes the adapted step size down (0.1 → 0.024 there) and the
  stage mixes slowly while it warms up. The chain is not biased — its covariance approaches the
  closed form along the chain and the frozen stage that follows it matches within 0.6 SE — but
  such a stage needs a budget, or a following stage with `hamiltonian_step_size=None`.
- **`Stage.pcn_beta=None` / `Stage.hamiltonian_step_size=None` carry-over** (2026-09-20). Both
  fields are now tri-state and default to `None`, meaning "the value carried from the last
  adaptive stage of that proposal type, else 0.5 / 0.1" — the former dataclass defaults, so the
  **effective value of every existing configuration is unchanged**. This makes the
  preliminary-stage → frozen-stage workflow work for pCN and the Hamiltonian family the way
  `proposal_sd_or_cov=None` already did for the random walk.
- **`adaptive_stats/<stage>/rank%04d.csv`** (2026-09-20, `16` §5 item 2). One row per completed
  adaptation period of an `adaptive=True` stage, written at stage end through the existing
  monitor; the columns depend on the proposal class and every row carries `rank_world` (see
  `docs/outputs.md`). `read_run` exposes it as `RunData.adaptive_stats[stage]` (`None` for a
  stage whose proposal did not adapt, so runs written before this file existed read back
  unchanged) and `post_processing.Samples` as `self.adaptive_stats[stage]`.
- **Pooled cross-rank hand-over interface on the adaptive proposals** (2026-09-20):
  `adapted_state() -> 1-D float64`, `set_pooled_state(states)` and `carry_over() -> dict`. The
  random walk pools `(n, mean, M2)` by the Chan–Golub–LeVeque parallel formula plus the mean
  `log σ`, pCN the mean `logit β`, the Hamiltonian family the mean `log ε̄`. Because the state
  has a fixed length, the buffer-shape mismatch of finding 2.5 is now structurally impossible.
- **`carry_over/<stage>.npz` and `Proposal.adapted_summary()`** (2026-09-21). The adapted
  proposal state handed from one adaptive stage to the next (`Proposal.carry_over()`) used to
  reach the user only as a stdout print; it is now also persisted to
  `sampling_output/carry_over/<stage>.npz` (one file per adaptive stage, written once by
  sampler rank 0 / `run_local`, see `docs/outputs.md`), alongside a new diagnostic-only
  `Proposal.adapted_summary()` (`base_cov`/`log_sigma`/`n_pooled`/`mean` for the random walk,
  `beta` for pCN, `step_size` for the Hamiltonian family — never consumed by `build_proposal`).
  `RunData.carry_over[stage]`/`Samples.carry_over[stage]` expose the pair (`None` for a
  non-adaptive stage or a run written before this file existed), and the "5. Proposal
  Adaptation" section of the HTML report gains a "Carried over to the next stage" subsection
  (the pooled covariance/sd/correlation tables for the random walk, a scalar for pCN/Hamiltonian).
  **This is a new output file and report subsection only — no sampler code path or RNG draw
  changed, so the sample stream, acceptance rate and surrogate accuracy of every configuration
  are byte-for-byte unaffected.**
- **`weighted_ess` column in `surrogate_quality_test.csv`** (WS6, finding S10, 2026-09-18).
  Effective sample size of the fixed test set's posterior weights (Kish's formula,
  `1/sum(w_i^2)` on the normalized weights); equals `n_test` when no weights are supplied.
  Diagnoses a concentrated posterior with few informative test points, where `weighted_rmse`
  can otherwise be essentially the error at one point (`10_manual_review_notes.md` §5e's
  `TestData` discussion). Purely additive: existing columns unchanged, appended as the 9th
  column; readers select columns by name so nothing else needed updating besides one
  hardcoded header list in `tests/mpi/test_mpi_surrogate.py`.
- **`Configuration.torch_threads`** (default `1`, author decision 2026-09-18, WS4;
  `library_notes/10_manual_review_notes.md` §7b "Torch threads" reopened and resolved).
  `SamplingFramework.run()`/`run_local()` now call `torch.set_num_threads(torch_threads)`
  deterministically on every rank that has torch loaded, instead of leaving it to whatever
  the launcher happens to do; `None` opts back out (torch's own default: all visible cores
  per process). Previously measured (2026-09-17) that this container's MPICH `mpiexec`
  already constrains every rank to 1 thread, so `torch_threads=1` is a no-op there; it
  matters under a launcher that does not itself throttle OpenMP (Slurm `srun`, OpenMPI, a
  bare `python`/`python -m mpi4py` process), where it prevents several ranks from each
  claiming every core for the NN surrogate. Recorded in `run_manifest.json` as both
  `configuration.torch_threads` (requested) and `environment.torch_num_threads` (effective,
  already existed). Performance-only, does not affect the posterior/acceptance
  rate/surrogate accuracy. See [`running.md`](docs/running.md#torch-cpu-threads).

- **`SamplingFramework.write_report(field_statistics_max_samples=...)`** (2026-09-18, finding F1 of
  `library_notes/14_grf_validation_2026-09-17.md`). The posterior field statistics call the solver
  once per decompressed posterior state; the existing `n_max_samples` sub-sampling of
  `compute_posterior_field_statistics` was unreachable from `write_report`, so a fast solver with
  long chains spent far longer reporting than sampling (measured 6·10⁶ states → 37 min report for
  6.5 min sampling). Default `None` keeps the exact all-states behaviour; `sampling_diffusion_grf.py`
  now passes 20 000.

- **Pool-mode report gaps are now announced, not silent** (2026-09-18, finding 2.7,
  `06_findings_consolidated.md`/`10_manual_review_notes.md` §8). When `use_solvers_pool=True`,
  the reporting rank has no live `Solver` object, so `write_report()`'s parameter-names,
  posterior-field-statistics and best-fit-solver-visualization sections had nothing to show and
  said nothing about why. `core.py::_write_report_rank0` now prints a `RuntimeWarning:`-prefixed
  stdout message, and `html_report_extended`'s new `pool_mode_note` argument fills each of those
  sections in the HTML with an explanatory note instead of a blank/generic one; a new "Best-fit
  Solver Visualization" section (previously nonexistent — the images were only ever saved as
  loose PNG files) shows the same note in pool mode, or the embedded image(s) otherwise. Purely
  additive: no existing section's content changed, verified with `mpiexec -n 4
  toy_examples/template_experiment.py` (pool mode: all three sections marked "Not available")
  and a non-pool run with a solver exposing `field_builder`/`coords`/`measurement_points`/
  `visualize_solution` (all three sections show real content, no note).
- **`stages_to_disp` accepts stage names, not just indices** (2026-09-18, WS9 bullet 5
  remainder). `Samples._resolve_stages` (and therefore `write_report`,
  `html_report_extended`, and every statistics/plot method built on it) now accepts a mix of
  positional indices and stage names as produced by `stages.stage_name()` (e.g.
  `"alg0001_DAMH-SMU"`); an unknown name raises `ValueError` naming the run's actual stages.
  Purely additive: int-only/`None` callers take the exact old code path.
- **`MPI_TAG_UB` start-up diagnostic** (2026-09-18, finding 2.11,
  `10_manual_review_notes.md` §2.19b). `SamplingFramework.run()` calls
  `modules.communication.check_tag_upper_bound()` once on rank 0 before role dispatch: MPI tags
  grow by one per full-model evaluation and are never reused, while the MPI standard only
  guarantees `MPI_TAG_UB >= 32767`. It warns (`RuntimeWarning`) when the projected tag count
  (`sum(min(max_evaluations, max_samples))` over the stages, plus the snapshot tag offset) does
  not fit, or when a stage has no evaluation/sample bound at all and `MPI_TAG_UB < 2^20`.
  **Diagnostic only**: it swallows every exception, never raises, never changes control flow, and
  is silent on every maintained example (MPICH 4.3.1 here reports `MPI_TAG_UB = 2^30 − 1`). The
  tags themselves still grow without bound — nothing about the wire protocol changed.
- **Idle throttling of the solvers-pool and collector service loops** (2026-09-18, findings
  4.2/M11, `10_manual_review_notes.md` §2.19c). Both loops were pure busy-waits on
  `Iprobe`/`Get_status` and pegged a core at 100% with nothing to do. They now drive a
  `ServiceLoopThrottle`: an iteration that did no real work counts as idle, the first 100
  consecutive idle iterations spin exactly as before, each further one sleeps 1 ms. **Timing
  only** — the same MPI calls in the same order, nothing sent, received or reordered, no effect on
  the posterior or on any output file. Measured (4 ranks, MPICH, 32-core container, 3 runs each):
  `toy_examples/typical_example.py` user CPU 4m17–4m27 → 3m36–3m41 at unchanged wall time;
  `toy_examples/minimal_example.py` 9.4–11.6 s → 8.6–9.4 s wall. The spin budget is what keeps a
  fast solver neutral: sleeping 1 ms on *every* idle iteration instead cost that example ~45 %
  wall time, because its analytic solver turns a request around in far less than 1 ms.
- **Run manifest** (WS4): every run (MPI or `run_local`) now writes
  `sampling_output/run_manifest.json` recording the full configuration/stages, seeds,
  surrogate/solver identity, package + git versions, MPI layout, and any unverified-option
  warnings. Pure addition; failures to write/finalize it are printed as warnings, never raised.
- **`run_local()`** (WS2): standalone, MPI-free single-chain runner
  (`surrDAMH.runner_local.run_local`), reproducing rank 0 of an equivalent MPI run with the same
  seeds; supports in-process DAMH-SMU (`updater=`), a fixed surrogate (`evaluator=`), and
  continuation.
- **`build_proposal()`** (WS2/WS7): per-stage proposal construction extracted into
  `surrDAMH/modules/proposal_builder.py`, shared by `process_SAMPLER` and `run_local`, carrying
  the pCN/Hamiltonian/adaptive fail-fast guards.
- **`SurrogateRestart(state_dir, mode="state"|"data"|"none")`** and a new optional
  `SamplingFramework(..., surrogate_restart=...)` argument (WS5). Applied on the collector rank
  only; replaces the two `tools.py` restart helpers (kept, marked superseded).
- **`Configuration.describe()` / `Stage.describe()`** (WS5), printed once on rank 0 at the start
  of `SamplingFramework.run()` and by `run_local()`: every dataclass field with its effective
  value, `*` on the posterior-affecting ones, plus a requested-vs-effective line for
  `use_surrogate_gradients`. New stdout at the start of every run.
- **`SolverSpec` stores an absolute `solver_module_path`** (WS5, finding M18): resolved on the
  launching rank so spawned solver children receive an absolute path through the broadcast spec,
  fixing a latent portability issue (a relative path already happened to work under this
  container's MPICH/Hydra launcher, but is not guaranteed to under every launcher).
  `Configuration.paths_to_append` is kept but documented as ineffective for spawned children.
- **`toy_examples/template_experiment.py`** (WS5): canonical ~100-line annotated starting point
  for a new experiment (problem → prior/likelihood → solver → surrogate → stages → run → report).
- **`read_run(output_dir) -> RunData`** (WS9a, `surrDAMH.read_run`,
  `surrDAMH/modules/run_data.py`): the single reader of the on-disk layout; see the output format
  v2 breaking-change entry for what it enforces.
- **`surrDAMH/post_processing.py` split into the package `surrDAMH/post_processing/`** (WS9b):
  `loading.py` (`Samples`/`StageSamples`), `statistics.py` (moments, ESS, R-hat, autocorrelation,
  best fits), `plots.py`, `html_report.py`. Import paths are unchanged — everything is re-exported
  from `surrDAMH.post_processing`; the split was verified to be a byte-identical, verbatim move
  (existing tests and a `summary.csv`/moments/ESS/R-hat dump of one toy run are unchanged before
  and after). The report also gained a "Run Configuration" section (from the manifest) and now
  aborts loudly (`MPI.COMM_WORLD.Abort(1)`) instead of hanging the trailing barrier if report
  generation fails.
- **Posterior-correctness tests for a multi-stage chain whose neural-network surrogate is built
  on the fly** (2026-09-17, `10_manual_review_notes.md` §5d). No test previously asserted the
  posterior of an `MH → DAMH-SMU → DAMH` chain driven by an NN surrogate (B2/B3 used a
  polynomial / kd-tree surrogate; B8 trained an MLP but used it only for Hamiltonian proposal
  gradients inside an MH stage). Added: `tests/mpi/test_mpi_posterior.py::test_b9_…` (real
  `mpiexec -n 4`, collector-trained MLP, both DAMH stages vs the closed form, retraining count
  reconstructed from `surrogate_quality.csv`, manifest/`weighting`/`output_normalization`
  asserted) and `tests/validation/test_gaussian_toy.py::test_V3c_…` (in-process DAMH-SMU,
  K ∈ {1, 5, 20}) / `::test_V3d_…` (in-process three-stage chain, also asserting that the frozen
  stage keeps the surrogate the SMU stage ended with). Tests only — no library change.
- **Collapsible `report_extended.html`, "Sampling Stages" and "Proposal Adaptation" sections**
  (2026-09-21, user request). Every top-level section and every per-stage block of the report
  is now a `<details>` element that is collapsed when the file is opened ("Expand all" /
  "Collapse all" buttons under the title; table-of-contents links unfold the section they point
  into via a few lines of inline JavaScript — the report is still one self-contained file). New
  section "Sampling Stages": the effective settings of every stage of the run, one column per
  stage and one row per `Stage` field (`*` marks the posterior-/acceptance-rate-affecting fields,
  as `Stage.describe()` does), from the new `html_report_extended(stages=...)` argument or, by
  default, from the `stages` list in `run_manifest.json`. New section "5. Proposal Adaptation"
  (the surrogate-quality and observation-histogram sections became 6 and 7): for every displayed
  stage whose proposal adapted, `Samples.plot_adaptation()` (new, `post_processing/plots.py`)
  plots the per-period `mean_acceptance_probability` the proposal was fed, with the target rate
  as a reference line, and every adapted parameter in `adaptive_stats/<stage>/rank%04d.csv`
  (`log_sigma`, `trace_C_over_d`, `shrinkage_delta` / `beta` / `log_step_size`,
  `log_step_size_bar`), one line per chain, plus a table of the last logged period of every
  chain; produced in the lightweight report too (one row per period, so cheap). The best-fit
  solver visualization block that `core._insert_best_fit_visualization_note` appends is a
  collapsed `<details>` as well. Report only — no sampling code changed.

- **`Solver.set_parameters` has a working default; `no_parameters`/`no_observations` are base-class
  attributes** (2026-09-21, intuitive-usage pass driven by `toy_examples/template_experiment_intuitive.py`).
  The default stores `np.asarray(parameters)` in `self.parameters` and raises `ValueError` when
  `no_parameters` is set and the size differs; `Solver.__init__(solver_id, output_dir, no_parameters=None,
  no_observations=None)` stores its arguments. Every solver in `surrDAMH/`, `toy_examples/` and `tests/`
  overrides `set_parameters` or already stored a `(no_parameters,)` array, so nothing else changes
  (unit + validation suite: 480 passed, 1 skipped). A solver that relied on the default storing a
  wrongly-sized array now fails at `set_parameters` instead of inside the model.

- **Package namespaces made self-describing** (2026-09-21, author-requested audit of the `__init__`
  files). New top-level names, all previously reachable only through a submodule: `surrDAMH.Stage`,
  `surrDAMH.SolverSpec`, `surrDAMH.run_local`, `surrDAMH.runner_local`. `surrDAMH.surrogates` now
  exports the `Updater`/`Evaluator` base classes, so writing a surrogate no longer needs the
  internal-looking `surrDAMH.surrogates.parent` path (`docs/writing_a_surrogate.md` updated).
  `surrDAMH`, `surrDAMH.distributions`, `surrDAMH.stages` and `surrDAMH.surrogates` got module
  docstrings that say which name to reach for and show a runnable snippet; `__all__` of the top-level
  package is grouped by purpose. `toy_examples/sampling_TSX.py` now builds its prior components from
  the exported `LognormalComponent`/`UniformComponent` instead of the `independent_components`
  submodule aliases. Purely additive: no name was removed or rebound.

- **`Solver.no_parameters`/`no_observations` are typed `int` for type checkers** (2026-09-22,
  author decision). They are now *declared* on `Solver` (`no_parameters: int`) instead of being
  class attributes defaulting to `None`, so Pylance/pyright knows a solver instance has them as
  plain `int`: `Configuration(no_parameters=my_solver.no_parameters, ...)` and numpy shape
  arguments built from them type-check (measured on `toy_examples/template_experiment_intuitive.py`:
  6 pyright errors before, 0 after). Runtime effect: a solver that sets neither attribute now raises
  `AttributeError` at the attribute the first time anything reads it, instead of silently reporting
  `None` that surfaced much later inside `Configuration` or the MPI layer; `set_parameters` reads it
  through `getattr(..., None)` and simply skips its size check for such a solver. No solver in the
  library, `toy_examples/` or `tests/` relied on the `None` state, and the library itself never reads
  these attributes (it uses `conf.no_parameters`), so nothing else changes.

- **Per-solver output directory is created on first use, not eagerly** (2026-09-22, author
  decision). `Solver.output_dir` is now a property: the constructor stores the path, and the
  directory is created the first time the attribute is read. `SamplingFramework` (per-rank
  solver) and `process_SOLVER` (per spawned child) now pass the `solver_output/rank<k>/` path
  without creating it. A solver that never writes leaves no empty directory behind; a solver that
  writes gets its directory exactly as before (verified: pool run with a writing solver -> the
  child that served requests got `solver_output/rank0/calls.txt`, the idle child created nothing;
  pool run with a non-writing solver -> no `solver_output/` at all). Assigning `self.output_dir`
  directly still works (property setter). Nothing in the library reads the attribute back, and no
  test or maintained solver relied on the eager creation. `docs/writing_a_solver.md` updated.

- **`target_rate` on a non-adaptive proposal prints a warning at construction** (2026-09-22).
  `RandomWalk(scale=0.5, adaptive=False, target_rate=0.5)` (and the `PCN`/`Hamiltonian` analogues)
  now print `Warning: RandomWalk(target_rate=0.5, adaptive=False): target_rate is only used when the
  proposal adapts; drop it or set adaptive=True` when built, where the offending line is still on
  screen; previously only `describe()` noted it at start-up. Behaviour unchanged: the value was and is
  ignored by the builder for a fixed proposal.

- **Updater constructors fully annotated and regrouped; class docstrings user-facing** (2026-09-22,
  author-requested: several parameters showed as `Unknown` in Pylance). `NeuralNetworkUpdater` had
  `no_parameters`, `no_observations`, `hidden_layer_sizes` unannotated; all 22 of its parameters, and
  every parameter of the other three updaters, now carry types (`RBFInterpolationUpdater.kernel` is a
  `Literal` of the eight scipy kernels instead of `str`). Keyword parameters after
  `no_parameters, no_observations` are reordered into groups -- model, training, data, runtime
  (`# --- group ---` comments in the signatures) -- same names, same defaults; every caller in the
  repository passes them by keyword (grep-verified), so nothing changes behaviour. Each class docstring
  is now a short user-facing lead plus an `Args:` section in the new order (what Pylance shows on the
  constructor call); the maintainer paragraphs moved to a comment block above the class. Guard test:
  `tests/unit/test_surrogates.py::test_every_updater_constructor_parameter_is_annotated`.

- **Solver hand-over rules checked at `SamplingFramework(...)` construction** (2026-09-22).
  `use_solvers_pool=True` without `solver_spec` raises `ValueError` explaining that the pool spawns
  processes that import the solver from a file, so a ready `solver_instance` cannot be used there
  (before: an `assert` on the pool rank inside `run()`); neither solver given, or both given, raise
  too. Manual check: `mpiexec -n 2` driver with the pool and a `solver_instance` -> the message on
  both ranks, exit 0, no hang.

- **Configuration fields carry attribute docstrings with their defaults** (2026-09-22, author: hovering a
  parameter did not show its default). Every field's trailing `#` comment became a docstring string
  ending in `Default: <value>.`, so Pylance can show description and default on the parameter itself
  (unverified in the IDE from here; the class docstring `Args:` stays). Corrected `Args` entries after
  the author's questions: `use_solvers_pool` (True needs `solver_spec`, a `solver_instance` needs
  False), `no_solvers`/`solver_maxprocs` (ignored without the pool), `min_snapshots_to_update` (0 =
  retrain every loop, useful for the network only), `save_snapshots_to_file` (parameters AND
  observations, tag, likelihood, status), `paths_to_append` (how to set `PYTHONPATH` for spawned
  processes), `debug` (collector rank only), `torch_threads` (network updater only).

- **Docstring clarifications from the template review** (2026-09-22): `SamplingFramework` `prior`
  (internal space: identical to the parameter space for `Normal`, standard normal for
  `PriorIndependentComponents`), `solver_spec`/`solver_instance` (rule stated inline),
  `surrogate_updater` (when required), `surrogate_test_data` (fixed, never extended);
  `PriorIndependentComponents` and the component classes got user-facing docstrings (the unit-test
  reference moved to a maintainer comment; `UnivariateComponent` described as "the prior of one
  parameter"). `UnivariateComponent.transform` raises `NotImplementedError` instead of `pass`, which
  pyright read as "returns None" and flagged every override.

- **`Configuration` and `Stage` parameter hovers: one source, with the default** (2026-09-22, three
  rounds with the author). Pylance shows, for a dataclass constructor parameter, the entry it extracts
  from the class docstring's `Args:` section -- and, if present, the field's attribute docstring as
  well, which duplicated the text; the parameter hover does not show the signature default. Final
  state: no attribute docstrings; every `Args:` entry of `Configuration` and `Stage` ends with
  `Default: <value>.` (or `Required.`; `unlimited` for the `sys.maxsize`/`inf` stopping rules),
  generated from the dataclass fields so they cannot drift. The earlier attribute-docstring attempt
  (same day) is gone.

- **Start-up note for snapshots streamed to a surrogate nobody will use** (2026-09-22, GRF scheme
  study, `library_notes/18_*` §5). A frozen-surrogate stage with `send_snapshots_to_collector=True` and
  no later stage that retrains makes the collector refit on every `min_snapshots_to_update` new
  snapshots for nothing; for RBF/KD-tree that stalled the samplers and made the frozen stage 2-3x
  slower than the retraining one (831 vs 397 s; 386 s once the flag was off). `stages.wasted_snapshot_notes`
  prints one `[note]` per such stage from `SamplingFramework.run()` (rank 0, collector runs only);
  nothing is dropped automatically because a later run may reuse the snapshots via
  `SurrogateRestart(mode="data")`. Test: `tests/unit/test_config_stages.py::test_wasted_snapshot_notes_flag_only_a_frozen_tail`.

- **Continuation: `SamplingRun.continue_sampling` / `continue_sampling_local`, lineage report,
  selection mask** (2026-10-08, API specs `continuation`/`lineage_report`;
  `docs/running.md#continuing-a-run`, `docs/outputs.md`). A finished run (or one rebuilt by
  `SamplingRun.load(output_dir, problem)`) can now be continued under a fresh
  `Configuration`/stage list without hand-wiring `initial_sample_type="continued"`,
  `continued_from_dir`, seeds, carry-over and surrogate restart separately:
  ```python
  run_b = run_a.continue_sampling(conf_b, stages_b, chains="continue",   # "continue"|"prior"|"lhs"
                                  problem=None, surrogate_updater=None, surrogate_evaluator=None,
                                  surrogate_initial_training_data=None, surrogate_test_data=None,
                                  surrogate_restart=None)                 # MPI, collective
  run_b = run_a.continue_sampling_local(conf_b, stages_b, chains="continue", problem=None,
                                        surrogate_updater=None, surrogate_evaluator=None)
  ```
  By default this reuses `run_a`'s last chain states (`chains="continue"`), merges the proposal
  carry-over of every adaptive stage of the lineage into `stages_b`'s first stage (so
  `RandomWalk(scale=None)` etc. start tuned), restores a saved surrogate checkpoint
  (`surrogates.reuse.SurrogateReused`) and held-out test data (`TestData.reuse`), and numbers
  `stages_b`'s directories/seeds after every earlier stage of the lineage. New `Configuration`
  fields, set by `continue_sampling` only (leave at the default otherwise):
  `stage_index_offset: int = 0`, `no_stages_lineage: int | None = None`,
  `lineage_generation: int = 0` — all three added to `POSTERIOR_AFFECTING_FIELDS`. Seeds gain a
  `1_000_000 * lineage_generation` shift (`modules.seeds.GENERATION_SEED_STRIDE`) so a
  continuation never reuses a stream of an earlier run of its lineage (requires
  `10 * no_stages_lineage * no_samplers < 1_000_000` for the whole lineage). `run_manifest.json`
  gains a `"lineage"` entry (`continued_from`, `generation`, `stage_index_offset`,
  `no_stages_lineage`, `same_problem`, `same_problem_lineage`, `chains`,
  `initial_sample_is_carried_over`, `dirs`); `modules.run_data.read_lineage(output_dir)` follows it
  recursively into one concatenated `RunData`, and `post_processing.Samples`/`SamplingRun.write_report`
  gain `include_previous` (`None`/`True`/`False`) to control whether the whole lineage or just this
  run is loaded. A new `post_processing_output/selection.json` (`post_processing/selection.py`,
  format 1) records a per-(stage, chain) include/burn-in mask plus `is_excluded`/`recommended`
  (reserved, `null` for now); `write_report(selection=...)` creates it with the defaults on first
  use, reads and applies it on every later call, and the report gains a final "Selection and
  re-run" section with the re-run snippet (`SamplingRun.load(output_dir).write_report()`).

  Behaviour changes flagged by the implementers, verified not to affect any run that does not use
  the new API:
  - (a) **`Stage.is_excluded` stages are now dropped from the default report** — the default
    selection sets `include=0` for every chain of an `is_excluded` (burn-in) stage, so it no
    longer appears in `report_extended.html`/`summary.csv` unless `selection=False` or an edited
    `selection.json` puts it back. Before the selection mask existed, `is_excluded` only affected
    the `+1`/`+0` multiplicity rule (`docs/outputs.md`) and such a stage still appeared in every
    report.
  - (b) **`LocalSurrogateManager` no longer re-adds snapshots an updater already restored**
    (`modules/algorithm_interfaces_local.py`): it now checks `updater.training_data_loaded` before
    calling `add_data` on the preloaded snapshots, the same rule `process_COLLECTOR.py` already
    applied on the MPI side. Previously a `run_sampling_local`/`continue_sampling_local` continuation
    that restored a network's/polynomial's training data via `load_state`/`load_training_data`
    counted and trained on those snapshots twice.
  - (c) **A run made by `continue_sampling(_local)` saves its surrogate state at the end**
    (`core._save_surrogate_state_of`, `SurrogateRestart(state_dir=...).save(updater)` on the
    collector rank / in process), so a chain of continuations keeps working without each script
    saving it by hand. A **plain** `run_sampling(_local)` call still does **not** save it
    automatically — call `SurrogateRestart(state_dir=...).save(updater)` yourself after it returns
    if a later continuation should be able to restore it (see `toy_examples/continue_sampling_example.py`).
  - (d) **`PolynomialSklearnUpdater` gained state persistence** (`save_state`/`load_state`,
    `supports_state_persistence() -> True`, alongside the pre-existing
    `save_training_data`/`load_training_data`): the stored snapshots and constructor
    hyperparameters are checkpointed; the ridge fit itself is deterministic and redone from the
    snapshots on the first `get_evaluator()` after a restore. `RBFInterpolationUpdater` and
    `KDTreeUpdater` still do not implement it, by design (see `docs/running.md#continuing-a-run`).
  - (e) **New log lines on a continued run**: `"Sampler at rank N carry-over from the previous run
    <key>: [...]"` (`process_SAMPLER.py`) / `"Local sampler - carry-over from the previous run
    <key>: [...]"` (`runner_local.py`) print the merged carry-over values a continuation's first
    stage starts from, once per key, when `initial_carry_over` is non-empty (never on a plain
    run).
  - (f) **A plain run's manifest now also carries the three new `Configuration` fields**
    (`stage_index_offset=0`, `no_stages_lineage=null`, `lineage_generation=0` in
    `configuration` — `_configuration_dict` dumps every dataclass field) and an absent `"lineage"`
    key; `core._lineage_of`/`modules.run_data._previous_link` treat both a plain run and the old
    manual `initial_sample_type="continued"` style as generation 0 of their own lineage when read
    back, so existing output directories remain readable.

  Not posterior-affecting for any existing script: `stage_index_offset=0`,
  `no_stages_lineage=None`, `lineage_generation=0` are every `Configuration`'s default, so a plain
  `run_sampling`/`run_sampling_local` call reproduces exactly the seed formula and manifest shape
  of before (plus the three new, inert configuration keys and an absent `"lineage"` key).

- **Automatic mode: `Problem.run_sampling_auto` / `run_sampling_local_auto`** (2026-10-08, API
  - Status: preliminary. Exact per installed surrogate; the on-the-fly network retraining is not yet covered by the
    ergodicity argument of note 18 (no Polyak averaging, no prior truncation, no exact safety kernel yet); all layout and
    surrogate numbers are placeholders pending the validation plan of note 25 §6.
  spec `auto`; `library_notes/25_robust_by_default_roadmap_2026-10-08.md` §4/§5;
  `docs/running.md#automatic-mode-2026-10-08`, `docs/outputs.md`, `docs/configuration.md`). A
  stage layout and a surrogate chosen from a budget instead of an explicit stage list, new module
  `surrDAMH/auto.py` (pure planner `plan_auto` -> `AutoPlan`):
  ```python
  run = problem.run_sampling_auto(conf, budget=80_000, mode="robust")        # MPI, collective call
  run = problem.run_sampling_auto(conf, time_limit=3600.0, mode="fast")      # wall-clock instead
  run = problem.run_sampling_local_auto(conf, budget=20_000, mode="robust")  # one chain, one process
  ```
  Exactly one of `budget=` (TOTAL exact evaluations over every chain, including the held-out set)
  / `time_limit=` (seconds); `mode` is `"robust"` (adaptive `RandomWalk()`) or `"fast"`
  (`Hamiltonian(num_steps=30, integrator="dimension_robust", mass=1.0)` on the surrogate's
  gradients) — **both exact**, `"fast"` only possibly slower-mixing on a hard problem.
  `surrogate_updater=`/`surrogate_test_data=` are expert overrides of the defaults below.

  **Layout rule** (every number a placeholder pending note 25 §6's validation, stated as such in
  the docstrings): `d = problem.no_parameters`, `C` = number of chains, `B` = per-chain budget
  after subtracting the held-out set. `B < 50·d`, or `conf.use_collector=False` (no collector to
  train a surrogate): one adaptive-MH stage with the whole budget. Otherwise: an excluded
  adaptive-MH warm-up of `n0 = clip(20·d, 0.05·B, 0.25·B)` evaluations per chain, then 4
  DAMH-SMU chunks (`subchain_length=1`, the mode's proposal) sharing `B - n0` as evenly as
  possible (remainder to the last chunk) — no frozen stage. A `time_limit` splits the same
  0.15/0.85 (÷4) shares over wall time instead, with a note that the `20·d` warm-up floor cannot
  then be enforced.

  **Surrogate defaults**: `NeuralNetworkUpdater(hidden_layer_sizes=(64, 64), activation="silu",
  solver="adamw", seed=0)` unless overridden; a held-out set
  (`min(256, max(2·d, 0.02·budget))`/`min(256, max(2·d, 64))` for a time budget, subtracted from
  `budget`) only when the plan uses a surrogate and `problem.solver_instance` is an in-process
  `Solver` — generated **on the collector rank** for an MPI run (so no sampler spends an
  evaluation on it) or before the run for `run_sampling_local_auto` (which does not monitor
  surrogate quality with it, only saves it). `conf.use_surrogate_gradients` is set to
  `mode == "fast"`; `min_snapshots_initial`/`min_snapshots_to_update` get size-dependent values
  (`max(1, min(10·d, n0·C // 2))` / `max(20, 2·d)`) **only if still at `Configuration`'s own
  default** — a value the user set explicitly is always kept, with a note in the plan.
  `conf.initial_sample_type` is set to `"lhs"` only if left at the default `"prior"`.

  The resolved plan is printed once at start-up (`AutoPlan.describe()`) and the explicit stage
  list it produced (`run.stages`) is what runs — copyable into a plain `run_sampling` call. It is
  recorded in `run_manifest.json`'s new `"auto"` entry and as `run.auto` (`mode`, `budget`,
  `time_limit`, `no_samplers`, `per_chain_budget`, `test_data_size`, `warm_up`, `chunks`,
  `stage_names`, `proposal`, `surrogate`, `conf_settings`, `notes`).

  **Deviations from the API spec, decided by the implementer** (code is the source of truth):
  `mode="fast"` raises `ValueError` if `conf.use_surrogate_gradients=False` or
  `conf.transform_before_surrogate=True` was set explicitly (the spec did not call these out as
  separate raises); a budget plan with a single MH stage (too small a budget, or no collector)
  uses the **whole** budget with **no** held-out set, instead of still subtracting one; a time
  budget's `min_snapshots_initial` is `max(1, 10·d)` (no `n0·C` is known without an evaluation
  count); `run_sampling_local_auto` never monitors surrogate quality with the held-out set (only
  saves it, since the local runner has no quality-monitoring loop); `budget` smaller than the
  number of chains and a non-positive/non-finite `time_limit` raise `ValueError` explicitly. Not
  implemented: `continue_sampling_auto` (an automatic continuation) — `conf`'s lineage fields
  must stay at their defaults or `plan_auto` raises.

  New example: `toy_examples/auto_mode_example.py`. Tests: `tests/unit/test_auto.py` (planner
  rules, local runs against the closed-form posterior), `tests/mpi/test_mpi_auto.py` (robust/fast
  MPI runs, `use_collector=False`).

- **`Stage.burn_in`; every chain visible in the report; the posterior aggregated over the
  selection mask** (2026-10-09, API spec `burn_in_report`; `docs/stages.md`, `docs/outputs.md`).
  New `Stage` field `burn_in: bool = False` (`surrDAMH/stages.py`, added to
  `POSTERIOR_AFFECTING_FIELDS` so it is marked `*` in `Stage.describe()`): a warm-up flag — the
  chain **continues** into the next stage exactly like a normal stage, but the stage's chains get
  `include=0` by default in the report's `selection.json`, so its samples are left out of the
  posterior unless the user edits the file. This contrasts with the pre-existing `is_excluded`
  (discarded exploration: the *next* stage restarts from the sample this stage started at).
  `burn_in=True` and `is_excluded=True` together raise `ValueError` from `Stage.__post_init__`.
  `Problem.run_sampling_auto`/`run_sampling_local_auto`'s warm-up stage now sets `burn_in=True`
  instead of `is_excluded=True` (see Bug fixes above) and `toy_examples/hamiltonian_neural_network_example.py`
  was updated to match.

  **Post-processing/report** (`post_processing/selection.py`, `loading.py`, `html_report.py`,
  `core.py`): `Samples(..., selection=...)` no longer drops chains at load time — every chain of
  every stage is always loaded (`Samples.chain_indices` holds every chain); the validated mask is
  kept on the object (`selection_mask`, `selection_burn_in`) and `Samples.posterior_view() ->
  Samples` returns the filtered subset as a shallow, array-sharing view (per-chain `burn_in` rows
  sliced off, nothing re-read or re-decompressed). `selection.json`'s per-stage entries gain a
  `burn_in_stage` key (the stage's `Stage.burn_in` flag, next to the existing `is_excluded`);
  `default_selection` sets `include=0` for a stage with either flag `True`.
  `html_report_extended` takes a `posterior:` argument (`self.posterior_view()` when omitted): the
  per-stage sections (traces, acceptance/pre-rejection, adaptation, carry-over, counters,
  histograms) are built from `self` (every chain), each chain's label getting a badge — `"in
  posterior"` or `"not in posterior (burn-in stage | excluded stage | excluded by user)"`
  (`Samples.chain_status`) — and the per-stage summary rows and `summary.csv` gain a
  `chains_in_posterior` column (`"<n>/<all>"`, e.g. `"2/4"`, appended last); the pooled sections
  (overall mean/covariance, combined histograms, ESS, R-hat, CpUS, best fits, combined observation
  histograms, posterior field statistics) are built from `posterior` instead, with their section
  header stating how many stages/chain-stage pairs contributed ("Aggregated over the chains with
  include = 1 in selection.json: ..."). A clearly visible user note, with the selection file's
  absolute path and the exact re-run command, appears at the top of the pooled "Overall analysis"
  section and again in the final "Selection and re-run" section, which also gains the
  `burn_in_stage`/`is_excluded` columns in its stage x chain table. `write_report()` now returns
  the full, unfiltered `Samples` (every chain) rather than a pre-filtered one; call
  `.posterior_view()` on it for the subset the report's pooled sections used.
  `core._write_report_rank0` builds `summary.csv` from the full `Samples` (so its `autocorr`/`CpUS`
  columns are over all chains of a stage) and passes `samples.posterior_view()` as
  `html_report_extended`'s `posterior=` (whose own CpUS figures, sections 4.4/4.8, are recomputed
  over the posterior chains only — the two numbers for a stage with any excluded chain can differ).

  **Behaviour change, retracting part of the 2026-10-08 continuation/selection-mask entry above**:
  a stage with `Stage.is_excluded=True` (or, now, `Stage.burn_in=True`) is **no longer dropped from
  the report/`summary.csv` entirely** by the default selection — only its chains are left out of
  the *posterior* (pooled sections); the stage and its per-chain traces/counters stay visible, each
  chain marked with its badge. The 2026-10-08 entry's point (a) ("`is_excluded` stages are now
  dropped from the default report") no longer holds.

  Tests: `tests/unit/test_burn_in_report.py` (the flag, default selection, the report showing
  every chain with correct badges and a pooled mean computed independently from the included
  chains' CSVs only, `summary.csv`'s `chains_in_posterior` column, the user note's exact text and
  placement, `posterior_view()` regression against `selection=False`, the Auto warm-up fix).

- **`selection.json` layout simplified** (2026-10-09, author decision, follow-up to the two entries
  above; `post_processing/selection.py`, `loading.py`, `html_report.py`, `docs/outputs.md`). The
  file written/read since 2026-10-08/09 is barely a day old, so it is replaced outright instead of
  kept compatible. New layout:
  ```json
  {"posterior": {"alg0000_MH-adaptive": [0, 0], "alg0001_DAMH-SMU": [1, 1]},
   "runs": ["<dir0>", "<dir1>"],
   "note": "..."}
  ```
  Removed keys: `format`, `lineage` (now `runs`, written only for a lineage of more than one run),
  `stages` (now `posterior`, a `{stage_name: [0/1, ...]}` object keyed directly by stage name
  instead of a list of `{"name": ..., "output_dir": ..., "include": [...], "burn_in": [...]}`
  objects), `is_excluded`/`burn_in_stage` (dropped entirely — the report's "Selection and re-run"
  table now reads the two `Stage` flags straight from the run manifest(s) instead), `recommended`
  (dropped; it was unused, reserved for not-yet-implemented verdict diagnostics). The per-chain
  row-drop entry, formerly `burn_in` inside each stage object, is now the top-level, OPTIONAL
  `drop_first_rows: {stage_name: [rows per chain]}` — never written by default, validated the same
  way (0 to the chain's compressed row count) when present. The file is written with a small custom
  dump (`selection._dump`): 1-space indent, but each stage's chain list stays on one line (plain
  `json.dump(..., indent=1)` would otherwise put one list element per line). A file still in the
  pre-2026-10-09 layout (has a `"stages"` key) is rejected by `validate_selection`/`read_selection`
  with `ValueError` naming the file and asking for it to be deleted so it is re-created with the
  current defaults — **no converter, by the same no-compat rule as the rest of this cycle**.
  `post_processing/loading.py`'s internal mask shape (`Samples.selection_mask`:
  `{stage_name: {"include": [...], "burn_in": [...]}}`) is UNCHANGED, so no consumer outside
  `selection.py`/`html_report.py`'s own prose needed a code change beyond wording ("include = 1 in
  selection.json" -> "posterior=1 in selection.json"; the user note no longer mentions dropped rows
  unless `drop_first_rows` is actually present in the file). Not posterior-affecting:
  purely a file-format and prose change, the selection semantics (which chains/rows feed the
  posterior) are identical. Tests updated in place (no behaviour removed):
  `tests/unit/test_lineage_report.py`, `tests/unit/test_burn_in_report.py`,
  `tests/mpi/test_mpi_continuation.py::test_lineage_report_with_selection`.

- **Bounded prior: `Problem(..., prior_bound=8.0)`** (2026-10-09, S0, note 25 D4; `docs/concepts.md`).
  The target is the prior truncated to `|u_i - m_i| <= R s_i` in the internal space (`problem.prior_bound`,
  `problem.prior_box = (lower, upper)`, `None` with `prior_bound=None`); proposals outside are rejected
  unevaluated: `SOLVER_TAG_OUT_OF_BOUNDS = -3` in `raw_data` (NaN observation blocks; MH `rejected`,
  DAMH sub-chain `prerejected`), `AlgorithmBase.counter_out_of_bounds`, `StageResult.counter_out_of_bounds`,
  a `notes` column `out_of_bounds` (appended last; the post-processing summary sums by column name now)
  and `- out-of-bounds proposals: N` in the stage-end line. Initial states outside the box raise
  `RuntimeError` at start-up; the manifest's `"problem"` entry records `prior_bound`/`prior_box`, and
  `continue_sampling` with a different bound raises unless `problem=` is passed. Tests:
  `tests/unit/test_prior_bound.py`.

- **Exact safety kernel and audit of pre-rejected proposals** (2026-10-09, S2, notes 24 §6 and 22 §4
  P1; `docs/stages.md`, `docs/outputs.md`). Two DAMH `Stage` fields in `[0, 1]`, default 0, both marked
  `*`: `exact_step_probability` (an outer iteration is an exact MH step with its own adaptive
  `RandomWalk()` -- carried `scale`, seed `seed0 + 5`, not pooled/carried, stage proposal not adapted
  on it -- scored with the installed surrogate first, accepted with the exact likelihood; the chain
  follows a mixture of two exact kernels) and `audit_prerejected` (the sub-chain's last
  surrogate-rejected proposal of a pre-rejected iteration is evaluated exactly, written as a
  `raw_data` row `state_type="audited"` and sent to the collector with multiplicity 0; the chain is
  unchanged). New stream `seed0 + 4` (`seeds.EXACT_STEP_SEED_OFFSET`, `EXACT_PROPOSAL_SEED_OFFSET = 5`,
  `SEED_FORMULA`, manifest `exact_step_seed`/`exact_proposal_seed`), drawn from only when a field is
  > 0. Outputs: `subchain_stats` column `kernel` (0 DA, 1 exact), `notes` columns `exact_steps,
  exact_accepted, audited, audit_hidden` (audits with `log L - log L~ > AUDIT_HIDDEN_NATS = 5`),
  `StageResult.counter_*` of the same names, stage-end line `- exact steps: a/b accepted - audited: n
  (hidden: m)`; audits count towards `max_evaluations`; `hist_G` drops `audited` rows and CpUS counts
  them as evaluations. An MH stage resets non-zero values with a warning; a frozen stage keeps the
  audit with a start-up note. Plain runs (both fields 0) are byte-identical in `samples`, `raw_data`,
  `last_sample`, `adaptive_stats`, `carry_over` and in `subchain_stats`/`notes` up to the appended
  columns (md5 of local MH/DAMH/adaptive-DAMH/Hamiltonian-DAMH references made before the change, and
  the plain reference scripts). **Behaviour change for Auto**: its DAMH chunks now set both fields to
  the placeholder 0.05 (`auto.AUTO_EXACT_STEP_PROBABILITY`, `AUTO_AUDIT_PREREJECTED`, manifest `auto`
  entry), so an Auto run's chunks follow a different sample stream than before. Tests:
  `tests/unit/test_safety_kernel.py`, `tests/mpi/test_mpi_s2.py`.
- **Polyak averaging of the received surrogate evaluators** (2026-10-09, S3, note 18 §6 N1, note 20
  §6, note 25 S3; `docs/stages.md`, `docs/outputs.md`, `docs/writing_a_surrogate.md`). New `Stage`
  field `polyak_averaging: bool = False` (marked `*`): the sampler installs, instead of each evaluator
  it receives, the running mean (weight `1/k`) of the parameters of the evaluators IT received in the
  stage, restarted at every stage (the state lives in the `AlgorithmBase` instance), so the installed
  surrogate moves by `O(1/k)` per update (diminishing adaptation, A5). One install point
  (`AlgorithmBase._install_evaluator` / `_polyak_candidate` / `_commit_evaluator`) used by
  `_initialize_current_approximation`, `_install_newest_evaluator` and
  `_refresh_surrogate_evaluator_if_needed`; the finiteness check runs on the AVERAGED evaluator and a
  refused candidate advances neither `k` nor the average; in a frozen stage the newest evaluator
  replaces the never-used first one (`k` stays 1). New `Evaluator.averageable_state()` (base `None`)
  and `Evaluator.with_averageable_state(state)` (base `NotImplementedError`), implemented by
  `PyTorchNNEvaluator` (float64 copies of the floating `state_dict` entries; a deep-copied model with
  the model's own dtype); other evaluators are installed unchanged with one printed note per stage.
  Outputs: `notes` column `polyak_updates` (appended last), `StageResult.counter_polyak_updates`,
  stage-end ` - polyak updates: k` (only when the flag is set). No MPI message changes. Chains may
  receive different subsets of the collector's versions, so their averages (and `k`) can differ.
  Plain runs (flag off) are byte-identical in `samples`, `raw_data`, `subchain_stats`, `last_sample`
  and `notes` up to the appended column (md5 of local MH -> DAMH-SMU runs with the NETWORK updater,
  random-walk and Hamiltonian, made before the change; the S2 references; the plain reference
  scripts). **Behaviour change for Auto**: its DAMH chunks now set `polyak_averaging=True`
  (`auto.AUTO_POLYAK_AVERAGING`, manifest `auto` entry `polyak_averaging`), so an Auto run with a
  network surrogate follows a different sample stream than before. Tests: `tests/unit/test_polyak.py`,
  `tests/mpi/test_mpi_s2.py::test_s3_polyak_averaging_with_network_and_collector`.
- **Global Gaussian surrogate error model** (2026-10-09, S1, note 22 §4 P3, note 25 S1; `docs/stages.md`,
  `docs/concepts.md`, `docs/outputs.md`, `docs/writing_a_surrogate.md`). New module
  `surrDAMH.modules.error_model` (MPI-free): `ErrorModel(bias, cov)` (`cov` `(m, m)`, or the `(m,)`
  diagonal above `ERROR_MODEL_FULL_COV_MAX_OBS = 50` observations), `estimate_error_model(residuals)`
  (mean; Ledoit-Wolf, `assume_centered=False`; `diag(var)` with fewer than `max(5, m+1)` residuals),
  `ErrorModel.as_state()/from_state()`, `corrected_likelihood(likelihood, model)` -> `CorrectedNormal`
  = `Normal(mean = y - eps, cov = Sigma_noise + Sigma_surr)` whose `logpdf` adds `-1/2 log det
  Sigma_tot + 1/2 log det Sigma_noise` (same normalisation convention as the exact `Normal.logpdf`).
  The collector (`process_COLLECTOR`) and the local runner's `LocalSurrogateManager` estimate the model
  of every newly trained evaluator from its residuals on all the updater's snapshots
  (`get_training_data_arrays()`, now also on `PolynomialSklearnUpdater`) and the held-out set, pooled
  with uniform weights, and attach it as `evaluator.error_model` (carried by the pickled evaluator: no
  new MPI message). New `Stage` field `surrogate_error_model: bool = False` (marked `*`): with it, every
  surrogate log-likelihood and its gradient (sub-chain, DA correction, `log_likelihood_approx` in
  `raw_data`, Hamiltonian potential, S2 exact-step scoring) uses `AlgorithmBase.likelihood_surrogate =
  corrected_likelihood(...)`, rebuilt at every evaluator install (after S3 averaging), so the re-scoring
  on refresh and the S0 pending re-score use the new model. Non-`Normal` likelihood: one note, flag
  ignored. `Evaluator.averageable_state()` (base) now returns the error model's entries
  (`error_model.bias`/`error_model.cov`; `PyTorchNNEvaluator` adds them to its weights) and
  `with_averageable_state` rebuilds it; a stage without the flag strips these entries before averaging,
  so S3 without S1 is unchanged. `surrogate_quality.csv` gains the columns `error_bias_norm,
  error_cov_trace` (appended). `run_sampling_local_auto` pools its held-out set into the local model.
  Plain runs (flag off) are byte-identical in `samples`, `raw_data`, `subchain_stats`, `notes`,
  `last_sample` (md5 of local MH -> DAMH-SMU runs with the NETWORK updater -- random walk, Hamiltonian,
  S2+S3 -- and the POLYNOMIAL updater -- random walk, S2+S3 -- made before the change; the plain reference
  scripts). The collector now makes one extra batched surrogate call per retraining (the residuals).
  **Behaviour change for Auto**: its DAMH chunks set `surrogate_error_model=True`
  (`auto.AUTO_SURROGATE_ERROR_MODEL`, manifest `auto` entry `surrogate_error_model`), so Auto runs
  follow a different sample stream. Tests: `tests/unit/test_error_model.py`,
  `tests/mpi/test_mpi_s2.py::test_s1_error_model_with_polynomial_updater_and_collector`.

- **Mechanisms of the adaptive-NN ergodicity paper, Section 6.5** (2026-10-23, author decision
  "everything in 6.5"; record `library_notes/37_paper_mechanisms_implementation_2026-10-23.md`).
  All off by default; with the defaults every CSV output is byte-identical to before (checked on four
  configurations against an untouched copy of the package); `run_manifest.json` gains keys
  (`installation_seed` per rank/stage, `exact_step_proposal` and `installation` in every stage dict).
  - `Stage.exact_step_proposal="prior"`: the S2 exact step proposes `y ~ prior` (independence sampler
    in the internal space, stream `seed0 + 5`, `modules/prior_proposal.PriorIndependence`) and accepts
    with `min{1, L(y)/L(x)}`; nothing adapts; out-of-box draws rejected unevaluated. Refused with
    `exact_step_scale`. Recorded in `proposal_resolution`. Default `"random_walk"` = S2 unchanged.
  - `Stage.installation=RandomisedInstallation(n_lambda, a_lambda)` (`surrDAMH.installation`): before
    outer iteration `i >= 1` one uniform from the new stream `seed0 + 6` (`INSTALLATION_SEED_OFFSET`)
    installs the newest received version iff `u < lambda_{i-1} = min{1, ((i-1)/n_lambda)^-a_lambda}`;
    receiving is decoupled from installing (newest received buffered, superseded versions dropped); the
    coin comes before the S2 kernel choice; the stage proposal and the exact step's random walk run in
    deferred adaptation and their live step/scale/beta changes only at installs
    (`Proposal.set_deferred_adaptation`, `apply_pending_adaptation`; live attributes documented in
    `docs/stages.md` "Deferred adaptation"). Outputs only when a rule is set: `notes` +
    `installations, versions_received` (last), `installations/<stage>/rank%04d.csv` (`iteration, version`).
  - `Stage.installation=AveragedInstallation(j0=1, a_zeta=1.0, weight_box=None)`: the running average of
    the RECEIVED versions with gain `zeta_j = min{1, (j/j0)^-a_zeta}` (the stage-start network is
    `average_0`, replaced by the first received version since `zeta_1 = 1`), model part clipped to
    `[-weight_box, weight_box]` before averaging (S1 error-model entries not clipped); refuses adaptive
    proposals and the adaptive exact random walk; refused together with `polyak_averaging=True`, which
    stays as the legacy running mean (stage-start network included, weight `1/k`).
  - `NeuralNetworkUpdater(input_clip=b)`: `clamp` of every input coordinate to `[-b, b]` before the first
    layer, in training and prediction (`PyTorchMLP.forward`); `vjp`/`jacobian` get zero gradient in a
    clipped coordinate; checkpoint hparam (a checkpoint without the key loads as `None`).
  - `Auto` unchanged (S2 random walk + legacy `polyak_averaging`). Tests: `tests/unit/test_prior_step.py`,
    `test_installation.py`, `test_deferred_adaptation.py`, `test_input_clip.py`,
    `tests/mpi/test_mpi_paper_mechanisms.py`; `./run_tests.sh unit` 900 passed.
  - Deferred adaptation, corrected the same evening after the first smoke run: in deferred mode the
    feedback recursions (random-walk log-scale, pCN logit, Hamiltonian dual averaging) advance ONCE per
    `apply_pending_adaptation()` (= per installation) on the mean acceptance gathered since the previous
    one, with the gain index counting installations; advancing them at every step on acceptances of the
    frozen live parameters ran open-loop (log step size fell to the -50 clip, leapfrog count at the cap).
    Default mode unchanged.
  - Experiment tooling (T1, same day): `Evaluator.version` = the collector's publication index
    (`surrogate_update_index`, 1 = first published; the local surrogate manager counts likewise), shipped
    with the pickled evaluator; the `installations/` CSV gains a third column `collector_version`, and a
    new `versions/<stage>/rank%04d.csv` (same columns, one row per RECEIVED evaluator) is written when
    `Stage.installation` is set, so a chain's installed version maps to `surrogate_quality_test.csv`
    (row `k` = publication `k`) and the installation lag is readable.

### Removed

- **Console output (2026-10-09, author request):** the chain start-up line no longer prints the initial sample (only with `Configuration(debug=True)`); it reads `Sampler at rank N started with an LHS/a prior/a user-specified/a continued (from <dir>) initial sample`. The `Saving last sample ...` and `Saving carry-over ...` lines are printed only in debug mode. `Samples.get_summary()` prints every column of the summary table (`to_string()`, 4 significant digits) instead of pandas' elided view.

- `Configuration.pickled_observations` and the raw-observation MPI transport it selected (WS8;
  see Breaking changes and Bug fixes above for the mechanism and the fix it delivers).
- `NeuralNetworkUpdaterBasic` and `surrDAMH/surrogates/torch_perceptron.py` (WS6; see Breaking
  changes above).
- `Stage`/`AlgorithmBase.artificial_acceptance_multiplicator` (WS0 §B5; see Breaking changes
  above).
- `Samples(..., load_posterior_surrogate=...)` and `StageSamples.posterior_surrogate` (WS9b; see
  Breaking changes above).
- Dead code with zero remaining callers, zero behaviour change (WS0 §A/§D, `13_dead_code_report.md`):
  `core.temptemptemp`, the stray `torch.mtia` import, `Algorithm_PARENT` (alias for
  `AlgorithmBase`), the commented-out old `HamiltonianInfinite.__init__`, and the dead
  `if False and ...` branch in `process_SOLVER.py`. **Not removed, by explicit author decision**
  (`10_manual_review_notes.md` §7b, 2026-09-17, "do not delete dead code"): `Stage.proposal`,
  `Proposal.subchain_length`, `TAG_READY_TO_RECEIVE`/`TAG_DATA`, `closest_point_distance*`,
  `PriorIndependentComponents.sd_approximation`, `Updater.supports_*_persistence`, and the five
  superseded `tools.py` helpers — all kept and documented as unused/superseded instead.
- Duplicate `toy_examples/` scripts, one kept per feature (WS10, author decision):
  `neural_network_surrogate_copy.py`, `typical_example_generic.py`, `post_processing_example.py`,
  `test_html_report_extended.py`.
- Obsolete experiment trees moved out of the maintained tree into the git-ignored
  `TSX_experiments_archived/` (WS10, decisions of 2026-09-13 and 2026-09-17):
  `TSX_complete_experiment_1..4`, `sampling_test_Hamilton`, all `out_TSX*` directories,
  `hpcse26_nn_training/`, `talk/`, the root-level `summarize_tsx2_results.py`/`analyze_tsx2_*.py`
  scripts, and assorted root scratch files (`KL_mode_selection_report.md`,
  `MLP_jacobian_stability_report.md`, `TSX2_*.md`, `test_cuda.ipynb`, `test_toy_shrinkable.html`,
  `loss_during_incremental_training.png`, `wait_for_experiment25_then_run_26.sh`). Only
  `toy_examples/` is maintained going forward.
- The `print("Stages:", ...)` progress line from `post_processing.statistics.calculate_CpUS`
  (2026-10-09, author request). Output-only; `calculate_CpUS`'s return value and `summary.csv`
  are unaffected.

### Documentation

- **New `docs/` tree** (WS11): `concepts.md` (Bayesian setting, MH/DAMH/DAMH-SMU derivation),
  `configuration.md` and `stages.md` (every field, posterior-affecting or not), `writing_a_solver.md`,
  `writing_a_surrogate.md` (the WS6 contract), `outputs.md` (format v2, the run manifest,
  `read_run`), `running.md` (process counts, `run_local`, continuation), and a `docs/README.md`
  index.
- **Docstrings added across the user-facing API** (WS11): `SamplingFramework`, `Configuration`,
  `Stage`, `run_local`, `Solver`/`SolverSpec`, the `Distribution` family, `Updater`/`Evaluator`,
  `TestData`, `Samples`/`read_run`, and the proposal classes, each noting which parameters are
  posterior-affecting or unverified.
- **`README.md` updated**: the maintained-examples table now matches the current
  `toy_examples/` tree (including `template_experiment.py`), the FEniCSx dependency of
  `sampling_TSX.py`/`sampling_diffusion_grf*.py` is documented, the "run from `toy_examples/`"
  relative-path contract is stated explicitly, and the `minimal_example.py` run command was
  corrected from `-n 4` to `-n 2` (WS0 §D12, WS5, WS11).
- **`library_notes/06_findings_consolidated.md`, `09_improvement_plan.md`, `10_manual_review_notes.md`
  and `library_notes/README.md` updated** (2026-09-17) with a per-finding/per-workstream status
  against the current tree and test suite, superseding their earlier "predicted"/in-progress
  framing.

- **User-facing docstring pass** (2026-09-21): `SamplingFramework` (workflow `run()` then `write_report()`,
  the two solver options, the surrogate arguments incl. `initial_snapshots` shapes, `TestData`,
  `SurrogateRestart`), `Solver`, `SolverSpec`, `Stage` (short lead: `proposal_sd_or_cov` or `adaptive=True`
  is mandatory for a random walk), `Normal`, `Distribution`. `Configuration` got a short docstring with an
  `Args:` section (Pylance shows it on the constructor; the trailing field comments never were) and its
  fields were **regrouped** (problem / MPI layout / solver / initial sample / surrogate training / outputs /
  performance) -- same names, types and defaults, so only the line order of `Configuration.describe()`
  and the key order of the configuration block in `run_manifest.json` change. The maintainer-level
  content of the old docstring lives on as a comment block above the dataclass.
