# MCMC algorithm layer — notes

Review date: 2026-09-12. Working-tree state (uncommitted changes included), branch `working_Kuba`.
**No code was executed** — everything below comes from reading the source. Findings are labelled
"confirmed by reading" or "suspected".

## Files covered

Read fully:
- `surrDAMH/modules/algorithms.py`
- `surrDAMH/modules/proposals.py`
- `surrDAMH/modules/algorithm_interfaces.py`
- `surrDAMH/modules/algorithm_interfaces_local.py`
- `surrDAMH/modules/algorithm_interfaces_mpi.py` (sampler-side view only)
- `surrDAMH/process_SAMPLER.py`
- `surrDAMH/stages.py`
- `surrDAMH/modules/lhs_normal.py`
- `surrDAMH/modules/continuation.py`
- `surrDAMH/modules/monitoring.py` (needed for the disk format)

Skimmed for context:
- `surrDAMH/core.py`, `surrDAMH/configuration.py`, `surrDAMH/distributions/parent.py`,
  `surrDAMH/distributions/normal.py`, `surrDAMH/distributions/independent_components.py`,
  `surrDAMH/surrogates/parent.py`, `surrDAMH/modules/communication.py` (`CommEvaluator_sampler` only).

## Purpose and role in the framework

This layer is the *chain driver*. It owns:
- the per-stage MCMC loop (`Algorithm_MH`, `Algorithm_DAMH`),
- the proposal kernels (`proposals.py`),
- the acceptance arithmetic in log domain,
- the bookkeeping written to `<output_dir>/sampling_output/…`,
- the backend-neutral contracts (`algorithm_interfaces.py`) that hide "where does an
  observation / a surrogate evaluator come from" behind three protocols
  (`ObservationProvider`, `SnapshotCollector`, `EvaluatorProvider`) plus an
  `AlgorithmConfiguration` protocol that lists exactly the six `Configuration` attributes the
  algorithms touch. MPI (`algorithm_interfaces_mpi.py`) and in-process
  (`algorithm_interfaces_local.py`) implementations are thin adapters; the algorithm module has no
  `mpi4py` import. This separation is clean and is the strongest part of the layer.

Everything else (solver pool, collector/surrogate training, post-processing) lives outside this scope.

## How it works

### Stage loop — `process_SAMPLER.run_SAMPLER` (`surrDAMH/process_SAMPLER.py:29`)

1. `comm_sampler = comm_world.Split(color=0, key=rank_world)` (`:33`) — samplers only.
2. Backends are built **once**, before the stage loop:
   - `MpiSolverPoolObservationProvider` if `conf.use_solvers_pool`, else the local `solver_instance` (`:35`).
   - If `conf.use_collector`: `MpiEvaluatorProvider.from_rank_collector(..., request_initial_evaluator=True)`
     and `MpiSnapshotSink.from_rank_collector(...)` (`:43`, `:48`). Otherwise a
     `LocalEvaluatorProvider` wrapping a user-supplied evaluator, or `None` (`:53`).
3. Initial sample (`:57`–`:67`): `"lhs"` → `lhs.lhs_normal(loc=prior.mean, scale=conf.lhs_scale,
   n=conf.no_samplers, seed=0)`, row `rank_world`; `"user_specified"` → `conf.initial_samples_distribution.rvs()`;
   `"continued"` → `conf.continued_samples[rank_world]` (loaded in `Configuration.__post_init__`,
   `surrDAMH/configuration.py:77`); default → `prior.rvs()`.
4. Per stage `i`: `seed0 = 10*(no_stages*rank_world + i)` (`:73`); proposal seed `seed0+1`,
   algorithm seed `seed0+2`.
5. Proposal selection (`:76`–`:126`) is a chain of `if`s on `stage.proposal_type`
   (`pCN` / `Hamiltonian` / `HamiltonianInfinite` / `block`), then `stage.adaptive`, then plain
   `GaussRandomWalk`. `proposal_sd_or_cov is None` falls back to `proposal_cov_adaptive`, the
   rank-averaged covariance produced by the previous adaptive stage.
6. Interface selection per stage (`:129`–`:149`): snapshot sink only if
   `stage.send_snapshots_to_collector`; evaluator provider if `algorithm_type == "DAMH"` **or** the
   proposal is Hamiltonian (gradients); `use_only_surrogate` replaces the observation provider by
   `commEvaluator.evaluator.as_solver()`.
7. `stage.name` is assigned here (`:152`–`:163`): `alg0000_MH`, `alg0000_MH-adaptive`,
   `alg0000_DAMH`, `alg0000_DAMH-SMU`. It is the directory name used by all output writers.
8. `alg_instance.run()` (`:177`), then: adaptive covariance `Allreduce`/mean (`:180`–`:185`),
   `initial_sample = alg_instance.current` unless `stage.is_excluded` (`:188`),
   `save_last_sample(...)` (`:191`), optional collector shutdown (`:194`–`:204`),
   `comm_sampler.Barrier()` (`:209`).

### Algorithm construction — `AlgorithmBase.__init__` (`surrDAMH/modules/algorithms.py:71`)

- `self.current = initial_sample` — **not copied**, the caller's `Sample` is mutated in place.
- `self._generator = np.random.RandomState(seed)` (`:95`) — the acceptance-decision RNG, separate
  from each proposal's own `RandomState`.
- `_prepare_run()` (`:120`): starts the clock; evaluates the initial sample **only if
  `current.observations is None`** (`:122`); writes the `subchain_stats` CSV header for DAMH.
- `_initialize_current_approximation()` (`:255`): pulls an `Evaluator`, computes
  `current.observations_approx` / `log_likelihood_approx`, and, if `conf.use_surrogate_gradients`,
  installs `log_likelihood_gradient` / `log_prior_gradient` on the proposal (`:268`–`:271`).

`Sample` (`:27`) is a small dataclass: `parameters`, `observations`, `observations_approx`,
`log_likelihood`, `log_prior`, `log_likelihood_approx`, `solver_tag`, plus `log_posterior`
properties and an explicit `copy()`.

### MH — `Algorithm_MH.run` (`:301`)

```
max_steps = min(stage.max_samples, stage.max_evaluations)
for i in range(max_steps):
    proposal.choose_group()
    proposed = Sample(proposal.propose_sample(current.parameters))
    evaluate proposed (full model)
    likelihood_part, prior_part = proposal.get_log_acceptance_probability(...)
    proposal.adapt(proposed, likelihood_part + prior_part)
    accept/reject; break on time_limit
```
Since each iteration costs exactly one full-model evaluation, `min(max_samples, max_evaluations)`
is the correct combined bound.

### DAMH — `Algorithm_DAMH` (`:330`)

Outer loop (`:429`): for each outer iteration, `_propose_new_sample_using_subchain()` (`:385`) runs a
surrogate-only MH sub-chain of exactly `stage.subchain_max_length` steps starting from
`self.current.copy()`:

- `_refresh_surrogate_evaluator_if_needed()` (`:331`) polls the evaluator provider **inside** the
  sub-chain when `stage.surrogate_model_updates` (DAMH-SMU).
- `_evaluate_surrogate_transition()` (`:339`) computes the approximate observations. On an evaluator
  change it re-evaluates **both** sub-chain states *and* refreshes `self.current.observations_approx`
  / `log_likelihood_approx` (`:345`–`:350`) — this is exactly the classic DAMH-SMU pitfall
  ("stale surrogate value at the current state") and it **is handled**.
- With `conf.state_dependent_approximation` (`:358`) the proposed state's approximation is shifted:
  `G̃(y) + G(x₀) − G̃(x₀)`, and the sub-chain current state is scored with `G(x₀)` directly.
- Inner acceptance uses the same `proposal.get_log_acceptance_probability`, so the inner kernel is
  an MH kernel targeting `π̃ ∝ prior · L̃`.
- On every inner **acceptance**, `correction_log_ratio += likelihood_part` (`:409`). Because the
  accepted steps chain together, this telescopes to `log L̃(x_K) − log L̃(x₀)`.

Outer step (`:434`–`:468`): if at least one inner step was accepted, the full model is evaluated at
the sub-chain endpoint and the decision is

```
exact_likelihood_log_ratio = proposed.log_likelihood - current.log_likelihood   # :461
accepted = draw(exact_likelihood_log_ratio - correction_log_ratio)              # :462
```

i.e. `log α = [log L(y) − log L(x)] − [log L̃(y) − log L̃(x)]`.
**This is the mathematically correct second-stage ratio** for a π̃-reversible sub-chain kernel:
`α = π(y)π̃(x) / (π(x)π̃(y))` and the prior cancels between `π` and `π̃`. The commented-out
"intended" line (`:460`, `log_acceptance_prob_exact - correction_log_ratio`) would have added a
spurious extra prior ratio `log p(y) − log p(x)` and would be **wrong**. The comment calls the
current code a "weird hot fix" — it is the opposite; it should be promoted to the documented formula.
If no inner step was accepted, the outer proposal equals the current state and the step is counted
as *pre-rejected* (no full-model evaluation) (`:469`–`:471`).

Stopping: outer loop bound `stage.max_samples`, `time_limit` check (`:486`), and
`counter_rejected + counter_accepted >= stage.max_evaluations` (`:489`) — i.e. `max_evaluations`
counts full-model evaluations only, correctly excluding pre-rejections.

### Proposals (`surrDAMH/modules/proposals.py`)

| class | line | acceptance contribution |
|---|---|---|
| `GaussRandomWalk` | `:50` | symmetric; `(ΔlogL, Δlogprior)` |
| `PCN` | `:84` | `(ΔlogL, 0.0)` — prior cancels by construction |
| `GaussRandomWalk_adaptive` | `:130` | inherits RW; adapts covariance every `period` proposals |
| `Hamiltonian` | `:190` | `(ΔlogL, Δlogprior + ΔK)` with `K = ½ pᵀM⁻¹p` |
| `HamiltonianInfinite` | `:310` | same, with a prior-preserving rotation instead of drift |
| `BlockProposal` | `:357` | delegates to the sub-proposal of the currently chosen group |

Checked and **correct**:
- pCN step `m + √(1−β²)(x−m) + β·η, η~N(0,C₀)` (`:122`), with `C₀` taken from
  `prior.get_covariance()` (`process_SAMPLER.py:81`). For `PriorIndependentComponents` the internal
  prior is exactly `N(0, I)` (`distributions/independent_components.py:52,60`), so the cancellation
  is exact and the `transform` mechanism does not break it — `logpdf` is always the *internal*
  log-density and `transform` is applied only before the solver / before saving.
- HMC momentum is drawn from `N(0, M)` with `M = diag(sd²)` or `M = cov` (`:216`–`:247`), matching
  the kinetic term `½pᵀM⁻¹p` used in the acceptance (`:258`); the gradient callbacks return
  `∇U = −∇log L` and `−∇log p` (`algorithms.py:273`, `:292`) so the leapfrog signs are right; the
  integrator is palindromic (half-kick / drifts / half-kick) and the final momentum flip does not
  change `K`. Valid HMC.
- `HamiltonianInfinite._apply_prior_kinetic_flow` (`:311`) is a rotation with determinant
  `cos² + sin² = 1` → volume preserving, and the palindromic structure + flip keeps the map an
  involution, so the MH acceptance `exp(−ΔH)` (which still contains the *full* prior ratio) targets
  the correct posterior for any prior. Note however that the rotation coefficients
  (`angle = ε/sd`, `q += (sin/sd)·p`, `p -= sd·sin·q`) solve `q̇ = M⁻¹p, ṗ = −C⁻¹q` with
  `M = diag(sd²)` and **`C = I`** — i.e. the "prior-preserving" property only holds for the standard
  normal internal prior. With `sd_or_cov ≠ 1` it degrades to a (still valid) preconditioned
  integrator whose linear part no longer matches the prior. Worth documenting.

### What is written to disk, and when

All files go through `SamplingOutputMonitor` (`surrDAMH/modules/monitoring.py:38`) to
`<conf.output_dir>/sampling_output/<data_name>/<stage.name>/rank####.csv`, opened lazily in `"w"`
mode and closed only in `_finalize_run` (`algorithms.py:233`).

| `data_name` | written when | row |
|---|---|---|
| `samples` | on each acceptance, before leaving the state (`:207`, via `:145`), and once in `_finalize_run` | `[weight, parameters…, log_posterior]`; `weight = 1 + counter_rejected_current`; parameters transformed iff `conf.transform_before_saving` |
| `raw_data` | every proposed state, iff `conf.save_snapshots_to_file` (`:150`) | `[state_type, parameters…, solver_tag, observations…, log_likelihood, log_prior]`; `state_type ∈ {accepted, rejected, prerejected}` |
| `notes` | once in `_finalize_run` (`:229`) | header + `[accepted, rejected, pre-rejected, sum, seed]` |
| `subchain_stats` | header in `_prepare_run` (`:124`), one row per outer DAMH iteration (`:472`) | `[iteration, subchain_max_length, subchain_accepted, subchain_acceptance_rate, correction_log_ratio, outer_proposed_changed, outer_accepted, rank_world]` |

The two `log_likelihood` / `log_prior` columns at the end of `raw_data` are **new in the uncommitted
diff** (`git diff -- surrDAMH/modules/algorithms.py`) — any post-processing that reads `raw_data`
positionally from the end will break.

Additionally `save_last_sample` (`surrDAMH/modules/continuation.py:12`) writes
`<output_dir>/sampling_output/last_sample/<stage.name>/rank####.npz` with keys `parameters`
(**float32**) and `no_parameters`, after every stage (`process_SAMPLER.py:191`).

Snapshots are forwarded to the collector as `[parameters, observations, weight]`
(`algorithms.py:215`), with `parameters` transformed iff `conf.transform_before_surrogate`.

## Potential bugs and risks

| ID | Sev | Location | Description / why it matters | Confidence | How to verify |
|---|---|---|---|---|---|
| A1 | high | `surrDAMH/core.py:154,158,163,164` vs `:6` | `Iterable` and `Any` are used in the annotations of `temptemptemp()` but never imported (`from typing import List, Literal`). Python 3.12 evaluates annotations eagerly, so the class body raises `NameError` → `import surrDAMH` fails, i.e. **nothing runs**. The whole method is dead (`return` at `:195`). | confirmed by reading | `python -c "import surrDAMH"` |
| A2 | high | `surrDAMH/process_SAMPLER.py:12` | `from torch.mtia import snapshot` — the name `snapshot` is never used (grep confirms). It makes torch a hard import on every sampler rank and targets a very recent `torch.mtia` API; on any torch without it, all samplers die at import. Almost certainly an IDE auto-import accident. | confirmed by reading | delete the line; `python -c "import surrDAMH.process_SAMPLER"` |
| A3 | high | `surrDAMH/process_SAMPLER.py:114` vs `surrDAMH/stages.py:35-37` | `GaussRandomWalk_adaptive` is constructed with `no_parameters` and `seed` only. `stage.adaptive_target_rate`, `adaptive_corr_limit`, `adaptive_sample_limit` are **never passed** → the proposal always uses `target_rate=0.25, corr_limit=0.3, period=10`. `TSX_complete_experiment_2/sampling_.py:450` sets `adaptive_target_rate=0.1`, which is silently ignored — results of that experiment were produced with 0.25. Changes acceptance rate and hence mixing/ESS. | confirmed by reading | grep for `target_rate` in `process_SAMPLER.py` (absent); print `my_Prop.target_rate` after construction |
| A4 | medium | `surrDAMH/stages.py:36` | `adaptive_corr_limit = None` has **no type annotation** → `dataclass` treats it as a plain class attribute, not a field. `Stage(adaptive_corr_limit=0.4)` raises `TypeError: unexpected keyword argument`. | confirmed by reading | `Stage(adaptive_corr_limit=0.4)` |
| A5 | high | `surrDAMH/modules/algorithms.py:358-373` | `state_dependent_approximation` combined with `subchain_max_length > 1`: `subchain_current.log_likelihood_approx` is always computed from `self.current.observations` (the **outer** current state's exact observations), even after the sub-chain has moved away from it. The correct shifted value at `x_i` is `G̃(x_i) + G(x₀) − G̃(x₀)`. The inner kernel is then not π̃-reversible and `correction_log_ratio` no longer telescopes → **the outer chain does not target the posterior**. For `subchain_max_length == 1` it is exactly right (and elegantly makes the correction exact at the current state). | confirmed by reading | unit test: 2-D Gaussian toy, `state_dependent_approximation=True`, `subchain_max_length=5`, deliberately biased surrogate; compare posterior mean/cov against plain MH |
| A6 | medium-high | `surrDAMH/modules/algorithms.py:389-413` (+ TODO at `:451`) | DAMH-**SMU** with `subchain_max_length > 1`: `_refresh_surrogate_evaluator_if_needed()` is called *inside* the sub-chain, so `correction_log_ratio` sums ratios computed under **different** surrogates and no longer equals `log L̃(x_K) − log L̃(x₀)` for any single `L̃`. The second-stage correction is then inconsistent → posterior bias of unknown size. With `subchain_max_length == 1` the refresh happens before both evaluations and the step is correct. | confirmed by reading | freeze the evaluator for the duration of one sub-chain (move the refresh call outside the `for` loop at `:389`) and compare posteriors on a toy problem |
| A7 | high (footgun) | `surrDAMH/modules/algorithms.py:200-205`, `stages.py:39` | `artificial_acceptance_multiplicator` is applied inside `_draw_acceptance_decision`, which is used by **both** the outer MH/DAMH acceptance **and** the surrogate sub-chain acceptance. Any value ≠ 1 makes acceptance probabilities exceed 1 → the chain no longer targets the posterior, *and* the inner kernel stops being π̃-reversible so the DAMH correction cannot compensate. No warning is printed. Currently 1.0 everywhere in tracked code. | confirmed by reading | grep shows only the definition and the single use; add a loud warning + a `notes` column when ≠ 1 |
| A8 | medium | `surrDAMH/modules/algorithms.py:185-189` | `_handle_rejection` guards with `if not self.current.solver_tag < 0:` but sends `self.proposed`. A *proposed* sample whose solver failed (negative tag, garbage/NaN observations) is still forwarded to the collector as training data; conversely valid proposals are dropped whenever the current state happens to carry a stale negative tag. Looks like `self.current` should be `self.proposed`. Poisons the surrogate. | suspected (logic inversion) | construct a solver returning `(nan_obs, -1)` for a region of parameter space, `save_snapshots_to_file=True`, check whether NaN rows reach the collector |
| A9 | medium | `surrDAMH/process_SAMPLER.py:180-185` | After an adaptive stage: `sendbuf = my_Prop.sd_or_cov; comm_sampler.Allreduce(sendbuf, recvbuf)`. `sd_or_cov` is 1-D until the first adaptation and 2-D afterwards (`proposals.py:63-66,184`). A rank whose chain did fewer than `period` proposals still holds a 1-D array → mismatched buffer shapes across ranks → MPI error or hang. Also `BlockProposal` has no `sd_or_cov` at all, so `stage.adaptive=True` with `proposal_type="block"` raises `AttributeError` here. | confirmed by reading | run an adaptive stage with `max_evaluations < period` (10) on ≥2 samplers |
| A10 | medium | `surrDAMH/process_SAMPLER.py:194-196` | `stages_will_use_surrogate = following_DAMH or following_onlySurr` — Python's `or` on lists returns the **first non-empty list**, so `following_onlySurr` is ignored whenever any later stage exists. A later `use_only_surrogate` stage that follows the last DAMH stage will have the collector already terminated. Should be `any(following_DAMH) or any(following_onlySurr)`. | confirmed by reading | stage list `[MH, DAMH, MH(use_only_surrogate)]`, `use_collector=True` |
| A11 | medium | `surrDAMH/modules/algorithms.py:120-123` | `_prepare_run` reuses `current.observations` from the previous stage instead of re-evaluating. After a `use_only_surrogate` stage those observations are **surrogate outputs**, so the first acceptance ratio of the next (exact) stage compares an exact `log L(y)` against a surrogate `log L(x)`. Affects the first accepted step; also makes `samples`/`raw_data` of the next stage start from a surrogate log-posterior. | confirmed by reading | run `[MH, MH(use_only_surrogate), MH]` and compare the first `samples` row's `log_posterior` against a manual exact evaluation |
| A12 | medium | `surrDAMH/modules/algorithms.py:266-271`, `core.py:74-109` | Gradient callbacks are installed **only if `conf.use_surrogate_gradients`**. `core._configure_surrogate_gradients` silently flips that flag to `False` (e.g. when `transform_before_surrogate=True`, or the surrogate has no gradients). A Hamiltonian stage then reaches `_leapfrog` with `self.log_likelihood_gradient` unset → `AttributeError` mid-run (after possibly hours). Nothing checks `proposal.needs_gradients` (the attribute exists, `proposals.py:16`, but is never read in `process_SAMPLER`/`algorithms`). | confirmed by reading | `proposal_type="Hamiltonian"` + `use_surrogate_gradients=False` |
| A13 | medium | `surrDAMH/process_SAMPLER.py:107-112` | For `proposal_type="block"`, only `BlockProposal`'s group-selection generator gets `seed0+1`. The sub-proposals come from `stage.block_proposal_list`, built once in the user script with the user's own seeds — **identical objects and seeds on every MPI rank**. All chains then draw the same proposal increments → chains are not independent, R-hat/ESS across chains become meaningless. The same objects are also reused across stages, so RNG state carries over. | confirmed by reading | print the first 3 proposed samples per rank for a block stage; they should differ, they will not |
| A14 | medium | `distributions/normal.py:80,84`, `distributions/independent_components.py:66` | `rvs()` uses the **global unseeded** `np.random`. With the default `initial_sample_type="prior"`, initial samples are not reproducible and are not derived from `seed0`. (`lhs` and `continued` are reproducible; `user_specified` depends on the user distribution.) | confirmed by reading | run the same script twice with `initial_sample_type="prior"`, compare the printed initial samples |
| A15 | medium | `surrDAMH/modules/proposals.py:160-187` | Adaptive proposal: (a) `np.cov` over only `period=10` proposals — with `no_parameters > 10` the sample covariance is singular, so `multivariate_normal` proposals collapse onto a low-dimensional subspace (relevant for the TSX / KL runs with many parameters); (b) `self.samples` grows without bound and `np.cov` is recomputed over **all** samples every `period` steps → O(n·d²) work per adaptation, quadratic overall, plus unbounded memory — `adaptive_sample_limit` clearly exists to cap this and is unused (see A3); (c) if all recent acceptance probabilities are 0 (or NaN), `aweights` sums to 0 → NaN covariance, propagated silently into the proposal; (d) `sd = sqrt(diag(cov))` with a zero variance → division by zero. | confirmed by reading | adaptive stage with `no_parameters=20`; inspect `np.linalg.matrix_rank(my_Prop.sd_or_cov)` after the first adaptation |
| A16 | medium | `surrDAMH/modules/algorithms.py:191-205` | Solver errors map to `(-inf, -inf)`. If the **current** state has `-inf` log-likelihood (e.g. the *initial* sample failed the solver), every ratio becomes `-inf − (−inf) = NaN`; `np.log(u) < NaN` is `False` → every proposal is rejected → the chain is dead for the whole stage with no error message. Also setting `log_prior = -inf` is unnecessary (the prior is computable) and additionally poisons `prior_part`. | confirmed by reading | solver that returns `tag=-1` for the initial sample; observe `accepted=0` in `notes` |
| A17 | low-medium | `surrDAMH/modules/algorithms.py:447` | In DAMH, `proposal.adapt()` is called **only** when the sub-chain accepted something, so the adaptive proposal's `current_rate` is the *second-stage* acceptance rate, not the overall one. `target_rate` therefore means different things in MH and DAMH stages; the adapted covariance is also computed from only the sub-chain endpoints. | confirmed by reading | compare `mean(aweights)` against `accepted/(accepted+rejected+prerejected)` from `notes` |
| A18 | low | `surrDAMH/modules/algorithms.py:440-447`, `proposals.py:252-260` | With a Hamiltonian proposal inside DAMH, the outer `get_log_acceptance_probability` reuses `p_current_start` / `p_proposed_end` left over from the **last inner** proposal. The returned value is not used for the decision (the code uses `exact_likelihood_log_ratio − correction_log_ratio`), but it *is* fed to `proposal.adapt()`. Harmless today; a trap if the hot fix is ever "cleaned up" back to `log_acceptance_prob_exact`. | confirmed by reading | n/a — read `:446` and `:462` together |
| A19 | low | `surrDAMH/modules/proposals.py:84-127`, `process_SAMPLER.py:77-83` | `PCN` assumes the internal prior is exactly `N(prior.mean, prior.get_covariance())`. Nothing checks Gaussianity — with a non-Gaussian `Distribution` (e.g. `FromScipy`) the prior ratio is silently dropped and the chain targets the wrong measure. `beta` is not validated in `(0,1]` (`beta>1` → `sqrt` of a negative → NaN). | confirmed by reading | add an assertion; or run pCN with a non-Gaussian prior and compare against RWMH |
| A20 | low | `surrDAMH/surrogates/parent.py:10-18`, `algorithms.py:104-118` | `use_only_surrogate` routes the full model through `SurrogateAsSolver`, which passes the **1-D** parameter vector straight to `Evaluator.__call__`, whose documented contract is `(n_datapoints, no_parameters)`. Everywhere else the algorithm wraps with `np.array([argument])` and `.ravel()`s (`algorithms.py:235-253`). Depending on the evaluator this either broadcasts wrongly or returns a `(1, n_obs)` array that then breaks `Normal.calculate_logpdf_uncorrelated`'s `np.dot`. | suspected | run a `use_only_surrogate` stage with the poly / RBF evaluator and check the shape of `get_observations()` |
| A21 | low | `surrDAMH/modules/lhs_normal.py:29-49` | "best of 5 LHS designs" does not work: `maxmin` is initialised to 0 and **never updated**, so the last candidate with `quality > 0` always wins. `LHS_final = np.zeros([n, n])` has the wrong shape when `no_parameters != n` (harmless only because it is always overwritten). The inner loop reuses the loop variable `i` of the outer loop. `import numpy.matlib` is deprecated in NumPy ≥ 1.19 and emits a warning. Affects only the spread of initial samples. | confirmed by reading | add `maxmin = quality` inside the `if` and compare the min-distance of the returned design |
| A22 | low | `surrDAMH/modules/continuation.py:15-19` | The last sample is stored as `float32`. Continuing a run therefore restarts from a truncated parameter vector — harmless statistically but it means "continued" is not bit-exact and small-scale parameters lose ~7 digits. | confirmed by reading | `np.load(...)["parameters"].dtype` |
| A23 | low | `surrDAMH/modules/algorithms.py:150-162,469-471` | `raw_data` rows are **ragged**: the observation block is skipped when `observations is None`, and the number of columns differs between `accepted`/`rejected` (exact observations) and `prerejected` (approximate observations). Moreover a `prerejected` row carries the *current* state's parameters and its **exact** `log_likelihood`/`log_prior` next to *approximate* observations — easy to misread downstream. | confirmed by reading | inspect a DAMH `raw_data` CSV and compare the column count of `prerejected` vs `accepted` rows |
| A24 | low | `surrDAMH/modules/monitoring.py:25-35` | Files are opened with `"w"` and never flushed until `_finalize_run`. A crash, an MPI abort, or a wall-clock kill loses the buffered tail of every CSV, and a re-run silently truncates the previous stage output. | confirmed by reading | kill a running sampler and inspect the CSV tail |
| A25 | low | `algorithms.py:25,416-428,491-503`; `proposals.py:286-308`; `algorithms.py:297` | Dead code: `evaluate_on_a_grid` is imported only for two large commented-out debug blocks inside `Algorithm_DAMH.run`; the `Algorithm_PARENT = AlgorithmBase` alias; a commented-out `HamiltonianInfinite.__init__`; `Proposal.subchain_length` and `Stage.proposal` are never read anywhere. | confirmed by reading | grep |
| A26 | low | `surrDAMH/modules/proposals.py:140-143` | `GaussRandomWalk_adaptive.__init__` does not call `super().__init__()` (its own TODO), so `needs_gradients` / `subchain_length` are missing. `BlockProposal.__init__` does `any(p.needs_gradients for p in list_of_proposals)` (`:378`) → `AttributeError` if an adaptive proposal is put in a block. | confirmed by reading | `BlockProposal(..., [GaussRandomWalk_adaptive(2)], [[0,1]])` |
| A27 | low (perf) | `algorithm_interfaces_local.py:207`, `algorithms.py:331-337` | `LocalEvaluatorProvider.evaluator_is_available()` returns `True` whenever *any* evaluator exists, not only when a *new* one is pending. In a DAMH-SMU stage without a collector this makes `_refresh_surrogate_evaluator_if_needed()` report "changed" on every sub-chain step → 3 surrogate calls per step instead of 1, and `self.current`'s approximation is recomputed every step. | confirmed by reading | count evaluator calls for a local DAMH-SMU stage |
| A28 | low | `surrDAMH/process_SAMPLER.py:152-163` | `alg_class` is left unbound if `stage.algorithm_type` is neither `"MH"` nor `"DAMH"` → `NameError` at `:166` (the `Literal` type hint is not enforced at runtime). `stage.name` is mutated on the shared `Stage` object, so re-running the same `Stage` list in one process silently renames output directories. | confirmed by reading | `Stage(algorithm_type="damh")` |
| A29 | low | `surrDAMH/stages.py:17` | The `Literal` for `proposal_type` omits `"block"`, which `process_SAMPLER.py:106` supports. Type checkers reject valid configurations. | confirmed by reading | mypy |
| A30 | low | `algorithms.py:227-233` + `process_SAMPLER.py:188-189` | The last state of stage *i* is written by `_finalize_run` **and** becomes the first state of stage *i+1*, where it is written again with its own weight. Combining stages in post-processing therefore double-counts one state per stage per chain (negligible for long chains, not for short ones). `_finalize_run` also never forwards that final state to the collector. | confirmed by reading | sum the `weight` column per stage and compare with `accepted+rejected+prerejected` from `notes` |

## Suggested improvements

Small-diff, roughly in order of value:

1. Delete `surrDAMH/process_SAMPLER.py:12` (A2) and fix `core.py`'s imports or delete the dead
   `temptemptemp` method (A1). These two are blockers for anything else.
2. Pass the stage's adaptive settings through (A3):
   `GaussRandomWalk_adaptive(no_parameters=…, seed=seed0+1, target_rate=stage.adaptive_target_rate or 0.25,
   corr_limit=stage.adaptive_corr_limit or 0.3, period=stage.adaptive_sample_limit or 10)`,
   and annotate `adaptive_corr_limit: float | None = None` in `stages.py` (A4). Then re-check which
   TSX experiments were run with the intended target rate.
3. Move `_refresh_surrogate_evaluator_if_needed()` out of the sub-chain loop
   (`algorithms.py:389-390` → before the loop) so the surrogate is frozen for the whole sub-chain
   (A6). If mid-sub-chain updates are wanted, recompute `correction_log_ratio` at the end as
   `log L̃_new(x_K) − log L̃_new(x₀)` instead of accumulating.
4. Fix `state_dependent_approximation` for `subchain_max_length > 1` (A5): compute
   `subchain_current`'s shifted approximation the same way as `subchain_proposed`'s, i.e.
   `G̃(x_i) + G(x₀) − G̃(x₀)`, and keep the special case `x_i == x₀ → G(x₀)` only implicitly
   (it falls out automatically). Alternatively assert `subchain_max_length == 1` when the flag is on.
5. Replace the "weird hot fix" comment block (`algorithms.py:448-462`) with a two-line derivation of
   `log α = Δlog L − Δlog L̃` and delete the commented alternative — the current code is the correct
   one and the comment invites a regression.
6. Rename `_draw_acceptance_decision`'s multiplier usage so it only applies to the *outer* decision,
   or drop `artificial_acceptance_multiplicator` entirely; at minimum print a prominent warning and
   record it in `notes` (A7).
7. `algorithms.py:187`: change `self.current.solver_tag` → `self.proposed.solver_tag` (A8), after
   confirming the intent with the author.
8. `process_SAMPLER.py:196`: `stages_will_use_surrogate = any(following_DAMH) or any(following_onlySurr)`
   and drop the now-redundant `any(...)` at `:198` (A10).
9. `process_SAMPLER.py:180-185`: before the `Allreduce`, normalise `sd_or_cov` to a full 2-D
   covariance (`np.diag(sd**2)` when 1-D) so all ranks agree on the shape (A9).
10. Add an explicit check after proposal construction: `if my_Prop.needs_gradients and not
    conf.use_surrogate_gradients: raise` (A12), and give `Proposal` class-level defaults
    (`needs_gradients = False`) instead of instance attributes so A26 cannot happen.
11. Give the distributions a seeded generator (`rvs(self, generator=None)`) so `initial_sample_type="prior"`
    becomes reproducible (A14); derive it from `seed0` in `process_SAMPLER`.
12. Re-seed the block sub-proposals per rank in `process_SAMPLER` (e.g. call a
    `reseed(seed0 + 100 + k)` on each) or deep-copy the list per rank (A13).
13. Move to `np.random.default_rng(np.random.SeedSequence(...).spawn(...))` instead of
    `RandomState` with the hand-rolled `10*(no_stages*rank + i)` arithmetic; it removes the
    "adjacent small seeds" concern and makes the per-chain/per-stage stream structure explicit.
14. Save `parameters` as `float64` in `continuation.py` (A22).
15. Make `raw_data` rectangular (write a fixed number of observation columns, NaN-filled) and add a
    header row so post-processing does not depend on positional offsets (A23) — especially now that
    two columns were just appended.
16. Flush the CSV writers periodically (or `open(..., buffering=1)`) so partial runs are usable (A24).
17. Docstrings/type hints: `AlgorithmBase`, `Algorithm_MH.run`, `Algorithm_DAMH.run`,
    `_propose_new_sample_using_subchain` and `_evaluate_surrogate_transition` have none; the last two
    are the mathematically subtle ones and deserve the derivation in prose. `_record_current_sample`
    redeclares `row: List[Any]` in both branches (mypy error). `Stage.block_proposal_groups` is typed
    `list[slice]` but used as index lists.
18. Remove the dead debug blocks and the `evaluate_on_a_grid` import (A25); delete the unreachable
    `except NotImplementedError` fallback in `_compute_surrogate_log_likelihood_gradient`
    (`algorithms.py:287-290`) — the base `Evaluator.vjp` already falls back to `jacobian`, so the
    handler can only fire when `jacobian` itself is missing, in which case it immediately re-raises.

## What should be tested or validated

None of the following were run.

1. **Import smoke test** (catches A1/A2): `python -c "import surrDAMH"` and
   `python -c "import surrDAMH.process_SAMPLER"`. Expected: no exception. Add as a pytest so this
   class of stray import cannot land again.
2. **MH correctness on a Gaussian toy.** Linear solver `G(x) = A x`, Gaussian likelihood, standard
   normal prior → analytically known Gaussian posterior. Run `mpiexec -n 4 python3 -m mpi4py …` with
   a single `Stage(algorithm_type="MH", max_samples=200000)`; compare the weighted sample mean and
   covariance from `sampling_output/samples/alg0000_MH/rank*.csv` against the closed form
   (weights are column 0). Expected: agreement within MC error.
3. **DAMH == MH in distribution.** Same toy, a deliberately *wrong but fixed* surrogate (e.g. `A' = 1.3·A`),
   `Stage(algorithm_type="DAMH", surrogate_model_updates=False, subchain_max_length=K)` for
   `K ∈ {1, 5, 20}`. Expected: the same posterior as (2) for every `K`. This is the single most
   valuable test — it pins down A5/A6 and would have caught a sign error in `correction_log_ratio`.
4. **DAMH-SMU with sub-chains** (A6): repeat (3) with `surrogate_model_updates=True` and a collector
   that retrains often. Expected: same posterior. If it drifts, the mid-sub-chain refresh is the cause.
5. **`state_dependent_approximation`** (A5): repeat (3) with the flag on, `K=1` and `K=5`.
   Expected: `K=1` matches, `K=5` currently should not.
6. **Unit test for the correction telescoping** (no MPI): instantiate `Algorithm_DAMH` with
   `LocalSolverAdapter` + `LocalEvaluatorProvider` (both already exist in
   `algorithm_interfaces_local.py`), run one `_propose_new_sample_using_subchain()` and assert
   `correction_log_ratio == subchain_current.log_likelihood_approx - initial.log_likelihood_approx`
   to machine precision. This is a cheap invariant that directly encodes the DAMH derivation.
7. **Proposal unit tests** (no MPI, pure numpy):
   - `PCN.propose_sample` applied repeatedly from a prior draw with no likelihood should leave
     `N(prior_mean, C₀)` invariant → compare the empirical covariance of 10⁵ pCN steps to `C₀`.
   - `Hamiltonian._leapfrog` on a quadratic `U` should conserve `H` to `O(ε²)`: assert
     `|ΔH| < c·ε²` and that halving `ε` reduces `|ΔH|` by ≈4.
   - `Hamiltonian`/`HamiltonianInfinite` reversibility: `leapfrog(leapfrog(q,p)) == (q,p)` after the
     two momentum flips, to ~1e-10.
   - `HamiltonianInfinite._apply_prior_kinetic_flow` must preserve `0.5*q@q + 0.5*p@M_inv@p` exactly
     when `sd_or_cov == 1.0` — this is the concrete check of the `C = I` observation above.
8. **Adaptive proposal** (A15): construct `GaussRandomWalk_adaptive(no_parameters=20, period=10)`,
   feed 10 `adapt()` calls, assert `np.linalg.matrix_rank(prop.sd_or_cov) == 20`. Expected to fail
   today (rank ≤ 10).
9. **Seed independence across chains** (A13/A14): run 4 samplers, `initial_sample_type="lhs"`,
   `proposal_type="block"`, and assert the first 10 rows of `samples/alg0000_*/rank0000.csv` and
   `rank0001.csv` differ.
10. **Solver-failure robustness** (A16): a solver returning `tag=-1` in a half-space. Expected: the
    chain keeps moving in the valid region; today, if the *initial* sample is in the invalid region,
    `notes` should show `accepted=0` for the whole stage.
11. **Continuation round-trip** (A22): run a short experiment, then a second with
    `initial_sample_type="continued"`, and assert the printed initial sample equals the last
    `samples` row of the source stage (currently only to float32).
12. **Output-format regression**: a golden-file test on one short toy run covering the column layout
    of `samples`, `raw_data`, `notes`, `subchain_stats` — the uncommitted `raw_data` change
    (two appended columns) shows this is needed.

## Open questions for the author

1. `algorithms.py:459-462` — the "2 lines of weird hot fix". My reading is that the *current* code is
   correct and the commented-out line would add a spurious prior ratio. Was the hot fix introduced
   empirically (acceptance rates looked wrong) or derived? Can the comment be replaced by the
   derivation? (A5/A6 hinge on this being deliberate.)
2. `_handle_rejection` (`:187`): is `if not self.current.solver_tag < 0` guarding the *current* state
   intentional, or should it be `self.proposed.solver_tag`? What is the intended policy for sending
   failed solver evaluations to the collector at all?
3. Is `subchain_max_length > 1` actually used together with `state_dependent_approximation` or with
   `surrogate_model_updates=True` in any of the TSX / hpcse26 experiments? That decides whether A5
   and A6 are latent or have already affected published numbers.
4. `adaptive_target_rate` / `adaptive_corr_limit` / `adaptive_sample_limit` are declared on `Stage`
   but never reach the proposal (A3). Were they ever wired, or were they added for a planned change?
   `TSX_complete_experiment_2/sampling_.py:450` sets `adaptive_target_rate=0.1` — should that run be
   re-labelled as target rate 0.25?
5. `artificial_acceptance_multiplicator`: what is it used for, and is applying it to the DAMH
   sub-chain intended? Should it be restricted to the outer acceptance and recorded in `notes`?
6. `HamiltonianInfinite`: the prior-preserving rotation implies prior covariance `C = I` with mass
   `M = diag(sd²)`. Is `sd_or_cov` meant purely as a mass/preconditioner, or was it meant to be the
   prior covariance? (If the latter, the rotation angles should be `ε/sd²` and the drift coefficient 1.)
7. For a Hamiltonian proposal in an `MH` stage, the surrogate used for gradients is fetched once at
   stage start and never refreshed (`_refresh_surrogate_evaluator_if_needed` exists only on
   `Algorithm_DAMH`). Intentional?
8. Should the last state of a stage be written by `_finalize_run` *and* re-written as the first state
   of the next stage (A30)? Post-processing that concatenates stages double-counts it.
9. `Stage.proposal: Proposal | None` is never read by `process_SAMPLER` — is a "user supplies a ready
   proposal instance" path planned, or can the field go?
10. `algorithm_interfaces_local.py` provides a complete standalone path (`LocalSolverAdapter`,
    `LocalSurrogateManager`) that nothing currently instantiates except `LocalEvaluatorProvider`.
    Is a non-MPI runner planned? It would make items 2–8 of the test list runnable in plain pytest.
