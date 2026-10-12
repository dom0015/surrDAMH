# `Stage` reference

`surrDAMH.stages.Stage` (`surrDAMH/stages.py`): one algorithm + proposal + stopping
rule. `list_of_stages` is run in order by every sampler rank (or by `run_sampling_local`);
`stage_name(stage, i)` derives its output-directory name (`alg%04d_<MH|MH-adaptive|DAMH|DAMH-SMU>`).

Proposal settings live in their own spec objects (`surrDAMH.proposals`: `RandomWalk`, `PCN`,
`Hamiltonian`, `Block`), passed as `Stage(proposal=...)` — see the second table below.

```python
from surrDAMH.proposals import RandomWalk, PCN, Hamiltonian, Block
from surrDAMH.stages import Stage

Stage(max_evaluations=1000)                                      # MH, random walk tuned online
Stage(proposal=RandomWalk(scale=0.5), max_evaluations=1000)      # MH, fixed step
Stage(algorithm="DAMH", subchain_length=5, max_evaluations=200)  # surrogate-accelerated
Stage(proposal=PCN(beta=0.2), time_limit=60)
```

## `Stage` fields

| Field | Type | Default | Meaning | Posterior-affecting? |
|---|---|---|---|---|
| `algorithm` | `"MH"\|"DAMH"` | `"MH"` | MH = plain Metropolis-Hastings; DAMH = delayed acceptance (see `docs/concepts.md`). | **yes** |
| `proposal` | `RandomWalk\|PCN\|Hamiltonian\|Block` | `RandomWalk()` | How new samples are proposed; a spec from `surrDAMH.proposals`. Each carries its own step (`scale`/`beta`/`step_size`) and `adaptive` switch — see the "Proposal specs" table below. | **yes** |
| `max_evaluations` | `int` | `sys.maxsize` | Stop after this many full-model (exact) evaluations. | no (chain length only) |
| `max_samples` | `int` | `sys.maxsize` | Stop after this many samples (outer-iteration chain states). If none of `max_evaluations`/`max_samples`/`time_limit` is set, `max_evaluations` defaults to 10 with a printed notice. | no (chain length only) |
| `time_limit` | `float` | `inf` | Stop after this many seconds. | no (chain length only) |
| `subchain_length` | `int` | `1` | DAMH only: surrogate-only MH steps between two exact-model evaluations. | **yes** (DAMH only) |
| `surrogate_model_updates` | `bool\|None` | `None` | Keep picking up newer surrogate evaluators during the stage. Tri-state on input, always a plain `bool` after `__post_init__`: `None` = "what this stage type has always done", i.e. `True` for DAMH (DAMH-SMU: the sub-chain surrogate is refreshed once per sub-chain) and `False` for MH. Since WS7 (2026-09-18) an **MH** stage whose proposal needs surrogate gradients (a `Hamiltonian` proposal, bare or inside a `Block`) may set it to `True`: the stage then polls the collector once per iteration and re-installs the proposal's gradient functions whenever a newer evaluator arrives, instead of keeping the one it fetched at stage start. The chain stays exact either way — an MH accept/reject test uses the exact model only, so any gradient field gives a valid MH kernel; only mixing/efficiency changes. On any other MH stage `True` is refused with a printed warning and resolved to `False` (before WS7 that happened silently). The stage directory name is unaffected (`_DAMH-SMU` remains a DAMH-only suffix; an opted-in MH stage is still `alg%04d_MH`). | **yes** (DAMH kernel; MH: proposal gradients only) |
| `use_only_surrogate` | `bool` | `False` | Replace the exact model with the surrogate entirely for this stage (via `SurrogateAsSolver`) — exploration only, the samples do not follow the posterior. | **yes** |
| `save_to_file` | `bool` | `True` | Write this stage's `samples/`/`notes/` CSVs. | no (output only) |
| `send_snapshots_to_collector` | `bool` | `True` | Whether this stage's exact-model evaluations feed the surrogate's training data. Forced to `False` when `use_only_surrogate=True`. | affects surrogate training data, not this stage's own posterior |
| `is_excluded` | `bool` | `False` | Discarded-exploration flag: if `True`, this stage's final sample is thrown away and the next stage **restarts** from the sample this stage started at (`process_SAMPLER`/`runner_local`: `sample_carried_to_next_stage` is skipped for an excluded stage). Also sets this stage's chains to 0 by default in the report's `selection.json` ("posterior"
entry, `docs/outputs.md`). Refused together with `burn_in=True` (`ValueError`). | **yes** (what the next stage starts from, and the default posterior selection) |
| `burn_in` | `bool` | `False` | Warm-up flag (2026-10-09): if `True`, the chain **continues** into the next stage exactly as after a normal stage — only this stage's chains get 0 by default in the report's `selection.json` ("posterior" entry, `docs/outputs.md`), so its samples are left out of the posterior unless the user edits the file. Refused together with `is_excluded=True` (`ValueError`). | **yes** (the default posterior selection) |
| `exact_step_probability` | `float` in `[0, 1]` | `0.0` | DAMH only (S2, 2026-10-09): probability that an outer iteration is an **exact** MH step instead of the delayed-acceptance step. The exact step has its own adaptive random walk (`RandomWalk()`, started from the carried `scale` if an earlier adaptive random-walk stage handed one over, seeded `seed0 + 5`, adapted on its own exact acceptances, not pooled across ranks, not carried over, no `adaptive_stats` trace); its proposal is scored with the installed surrogate first (both observation blocks in `raw_data`), then with the exact model, and accepted with the exact likelihoods. The stage proposal is **not** adapted on exact iterations (a Hamiltonian stage proposal's momenta are untouched). Both kernels keep the exact posterior invariant, so the mixture does too, for any surrogate — and only the exact step can move the chain into a region the surrogate wrongly rules out (note 22's blind spot), also in a frozen stage. Cost: one exact evaluation per exact step (counted in `accepted`/`rejected` and towards `max_evaluations`). The "which kernel" draw uses its own stream (`seed0 + 4`), drawn from only when the field is > 0. On an MH stage a non-zero value is reset to 0 with a printed warning; outside `[0, 1]` → `ValueError`. Off for explicit stage lists; Auto sets 0.05 on its DAMH chunks. | **yes** |
| `audit_prerejected` | `float` in `[0, 1]` | `0.0` | DAMH only (S2, 2026-10-09): probability that a **pre-rejected** iteration's last surrogate-rejected proposal is evaluated with the exact model anyway. The decision is not revisited — the chain's law is unchanged — but the point is written to `raw_data` (`state_type="audited"`, both observation blocks, the exact log-likelihood) and sent to the collector as a snapshot with multiplicity 0 (not with `send_snapshots_to_collector=False`), so a surrogate that hides a region of high posterior gets corrected; `notes` counts `audited` and `audit_hidden` (audits with `log L − log L~ > 5` nats). Non-finite and out-of-box sub-chain proposals are never candidates. Cost: one exact evaluation per audit (towards `max_evaluations`). Shares the `seed0 + 4` stream with the exact-step draw. On a frozen stage (`surrogate_model_updates=False`) it is kept, with a start-up note that it only measures there; on an MH stage it is reset to 0 with a warning. Auto sets 0.05 on its DAMH chunks. | **yes** (acceptance/surrogate training; not the chain's law) |
| `exact_step_scale` | `float > 0` or `None` | `None` | DAMH only, used only with `exact_step_probability > 0` (study 29, 2026-10-12). `None` = the exact step's random walk adapts as described for `exact_step_probability`. A number `s` = that random walk is **non-adaptive** with covariance `s * (2.38^2/d) * Sigma_hat`, `Sigma_hat` = the chain's pooled posterior-statistics covariance estimate at stage start (identity, with a note, while fewer than 100 states have been seen). Recorded per stage in `run_manifest.json` `proposal_resolution` (`exact_step_scale`, `exact_step_covariance_source`, `exact_step_covariance_trace`). Reset to `None` with a warning on an MH stage. | **yes** (acceptance) |
| `polyak_averaging` | `bool` | `False` | S3 (2026-10-09): install, instead of each received surrogate evaluator, the running **mean** of the parameters of the evaluators this chain received in this stage (weight `1/k` for the `k`-th, restarted at every stage), so the installed surrogate changes by `O(1/k)` per update — the diminishing-adaptation condition (A5) of the ergodicity argument for on-the-fly retraining (`library_notes/18_*` §6 N1, `20_*` §6). Meaningful on a stage that receives evaluators (DAMH, or MH with a gradient proposal and `surrogate_model_updates=True`); ignored elsewhere. Only evaluators with `averageable_state()` (the `NeuralNetworkUpdater` network) are averaged; others are installed unchanged with one printed note per stage. The averaged evaluator is what is checked for finiteness, what the sub-chain, the exact step's pre-scoring and the Hamiltonian gradients use; the delayed-acceptance correction stays exact because the installed surrogate still changes only between sub-chains. Each chain averages what IT received — chains may receive different subsets of the collector's versions, so their averages can differ. `notes` column `polyak_updates` = `k` at stage end. Auto sets `True` on its DAMH chunks. | **yes** (surrogate used; not the chain's law for a fixed surrogate) |
| `surrogate_error_model` | `bool` | `False` | S1 (2026-10-09): use the global Gaussian approximation-error model of the surrogate. Every **surrogate** log-likelihood of the stage — the DAMH sub-chain, the delayed-acceptance correction, `log_likelihood_approx` in `raw_data`, the Hamiltonian potential and its gradient (`J~ᵀ Σ_tot⁻¹ (G~ + ε̄ − y)`), the S2 exact step's surrogate scoring — becomes `N(y; G~(u) + ε̄, Σ_noise + Σ_surr)`, where `ε̄`/`Σ_surr` are the mean/covariance of the residuals `G − G~` of the installed evaluator, estimated by the collector (or the local surrogate manager) after every retraining on all the updater's snapshots plus the held-out set (`surrogate_test_data`), Ledoit–Wolf shrinkage up to 50 observations, diagonal above (`surrDAMH.modules.error_model`, `Evaluator.error_model`). The exact likelihood is never changed and the correction uses the same corrected surrogate likelihood, so the chain still targets the exact posterior; what changes is which proposals pass the surrogate stage (a biased or over-confident surrogate pre-rejects less wrongly). Needs a `Normal` likelihood — otherwise one printed note and the flag is ignored; an evaluator without a model (e.g. a user-given fixed evaluator) is used uncorrected with one note per stage. With `polyak_averaging` the error model is averaged together with the network. Auto sets `True` on its DAMH chunks. | **yes** (acceptance; not the chain's target) |
| `name` | `str\|None` | `None` | Set by the library (`stage_name`), not by the user — the stage's output-directory name. | n/a |

**`burn_in` vs. `is_excluded`.** Both flags only change the report's *default* selection — they
set the stage's entry in `selection.json`'s `posterior` to 0 for every chain (the user may still flip it back
to `1` by hand, see `docs/outputs.md`) — and both are refused together (`ValueError`). They differ
in what happens to the chain itself: `burn_in=True` is a warm-up — the chain **continues** into the
next stage exactly as after a normal stage, only the samples are excluded from the posterior by
default. `is_excluded=True` is discarded exploration — the next stage **restarts** from the sample
this stage itself started at, throwing away everything the stage did. Both are in
`POSTERIOR_AFFECTING_FIELDS` and so are marked `*` in `Stage.describe()`.

`Stage.adaptive` is a **read-only property**, not a field: `bool(getattr(self.proposal,
"adaptive", False))`, i.e. it mirrors `proposal.adaptive` and is always `False` for a `Block`
(a `Block` has no `adaptive` field/kwarg at all — see below). There is no `Stage(adaptive=...)`
kwarg any more; set `adaptive=` on the proposal spec instead.

## Proposal specs (`surrDAMH.proposals`)

Small, picklable descriptions of *which* proposal a stage uses and with which settings. The
sampler turns a spec into the matching runtime class of `surrDAMH.modules.proposals`
(`modules.proposal_builder.build_proposal`), one fresh object per stage and per chain. A spec
holds no random state and can be reused across several stages.

The step field of every proposal (`scale` / `beta` / `step_size`) is optional. Leave it `None`
and the stage uses the value tuned by the previous adaptive stage of the same proposal type,
else that proposal's own default; `adaptive` then defaults to `True` ("adapt unless the step is
pinned"). Give the step and the proposal is fixed unless you also pass `adaptive=True`, which
then tunes it starting from the given value.

| Spec | Field | Type | Default | Meaning | Posterior-affecting? |
|---|---|---|---|---|---|
| `RandomWalk` | `scale` | `float\|ArrayLike\|None` | `None` | Step in the prior's internal space: a scalar sd, a vector of per-parameter sds, or a covariance matrix. `None`: the covariance tuned by the previous adaptive random-walk stage, else the prior covariance scaled by `2.38²/d` (sd `1.0` + warning if the prior has no `get_covariance()`). | **yes** |
| `RandomWalk` | `adaptive` | `bool\|None` | `None` | Tune `scale` online (shrinkage-regularised Haario covariance estimate + Robbins–Monro log-scale). `None` = `True` iff `scale` is `None`. `adaptive=True` with a `scale` starts tuning from that value; `adaptive=False` without a `scale` runs the whole stage at the default (a warning is printed). | **yes** |
| `RandomWalk` | `target_rate` | `float\|None` | `None` | Acceptance rate the tuning aims at; `None` = `0.234` (the high-dimensional RWM optimum). | **yes**, when `adaptive=True` |
| `PCN` | `beta` | `float\|None` | `None` | Step in `(0, 1)`. `None`: the value tuned by the previous adaptive pCN stage, else `0.5`. Needs a Gaussian internal prior (`Normal`/`PriorIndependentComponents`). | **yes** |
| `PCN` | `adaptive` | `bool\|None` | `None` | Tune `beta` online (Robbins–Monro on the logit of `β`); `None` = `True` iff `beta` is `None`. | **yes** |
| `PCN` | `target_rate` | `float\|None` | `None` | `None` = `0.234`. | **yes**, when `adaptive=True` |
| `Hamiltonian` | `step_size` | `float\|None` | `None` | Leapfrog step; `None`: the value tuned by the previous adaptive Hamiltonian stage, else `0.1`. | **yes** |
| `Hamiltonian` | `num_steps` | `int\|None` | `None` (= `10`) | Leapfrog steps per proposal; `None` = `10` unless `integration_time` is given (both given → `ValueError`). Never tuned. | **yes** |
| `Hamiltonian` | `integration_time` | `float\|None` | `None` | C1 (2026-10-11): trajectory length `T` in time instead of `num_steps`: `num_steps = max(1, round(T / step_size))` (at most 1000), recomputed whenever the dual averaging changes the step, so the trajectory length stays `T` while the step adapts (`16` §5 item 4). The adaptation trace gains a `num_steps` column (appended, only for such a stage). Auto's fast mode uses `T = 3.0` with jitter 0.5 (placeholder; author decision 2026-10-11, C1b evidence: better than π/2 on both means and second moments). | **yes** |
| `Hamiltonian` | `integration_time_jitter` | `float` in `[0, 1)` | `0.0` | C1b (2026-10-11), only with `integration_time` (`> 0` without it → `ValueError`): every trajectory draws `u ~ Uniform(1 − j, 1 + j)` from the proposal's own generator (before its momentum) and integrates `max(1, round(T u / step_size))` steps (same cap); the trace's `num_steps` column stays the nominal `round(T / step)`. `0.0` draws nothing (bit-identical to a stage without the field). Valid: the length is drawn independently of the state, so each trajectory is a reversible kernel and so is the mixture. Removes the antithetic second-moment deficit of a fixed `T` (CHANGELOG 2026-10-11, C1b: 10-D, `T = 3`: ESS/eval of `x²` 0.109 → 0.318). Auto's fast mode uses `0.5`. | **yes** |
| `Hamiltonian` | `mass` | `float\|ArrayLike\|"covariance"` | `1.0` | Mass matrix `M` (momenta `p ~ N(0, M)`, kinetic energy `½ pᵀM⁻¹p`): scalar or per-parameter vector (`M = diag(value²)`) or a full SPD matrix; `1.0` is the natural choice for the standard-normal internal prior. `"covariance"` (C1, 2026-10-11): `M = Σ̂⁻¹`, the regularised precision of the chain's **posterior statistics** (see "Posterior statistics" below: all previous stages and runs, pooled over the chains, shrinkage + relative ridge as for the random walk), resolved at the start of the stage and **fixed within it** (no new adaptation theory); the identity with a printed note while the statistics hold fewer than 100 states (`MASS_MIN_STATES`). `16` §5 item 5: `M = Σ̂` is a poor mass, `M = Σ̂⁻¹` the good one. Both integrators take a full matrix (the dimension-robust one rotates each eigen-coordinate of `M` with angular frequency `1/√λᵢ`). The resolved source is recorded in `run_manifest.json` (`proposal_resolution`). | **yes** |
| `Hamiltonian` | `integrator` | `"leapfrog"\|"dimension_robust"` | `"leapfrog"` | `"leapfrog"` = standard HMC; `"dimension_robust"` = the prior part of the dynamics is solved exactly (harmonic-oscillator rotation) and only the likelihood gradient is integrated numerically — the dimension-robust "infinite-dimensional" HMC scheme (formerly the separate `"HamiltonianInfinite"` proposal type), recommended for a standard-normal internal prior. | **yes** |
| `Hamiltonian` | `adaptive` | `bool\|None` | `None` | Tune `step_size` online (dual averaging); `None` = `True` iff `step_size` is `None`. `mass` and `num_steps` are never tuned (with `integration_time`, `num_steps` follows the step). With `step_size=None` and a resolved mass within `DA_CONTINUE_MASS_TOLERANCE = 0.2` of the previous adaptive Hamiltonian stage's (relative spectral-norm change `r = ‖M_new − M_old‖₂ / ‖M_old‖₂`, C1b 2026-10-11; C1 required an identical mass), the dual averaging **continues** that stage's recursion (`m`, `H̄`, `log ε̄`, `μ` carried; no restart at `μ = log(10 ε₀)`); after a larger change it restarts from the carried step **rescaled to the new mass**, `ε₀ = ε_carried · (det M_new / det M_old)^(1/(2d))` (for `M = Σ̂⁻¹`: `(det Σ̂_old / det Σ̂_new)^(1/(2d))` — a larger covariance estimate means a lighter mass and a smaller step), with the small offset `μ = log(DA_RESTART_FACTOR · ε₀)`, `DA_RESTART_FACTOR = 2` (placeholders); printed note, branch and `r` in `proposal_resolution`. | **yes** |
| `Hamiltonian` | `target_rate` | `float\|None` | `None` | `None` = `0.8` (`16` §5 item 4 — `0.65` overshot by ~2x into the `εL ≈ 2π` resonance in the prototype). | **yes**, when `adaptive=True` |
| `Block` | `groups` | `list[list[int]]` | `[]` (must be non-empty) | Parameter indices of each group, e.g. `[[0, 1], [2, 3, 4]]`; the groups must cover all parameters without overlap. | **yes** |
| `Block` | `proposals` | `list[RandomWalk\|PCN\|Hamiltonian]` | `[]` (must match `groups` in length) | One proposal spec per group. Their steps must be given — a block proposal does not adapt; `adaptive=True` on a sub-proposal raises (see "Invalid combinations" below). A `Block` cannot contain another `Block`. | **yes** |

`Block.adaptive` is a **class attribute fixed at `False`**, not a dataclass field: a `Block`
never adapts and has no `adaptive` kwarg to pass in the first place (`Block(adaptive=True)`
raises a plain `TypeError` from the dataclass constructor, not a `ValueError` about adaptivity
— see below).

Note for an **MH** stage with a `Hamiltonian` proposal: the only acceptance statistic available
there is the exact-model one, which also pays for the *surrogate gradient field's* error, so an
under-trained surrogate pushes the adapted step size down (measured: 0.1 → 0.024 on the
`tests/mpi` B12 case) and the stage mixes slowly while it warms up — give such a stage enough
evaluations, or freeze the step size in a following stage with `Hamiltonian(step_size=None)`
(fixed, i.e. `adaptive=False`, carrying the previous value).

**DAMH target-rate semantics.** Every adaptive proposal adapts its step on the **sub-chain's
own per-step acceptance against the surrogate** — the acceptance of the moves it actually
generates — with one update per sub-chain step, on every outer iteration (pre-rejected ones
included): `RandomWalk` and `PCN` run their Robbins–Monro recursion (target 0.234) once per
sub-chain step (since 2026-10-10, S6), the `Hamiltonian` family dual-averages its step size on the
same probabilities. The random walk's covariance SHAPE is still estimated once per outer
iteration from the chain state. The outer DAMH acceptance is reported but not targeted: it
measures the surrogate error, not the step — scoring only moved iterations tends to 1 and made
the scale diverge (`library_notes/16_adaptivity_options_research_2026-09-20.md` §2 item 1), and
scoring the whole multi-step move (the rule from 2026-09-20 to S6) left the per-step random-walk
step 1.5× / 6.7× too large at `subchain_length` 5 / 20 and drove pCN's β to 1 (CHANGELOG
2026-10-10). The adapted step is therefore the same for any `subchain_length`, which becomes a
separate cost knob. Cross-check `subchain_acc_rate`/`outer_acc_given_move` in the run summary and
the per-period `adaptive_stats/` trace (`docs/outputs.md`).

## Invalid combinations that raise

Raised at `build_proposal`/`stage_name` time — i.e. when `Problem.run_sampling()` or
`run_sampling_local()` actually reaches this stage, not eagerly at `Stage()`/spec construction:

- Unknown `algorithm` (anything other than `"MH"`/`"DAMH"`) — `ValueError` from `stage_name`,
  and it fires on every rank at start-up (stage names are assigned before role dispatch) rather
  than only when a sampler reaches that stage.
- A DAMH or Hamiltonian-proposal (bare or inside a `Block`) **first** stage with no evaluator
  available yet — see `docs/configuration.md`'s "DAMH without a collector" table; without a
  collector-side start-up handshake success or a fixed `surrogate_evaluator`, this deadlocks (or
  now raises, once the WS8 handshake covers it — see
  `library_notes/10_manual_review_notes.md` §2.11).
- `PCN` with a prior that is not `Normal`/`PriorIndependentComponents` — `ValueError` from
  `build_proposal` (pCN's acceptance ratio drops the prior term, which is only valid for a
  Gaussian internal prior).
- A `Hamiltonian` proposal (bare or inside a `Block`) without
  `Configuration.use_surrogate_gradients=True` in effect — `AssertionError` from
  `build_proposal`.
- A `RandomWalk`/`PCN`/`Hamiltonian` sub-proposal inside a `Block` with `adaptive=True` —
  `ValueError` from `build_proposal` (`surrDAMH/modules/proposal_builder.py`, the `Block`
  branch: it checks `sub_spec.adaptive` for each sub-proposal before building it; give every
  sub-proposal an explicit step instead). This check is **not** in `Block.__post_init__` itself
  — that only validates that `groups`/`proposals` have matching, non-empty length and that no
  sub-proposal is itself a `Block`. Note also that a bare sub-proposal spec with no step given
  (e.g. `RandomWalk()` inside a `Block`) resolves its own `adaptive` to `True` in its own
  `__post_init__` regardless of being inside a `Block` — so it passes construction and only
  fails later, at `build_proposal` time, when the stage actually runs.
- `Block(..., adaptive=True)` — a **different** error at a **different** (construction) time
  from the previous item: `Block` has no `adaptive` field/kwarg at all (`adaptive = False` is a
  plain class attribute, not a dataclass field), so this raises `TypeError: __init__() got an
  unexpected keyword argument 'adaptive'` immediately, not a `ValueError` from `build_proposal`.
- Any field name from the pre-2026-09-21 `Stage` API (the proposal-related kwargs that moved
  into `surrDAMH.proposals`, or `adaptive_corr_limit=`/`adaptive_sample_limit=`) passed to
  `Stage(...)` or a proposal spec — all raise `TypeError: unexpected keyword argument` (no
  compatibility shim); use the new proposal specs instead.

## Posterior statistics (C1, 2026-10-11)

Every chain owns ONE persistent estimate of the posterior mean and covariance
(`surrDAMH.modules.posterior_statistics.PosteriorStatistics`: Welford `(n, mean, M2)` of the
post-decision chain states in the internal space). The algorithm updates it once per outer
iteration of **every** stage, whatever the proposal (MH, DAMH delayed-acceptance and pre-rejected
iterations, S2 exact steps); at every stage end the samplers pool it with one `Allgather` of
each chain's INCREMENT (the states of this stage) and install `start ⊕ increment_1 ⊕ … ⊕ increment_R`
(Chan–Golub–LeVeque combination, identical on every rank; every state of every chain is counted
exactly once — fixed 2026-10-11, before which the shared start was counted once per sampler at
every stage end) and the pooled object goes on to the next
stage, into `carry_over/<stage>.npz` (`carry_over__posterior_statistics`) and, through
`continue_sampling` / `continue_sampling_local`, to continued runs (`chains="prior"`/`"lhs"` too;
`problem=` drops it with a printed note, a different problem being a different posterior). It is
never reset. Consumers: the adaptive random walk's shape (below), `Hamiltonian(mass="covariance")`
and (C1b, 2026-10-11) the S2 exact step's own random walk, which reads its SHAPE from them —
immediately at construction once they hold 100 states, then at every 10th of its own steps — while
its scale keeps adapting on its own exact acceptances; it never updates them (the algorithm does,
once per outer iteration, exact steps included). This changes every stage with
`exact_step_probability > 0`.

**Burn-in discount (C1b, 2026-10-11).** At the end of a `burn_in=True` stage — after the pooling and
the adaptive hand-over, so the burn-in stage's own carried proposal is unchanged — the pooled
statistics keep their mean and covariance but only the weight of `max(100, 10 d)` states
(`M2 ← M2 (n_eff − 1)/(n − 1)`, `n ← n_eff`; `BURN_IN_STATISTICS_PER_DIMENSION = 10`,
placeholder), so the warm-up's early transient no longer dominates what the following stages build
on; recorded as `summary__posterior_statistics_discounted_from_n` in `carry_over/` and
`posterior_statistics_discounted_from_n` in `proposal_resolution`. `is_excluded` stages are not
discounted (their states are not a transient of the next stage: the chain restarts). The
estimator's shrinkage `δ = min(1, 2d/N)` uses the UNDISCOUNTED count `N = n_seen` of states ever
seen (author decision 1, 2026-10-11; carried as `carry_over__posterior_statistics_n_seen`), so the
discount does not make the following `mass="covariance"` stage's mass more isotropic.

## Cross-stage carry-over of an adaptive stage

At the end of a stage whose `proposal.adaptive` is `True` every sampler rank exchanges its
adapted state (`Proposal.adapted_state()`, a fixed-length float64 vector) with an `Allgather`,
combines the same pooled state (`set_pooled_state`) and derives the same carry-over
(`carry_over()`), which is merged into a `dict` keyed by the spec's step-field name
(`surrDAMH.proposals.STEP_FIELD`) and consumed by every later stage for the fields it leaves
`None`:

| proposal spec | runtime class | pooled | carried field |
|---|---|---|---|
| `RandomWalk(adaptive=True)` | `GaussRandomWalk_adaptive` | mean effective log-scale `log σ + ½ log tr(base_cov)` (converted back to `log σ` against the base re-installed from the pooled posterior statistics, 2026-09-21; since C1 the statistics are pooled separately, see above) | `scale` |
| `PCN(adaptive=True)` | `PCN_adaptive` | mean `logit β` | `beta` |
| `Hamiltonian(adaptive=True)` | `Hamiltonian_adaptive` (`integrator="leapfrog"`) or `HamiltonianInfinite_adaptive` (`integrator="dimension_robust"`) | mean `log ε̄` (and, C1, the mean `[m, H̄, log ε̄, μ]`) | `step_size`, plus `dual_averaging` and `hamiltonian_mass` (C1, for continuing the recursion) |

**Adaptive random walk after an earlier stage (C1, a behaviour change).** The random walk's shape
is estimated from the chain's posterior statistics, not from its own stage's states: a later
adaptive random-walk stage starts with the pooled statistics of all previous stages
(`statistics.n >= warmup` already) and installs its shape at its first period boundary (10
iterations) instead of after 100 iterations of its own. The first stage of a run starts from empty
statistics and is bit-for-bit unchanged.

Before 2026-09-20 the ranks averaged their per-rank *covariances* instead, which the study
measured disagreeing by 6–17x between ranks (`15` §2.4). `run_sampling_local` runs the same
code path with a single row, which keeps its own adapted state exactly.

## `Stage.describe()`

`Stage.describe(index)` returns a short block listing every field with its **effective**
value (after `__post_init__`'s silent corrections — the proposal spec is shown with all its own
fields, e.g. `proposal=RandomWalk(scale=None, adaptive=True, target_rate=None)`), marking the
posterior-affecting ones with `*` and appending `[note]` lines for the fields that do not apply
to the stage as configured. `Problem.run_sampling()` and `run_sampling_local()` print one block
per stage on rank 0 at start-up — see [`configuration.md`](configuration.md#configurationdescribe--stagedescribe).
