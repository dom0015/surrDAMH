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
| `Hamiltonian` | `num_steps` | `int` | `10` | Leapfrog steps per proposal. Never tuned. | **yes** |
| `Hamiltonian` | `mass` | `float\|ArrayLike` | `1.0` | Mass matrix: scalar, per-parameter vector, or matrix; `1.0` is the natural choice for the standard-normal internal prior. Never tuned, and **not** carried over from an adaptive random-walk stage's covariance: `16` §5 item 5 measured `M = Σ̂` as a poor mass (`M = Σ̂⁻¹` is the good one), and carrying the inverse needs the step size re-adapted at fixed integration time, which is an open decision — set the mass explicitly. | **yes** |
| `Hamiltonian` | `integrator` | `"leapfrog"\|"dimension_robust"` | `"leapfrog"` | `"leapfrog"` = standard HMC; `"dimension_robust"` = the prior part of the dynamics is solved exactly (harmonic-oscillator rotation) and only the likelihood gradient is integrated numerically — the dimension-robust "infinite-dimensional" HMC scheme (formerly the separate `"HamiltonianInfinite"` proposal type), recommended for a standard-normal internal prior. | **yes** |
| `Hamiltonian` | `adaptive` | `bool\|None` | `None` | Tune `step_size` online (dual averaging); `None` = `True` iff `step_size` is `None`. `mass` and `num_steps` are never tuned. | **yes** |
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

**DAMH target-rate semantics.** For `RandomWalk` and `PCN` the feedback is the *overall*
acceptance of an outer iteration — the exact-posterior acceptance probability of the sub-chain
endpoint when the sub-chain moved, and **0** on a pre-rejected iteration (`Proposal.adapt` is
called on every outer iteration since 2026-09-20). Scoring only the moved iterations feeds back
the conditional second-stage rate, which tends to 1 as the surrogate improves and made the scale
diverge (`library_notes/16_adaptivity_options_research_2026-09-20.md` §2 item 1). For the
`Hamiltonian` family the step size is instead dual-averaged on the **sub-chain's own acceptance
against the surrogate**, one update per sub-chain step: the leapfrog integrates the surrogate
gradient field, and the outer DAMH rate is flat over a 20x step range (`16` §5 item 4, `15`
§2.7). At `subchain_length > 1` the `RandomWalk`/`PCN` target scores the whole multi-step move,
so the per-step scale ends above its own optimum (≈ 3.3x at `subchain_length=5` on both examples
of `toy_examples/out_damh_adapt_fix_2026-09-20/`) — prefer `subchain_length=1` in an adaptive
DAMH stage. Cross-check `subchain_acc_rate`/`outer_acc_given_move` in the run summary and the
per-period `adaptive_stats/` trace (`docs/outputs.md`).

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

## Cross-stage carry-over of an adaptive stage

At the end of a stage whose `proposal.adaptive` is `True` every sampler rank exchanges the
*sufficient statistics* of its adaptation (`Proposal.adapted_state()`, a fixed-length float64
vector) with an `Allgather`, combines the same pooled state (`set_pooled_state`) and derives the
same carry-over (`carry_over()`), which is merged into a `dict` keyed by the spec's step-field
name (`surrDAMH.proposals.STEP_FIELD`) and consumed by every later stage for the fields it
leaves `None`:

| proposal spec | runtime class | pooled | carried field |
|---|---|---|---|
| `RandomWalk(adaptive=True)` | `GaussRandomWalk_adaptive` | `(n, mean, M2)` of all chains' states (Chan et al. parallel combination) + mean effective log-scale `log σ + ½ log tr(base_cov)` (converted back to `log σ` against the pooled base, 2026-09-21) | `scale` |
| `PCN(adaptive=True)` | `PCN_adaptive` | mean `logit β` | `beta` |
| `Hamiltonian(adaptive=True)` | `Hamiltonian_adaptive` (`integrator="leapfrog"`) or `HamiltonianInfinite_adaptive` (`integrator="dimension_robust"`) | mean `log ε̄` | `step_size` |

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
