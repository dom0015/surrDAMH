# `Stage` reference

`surrDAMH.stages.Stage` (`surrDAMH/stages.py`): one algorithm + proposal + stopping
rule. `list_of_stages` is run in order by every sampler rank (or by `run_local`);
`stage_name(stage, i)` derives its output-directory name (`alg%04d_<MH|MH-adaptive|DAMH|DAMH-SMU>`).

| Field | Type | Default | Meaning | Posterior-affecting? |
|---|---|---|---|---|
| `algorithm_type` | `"MH"\|"DAMH"` | `"MH"` | MH = plain Metropolis-Hastings; DAMH = delayed acceptance (see `docs/concepts.md`). | **yes** |
| `proposal_type` | `"RWMH"\|"pCN"\|"Hamiltonian"\|"HamiltonianInfinite"\|"block"` | `"RWMH"` | Which `Proposal` `build_proposal` constructs. | **yes** |
| `proposal` | `Proposal\|None` | `None` | **Never read.** `build_proposal` always constructs a fresh proposal from `proposal_type` and the fields below; setting this has no effect. | ignored |
| `proposal_sd_or_cov` | `float\|ArrayLike\|None` | `None` | Proposal scale/covariance (RWMH, adaptive RWMH, Hamiltonian mass); `None` carries over the previous adaptive random-walk stage's covariance (RWMH stages only). With nothing to carry over, an **adaptive** RWMH stage starts from the prior covariance × `2.38²/d` (sd `1.0` + warning if the prior has no `get_covariance()`) — the adaptive random walk keeps its effective covariance continuous when its estimate replaces the start, so the start only shapes the warm-up (2026-09-21); a non-adaptive RWMH stage raises. A **Hamiltonian** stage with `None` gets the mass `1.0`, never the carried covariance: `16` §5 item 5 measured `M = Σ̂` as a poor mass (`M = Σ̂⁻¹` is the good one), and carrying the inverse needs the step size re-adapted at fixed integration time, which is an open decision — set the mass explicitly on a Hamiltonian stage. | **yes** |
| `pcn_beta` | `float\|None` | `None` | pCN step size in `(0, 1)`. Only for `proposal_type="pCN"`. `None` (since 2026-09-20) = the β carried over from the last adaptive pCN stage of the run, else `0.5` — the former dataclass default, so no existing configuration changes value. | **yes** |
| `hamiltonian_num_steps` | `int` | `10` | Leapfrog steps. Only for the two Hamiltonian proposal types. | **yes** |
| `hamiltonian_step_size` | `float\|None` | `None` | Leapfrog step size. Only for the two Hamiltonian proposal types. `None` (since 2026-09-20) = the dual-averaged `ε̄` carried over from the last adaptive Hamiltonian stage of the run, else `0.1` — the former dataclass default, so no existing configuration changes value. | **yes** |
| `block_proposal_groups` | `list[slice]\|None` | `None` | Parameter-index groups. Only for `proposal_type="block"`. | **yes** |
| `block_proposal_list` | `list[Proposal]\|None` | `None` | One sub-proposal per group. Only for `proposal_type="block"`. | **yes** |
| `adaptive` | `bool` | `False` | Use the adaptive variant of this stage's proposal: `GaussRandomWalk_adaptive` (shrinkage-regularised Haario covariance + Robbins–Monro log-scale), `PCN_adaptive` (Robbins–Monro on the logit of β) or `Hamiltonian_adaptive`/`HamiltonianInfinite_adaptive` (dual-averaged leapfrog step size; the mass and `hamiltonian_num_steps` are **not** adapted). Supported for all four since 2026-09-20 — pCN is no longer forced back to `False`. `proposal_type="block"` still raises; adapt the sub-proposals instead. The adapted state is pooled across MPI sampler ranks at the end of the stage and handed to the later stages through the fields they leave `None`. Note for an **MH** stage with a Hamiltonian proposal: the only acceptance statistic available there is the exact-model one, which also pays for the *surrogate gradient field's* error, so an under-trained surrogate pushes the adapted step size down (measured: 0.1 → 0.024 on the `tests/mpi` B12 case) and the stage mixes slowly while it warms up — give such a stage enough evaluations, or freeze the step size in a following stage with `hamiltonian_step_size=None`. | **yes** |
| `max_samples` | `int` | `sys.maxsize` | Stop after this many outer iterations. | no (chain length only) |
| `max_evaluations` | `int` | `sys.maxsize` | Stop after this many full-model evaluations. If none of the three stopping fields is set, defaults to 10 with a printed notice. | no (chain length only) |
| `time_limit` | `float` | `inf` | Stop after this many seconds. | no (chain length only) |
| `send_snapshots_to_collector` | `bool` | `True` | Whether this stage's evaluations feed the surrogate's training data. Forced to `False` when `use_only_surrogate=True`. | affects surrogate training data, not this stage's own posterior |
| `subchain_max_length` | `int` | `1` | DAMH sub-chain length (surrogate-only MH steps per outer iteration). | **yes** (DAMH only) |
| `surrogate_model_updates` | `bool\|None` | `None` | Keep picking up newer surrogate evaluators during the stage. Tri-state on input, always a plain `bool` after `__post_init__`: `None` = "what this stage type has always done", i.e. `True` for DAMH (DAMH-SMU: the sub-chain surrogate is refreshed once per sub-chain) and `False` for MH. Since WS7 (2026-09-18) an **MH** stage whose proposal needs surrogate gradients (`"Hamiltonian"`/`"HamiltonianInfinite"`, or a `"block"` proposal containing such a sub-proposal) may set it to `True`: the stage then polls the collector once per iteration and re-installs the proposal's gradient functions whenever a newer evaluator arrives, instead of keeping the one it fetched at stage start. The chain stays exact either way — an MH accept/reject test uses the exact model only, so any gradient field gives a valid MH kernel; only mixing/efficiency changes. On any other MH stage `True` is refused with a printed warning and resolved to `False` (before WS7 that happened silently). The stage directory name is unaffected (`_DAMH-SMU` remains a DAMH-only suffix; an opted-in MH stage is still `alg%04d_MH`). | **yes** (DAMH kernel; MH: proposal gradients only) |
| `use_only_surrogate` | `bool` | `False` | Replace the exact model with the surrogate entirely for this stage (via `SurrogateAsSolver`). | **yes** |
| `save_to_file` | `bool` | `True` | Write this stage's `samples/`/`notes/` CSVs. | no (output only) |
| `is_excluded` | `bool` | `False` | If `True`, the next stage starts from this stage's final sample without re-deriving it as a fresh start (see `algorithms.sample_carried_to_next_stage`). | **yes** (affects what the next stage starts from) |
| `adaptive_target_rate` | `float\|None` | `None` | Target acceptance rate of the adaptive proposal. `None` → the proposal class's own default: **0.234** for `GaussRandomWalk_adaptive` and `PCN_adaptive` (the high-dimensional RWM optimum; `16` §5 items 2/6), **0.8** for the dual-averaged Hamiltonian step size (`16` §5 item 4 — 0.65 overshot by ~2× into the `εL ≈ 2π` resonance in the prototype). **DAMH semantics:** for RWMH and pCN the feedback is the *overall* acceptance of an outer iteration — the exact-posterior acceptance probability of the sub-chain endpoint when the sub-chain moved, and **0** on a pre-rejected iteration (`Proposal.adapt` is called on every outer iteration since 2026-09-20). Scoring only the moved iterations feeds back the conditional second-stage rate, which tends to 1 as the surrogate improves and made the scale diverge (`library_notes/16_adaptivity_options_research_2026-09-20.md` §2 item 1). For the Hamiltonian family the step size is instead dual-averaged on the **sub-chain's own acceptance against the surrogate**, one update per sub-chain step: the leapfrog integrates the surrogate gradient field, and the outer DAMH rate is flat over a 20× step range (`16` §5 item 4, `15` §2.7). At `subchain_max_length > 1` the RWMH/pCN target scores the whole multi-step move, so the per-step scale ends above its own optimum (≈ 3.3× at `subchain_max_length=5` on both examples of `toy_examples/out_damh_adapt_fix_2026-09-20/`) — prefer `subchain_max_length=1` in an adaptive DAMH stage. Cross-check `subchain_acc_rate`/`outer_acc_given_move` in the run summary and the per-period `adaptive_stats/` trace (`docs/outputs.md`). | **yes**, when set together with `adaptive=True` |
| `name` | `str\|None` | `None` | Set by the runner (`stage_name`), not by the user. | n/a |

## Invalid combinations that raise

Raised at `build_proposal`/`stage_name` time — i.e. when `SamplingFramework.run()` or
`run_local()` actually reaches this stage, not eagerly at `Stage()` construction:

- Unknown `algorithm_type` (anything other than `"MH"`/`"DAMH"`) — `ValueError` from `stage_name`, and it now fires on every rank at start-up (stage names are assigned before role dispatch) rather than only when a sampler reaches that stage.
- A DAMH or Hamiltonian-family (`"Hamiltonian"`/`"HamiltonianInfinite"`, or `"block"` containing one) **first** stage with no evaluator available yet — see `docs/configuration.md`'s "DAMH without a collector" table; without a collector-side start-up handshake success or a fixed `surrogate_evaluator`, this deadlocks (or now raises, once the WS8 handshake covers it — see `library_notes/10_manual_review_notes.md` §2.11).
- `proposal_type="pCN"` with a prior that is not `Normal`/`PriorIndependentComponents` — `ValueError` from `build_proposal` (pCN's acceptance ratio drops the prior term, which is only valid for a Gaussian internal prior).
- A Hamiltonian-family proposal (bare or inside a `"block"`) without `Configuration.use_surrogate_gradients=True` in effect — `AssertionError` from `build_proposal`.
- `proposal_type="block"` with `adaptive=True` — `ValueError` from `build_proposal` (adapt the sub-proposals instead).
- `adaptive_corr_limit=` / `adaptive_sample_limit=` — `TypeError` from `Stage(...)`: both fields were removed with the adaptive random walk's internals on 2026-09-20 (no compatibility shim).

Since 2026-09-20 `proposal_type="pCN"` with `adaptive=True` is **supported** (it used to be
silently forced back to `adaptive=False`).

## Cross-stage carry-over of an adaptive stage

At the end of an `adaptive=True` stage every sampler rank exchanges the *sufficient
statistics* of its adaptation (`Proposal.adapted_state()`, a fixed-length float64 vector) with
an `Allgather`, combines the same pooled state (`set_pooled_state`) and derives the same
carry-over (`carry_over()`), which is merged into a `dict` keyed by `Stage` field name and
consumed by every later stage for the fields it leaves `None`:

| proposal | pooled | carried field |
|---|---|---|
| `GaussRandomWalk_adaptive` | `(n, mean, M2)` of all chains' states (Chan et al. parallel combination) + mean effective log-scale `log σ + ½ log tr(base_cov)` (converted back to `log σ` against the pooled base, 2026-09-21) | `proposal_sd_or_cov` |
| `PCN_adaptive` | mean `logit β` | `pcn_beta` |
| `Hamiltonian_adaptive` / `HamiltonianInfinite_adaptive` | mean `log ε̄` | `hamiltonian_step_size` |

Before 2026-09-20 the ranks averaged their per-rank *covariances* instead, which the study
measured disagreeing by 6–17× between ranks (`15` §2.4). `run_local` runs the same code path
with a single row, which keeps its own adapted state exactly.

## `Stage.describe()`

`Stage.describe(index)` returns a short block listing every field with its **effective**
value (after `__post_init__`'s silent corrections), marking the posterior-affecting ones
with `*` and appending `[note]` lines for the fields that do not apply to the stage as
configured. `SamplingFramework.run()` and `run_local()` print one block per stage on rank 0
at start-up — see [`configuration.md`](configuration.md#configurationdescribe--stagedescribe).
