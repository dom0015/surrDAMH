# `Stage` reference

`surrDAMH.stages.Stage` (`surrDAMH/stages.py`): one algorithm + proposal + stopping
rule. `list_of_stages` is run in order by every sampler rank (or by `run_local`);
`stage_name(stage, i)` derives its output-directory name (`alg%04d_<MH|MH-adaptive|DAMH|DAMH-SMU>`).

| Field | Type | Default | Meaning | Posterior-affecting? |
|---|---|---|---|---|
| `algorithm_type` | `"MH"\|"DAMH"` | `"MH"` | MH = plain Metropolis-Hastings; DAMH = delayed acceptance (see `docs/concepts.md`). | **yes** |
| `proposal_type` | `"RWMH"\|"pCN"\|"Hamiltonian"\|"HamiltonianInfinite"\|"block"` | `"RWMH"` | Which `Proposal` `build_proposal` constructs. | **yes** |
| `proposal` | `Proposal\|None` | `None` | **Never read.** `build_proposal` always constructs a fresh proposal from `proposal_type` and the fields below; setting this has no effect. | ignored |
| `proposal_sd_or_cov` | `float\|ArrayLike\|None` | `None` | Proposal scale/covariance (RWMH, adaptive RWMH, Hamiltonian mass); `None` carries over the previous adaptive stage's covariance. | **yes** |
| `pcn_beta` | `float` | `0.5` | pCN step size in `(0, 1]`. Only for `proposal_type="pCN"`. | **yes** |
| `hamiltonian_num_steps` | `int` | `10` | Leapfrog steps. Only for the two Hamiltonian proposal types. | **yes** |
| `hamiltonian_step_size` | `float` | `0.1` | Leapfrog step size. Only for the two Hamiltonian proposal types. | **yes** |
| `block_proposal_groups` | `list[slice]\|None` | `None` | Parameter-index groups. Only for `proposal_type="block"`. | **yes** |
| `block_proposal_list` | `list[Proposal]\|None` | `None` | One sub-proposal per group. Only for `proposal_type="block"`. | **yes** |
| `adaptive` | `bool` | `False` | Use `GaussRandomWalk_adaptive` instead of the fixed-covariance proposal. Silently forced to `False` when `proposal_type="pCN"` (printed warning, not an exception). | **yes** |
| `max_samples` | `int` | `sys.maxsize` | Stop after this many outer iterations. | no (chain length only) |
| `max_evaluations` | `int` | `sys.maxsize` | Stop after this many full-model evaluations. If none of the three stopping fields is set, defaults to 10 with a printed notice. | no (chain length only) |
| `time_limit` | `float` | `inf` | Stop after this many seconds. | no (chain length only) |
| `send_snapshots_to_collector` | `bool` | `True` | Whether this stage's evaluations feed the surrogate's training data. Forced to `False` when `use_only_surrogate=True`. | affects surrogate training data, not this stage's own posterior |
| `subchain_max_length` | `int` | `1` | DAMH sub-chain length (surrogate-only MH steps per outer iteration). | **yes** (DAMH only) |
| `surrogate_model_updates` | `bool` | `True` | DAMH-SMU: surrogate keeps retraining during the stage. Forced to `False` when `algorithm_type="MH"`. | **yes** (DAMH only) |
| `use_only_surrogate` | `bool` | `False` | Replace the exact model with the surrogate entirely for this stage (via `SurrogateAsSolver`). | **yes** |
| `save_to_file` | `bool` | `True` | Write this stage's `samples/`/`notes/` CSVs. | no (output only) |
| `is_excluded` | `bool` | `False` | If `True`, the next stage starts from this stage's final sample without re-deriving it as a fresh start (see `algorithms.sample_carried_to_next_stage`). | **yes** (affects what the next stage starts from) |
| `adaptive_target_rate` | `float\|None` | `None` | Target acceptance rate of the adaptive proposal; for DAMH this is the SECOND-STAGE (outer) rate, since `Proposal.adapt` is only called once per outer step, on the sub-chain endpoint (decision 2 in `library_notes/09_improvement_plan.md` §3 is deferred — this is what the number means today). `None` → `GaussRandomWalk_adaptive`'s default `0.25`. | **yes**, when set together with `adaptive=True` (wired since G1, 2026-09-17) |
| `adaptive_corr_limit` | `float\|None` | `None` | Max \|correlation\| the adapted proposal covariance may have. `None` → default `0.3`. | **yes**, when set together with `adaptive=True` (wired since G1) |
| `adaptive_sample_limit` | `int\|None` | `None` | Keep only the last N proposals/acceptance weights when re-estimating the covariance and the current acceptance rate. `None` → unbounded history, accumulating for the whole stage. | **yes**, when set together with `adaptive=True` (wired since G1) |
| `name` | `str\|None` | `None` | Set by the runner (`stage_name`), not by the user. | n/a |

## Invalid combinations that raise

Raised at `build_proposal`/`stage_name` time — i.e. when `SamplingFramework.run()` or
`run_local()` actually reaches this stage, not eagerly at `Stage()` construction:

- Unknown `algorithm_type` (anything other than `"MH"`/`"DAMH"`) — `ValueError` from `stage_name`, and it now fires on every rank at start-up (stage names are assigned before role dispatch) rather than only when a sampler reaches that stage.
- A DAMH or Hamiltonian-family (`"Hamiltonian"`/`"HamiltonianInfinite"`, or `"block"` containing one) **first** stage with no evaluator available yet — see `docs/configuration.md`'s "DAMH without a collector" table; without a collector-side start-up handshake success or a fixed `surrogate_evaluator`, this deadlocks (or now raises, once the WS8 handshake covers it — see `library_notes/10_manual_review_notes.md` §2.11).
- `proposal_type="pCN"` with a prior that is not `Normal`/`PriorIndependentComponents` — `ValueError` from `build_proposal` (pCN's acceptance ratio drops the prior term, which is only valid for a Gaussian internal prior).
- A Hamiltonian-family proposal (bare or inside a `"block"`) without `Configuration.use_surrogate_gradients=True` in effect — `AssertionError` from `build_proposal`.

Not an exception, a silent correction with a printed warning: `proposal_type="pCN"`
with `adaptive=True` is forced back to `adaptive=False` in `Stage.__post_init__`
(adaptive pCN is not supported).

## `Stage.describe()`

`Stage.describe(index)` returns a short block listing every field with its **effective**
value (after `__post_init__`'s silent corrections), marking the posterior-affecting ones
with `*` and appending `[note]` lines for the fields that do not apply to the stage as
configured. `SamplingFramework.run()` and `run_local()` print one block per stage on rank 0
at start-up — see [`configuration.md`](configuration.md#configurationdescribe--stagedescribe).
