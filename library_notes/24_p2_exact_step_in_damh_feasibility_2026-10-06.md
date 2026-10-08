# 24 — Mixing the library's step kernels inside one stage (exact MH, DAMH, gradient-driven MH, surrogate-only): what is possible, what it costs, how to write it (2026-10-06)

**Design note**, not a record of implemented work. Companion of note 23 (mixture of *proposals*). It grew
out of P2 of `22_surrogate_blind_spot_2026-10-06.md` §4 (*in every DAMH iteration, with probability
`p_exact` perform a plain MH step on the exact model*), and generalises it: a stage whose transition kernel
is a **mixture of the step kernels the library provides**, each with its own proposal and sub-chain, drawn
with fixed probabilities per outer iteration. §2 lists every kernel the library has today and whether it
can be a component; §3 is the user interface; §4 the implementation; §5 the cost and effect; §6 the P2
special case in detail (the first instance worth building); §7 validity. Nothing was implemented or run;
all statements come from reading `surrDAMH/modules/algorithms.py`, `stages.py`, `modules/seeds.py`,
`modules/manifest.py`, both runners and the post-processing readers at commit `c518457` plus the
uncommitted working tree, and from the numbers already in notes 19 and 22.

---

## 1. Verdict

* A stage-level **mixture of step kernels** is feasible and additive. Every kernel of the library that
  leaves the exact posterior `π` invariant can be a component: exact MH with any proposal (random walk,
  pCN, Hamiltonian on surrogate gradients, `Block`, and note 23's `Mixture`), and DAMH with any proposal
  and any sub-chain length. The one kernel that **cannot** be a component is the surrogate-only MH step
  (`use_only_surrogate=True`): it leaves `π̃`, not `π`, invariant, and a mixture of kernels with
  different invariant laws has neither (§2.3).
* Size: **~750–900 lines** for the general facility (kernel specs, a mixture algorithm class built from
  the existing MH/DAMH iteration bodies, per-component proposals/seeds/adaptation/statistics, readers,
  tests, docs). The narrow P2 (`DAMH` step + exact `MH` step with the *same* proposal) is **~250–300
  lines** and is a strict special case; the general design below is laid out so that P2 is its first
  component pair, not a throw-away.
* Runs that do not use the facility stay bit-identical: the existing `Algorithm_MH` / `Algorithm_DAMH`
  classes and the string `Stage.algorithm` path are untouched except for factoring their iteration bodies
  into methods (guarded by the A1/A2 byte-identity tests).
* No MPI message changes, no collector changes; `run_local` parity is automatic (shared algorithm
  class, shared builder, seeds from `modules/seeds.py`).
* Unique property of a mixture containing an exact step: it is the only remedy of note 22 that acts in a
  **frozen** stage, and it removes the surrogate-boundedness assumption A4 from the Doeblin bound of
  note 18 (§7).

---

## 2. The kernels the library provides, and whether each can be a component

A "step kernel" here is what one outer iteration of a stage does to the chain state. Today a stage runs
one kernel for its whole length; the kernel is determined by `Stage.algorithm`, `Stage.proposal`,
`Stage.subchain_length`, `Stage.use_only_surrogate` and, as a *policy* rather than a kernel,
`Stage.surrogate_model_updates`.

| # | kernel as the library has it today | invariant law | component? | what a component needs | remarks |
|---|---|---|---|---|---|
| K1 | **exact MH**, `Algorithm_MH` with `RandomWalk` / `PCN` (fixed or adaptive) | `π` | **yes** | solver only | the P2 "exact step"; pCN keeps its Gaussian-internal-prior requirement |
| K2 | **exact MH with a Hamiltonian proposal** on surrogate gradients (`Algorithm_MH`, `Hamiltonian`, `use_surrogate_gradients=True`; optional per-iteration gradient-surrogate refresh, WS7) | `π` | **yes** | solver + evaluator (gradients) | the trajectory depends on the surrogate, the acceptance does not; a valid reversible MH move for any gradient field. In a mixture the evaluator poll happens once per outer iteration, before the chosen component acts (§4.3) |
| K3 | **exact MH with a `Block` proposal** (optionally containing Hamiltonian groups) | `π` | yes | as K1/K2 | `choose_group()` is called by the mixture for the chosen component |
| K4 | **exact MH with a note-23 `Mixture` proposal** | `π` | yes | as its components | proposal-level and kernel-level mixtures compose without interaction |
| K5 | **DAMH, sub-chain length `K`, any proposal** (`Algorithm_DAMH`), surrogate frozen during the sub-chain | `π` | **yes** | solver + evaluator | several DAMH components with different `K` / proposals may coexist (e.g. a Hamiltonian `K = 1` step and a random-walk `K = 20` sub-chain) |
| K6 | **DAMH with Hamiltonian** (the note-19 scheme) | `π` | yes | solver + evaluator (values + gradients) | = K5 with gradients; dual averaging adapts on the sub-chain acceptances, per component |
| K7 | **surrogate-only MH** (`use_only_surrogate=True`: the evaluator is wired in as the solver, `commSolver_stage = evaluator.as_solver()`) | `π̃` | **no** | — | a mixture of a `π`-invariant and a `π̃`-invariant kernel is invariant for neither; the chain would be biased by an unquantified amount. Also a stage-level wiring (one observation provider per stage), not a per-iteration switch. The only legitimate way to use surrogate-only moves inside a `π`-invariant kernel is as the inner sub-chain of a DAMH step, which is K5 |
| — | **DAMH-SMU vs frozen** (`surrogate_model_updates`) | — | not a kernel | — | a *policy* on when the shared evaluator is refreshed; applies to every DAMH / gradient component of the stage alike (§4.3). "Component A updates, component B is frozen" is not meaningful with one evaluator per chain |
| — | **stage layout** (`is_excluded`, `save_to_file`, `send_snapshots_to_collector`, stopping rules) | — | not a kernel | — | stay stage-level |

So every `π`-invariant step of the library is mixable (K1–K6, in any combination and multiplicity), and
exactly one is not (K7). Beyond mixtures, a deterministic **cycle** (e.g. nine DAMH steps then one exact
step) is also `π`-invariant (a composition of `π`-invariant kernels), has the same expected cost with
less variance, but is not reversible and makes the per-component adaptation bookkeeping and the theory
slightly less clean; the random mixture is recommended first, a `Cycle` spec with the same components is
a ~30-line extension.

### 2.1 Pairwise interactions worth knowing

* **K5 with several components, or K5 + K2**: all surrogate-using components share one evaluator per
  chain. Each DAMH sub-chain still sees a frozen surrogate (the poll is done once per outer iteration,
  before the component runs), so the telescoping of `correction_log_ratio` is untouched.
* **Any exact component (K1–K4) + any DAMH component**: after an exact component moves the chain, the
  surrogate terms cached on `self.current` (`observations_approx`, `log_likelihood_approx`) belong to the
  old state. The next DAMH component trusts them (§4.4). This is the one correctness trap of the whole
  facility and must be handled centrally.
* **K2 or K6 + a random-walk component**: the sampler computes a surrogate gradient on every surrogate
  evaluation when `proposal.needs_gradients` is true; with a per-component proposal the flag can be read
  from the *chosen* component, so random-walk iterations do not pay for gradients (unlike note 23 §3.3,
  where one proposal object serves all iterations).
* **Adaptive proposals in several components of one type** (two adaptive random walks): the same
  carry-over keying problem as note 23 §4.1; same V1 rule (raise), same later extension (positional).
* **Two Hamiltonian components with different `num_steps`** (short/long trajectories): allowed; each
  dual-averages its own step size on its own sub-chain acceptances.

### 2.2 What a mixture does not give

* No kernel of the library proposes from the prior or from a fixed independence density, so an
  "independence sampler" component would be new code (a trivial `Independent(prior)` proposal, ~30 lines,
  valid, occasionally useful for multimodality) — out of scope here.
* A mixture does **not** fix a *pessimistic* surrogate by itself in an SMU stage faster than the exact
  component supplies in-region snapshots (§5.2); it is a complement to P1/P3 of note 22, not a replacement.

### 2.3 Why K7 is excluded, in one line

If `P₁ π = π` and `P₂ π̃ = π̃` with `π ≠ π̃`, then `((1−p)P₁ + pP₂) π = (1−p) π + p P₂ π ≠ π` unless
`P₂ π = π`, which a surrogate-MH kernel does not satisfy. The stage would sample an unknown
interpolation. If an "exploration" mixture is ever wanted, it must be labelled like `use_only_surrogate`
(samples do not follow the posterior) and excluded from reports' posterior estimates; not recommended.

---

## 3. User interface

### 3.1 Recommended: kernel specs, `Stage.algorithm` accepts a `Mixture`

Mirror the 2026-09-21 proposal-spec design (`surrDAMH.proposals`): small picklable dataclasses, one
module, dispatch in one builder.

```python
from surrDAMH.stages import Stage
from surrDAMH.proposals import RandomWalk, Hamiltonian, PCN
from surrDAMH.kernels import MH, DAMH, Mixture          # new module

stages = [
    Stage(max_evaluations=20000),                        # unchanged: MH, adaptive random walk
    # P2: DAMH with a Hamiltonian step, plus an exact random-walk step 10 % of the time
    Stage(algorithm=Mixture([DAMH(proposal=Hamiltonian(num_steps=30, integrator="dimension_robust")),
                             MH(proposal=RandomWalk())],
                            probabilities=[0.9, 0.1]),
          max_evaluations=20000),
    # two DAMH components: cheap long random-walk sub-chains most of the time, a Hamiltonian step sometimes
    Stage(algorithm=Mixture([DAMH(proposal=RandomWalk(adaptive=False), subchain_length=20),
                             DAMH(proposal=Hamiltonian(step_size=0.05, adaptive=False))],
                            probabilities=[0.8, 0.2]),
          surrogate_model_updates=False, max_evaluations=20000),
]
```

Rules:

* `MH(proposal=...)` and `DAMH(proposal=..., subchain_length=1)` are the kernel specs; their fields are
  exactly the stage fields they replace. `Mixture(kernels, probabilities)` validates equal lengths,
  positive weights (normalised), no nested `Mixture`, no `use_only_surrogate` component (there is none to
  write: K7 is not a kernel spec), and at most one adaptive proposal per proposal type (note 23 §4.1
  rule, V1).
* `Stage.algorithm` keeps accepting `"MH"` / `"DAMH"`; with a string, `Stage.proposal` and
  `Stage.subchain_length` are used as today. With a `Mixture`, giving `Stage.proposal` or
  `Stage.subchain_length` is an error ("the components carry them"), so there is one way to say each
  thing. The string path is unchanged and remains the 95 % case.
* Derived stage properties generalise mechanically: `Stage.adaptive = any(component.proposal.adaptive)`,
  `proposal_needs_gradients = any(...)`, `stage_needs_surrogate = any(component is DAMH or needs
  gradients)`, `stage_name → alg000i_MIX` (`-SMU` suffix if it updates), `describe()` prints the
  components one per line. `surrogate_model_updates` resolves as for DAMH when any component is DAMH,
  as for MH-with-gradients when only K2 components use the surrogate.
* P2 is then the first example above. The narrow alternative, a `Stage.exact_step_probability` field on
  a DAMH stage (note 22's wording), is a 15-line facade over the same machinery; it can be offered as
  sugar, but the `Mixture` form is the one to build, because it also covers K5+K5, K5+K2 and the
  different-proposal exact step (§6.5) without new fields.

### 3.2 Alternatives considered

* A `Stage.kernel` field next to `algorithm`: two fields for one concept; rejected.
* Making `Stage` itself nestable (`Stage(mixture_of=[Stage(...), Stage(...)])`): stopping rules, file
  flags and `is_excluded` do not mix; rejected.
* A probability field per existing stage field (`exact_step_probability`, `hamiltonian_step_probability`,
  …): does not scale past two components; only acceptable as sugar for P2.

---

## 4. Implementation

| Piece | Lines | Notes |
|---|---|---|
| `surrDAMH/kernels.py`: `MH`, `DAMH`, `Mixture` dataclasses + validation; `KernelSpec` union | ~60 | pattern: `surrDAMH/proposals.py` |
| `stages.py`: `algorithm: Literal["MH","DAMH"] | Mixture`; conflict checks; derived properties; `stage_name`; `describe` | ~40 | |
| `proposal_builder.py`: `build_proposals_for_stage` returning one runtime proposal per component, seeded `subproposal_seed(seed0 + 1, k)` (reuse of the `BlockProposal` seed map), each built with the merged `carried` dict | ~30 | `stage.adaptive` hand-over: per-component `adapted_state` concatenated, as in note 23 §2 |
| `algorithms.py`: factor the iteration bodies — `_mh_iteration(proposal)` out of `Algorithm_MH.run`, `_damh_iteration(proposal, subchain_length)` out of `Algorithm_DAMH.run` (the sub-chain routine already takes the length from `self.stage`; make it a parameter) | ~70 moved | `Algorithm_MH`/`Algorithm_DAMH` call the methods; A1/A2 byte-identity + posterior tests guard the move |
| `algorithms.py`: `Algorithm_Mixture(AlgorithmBase)`: run loop (poll evaluator once, draw component from the component stream, `choose_group()` on its proposal, dispatch, `adapt()` on that proposal only, §4.4 consistency, per-component counters, `subchain_stats` row, stopping rules) | ~120 | |
| `modules/seeds.py`: `KERNEL_CHOICE_SEED_OFFSET = 4`; `SEED_FORMULA`; `manifest._seeds_dict` | ~15 | `tests/unit/test_manifest.py` pins the formula string |
| runners (`process_SAMPLER`, `runner_local`): wiring by the generalised stage properties (evaluator when any component needs it; snapshot communicator as today); hand-over loop over components | ~20 each | no message changes |
| outputs: `subchain_stats` gains a `kernel` column (component index; `−1`/absent for single-kernel stages, or always present — decision); new `kernel_stats/<stage>/rank%04d.csv` with one row per component (`kernel, proposed, accepted, rejected, prerejected`); `adaptive_stats` per component as note 23 §4.3 option A | ~60 | output-format change (additive) — needs author approval and `docs/outputs.md` |
| post-processing: `run_data` loader for the new file, report table "kernels of stage i", adaptive-stats per component | ~50 | |
| tests: unit via `run_local` on the linear-Gaussian problem — `Mixture([DAMH(...)],[1])` byte-identical to the plain DAMH stage *is not expected* (different seeds) and must not be asserted; instead: posterior moments for K5+K1 at `p = 0.5`, K5+K5, K5+K2; `adapt()` routed to the chosen component only; §4.4 consistency (surrogate terms of `current` equal a fresh evaluation after every iteration, instrumented); K7 rejected at construction; conflict checks; seeds in manifest; MPI: a B2 variant with `Mixture([DAMH(RandomWalk()), MH(RandomWalk())],[0.7,0.3])` through the collector | ~250 | |
| docs: `docs/stages.md`, `docs/outputs.md`, `docs/concepts.md` (one paragraph on mixtures), `CHANGELOG.md` | ~60 | |

### 4.1 Random streams

One **component-choice** stream per chain and stage (`seed0 + 4`; the stride of 10 leaves room), one
acceptance stream as today (`seed0 + 2`), one proposal stream per component (`subproposal_seed(seed0 +
1, k)`). The single-kernel classes are not touched, so existing runs stay bit-identical; a mixture stage
is a new kind of stage with its own streams.

### 4.2 Adaptation and hand-over

`adapt()` goes to the proposal of the component that ran, with that component's own feedback (outer
acceptance for a DAMH component, exact acceptance for an MH component, sub-chain acceptances for a
Hamiltonian dual averaging). Each adaptive proposal therefore targets its own kernel's rate — the correct
signal — on a random thinning of the iterations. Hand-over: concatenate the components' `adapted_state()`
vectors, `Allgather`, split, `set_pooled_state` per component, merge `carry_over()` dicts with the note-23
rule. Diminishing adaptation holds iff it holds per component (fixed weights).

### 4.3 Evaluator policy

One poll per outer iteration, before the chosen component runs (today: once per sub-chain in DAMH, once
per iteration in gradient-MH — the same cadence). The component then runs on a frozen evaluator, so the
DA correction telescopes as before. `surrogate_model_updates` is one stage-level switch.

### 4.4 The consistency invariant (the one trap)

Invariant to maintain after every iteration: *if any component uses the surrogate, `self.current`
carries `observations_approx` / `log_likelihood_approx` evaluated with the currently installed
evaluator.* It can break in two ways: an **exact component accepted** a proposal whose surrogate terms
were never computed, or the **evaluator was refreshed** since `current` was last scored (already handled
for DAMH by `_evaluate_surrogate_transition(surrogate_evaluator_changed=True)`). Cheapest uniform
handling: score the exact component's proposal with the surrogate before its decision (one cheap
surrogate call; it also fills `obs_approx_*` in `raw_data` and yields the P6 audit statistic, §6.3), and
re-score `current` after any evaluator refresh exactly as DAMH does now. A unit test should assert the
invariant after every iteration of a mixed run.

---

## 5. Cost and effect

### 5.1 Cost model

Per outer iteration a component `c` costs `e_c` exact evaluations in expectation and produces a move of
"quality" `m_c` (ESS contribution). For the library's kernels: exact MH `e = 1`; DAMH `e = 1 − r_c` with
`r_c` the pre-rejection rate of that component (note 19: 0.19 for the Hamiltonian `K = 1` step, ~0.65 for
random-walk sub-chains). The mixture costs `Σ p_c e_c` per iteration. Relative to running the dominant
DAMH component alone, adding an exact component with weight `p` costs `p · r / (1 − r)` more evaluations:

| dominant component (note 19, stage 3) | `r` | `K` | `p = 0.05` | `p = 0.1` | `p = 0.2` | quality of the exact step vs the DA move |
|---|---|---|---|---|---|---|
| DAMH + Hamiltonian, `K = 1` | 0.19 | 1 | +1.2 % | +2.3 % | +4.7 % | the same (one trajectory either way, if the exact step also uses the Hamiltonian proposal) |
| DAMH + random-walk sub-chain | 0.65 | 20–200 | +9 % | +19 % | +37 % | one random-walk step instead of a `K`-step sub-chain: the `p` fraction of iterations mixes far worse |

Mixtures of two DAMH components (K5+K5) cost the weighted mean and are the natural way to get "cheap
most of the time, well-mixing sometimes" without a stage boundary.

### 5.2 Effect in the blind spot of note 22 (mechanism, nothing measured)

Setting: region `R` of high exact posterior where the surrogate is pessimistic by `e⁻⁵⁰`.

1. **Entering `R`.** A DAMH component enters with probability ∝ `e⁻⁵⁰`. An exact component enters at
   the rate of plain exact MH times `p`: "never" becomes "`1/p` times the exact-MH waiting time". A
   Hamiltonian exact component is weaker here than a random-walk one: its trajectory still follows `∇Ũ`
   and is not attracted into `R` (note 22 §1.3); only the acceptance ignores the surrogate.
2. **Inside `R` with the surrogate still wrong.** Each DAMH iteration from `x ∈ R` accepts leaving in the
   sub-chain (`L̃(y)/L̃(x) ≈ e⁺⁵⁰`), evaluates `G(y)` for `y ∉ R`, and rejects (`α₂ ≈ e⁻⁵⁰`): one exact
   evaluation **outside** `R` per iteration and no movement. Only the exact component moves the chain
   inside `R`; mixing there runs at rate `p` until the surrogate is corrected.
3. **Learning `R`** (SMU stage, snapshots on). In-`R` snapshots come from the exact component only: a
   rejected exact proposal at once (multiplicity 0), an accepted one when the state is later left (with
   the large multiplicity accumulated in item 2 — under `weighting="multiplicity"` these dominate the
   retraining, which is the right direction). Rate ≈ `p` in-`R` snapshots per iteration while in `R`. How
   many the network needs is the experimental question; the manual check of note 22 P1 applies verbatim.
4. **Frozen stage.** Items 1–2 hold; the chain is not confined to the surrogate's idea of the posterior
   and its stationary law is exact, at the price of mixing in `R` at rate `p`. Nothing else in note 22
   does this.
5. **Optimistic errors** are unaffected (DA already evaluates and rejects there).

### 5.3 Relation to "interleave short MH stages" (note 22 P7)

That is a mixture realised in bursts with no code: stage boundaries cost a carry-over and a fresh
proposal object, the exact bursts are invisible to the DAMH adaptation, the frozen-stage protection is
only as fine as the stage lengths, and the per-iteration audit statistic (§6.3) is unavailable. The
mixture is the same idea as one homogeneous reversible kernel.

---

## 6. The P2 special case: `Mixture([DAMH(...), MH(...)], [1 − p, p])`

### 6.1 Body of the exact component

```
proposed = proposal_c.propose_sample(current)
[§4.4: proposed.observations_approx / log_likelihood_approx from the installed surrogate]
_evaluate_proposed_sample()                              # exact model; non-finite proposals as in MH (tag −2)
log α = proposal_c.get_log_acceptance_probability(exact log L's, log priors)
decision with the acceptance stream; _handle_acceptance() / _handle_rejection()
```

`_handle_acceptance` / `_handle_rejection` are reused unchanged, so the `multiplicity` bookkeeping (A30),
the `samples` file and the snapshot transport need no change; the identity
`sum(multiplicity) == accepted + rejected + prerejected (+1)` of `docs/outputs.md` still holds because
exact-step outcomes land in `accepted` / `rejected`. `max_evaluations` counts `accepted + rejected`, so
exact steps consume the budget like any exact evaluation.

### 6.2 Output rows of an exact iteration

`raw_data`: `state_type` stays `accepted` / `rejected` (every reader filters only `== "prerejected"`:
`post_processing/loading.py:214`, `statistics.py:345/461`, `plots.py:387`); the iteration is identified
by the `kernel` column of `subchain_stats`, which is 1:1 with `raw_data` rows per rank. `subchain_stats`
exact rows: `subchain_accepted = 0`, `subchain_acceptance_rate = NaN`, `correction_log_ratio = 0`,
`outer_proposed_changed = 1`.

### 6.3 The audit statistic for free

With §4.4, every exact-step row carries `log L(y)` (exact, as the decision used) and `obs_approx_*`, so
`log L(y) − log L̃(y)` at points chosen **independently of the surrogate** can be computed from `raw_data`
— the note-22 P6 statistic that nothing measures today. With P1 (audit of pre-rejected proposals) the
other half of the picture is added; P1 and P2 share the per-stage probability, the separate stream, the
branch in the DAMH iteration and the `notes`/`docs` changes, so they should be built together.

### 6.4 Adaptation on exact iterations

Per §4.2 the exact component's proposal adapts on the exact acceptance; the DAMH component's proposal
never sees exact-step outcomes. In the degenerate "same proposal object for both components" variant
(note 22's wording) this separation is lost; recommend **separate proposal objects** (the `Mixture` spec
gives them naturally) and, if a shared object is ever wanted, skipping `adapt()` on exact iterations.

### 6.5 Which proposal for the exact step

For a Hamiltonian DAMH component, pair it with `MH(proposal=RandomWalk())` (carried covariance, adaptive
on its own exact acceptance) rather than with a second Hamiltonian: the random walk proposes
isotropically and is the scientifically better blind-spot remedy (§5.2 item 1); the Hamiltonian exact
step is still valid and mixes as well as the DA step where the surrogate is good. Both are expressible
without new fields.

### 6.6 Minimal P2 build (if the general facility is postponed)

`Stage.exact_step_probability: float = 0.0` on a DAMH stage; `Algorithm_DAMH.run` draws from `seed0 + 4`
and runs `_mh_iteration(self.proposal)` with the same proposal object, skipping `adapt()` on exact
iterations; §4.4 scoring; `kernel`/`exact_step` column in `subchain_stats`, two `notes` columns;
`p = 0` byte-identical (test), `p = 1` → every row exact and `prerejected = 0` (test), posterior moments
at `p = 0.5` (test). ~250–300 lines. Everything but the field name is reused when the general facility
is built later.

---

## 7. Validity

Kernel of one iteration: `P_γ = Σ_c p_c K_c,γ`, `γ` the installed surrogate. Each component is
`π`-reversible for every `γ` (exact MH: standard; DAMH: Christen–Fox, as derived in
`Algorithm_DAMH.run`; a Hamiltonian step on a surrogate gradient field: a valid reversible MH move for any
gradient field, as the `BlockProposal` docstring records), so the mixture is `π`-reversible (Tierney
1994) with fixed weights. K7 is excluded by §2.3.

DAMH-SMU (note 18): Lemma 2 (kernel continuity in the surrogate) holds with its constant multiplied by
the total weight of the surrogate-using components, so **A5 (diminishing adaptation of the surrogate) is
still required** whenever any component uses the surrogate. Lemma 3 (simultaneous uniform ergodicity)
improves qualitatively when a random-walk or pCN **exact** component with weight `p` is present:
`P_γ(x, ·) ≥ p K^{MH}(x, ·) ≥ p · q_min · a · λ(· ∩ X)` with
`a = (p_min/p_max)(q_min/q_max) e^{−2B_ℓ}` from **A1–A3 alone** — no `K`-th power, no dependence on the
surrogate bound `B̃`. So **A4 is no longer needed for the Doeblin minorisation**; whether Lemma 2's
constant still uses A4 must be re-checked in note 18 (it says boundedness is used in Lemmas 2–3). The
minorisation constant remains astronomically small in any real problem; the gain is that it no longer
depends on what the network predicts. A mixture of DAMH components only (K5+K5, K5+K6) changes nothing in
note 18's assumptions. An exact component with a Hamiltonian proposal gives no Doeblin bound (A2 fails for
HMC, as today).

---

## 8. Decisions for the author

1. Build the general `Mixture` kernel spec (§3.1), or the minimal P2 facade first (§6.6)? Recommendation:
   general spec, with P2 as the first tested pair and P1 built alongside.
2. Output format: `kernel` column in `subchain_stats` (always, or only for mixture stages), the new
   `kernel_stats/` file, per-component `adaptive_stats` (note 23 §4.3 option A).
3. Carry-over rule for two adaptive proposals of one type: raise (V1, as note 23 §4.1).
4. Allow a `Cycle` (deterministic schedule) spec alongside `Mixture`? Recommendation: later, if asked.
5. Seed offset `+4` for the component-choice stream; per-component proposal seeds via `subproposal_seed`.

## 9. Relation to note 23

A `Mixture` **proposal** (note 23) cannot give P2: all its components are screened by the surrogate
pre-rejection. The kernel-level `Mixture` of this note mixes **algorithms**; a DAMH component may itself
carry a proposal-level `Mixture`, and the two compose without interaction. The two specs should carry
different names in the sampling script to avoid confusion — e.g. `surrDAMH.proposals.Mixture` and
`surrDAMH.kernels.Mixture` imported under their module, or `MixtureProposal` / `MixtureKernel`; the
examples in §3.1 assume the module-qualified form.
