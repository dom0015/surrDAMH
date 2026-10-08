# 23 — A `Mixture` proposal (mixture of kernels): feasibility, size, and what it does to adaptation (2026-10-06)

**Design note**, not a record of implemented work. The question (author, 2026-10-06): how difficult,
how impactful and how large would it be to add a proposal spec `Mixture(list_of_proposals,
list_of_probabilities)` that draws one of several kernels per iteration; are all current proposals
mixable; does adaptation survive; what else does it drag in. Nothing was implemented; every statement
below is from reading the code at commit `c518457` plus the uncommitted working tree.

---

## 1. Verdict

A fixed-weight `Mixture` is a **moderate, low-risk, additive** change. The runtime mechanics already
exist in `BlockProposal` (`surrDAMH/modules/proposals.py`): random choice of a sub-kernel per outer
iteration (`choose_group`), dispatch of `propose_sample` and `get_log_acceptance_probability` to the
chosen sub-proposal, per-rank/per-stage reseeding of the sub-proposals (`subproposal_seed`), gradient
forwarding (`set_gradient_functions`). New are (i) routing `adapt()` and the cross-rank hand-over
(`adapted_state` / `set_pooled_state` / `carry_over`) through the components and (ii) two design
decisions: how carried step values are keyed, and how the per-component adaptation traces are written.

Estimate **450–600 lines**, about half of it tests and docs; ~250 lines of library code.
No MPI message changes (the hand-over already `Allgather`s a shape-agnostic `adapted_state()` vector),
`run_local` shares `build_proposal`, so continuation parity is automatic. Runs that do not use the new
spec are bit-identical to today.

**Finding.** An equal-weight mixture of *fixed* kernels is reachable today without code: `Block.__post_init__`
only checks `len(groups) == len(proposals)` and no nesting, not that the groups partition the parameters,
so

```python
Block(groups=[list(range(d)), list(range(d))],
      proposals=[RandomWalk(scale=0.1), RandomWalk(scale=1.0)])
```

runs as a 50/50 mixture of two fixed random walks (`BlockProposal._full_gradient_argument` is the
identity for a full group; `group_probabilities` exists at runtime but the builder never passes it, so
the weights are uniform). Missing are weights, adaptation, carry-over and an honest name.

---

## 2. Where the lines go

| Piece | Lines | Notes |
|---|---|---|
| `surrDAMH/proposals.py`: `Mixture` dataclass (`proposals`, `probabilities`), validation (equal lengths, positive weights normalised, no nested `Mixture`, no `Mixture` inside `Block`), `adaptive` = any component adapts, `needs_gradients` = any; add to `ProposalSpec` | ~45 | mirrors `Block` |
| `surrDAMH/stages.py`: the two `isinstance` tuples (`__post_init__`, `_check_stages` hint), docstrings | ~10 | `STEP_FIELD.get(type(spec))` already returns `None` for composite specs |
| `surrDAMH/modules/proposals.py`: `MixtureProposal(Proposal)` | ~120–150 | `__init__` (weights, own generator, reseed components), `reseed`, `choose_group` (draw index, then forward `choose_group` to the chosen component so a `Block` component picks its group), `propose_sample`, `get_log_acceptance_probability`, `set_gradient_functions` (forward unchanged, full dimension), `adapt` (route to the component that proposed), `adapted_state` (concatenate the components' fixed-length vectors), `set_pooled_state` (split by the components' known lengths, forward), `carry_over` (merge, §4.1 rule), `adapted_summary` (keys prefixed `component<k>.`) |
| `surrDAMH/modules/proposal_builder.py`: `Mixture` branch; factor the `Block` branch into `_build_block` so a `Block` can be a component; components are built with the merged `carried` dict (unlike `Block`, whose sub-proposals get `carried={}`) | ~40 | |
| Adaptation-trace output for a stage with several adapting components (§4.3): writer `AlgorithmBase._write_adaptation_stats`, loader `modules/run_data.py`, `post_processing/plots.py` (adaptive-stats plot), `html_report.py` | ~50–60 | output-format change for mixture stages only |
| `modules/continuation.save_carry_over`: prefixed summary keys | ~5 | |
| Tests: unit (spec validation; dispatch; `adapt` routed to the proposing component only; `adapted_state`/`set_pooled_state` round trip with 2+ rows; carry-over rule; builder guards incl. pCN-needs-Gaussian-prior per component and Hamiltonian-needs-gradients; component seeds distinct across ranks and from the top-level stream; manifest JSON-safe) + one MPI case (4 ranks, adaptive mixture stage followed by a stage consuming its carry-over; Gaussian target moments) | ~200 | pattern: `tests/unit/test_proposals.py::test_block_proposal_*`, `tests/mpi/test_mpi_posterior.py` |
| `CHANGELOG.md`, `docs/outputs.md` (adaptive_stats layout), stage docstring, a toy example line | ~40 | |

`manifest._json_safe` already recurses into lists of dataclasses: zero change there.
`AlgorithmBase._warn_about_nonfinite_proposal` uses `getattr(proposal, "step_size", None)`: fine.

---

## 3. Mixability of the current proposals

| Component | Mixable | Remarks |
|---|---|---|
| `RandomWalk` (fixed / adaptive) | yes | symmetric kernel, no state between calls |
| `PCN` (fixed / adaptive) | yes | keeps its Gaussian-internal-prior requirement; the builder's `_prior_is_gaussian` check applies per component |
| `Hamiltonian` / `dimension_robust` (fixed / adaptive) | yes | `get_log_acceptance_probability` reads the momenta stored by the last `propose_sample` of the *same object*; consistent as long as the component is fixed for the whole outer iteration (§3.2) |
| `Block` | yes, as a component | natural use: block update vs full update; the mixture forwards `choose_group`. `Mixture` *inside* `Block` and nested `Mixture`: forbid for now |

### 3.1 Validity

Every component is a reversible kernel for the same target. A fixed-weight mixture of reversible kernels
is reversible (Tierney 1994), so the posterior is unchanged in an MH stage. In a DAMH stage the sub-chain
kernel must be reversible w.r.t. the surrogate posterior `π̃` for the delayed-acceptance correction
(`Algorithm_DAMH.run`, the telescoping `correction_log_ratio`): a mixture of `π̃`-reversible kernels is
`π̃`-reversible, and so is any power of it, so both "draw the component once per outer iteration" and
"draw it per sub-chain step" are valid (in the latter case the realised path's correction still telescopes
to `log L̃(y) − log L̃(x)`, and the transition density that enters the outer ratio is the marginal
`M^n`, which is reversible).

### 3.2 Recommendation: draw the component once per outer iteration

This is what `choose_group` already does (`Algorithm_MH.run` line ~661, `Algorithm_DAMH.run` line ~830,
before `_propose_new_sample_using_subchain`). Keep it: the list `subchain_log_acceptance_probabilities`
passed to `adapt()` then belongs unambiguously to one component (the Hamiltonian dual averaging adapts on
it, note 16 §5 item 4), and the Hamiltonian momenta stored on the component stay consistent between the
sub-chain's `propose_sample` and the outer `get_log_acceptance_probability`.

### 3.3 Gradient cost

`AlgorithmBase._evaluate_surrogate_transition` computes a surrogate gradient (`vjp`) on every surrogate
evaluation when `proposal.needs_gradients` is true. With an RW+HMC mixture the random-walk iterations pay
for an unused gradient. Not a correctness issue; fixable later by letting `needs_gradients` follow the
chosen component at evaluation time (the stage-level decision to request an evaluator must stay "any").

---

## 4. Adaptation

Routing is straightforward: `MixtureProposal.adapt` forwards each call to the component that produced the
proposal (`-inf` for a non-finite or pre-rejected iteration included, exactly as today). Consequences:

* **Scale recursions** (Robbins–Monro on `log_sigma` / `logit_beta`, dual averaging on `log ε`) see
  only their own component's acceptance probabilities — which is what they should target.
* **Random-walk shape** (Welford mean/M2 of the post-decision chain state) sees a random subsample of
  the chain — still an unbiased covariance estimate, with fewer points. Feeding every random-walk
  component the chain state while routing the acceptance would need `adapt()` split into two concerns;
  not worth it for a first version.
* **Diminishing adaptation / ergodicity**: the adapted parameter is the tuple of component parameters;
  with *fixed* weights the mixture's adaptation diminishes iff each component's does, and containment is
  inherited, so the Roberts–Rosenthal argument that covers the components covers the mixture. Adapting
  the weights themselves (not requested) would need a separate argument; do not do it.
* The "fixed safety kernel + adaptive kernel" mixture that notes 16 (§2, Vihola/Roberts–Rosenthal
  `β`-mixture) and 20 (§(d), Sherlock et al. defensive mixture) cite as the alternative to clipping
  becomes expressible: `Mixture([RandomWalk(scale=s0), RandomWalk()], [0.05, 0.95])`.

### 4.1 Carry-over keying (decision needed)

`process_SAMPLER` / `runner_local` merge `carry_over()` dicts into one `carried` dict keyed by the step
field of the spec *type* (`"scale"`, `"beta"`, `"step_size"`), and `build_proposal` resolves a `None`
step from it. Two adaptive random walks in one mixture would overwrite each other's `"scale"`.

Recommendation for V1: **raise in the builder if two adaptive components share a type**; otherwise merge
the component carry-overs into the flat dict. Pass the merged `carried` into the components so a
mixture both consumes and produces carry-over (a following plain `RandomWalk()` stage then starts from
the mixture's adaptive random walk). A positional scheme (`carried["mixture"] = [per-component dicts]`,
consumed by a following `Mixture` of identical structure) can be added later if ever needed.

### 4.2 "Adapt unless pinned" inside a mixture is a trap

Two unpinned `RandomWalk()` components both drive their acceptance to 0.234 and converge to the same
kernel; the mixture degenerates to a plain adaptive random walk. The sensible use pins one component
(the big-jump or safety kernel) or gives different `target_rate`s. Keep the default (`adaptive = step
is None`) for consistency with the other specs, but print a warning when two components of one type both
adapt — the 4.1 rule makes this an error anyway in V1.

### 4.3 Adaptation-trace files (decision needed)

`_write_adaptation_stats` writes one `adaptive_stats/<stage>/rank%04d.csv` with the single
`proposal.adaptation_stats_header`; the components have different headers (random walk 5 columns, pCN 3,
Hamiltonian 4), and `plots.py` treats every non-meta column as an adapted parameter. Options:

* **A (recommended)**: per-component subdirectory `adaptive_stats/<stage>/component<k>/rank%04d.csv`,
  each with its own header; loader returns a list per stage for a mixture; plot/report iterate.
  Output-format change confined to mixture stages; must be documented in `docs/outputs.md`.
* B: one file with a `component` column and the union of columns, NaN-padded. Smaller change, uglier
  data (`n` means different things per row).

Carry-over summaries (`adapted_summary`) get prefixed keys (`component0.base_cov`, ...); the HTML
report's carry-over block is key-driven and needs no change beyond cosmetics.

### 4.4 Diagnostics worth adding (optional, ~20 lines)

Per-component proposal and acceptance counters printed at the end of the stage and written to `notes`,
so a user can see that the 5 % safety kernel was actually drawn and how often it was accepted.

---

## 5. Caveats and what this is not

* A `Mixture` of **proposals** is **not** P2 of note 22 (an *exact* MH step with probability `p` inside
  a DAMH stage): P2 mixes *algorithms* (DA step vs exact step) and lives in `Algorithm_DAMH.run`, not in
  the proposal; see note 24.
* An RW+HMC mixture does **not** cure the surrogate blind spot of note 22: every component is still
  screened by the surrogate pre-rejection. It only removes the dependence on the surrogate *gradient*
  for the random-walk fraction of iterations.
* A one-component `Mixture([RandomWalk(...)], [1.0])` does not reproduce the plain stage bit for bit:
  the component is reseeded via `subproposal_seed(seed, 0)` and the mixture's own generator consumes a
  draw per iteration.
* Behaviour-change rule (CLAUDE.md): nothing changes for existing configurations; the only new output is
  the §4.3 layout for mixture stages.

## 6. Suggested split if built

One **opus** agent: spec, `MixtureProposal`, builder branch, carry-over rule, unit tests of the
mathematics (routing, pooled-state round trip). One **sonnet** agent: adaptation-trace output path
(writer/loader/plot/report), MPI smoke test, docs and CHANGELOG. Review both diffs; run
`python -m pytest tests -q` and one 4-rank toy run with `Mixture([RandomWalk(scale=0.3), RandomWalk()],
[0.1, 0.9])` followed by `Stage(proposal=RandomWalk())` to see the carry-over consumed.
