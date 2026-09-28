# `Stage` with proposal objects — design (2026-09-21, implemented the same day)

Status: **implemented 2026-09-21** after the author answered §4: short names (`scale`, `algorithm`,
`subchain_length`); one `Hamiltonian` spec with `integrator="leapfrog"|"split"` instead of an
`infinite` flag; new module `surrDAMH/proposals.py`; no string shorthand; kept as one separable
change set. Evidence: `10_manual_review_notes.md` §2.28, `CHANGELOG.md` "Breaking changes".
Written after the intuitive-usage pass (`10_manual_review_notes.md` §2.26/§2.27); the author
chose this direction over "flat Stage with cleaned names" on 2026-09-21.

## 1. Problem

`Stage` has 21 fields from five unrelated groups. Nine of them belong to exactly one
`proposal_type` and are dead weight for every other stage:

| group | fields today |
|---|---|
| algorithm | `algorithm_type`, `subchain_max_length`, `surrogate_model_updates`, `use_only_surrogate` |
| proposal (random walk) | `proposal_sd_or_cov`, `adaptive`, `adaptive_target_rate` |
| proposal (pCN) | `pcn_beta` (+ `adaptive`, `adaptive_target_rate`) |
| proposal (Hamiltonian) | `hamiltonian_step_size`, `hamiltonian_num_steps`, `proposal_sd_or_cov` as the mass (+ `adaptive`, `adaptive_target_rate`) |
| proposal (block) | `block_proposal_groups`, `block_proposal_list` |
| stopping | `max_evaluations`, `max_samples`, `time_limit` |
| output / chaining | `save_to_file`, `send_snapshots_to_collector`, `is_excluded`, `name` |
| dead | `proposal` (never read) |

Consequences for a user: the tooltip of `Stage(` lists every knob of every proposal;
`proposal_sd_or_cov` means "sd" for one proposal and "mass" for another; `adaptive`
means three different algorithms; the "adapt unless pinned" default (§2.27) has to
explain which of three fields pins which proposal.

## 2. Proposed user-facing API

```python
from surrDAMH.stages import Stage
from surrDAMH.proposals import RandomWalk, PCN, Hamiltonian, Block

Stage(max_evaluations=1000)                                     # == proposal=RandomWalk(): tuned online
Stage(proposal=RandomWalk(scale=0.5), max_evaluations=1000)     # fixed step, exactly as given
Stage(proposal=RandomWalk(scale=0.5, adaptive=True), ...)       # tune, starting from 0.5
Stage(algorithm="DAMH", proposal=RandomWalk(), subchain_length=5, max_evaluations=200)
Stage(proposal=PCN(beta=0.2), time_limit=60)
Stage(algorithm="DAMH", proposal=Hamiltonian(step_size=0.05, num_steps=100), ...)
Stage(proposal=Block(groups=[slice(0, 2), slice(2, 5)],
                     proposals=[RandomWalk(scale=0.5), PCN(beta=0.3)]))
```

Proposal *specs* are small picklable dataclasses with no RNG and no state (the runtime
classes in `modules/proposals.py` stay as they are and are built from the spec):

| spec | fields (all keyword) | `adaptive=None` means |
|---|---|---|
| `RandomWalk` | `scale=None` (scalar sd, sd vector or covariance, internal space), `adaptive=None`, `target_rate=None` | adapt iff `scale is None` |
| `PCN` | `beta=None`, `adaptive=None`, `target_rate=None` | adapt iff `beta is None` |
| `Hamiltonian` | `step_size=None`, `num_steps=10`, `mass=1.0`, `infinite=False`, `adaptive=None`, `target_rate=None` | adapt iff `step_size is None` |
| `Block` | `groups: list[slice]`, `proposals: list[spec]` | never (sub-proposals adapt individually) |

`Stage` keeps only: `algorithm` ("MH"/"DAMH"), `proposal` (spec, default `RandomWalk()`),
`max_evaluations`, `max_samples`, `time_limit`, `subchain_length`, `surrogate_model_updates`,
`use_only_surrogate`, `save_to_file`, `send_snapshots_to_collector`, `is_excluded`, `name`.
Every proposal knob leaves `Stage`; Pylance shows each proposal's own arguments on
`RandomWalk(` etc., and `Stage(` shrinks to twelve fields of which a user typically sets
three.

## 3. Mechanics (what changes inside the library)

- `modules/proposal_builder.build_proposal(stage, conf, prior, seed, carried)`: dispatch on
  `type(stage.proposal)` instead of the string; each spec gets a `build(no_parameters, prior,
  seed, carried)` or the builder keeps one `if/elif` per spec class. Same runtime classes.
- Carry-over dict (`Proposal.carry_over()`, `process_SAMPLER`/`runner_local`,
  `save_carry_over`/`load_carry_over`, `RunData.carry_over`): keys become the spec field
  names (`"scale"`, `"beta"`, `"step_size"`) instead of `"proposal_sd_or_cov"`,
  `"pcn_beta"`, `"hamiltonian_step_size"`. The `.npz` files of already-finished runs keep
  the old keys: `load_carry_over` is only used for reporting, so a one-line key rename on
  read is enough (no converter for `out_*`, decision 6).
- `stage_name()`: `-adaptive` suffix from `stage.proposal.adaptive` (resolved).
- `Stage.describe()`: one extra line `proposal=RandomWalk(scale=None, adaptive=True, ...)`.
- `modules/manifest.build_run_manifest`: `stages[i]` gets a nested `"proposal": {"type":
  "RandomWalk", ...fields}`. `read_run` reads only stage names from the manifest, so
  old manifests stay readable; the HTML report's "Sampling Stages" section renders whatever
  the stage dict holds (to be checked when implementing: `post_processing/html_report.py`
  reads `stages` as `Stage` objects for the live run and as the manifest copy otherwise).
- Removals with no compatibility layer (author decision 2026-09-17, "no compat"):
  `Stage(proposal_type=..., proposal_sd_or_cov=..., pcn_beta=..., hamiltonian_*=...,
  block_proposal_*=..., adaptive=..., adaptive_target_rate=...)` raise `TypeError`, as
  the removed `Configuration` fields do. `algorithm_type` → `algorithm`,
  `subchain_max_length` → `subchain_length` are renames of the same kind.
- Sweep: 13 `toy_examples` scripts, ~150 `Stage(` sites in `tests/`, `docs/stages.md`,
  `docs/concepts.md` references, `CHANGELOG.md`. Mechanical; one subagent per directory.

Not posterior-affecting for any configuration that is translated one-to-one (the runtime
proposal objects, seeds and defaults do not change). The translation must be checked with
`tests/validation` and `tests/mpi` as usual, and the manifest diff of one run before/after.

## 4. Open points for the author

1. Names: `scale` (vs. keeping `sd_or_cov`), `algorithm` (vs. `algorithm_type`),
   `subchain_length` (vs. `subchain_max_length`). Recommendation: the short forms above.
2. Hamiltonian: one spec with `infinite=False/True`, or two specs `Hamiltonian` /
   `HamiltonianInfinite`? Recommendation: one spec with a flag; the two runtime classes stay.
3. Module: new `surrDAMH/proposals.py` exporting the specs (runtime classes remain in
   `surrDAMH/modules/proposals.py`), re-exported as `surrDAMH.proposals`. Alternative: put
   the specs next to `Stage` in `stages.py`.
4. Should a string shorthand be accepted (`Stage(proposal="pCN")`)? Recommendation: no —
   one way to do it, and the spec constructor is where the tooltip lives.
5. Timing: this touches every script and test, so it should be its own PR with nothing
   else in it.
