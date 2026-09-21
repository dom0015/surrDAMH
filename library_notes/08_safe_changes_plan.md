# Safe-changes plan — status

Originally a plan (2026-09-12) for changes that don't alter the sample stream of any
configuration that already worked. All groups A–G are applied; re-verified 2026-09-18 by
reading the current code (spot-checking at least one concrete line per item, not just trusting
the earlier claim).

## A–E: applied in full

| Group | Content | Status |
|---|---|---|
| A (A1–A3) | Import-time crashes: dead `temptemptemp`, stray `torch.mtia` import, `TestData` call fix | **fixed** |
| B (B1–B8) | Fail-fast guards: unconditional stop-signal poll, Hamiltonian-needs-gradients assert, non-finite initial state raises, `pCN` beta assert, `state_dependent_approximation=True` warns | **fixed** — B2 (a print warning) superseded by something stronger: the WS8 start-up handshake now hard-errors instead of just warning; B5 (`artificial_acceptance_multiplicator`) removed the field outright rather than warning |
| C (C1–C12) | Latent-bug fixes on paths no script reached: list-`or` bug, `proposed.solver_tag` gate, kd-tree zero guard, `Gaussian_process` outer product, `Stage.adaptive_corr_limit` annotation, `"block"` in the `Literal`, `Proposal` class defaults, float64 MPI boundary, `no_unique_samples` fix | **fixed** — C12 (archive root TSX scripts) superseded, scripts archived instead |
| D (D1–D14) | Dead code/comments/docstrings, zero behaviour change | **fixed** |
| E (E1–E6) | Packaging: `setup.py` deps, `python_requires`, `.gitignore`, stray root files, `save_snapshots` output path | **fixed** |

## G: changes the sample stream — all six decided and applied

Author decision 2026-09-17: "do G1–G6 as recommended," together with A30 (finding 5.7).
Evidence and measured before/after deltas: `10_manual_review_notes.md` §2.13. Two deviations
from the plan's suggested defaults, both deliberate:

| # | Change | What it changes | Deviation from the plan text |
|---|---|---|---|
| G1 | Wire `Stage.adaptive_*` fields into `GaussRandomWalk_adaptive` | scripts that set the fields (none in-tree) start honouring them | `adaptive_sample_limit` default is `None` (unbounded — today's behaviour), not 10, so unset stages stay bit-identical |
| G2 | Normalise `sd_or_cov` to 2-D before the adaptive `Allreduce`; block+adaptive → `ValueError` | a rank that never adapted now hands over a full covariance (different RNG stream, same distribution) for the edge case; block+adaptive now fails fast instead of `AttributeError` later | none |
| G3 | Fix `lhs_normal`'s "best of 5" selection (`maxmin` was never updated) | every `initial_sample_type="lhs"` run with 3+ chains gets a different (better-spread) design | none |
| G4 | Seed `Distribution.rvs()` from a per-rank/per-stage generator | `initial_sample_type="prior"` runs become reproducible (they weren't before, so there's no "before" value to match) | none |
| G5 | Re-seed `BlockProposal` sub-proposals per rank/stage | every block-proposal run changes (chains no longer share identical increments — this was a correctness defect, not a feature) | sub-proposal seed formula is `2**31 + 1000*block_seed + index`, not the plan's `seed + 100*(k+1)` (which collides for 11+ samplers with 2 groups) |
| G6 | Line-buffer CSV writers | none (I/O only) | none |
| A30 | Stage-boundary state counted once | every multi-stage run's per-stage first-row weight | neither of the plan's sketched options (B)/(C) — see finding 5.7 |

G7 (default `state_dependent_approximation`) and G8 (`raw_data` rectangularisation) were
decided separately: G7 keeps `False` (decision 1); G8 is superseded by output format v2
(finding 5.6), which rectangularised `raw_data` outright.

## H: explicitly deferred (still open, tracked in `06`/`09`/`10`)

H1 (sub-chain semantics beyond what WS3 fixed — finding 1.1), H2 (zero-weight NN training,
decided as the `weighting=` policy, finding 3.1), H3 (collector busy-wait, 1 GiB buffer,
snapshot batching — Tier 4), H4 (RBF fallback, polynomial hardening — Tier 3), H5 (narrowing
`except BaseException` — done in WS9b, superseding this item), H6 (duplicate examples — done
in WS10, superseding this item).
