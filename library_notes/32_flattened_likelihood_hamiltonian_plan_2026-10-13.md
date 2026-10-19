# 32 — Plan: flattened-likelihood Hamiltonian component, and its comparison with the tempered-momentum one (2026-10-13)

**Design note / plan.** **§1 implemented 2026-10-13** (`Hamiltonian(likelihood_weight=τ)`, both integrators via
`_likelihood_force`, byte-identical at τ = 1; `subchain_stats` `component` column for MIXTURE stages only, −1 on exact-step
rows — D-32-3 taken as "mixture stages"; D-32-4 relative step NOT implemented by author decision; ~69 library lines, 17
tests + 1 MPI: `tests/unit/test_flattened_hamiltonian.py`, `tests/unit/test_subchain_component_column.py`,
`tests/mpi/test_mpi_flattened.py`); `bimodal3d.py` generalised to any `d` (§2.1). §2 = study 33
(`33_flattened_vs_tempered_study_2026-10-13.md`), half steps as absolute numbers per problem. Follows study 31 (`31_c3_efficiency_study_2026-10-12.md`): the C3
tempered-momentum component (note 30) crosses the valleys of the 3-D bimodal problem only with a halved step and
β = 0.001, and never works on B2, where the prior draw `PCN(beta=1)` is the only kernel that passes. The alternative
studied here keeps the ordinary kinetic energy and lowers the barrier the trajectory sees instead: the leapfrog
integrates the gradient of `τ · log L + log prior` with `0 ≤ τ < 1`, the acceptance uses the exact energy. Its
`τ = 0` limit with the dimension-robust integrator is the prior rotation, whose quarter period is exactly the prior
draw — so the option interpolates between plain Hamiltonian and `PCN(beta=1)` along one line. The question of the
comparison is which of the two mechanisms crosses at lower cost, and which survives the dimension.

## 1. The option

### 1.1 Dynamics and validity
Untempered trajectory: kicks `p ← p − ε (∇log L + ∇log prior)` (leapfrog) or `p ← p − ε ∇log L` with the exact prior
rotation (`dimension_robust`). Flattened: every kick uses `τ ∇log L` instead of `∇log L`; the prior part, the position
updates, the half kicks, the finiteness checks and the flip are unchanged. A kick by any position-dependent force keeps
the map volume preserving (shear) and, with the symmetric sequence, an involution under the flip — the same argument as
note 30 §2 — so the Metropolis test with the EXACT `H` preserves the exact (box-truncated) posterior, in MH and in DAMH
stages alike. The endpoint energy error is, to first order, `(1 − τ)` times the change of `log L` along the path: a
jump between equally likely modes is accepted, a jump to a worse place is penalised in proportion.

Combination with C3 (both on) is valid by the same argument; the study tests each alone and one combined row.

### 1.2 Interface
`Hamiltonian` gains one field, `likelihood_weight: float = 1.0` in `[0, 1]` (τ). Validation: range; `likelihood_weight
< 1` with resolved `adaptive=True` raises for the same reason as C3's D-C3-2 (its acceptance is lower by design; the
dual averaging would shrink the step; see note 30 §4). `describe()` one note line; `proposal_resolution` gains
`likelihood_weight` only when `< 1`; the manifest `stages` entries list it for every Hamiltonian spec (as C1/C3 fields do;
to be stated in the CHANGELOG). `τ = 0` is allowed (pure prior dynamics; with `dimension_robust` and `T = π/2` this is a
prior draw, with `T` jittered a partial one).

### 1.3 Implementation (≈ 20 library lines)
| piece | where | lines |
|---|---|---|
| field, validation, docstring | `surrDAMH/proposals.py` | 10 |
| `__init__` kwarg; in both `_leapfrog`s the six gradient calls (lines ~907/919/922 and ~994/1006/1009) become `self._lw * self.log_likelihood_gradient(q)` ONLY when `self._lw != 1.0` — a branch, so the untempered path executes the same float operations (byte-identity) | `surrDAMH/modules/proposals.py` | 8 |
| builder pass-through + resolution; `describe()` line | `proposal_builder.py`, `stages.py` | 6 |

### 1.4 Tests (≈ 80 lines, `tests/unit/test_flattened_hamiltonian.py`)
Off = byte-identical (explicit `1.0` ≡ default, machine-independent; the pinned C3 fingerprints must still pass);
involution and unit Jacobian of the map at `τ = 0.3` on the 2-D quadratic of C3's tests, both integrators; invariance on a
2-D Gaussian via `run_sampling_local` with an exact evaluator at `τ = 0.3` (moments within 4 SE); `τ = 0`,
`dimension_robust`, `integration_time = π/2`, no jitter, `mass = 1`: the proposal equals `m + η` in law — check that the
proposed point is independent of the start (correlation of 2 000 proposals with their starts ≈ 0); spec validation;
the mixture builds and the pinned component takes the carried step.

## 2. Comparison study (note 33 when run)

### 2.1 Problems
* **B2** and **M3-{5, 10, 20}** exactly as study 31 (its `scripts/problems.py`, references, stage layout, budget
  4 000 evaluations per chain, 8 samplers + collector, network (32, 32), error model on, Polyak on).
* **M-d, the dimension test** (the reason the roadmap wanted a Hamiltonian crossing mechanism): generalise
  `toy_examples/solver_examples/bimodal3d.py` to `no_parameters = d` with `G(u) = (Σ u_i)² / d²`, `y = 1 + d σ²/2`,
  `σ = (2Δ)^{-1/2}`: modes exactly at `±(1, …, 1)`, saddle at the origin, `log π(mode) − log π(0) = Δ` (same algebra as
  the 3-D case; the perpendicular part is `N(0, I_{d−1})`, the reference is the same 1-D quadrature in `t = u·1/√d`).
  Run `d ∈ {3, 10, 30}` at `Δ = 10`, network (32, 32) for `d ≤ 10`, (64, 64) for 30, `min_snapshots_initial = 10 d`,
  held-out 200. **Prediction to test**: the prior draw's crossings fall exponentially with `d`; the flattened
  component with `τ > 0` should fall more slowly; the tempered-momentum one should not fall. Any of the three failing
  that is the result.

### 2.2 Configurations (measured stage 3 of the study-31 layout; all mixtures `[H_ADAPT, component]`)
| label | component | grid |
|---|---|---|
| plain | `H_ADAPT` alone | 1 |
| prior draw | `PCN(beta=1)`, `p = 0.2` | 1 |
| tempered-momentum, study-31 best | `tempered_fraction 0.35, tempered_beta 0.001, T_t = 3T, step ½ carried` (needs the relative-step option of item 21(f), or the absolute half step per problem as study 31 did), `p = 0.3` | 1 |
| flattened, carried step | `likelihood_weight τ ∈ {0.5, 0.2, 0.05, 0}`, `T_t ∈ {T, 3T}`, `p = 0.3`, carried step | 8 |
| flattened, half step | `τ ∈ {0.2, 0.05}`, `T_t = 3T`, step ½ carried | 2 |
| combined | `τ = 0.2` + `tempered_beta 0.03, f 0.35`, `T_t = 3T`, half step | 1 |

14 configurations × (B2, M3-5, M3-10, M3-20, M10-10, M30-10) × 3 repetitions ≈ 250 runs ≈ 1.5–2 h at study-31
timings (the `d = 30` runs are slower; budget 3 000 per chain there if needed, say so).

### 2.3 Metrics (study 31's `analyze.py` plus)
Crossings per 1 000 measured-stage iterations and the mode mass; **mode ESS** (study 31's between-chain estimate from
the per-chain mode fractions) and the **split R-hat of the mode indicator** (`1[t > 0]`, 2 splits per chain, Vehtari et
al. 2021 rank-normalised not needed for a 0/1 variable) as the mixing verdict — `R̂_mode < 1.05` and all chains crossed
at least once = "mixed"; min-over-parameters ESS/eval and ESS/s reported but NOT used for ranking (item 21(h));
out-of-box and pre-rejection rates; outer acceptance; wall time; for the flattened component the acceptance of its own
iterations is not observable per component (no component column in `subchain_stats`, item 21(d)) — add the drawn
component index as a `subchain_stats` column before this study (~10 lines, output-format addition for mixture stages;
decision D-32-3) so that per-component acceptance and crossings-by-component can finally be reported.

### 2.4 Questions the report answers
(i) crossings and mixing verdict per mechanism and problem; (ii) cost per crossing in exact evaluations, in network
gradients (`T_t/ε` × share) and in seconds; (iii) does the flattened component need the halved step (expected not —
it keeps the prior pull); (iv) the dimension curve `d = 3 → 10 → 30` for the three mechanisms; (v) does combining help;
(vi) recommendation for the C6 placeholders with numbers, or the statement that neither mechanism is robust enough.

## 3. Decisions for the author
* D-32-1: field name `likelihood_weight` (τ) vs `tempered_potential`/`flatten`. Recommendation: `likelihood_weight`.
* D-32-2: allow `τ = 0`. Recommendation: yes (it is the prior-draw limit and a useful control).
* D-32-3: add the drawn component index to `subchain_stats` for mixture stages (output-format addition) before the
  study. Recommendation: yes, it unblocks per-component acceptance and crossing attribution, which study 31 lacked.
* D-32-4: the relative-step option for pinned components (`step_factor`, item 21(f)) before the study, so the
  "½ carried" rows are expressed in the API rather than as per-problem absolute numbers. Recommendation: yes (~20 lines).
