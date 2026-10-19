# 30 — C3 plan: tempered-trajectory Hamiltonian component and the prior-draw component (2026-10-12)

**Design note / plan.** **Implemented 2026-10-12 (same day)** as specified in §3–§6 with decisions D-C3-1 (reached
temperature), D-C3-2 (adaptive refused) taken as recommended; ~120 library lines (docstrings and validation included), 21
unit tests + 1 MPI test (`tests/unit/test_tempered_hamiltonian.py`, `tests/mpi/test_mpi_tempered.py`); untempered runs
byte-identical (pinned hashes). Deviations: the `tempered_skipped` count is a CONDITIONAL last column of `notes` (only for
tempering stages); the manifest's `stages` entries now list the two fields for every Hamiltonian spec; the MPI test uses a
fixed perturbed-linear evaluator instead of the polynomial surrogate (it has no gradients). §7 validation: study 31
(`31_c3_efficiency_study_2026-10-12.md`) on B2 and the new 3-D bimodal problem `toy_examples/solver_examples/bimodal3d.py`. Fills in the one-row item C3 of note 25 §9 ("tempered-dynamics
Hamiltonian component (`tempered_fraction`, `tempered_beta`, longer `T`; acceptance unchanged) — the crossing
mechanism that scales with dimension; prior-draw `PCN(beta=1)` component for low d"). Prerequisite C2 (`Mixture`,
adaptive `Block`) is done (2026-10-12, note 25 §9 status line). Consumer: C6 (the Auto proposal choice when the
warm-up chains disagree).

## 1. What the component is

**Neal's tempered trajectory** (Neal 2011, *MCMC using Hamiltonian dynamics*, §5.5.2 "tempered trajectories"),
not a tempered potential: the leapfrog runs on the ordinary posterior gradient, and the momentum is multiplied by a
constant factor `√α` at each of the first `n_t` steps ("heating") and divided by `√α` at each of the last `n_t`
("cooling"). Between heating and cooling the trajectory has the kinetic energy of a chain at temperature `α^{n_t}`,
so it can climb a barrier of order `n_t log α` nats that an untempered trajectory of the same length cannot; cooling
brings it back to a state the ordinary acceptance test can accept. The row's names fit this construction and not
the tempered potential (which needs a temperature schedule, not a fraction and a factor); this note fixes that
reading.

**Why it is the mechanism "that scales with dimension".** The prior-draw component (`PCN(beta=1)`: an independent
prior sample, acceptance = the likelihood ratio) crosses any barrier in low dimension but its acceptance decays
like the prior-to-posterior volume ratio, exponentially in `d`. A tempered trajectory moves along the posterior's
own dynamics and pays only for the energy it fails to return, which is set by `n_t`, `α` and the step, not by `d`
directly. Note 28 (bimodal 2-D, 11-nat barrier): only large moves crossed (pCN β = 1, random walk scale ≥ 1.5);
the ordinary Hamiltonian trapped chains in one arm.

## 2. Validity (why "acceptance unchanged")

Write one trajectory step as `F_j` (position update: `q += ε M⁻¹ p` for `leapfrog`, the exact prior rotation for
`dimension_robust`) and `K_j` (momentum kick by `ε` times the gradient), with the half kicks at both ends. The
untempered map is `K/2 · F_1 · K_1 · F_2 · … · K_{L-1} · F_L · K/2`, which is volume preserving and, with the final
momentum flip, an involution — hence the Metropolis test `min(1, exp(H(q,p) − H(q',p')))` is exact.

Tempering inserts the scaling `S: p ↦ √α p` immediately **before** `F_j` for `j = 1..n_t` and its inverse `S⁻¹`
immediately **after** `F_j` for `j = L+1−n_t..L`:

    K/2 · S F_1 K_1 · S F_2 K_2 · … · K_{L-n_t} · F_{L-n_t+1} S⁻¹ K_{L-n_t+1} · … · F_L S⁻¹ · K/2

* **Volume**: `S` has Jacobian `α^{d/2}`, `S⁻¹` has `α^{-d/2}`; equal counts `n_t` ⇒ the product has Jacobian 1.
* **Reversibility**: the operator list is a palindrome with `S ↔ S⁻¹` swapped, and `S` commutes with the momentum
  flip, so (flip ∘ map) is again an involution. The reversed trajectory heats where the forward one cooled.
* Therefore the existing `get_log_acceptance_probability` (likelihood + prior + `½ pᵀ M⁻¹ p` difference with
  `p_current_start` and `p_proposed_end`) is **unchanged**; only the integrators change. `n_t ≤ ⌊L/2⌋` is required
  (heating and cooling must not overlap); with `n_t = 0` the map is bit-identical to today's.
* The same argument holds for `dimension_robust`: the prior rotation `F` is a symplectic map and is reversible under
  the flip; the scaling is inserted around it exactly as above.
* Divergent-trajectory abandonment (non-finite `q`/`p` → return, rejected without a model call) stays as is; a
  heated trajectory is more likely to diverge, which is the correct signal (acceptance 0).
* In a DAMH stage the trajectory runs on surrogate gradients; the delayed-acceptance correction keeps the exact
  posterior invariant for any proposal map, so nothing in the DAMH argument changes. The surrogate is evaluated far
  from its training data more often (study 29 §3: out-of-box pre-rejections), which costs efficiency, not validity;
  the error model (S1) and the audit (S2) are the existing tools for that.

## 3. Interface

`Hamiltonian` gains two fields (defaults = off, so every existing configuration is byte-identical):

| field | type / default | meaning |
|---|---|---|
| `tempered_fraction` | `float` in `[0, 0.5]`, `0.0` | share of the trajectory's steps tempered at EACH end: `n_t = min(⌊f L⌋, ⌊L/2⌋)` per trajectory (with jitter, `L` is the trajectory's own count) |
| `tempered_beta` | `float` in `(0, 1]`, `1.0` | inverse temperature reached between heating and cooling: the per-step factor is `α = β^{-1/n_t}`, so the hottest kinetic temperature is `1/β` **independently of `L`** (and so of the adapted step) |

Rationale for `β` instead of the per-step `α`: `L = T/ε` changes whenever the step adapts or the jitter draws a
new length; a fixed per-step factor would make the reached temperature depend on the step. `n_t = 0` (short
trajectories, `L < 2`) ⇒ no tempering; counted per stage (`notes` column `tempered_skipped`) and printed at stage
end, because on G20-like problems the adapted step gives `L = 1–3` (study 29 §2.3) and the component would be
silently ineffective.

Validation in `__post_init__`: `tempered_fraction > 0` requires `tempered_beta < 1` and vice versa (one without
the other is a configuration error, not a silent no-op); `tempered_fraction > 0` with `adaptive=True` raises (see
§4). `describe()`/manifest (`proposal_resolution`): `tempered_fraction`, `tempered_beta`.

**Prior-draw component**: `PCN(beta=1.0)` already works (`PCN` runtime asserts `0 < β ≤ 1`; `β = 1` proposes
`m + η`, the acceptance is the likelihood ratio). Nothing to implement; the `PCN` docstring says "`beta` in (0, 1)"
and should say `(0, 1]` with one sentence on the prior draw.

**Mixture use** (the intended way; C2):

    Mixture([Hamiltonian(integrator="dimension_robust", mass="covariance", integration_time=3, integration_time_jitter=0.5),
             Hamiltonian(integrator="dimension_robust", mass="covariance", integration_time=9, integration_time_jitter=0.5,
                         tempered_fraction=0.3, tempered_beta=0.1, adaptive=False)],
            probabilities=[0.8, 0.2])

The tempered component is pinned (`adaptive=False`, `step_size=None`): it takes the carried `step_size` of the
last adaptive Hamiltonian stage/chunk (in Auto: the previous chunk's frozen step) and the same `mass="covariance"`.
C2's V1 rule allows this (one adaptive Hamiltonian per mixture). Its `T` is the ordinary one times 2–4 ("longer
`T`" of the row): the heated middle part needs room to travel.

## 4. Adaptation: the tempered component does NOT adapt its step

The dual averaging targets a sub-chain/MH acceptance rate; a tempered component's acceptance is lower by design
and depends on the barrier, so the recursion would shrink its step without bound (the ceiling-below-target
mechanism of study 29 §2.6, here by construction). Hence `tempered_fraction > 0` with `adaptive=True` is refused.
What is tuned instead: nothing in C3. `β`, `f` and the mixing probability are placeholders for C6/C8 (note 25 §6
validation). The component reads the ordinary component's step through the carry-over at stage/chunk start; within
a chunk it is fixed.

## 5. Implementation (≈ 60 library lines)

| piece | where | lines |
|---|---|---|
| spec fields + validation + docstring | `surrDAMH/proposals.py` `Hamiltonian` | 20 |
| runtime: `tempered_fraction`/`tempered_beta` kwargs, `_tempering_counts(L) -> (n_t, sqrt_alpha)`, the two `_leapfrog` methods get the `S`/`S⁻¹` lines at the places of §2 (one helper `_temper(p, j, L)` used by both; the untempered path must not change a single float operation) | `surrDAMH/modules/proposals.py` `Hamiltonian`, `HamiltonianInfinite` | 30 |
| builder pass-through, `resolution` entries, `describe()` note, `notes` counter `tempered_skipped` | `proposal_builder.py`, `stages.py`, `algorithms.py` (one counter, like `out_of_bounds`) | 15 |
| `PCN` docstring `(0, 1]` | `surrDAMH/proposals.py` | 2 |

Not in scope: Auto (C6 decides when to use the mixture), per-component trace files (C2 cut), any change to the
acceptance code or the exact step (S2).

## 6. Tests (≈ 120 lines)

1. **Off = byte-identical**: `tempered_fraction=0` gives the same sample stream as today (both integrators;
   reuse the pinned-fingerprint pattern of `tests/unit/test_mixture.py` or compare two runs with the field absent
   vs 0.0 — the latter is machine-independent and sufficient).
2. **Map properties on a 2-D quadratic potential** (exact gradients, both integrators, `L = 6`, `n_t = 2`,
   `α = 2`): (a) involution — apply the map, flip `p`, apply again, flip: returns `(q, p)` to 1e-12;
   (b) volume — finite-difference Jacobian determinant of `(q,p) ↦ (q',p')` equals 1 to 1e-6;
   (c) `n_t` and `α` from `(f, β, L)`: `n_t ≤ ⌊L/2⌋`, `α^{n_t} = 1/β`, `L < 2 ⇒ n_t = 0`.
3. **Detailed balance / invariance**: MH stage with the tempered Hamiltonian alone on a 2-D Gaussian posterior
   (`run_sampling_local`, exact gradient evaluator): moments within SE; acceptance lower than untempered but
   > 0.05 at `β = 0.5`.
4. **Spec validation**: the four error cases of §3/§4.
5. **Mixture**: `Mixture([Hamiltonian(), Hamiltonian(tempered..., adaptive=False)])` builds; the pinned one takes
   the carried step; manifest shows the two fields and `adaptive_stats_component == 0`.
6. **MPI** (one case, `tests/mpi`): the §3 mixture in a DAMH stage on the B-series linear-Gaussian problem with the
   polynomial surrogate, 4 ranks: completes, moments within 4 SE, `tempered_skipped` counted.

## 7. Validation runs before the component enters Auto (C8 material, not C3)

* **B2** (study 29 setup, `toy_examples/tmp_step_size_study_2026-10-12/scripts/problems.py`): arm crossings per run and
  left-arm mass for `Mixture([H adaptive, H tempered])` with `p_t ∈ {0.1, 0.2, 0.4}`, `β ∈ {0.3, 0.1, 0.03}`,
  `f ∈ {0.2, 0.35}`, `T_t = 3 T`, vs the plain adaptive Hamiltonian and vs `Mixture([H, PCN(beta=1)])`; 3 reps.
  Metric: crossings per 1 000 exact evaluations and the fraction of crossings made by the tempered component
  (`subchain_stats` has no component column for mixtures — add the chosen component index to `subchain_stats`
  when this study is run; ~10 lines, output-format addition).
* **Bimodal in `d = 10, 20`**: solver `G(u) = (u_1², u_2, …, u_d)` observed at `(a², 0, …, 0)` with `a = 2`,
  noise sd 0.3: posterior bimodal in `u_1` (modes near `±a`, barrier `≈ a⁴/(2·0.3²)` nats scaled by the prior), Gaussian
  in the rest; reference by 1-D quadrature in `u_1`. Same grid as B2. This is the "scales with dimension" check:
  prior-draw crossings should collapse with `d`, tempered ones should not.
* Cost: `n_t` extra momentum scalings are free; the cost is the longer `T_t` (2–4× the gradients of the ordinary
  component on 10–40 % of the iterations) — report ESS/eval, ESS/s and ESS per network gradient (study 29 §3
  caveat).

## 8. Decisions for the author

* D-C3-1: `tempered_beta` as the reached inverse temperature (this note) vs Neal's per-step `α`. Recommendation:
  reached temperature (independent of the adapted step).
* D-C3-2: refuse `adaptive=True` on a tempered component (this note) vs allow with a separate target rate.
  Recommendation: refuse in C3; revisit after the §7 runs show what acceptance a useful tempered component has.
* D-C3-3: placeholders for C6: `p_t = 0.2`, `β = 0.1`, `f = 0.3`, `T_t = 3 T`. To be set by §7, not by this note.
