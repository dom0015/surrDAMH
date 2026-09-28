# DAMH-SMU as an adaptive MCMC algorithm: assumptions, proof of validity, and what they demand of the NN surrogate (2026-09-22)

Scope: the DAMH stage with `surrogate_model_updates=True` (DAMH-SMU), i.e. the surrogate is
retrained from the chain's own exact evaluations while the chain runs. Fixed-surrogate DAMH
is only used as a lemma. `use_only_surrogate=True` stages are out of scope (they do not target
the posterior by construction).

Status of the content: §3–§4 is a self-contained proof modulo two textbook facts (maximal
coupling; Doeblin's theorem) and one cited theorem (the WLLN of Roberts & Rosenthal 2007,
used but not re-proved). §5–§6 is my own reading of the code against the assumptions;
line numbers refer to the working tree on 2026-09-22 and nothing was executed.

---

## 0. Result first

DAMH-SMU is a valid MCMC algorithm — `sup_A |P(X_n ∈ A) − π(A)| → 0` and ergodic averages of
bounded functions converge in probability to their posterior expectations — under the
following assumptions:

| | Assumption | Structural (code) or analytic? | NN surrogate today |
|---|---|---|---|
| I1 | One surrogate version for the **entire** outer transition: all `K` sub-chain tests and the stage-2 ratio `ℓ̃_γ(y) − ℓ̃_γ(x)` use the same version | code invariant, **holds** (`algorithms.py:692-702, 746`) | ok |
| I2 | Stage 2 always uses the exact solver (the only skip is the no-op `y = x`) | code invariant, **holds** (`algorithms.py:797-798, 840-864`) | ok |
| I3 | Installed evaluator is a frozen snapshot | **holds** for NN (`torch_perceptron_minibatches.py:93, 109-118`); MPI pickling holds for all | ok |
| A1 | Compact state space `X` (prior with bounded support) | analytic | **fails** with Gaussian priors on `ℝ^d` — see §5.3 |
| A2 | Stage-1 proposal density bounded above and below on `X × X` | analytic; holds for RW / pCN on compact `X` | ok |
| A3 | Exact log-likelihood bounded on `X` | analytic; holds for a continuous forward map | ok modulo solver failures |
| A4 | Surrogate log-likelihood **uniformly** bounded over every version that can ever be installed | must be **engineered** | **fails** — unbounded outputs, unbounded weights |
| A5 | Diminishing adaptation: `sup_x |ℓ̃_{n+1}(x) − ℓ̃_n(x)| → 0` in probability | must be **engineered** | **fails** — fixed `lr × iterations_batch` per retrain, retrain on every snapshot |
| A6 | Fresh randomness: the draws used for step `n+1` are independent of everything that decided which surrogate is installed at step `n` | code invariant, **holds** (async MPI included) | ok |

A1–A4 give **simultaneous uniform ergodicity** of the family of DAMH kernels; A5 gives
**diminishing adaptation**; together they are the hypotheses of Roberts & Rosenthal (2007).
A4 and A5 are the two that the NN surrogate does not satisfy today, and §6 gives three
implementable ways to make it satisfy them (Polyak-averaged weight publication, a decaying
training budget, or randomised installation), each with the bound that closes the proof.
Without A1 the proof does not go through; on `ℝ^d` containment has to be assumed (§5.3).

What is *not* needed: surrogate accuracy, surrogate convergence to the true model,
consistency of the fit, independence of the training data from the chain. The surrogate
only has to stop moving (A5) and stay tame (A4).

---

## 1. Setting and notation

* State space `X ⊂ ℝ^d` with Lebesgue measure `λ`; prior density `p`; exact log-likelihood
  `ℓ(x) = log L(x)`; posterior `π(dx) ∝ p(x) e^{ℓ(x)} dx`.
* A surrogate is an index `γ ∈ Γ` (for the NN: the weight vector after clipping/projection,
  see §6); it defines a surrogate log-likelihood `ℓ̃_γ(x)` and the stage-1 target
  `π̃_γ(dx) ∝ p(x) e^{ℓ̃_γ(x)} dx`. In the code `ℓ̃_γ(x)` is the likelihood evaluated at
  `observations_approx = G̃_γ(x)` (`algorithms.py:36-39, 296-302`).
* Stage-1 proposal density `q(x, y)` (RW or pCN; the Hamiltonian sub-chain is discussed in
  §5.4). Stage-1 MH kernel for surrogate `γ`:

      M_γ(x, dy) = q(x, y) α₁^γ(x, y) dy + δ_x(dy) (1 − ∫ q(x, z) α₁^γ(x, z) dz),
      α₁^γ(x, y) = min{1, π̃_γ(y) q(y, x) / (π̃_γ(x) q(x, y))}.

  Sub-chain of length `K` (`subchain_length`): `Q_γ = M_γ^K`.
* Outer DAMH kernel:

      P_γ(x, dy) = Q_γ(x, dy) α₂^γ(x, y) + δ_x(dy) (1 − ∫ Q_γ(x, dz) α₂^γ(x, z)),
      α₂^γ(x, y) = min{1, exp[(ℓ(y) − ℓ(x)) − (ℓ̃_γ(y) − ℓ̃_γ(x))]}      (`algorithms.py:827-828`).

  The pre-rejection branch (`algorithms.py:840-864`) is the event that `Q_γ` returned `y = x`;
  then `α₂ = 1` and the kernel above already describes it.
* Adaptive process: `(X_n, Γ_n)_{n ≥ 0}`; `𝓕_n` = σ-field of everything observed up to outer
  step `n` (all chains' states, all evaluations, all published surrogates, the collector's
  own randomness and message timings). The algorithm satisfies

      P(X_{n+1} ∈ A | 𝓕_n) = P_{Γ_n}(X_n, A)          (★)

  with `Γ_n` `𝓕_n`-measurable. (★) is exactly what invariants I1, I2, I3, A6 buy: I1 makes
  the transition a transition of *one* kernel `P_γ`, I2 makes that kernel the one written
  above, I3 makes `γ` a fixed object during the step, A6 makes the conditional law correct.
* Total variation: `‖μ − ν‖ := sup_A |μ(A) − ν(A)| ∈ [0, 1]`.
* `D_n := sup_{x ∈ X} ‖P_{Γ_{n+1}}(x, ·) − P_{Γ_n}(x, ·)‖`.
* `Δ_n := sup_{x ∈ X} |ℓ̃_{Γ_{n+1}}(x) − ℓ̃_{Γ_n}(x)|`.

## 2. Assumptions (precise form)

* **A0** (standing regularity) `X` is Polish, `Γ` is a measurable space, `(x, γ) ↦ ℓ̃_γ(x)` is
  jointly measurable and `x ↦ ℓ̃_γ(x)` is continuous for each `γ`. This makes
  `(x, γ) ↦ P_γ(x, A)` jointly measurable (so `P_{Γ_n}(X_n, A)` is a random variable) and the
  suprema defining `Δ_n`, `D_n` measurable. Satisfied by any finite MLP with a continuous
  activation, indexed by its weight vector.
* **A1** `X` is compact with `λ(X) ∈ (0, ∞)` and `0 < p_min ≤ p(x) ≤ p_max` on `X`.
* **A2** `0 < q_min ≤ q(x, y) ≤ q_max < ∞` for all `x, y ∈ X`.
* **A3** `|ℓ(x)| ≤ B_ℓ < ∞` on `X`.
* **A4** `sup_{γ ∈ Γ} sup_{x ∈ X} |ℓ̃_γ(x)| ≤ B̃ < ∞`, where `Γ` is the set of surrogates that
  can ever be installed.
* **A5** `Δ_n → 0` in probability.
* **A6** (★) holds.

Remarks. A1 excludes Gaussian priors on `ℝ^d` unless truncated (§5.3). A2 holds for a
Gaussian random walk and for pCN restricted to a compact `X` (a Gaussian density is bounded
above and, on a bounded set of arguments, below). A3 holds if the forward map is continuous
on `X`; the failed-solver convention `ℓ = −∞` (`algorithms.py:47`) violates it unless the
failure set is `π`-null — see §5.2. A4 says nothing about accuracy; it forbids the surrogate
from predicting arbitrarily bad misfits. A5 is the only assumption that couples consecutive
surrogate versions.

## 3. Theorem

Under A0–A6,

1. `sup_A |P(X_n ∈ A) − π(A)| → 0` as `n → ∞` (for any starting point and any initial
   surrogate), and
2. for every bounded measurable `f`, `(1/n) Σ_{i=1}^{n} f(X_i) → π(f)` in probability.

Part 1 is proved in full below (Lemmas 1–3 and Proposition 4). Part 2 is the WLLN of
Roberts & Rosenthal (2007) for adaptive chains satisfying diminishing adaptation and
containment; Lemma 3 gives the stronger simultaneous uniform ergodicity, so their hypotheses
hold. I have not re-derived part 2 (it is a bounded-differences/martingale argument on top of
the same coupling) and the theorem number in RR07 is not re-checked here.

## 4. Proof

### Lemma 1 (each `P_γ` is `π`-reversible)

`M_γ` is a Metropolis–Hastings kernel with target `π̃_γ`, hence `π̃_γ(dx) M_γ(x, dy)` is a
symmetric measure on `X × X`. A power of a reversible kernel is reversible for the same
measure (induction: `π̃(dx) M^{k+1}(x, dy) = ∫ π̃(dx) M(x, dz) M^k(z, dy)`, apply symmetry of
`π̃(dx) M(x, dz)` then of `π̃(dz) M^k(z, dy)`), so `π̃_γ(dx) Q_γ(x, dy)` is symmetric.

Write `r_γ(x) := e^{ℓ(x) − ℓ̃_γ(x)}`, finite and positive whenever `ℓ` and `ℓ̃_γ` are
real-valued (A3–A4 are more than enough; boundedness is used only in Lemmas 2–3); note
`π(dx) = c_γ r_γ(x) π̃_γ(dx)` and `α₂^γ(x, y) = min{1, r_γ(y)/r_γ(x)}` (the prior cancels,
which is why the code only needs likelihood ratios). On all of `X × X` — including the atom
of `Q_γ` at `y = x` (the all-rejected sub-chain), where `α₂ = 1` and `min{r, r} = r` —

    π(dx) Q_γ(x, dy) α₂^γ(x, y) = c_γ π̃_γ(dx) Q_γ(x, dy) · r_γ(x) min{1, r_γ(y)/r_γ(x)}
                                = c_γ π̃_γ(dx) Q_γ(x, dy) · min{r_γ(x), r_γ(y)},

a symmetric measure. The outer-rejection part `π(dx) δ_x(dy)(1 − …)` is carried by the
diagonal, and any measure of the form `μ(dx) δ_x(dy)` is invariant under `(x, y) ↦ (y, x)`.
Hence `π(dx) P_γ(x, dy)` is symmetric, i.e. `P_γ` is `π`-reversible and `π P_γ = π`. ∎

(This is Christen & Fox 2005 in measure form. Nothing about the quality of `ℓ̃_γ` is used;
what is used is that `ℓ̃_γ` is one fixed function during the transition — invariant I1 — and
that stage 2 evaluates the true `ℓ` — invariant I2.)

### Lemma 2 (kernel continuity in the surrogate ⇒ diminishing adaptation)

For `γ, γ' ∈ Γ` put `Δ := sup_{x ∈ X} |ℓ̃_γ(x) − ℓ̃_{γ'}(x)|`. Then

    sup_{x ∈ X} ‖P_γ(x, ·) − P_{γ'}(x, ·)‖ ≤ (8K + 4) Δ.

In particular `D_n ≤ (8K + 4) Δ_n`, so A5 implies `D_n → 0` in probability.

Proof. (i) `a ↦ min{1, e^a}` is 1-Lipschitz on `ℝ`. The log-ratio inside `α₁` changes by at
most `2Δ` between `γ` and `γ'` (two surrogate evaluations, everything else identical), so
`|α₁^γ(x, y) − α₁^{γ'}(x, y)| ≤ 2Δ`; likewise `|α₂^γ − α₂^{γ'}| ≤ 2Δ`.

(ii) One stage-1 step. For a measurable `A`,

    |M_γ(x, A) − M_{γ'}(x, A)| ≤ ∫_A q(x, y) |α₁^γ − α₁^{γ'}| dy + 1_A(x) |∫ q(x, z)(α₁^γ − α₁^{γ'}) dz| ≤ 4Δ,

so `sup_x ‖M_γ(x, ·) − M_{γ'}(x, ·)‖ ≤ 4Δ`. (In the `sup_A` convention this is actually
`≤ 2Δ`, since `sup_A |μ(A) − ν(A)| = ½ |μ − ν|(X)` for probability measures; the cruder `4Δ`
is kept so that the constant below is a plain upper bound. The sharp constant of the lemma
is `(4K + 2)Δ`.)

(iii) `K` steps. With `‖·‖` the sup-over-`x` TV distance between kernels, the telescoping
`M_γ^K − M_{γ'}^K = Σ_{j=0}^{K−1} M_γ^{j} (M_γ − M_{γ'}) M_{γ'}^{K−1−j}` and the facts that
Markov kernels are TV-contractions on the left and on the right give
`sup_x ‖Q_γ(x, ·) − Q_{γ'}(x, ·)‖ ≤ 4KΔ`.

(iv) Outer step. Using `|∫ f d(μ − ν)| ≤ osc(f) · ‖μ − ν‖ ≤ ‖μ − ν‖` for `0 ≤ f ≤ 1` — valid
because `μ − ν = (Q_γ − Q_{γ'})(x, ·)` has zero total mass (without that, the step would
need `2 sup f`) —

    |P_γ(x, A) − P_{γ'}(x, A)|
      ≤ |∫_A (Q_γ − Q_{γ'})(x, dy) α₂^γ(x, y)| + |∫_A Q_{γ'}(x, dy)(α₂^γ − α₂^{γ'})(x, y)|
        + 1_A(x) |∫ (Q_γ − Q_{γ'})(x, dz) α₂^γ + ∫ Q_{γ'}(x, dz)(α₂^γ − α₂^{γ'})|
      ≤ (4KΔ + 2Δ) + (4KΔ + 2Δ) = (8K + 4)Δ.  ∎

The constant is crude and irrelevant; what matters is that the DAMH kernel is *Lipschitz in
the surrogate log-likelihood in sup norm*. That is the whole content of "diminishing
adaptation" for this algorithm, and it is why A5 is stated in terms of `ℓ̃` rather than
weights.

### Lemma 3 (simultaneous uniform ergodicity)

Under A1–A4 there is `ε > 0` and a probability measure `ν` on `X` such that
`P_γ(x, ·) ≥ ε ν(·)` for all `x ∈ X`, `γ ∈ Γ`. Consequently

    ‖P_γ^n(x, ·) − π‖ ≤ (1 − ε)^n      for all n, x ∈ X, γ ∈ Γ.

Proof. By A1, A2, A4,

    α₁^γ(x, y) ≥ (p_min / p_max) (q_min / q_max) e^{−2B̃} =: a₁ > 0,

so `M_γ(x, A) ≥ q_min a₁ λ(A)` for `A ⊂ X` (proposals falling outside `X` have prior zero
and are rejected into the atom, which only helps) and, composing `K` times through the
absolutely continuous part, `Q_γ(x, A) ≥ (q_min a₁)^K λ(X)^{K−1} λ(A)`. By A3, A4,
`α₂^γ ≥ e^{−2B_ℓ − 2B̃} =: a₂ > 0`. Hence

    P_γ(x, A) ≥ a₂ Q_γ(x, A) ≥ ε ν(A),   ν := λ(· ∩ X)/λ(X),   ε := a₂ (q_min a₁)^K λ(X)^K,

with `ε ∈ (0, 1]` independent of `x` and `γ`. `P_γ` has invariant measure `π` (Lemma 1), and
Doeblin's theorem — the one-step contraction `‖μP_γ − νP_γ‖ ≤ (1 − ε)‖μ − ν‖` in the `sup_A`
convention, with `‖δ_x − π‖ ≤ 1` — gives `‖P_γ^n(x, ·) − π‖ ≤ (1 − ε)^n`. ∎

(`ε` is astronomically small in any real problem, e.g. `(q_min a₁)^K` with `K = 10`. That is
irrelevant for the qualitative theorem, and it is the reason nothing quantitative should be
read into it. It is also why containment — the weaker condition in RR07 — is the condition
one would want in practice; SUE is simply what compactness gives for free.)

### Proposition 4 (simultaneous uniform ergodicity + diminishing adaptation ⇒ ergodicity)

Assume (★), `D_n → 0` in probability, and: for every `ε > 0` there is `N` with
`‖P_γ^N(x, ·) − π‖ ≤ ε` for all `x, γ`. Then `sup_A |P(X_n ∈ A) − π(A)| → 0`.

Proof (coupling; this is the argument of Roberts & Rosenthal 2007 written out for this
setting). Fix `ε > 0` and the corresponding `N`. For `n ≥ N` put `m := n − N`.

*Construction.* Enlarge the probability space so that, from time `m` on, the algorithm's
chain `X` is generated jointly with an auxiliary chain `Y` (a fresh `Y` for each `n`, i.e. a
triangular array): set `Y_m := X_m`; for `k = m, …, n − 1`, given `𝓕_k` and `Y_{m:k}`, draw
`(X_{k+1}, Y_{k+1})` from a maximal coupling of `P_{Γ_k}(X_k, ·)` and `P_{Γ_m}(Y_k, ·)` (a
jointly measurable selection of maximal couplings exists on a Polish space, A0). Then,
conditionally on `(𝓕_k, X_{k+1})`, draw the step's *internal* variables — the sub-chain path
and the exact evaluations at rejected stage-2 proposals — from their conditional law under
the algorithm, and let the collector update `Γ_{k+1}` from those, the accepted state, and its
own fresh randomness exactly as the code does. This matters because the surrogate is trained
on rejected proposals too (`algorithms.py:348-351`), so `Γ_{k+1}` is *not* a function of the
`X`-path alone. Since the conditional law of `X_{k+1}` given `𝓕_k` and `Y_{m:k}` is
`P_{Γ_k}(X_k, ·)`, the joint law of `(X, Γ, internals)` is exactly that of the algorithm,
i.e. (★) holds; the auxiliary chain only reads `Γ_m`, which is fixed at time `m`.

*The auxiliary chain is close to `π`.* The `Y`-marginal of each maximal coupling is
`P_{Γ_m}(Y_k, ·)` by construction, so by the tower property, conditionally on `𝓕_m`,
`(Y_k)_{k ≥ m}` is a time-homogeneous Markov chain with kernel `P_{Γ_m}` started at `X_m`.
Applying the
hypothesis with `γ = Γ_m`, `x = X_m` (both `𝓕_m`-measurable),

    ‖P(Y_n ∈ · | 𝓕_m) − π‖ ≤ ε   a.s.,   hence   sup_A |P(Y_n ∈ A) − π(A)| ≤ ε.

*The two chains rarely separate.* On `{X_k = Y_k = x}` the maximal coupling fails with
conditional probability exactly `‖P_{Γ_k}(x, ·) − P_{Γ_m}(x, ·)‖` (this is where the `sup_A`
convention is load-bearing) `≤ Σ_{j=m}^{k−1} D_j` (triangle inequality and the definition
of `D_j` as a sup over `x`). Since `X_m = Y_m`, `{X_n ≠ Y_n} ⊂ ∪_k {X_k = Y_k, X_{k+1} ≠ Y_{k+1}}`,
and since `1 ∧ (a + b) ≤ 1 ∧ a + 1 ∧ b` for `a, b ≥ 0`,

    P(X_n ≠ Y_n) ≤ Σ_{k=m}^{n−1} E[1 ∧ Σ_{j=m}^{k−1} D_j] ≤ N Σ_{j=n−N}^{n−2} E[1 ∧ D_j].

Each `E[1 ∧ D_j] → 0` as `j → ∞` (convergence in probability of a variable bounded by 1),
and the sum has at most `N` terms with `N` fixed, so `P(X_n ≠ Y_n) → 0` as `n → ∞`.

*Conclusion.* For every `A`,
`|P(X_n ∈ A) − π(A)| ≤ P(X_n ≠ Y_n) + |P(Y_n ∈ A) − π(A)| ≤ P(X_n ≠ Y_n) + ε`, so
`limsup_n sup_A |P(X_n ∈ A) − π(A)| ≤ ε`; `ε` was arbitrary. ∎

### Proof of the Theorem

Lemma 1 gives `π P_γ = π` for all `γ`, which Lemma 3 uses. Lemma 3 gives the uniform-in-`γ`
convergence hypothesis of Proposition 4 (with `N = ⌈log ε / log(1 − ε_Doeblin)⌉`). Lemma 2 and
A5 give `D_n → 0` in probability. A6 gives (★). Proposition 4 then gives part 1. Part 2 is
RR07's WLLN under the same hypotheses. ∎

### Remarks on the proof

* **Where the correlation problem went.** The naive argument "each `P_γ` is `π`-invariant,
  so if `X_n ~ π` then `X_{n+1} ~ π`" is false because `Γ_n` and `X_n` are dependent
  (`E[(P_{Γ_n} f)(X_n)]` need not equal `π(f)`). Proposition 4 never uses invariance of the adaptive
  step; it freezes the surrogate at time `m` for the *auxiliary* chain, uses invariance and
  uniform ergodicity only for that frozen kernel, and pays for the freeze with `Σ D_j` over a
  window of fixed length `N`. That is exactly why A5 is stated as a *per-step* change going to
  zero: a bounded total change would not do, and neither would "change only rarely" unless
  the rare changes are themselves small or their probability vanishes (§6.4).
* **Multiple chains.** With `m` samplers, each chain `i` satisfies (★) with respect to the
  *global* filtration `𝓕_n` (its own `Γ_n^{(i)}` is `𝓕_n`-measurable, whatever the other
  chains contributed to it), so Lemmas 1–3 and Proposition 4 apply to each chain separately.
  No product-chain argument is needed — and none is available in the asynchronous case,
  where the `m` chains do not share a step index.
* **Asynchronous collector.** Which version a sampler installs at step `n` depends on message
  arrival times. What the proof needs is only that the install decision for step `n + 1` is
  made from information available at the end of step `n` (this is invariant I1: the
  evaluator is polled once, before the sub-chain starts, `algorithms.py:746`) and that the
  fresh uniforms and proposal noise used for step `n + 1` are independent of it. Then (★)
  holds and asynchrony costs nothing in the proof. What *would* break (★) is an evaluator
  that changes while a step is being computed; I3 excludes it for the NN and MPI paths (the
  `run_local` polynomial aliasing noted in the previous review is the one exception in the
  code base and does not concern the NN).
* **What the theorem does not give.** No rate, no CLT, no strong law. Those need the
  stochastic-approximation machinery of Andrieu & Moulines (2006) or Saksman & Vihola (2010)
  and, for a rate, a quantitative version of A5 (`Σ_n Δ_n^2 < ∞` or similar). Not attempted.

---

## 5. The analytic assumptions against the code

### 5.1 A2 (proposal bounded above and below)

RW: `q(x, y) = φ_Σ(y − x)`, so `q_max = φ_Σ(0)` and `q_min = inf_{‖v‖ ≤ diam X} φ_Σ(v) > 0`.
pCN: `q(x, y) = N(y; ρx, (1 − ρ²)C)`, same argument on compact `X`. Fine.

### 5.2 A3 (exact log-likelihood bounded)

Gaussian likelihood ⇒ `ℓ ≤ const`; the lower bound needs `sup_X ‖G(x) − d‖ < ∞`, i.e. a
forward map bounded on `X` — true for a continuous map on a compact set. The solver-failure
convention `ℓ = −∞` (`algorithms.py:47, 254`) makes `α₂ = 0` at failed points, which keeps
Lemma 1 (the displayed identity holds with both sides zero, so reversibility w.r.t. `π`
restricted to the success set survives) but breaks Lemma 3 on the failure set. If the
failure set has prior mass zero nothing changes; otherwise the target is `π` restricted to the
success set and A3 should be read on that set with the minorisation restricted accordingly.
Not a practical obstacle, but the theorem as stated assumes solver failures are `π`-null.

### 5.3 A1 (compactness) — the assumption that actually bites

The toy examples use Gaussian priors on `ℝ^d`. Two honest options:

* **Truncate the prior** to a box that carries all but a negligible prior mass. This *changes
  the posterior* (by the truncated mass) and must be declared as such; it is not a
  reparametrisation. With the box at, say, 8 prior standard deviations the change is far below
  Monte Carlo error, but it is a change. Under truncation A1–A3 hold and the proof applies.
* **Keep `ℝ^d` and assume containment.** The proof then loses Lemma 3. What survives of it:
  by A4, every stage-1 target satisfies `e^{−2B̃} ≤ dπ̃_γ/dp ≤ e^{2B̃}` uniformly in `γ`
  (`dπ̃_γ/dp = e^{ℓ̃_γ}/Z_γ` with `Z_γ ∈ [e^{−B̃}, e^{B̃}]`), i.e. all surrogate targets are
  uniformly comparable to the prior, and `P_γ ≥ a₂ Q_γ`. For **pCN sub-chains** this is more
  than plausibility: Hairer, Stuart & Vollmer (2014) prove a spectral gap for pCN under a
  bounded (locally Lipschitz) log-likelihood perturbation of the Gaussian reference, which
  is A4 (plus a gradient bound), so the *stage-1* family `{M_γ}` is simultaneously
  geometrically ergodic; what is not done here is to carry that through the `K`-fold
  composition and the outer acceptance to `{P_γ}`. For **RW sub-chains** the situation is
  weaker: Jarner & Hansen (2000)'s conditions constrain the tails *and* `∇ log π̃_γ`, and a
  bounded oscillating `ℓ̃_γ` does not preserve them, so A4 alone does not transfer — a uniform
  bound on `∇ℓ̃_γ` would be needed as well. Saksman & Vihola (2010) and Bai, Roberts &
  Rosenthal (2011) are the references for containment of adaptive families on unbounded
  domains. I have not carried out the simultaneous-drift verification for the outer DAMH
  kernel and do not claim it. On `ℝ^d`, containment is an assumption, and A4 (with a gradient
  bound, for RW) is what makes it a reasonable one.

### 5.4 Hamiltonian stage-1 proposals

Lemma 2 needs, in addition, `sup_x ‖∇ℓ̃_{γ} − ∇ℓ̃_{γ'}‖ → 0` (the proposal itself depends on
the surrogate through its gradient), and Lemma 3's density-based minorisation does not apply
as written to a deterministic-flow proposal. The structure of the argument is unchanged; the
lemmas would have to be redone for the leapfrog kernel. Not done here. Everything in §6
therefore refers to RW/pCN sub-chains; for Hamiltonian sub-chains the same NN measures are
needed for the gradient as well.

---

## 6. The NN surrogate: what fails, and how to satisfy A4 and A5

### 6.1 What the code does today (`surrDAMH/surrogates/torch_perceptron_minibatches.py`)

* The network is **warm-started**: `self.model` persists across retrains, `train()` runs
  `iterations_batch` (default 100) optimiser steps at `learning_rate` (default `1e-3`) over the
  whole snapshot set (`:791-821`, `:610-639`), and `add_data()` may run another
  `iterations_batch` steps on new + replay rows (`:767-776`). Optimiser: AdamW with
  `weight_decay=1e-4` by default (`:421-425`).
* `gradient_clip_norm` exists but defaults to `None` (`:295, 559-561`).
* Outputs are an affine de-normalisation of a linear last layer (`:783-785`); default
  activation `silu`; no output clipping, no weight projection.
* The collector retrains as soon as `min_snapshots_to_update = 1` new snapshot has arrived
  (`process_COLLECTOR.py:319`, `configuration.py:271`), for the entire run.
* The published evaluator deep-clones the weights (`:93, 109-118`), so I3 holds.

### 6.2 A4 (uniform bound on `ℓ̃`)

For a Gaussian likelihood `ℓ̃_γ(x) = −½ ‖Σ^{−1/2}(G̃_γ(x) − d)‖² + const`, so A4 is equivalent
to a uniform bound on the noise-normalised surrogate residual over `X × Γ`. Today there is
none: an MLP with unbounded weights is unbounded on `X`, and nothing bounds the weights
(weight decay penalises, it does not constrain). Two ways to get A4, both of which keep
Lemma 1 intact because they are fixed measurable transforms of `γ`:

* **Clip in observation space**: `G̃ ← clip(G̃, d − c·σ, d + c·σ)` componentwise (or clip the
  normalised residual norm at `R`). Then `|ℓ̃| ≤ ½ R² + const` for every weight vector, so
  `Γ` can be all of weight space. This is the cheapest route and has a useful side effect:
  it stops stage 1 from acting on surrogate predictions that are wildly outside the data
  range, i.e. exactly where the surrogate is untrustworthy. Efficiency effect only.
* **Bound the weights**: project onto `‖θ‖_∞ ≤ W` after every retrain (`Θ` compact). With
  `X` compact and a continuous activation, `G̃_θ` is then bounded on `X × Θ`, and moreover
  Lipschitz in `θ` uniformly in `x` (needed in 6.3). This is the route that makes the weight-
  space schemes below rigorous.

Either bound propagates to `ℓ̃` via `|ℓ̃ − ℓ̃'| ≤ R ‖Σ^{−1/2}(G̃ − G̃')‖` where `R` bounds the
normalised residual — so from now on it is enough to control `sup_x ‖G̃_{n+1}(x) − G̃_n(x)‖`.

### 6.3 A5 (diminishing adaptation) — why it fails now

Let `θ_n` be the installed weights at outer step `n`. With `Θ` compact (6.2) the network is
`L_Θ`-Lipschitz in `θ` uniformly on `X`, so

    Δ_n ≤ R L_Θ ‖θ_{n+1} − θ_n‖.                                   (†)

A5 therefore follows from `‖θ_{n+1} − θ_n‖ → 0` in probability. Today it does not hold:
Adam-type optimisers move every coordinate by `O(learning_rate)` per step irrespective of the
gradient size (that is what the second-moment normalisation does), so one retrain moves `θ`
by up to `O(iterations_batch · learning_rate) = O(0.1)` per coordinate, at every retrain, for
the whole run, and the growing data set does not shrink it (the minibatch noise floor keeps
the iterate bouncing at that scale). Retraining on every new snapshot makes this happen
essentially at every outer step. So `Δ_n` is bounded but **does not go to zero** — the
theorem's hypothesis fails, and no conclusion about convergence to `π` is available for the
current default configuration. (This is the same defect note 16 diagnosed for windowed
proposal adaptation: bounded per-update change that does not decay.)

### 6.4 Three ways to satisfy A5 with the NN, with the bound each one delivers

**(N1) Publish Polyak-averaged weights** — recommended.
Let the collector train as it does now (fixed `lr`, fixed `iterations_batch`; nothing about
the learning needs to change), producing raw weights `θ̂_k` after the `k`-th retrain. Publish
instead the running average

    θ̄_k = θ̄_{k−1} + η_k (θ̂_k − θ̄_{k−1}),   η_k = k^{−a}, a ∈ (½, 1]   (η_k = 1 for a warm-up phase).

With `Θ` compact, `‖θ̄_k − θ̄_{k−1}‖ ≤ η_k diam Θ`. Between two consecutive *installs* at
outer steps `n` and `n + 1` the collector may have retrained `r_n ≥ 1` times (with
`min_snapshots_to_update = 1` and up to `K + 1` snapshots per outer step, `r_n > 1` is the
normal case), so by (†) and the triangle inequality

    Δ_n ≤ R L_Θ diam Θ · Σ_{k ∈ (k(n), k(n+1)]} η_k ≤ R L_Θ diam Θ · r_n η_{k(n)}

**deterministically**, where `k(n) → ∞` is the retrain count at step `n`. This vanishes
provided `r_n` stays bounded, which should be enforced (cap the number of retrains folded
into one published version, or publish after every retrain — then `r_n ≤ K + 1`
automatically). A5 holds; nothing else in the proof changes. Cost: one extra weight vector in
the collector and a projection onto `Θ`. Averaging weights of a warm-started, single-trajectory network is the SWA/Polyak–
Ruppert setting and is meaningful; it would *not* be meaningful across re-initialisations
(permutation symmetry), so the warm start in the code is load-bearing. The averaged network
lags the raw one early on — hence the warm-up with `η_k = 1` — and is typically at least as
accurate later. Publishing continues at every retrain, so the sampler still sees frequent
updates; they just shrink like `k^{−a}`.

**(N2) Decaying training budget** (Robbins–Monro style).
Use `learning_rate_k · iterations_k → 0` with `Σ_k learning_rate_k · iterations_k = ∞`, e.g.
`lr_k = lr_0 k^{−a}`, `a ∈ (½, 1]`, fixed `iterations_batch`, plus `gradient_clip_norm` set
(for SGD the displacement bound needs it; for Adam the `O(lr)` per-step bound already
holds). Then `‖θ̂_{k} − θ̂_{k−1}‖ ≤ C lr_k iterations_k → 0` deterministically and (†) gives
A5, with the same "retrains per install" factor `r_n` as in (N1). The divergent-sum condition
is not needed for A5; it keeps the network able to learn indefinitely. This is the
Andrieu–Moulines stochastic-approximation regime and the natural one if a CLT is ever wanted.
Cost: learning slows down, and the decay exponent is one more tuning knob. Note that the
Adam optimiser is rebuilt with `lr = learning_rate_init` on rollback (`:810`), so a schedule
has to live in the updater, not in the optimiser object.

**(N3) Randomised installation** (Conrad, Marzouk, Pillai & Smith 2016).
Leave collector training alone; in `_refresh_surrogate_evaluator_if_needed`
(`algorithms.py:498-530`) install a pending evaluator at outer step `n` only with probability
`β_n → 0`, `Σ β_n = ∞` (e.g. `β_n = min(1, c n^{−a})`, `a ∈ (0, 1]`), with the coin drawn
*before* the step's other randomness (A6). Then `P(D_n ≠ 0) ≤ β_n → 0`, so `D_n → 0` in
probability; Proposition 4 needs only `E[1 ∧ D_j] ≤ P(D_j ≠ 0) ≤ β_j → 0`, using `D_j ≤ 1`.
A4 is still required for Lemma 3. This is the smallest code change
and does not touch the learning at all; the price is that the sampler runs on a stale
surrogate between installs (efficiency, not validity), and that the per-install change is
`O(1)`, so the "window" bound in Proposition 4 is only small in probability, not
deterministically — fine for the theorem, but (N1) is the cleaner statement.

**What does *not* work: a deterministic sparse schedule.** Refitting/installing at times
`n_k` with `n_{k+1} − n_k → ∞` but `O(1)` change per install violates A5 as stated
(`P(D_{n_k} > ε) = 1` along the subsequence), and Proposition 4's window bound is `O(1)` for
`n` just after each `n_k`. What one still gets under Lemma 3, conditioning on `𝓕_{n_k}` and
using that the kernel is fixed on `(n_k, n_{k+1}]`, is
`sup_A |P(X_n ∈ A) − π(A)| ≤ (1 − ε)^{n − n_{k(n)}}` — convergence along every sequence of
times whose distance from the last install tends to infinity, and hence the WLLN for
ergodic averages. What is lost is control of the marginal in the `O(1)` steps after each
install, so it is not validity in the sense of the Theorem, and with a realistic (tiny) `ε`
the "distance from the last install" needed is not small. Do not rely on "we only retrain
every 1000 steps" as a validity argument; it is an efficiency device.

**Finite adaptation** — stop installing after a fixed number of exact evaluations — is the
degenerate case `Δ_n = 0` eventually and is covered trivially; it is what note 16 recommends
for the final stage and is not discussed further here as per the question.

### 6.5 Summary for the NN surrogate

| Assumption | Minimal change that satisfies it | Bound it delivers |
|---|---|---|
| A4 | clip the surrogate residual at `R` (observation space), or project weights onto `‖θ‖_∞ ≤ W` | `|ℓ̃| ≤ ½R² + c` for all versions |
| A5 | (N1) publish Polyak-averaged weights `η_k = k^{−a}` (needs weight bound) | `Δ_n ≤ R L_Θ diam Θ · k(n)^{−a}` deterministic |
| A5 | (N2) `lr_k · iterations_k → 0`, `Σ = ∞`, gradient clipping | `Δ_n ≤ C lr_k iterations_k` deterministic |
| A5 | (N3) install with probability `β_n → 0`, `Σ β_n = ∞` | `P(D_n ≠ 0) ≤ β_n` |
| A1 | truncate the prior to a box (declared change of target), or assume containment on `ℝ^d` | Lemma 3 / assumption |

With A4 by clipping **and weight projection onto `Θ`** (the projection is what (N1)'s bound
needs; clipping alone does not give it), A5 by (N1), plus a truncated prior, every
hypothesis of the theorem is met and DAMH-SMU with the NN surrogate is a provably valid MCMC algorithm. Without A5 it
is an algorithm each of whose transitions is exactly `π`-reversible (Lemma 1 holds today)
and whose convergence to `π` is *not* established — that is the precise status of the
present default configuration.

---

## 7. Not verified / limits of this note

* Nothing was run. Line numbers are from reading the working tree on 2026-09-22.
* The proof was independently refereed (Opus, 2026-09-22); the referee's two necessary
  repairs (internal variables in Proposition 4's construction; standing measurability A0)
  and its tightenings (Lemma 1 labelling, Lemma 2 sharp constant, per-chain instead of
  product argument, `r_n` factor in (N1)/(N2), the `(1 − ε)^{n − n_k}` remark, the
  Jarner–Hansen caveat) are incorporated above.
* Bibliographic page ranges in §8 were checked from memory only; the Bai–Roberts–Rosenthal
  page range in particular is unverified.
* The WLLN (Theorem part 2) is cited, not proved; RR07 theorem numbering not re-checked.
* Containment on `ℝ^d` (§5.3) is not proved; the references given are for related but not
  identical kernel families.
* Hamiltonian sub-chains (§5.4) are not covered by Lemmas 2–3 as written.
* The Lipschitz-in-weights constant `L_Θ` in (†) exists for any finite MLP with Lipschitz
  activation on compact `X × Θ`, but no attempt is made to bound it; only its finiteness is
  used.
* The claim that Adam moves each coordinate by `O(lr)` per step is the standard qualitative
  property of the update rule (`|Δθ_i| ≲ lr` when `|m̂_i| ≲ √v̂_i`), not an exact bound in
  every regime; it is used only to argue that the *current* scheme does not satisfy A5, which
  is also evident from the fixed, non-decaying budget.

## 8. References

* Christen, J. A. & Fox, C. (2005). MCMC using an approximation. *J. Comput. Graph. Statist.* 14, 795–810.
* Roberts, G. O. & Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive MCMC algorithms. *J. Appl. Prob.* 44, 458–475.
* Bai, Y., Roberts, G. O. & Rosenthal, J. S. (2011). On the containment condition for adaptive Markov chain Monte Carlo algorithms. *Adv. Appl. Stat.* 21, 1–54.
* Andrieu, C. & Moulines, É. (2006). On the ergodicity properties of some adaptive MCMC algorithms. *Ann. Appl. Probab.* 16, 1462–1505.
* Saksman, E. & Vihola, M. (2010). On the ergodicity of the adaptive Metropolis algorithm on unbounded domains. *Ann. Appl. Probab.* 20, 2178–2203.
* Conrad, P. R., Marzouk, Y. M., Pillai, N. S. & Smith, A. (2016). Accelerating asymptotically exact MCMC for computationally intensive models via local approximations. *JASA* 111, 1591–1607.
* Sherlock, C., Golightly, A. & Henderson, D. A. (2017). Adaptive, delayed-acceptance MCMC for targets with expensive likelihoods. *JCGS* 26, 434–444.
* Cui, T., Fox, C. & O'Sullivan, M. J. (2011). Bayesian calibration of a large-scale geothermal reservoir model by a new adaptive delayed acceptance Metropolis Hastings algorithm. *Water Resour. Res.* 47.
* Hairer, M., Stuart, A. M. & Vollmer, S. J. (2014). Spectral gaps for a Metropolis–Hastings algorithm in infinite dimensions. *Ann. Appl. Probab.* 24, 2455–2490.
* Jarner, S. F. & Hansen, E. (2000). Geometric ergodicity of Metropolis algorithms. *Stoch. Proc. Appl.* 85, 341–361.
* Related notes: `16_adaptivity_options_research_2026-09-20.md` (same machinery for proposal adaptation; §2 item 5 on non-decaying windowed updates), `06_findings_consolidated.md` findings 1.1–1.2 (the fixes that make invariant I1 true), `docs/concepts.md:59-103` (per-transition exactness).
