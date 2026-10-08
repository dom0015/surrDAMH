# Diminishing adaptation for the on-the-fly NN surrogate: how to get it, and whether ergodicity follows (2026-09-28)

Scope: DAMH-SMU with the torch MLP surrogate retrained by the collector from the chain's own
exact evaluations. Question asked by the author: *without changing the library*, how can
diminishing adaptation be ensured for such a surrogate, and is the sampler then ergodic? This
note builds on the proof note `18_damh_smu_validity_proof_2026-09-22.md` (cited as "note 18p";
its assumptions A0–A6, invariants I1–I3, Lemmas 1–3, Proposition 4 and the mechanisms N1–N3 are
used without restating them), a literature check with web access (§3, raw file in
`out_diminishing_adaptation_2026-09-28/literature_review.md`), and a standalone side experiment
(§4, code and outputs in `out_diminishing_adaptation_2026-09-28/`; nothing in `surrDAMH/` was
changed). Sections marked **[analysis]** are reasoning, **[verified]** were checked online or
computed, **[not verified]** were not.

---

## 0. Result first

1. **Diminishing adaptation (DA) is a property of the *installed* surrogate sequence seen by one
   chain, not of the collector's training.** It can therefore be enforced entirely on the sampler
   side, per stage, without touching the collector: the sampler owns the install decision
   (`algorithms.py:505-535`, one call per outer step) and, for the NN, the installed weights. Two
   mechanisms need nothing from the collector — (N3) install with a decaying probability, and
   (N1′) a sampler-side Polyak average of the received weights, indexed by the sampler's own
   install count. Both give the "aggressive preliminary stage / DA production stage" split the
   author asked about, because the *same* published stream can be consumed either way.
   Neither requires the collector to know which stage any chain is in (it does not).
2. **A finite non-DA prefix costs nothing.** The Roberts–Rosenthal theorem is about the tail:
   `D_n → 0` in probability. An aggressive preliminary stage followed by a DA production stage
   is one adaptive chain whose adaptation eventually diminishes; the theorem applies to the whole
   run. So "version 1 for burn-in, version 2 for production" is exactly the regime the theory
   permits, provided the production stage also pins the proposal (`adaptive=False`), because
   proposal adaptation is a second adaptation that must diminish too (note 16).
3. **DA alone is not ergodicity.** It needs *containment* as well (RR07; their Example shows DA
   without containment can fail). On a compact state space with the surrogate log-likelihood
   bounded (A1 + A4) containment comes for free as simultaneous uniform ergodicity (note 18p
   Lemma 3). On `ℝ^d` with a Gaussian prior containment is an *assumption* for the outer DAMH
   kernel; nothing in this note or the literature found proves it for DAMH. The honest options
   remain: truncate the prior to a large box (declared change of target, below Monte Carlo
   error at 8 prior sd), or state containment as a hypothesis.
4. **A4 (uniform bound) and the Lipschitz-in-weights step of A5 can be obtained without
   compactness of the prior**, by clipping the *inputs* of the network to a box before the first
   layer and clipping the *outputs* to `d ± cσ`. The surrogate is then constant outside the box
   and bounded everywhere; accuracy there is irrelevant for validity (Lemma 1 needs none). This
   leaves A1 needed only for Lemma 3, i.e. for containment.
5. **What ergodicity then means**: convergence of the marginal law of `X_n` to `π` in total
   variation and a weak law of large numbers for bounded functions. No rate, no CLT, no
   finite-`n` bias bound. DA does not remove the bias of the early part of the chain; it makes
   the chain asymptotically correct. A frozen final stage gives strictly more (a homogeneous,
   uniformly ergodic chain with a CLT) and is the right default whenever the surrogate has
   converged; DA training buys something only when the surrogate is still improving in production.
6. **Literature (§3)**: no published ergodicity proof exists for DA-MCMC with a neural network
   retrained during production; the closest analogue is Sherlock, Golightly & Henderson 2017
   (k-NN surrogate, randomised updates, exact-kernel mixture) — note 18p attributed the
   randomised-installation idea to Conrad et al. 2016, which is the wrong paper. Laitinen &
   Vihola (2026) upgrade the conclusion: under the note's own uniform ergodicity, N1′ and N3
   with any decay exponent give a **strong** law of large numbers, not only a weak one; a CLT
   would need the installed surrogate's asymptotic variance to converge, which nothing
   guarantees for a network.
7. **Experiment (§4)**: in a standalone 4-d toy, today's scheme has `Δ_n` flat at ≈ 0.1
   log-likelihood units for 20 000 steps (A5 fails, as claimed); Polyak averaging gives a clean
   `n^{−0.8}` decay *and* the most accurate surrogate and the best efficiency of all learning
   schemes; learning-rate decay gives the decay but stalls the surrogate; randomised installation
   gives it only in probability; output clipping costs nothing. No scheme, today's included,
   shows a posterior bias above the Monte Carlo noise of a 20 000-step chain.

---

## 1. [analysis] What "diminishing adaptation" is for this algorithm, precisely

Note 18p Lemma 2: `sup_x ‖P_γ(x,·) − P_{γ'}(x,·)‖ ≤ (8K+4) · sup_x |ℓ̃_γ(x) − ℓ̃_{γ'}(x)|`.
So DA for DAMH is a statement about the *function* `ℓ̃` installed at consecutive outer steps of
one chain:

    Δ_n = sup_{x∈X} |ℓ̃_{Γ_{n+1}}(x) − ℓ̃_{Γ_n}(x)| → 0 in probability.        (A5)

Three consequences that matter for design:

* **The clock is the chain's outer step `n`, not the collector's retrain counter `k`.** With
  `min_snapshots_to_update = 0` (the default since 2026-09-23) the collector retrains whenever
  it is idle, so the number of retrains `r_n` between two consecutive installs on one chain is
  set by wall-clock ratios (solver time vs. training time), not by the algorithm. Any bound of
  the form `Δ_n ≤ C · r_n · η_{k(n)}` (note 18p N1, N2) therefore needs `r_n` bounded in
  probability. That holds when every chain advances at a bounded rate relative to the collector,
  which is the normal MPI situation, but it is an *extra* assumption about timing, and a stalled
  chain (slow solver) breaks the deterministic version of the bound. A sampler-side mechanism
  whose clock is the chain's own install count removes the issue entirely (§2.1).
* **`sup_x` is over the whole state space.** Two different MLPs with SiLU activations grow
  linearly in different directions as `‖x‖ → ∞`, so on `ℝ^d` the sup of their difference is not
  small even when they agree on the data. Either `X` is compact (A1), or the network's inputs are
  clipped to a box so that both functions are constant outside it (§2.3). With input clipping the
  sup reduces to a sup over the box.
* **Multiple chains do not interact in A5.** Each chain has its own installed sequence; the
  proof is per chain with the global filtration (note 18p, Remarks). What the chains share is
  the training data; that affects which `Γ` gets installed, not the form of the condition.

## 2. [analysis] Mechanisms, re-examined for a per-stage design

The collector trains one network for all chains and does not know their stages. A production
mode must therefore be *sampler-side*. Of note 18p's three mechanisms, N3 already is; N1 can be
moved to the sampler; N2 cannot.

### 2.1 N1′: sampler-side Polyak average of received weights (new here)

The sampler already receives every published version as a frozen deep copy of the weights
(`PyTorchNNEvaluator.clone_model_to_cpu`, `torch_perceptron_minibatches.py:109-118`). Let
`θ̂_j` be the weights of the `j`-th version this chain *installs* (its own counter). Keep

    θ̄_j = θ̄_{j−1} + η_j (θ̂_j − θ̄_{j−1}),   η_j = j^{−a}, a ∈ (½, 1],   η_j = 1 for j ≤ j₀,

and evaluate the surrogate with `θ̄_j`. With weights projected onto the box `‖θ‖_∞ ≤ W` (the
collector need not do it: project the received `θ̂_j` on the sampler; the box is convex, so the
average stays inside), and inputs clipped to a box `𝔹` (§2.3), the map `θ ↦ G̃_θ(x)` is
`L`-Lipschitz uniformly in `x` (finite MLP, Lipschitz activation, compact domain), and

    Δ_n ≤ R · L · ‖θ̄_{j(n+1)} − θ̄_{j(n)}‖ ≤ R L diam(Θ) · η_{j(n)} · 1{install at n}

**deterministically**, with at most one install per outer step by construction (one poll per
sub-chain, `algorithms.py:753`). No `r_n` factor, no timing assumption. `R` is the clipping
radius of the normalised residual (§2.3). Cost: one extra weight vector per sampler, one
`state_dict` blend per install (microseconds at this network size). The collector, the MPI
messages and the preliminary stages are untouched; a preliminary stage simply uses `θ̂_j`.

Two caveats. (i) Weight averaging is meaningful only along a single warm-started trajectory —
which is what the collector does (`self.model` persists; note 18p §6.1) — and only if the
collector never re-initialises the network (a rollback that rebuilds the optimiser keeps the
weights, `:810`, so that is fine; a fresh `NeuralNetworkUpdater` would not be). (ii) The averaged
network lags the raw one by `O(1/η_j)` installs; with `a = 0.7` and `j = 1000` that is a lag of
~125 installs, which is why the warm-up `η = 1` phase should cover the preliminary stage and
the production stage should start its counter at `j₀` with the current raw weights.

### 2.2 N3: randomised installation

Install the pending version at outer step `n` of the production stage with probability
`β_n = min(1, (n/n₀)^{−a})`, `a ∈ (0,1]`, coin drawn from a dedicated per-(chain, stage)
stream before the step's proposal noise and acceptance uniforms — the seed layout
(`modules/seeds.py`, stride 10, offsets 1–3 used) leaves room for such a stream without moving
any existing one, which is what the project rule on seeds requires. Then
`P(Δ_n ≠ 0) ≤ β_n → 0`, so A5 holds in probability; `Σ β_n = ∞` keeps the number of installs up
to `n` growing like `n^{1−a}`, so the chain keeps benefiting from training. The pending version
is the *latest* one when the coin succeeds (versions skipped in between are dropped, as today),
so the per-install change is `O(1)` and the bound is only in probability — enough for RR07 and
Proposition 4. This is the smallest change, is exactly what the current install path does with
one extra `if`, and needs A4 (clipping) but neither weight projection nor a Lipschitz constant.

### 2.3 A4 and the Lipschitz step without a compact prior

* **Output clipping** `G̃ ← clip(G̃, d − cσ, d + cσ)` componentwise (note 18p §6.2). Gives
  `|ℓ̃| ≤ ½ m c²` for `m` observations, uniformly over all weights. Efficiency effect: none
  where the surrogate is accurate (the true likelihood at a 5σ residual is `e^{−12.5}` per
  component). Side effect for the gradient (Hamiltonian sub-chains): zero gradient in the
  clipped region — acceptable, note 18p §5.4 does not cover Hamiltonian sub-chains anyway.
* **Input clipping** `x ← clip(x, −b, b)` in standardised coordinates before the first layer,
  `b` ≈ 8 (the inputs are standardised since 2026-09-22, so the box is dimension-free). Then
  `G̃_θ` is evaluated on a compact set for every `x ∈ ℝ^d`, `θ ↦ G̃_θ(x)` is Lipschitz uniformly
  in `x`, and `sup_x |ℓ̃_θ − ℓ̃_{θ'}|` is a sup over the box. This is what makes N1′'s
  deterministic bound hold on `ℝ^d`. It does *not* give containment (Lemma 3 still needs A1);
  it only removes A1 from the *adaptation* side of the theorem.
* **Weight projection** onto `‖θ‖_∞ ≤ W` is needed by N1′ (for `diam Θ` and `L`), not by N3.
  With AdamW at `weight_decay=1e-4` the weights of a 2×32 SiLU network stay `O(1)`; `W = 10`
  should never bind (§4 reports whether it did in the toy).

### 2.4 N2 (decaying learning rate) is not a per-stage mechanism

It lives in the collector and throttles the one network every chain uses. It also spends the
schedule on idle retrains (`k` counts collector loops, not data), so with
`min_snapshots_to_update = 0` the learning rate would decay at wall-clock speed. Keep it out of
the per-stage design; it remains the natural route if a CLT (Andrieu–Moulines regime) is ever
wanted, and would then have to be indexed by the number of snapshots, not by retrains.

### 2.5 What does *not* work, restated

A deterministic sparse install schedule ("install every 1000 steps") violates A5 along the
install times (note 18p §6.4). A *gated* install ("install only if the change on a test set is
below `ε_n → 0`") looks like DA but the sup over a finite test set is not the sup over `X`, and
if the gate never opens the chain silently freezes; useful as a diagnostic, not as a proof
route. Freezing (`surrogate_model_updates=False`) is the degenerate case `Δ_n = 0` and is
always valid.

## 3. Literature check

Source: `out_diminishing_adaptation_2026-09-28/literature_review.md` (Opus with web access,
325 lines; every claim there is tagged full-text / abstract-only / memory). I re-checked the two
load-bearing statements below against the arXiv full texts myself (marked ✔). Everything else is
as tagged in that file.

### 3.1 Corrections to note 18p

| Note 18p says | Literature says | Effect |
|---|---|---|
| N3 (randomised installation) is Conrad, Marzouk, Pillai & Smith 2016 | Conrad et al.'s `β_t` is the probability of *refining* (adding an exact evaluation), their chain is *approximate* (not DA), and their Theorem B.1 / Lemma B.3 need the approximation to **converge to the truth**. The randomised-adaptation mechanism is **Sherlock, Golightly & Henderson 2017** (Algorithm 1 step 4: new points enter the k-NN tree with probability `p_i → 0`; Theorem 1 proved as Theorem 6 via RR07) and, as a prototype, RR07 Corollary 16. | Citation fixed; the mechanism and its proof are unchanged. Validity needs only `β_n → 0`; `Σ β_n = ∞` is an efficiency condition, not a validity one. |
| Theorem part 2 cites RR07 "WLLN" without number | RR07 (final preprint): Theorem 5 = SUE + DA ⇒ ergodic (the note's Proposition 4); Theorem 13 = containment + DA; Theorem 23 = WLLN for bounded `g`. Example 24: DA + SUE do **not** give a strong law in general. | Numbers can be filled in. |
| "No rate, no CLT, no strong law" (§4 Remarks) | **Laitinen & Vihola (EJP 2026, arXiv 2408.14903)** ✔: under SUE `d_TV(P_s^k(x,·), π) ≤ Cρ^k` (their A3, which note 18p Lemma 3 delivers with `C = 1`, `ρ = 1 − ε`), a filtration that allows internal variables (their A1, i.e. (★)), and `D_k` as in A5: Lemma 11 — if `Σ_k E[D_k]/k^p < ∞` the adaptation is "strongly p-waning"; Theorem 9 — p = 1 gives the **SLLN**, p = ½ plus their (A4) (the asymptotic variances of the installed kernels average to a limit `σ²_ϕ`) gives a **CLT** with that variance. | A strong law is available for N1′ and N3 (§5). A CLT is not, because (A4) needs the installed kernels' asymptotic variances to settle, which a trained network does not guarantee. |
| Deterministic sparse schedules are only "an efficiency device" (§6.4) | **Air MCMC (Chimisov, Łatuszyński & Roberts 2018, arXiv 1801.09309)** ✔: adaptation at times `N_j = Σ_{k≤j} n_k` with *lags* `c₁k^β ≤ n_k ≤ c₂k^β`, `β > 0`, under simultaneous geometric drift (their Assumptions 1–2): WLLN for any `β > 0` with an MSE rate `N^{−min(1, 2β/(1+β))}`, **SLLN for `β > ½`**, CLT for `β > 1` plus convergence of the adapted parameter. Their §3.4 Example 1: marginals need not converge without DA; Theorem 5 restores it with randomised lags. LV26 Theorem 20 is the SUE analogue (SLLN if `Σ_j E[1/τ_j] < ∞`). | The note is right that a sparse schedule does not give convergence of marginals, but understates the LLN: with lags growing like `k^β`, `β > ½`, ergodic averages converge a.s. A *fixed* lag ("every 1000 steps") gives neither, as the note said. |
| A4 (uniform surrogate bound) is needed for SUE | Sherlock et al. 2017 mix an **exact-likelihood MH kernel with fixed probability `β`** into every step; the minorisation then comes from the exact kernel alone (proof of their Theorem 6), so SUE holds with **no bound on the surrogate**. | An alternative to output clipping (§2.3). In DAMH the cost is small: an exact-MH step and a DAMH step each call the solver at most once, so a fraction `β` of steps merely lose the surrogate screening. Compactness (A1) is still needed for the exact kernel's minorisation. |

### 3.2 What the general theory delivers (verified numbers)

| Result | Needs | Reference |
|---|---|---|
| Marginals → π in TV | SUE + DA | RR07 Thm 5; containment + DA: RR07 Thm 13; Atchadé & Fort 2010 Thm 2.1 |
| DA alone is insufficient | — | Bai, Roberts & Rosenthal 2011 Example 1 / Prop. 1 (two-state chain, `θ_n = (n+2)^{−r}`), Example 2 / Prop. 3; RR07 Example 4 |
| WLLN, bounded `f` | SUE + DA | RR07 Thm 23; LV26 Thm 9(i) with Lemma 10 |
| SLLN | SUE + `Σ E[D_k]/k < ∞` | LV26 Thm 9(ii) + Lemma 11; also Andrieu & Moulines 2006 Thm 8, Atchadé–Fort Thm 2.5, Saksman–Vihola Thm 10 in their own settings |
| CLT | SUE + `Σ E[D_k]/√k < ∞` + convergence of asymptotic variances (LV26 A4) | LV26 Thm 9(iii); AM06 Thm 9 and Air Thm 1(iv) need the adapted parameter to converge a.s. |
| Containment on ℝ^d | simultaneous geometric drift | RR07 Thms 18–19; BRR 2011 Thm 3, Thm 6 (adaptive Metropolis with a **fixed** target, light/exponential/hyperbolic tails) |
| DA is π-invariant for any approximation | positivity of the approximation where the proposal is positive | Christen & Fox 2005 Thm 1; efficiency only via Peskun ordering (§3 there) |

### 3.3 Surrogate-based adaptive MCMC with proofs

* **Sherlock, Golightly & Henderson 2017** is the closest published analogue of DAMH-SMU: DA with a
  k-NN surrogate built from *all* exact evaluations, rejected proposals included (as here,
  `algorithms.py:348-351`), updated with probability `p_i → 0`, plus the defensive exact kernel.
  Ergodicity via RR07. Their §5.1 states the feedback-loop caveat plainly: "between adaptations
  our algorithm targets the true posterior distribution, but it is perturbed every time an
  adaptation occurs".
* **Conrad et al. 2016 / 2018** prove TV convergence of an *approximate* chain (no second stage)
  whose local polynomial surrogate is refined at random; the argument needs the surrogate to
  converge to the truth and gives no LLN or CLT (their Remark 3.5). Their Example B.13 shows the
  feedback loop can be permanent if refinement stops (`Σ β_t < ∞`): the chain never visits where
  the surrogate is wrong, so it is never corrected. Not applicable to DAMH's validity, but the
  right reference for why on-the-fly training should not stop while the chain may still discover
  new regions.
* **Cui, Fox & O'Sullivan 2011** (adaptive DA with an error model) assert DA + RR07; details in an
  unseen tech report. **Lykkegaard et al. 2023** (MLDA) justify their adaptive error model by
  assertion.
* **Neural-network surrogates trained during the production run: no ergodicity, LLN or CLT
  proof was found** (search terms in the review's §5). Every NN/DA paper found freezes the
  surrogate after a warm-up (Deveney, Mueller & Shardlow 2023; Bérešová et al. 2026,
  arXiv 2606.14743) or trains it offline. A proof for DAMH-SMU with an NN would therefore be new.

### 3.4 Optimisation facts used by the mechanisms

* Weight averaging along a single warm-started trajectory is standard practice (SWA, Izmailov et
  al. 2018 §3.2). The bound `‖θ̄_k − θ̄_{k−1}‖ ≤ η_k diam Θ` is exact provided the *raw* iterate is
  projected onto the convex box `Θ` (the average then stays inside).
* Adam's per-coordinate step is bounded by `α(1−β₁)/√(1−β₂)` in the worst case (≈ 3.16 α with
  PyTorch defaults) and by `α` otherwise (Kingma & Ba 2015 §2.1); "`|Δθ| ≈ α`" is informal.
  AdamW adds `lr·λ·|θ_i|`, bounded on `Θ`. So "O(lr) per step" holds with a constant.
* Decaying learning rate (N2): for nonconvex SGD with `Σ α_k = ∞`, `Σ α_k² < ∞`, Bottou, Curtis &
  Nocedal 2018 Thms 4.9–4.10 control only gradient norms, not the weights; Adam needs `α → 0`
  and `β₂ → 1` to converge at all (Reddi et al. 2018; Défossez et al. 2022). All for a *fixed*
  objective; no theory found for a growing, chain-dependent training set.

## 4. [computed] Side experiment

Standalone Python (no `surrDAMH` import), Opus-written, reviewed by me for the install/averaging
logic: `out_diminishing_adaptation_2026-09-28/scripts/{problem,reference,toy_damh_smu,run_all,
ess,analyze}.py`; raw data `runs/`, figures `figures/`, full write-up `RESULTS.md`. All numbers
below are copied from `runs/summary.csv` (mean ± sd over 4 replicates).

**Setup.** `d = 4`, prior `N(0, I)`, `G: ℝ⁴ → ℝ⁶`, `G_i(x) = a_i·x + b_i sin(c_i·x)`, noise sd
0.1; posterior unimodal, moderately non-Gaussian (skewness up to 0.7). Two independent exact
references of 6.4·10⁵ thinned samples each. DAMH-SMU with `K = 5` RW sub-steps, one installed
version per outer step, stage 2 exact; MLP 4-32-32-6 SiLU, AdamW, 200 pilot snapshots; after
*every* outer step 20 optimiser steps at lr 1e-3 (today's `min_snapshots_to_update = 0`
regime); `N = 20 000` outer steps; `Δ_n` measured as the max over 600 test points (500 from
the posterior + 100 from the prior) of `|ℓ̃_{n+1} − ℓ̃_n|` between installed versions.

| Scheme | mechanism |
|---|---|
| EX | exact model in stage 1 (Monte Carlo noise scale) |
| S0 | frozen pilot surrogate |
| S1 | today's default: install every retrain |
| S2 | N1: install Polyak average, `η_k = k^{−0.7}` after 200 warm-up, `‖θ‖_∞ ≤ 10` |
| S3 | N2: `lr_k = 10^{−3} k^{−0.7}`, grad-clip 2 |
| S4 | N3: install with probability `min(1, (n/50)^{−0.7})`, coin first |
| S5 / S6 | S1 / S2 plus output clipping at `d ± 5σ` (A4) |

| | EX | S0 | S1 default | S2 Polyak | S3 lr-decay | S4 rand. install | S5 S1+clip | S6 S2+clip | ref-vs-ref |
|---|---|---|---|---|---|---|---|---|---|
| stage-2 acceptance | 1 | 0.28 ± 0.06 | 0.987 | 0.992 | 0.926 ± 0.011 | 0.987 | 0.987 | 0.992 | |
| surrogate RMSE, final (σ units) | 0 | 1.04 | 0.0078 ± 0.0015 | **0.0021 ± 0.0003** | 0.040 ± 0.004 | 0.0084 | 0.0073 | **0.0021** | |
| `Δ_post`, last 10 % | – | 0 | 0.102 ± 0.009 | 7.2e-5 | 0.0030 | 0.0013 ± 0.0007 | 0.103 | 7.1e-5 | |
| `Δ_all`, last 10 % | – | 0 | 11.6 ± 2.4 | 0.0096 | 0.054 | 0.16 ± 0.11 | 1.30 | 0.0011 | |
| slope of `Δ_post` vs `n`, last half | – | – | −0.07 ± 0.07 | **−0.81 ± 0.06** | −0.75 ± 0.07 | noise (installs rare) | −0.04 | −0.83 | |
| slope of `Δ_all` vs `n`, last half | – | – | **+0.33 ± 0.08** | −0.39 ± 0.04 | −0.72 ± 0.05 | noise | +0.29 | −0.36 | |
| posterior mean err, max_j (/sd_ref) | 0.047 ± 0.059 | 0.115 | 0.054 ± 0.034 | 0.065 | 0.063 | 0.025 | 0.062 | 0.070 | 0.014 |
| posterior sd err, max_j (rel.) | 0.062 ± 0.032 | 0.069 | 0.029 ± 0.021 | 0.045 | 0.023 | 0.031 | 0.030 | 0.047 | 0.0066 |
| KS, max_j | 0.028 ± 0.018 | 0.062 | 0.031 | 0.031 | 0.035 | 0.020 | 0.034 | 0.032 | 0.0075 |
| ESS / exact evaluation | 0.038 ± 0.008 | 0.0087 | 0.032 ± 0.003 | 0.037 ± 0.005 | 0.029 ± 0.001 | 0.036 | 0.033 | 0.036 | |
| installs in 20 000 steps | 0 | 0 | 19 999 | 19 999 | 19 999 | 884 ± 21 | 19 999 | 19 999 | |

Findings (see `figures/delta_vs_n.png`):

1. **Today's default does not satisfy A5.** `Δ_n` over posterior points stays flat at ≈ 0.1
   log-likelihood units for the whole run (slope −0.07); over the full test set it *grows*
   (+0.33), driven by the prior-region points where the data do not pin the network. The raw
   weights keep moving by ≈ 0.04 in ℓ₂ per retrain throughout — the "Adam noise floor" of
   note 18p §6.3, observed.
2. **Polyak averaging (S2/S6) gives `Δ_n → 0` with a clean power law**: slope −0.81 on
   posterior points (design: −0.7), `Δ_post` four orders of magnitude below S1 at the end. The
   weight box never bound (max `|θ|` 2.0–2.6 against `W = 10`). **And the averaged network is
   the most accurate surrogate of the study** (RMSE 0.0021σ vs 0.0078σ for the raw weights):
   averaging removes the minibatch noise, as SWA would predict. Stage-2 acceptance and
   ESS/evaluation are the best of the learning schemes, within noise of the exact-model chain.
3. **Learning-rate decay (S3) also gives `Δ_n → 0`** (slope −0.72 on both sets) but pays for
   it: lr reaches 1e-6, the surrogate stalls at 0.040σ, stage-2 acceptance drops to 0.93, and
   ESS/evaluation is the lowest of the learning schemes.
4. **Randomised installation (S4)** installs 884 times instead of 19 999; installs get rarer
   (rate 0.08 early → 0.015 late) but each is as large a jump as an S1 step (`Δ_post` 0.1–0.16
   per install), so the *bin-averaged* `Δ_n` falls only in probability and its slope estimate is
   noise, as the theory says. Surrogate accuracy and efficiency equal S1's.
5. **Output clipping costs nothing measurable** (S5 vs S1, S6 vs S2 identical in acceptance,
   ESS, posterior errors); it only removes the growth of `Δ_all` at the prior points.
6. **No posterior bias is detectable in any scheme, S1 included.** All errors sit inside the
   spread of the exact-model chain; the reference-vs-reference values are 3–10× below the Monte
   Carlo error of a 20 000-step chain, so the setup can only exclude biases above ≈ 0.05 sd.
   S1's failure is of the theorem's hypothesis, not an observed bias — one cheap 4-d problem
   with a surrogate that is already accurate after the pilot phase.

Confounders stated by the runner: wall times are unreliable (another MPI job occupied ~24 of
32 cores during the runs; only S3's +20 % for gradient clipping is trustworthy; training is
≈ 97 % of wall time in every learning scheme); comparisons are paired by seed, not
independent; problem constants were chosen by a small screen for unimodality.

## 5. [analysis] Is the sampler then ergodic? Assembled answer

Take a production stage with a pinned proposal (`adaptive=False`), `surrogate_model_updates=True`,
the NN surrogate with input clipping to a box and output clipping to `d ± cσ` (§2.3), and one of
N1′ or N3 on the sampler side. Then, chain by chain:

**(a) Diminishing adaptation holds.**
* N1′: `D_n ≤ (8K+4) R L diam(Θ) · j(n)^{−a} · 1{install at n}` deterministically (Lemma 2 of
  note 18p plus §2.1). Since at most one install happens per outer step, `j(n) ≤ n`, and
  `D_n = 0` at non-install steps, `Σ_n D_n / n^p ≤ C Σ_j j^{−a} / j^p`, which is finite for
  `p = 1` and any `a > 0`, and for `p = ½` when `a > ½`.
* N3: `E[D_n] ≤ P(install at n) ≤ β_n = (n/n₀)^{−a}`, so `Σ E[D_n]/n < ∞` for any `a > 0` and
  `Σ E[D_n]/√n < ∞` for `a > ½`.

**(b) On a compact state space (truncated prior): ergodic, with a strong law.** A1–A4 give SUE
(Lemma 3 of note 18p, `C = 1`, `ρ = 1 − ε`), which is LV26's (A3); (★) is their (A1). With (a):
* marginals `→ π` in TV (RR07 Thm 5 / note 18p Prop. 4);
* WLLN (RR07 Thm 23);
* **SLLN for bounded `f`** (LV26 Thm 9(ii) + Lemma 11, `p = 1`) — for any decay exponent
  `a > 0` in either mechanism;
* a CLT needs `a > ½` **and** LV26 (A4), the convergence of the averaged asymptotic variances
  of the installed kernels. That would follow if the installed surrogate converged (to anything);
  nothing guarantees that for a network trained on a growing data set, so **no CLT is claimed**.
  Consequently ESS-based error bars in a production DA-training stage are heuristic, exactly as
  they are for a frozen stage, only without the CLT that a homogeneous chain would have.

A finite aggressive prefix (the preliminary stages, today's default training) changes none of
this: (a) is a statement about the tail, and the LV26 sums are unaffected by finitely many terms.

**(c) On ℝ^d with a Gaussian prior: ergodic *if* containment holds, and containment is
unproven.** Input clipping keeps (a) intact on ℝ^d (§2.3), so what is missing is only Lemma 3.
The literature on containment (BRR 2011 Thm 6, Saksman–Vihola 2010) covers adaptive
*proposals* with a fixed target; the family `{P_γ}` here has a *moving stage-1 target* and a
K-fold sub-chain, and no published result covers it. The plausible route (note 18p §5.3:
all `π̃_γ` are within `e^{±2B̃}` of the prior, pCN has a uniform spectral gap under bounded
log-likelihood perturbations, Hairer–Stuart–Vollmer 2014) is unproven for the outer kernel,
and for RW sub-chains it would need a uniform gradient bound as well. So on ℝ^d the honest
statement is: *every transition is exactly π-reversible (Christen–Fox), the adaptation
diminishes, and convergence follows under the containment hypothesis* — which is the same
status as most adaptive-Metropolis practice, and strictly better than today's default, for
which DA itself fails.

**(d) Without A4 (no clipping)**: replace clipping by the Sherlock et al. defensive mixture
(an exact MH step with probability `β` per outer step). SUE then comes from the exact kernel on
a compact `X`; no bound on the surrogate is needed. Cost: a fraction `β` of steps without
surrogate screening. Clipping is simpler and also improves robustness, so it remains the
recommendation; the mixture is the fallback if clipping is ever unwanted (e.g. for a
non-Gaussian likelihood where "`d ± cσ`" has no meaning).

**(e) What is *not* delivered by any of this**: a finite-`n` bias bound. DA says the chain
is asymptotically correct; the samples drawn while the surrogate was still moving are biased
by an amount nobody bounds. Discarding the preliminary stages is therefore not optional. If a
guarantee stronger than (b) is wanted for the production samples — a homogeneous chain with a
CLT and standard error bars — freeze the surrogate; that is the degenerate case `D_n = 0`.

## 6. Recommendation (nothing implemented; for the author to decide)

**Design: two consumption modes of one published stream, chosen per stage.**

| | preliminary stage (today's behaviour) | production stage (new option) |
|---|---|---|
| collector | unchanged: trains whenever idle, publishes raw weights | unchanged |
| sampler install | every arriving version (`algorithms.py:505-535`) | same poll; installed weights = sampler-side Polyak average `θ̄_j`, `η_j = j^{−a}`, `a ≈ 0.7`, `j` = this chain's install count in the stage, started from the raw weights at stage entry |
| NN evaluator | as today | inputs clipped to `[−b, b]^d` in standardised coordinates, outputs clipped to `d ± cσ`, received raw weights projected to `‖θ‖_∞ ≤ W` before averaging |
| proposal | adaptive allowed | pinned (`adaptive=False`), as note 16 already recommends |
| guarantee | none (finite prefix; burn-in) | DA deterministic; on a truncated prior: TV convergence + SLLN; on ℝ^d: same under containment |

Why N1′ rather than N3 as the primary: the experiment shows the averaged network is both the
most accurate surrogate and the most efficient sampler, so the guarantee is free; N3 gives
only in-probability DA and a stale surrogate between installs. N3 remains the fallback for a
surrogate whose weights cannot be averaged (a re-initialised network, a non-NN updater that
one nevertheless wants under DA), because it needs nothing but the install coin.

**What it would touch if implemented** (estimate, not done): a `Stage` field for the mode; a
dedicated per-(chain, stage) random stream for nothing (N1′ needs no coin) or one seed offset
for N3; `PyTorchNNEvaluator` gains input/output clipping and a `blend(weights, η)` step; the
install path gains the averaging; `Configuration` gains `a`, `W`, `b`, `c` with defaults 0.7,
10, 8, 5. No MPI message changes. Behaviour change: every production-mode stage produces a
different sample stream from today (flagged per the project rule); preliminary stages and all
non-NN surrogates are unaffected. Tests: unit test of the averaging bound
`‖θ̄_j − θ̄_{j−1}‖ ≤ η_j diam Θ` and of clipping; an MPI run on the GRF example comparing a
production-mode DA stage against a frozen stage (posterior sd, ESS/eval, `Δ_n` on a test set).

**Two things to decide first.**
1. Whether to truncate the prior (declared change of target, needed for the *proof* of
   containment) or to state containment as an assumption on ℝ^d. The library changes nothing
   either way; it is a documentation decision.
2. Whether a production stage with DA training is wanted at all, given §5(e): a frozen stage
   gives a homogeneous chain with a CLT; DA training gives only an SLLN and helps only if the
   surrogate is still improving. The "surrogate still learning" diagnostic proposed on
   2026-09-23 (RMSE trend over the last third of retrainings) would tell the user which case
   they are in; it is worth implementing before either mode.

## 7. Not verified / limits

* Nothing in `surrDAMH/` was run or changed; the experiment is a standalone re-implementation
  of DAMH-SMU (sequential, one chain, retrain after every outer step), not the MPI code path.
  In the library the collector's retrain count per install is set by wall-clock ratios (§1);
  N1′'s deterministic bound is indexed by installs, so this does not affect the argument, but
  the experiment does not exercise asynchrony.
* The Lipschitz-in-weights constant `L` and `diam Θ` are finite but not estimated; the DA
  bound is qualitative. `R L diam Θ` with `W = 10` and 32-wide layers is astronomically large;
  the *observed* `Δ_n` in §4 is what matters in practice and it is tiny.
* Containment on ℝ^d for the outer DAMH kernel with a moving stage-1 target: unproven here
  and, per §3.3, not in the literature.
* LV26 (A4) for a CLT: not established; no CLT is claimed for any DA-training scheme.
* Hamiltonian sub-chains: not covered (note 18p §5.4); the gradient of a clipped network is
  zero outside the clip region, which would need a separate look.
* The experiment is one 4-d problem with a surrogate that was already accurate after the pilot;
  it cannot detect biases below ≈ 0.05 posterior sd and did not test a surrogate that is wrong
  in a region the chain visits late (the Conrad et al. Example B.13 situation).
* Literature: RR07 theorem numbers are from the final preprint; Cui–Fox–O'Sullivan's tech
  report was not seen; Fort–Moulines–Priouret and Davis et al. 2022 abstract-only; Polyak–
  Juditsky step conditions from memory. Full tags in the review file.
* Wall times in §4 are confounded by a concurrent MPI job on the machine.
