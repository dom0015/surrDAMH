# Adaptivity options that would remove guessed sampling parameters — literature research (2026-09-20)

**Dated record.** Requested by the author on 2026-09-20: "research suitable adaptivity options that can
be implemented into the library such that users do not have to guess sampling parameters; use papers
from the internet; write a report with references; test algorithms in Python but do not implement
them into the library." **No library, test or docs code was changed.** Everything here is a
recommendation for later implementation, tied to the measurements of `15_adaptivity_study_2026-09-18.md`
(cited below as "the study", §-references to that file).

How it was produced: three literature reviews (adaptive random-walk covariance methods; HMC/pCN/
function-space proposals; delayed-acceptance and surrogate-specific adaptation) and one numpy
prototyping track, run in parallel by subagents; every reference was verified by the reviewing agent
by fetching its landing page or full text, and the claims this report leans on hardest were re-checked
by the manager against the primary PDF or source file (marked **✔✔** in §8; **✔** = verified by the
agent; *record* = bibliographic record only; *unverified* = neither). Raw material — the three full
reviews (`lit_rw/REVIEW_RW.md`, `lit_hmc/REVIEW_HMC.md`, `lit_da/REVIEW_DA.md`, 29 + 36 + 26
references) and the prototype scripts/CSV/plots (`proto/`) — is in
`toy_examples/out_adaptivity_research_2026-09-20/` (an `out_*` directory; the author decides whether
to keep it).

---

## 1. Result first

**The literature supports one architectural principle and one mechanism per proposal family.**

*Principle.* Adapt the proposal **where every proposal is seen and where the acceptance statistic has
an optimum** — on the surrogate sub-chain in a DAMH stage, on the MH chain in a plain stage — and
leave the delayed-acceptance correction untouched; the DA correction is exact for any fixed first-stage
proposal, so adaptation becomes an ordinary adaptive-MCMC question (diminishing adaptation +
containment, Roberts & Rosenthal 2007) instead of the unbounded feedback loop the study measured
(§3.3). This is what tinyDA ships, what Sherlock–Thiery–Golightly 2021 §6 recommend for the
surrogate-transition method ("the optimal scaling for π is also a sensible scaling for π_a"), and —
independently — the only statistic against which an HMC step size can be dual-averaged in this
library (the outer DAMH rate is flat at 0.97 over a 20× step range, study §2.7).

*Mechanisms* (details in §3, prototype evidence in §4, final ranking in §5). The prototypes moved one
thing relative to the literature: for the random walk, a **shrinkage-regularised AM covariance +
Robbins–Monro scale** beat RAM on every d ≥ 10 problem within 20 000 evaluations (0.64–1.01 vs
0.14–0.67 of oracle), so RAM — the literature's first choice — becomes the robust fallback.

| proposal | mechanism from the literature | guessed knobs it removes | knobs it adds (all with published defaults) | maps to study defect |
|---|---|---|---|---|
| Gaussian random walk (primary, from prototypes) | covariance of **accepted states** with vanishing shrinkage `(1−δ)Ĉ + δ(trĈ/d)I`, `δ=min(1,2d/n)` (Haario 2001 estimator; Ledoit–Wolf-style target; Stan's practice) + Robbins–Monro log-scale `log σ ← log σ + n^{−0.7}(α−0.234)` (Andrieu–Thoms 2008 Alg. 4); **pool samples across MPI ranks** (Craiu–Rosenthal–Yang 2009; Solonen et al. 2012) | initial covariance, `adaptive_corr_limit`, `adaptive_sample_limit`, adaptation length for the hand-over | `γ_n`, `δ` (neither needed tuning) | gated shape update (§2.2-1), non-PSD clip (§2.2-2), unstable window (§2.3), 6–17× rank disagreement (§2.4), lifetime-average hand-over (frozen acceptance 0.22–0.25 vs 0.07–0.44) |
| Gaussian random walk (fallback, literature's first choice) | **Robust Adaptive Metropolis** (Vihola 2012): one rank-one Cholesky update per proposal, `S Sᵀ ← S(I + η_n(α_n−α*)UUᵀ/‖U‖²)Sᵀ`, `α*=0.234`, `η_n=min{1, d·n^{−2/3}}` | same, plus the d² sample appetite (no covariance estimate at all; 10⁴× wrong start still fine) | `α*`, `γ` only | same list; PSD structural; slower shape learning at d = 20 (prototype §4.1) |
| any proposal inside DAMH | adapt on the **sub-chain's own** acceptance (tinyDA; Sherlock et al. 2021 §6), update parameters only at outer-iteration boundaries | the DA-stage proposal scale; the endpoint-vs-per-step question; makes adaptation independent of `subchain_max_length` | — | divergence of `adaptive=True` in DAMH (§3.3, §2.5) |
| Hamiltonian / HamiltonianInfinite | **dual averaging of the step size** (Hoffman & Gelman 2014 eq. 6: `γ=0.05, t₀=10, κ=0.75, μ=log 10ε₀`) to `δ≈0.65` (Beskos et al. 2013) **on the sub-chain acceptance, at fixed integration time `T=ε·L`**, `L=⌈T/ε⌉`; jitter `L` (Neal 2011 §4.2) | `hamiltonian_step_size`; silent 100 % pre-rejection at ε=0.5 | `T`, `δ`, warm-up length | 3× ESS between ε 0.02 and 0.05, flat outer rate (§2.7, §3.2) |
| Hamiltonian mass | **likelihood-informed mass from the surrogate Jacobian**: `M = F̃ + I` (Beskos et al. 2017 eq. 35, `K⁻¹ = F̃ + C⁻¹`) for `HamiltonianInfinite`, `M = Σ̂⁻¹` for `Hamiltonian` (Neal 2011 §4.1; Stan); LIS threshold λ ≥ 0.1 (Cui et al. 2014); **must ship with step-size re-adaptation** because metric and step are inseparable | `proposal_sd_or_cov` on Hamiltonian stages | LIS threshold | 3.3× mass effect and the toy/GRF "contradiction" (§3.2), which §3.3 below resolves |
| pCN | per-mode proposal variances in the prior KL basis (Hu, Yao & Li 2017: `ρ=0.99`, `λ_j = min{λ̂_j, α_j}`, diminishing-adaptation proof) + Robbins–Monro on β | `pcn_beta` (partly) | `ρ` | nothing adapts β today (§4.1) |
| surrogate / DA machinery | report `η` (surrogate/exact cost ratio), `α₁`, `α₂|₁`, ESS per exact evaluation **and** per second (Sherlock et al. 2021 §2.3 needs exactly `η` and `α̂₂|₁`); optional adaptive Gaussian error model (Cui–Fox–O'Sullivan 2019; Lykkegaard et al. 2021, 2023); K from the exact-stage acceptance ∈ [0.2, 0.4] (Lykkegaard et al. 2021) or randomised `K~U{1..J}` (MLDA) | `subchain_max_length` (rule, not adaptation); "is the proposal family right for my solver cost" (14× per-second disagreement, §3.3) | — | nothing reports ESS or `η` today (§4.2 last rows) |

**Not recommended** on this evidence: NUTS (recursive control flow vs the fixed-length sub-chain and
MPI batching; validity inside a DA step uncovered by any source), RMHMC (5–6 implicit fixed-point
iterations per step), MEADS (changes the proposal family to generalized HMC), emcee-style ensembles
(need ≫ d walkers moving together — conflicts with one-chain-per-rank + shared surrogate), textbook AM
without acceptance feedback, and any Hamiltonian step-size adaptation driven by the *outer* DAMH rate.

**Where theory is missing** (so the report says "structural argument", not "cited"): no source gives a
tuning rule or modified acceptance target for HMC under delayed acceptance; no diffusion theory exists
for the surrogate-transition method with K > 1 (Sherlock et al. 2021 say so explicitly); no
published adaptive-MCMC paper plugs Ledoit–Wolf shrinkage into AM (the study's "ungate + shrink"
prototype stands on Stan's regularisation practice and its own §2.6 measurement); no pCN-specific
optimal acceptance theorem exists (0.234 borrowed from RWM in Cotter et al. 2013; CUQIpy hardcodes
`0.44 #TODO: 0.234`).

---

## 2. What the literature says about the defects the study measured

Each item names the measurement, the explanation the literature gives, and the reference.

1. **`adaptive=True` inside a DAMH stage diverges (2.8 M iterations, 229 exact evaluations, sd 0.3 → 3.9; §3.3).**
   In Sherlock–Thiery–Golightly's notation the library's `adapt()` is fed the *conditional stage-two*
   acceptance `α₂|₁` (only iterations whose sub-chain moved reach it), not the overall rate
   `α₁₂ = α₁·α₂|₁`. They show `α̂₂|₁` "is a measure of the accuracy of π_a: if the approximation were
   perfect, this would equal one" (✔✔). It has no optimum at 0.25, so coercing it to 0.25 by inflating
   the scale is an unbounded loop — the divergence needs no bug to explain it. Fix: adapt on the
   sub-chain (tinyDA `DAChain._sample_coarse` → `proposal.adapt(accepted=self.accepted_coarse)` ✔✔),
   or score pre-rejected iterations as acceptance 0 (that yields `α₁₂`) and set the target from
   Sherlock et al.'s Figure 1 using measured `η` and `α̂₂|₁` rather than 0.234.
2. **With K > 1 the adapted per-step scale is 2.4–9.3× too large at K=5 and 5–10× too small at K=20 (§2.5).**
   No diffusion theory exists for K > 1 ("our theory also relies on independence between the components
   of x*−x, which does not hold after multiple sub-iterations", Sherlock et al. 2021 §6 ✔✔); their
   practical advice is *not* a √K correction but "the optimal scaling for π is also a sensible scaling
   for π_a, whilst the number of sub-iterations should be no more than is required for approximate
   convergence, and fewer if π_a is relatively expensive to evaluate" (✔✔). Under sub-chain adaptation
   the question disappears; K becomes a separate knob (rules in §3.2).
3. **Textbook AM `(2.38²/d)(Ĉ+εI)` collapsed from d=10 up when Ĉ came from proposed samples (§2.6).**
   Andrieu & Thoms 2008 Algorithm 3 (Rao-Blackwellised AM) uses
   `Σ ← Σ + γ[α(Y−μ)(Y−μ)ᵀ + (1−α)(X−μ)(X−μ)ᵀ − Σ]`; the library's
   `np.cov(proposed_samples, aweights=α)` (`proposals.py:331` ✔✔) keeps only the first term and drops
   the `(1−α)(X−μ)(X−μ)ᵀ` anchor to the chain state. That this is *the* cause was an inference from
   the two formulae — **and the prototype refutes it** (§4.1): accepted-state vs proposed-sample
   estimation moves the result ≤ 2× in both directions, while the ridge/shrinkage is worth 16–19× at
   d = 20. The collapse is a *regularisation* failure (a sticky chain under-estimates the narrow
   directions and AM has no acceptance feedback to correct it), which is also why the shipped `coef`
   feedback was load-bearing.
4. **Entrywise correlation clipping to 0.3 breaks positive-definiteness at d ≥ 10 and binds on GRF where max|ρ|=0.17 (§2.2, §3.1).**
   Clipping appears nowhere in the adaptive-MCMC literature. Containment (the second condition of
   Roberts & Rosenthal 2007 Thm 13) is enforced either by Haario's `+εI` or by Roberts–Rosenthal's
   `β=0.05` mixture with a fixed `N(x,(0.1)²I/d)` kernel; Vihola 2011 shows the `εI` lower bound is not
   even necessary for AM's eigenvalues to avoid collapse. RAM makes PSD structural (Vihola 2012 Prop. 1).
5. **A bounded history (`adaptive_sample_limit`) is unstable — 4 orders of magnitude blow-up (§2.3).**
   This is the theoretical name "diminishing adaptation fails": with an unbounded history each update
   changes the kernel by O(1/n); with a window the change does not go to zero. The fix is a
   decaying-weight recursion (RAM's `η_n`, Andrieu–Thoms `γ_n`, Welford/Solonen pooled update), not a window.
6. **Per-rank covariances differ 6.6–17× and are averaged by `Allreduce` (§2.4, §3.1).**
   Inter-chain adaptation in the literature *pools samples* ("setting the proposal covariance to the
   sample covariance matrix of all the available samples", Craiu–Rosenthal–Yang 2009); averaging finished
   covariances has no precedent. Solonen et al. 2012 measured 10 pooled chains converging in ≈200 steps
   against >1500 for one chain, saturating around 10 chains — with 3–6 sampler ranks expect a 3–6×
   shorter adaptation transient.
7. **The preliminary adaptive stage never learned the shape at d=20 in 90 s / 137 k evaluations (§3.1).**
   Roberts & Rosenthal 2009 measured AM's sample appetite: sub-optimality factor `b` from 193 to 1.086
   after 500 000 iterations at d=100, ≈2·10⁶ at d=200 — of order 40–50·d² samples. At d=20 that is
   10⁴–10⁵ evaluations *per pooled chain set*. RAM sidesteps this by never forming a covariance
   estimate (Table 1: RAM 1.03–1.61 RMSE at d=32 vs AM 33.87 ✔✔).
8. **Hamiltonian: step 0.02 beats the shipped 0.05 by 3×, the outer acceptance is flat at 0.97, ε=0.5 pre-rejects 100 % silently (§2.7, §3.2).**
   The outer DAMH acceptance measures surrogate error, not integrator energy error, so it cannot drive
   step-size adaptation; the sub-chain's own Metropolis probability is the HMC acceptance of Hoffman &
   Gelman / Beskos et al. 2013 and can be coerced to 0.65. A second reading of the study's two sweeps
   (HMC review §1, analysis not measurement): both are consistent with integration time `T = ε·L ≈ 1–2`
   rad being the optimum (0.02×100 → T=2 best; 0.05×20 → T=1 second; 0.05×100 → T=5; L=400 → T=20
   worst), so "3× for ε=0.02" may be a trajectory-length effect. Dual averaging at fixed L would chase
   the wrong thing; every production implementation parametrises by T (Stan, PyMC, BlackJAX, TFP).
9. **Mass carry-over: `M=Σ̂` helped `HamiltonianInfinite` 3.3× on GRF but froze the fully-informed toy; `Hamiltonian` wanted `Σ̂⁻¹` (§3.2).**
   Resolved in §3.3 below: the canonical ∞-HMC has no free mass (whitened: `M=I`), the literature's
   data-informed choice is `K⁻¹ = F̃ + C⁻¹` (posterior precision, ✔✔), and the two measurements differ
   only because a mass change silently rescales the *effective step* per direction — `M=Σ̂` won at the
   shipped ε=0.05 because it accidentally landed near the split integrator's stability optimum.
10. **DAMH-Hamiltonian is 14× slower per second than DAMH-pCN K=5 on the 0.19 ms GRF solver (§3.3).**
    Christen & Fox 2005: DA is dominated by plain MH in asymptotic variance (Peskun) and can only win on
    cost; with a good, free surrogate the speedup ceiling is `1/acceptance rate` — at the GRF pCN
    stage's 0.69 acceptance that is ≈1.45×. Sherlock et al. 2021 make the cost ratio `η` a *required*
    tuning input and call a surrogate only 10× cheaper than the model "of dubious utility" (✔✔).
    None of `η`, `α₁`, `α₂|₁`, ESS/evaluation or ESS/second is computable from today's output.

---

## 3. The options, by family

### 3.1 Gaussian random walk: RAM as the primary, regularised AM as the fallback

**RAM — Vihola 2012** (✔✔ full text). Proposal `Y = X + S U`, `U ~ N(0,I)` (or Student); after each
proposal `S Sᵀ ← S (I + η_n (α_n − α*) U Uᵀ/‖U‖²) Sᵀ`, `S` the Cholesky factor of the right-hand side
(exists for any `α_n` — Proposition 1), `α* = 0.234`, `η_n = min{1, d·n^{−2/3}}` (the paper's own
experimental choice; `γ = 1` "is not advisable" because a bad start is then never forgotten). Cost
O(d²) per step as a rank-one Cholesky update — "only a constant factor" over generating the proposal.
Fixed point: for elliptically symmetric targets `S_* S_*ᵀ ∝ Σ` (Thm 7), `S_n S_nᵀ → S_* S_*ᵀ` a.s. (Thm
18). Evidence (Table 1, RMSE of Gaussian HPD quantiles, 500 k iterations): AM 0.21 / 0.33 / 1.25 / 6.83
/ **33.87** at d = 2…32; RAM from `s₁=I` 0.21 / 0.27 / 0.37 / 0.52 / **1.03**; RAM from `s₁ = 10⁴·I`
0.22 / 0.28 / 0.45 / 0.75 / **1.61** (AM from the same start: 58.20). Not suitable for strongly
multi-modal targets. Shipped by `adaptMCMC` (R, `gamma=2/3` default) and optionally `pymcmcstat`.

Why it fits this library: shape and scale are one object (the gating defect cannot exist); PSD is
structural (`corr_limit` deleted, not retuned); no history and no covariance estimate (`sample_limit`
and the O(n²) cost go away; the d² sample appetite goes away); `η_n → 0` gives diminishing adaptation;
what is handed to the next stage is the current `S`, not a lifetime average; the user's initial
covariance stops mattering. Open question the literature does not answer (RW review §9): RAM's
behaviour when `α_n` is the outer DA ratio — moot under sub-chain adaptation, where `α_n` is the
sub-chain's own acceptance.

**Fallback — regularised AM with acceptance feedback.** Roberts & Rosenthal 2009:
`Q_n = (1−β) N(x, 2.38² Σ_n/d) + β N(x, 0.1² I/d)`, `β = 0.05`, for `n > 2d`; Andrieu & Thoms 2008
Alg. 4: `log λ ← log λ + γ_n(α − 0.234)` with `γ_n ∝ n^{−α}`, `α ∈ [0.5, 0.7]` (their "adaptation
penalty ~ N^{−α}/(1−α)"), and the complete Rao-Blackwellised recursion of Alg. 3. This keeps the
library's structure and the `coef` feedback the study found load-bearing, removes `corr_limit`,
`sample_limit` and `ε`, but not the d² appetite.

**Target rate: keep 0.234, do not expose it prominently.** Roberts & Rosenthal 2001: "any algorithm
with acceptance rate between say 0.15 and 0.5 will be at least 80 % efficient"; Rosenthal 2010: "the
asymptotics approximately apply whenever d ≥ 5"; 0.44 for d = 1 / Metropolis-within-Gibbs. The
study's 25 % spread over 0.234/0.25/0.4 (§2.3) is exactly this flatness. Sherlock & Roberts 2009 warn
that 0.234 fails for extremely eccentric targets — a caveat for the rank-2 GRF posterior.

**Multi-chain pooling.** Craiu–Rosenthal–Yang 2009 (INCA): after burn-in all K chains adapt from the
pooled samples, chains independent given the adapted parameter, so the ergodicity proofs go through;
they stop pooling when Gelman–Rubin `R < 1.1`. Solonen et al. 2012 give the pooled Welford recursion
applied once per chain per sweep. For RAM the analogue is to apply each rank's update locally and
synchronise `S` periodically (DIAM, Chen et al. 2016, "periodically synchronized concurrent chains").

**Shape metric worth adopting in reports.** Roberts–Rosenthal's sub-optimality factor
`b = d·Σλᵢ⁻²/(Σλᵢ⁻¹)²` (λᵢ eigenvalues of `Σ_p^{1/2} Σ^{−1/2}`, `b ≥ 1`, `b = 1` ⇔ proportional) says
directly how far a proposal covariance is from optimal — better than the Frobenius shape error the study used.

### 3.2 Delayed acceptance and the surrogate: where to adapt, K, error model, diagnostics

**Adapt on the sub-chain** (tinyDA ✔✔: `GaussianRandomWalk`/`AdaptiveMetropolis`/`CrankNicolson` all
share `period=100`, `alpha_star=0.24`, windowed `acceptance_rate = mean(accepted[-period:])`,
`scaling ← exp(log scaling + γ^{−k}(â − α*))` with `γ = 1.01`, `k` the adaptation count; note this
decay is geometric, so the total adaptation is finite — valid, but unlike the polynomial `n^{−γ}`
schedules it cannot recover from a start that is wrong by more than the finite budget allows).
**Manager's design note (not from the literature):** the surrogate-transition acceptance
`π(y)π_a(x)/(π(x)π_a(y))` requires the K-step sub-chain kernel to be `π_a`-*reversible*, and a
composition of reversible kernels with *different* parameters is `π_a`-invariant but not reversible in
general. So proposal parameters must be frozen for the duration of one sub-chain and updated only at
outer-iteration boundaries (with `subchain_max_length=1`, tinyDA's default, the issue does not arise;
for K > 1 tinyDA's per-step call with `period=100` can change the scale mid-sub-chain). With the
surrogate itself being updated during a DAMH-SMU stage the sub-chain target `π_a` moves as well; the
outer correction stays exact and the adaptation just tracks a moving target — freeze everything in the
final fixed-surrogate stage.

**What the DA theory prescribes for tuning** (Sherlock–Thiery–Golightly 2021 ✔✔). Efficiency
`Eff_da(μ) = μ² α₁₂(μ)/(η + α₁(μ))` vs `Eff_rwm(μ) = 2μ²Φ(−μ/2)` (optimum μ̂ ≈ 2.38); recipe §2.3:
tune plain RWM to `λ̂_rwm`, measure `α̂₂|₁ = α₁₂/α₁` and `η` (timing), read `λ̂_da/λ̂_rwm` from their
Fig. 1; worked example: `α̂₂|₁ ≈ 0.75`, `1/η ≈ 2·10⁴` → `λ̂_da ≈ 2.9 λ̂_rwm`, minESS/sec 1.28 → 17.4, "a
thirteenfold improvement" (grid optimum 18.7). Banterle et al. 2019 Prop. 4: the optimal DA acceptance
depends only on the cost ratio, not on the target. Christen & Fox 2005: ceiling `1/acceptance rate`.
Consequence for the library: the *second-stage* proposal scale should be larger than plain-RWM
optimal by a factor that depends on `α̂₂|₁` and `η` — both must be measured and reported.

**Sub-chain length K.** No paper adapts K online. Rules that exist: Lykkegaard–Dodwell–Moxey 2021 tune
the offset length so the *exact-stage* acceptance lands in [0.2, 0.4], falling back to K=2 when the
surrogate is weak ("longer subchains tended to diverge"); MLDA (Lykkegaard et al. 2023) draws
`K ~ U{1,…,J}` (randomisation makes their variance-reduced estimator unbiased and decouples the
proposal from one endpoint); Sherlock et al. 2021 §6: "no more than is required for approximate
convergence, and fewer if π_a is relatively expensive". Defaults elsewhere: tinyDA `subchain_length=1`,
PyMC MLDA `subsampling_rates=5`, neither with guidance.

**Adaptive Gaussian error model on the surrogate likelihood.** Cui–Fox–O'Sullivan 2019 (bias
`B = F − F̂ ~ N(μ_B, Σ_B)`, recursive moments, adaptation strength `δ = min{0.01, √(N/n)}`; with a fixed
surrogate DA "requires many times more iterations" than exact MCMC, with the adaptive state-dependent
error model "virtually the same number"); Lykkegaard et al. 2021 eqs 22–23 (`Σ_bias + Σ_e`, shift by
`μ_bias`; caveat: bias and observation noise are "observationally equivalent, and not well-defined");
MLDA eqs 27–28; validity by diminishing adaptation. Shipped by tinyDA (`adaptive_error_model ∈ {None,
'state-independent', 'state-dependent'}`) and PyMC MLDA. It raises `α₂|₁`, which is what buys a larger
scale and a safer K. The collector already stores the (surrogate, model) output pairs needed.

**Refinement / freezing.** Conrad–Marzouk–Pillai–Smith 2016: refine with probability `β_t = 0.01 t^{−0.2}`
(`Σβ_t = ∞` required and sharp) or when a leave-one-out perturbation of the *acceptance probability*
exceeds `γ_t = 0.1 t^{−0.1}` — chosen because an error in acceptance probability is problem-independent
whereas an error in forward output or log-likelihood is not. Sherlock–Golightly–Henderson 2017 (the
closest published design to this library — kNN surrogate built from the chain's own evaluations inside
DA, ergodicity proof): tree update with probability `p_i = (1+ci)^{−1}`, `c ∈ {10⁻⁴,10⁻³,10⁻²}` gave
relative ESS 7.28/6.80/4.67 and no adaptation was worst; merge distance `ε ≈ √(2 q_{χ²_d}(1/2n))` so a
new snapshot is stored only if its ε-ball holds < ½ an existing point in expectation; proposal scale
`ξ²V` with `V` from a plain-RWM run, `ξ = 3` optimal (6.8× over tuned pseudo-marginal RWM); they state
the open problem the study hit: "It might also be possible to adaptively update the scaling of the
proposal; however the mechanism to use is less obvious". Under DA, freezing the surrogate is a *cost*
decision, never a correctness one; the natural freeze statistic is `α̂₂|₁` (or Conrad's indicator), not
the held-out RMSE.

**Diagnostics.** Vats–Flegal–Jones 2019 (multivariate ESS with an a-priori minimum-ESS termination
rule), Vehtari et al. 2021 (rank-normalised split-R̂, bulk/tail ESS), Geyer 1992 (the initial-sequence
ESS the library already uses in post-processing). No software documents an "ESS per second" column
(Stan's posterior-analysis page defines ESS/MCSE only), but every DA paper reports minESS/sec, and the
study's 14× per-evaluation/per-second disagreement is exactly `η`.

### 3.3 Hamiltonian family: step size, trajectory length, mass — and the whitened ∞-HMC question

**Step size — dual averaging** (Hoffman & Gelman 2014 eq. 6, ✔): `x = log ε`, `H_t = δ − α_t`,
`x_{t+1} = μ − (√t/γ)·(1/(t+t₀))·Σ_{i≤t} H_i`, `x̄_{t+1} = η_t x_{t+1} + (1−η_t) x̄_t`, `η_t = t^{−κ}`;
Algorithm 5 constants `γ = 0.05, t₀ = 10, κ = 0.75, μ = log(10 ε₀)`; freeze at `x̄` after warm-up.
Target: Beskos–Pillai–Roberts–Sanz-Serna–Stuart 2013 prove 0.651 optimal for product targets
(`h ∝ d^{−1/4}`); Hoffman & Gelman: "HMC's best performance seems to occur around δ = 0.65"; production
defaults sit higher (Stan `adapt_delta=0.8`, PyMC `target_accept=0.8`, BlackJAX window adaptation 0.8;
BlackJAX ChEES uses 0.651). Neal 2011 states no optimal rate (misattribution corrected by the review).
In this library the statistic must be the **sub-chain's** Metropolis probability on the surrogate
posterior (its complement is the logged pre-rejection rate), never the outer DAMH rate; and the
adaptation must run at **fixed integration time `T`**, `L = ⌈T/ε⌉`, or ε adaptation moves the trajectory
length too (§2 item 8). Surrogate gradients change nothing here: Rasmussen 2003, Zhang–Shahbaba–Zhao
2017, Li et al. 2019, Lan et al. 2016 and Cao–O'Leary-Roseberry–Ghattas 2024 (derivative-informed
neural operator driving DA geometric MCMC — the closest published analogue of this library) all drive
the integrator with an approximate gradient and correct exactly, with no modified target.

**Trajectory length.** Jitter `L` over a small interval (Neal 2011 §4.2) — three lines, and especially
relevant to `HamiltonianInfinite`, whose free flow is an *exact* rotation and therefore exactly periodic
in the prior directions (T = 2π returns them to the start). NUTS: not recommended (see §1). **ChEES-HMC**
(Hoffman–Radul–Sountsov 2021; formulas from the TFP/BlackJAX implementations, the PDF was not
machine-readable) adapts the trajectory length by gradient descent on a change-in-estimator criterion
across ≥ 2 parallel chains, jittered `U[(1−a)·max, max]`, paired with dual averaging to 0.651 — the
best structural fit for an MPI code with several chains and the strongest option *not* in the top
list, ranked lower only because it needs a cross-rank reduction every iteration. SNAPER (Sountsov &
Hoffman 2021) additionally learns a diagonal mass along the principal component and is "stable when
combined with mass-matrix adaptation".

**Mass matrix.** Stan's windowed warm-up (✔ from `covar_adaptation.hpp`):
`Σ̂ ← (n/(n+5)) Σ̂ + 10⁻³·(5/(n+5))·I`, metric `M⁻¹ = Σ̂`, windows 75/25(doubling)/50, step size
re-adapted in the final window. Neal 2011 §4.1: momentum covariance `Σ⁻¹`. Girolami & Calderhead 2011:
`M = G(θ)` (Fisher), 5–6 implicit iterations per step — too costly unless emulated.

**The whitened ∞-HMC question (study §3.2), resolved from primary sources (✔✔ Beskos et al. 2011, 2017;
`proposals.py:494–537`).** (a) The canonical Hilbert-space HMC has no free mass: with `M = L` (prior
precision) every direction oscillates at angular frequency 1 and the split integrator rotates all
directions by the same angle; whitened (`C = I`) that is `M = I`, the shipped default.
`HamiltonianInfinite`'s per-direction angle `ε/√λ_M` with `p/√λ_M` mixed in (prior-preserving for any
PD `M`) is a generalisation beyond the published algorithm — effectively a per-direction pCN β, not a
metric in the literature's sense. (b) The literature's data-informed choice is the *posterior*
precision, not the prior: Beskos et al. 2017 eq. (35) `K⁻¹(u) := F̃(u) + C⁻¹`, `F̃` the Fisher operator
truncated to the first D₀ prior modes (D₀ = 25 of 100 in their first example), with plain ∞-MALA/∞-HMC
recovered "under the setting K = C, when scales are tuned to the prior". In the library's
parametrisation that is `M = F̃ + I = Σ̂⁻¹` (whitened Gauss–Newton posterior precision). (c) The two
measurements do not contradict each other: the frequency of eigen-direction i is `1/√(λ_Σ λ_M)`, so
`M = Σ⁻¹` gives frequency 1 everywhere (the toy result); what differs is the *integrator's* stability
constraint, which for the split integrator involves only the likelihood Hessian `h` (`λ_Σ = 1/(1+h)`):
roughly `ε·√(h/λ_M) ≲ 2`. At the GRF numbers (`λ_Σ = 0.044`, `h ≈ 21.7`, `ε = 0.05`): `M=I` → 0.23,
`M=Σ̂` → 1.11, `M=Σ̂⁻¹` → 0.049. `M=Σ̂` won by accidentally setting the right *effective step*; `M=Σ̂⁻¹`
shrank the effective step 5× at the shipped ε and wasted the budget; on the toy, `M=Σ` put every
direction at the stability limit, hence the frozen chain. **Rule: a mass change rescales the effective
step by `√(λ_M,old/λ_M,new)` per direction, so any mass carry-over must be followed by step-size
re-adaptation** — exactly Stan's/BlackJAX's ordering (mass in slow windows, ε in the final fast window).

**Likelihood-informed subspace from the surrogate Jacobian.** Cui–Martin–Marzouk–Solonen–Spantini 2014:
eigen-decompose the prior-preconditioned Gauss–Newton Hessian `LᵀJᵀΓ⁻¹JL`, keep `λ ≥ τ = 0.1`; Spantini
et al. 2015 prove these directions optimal for the low-rank posterior update; DILI (Cui–Law–Marzouk
2016) adapts the subspace online (Alg. 4) and uses pCN in the complement — it removes the *shape* knob,
not the step knob (full-text search: no Robbins–Monro/target-acceptance rule; the brief's premise was
wrong). hIPPYlib-MUQ: every curvature-informed sampler still carries a scalar τ. The library-specific
opportunity: the NN surrogate already exposes `jacobian`/`vjp` and the GRF example has 4 observations,
so `JᵀΓ⁻¹J` is a 4-row product — essentially free — and an erroneous LIS degrades efficiency, not
correctness, because the accept/reject stays exact.

### 3.4 pCN

Cotter–Roberts–Stuart–White 2013 tuned β empirically to ≈0.234 acceptance (borrowed from RWM);
Pillai–Stuart–Thiéry 2012's 0.574 is for MALA, not pCN; Hairer–Stuart–Vollmer 2014 prove the
dimension-independent spectral gap without an acceptance number. CUQIpy: Robbins–Monro on the scale
with `ζ = 1/√(n+1)`, target `0.44 #TODO: 0.234` (✔✔). **Adaptive pCN — Hu, Yao & Li 2017 (✔✔ full
text):** generalised pCN whose proposal covariance `B` shares the prior's KL eigenfunctions (it *must*
commute with `C₀`, Prop. 2.3); the first J eigenvalues are running posterior-variance estimates
**capped at the prior eigenvalue**, `λ_j = min{λ̂_j, α_j}` (keeps `I − β²BL` positive), `λ_j = α_j`
beyond; `J = min{j : Σ_{i≤j} α_i / Σ α_i > ρ}`, `ρ = 0.99` → J = 14 in both examples; a fixed number of
plain-pCN steps before adapting; β fixed at 1/5 (≈20 % acceptance, where plain pCN needed β = 1/300);
diminishing-adaptation proof. In this library the internal prior *is* `N(0,I)`, so "diagonal in the KL
basis" is automatic and the method reduces to per-coordinate variances — the lowest-effort adaptive
option in this report, aimed at the preliminary stage that actually feeds the surrogate (study §4.3).

---

## 4. Prototype evidence (`proto/RESULTS_PROTO.md`, `results_proto.csv`: 1605 runs)

Standalone numpy re-implementations (formulas in `RESULTS_PROTO.md` §1, checked by the manager against
Vihola 2012 step R3 and Hoffman & Gelman 2014 Alg. 5) on the study's closed-form toy family (whitened
prior, `G = Ax`, prescribed posterior `C = Q diag(λ) Qᵀ`, λ log-spaced 0.5 → 0.5/κ; d ∈ {2,10,20},
κ ∈ {1,10,100}); the shipped `GaussRandomWalk_adaptive` imported unmodified as the baseline. Metrics
fixed beforehand: min-over-coordinates Geyer ESS per evaluation ÷ oracle RW `(2.38²/d)·C`; shape error
`‖P/trP − C/trC‖_F/‖C/trC‖_F`; posterior check `max|mean−μ|/SE ≤ 4` (batch means). 3 seeds; only
differences ≳ 1.5× read as real. Every adaptive method starts from the naive isotropic `sd = 1`.
**Correctness: 0 posterior-check failures among the 248 runs with ESS ≥ 200; all 255 failures have
ESS < 200 — no detectable bias in any method** (the check cannot discriminate on the hard problems).

**4.1 Random walk, 20 000 evaluations (ESS/oracle, mean of 3 seeds)**

| problem | shipped | ungate+shrink (study §2.6) | RAM | AM acc. states, ε=1e-6 | AM acc. + vanishing shrinkage | **AM-shrink + Robbins–Monro** | RM, true shape (ceiling) |
|---|---|---|---|---|---|---|---|
| d2 κ10 | 0.52 | 0.88 | 0.89 | 0.94 | **1.02** | 0.87 | 0.88 |
| d10 κ10 | 0.31 | 0.92 | 0.67 | 0.31 | 0.82 | **1.01** | 1.00 |
| d10 κ100 | 0.06 | 0.26 | 0.34 | 0.03 | **0.70** | 0.64 | 0.86 |
| d20 κ10 | 0.18 | 0.25 | 0.34 | 0.02 | 0.52 | **0.91** | 0.98 |
| d20 κ100 | 0.05 | **0.03** | 0.14 | 0.01 | 0.42 | **0.65** | 1.02 |

* **Scale is solved, shape is the whole gap**: Robbins–Monro handed the *true* shape is 0.88–1.02 of
  oracle everywhere; RAM and RM hit their target acceptance within 0.002 on every problem; the shipped
  rule's seed spread (0.03–0.82 at d10κ10) exceeds most differences the study ranked on.
* **The combination "shrinkage-regularised AM covariance of accepted states + Robbins–Monro log-scale"**
  (`Ĉ ← (1−δ)Ĉ + δ·(trĈ/d)I`, `δ = min(1, 2d/n)`; `log σ ← log σ + n^{−0.7}(α − 0.234)`;
  `P = σ²(2.38²/d)(Ĉ + 10⁻⁶I)`) — a configuration from two papers, not one — is best on 3/5 problems,
  1.5–4.6× the next best at d = 20, has the lowest shape error at every d ≥ 10 (0.077–0.252 vs RAM
  0.244–0.632, shipped 0.59–0.67, identity 0.56–0.78), emits no numpy warnings. Caveat: over-scales at
  d = 2 (scale ratio 1.41, the `2.38²/d` constant is asymptotic; dropping it was not tested).
* **RAM is the robust fallback**: never worse than shipped (1.7–5.6×), smallest seed spread (0.04–0.16),
  two constants, structurally no indefinite covariance. `γ = 0.9` fails (acceptance 0.006 at d20κ100):
  ship `γ = 2/3`. `α*` 0.234 vs 0.25 within noise.
* **The shipped rule emitted 19 750 "covariance is not symmetric positive-semidefinite" warnings in one
  20 000-evaluation run** (51 782 over 17 of 78 runs): once the clip makes the matrix indefinite,
  essentially every later proposal is drawn through numpy's SVD fallback from a covariance nobody specified.
* **Textbook AM's collapse is the ridge, not the estimator** (§2 item 3 revised): accepted-state vs
  proposed-sample covariance moves the result ≤ 2× *in both directions*; ε = 10⁻³ vs 10⁻⁶ is worth
  3.2× / 19× / 19× / 16× on d10κ10 / d10κ100 / d20κ10 / d20κ100. Diminishing-gain AM
  (`P ← P + n^{−0.7}(target−P)`) and unregularised AM frozen after n₀ are both poor (0.03–0.36; frozen
  acceptance up to 0.85).
* **The study's §2.6 favourite "ungate + shrink" is retracted**: 0.03 of oracle at d20κ100 with shape
  error 1.015 (worse than the identity), and the worst adapt-then-freeze hand-over of all — the `coef`
  feedback it inherits is the part that misbehaves.
* **Adapt-then-freeze hand-over** (n₀ = 3000, frozen stage 20 000 evaluations, min–max over seeds and
  problems): RM 0.223–0.246, AM-shrink+RM 0.222–0.252, RAM 0.195–0.267, **shipped 0.07–0.44** (the
  study's §2.4 "anywhere in 0.02–0.65", cause: `current_rate` is a lifetime mean). Frozen-stage
  ESS/eval at d10κ100: combination 0.0125 vs shipped 0.00032 (39×).

**4.2 DAMH stage, fixed wrong surrogate `A' = 1.3A`, K ∈ {1,5}, d ≤ 10, every run started at the oracle covariance**

| policy | scale drift vs per-step oracle | pre-rejection | exact evals reached / 5000 | divergences |
|---|---|---|---|---|
| shipped information (adapt only on moved iterations) — shipped rule, RAM **and** Robbins–Monro alike | 3.1–97× | 0.90–1.00 | **8–569** | **17 / 18 cells** |
| DA-aware: every outer iteration counts, pre-rejected = α 0 | 0.74–2.84× (K=5 drift 1.6–2.8× is the endpoint-vs-step effect) | 0.35–0.69 | 5000 | 0 / 18 |
| sub-chain-local: adapt the sub-chain proposal on its own acceptance, α* 0.234 | 0.98–1.35× at d2/d10κ10; 3.6× at d10κ100 (tunes to the *surrogate*) | 0.13–0.76 | 5000 | 0 / 18 |

**The divergence is the information restriction, not the rule** — the §2 item 1 explanation, isolated.
Per *total* (surrogate + exact) evaluation the shipped policy is 2–3 orders of magnitude behind either
fix. Not verified: whether DA-aware adaptation has a fixed point when pre-rejection is intrinsically
≈ 0.6 (it creeps downward, i.e. safely); d > 10; a real surrogate; on d10κ100 every adaptive DAMH
policy mixes poorly at this budget (`post_dev_se` 5.8–23.6 with ESS 8–45 — could not check bias).

**4.3 HMC dual averaging (exact gradients, standard leapfrog and the `HamiltonianInfinite` split integrator, whose rotation was first checked to conserve `½q·q + ½pᵀM⁻¹p` to 2·10⁻¹⁴ for M = I, Σ, Σ⁻¹)**

* Fixed-step grid, mass I: acceptance `0.999 → 0.99 → 0.96 → 0.72–0.78 → 0.000` between ε = 0.1 and
  0.2 (stability limit `2√λ_min = 0.141`) while ESS varies 40× on the flat part — the toy counterpart of
  the study's flat 0.97 and the silent `step_size = 0.5` failure.
* Dual averaging lands ε̄ within 1.2–2× of the grid optimum and **below the cliff** with mass I; with
  mass Σ⁻¹ at *fixed L* δ = 0.65 overshoots ~2× into the `εL ≈ 2π` resonance, δ = 0.8 does not —
  **δ = 0.8 at fixed L**; at fixed T (the literature's parametrisation, §3.3) the resonance would not
  move with ε, so 0.65 may be fine there — *untested*. Warm-up ≥ 2000 evaluations (converges from
  above; 500 far too short); ε₀ ∈ {0.01, 0.1, 1} → ε̄ 0.61 / 0.68 / 0.66 — the initial step is no
  longer a guess.
* **Mass (with L scanned so the effective `εL` is not confounded)**, ESS/eval ÷ RW oracle, best cell:
  fully-informed d20κ100 — `M=I` 2.08, `M=Σ` **0.06**, `M=Σ⁻¹` **19.0**; partially-informed d20
  (λ = 0.05, 0.5, 18×0.95) — `I` 9.8, `Σ` 1.23, `Σ⁻¹` **29.5**. **`Σ⁻¹` wins on both, `Σ` is bad on
  both**; at the fixed L = 5 of the study the ranking inverts on the partially-informed target purely
  through `ε̄L ≈ 5.9 ≈ 2π`. This is the experimental side of §3.3(c): the GRF's `Σ̂` win was an
  effective-step/trajectory-length artefact, and any mass carry-over must re-adapt ε *and* T. Caveat:
  the best cells sit on the ESS estimator ceiling `0.9/(L+1)`, so their mutual ranking is not resolved.

**4.4 pCN β by Robbins–Monro (logit scale, `n^{−0.7}`)**: target 0.234 recovers 0.77–0.99 of the
grid-optimal β's ESS and lands within one grid step of the best β on all five problems; target 0.5
converges to roughly half the optimal β and loses 1.03–2.14× everywhere. Optimal β spans 0.8 (d2κ10)
to 0.1 (κ = 100) — a fixed default cannot fit — and pCN acceptance *is* informative (0.97 → 0.11 across
the grid), unlike the HMC step size.

**Limits of the prototypes**: single chain (no MPI/pooling), Gaussian targets only, d ≤ 20, one budget
for the combination, a fixed analytic wrong surrogate rather than an NN, no on-the-fly surrogate
updates, P5 at d ≤ 10. Nothing was run against the library except the shipped proposal class.

---

## 5. Ranked recommendation for the intended 3-stage workflow (preliminary → DAMH + Hamiltonian + NN updates → DAMH frozen)

Ranked by (evidence: literature × prototype) × (knobs removed) ÷ (size of change). Nothing here is
implemented; each item says what would change for a user and what must be decided.

1. **Feed the DAMH adaptation every outer iteration, scoring a pre-rejected one as α = 0** (~2 lines in
   `Algorithm_DAMH`'s call to `adapt()`). **Implemented 2026-09-20 at the author's request** — see
   CHANGELOG "Behaviour changes", `10` §2.21 for the old-vs-new example comparison, and
   `toy_examples/out_damh_adapt_fix_2026-09-20/RESULTS_COMPARISON.md`. Literature: `α₂|₁` → 1 has no optimum (§2 item 1). Prototype:
   17/18 → 0/18 divergences, for every adaptation rule (§4.2). Removes "is `adaptive=True` safe in a
   DAMH stage" (study §4.2). Decision-free. The structural version — adapt the *sub-chain* proposal on
   its own acceptance with parameters frozen within a sub-chain (tinyDA; Sherlock et al. 2021 §6) —
   also never diverged and controls scale best, but tunes to the surrogate posterior (3.6× drift with a
   30 % wrong surrogate); keep it as the design for item 4, where the outer rate is uninformative.
2. **Replace `GaussRandomWalk_adaptive`'s internals.** **Implemented 2026-09-20 at the author's request**
   (combination chosen; `2.38²/d` kept; covariance of post-decision chain states via running Welford
   statistics; relative ridge; target field kept, default 0.234; pooled hand-over by `Allgather` of
   sufficient statistics; `adaptive_corr_limit`/`adaptive_sample_limit` removed) — evidence `10` §2.22,
   efficiency `toy_examples/out_adaptivity_impl_2026-09-20/RESULTS_EFFICIENCY.md`. Primary: shrinkage-regularised covariance of
   accepted states + Robbins–Monro log-scale to 0.234 (§4.1; Haario 2001 + Ledoit–Wolf-style vanishing
   shrinkage + Andrieu–Thoms 2008 Alg. 4). Fallback: RAM (Vihola 2012), the literature's first choice
   and the most robust prototype, slower on shape at d = 20 within 20 000 evaluations. Either way:
   pool samples across ranks instead of averaging covariances (Craiu et al. 2009; Solonen et al. 2012),
   persist the adapted covariance / `S` and a per-period `adaptive_stats.csv` (study §4.3 item 5).
   Removes `adaptive_corr_limit`, `adaptive_sample_limit`, the initial-covariance guess and the
   "how long to adapt" question for the hand-over (frozen acceptance 0.22–0.25 instead of 0.07–0.44).
   **Needs a decision:** combination vs RAM; whether the `2.38²/d` constant is dropped (RM absorbs the
   scale; fixes the d = 2 over-scaling — untested). Changes the sample stream of every adaptive run.
3. **Report efficiency and the DA inputs**: ESS per exact evaluation and per second, surrogate-call
   count, `η`, `α₁`, `α₂|₁` per stage in `summary.csv`; a pre-rejection / no-move guard that names the
   knob (study §4.3 items 4, 8). Sherlock et al. 2021 §2.3 turns `η` and `α̂₂|₁` into the DA-stage scale
   (13× in their example); Christen & Fox's `1/acceptance` ceiling tells the user when DA cannot pay.
   Decision-free, sampler unchanged.
4. **Dual-averaging step size for `Hamiltonian`/`HamiltonianInfinite`** — **step-size part implemented
   2026-09-20** (`Hamiltonian_adaptive`/`HamiltonianInfinite_adaptive`, δ = 0.8, Hoffman–Gelman constants,
   one update per sub-chain step in DAMH, `hamiltonian_step_size=None` carries `ε̄`; `num_steps` kept, `T`
   not exposed, mass unchanged — `10` §2.22) — on the *sub-chain* acceptance,
   parametrised by integration time `T` (`L = ⌈T/ε⌉`, jittered), Hoffman–Gelman constants, freeze at
   `ε̄` for the fixed-surrogate stage; `δ = 0.8` measured at fixed L (0.65 overshoots into the `εL ≈ 2π`
   resonance; at fixed T untested), warm-up ≥ 2000 evaluations, `ε₀` irrelevant over two decades.
   Removes `hamiltonian_step_size` (3× and the silent 100 %-pre-rejection failure). **Needs a decision:**
   expose `T` instead of `num_steps` (a user-facing change); adapt during the NN-update stage while
   `π_a` moves, freeze in stage 3.
5. **Mass = whitened posterior precision `Σ̂⁻¹` for both classes** — from the preliminary stage's pooled
   covariance or, better, from the surrogate's Gauss–Newton Hessian `F̃ + I` (Beskos et al. 2017
   eq. 35; LIS threshold 0.1, Cui et al. 2014) — **only together with item 4**, because a mass change
   rescales the effective step per direction (§3.3(c); prototype §4.3: `Σ⁻¹` 19–29× the RW oracle on
   both target types, `Σ` 0.06–1.2). This **resolves the study's §4.3 item 9(a)** ("evidence is split"):
   it was not split, the GRF `Σ̂` win was a step-length artefact at the shipped ε = 0.05. Removes the
   mass guess. **Needs a decision:** carry the mass automatically when `proposal_sd_or_cov=None`.
6. **pCN β by Robbins–Monro to 0.234** — **implemented 2026-09-20** (`PCN_adaptive`, logit-scale
   recursion, `pcn_beta=None` carries β, the pCN `adaptive=True` force-off removed; per-mode variances not
   done — `10` §2.22) — (§4.4; 0.77–0.99 of the grid optimum, optimal β spans 0.1–0.8),
   with Hu–Yao–Li 2017 per-mode variances (`ρ = 0.99`, cap at the prior eigenvalue) as the extension
   that also learns the shape. Removes `pcn_beta`. Decision-free.
7. **`subchain_max_length`: a rule and a warning, not adaptation** — report the exact-stage acceptance
   and warn outside [0.2, 0.4] (Lykkegaard et al. 2021); optionally randomised `K ~ U{1..J}` (MLDA).
8. **Adaptive Gaussian error model** on the surrogate likelihood (Cui et al. 2019; Lykkegaard et al.
   2021/2023) — raises `α₂|₁`, hence the safe scale and K; the collector has the pairs. Literature only,
   not prototyped; moderate effort; document the bias/noise non-identifiability.
9. **ChEES trajectory-length adaptation** across ranks (Hoffman et al. 2021) — the principled answer to
   `num_steps` for an MPI code with several chains; needs a per-iteration cross-rank reduction; not prototyped.

**Retracted from the study (`15` §4.3) on this evidence:** item 1–2's "ungate + vanishing shrinkage"
as the fix (fails at d20κ100 and has the worst hand-over — the `coef` feedback must go, §4.1);
item 9(a)'s "no blanket rule for the mass" (there is one: `Σ̂⁻¹` + step re-adaptation). **Confirmed:**
item 3 (with the exact fix), item 4, item 5, item 8, item 10 (don't expose `period`; don't adopt a
windowed rate alone).

**Do not adopt, with evidence:** unregularised textbook AM (`ε = 10⁻⁶`); AM with a diminishing gain on
the covariance; freezing an unregularised AM covariance; `M = Σ̂` for any Hamiltonian class; RAM with
`γ = 0.9`; pCN target 0.5; step-size adaptation from the outer DAMH rate; entrywise correlation
clipping; averaging per-rank covariances; a bounded sample window; NUTS/RMHMC/MEADS/ensembles for the
reasons in §1.

---

## 6. What could not be verified

* No source gives a tuning rule or acceptance target for HMC under delayed acceptance; §3.3's
  "sub-chain acceptance, fixed T" is a structural argument. The "T ≈ 1–2 rad" reading of the study's
  sweeps is an interpretation of one-run-per-point data (±40 % scatter), not a measurement.
* The reversibility caveat for adapting inside a K > 1 sub-chain (§3.2) is the manager's reasoning,
  not a citation; it should be checked against Liu's surrogate-transition derivation (not fetched).
* The `(1−α)(X−μ)(X−μ)ᵀ` explanation of the AM collapse is an inference from formulae (§2 item 3).
* ChEES/MEADS formulas come from the TFP/BlackJAX source, not the (non-machine-readable) PDFs; SNAPER's
  venue unconfirmed; Davis et al. 2022's refinement exponent not read (PDF over fetch limit); Cui–Fox–
  O'Sullivan 2011, Conrad et al. 2018, Solonen et al. 2012 (in the DA review; the RW review read its
  full text), Gong & Flegal 2016, Vehtari et al. 2021 verified as records only; Geyer 1992, Nesterov
  2009, Vihola 2011, Gelman–Roberts–Gilks 1996, Roberts–Gelman–Gilks 1997, Liu's surrogate-transition
  paper not fetched.
* Girolami & Calderhead 2011 technical quotes were taken from the withdrawn preprint arXiv:0907.1100.
* Nothing in §1–§3 was run against the library; the numbers quoted about the library are the study's.
* Prototype limits (§4): single chain, Gaussian targets, d ≤ 20, 3 seeds (differences < 1.5× unresolved);
  the shrinkage+RM combination tested at one budget on five problems and over-scales at d = 2; the
  HMC best cells sit on the ESS-estimator ceiling `0.9/(L+1)`; `δ = 0.8` measured at fixed L only; the
  DA track uses a fixed analytic wrong surrogate (`1.3A`), d ≤ 10, and the DA-aware recursion's fixed
  point under intrinsic ≈ 0.6 pre-rejection is not verified; on d10κ100 no adaptive DAMH policy mixed
  well enough to clear the posterior check at 5000 exact evaluations (could not check bias).

---

## 7. Pointers

* Full reviews with per-reference verification status:
  `toy_examples/out_adaptivity_research_2026-09-20/lit_{rw,hmc,da}/REVIEW_*.md`.
* Prototype scripts, `results_proto.csv`, plots and `RESULTS_PROTO.md`:
  `toy_examples/out_adaptivity_research_2026-09-20/proto/`.
* Open-list entry: `10_manual_review_notes.md` §8(b3). Findings table: `06_findings_consolidated.md`
  Tier 7 (note added 2026-09-20 on the retracted/confirmed items).

---

## 8. References

Verification marks: **✔✔** manager re-checked against the primary PDF/source in this session;
**✔** reviewing agent fetched the landing page and/or full text; *record* = bibliographic record only;
*unverified* = neither (cited through another verified paper).

**Adaptive Metropolis, validity, optimal scaling, multi-chain**

- Andrieu, C., Moulines, É. (2006). On the ergodicity properties of some adaptive MCMC algorithms. *Ann. Appl. Probab.* 16(3), 1462–1505. arXiv:math/0610317. ✔
- Andrieu, C., Thoms, J. (2008). A tutorial on adaptive MCMC. *Statistics and Computing* 18(4), 343–373. doi:10.1007/s11222-008-9110-y. ✔ (full text)
- Atchadé, Y. F., Rosenthal, J. S. (2005). On adaptive Markov chain Monte Carlo algorithms. *Bernoulli* 11(5), 815–828. doi:10.3150/bj/1130077595. ✔
- Chen, Y., Keyes, D., Law, K. J. H., Ltaief, H. (2016). Accelerated dimension-independent adaptive Metropolis. *SIAM J. Sci. Comput.* 38(5), S539–S565. arXiv:1506.05741. ✔
- Craiu, R. V., Rosenthal, J. S., Yang, C. (2009). Learn from thy neighbor: parallel-chain and regional adaptive MCMC. *JASA* 104(488), 1454–1466. doi:10.1198/jasa.2009.tm08393. ✔ (authors' full text)
- Gelman, A., Roberts, G. O., Gilks, W. R. (1996). Efficient Metropolis jumping rules. *Bayesian Statistics 5*, 599–608, OUP. *unverified* (via Roberts & Rosenthal 2001)
- Haario, H., Saksman, E., Tamminen, J. (2001). An adaptive Metropolis algorithm. *Bernoulli* 7(2), 223–242. ✔ (landing page; formulae quoted via Roberts & Rosenthal 2009 and Vihola 2012)
- Haario, H., Laine, M., Mira, A., Saksman, E. (2006). DRAM: efficient adaptive MCMC. *Statistics and Computing* 16(4), 339–354. doi:10.1007/s11222-006-9438-0. *record* (title/venue verified, abstract from search text)
- Ledoit, O., Wolf, M. (2004). A well-conditioned estimator for large-dimensional covariance matrices. *J. Multivariate Anal.* 88(2), 365–411. doi:10.1016/S0047-259X(03)00096-4. ✔ (RePEc record)
- Roberts, G. O., Gelman, A., Gilks, W. R. (1997). Weak convergence and optimal scaling of random walk Metropolis algorithms. *Ann. Appl. Probab.* 7, 110–120. *unverified* (via Rosenthal 2010 / Roberts & Rosenthal 2001)
- Roberts, G. O., Rosenthal, J. S. (2001). Optimal scaling for various Metropolis–Hastings algorithms. *Statistical Science* 16(4), 351–367. doi:10.1214/ss/1015346320. ✔ (full text)
- Roberts, G. O., Rosenthal, J. S. (2007). Coupling and ergodicity of adaptive Markov chain Monte Carlo algorithms. *J. Appl. Probab.* 44(2), 458–475. ✔ (authors' full text)
- Roberts, G. O., Rosenthal, J. S. (2009). Examples of adaptive MCMC. *J. Comput. Graph. Statist.* 18(2), 349–367. doi:10.1198/jcgs.2009.06134. ✔ (full preprint, http://probability.ca/jeff/ftpdir/adaptex.pdf)
- Rosenthal, J. S. (2010). Optimal proposal distributions and adaptive MCMC. In *Handbook of MCMC* (Brooks, Gelman, Jones, Meng, eds.). https://probability.ca/jeff/ftpdir/galinart.pdf ✔ (full text)
- Saksman, E., Vihola, M. (2010). On the ergodicity of the adaptive Metropolis algorithm on unbounded domains. *Ann. Appl. Probab.* 20(6), 2178–2203. arXiv:0806.2933. ✔
- Sherlock, C., Roberts, G. O. (2009). Optimal scaling of the random walk Metropolis on elliptically symmetric unimodal targets. *Bernoulli* 15(3), 774–798. arXiv:0909.0856. ✔ (abstract)
- Sherlock, C., Fearnhead, P., Roberts, G. O. (2010). The random walk Metropolis: linking theory and practice through a case study. *Statistical Science* 25(2), 172–190. arXiv:1011.6217. ✔ (full text)
- Solonen, A., Ollinaho, P., Laine, M., Haario, H., Tamminen, J., Järvinen, H. (2012). Efficient MCMC for climate model parameter estimation: parallel adaptive chains and early rejection. *Bayesian Analysis* 7(3), 715–736. doi:10.1214/12-BA724. ✔ (full text, RW review)
- Vihola, M. (2011). Can the adaptive Metropolis algorithm collapse without the covariance lower bound? *Electron. J. Probab.* 16, 45–75. arXiv:0911.0522. *unverified*
- Vihola, M. (2012). Robust adaptive Metropolis algorithm with coerced acceptance rate. *Statistics and Computing* 22(5), 997–1008. doi:10.1007/s11222-011-9269-5; arXiv:1011.4381. **✔✔** (full text; Table 1 and η_n re-checked)
- `adaptMCMC` (R): https://search.r-project.org/CRAN/refmans/adaptMCMC/html/MCMC.html ✔ — RAM, `gamma=2/3`.
- `pymcmcstat` — Miles, P. R. (2019). *JOSS* 4(38), 1417. doi:10.21105/joss.01417; source `samplers/Adaptation.html`. ✔
- PyMC `step_methods/metropolis.py` ✔ — scalar tuning only, `tune_interval=100`.
- Goodman, J., Weare, J. (2010). Ensemble samplers with affine invariance. *CAMCoS* 5(1), 65–80. ✔; Foreman-Mackey, D., Hogg, D. W., Lang, D., Goodman, J. (2013). emcee: the MCMC hammer. *PASP* 125(925), 306. ✔

**HMC, function-space proposals, likelihood-informed methods**

- Beskos, A., Pinski, F. J., Sanz-Serna, J. M., Stuart, A. M. (2011). Hybrid Monte Carlo on Hilbert spaces. *Stoch. Proc. Appl.* 121(10), 2201–2230. doi:10.1016/j.spa.2011.06.003. ✔ (full text)
- Beskos, A., Pillai, N., Roberts, G., Sanz-Serna, J.-M., Stuart, A. (2013). Optimal tuning of the hybrid Monte Carlo algorithm. *Bernoulli* 19(5A), 1501–1534. doi:10.3150/12-BEJ414. ✔
- Beskos, A., Girolami, M., Lan, S., Farrell, P. E., Stuart, A. M. (2017). Geometric MCMC for infinite-dimensional inverse problems. *J. Comput. Phys.* 335, 327–351. arXiv:1606.06351. **✔✔** (eqs 33–35 and "K = C" re-checked in the full PDF)
- Betancourt, M. (2017). A conceptual introduction to Hamiltonian Monte Carlo. arXiv:1701.02434. ✔
- Cao, L., O'Leary-Roseberry, T., Ghattas, O. (2024). Derivative-informed neural operator acceleration of geometric MCMC for infinite-dimensional Bayesian inverse problems. arXiv:2403.08220. ✔ (abstract)
- Cotter, S. L., Roberts, G. O., Stuart, A. M., White, D. (2013). MCMC methods for functions: modifying old algorithms to make them faster. *Statistical Science* 28(3). arXiv:1202.0709. ✔
- Cui, T., Martin, J., Marzouk, Y. M., Solonen, A., Spantini, A. (2014). Likelihood-informed dimension reduction for nonlinear inverse problems. *Inverse Problems* 30, 114015. arXiv:1403.4680. ✔ (full text)
- Cui, T., Law, K. J. H., Marzouk, Y. M. (2016). Dimension-independent likelihood-informed MCMC. *J. Comput. Phys.* 304, 109–137. arXiv:1411.3688. ✔ (full text)
- Girolami, M., Calderhead, B. (2011). Riemann manifold Langevin and Hamiltonian Monte Carlo methods. *JRSS-B* 73(2), 123–214. ✔ (bibliography; quotes from preprint arXiv:0907.1100)
- Hairer, M., Stuart, A. M., Vollmer, S. J. (2014). Spectral gaps for a Metropolis–Hastings algorithm in infinite dimensions. *Ann. Appl. Probab.* 24(6), 2455–2490. arXiv:1112.1392. ✔
- Hoffman, M. D., Gelman, A. (2014). The No-U-Turn Sampler. *JMLR* 15, 1593–1623. arXiv:1111.4246. ✔ (full text)
- Hoffman, M., Radul, A., Sountsov, P. (2021). An adaptive-MCMC scheme for setting trajectory lengths in Hamiltonian Monte Carlo (ChEES). *AISTATS*, PMLR 130, 3907–3915. ✔ (abstract; formulas from TFP/BlackJAX source)
- Hoffman, M. D., Sountsov, P. (2022). Tuning-free generalized Hamiltonian Monte Carlo (MEADS). *AISTATS*, PMLR 151, 7799–7813. ✔ (abstract)
- Hu, Z., Yao, Z., Li, J. (2017). On an adaptive preconditioned Crank–Nicolson MCMC algorithm for infinite dimensional Bayesian inference. *J. Comput. Phys.* 332, 492–503. arXiv:1511.05838. **✔✔** (full PDF: ρ=0.99, J=14, β=1/5, λ_j cap, commutation, β=1/300)
- Kim, K.-T., Villa, U., Parno, M., Marzouk, Y., Ghattas, O., Petra, N. (2023). hIPPYlib-MUQ. *ACM TOMS*. doi:10.1145/3580278; arXiv:2112.00713. ✔ (full text)
- Lan, S., Bui-Thanh, T., Christie, M., Girolami, M. (2016). Emulation of higher-order tensors in manifold Monte Carlo methods for Bayesian inverse problems. *J. Comput. Phys.* 308, 81–101. arXiv:1507.06244. ✔
- Li, L., Holbrook, A., Shahbaba, B., Baldi, P. (2019). Neural network gradient Hamiltonian Monte Carlo. *Comput. Statist.* 34, 281–299. arXiv:1711.05307. ✔ (bibliography)
- Martin, J., Wilcox, L. C., Burstedde, C., Ghattas, O. (2012). A stochastic Newton MCMC method for large-scale statistical inverse problems. *SIAM J. Sci. Comput.* 34(3), A1460–A1487. ✔ (full text)
- Neal, R. M. (2011). MCMC using Hamiltonian dynamics. *Handbook of MCMC*, ch. 5. arXiv:1206.1901. ✔ (full text)
- Nesterov, Y. (2009). Primal-dual subgradient methods for convex problems. *Math. Program.* 120(1), 221–259. *unverified* (via Hoffman & Gelman)
- Petra, N., Martin, J., Stadler, G., Ghattas, O. (2014). A computational framework for infinite-dimensional Bayesian inverse problems, Part II: stochastic Newton MCMC. *SIAM J. Sci. Comput.* 36(4), A1525–A1555. arXiv:1308.6221. ✔
- Pillai, N. S., Stuart, A. M., Thiéry, A. H. (2012). Optimal scaling and diffusion limits for the Langevin algorithm in high dimensions. *Ann. Appl. Probab.* 22(6), 2320–2356. arXiv:1103.0542. ✔
- Rasmussen, C. E. (2003). Gaussian processes to speed up hybrid Monte Carlo for expensive Bayesian integrals. *Bayesian Statistics 7*, OUP. ✔ (existence only)
- Sountsov, P., Hoffman, M. D. (2021). Focusing on difficult directions for learning HMC trajectory lengths (SNAPER). arXiv:2110.11576. ✔ (venue unconfirmed)
- Spantini, A., Solonen, A., Cui, T., Martin, J., Tenorio, L., Marzouk, Y. (2015). Optimal low-rank approximations of Bayesian linear inverse problems. *SIAM J. Sci. Comput.* 37(6), A2451–A2487. arXiv:1407.3463. ✔
- Zhang, C., Shahbaba, B., Zhao, H. (2017). Hamiltonian Monte Carlo acceleration using surrogate functions with random bases. *Statistics and Computing* 27. arXiv:1506.05555. ✔
- Stan reference manual, MCMC sampling; `stan/src/stan/mcmc/covar_adaptation.hpp`. ✔ (regularisation quoted from source)
- PyMC `step_methods/hmc/nuts.py` ✔; BlackJAX `window_adaptation`/`chees_adaptation`/`meads_adaptation` ✔; TensorFlow Probability `gradient_based_trajectory_length_adaptation.py`, `snaper_hmc.py` ✔.
- CUQIpy `cuqi/sampler/_pcn.py`, `_hmc.py`, `_langevin_algorithm.py` (https://github.com/CUQI-DTU/CUQIpy) **✔✔** (`star_acc = 0.44 #TODO: 0.234`, Robbins–Monro with ζ=1/√(n+1)); Alghamdi et al. (2024) *Inverse Problems* 40(4) 045010; Riis et al. arXiv:2305.16949. ✔

**Delayed acceptance, surrogates, multi-fidelity, diagnostics**

- Banterle, M., Grazian, C., Lee, A., Robert, C. P. (2019). Accelerating Metropolis–Hastings algorithms by delayed acceptance. *Foundations of Data Science* 1(2), 103–128. arXiv:1503.00996. ✔ (full text)
- Bérešová, S., Béreš, M., Luber, T., Sysala, S. (2026). Delayed acceptance sampling with Hamiltonian proposal subchains for random field materials inference. arXiv:2606.14743. ✔ (this library's method paper)
- Christen, J. A., Fox, C. (2005). Markov chain Monte Carlo using an approximation. *J. Comput. Graph. Statist.* 14(4), 795–810. doi:10.1198/106186005X76983. ✔ (full preprint)
- Conrad, P. R., Marzouk, Y. M., Pillai, N. S., Smith, A. (2016). Accelerating asymptotically exact MCMC for computationally intensive models via local approximations. *JASA* 111(516), 1591–1607. arXiv:1402.1694. ✔ (full text)
- Conrad, P. R., Davis, A. D., Marzouk, Y. M., Pillai, N. S., Smith, A. (2018). Parallel local approximation MCMC for expensive models. *SIAM/ASA JUQ*. arXiv:1607.02788. *record*
- Cui, T., Fox, C., O'Sullivan, M. J. (2011). Bayesian calibration of a large-scale geothermal reservoir model by a new adaptive delayed acceptance Metropolis Hastings algorithm. *Water Resour. Res.* 47, W10521. doi:10.1029/2010WR010352. *record*
- Cui, T., Fox, C., O'Sullivan, M. J. (2019). A posteriori stochastic correction of reduced models in delayed-acceptance MCMC. *Int. J. Numer. Meth. Eng.* 118(10), 578–605. arXiv:1809.03176. ✔ (arXiv HTML)
- Davis, A. D., Marzouk, Y., Smith, A., Pillai, N. (2022). Rate-optimal refinement strategies for local approximation MCMC. *Statistics and Computing* 32(4), 60. arXiv:2006.00032. ✔ (abstract only)
- Dodwell, T. J., Ketelsen, C., Scheichl, R., Teckentrup, A. L. (2015). A hierarchical multilevel MCMC algorithm with applications to UQ in subsurface flow. *SIAM/ASA JUQ* 3, 1075–1108. arXiv:1303.7343. ✔
- Efendiev, Y., Hou, T., Luo, W. (2006). Preconditioning MCMC simulations using coarse-scale models. *SIAM J. Sci. Comput.* 28(2), 776–803. ✔ (repository page)
- Franks, J., Vihola, M. (2020). Importance sampling correction versus standard averages of reversible MCMCs in terms of the asymptotic variance. *Stoch. Proc. Appl.* 130(10), 6157–6183. arXiv:1706.09873. ✔
- Geyer, C. J. (1992). Practical Markov chain Monte Carlo. *Statistical Science* 7(4), 473–483. *unverified* (Project Euclid blocked; the library already uses the estimator)
- Gong, L., Flegal, J. M. (2016). A practical sequential stopping rule for high-dimensional MCMC. *J. Comput. Graph. Statist.* 25(3), 684–700. arXiv:1403.5536. *record*
- Lykkegaard, M. B., Dodwell, T. J., Moxey, D. (2021). Accelerating uncertainty quantification of groundwater flow modelling using a deep neural network proxy. *CMAME* 383, 113895. arXiv:2007.00400. ✔ (full text)
- Lykkegaard, M. B., Dodwell, T. J., Fox, C., Mingas, G., Scheichl, R. (2023). Multilevel delayed acceptance MCMC. *SIAM/ASA JUQ* 11(1), 1–30. arXiv:2202.03876. ✔ (full text)
- Quiroz, M., Tran, M.-N., Villani, M., Kohn, R. (2018). Speeding up MCMC by delayed acceptance and data subsampling. *J. Comput. Graph. Statist.* 27(1), 12–22. arXiv:1507.06110. ✔
- Sherlock, C., Golightly, A., Henderson, D. A. (2017). Adaptive, delayed-acceptance MCMC for targets with expensive likelihoods. *J. Comput. Graph. Statist.* 26(2). arXiv:1509.00172. ✔ (full text)
- Sherlock, C., Thiery, A. H., Golightly, A. (2021). Efficiency of delayed-acceptance random walk Metropolis algorithms. *Ann. Statist.* 49(5), 2972–2990. arXiv:1506.08155. **✔✔** (full PDF; §2.3 recipe, §6 quotes, 13-fold example re-checked)
- Vats, D., Flegal, J. M., Jones, G. L. (2019). Multivariate output analysis for MCMC. *Biometrika* 106(2), 321–337. doi:10.1093/biomet/asz002. ✔
- Vehtari, A., Gelman, A., Simpson, D., Carpenter, B., Bürkner, P.-C. (2021). Rank-normalization, folding, and localization: an improved R̂. *Bayesian Analysis* 16(2), 667–718. arXiv:1903.08008. *record*
- tinyDA (Lykkegaard, M. B.), https://github.com/mikkelbue/tinyDA — `chain.py`, `proposal.py`. **✔✔** (sub-chain `adapt(accepted=self.accepted_coarse)`; `period=100`, `alpha_star=0.24`, `gamma=1.01`)
- PyMC `step_methods/mlda.py` (v4.1.0) and MLDA introduction notebook. ✔ (`subsampling_rates=5`, adaptive error model, no guidance on the rate)
- Stan reference manual, Posterior analysis. ✔ (no ESS-per-second convention)
