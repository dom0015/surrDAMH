# State of the art and positioning for the paper "Establishing ergodicity of adaptive MCMC with neural-network surrogate models by algorithmic means" (2026-10-23)

Paper draft: `pdf_reports/adaptive_nn_ergodicity_paper/main.tex` (state of 2026-10-23: Sections 2–7 written,
intro/experiments/conclusion TODO). This note answers two questions: (1) what is the state of the art and
where does the draft overlap with existing work; (2) how to frame the paper, what the contribution is, and
what is not new.

Method: five parallel web searches (adaptive-surrogate MCMC; safe-component adaptive-MCMC theory;
surrogate-gradient and adaptive HMC; NN surrogates in Bayesian inversion incl. the authors' own record;
limit theorems and ESS estimation), each reading arXiv/publisher full texts where available. Every item
below is marked **V** (seen on a web page, full text or abstract, this session) or **U** (from memory or a
search snippet only; check before citing). Theorem numbers of Laitinen & Vihola refer to arXiv:2408.14903v1;
the EJP version (vol. 31, 1–22, doi 10.1214/26-EJP1509) is confirmed to exist but its numbering was not seen.
Nothing in the repository was changed except this note, its README row and a PLAN.md pointer.

---

## 1. Executive summary

1. **The mechanism of the main theorem (Theorem 5.1) is in print: Sherlock, Golightly & Henderson (JCGS 2017).**
   Their adaptive delayed-acceptance sampler mixes, at a fixed rate β, a *fixed exact MH kernel* with the
   adaptive DA kernel; Assumption 1 is a whole-space minorisation of the fixed kernel; the surrogate (a k-NN
   / KD-tree average of past exact evaluations) is updated with probability p_i → 0; ergodicity follows from
   Roberts & Rosenthal (2007) Thm 5. Their Appendix D.1 says verbatim that on a non-compact space Assumption 1
   holds "if the proposal density q(θ*|θ) = q(θ*) is an independence proposal which satisfies q(θ*) > δπ(θ*)".
   The draft's prior-proposal step is the instance q = p, δ = Z e^{−B₊}. The draft currently cites SGH17 only
   for randomised installation (Rem. 6.3(ii)); it must cite it for the mixture-and-minorisation argument too,
   otherwise a referee will call Section 5 unacknowledged prior art.
2. **The same "independence component with uniformly bounded π/q ⇒ SUE on ℝ^d" argument appears in**
   Pompe, Holmes & Łatuszyński (Ann. Stat. 2020, Thm 4.1, condition (4.4)), Andrieu & Moulines (AAP 2006,
   §7.4 Remark 7, defensive fixed component ζ with ζ/π ≥ e), Holden, Hauge & Holden (AAP 2009, who say the
   Doeblin condition "will usually be satisfied in a Bayesian approach using a prior with heavy tails"),
   Giordani & Kohn (JCGS 2010, defensive component g₀ with π/g₀ ≤ K), Brofos et al. (AISTATS 2022, on compact
   spaces). Andrieu & Thoms (2008, p. 350) state the idea as folklore and object that it "poses the problem of
   the practical choice of the non-adaptive component". **The draft's genuine point is the answer to that
   objection**: for a Bayesian posterior with bounded likelihood the *prior* is a canonical, model-free safe
   component, with an explicit constant and no assumption on G, the surrogate, the sub-chain or the tails.
3. **The Roberts & Rosenthal (2009) analogy in the draft is loose and should be corrected**: their fixed
   component β N(x, 0.1²I/d) is a *proposal* component that secures *containment via drift* (their §8,
   V = π^{−1/2}), not SUE; random-walk kernels are never uniformly ergodic on ℝ^d. Cite SGH17 / PHL20 / AM06 /
   HHH09 / GK10 as the correct analogues.
4. **Every published proof for a surrogate-adaptive sampler obtains containment from compactness, tail
   envelopes or drift, or skips it.** Conrad et al. (JASA 2016) need a compact Θ (Thm 3.4) or a "Gaussian
   envelope" the authors call "quite strong" (Thm 3.3); Conrad et al. (JUQ 2018) compact Θ; Cui, Fox &
   O'Sullivan (WRR 2011 / IJNME 2019) compact X and continuity/Lipschitz; Parno & Marzouk (JUQ 2018)
   super-exponentially light targets + compact map coefficients; Wang, Mohamed & de Freitas (ICML 2013,
   adaptive HMC) compact support and a finite parameter set; Zhang, Shahbaba & Zhao (Stat. Comput. 2017,
   surrogate HMC with a random-basis network updated w.p. a_t → 0) claim ergodicity "by Theorem 5 of
   [RR2007]" *without checking SUE/containment*; Lykkegaard et al. (MLDA, JUQ 2023) and Lykkegaard, Dodwell &
   Moxey (CMAME 2021) assert it in a remark ("de facto ergodic"). **No paper was found that proves
   ergodicity of a delayed-acceptance sampler whose global NN surrogate is retrained without limit, on an
   unbounded space, with no condition on the forward map.** That is the gap the draft fills, and it should be
   stated exactly in these words.
5. **Randomised installation (Section 6.1) is a standard device**: RR2007 Cor. 7/16 ("adapt with probability
   p(n) → 0"), SGH17 (p_i → 0), Zhang et al. 2017 (a_t → 0, with HMC and a network), Strathmann et al. 2015
   (KMC), Wang et al. 2013, Air MCMC §3.4, Laitinen & Vihola Example 22 (activation probabilities
   η_k ≤ c log^{−(1+ε)} k). Present it as such; the contribution is that it is combined with SUE *rigorously*
   and applied jointly to network versions and HMC parameters.
6. **Averaged installation (Section 6.2) is the most novel technical piece, but not new in principle**:
   Giordani & Kohn (2010, eq. 9) already average adaptive proposal parameters; Cui, Fox & O'Sullivan (2019,
   Thm 3) prove DA via Lipschitz dependence of the DA kernel on the error-model index; Wang, Chen, Kanagawa &
   Oates (RLMH, arXiv 2024, Lemma 4) use summable learning rates + gradient clipping (parameters stay in a
   ball) + Lipschitz-in-parameters for NN-parametrised *proposals*; AM06 Prop. 19–20 and LV Lemma 16 are the
   general continuity route. New: the TV-Lipschitz bound of the DAMH kernel in the surrogate log-likelihood
   (Lemma 6.5), the perceptron bound in a weight box with input clipping, and Polyak averaging of *surrogate
   weights*. No precedent for Polyak-averaged surrogate weights was found.
7. **Limit theorems (Section 6.3)**: ergodicity and WLLN are RR2007 Thm 5/23; SLLN under Σλ_n/n < ∞ is LV
   Thm 20 directly (or Thm 9 + Lemma 11), also FMP 2011 A4 with V ≡ 1. RR2007 Example 24 is a
   randomised-adaptation counterexample (adaptive rule used w.p. 1/k per block): DA holds, SLLN fails, so the
   summability condition in Thm 6.11(iv) is necessary in kind and should be motivated by that example. "No
   CLT" is a choice consistent with LV Lemma 18, FMPV 2014 A4, Air Thm 1(iv) and Atchadé 2011: a CLT needs the
   installed kernels to converge, which nothing guarantees for NN weights. The coupling bound (Prop. 6.10) is
   the RR2007 proof with constants, of the same flavour as Holden et al. 2009 Thm 2 and Brofos et al. Thm 4.1.
8. **Wang (Bernoulli 2022) Thm 2**: the worst-case TV distance of an independence sampler is *exactly*
   (1 − 1/w*)^n with w* = sup π/q. With q = p: 1 − Z e^{−sup ℓ}. So δ/w = Z e^{−B₊} is sharp when B₊ = sup ℓ,
   which quantifies the draft's remark that the bound is qualitative. Cotter et al. (2013, §4.4) report the
   prior independence sampler (pCN with δ = 2) accepting 12 of 10⁶ proposals: good evidence for Rem. 5.3(ii).
9. **Section 7 (efficiency)**: the lazy-chain formula τ_w = (τ + w)/(1 − w) is Łatuszyński & Roberts (MCAP
   2013) Corollary 1 verbatim; for the mixture wQ + (1 − w)P with Q the prior step the same expression is an
   *upper bound* by Peskun (Tierney 1998), since Q's off-diagonal part dominates that of the identity. The
   intro's open question ("is it valid to estimate autocorrelation while the chain adapts?") has one rigorous
   answer in the literature, **Atchadé (Ann. Stat. 2011)**: lag-window estimators on an adaptive run converge
   to a *random* limit and are consistent for the asymptotic variance only if the adapted parameter converges
   to a deterministic limit, with increments O(n^{−α}), α > 1/2 (randomised installation violates this: O(1)
   increments). The draft's recommendation (frozen tail or replicate runs) is what the literature supports.
   Geyer's initial-sequence estimator is "asymptotically conservative" (lim inf ≥ σ² a.s., Thm 3.2), not a
   "consistent overestimate"; reword.
10. **The open item on HMC (Section 6.4(v)) is genuinely open**: no TV- or Wasserstein-Lipschitz bound of the
    Metropolised HMC kernel in step size, mass matrix or gradient field, uniform in the state, was found.
    Johndrow & Mattingly (2017) note that such kernels are typically *not* TV-close but W₁-close; Brown &
    Rosenthal (JAP 2025) give a Wasserstein analogue of RR2007 that could be the route. Atchadé's truncated
    drift (MCAP 2006) is the Langevin precedent for a bounded-force device. Wang et al. 2013 obtain SUE for
    adaptive HMC only on a compact space via the MALA component. The draft's observation that the prior step
    makes HMC irreducibility irrelevant (cf. Durmus, Moulines & Saksman 2020: "a mixture of Markov kernels is
    irreducible as soon as one component of the mixture is irreducible"; Livingstone et al. 2019 use
    P[L = 1] > 0 for the same purpose) is correct and appears not to have been made for NN-surrogate HMC.
11. **The authors' own record gives a clean narrative.** Bérešová, PANM 2022: "the algorithm becomes an adaptive
    MCMC method … In situations when it is not possible to prove the ergodicity of the DAMH-SMU algorithm, the
    validity of the sampling process can be ensured simply by stopping the adaptations at some point."
    Bérešová, Béreš, Luber & Sysala, arXiv:2606.14743 (June 2026): "stepwise exactness is not enough to
    guarantee ergodicity", "the standard diminishing-adaptation and containment conditions are not satisfied"
    for online-trained networks, hence the network "is updated only during a burn-in phase". The new paper
    removes that freeze with a proof. Say so in the introduction.
12. **Closest methodological neighbour**: Deveney, Müller & Shardlow (SIAM/ASA JUQ 2023): deep surrogate,
    delayed acceptance, HMC first stage (K = 1), surrogate refined during warm-up then frozen, Hellinger bound
    for the surrogate posterior, no adaptive-chain theory. Cui, Marzouk & Willcox (IJNME 2015): L-step
    sub-chain on a reduced-model posterior + exact correction, basis enriched during sampling, justified by
    *finite* adaptation, "always converges to the full posterior distribution, regardless of the accuracy of
    the reduced-order model" — the same message as the draft's Rem. 6.12(ii), to be cited for it.

---

## 2. State of the art, by strand

### 2.1 Exact (delayed-acceptance) samplers whose approximation is adapted during the run

| Work | Surrogate | Updated during run | How DA is obtained | How containment/SUE is obtained | Assumptions on G / posterior | Limit theorems | Status |
|---|---|---|---|---|---|---|---|
| Christen & Fox, JCGS 14(4) 2005, 795–810 | any fixed approximation | no | – | – | fixed approximation | invariance/ergodicity of DA | V (citation) |
| Efendiev, Hou & Luo, SISC 28(2) 2006, 776–803, doi 10.1137/050628568 | coarse-scale model | no | – | – | fixed | convergence conditions | V (abstract) |
| Cui, Fox & O'Sullivan, WRR 47 (2011) W10521, doi 10.1029/2010WR010352; theory in Univ. Auckland Report 687 | reduced model + adaptive Gaussian error model, AM proposal | yes (recursive moments) | changes O(1/n) | **compact X**, kernels ergodic, q uniformly bounded, joint continuity, compact index sets (RR2007 Thm 1 + Cor 5) | compact, continuous | TV | V (report full text) |
| Cui, Marzouk & Willcox, IJNME 102(5) 2015, 966–990, doi 10.1002/nme.4748, arXiv:1403.4290 | reduced basis (POD), L-step sub-chain, exact correction | yes, finitely many (basis cap + stopping rule) | **finite adaptation** | ergodicity of each fixed kernel | – | TV | V |
| **Sherlock, Golightly & Henderson, JCGS 26(2) 2017, 434–444, doi 10.1080/10618600.2016.1231064, arXiv:1509.00172** | k-NN / KD-tree average of exact evaluations | yes, points added w.p. p_i → 0 | randomised update | **mixture β·(fixed exact MH) + (1−β)·DA; Assumption 1 q·α_MH ≥ δν; App. D.1: independence proposal with q ≥ δπ works on non-compact spaces** | Assumption 1 only | TV; Thm 2 β ∝ stage-1 acceptance; Thm 3 adapt every other step on a compact rectangle | V (full text) |
| Cui, Fox & O'Sullivan, IJNME 118(10) 2019, 578–605, doi 10.1002/nme.6028, arXiv:1809.03176 | reduced model + stochastic correction | yes | ‖Ξ_{n+1} − Ξ_n‖ → 0 via **Lipschitz dependence of the kernel on the index** (Thm 3) | **compact X, V, E**, Lipschitz target and q (Thm 2) | Lipschitz | TV | V (full text) |
| Zhang, Shahbaba & Zhao, Stat. Comput. 27(6) 2017, 1473–1490, doi 10.1007/s11222-016-9699-1, arXiv:1506.05555 | random-basis one-hidden-layer network, drives HMC; exact MH with true Hamiltonian | yes (ARNS-HMC), w.p. a_t → 0, Σa_t = ∞ | randomised update | **not checked** ("By Theorem 5 of [RR2007], the resulting algorithm is ergodic") | none stated | claimed only | V (full text) |
| Lykkegaard, Dodwell & Moxey, CMAME 383 (2021) 113895 (vol./no. U), arXiv:2007.00400 | offline DNN + adaptive error model | error model only | "naturally" (heuristic) | "does not in theory guarantee ergodicity … de facto ergodic process" | – | none | V (text) |
| Lykkegaard, Dodwell, Fox, Mingas & Scheichl (MLDA), SIAM/ASA JUQ 11(1) 2023, 1–30, doi 10.1137/22M1476770, arXiv:2202.03876 | coarse models + adaptive error model, sub-chains of arbitrary length | error model | remark: recursive moments converge | remark: "diminishing adaptation and bounded convergence which is sufficient"; NN bias models fine "as long as they are either constructed a priori or exhibit diminishing adaptation and bounded convergence" | – | detailed balance proved (R1 citation for K > 1) | V |
| Benavoli, Wyse & White, arXiv:2109.13891 (2021) | GP learned on the fly, two-stage MH/MALA | yes, continuously | GP variance decrease (Prop. 2.3, informal) | "generally satisfied under some regularity conditions" | – | none | V |
| Deveney, Müller & Shardlow, SIAM/ASA JUQ 11(3) 2023, 970–995, doi 10.1137/22M1513113, arXiv:2204.02272 | PINN surrogate; DA with HMC first stage, K = 1 | warm-up only, then frozen; ε, M frozen after warm-up | freeze | – | – | Hellinger convergence of the *surrogate posterior* | V (full text) |
| Bérešová, Béreš, Luber & Sysala, arXiv:2606.14743 (June 2026) | MLP shared via collector; Hamiltonian sub-chains | burn-in only | freeze ("DA and containment are not satisfied") | – | – | π-invariance per fixed surrogate | V (full text) |
| Bon, Lee & Drovandi, Stat. Comput. 31:62 (2021), arXiv:2009.03699 | surrogate likelihood inside SMC with DA | between SMC iterations | SMC sidesteps adaptive-MCMC theory | – | – | – | V (abstract) |

### 2.2 Asymptotically exact surrogate MCMC without an exact correction step

- **Conrad, Marzouk, Pillai & Smith, JASA 111(516) 2016, 1591–1607, doi 10.1080/01621459.2015.1096787, arXiv:1402.1694** (V, full text). Accept/reject uses the local polynomial/GP surrogate; exactness is asymptotic through infinite refinement. Refinement: random with probability β_t (Σβ_t = ∞ needed) + leave-one-out cross-validation indicator ≥ γ_t. Theory is for a modified algorithm (constant β, quadratic interpolation, RW proposal): Thm 3.4 compact Θ with target and proposal densities bounded away from 0 and ∞; Thm 3.3 non-compact under a "Gaussian envelope" tail assumption (called "quite strong" by the authors) + drift V = 1/p_∞. Coupling proof. TV only; Remark 3.5: LLN/CLT "believed" to hold.
- **Conrad, Davis, Marzouk, Pillai & Smith, SIAM/ASA JUQ 6(1) 2018, 339–373, doi 10.1137/16M1084080, arXiv:1607.02788** (V). Several chains share one approximation (the draft's collector pattern). Thm 1: compact Θ, C^∞ densities bounded below, constant β. Thm 3: mMALA with surrogate gradients on [0,1]^d. Remark 3.2: cannot prove parallel chains never hurt ("mal-adaption").
- **Davis, Marzouk, Smith & Pillai, Stat. Comput. 32:60 (2022), doi 10.1007/s11222-022-10123-0, arXiv:2006.00032** (V). Deterministic refinement thresholds decaying with t; drift for the exact kernel, geometric ergodicity on sublevel sets, smoothness; MSE ≤ C log³T / T. Quantitative where the draft is qualitative.
- **Parno & Marzouk, SIAM/ASA JUQ 6(2) 2018, 645–682, arXiv:1412.5492** (V). Exact MH with an adaptive transport-map *proposal*; Thm 5.7 needs a super-exponentially light continuous target and map coefficients in a compact set with bi-Lipschitz maps; containment via SSAGE (Bai, Roberts & Rosenthal Thm 3). The authors note the compactness is stronger than what is enforced in practice.

### 2.3 Inexact surrogate posteriors and their error analysis (the alternative the draft avoids)

Marzouk & Xiu, CiCP 6(4) 2009, 826–847 (V); Stuart & Teckentrup, Math. Comp. 87(310) 2018, 721–753, arXiv:1603.02004 (V); Lie, Sullivan & Teckentrup, SIAM/ASA JUQ 6(4) 2018, 1600–1629, arXiv:1712.05717 (V); Sprungk, Inverse Problems 2020, arXiv:1906.07120 (V: local Lipschitz stability of the posterior in TV/Hellinger/Wasserstein/KL under bounded likelihood, with constants that **grow as the posterior concentrates**); Latz, SIAM/ASA JUQ 8(1) 2020, 451–482 (V); Lu et al., JCP 2023, arXiv:2210.03008 (V from snippet: KL bound for neural-operator posteriors, ill-conditioned when the prior is uninformative, data high-dimensional or noise small); Kaipio & Somersalo, JCAM 198(2) 2007, 493–504 (V, citation). Inexact NN pipelines: Yan & Zhou, CiCP 28(5) 2020, 2180–2205, arXiv:1911.08926 (V; offline DNN + online shallow corrections; samples from the approximate posterior; **no convergence theorem**, contrary to a search-engine summary); Yan & Zhou arXiv:1807.00618 (JCP 2019, U) multi-fidelity PC with KL/Hellinger bounds; Wang & Li, Neural Comput. 30(11) 2018 ("a rigorous convergence analysis of the algorithm is still lacking", V); Li & Marzouk, SISC 36(3) 2014 (V); Cleary et al. (CES), JCP 424 (2021) 109716 (V); Drovandi, Moores & Boys, CSDA 118 (2018) 1–17 (V; GP-GIMH shows bias); Zhou & Tartakovsky, SERRA 2021, arXiv:2003.02322 (V); Cao, O'Leary-Roseberry & Ghattas, JMLR 26 (2025), arXiv:2403.08220 (V; offline derivative-informed neural operator in DA geometric MCMC, 3–9× over geometric MCMC); Zacchei, Conti, Frangi & Manzoni, arXiv:2512.16430 (Dec 2025; offline NNs in MLDA) (V); Jimenez-Beltran, Teckentrup, Vergari & Zygalakis, arXiv:2606.20417 (June 2026; proposes a short DA run as a faithfulness check of a surrogate posterior) (V); Lykkegaard, Fox, Higdon, Reese & Moulton, arXiv:2405.00397, Handbook of MCMC 2nd ed. (V; general chapter on approximate models in MCMC). Use this strand for one paragraph: inexact bounds need accuracy and stability constants that deteriorate exactly when data are informative; the exact route needs neither.

### 2.4 Safe/fixed components and independence samplers in adaptive-MCMC theory

- **Roberts & Rosenthal, JAP 44 (2007) 458–475** (V, preprint probability.ca/jeff/ftpdir/adapt.pdf): Thm 5 (SUE + DA), Thm 13 (containment + DA), Thm 23 (WLLN), Cor. 7/16 (adapt w.p. p(n) → 0), **Example 24** (SLLN fails under the conditions of Thm 5: blocks [2^{(k−1)²}, 2^{k²}], adaptive rule used w.p. 1/k per block), §10 (parallel-chain adaptation). No fixed-kernel mixture lemma.
- **Roberts & Rosenthal, JCGS 18(2) 2009, 349–367, doi 10.1198/jcgs.2009.06134** (V): eq. (3) Q_n = (1−β)N(x, 2.38²Σ_n/d) + βN(x, 0.1²I_d/d), β = 0.05; "Restricting β > 0 … ensures that (2) [containment] is satisfied, at least for … log-concave outside some arbitrary bounded region (see Section 8)"; §8 obtains containment via simultaneous geometric drift V = π^{−1/2}. **Containment via drift, not SUE.**
- **Andrieu & Thoms, Stat. Comput. 18(4) 2008, 343–373** (V), p. 350: "Another possibility consists of considering mixtures of adaptive and non-adaptive MCMC proposal distributions, the non adaptive components ensuring stability e.g. (Roberts and Rosenthal 2007): again while this type of strategy generally ensures that the theory works, it poses the problem of the practical choice of the non-adaptive component, and might not always result in efficient strategies."
- **Pompe, Holmes & Łatuszyński, Ann. Stat. 48(5) 2020, 2930–2952, doi 10.1214/19-AOS1916, arXiv:1812.02609** (V): JAMS; jump moves are independence-type proposals R_{γ,J,i}; condition (4.4) sup_x sup_γ π(x)/R(x) < ∞; Thm 4.1 ergodic; proof: P̃_γ ≥ a c₁ π̃_γ ⇒ (1 − a c₁)^n uniformly, i.e. SUE on ℝ^d; drift/containment fallback (Lemma 3.5). Identical argument to Thm 5.1, with R = prior as a special case.
- **Andrieu & Moulines, AAP 16(3) 2006, 1462–1505, doi 10.1214/105051606000000286** (V): §7.1 IMH geometric iff q/π ≥ ε, whole space (1, ε)-small; §7.4 adaptive IMH with q_θ = (1−ι) q̃_ξ(θ) + ι ζ, ζ fixed; Remark 7: this "automatically ensure[s]" the uniform bound with ε = eι; Prop. 19–20: the fixed component also controls ‖P_q − P_{q′}‖ (continuity, i.e. DA).
- **Holden, Hauge & Holden, AAP 19(1) 2009, 395–413, doi 10.1214/08-AAP545** (V): strong Doeblin q_i ≥ a_i π; Thm 2 ‖p_i − π‖ ≤ 2∏(1 − a_j); "It is possible to obtain a_i > 0 by making the tails in the proposal function sufficiently heavy. This will usually be satisfied in a Bayesian approach using a prior with heavy tails. Alternatively, we may replace q_i by (1 − ε)q_i + εg".
- **Giordani & Kohn, JCGS 19(2) 2010, arXiv:0801.1864** (V): q_n = ω₁g₀ + (1−ω₁)ḡ_n, π/g₀ ≤ K ("similar to the defensive mixtures approach studied by Hesterberg"); Thm 1 TV, Thm 2 SLLN; **eq. (9) averages the adaptive parameters** λ̄_n = (1−β_n)λ*_n + β_n λ̄_{n−1}.
- **Brofos, Gabrié, Brubaker & Lederman, AISTATS 2022, PMLR 151:5949–5986, arXiv:2110.13216** (V): adaptive IMH with normalizing flows; Thm 4.1 product bound; Prop. I.1–I.2 mixture αK_θ + (1−α)K′: DA transfers, minorisation ⇒ containment; App. N: defensive mixture with a fixed density ⇒ SUE on **compact** spaces ("the fact that there was a global limit to how bad any proposal distribution could be allowed us to invoke simultaneous uniform ergodicity").
- Johnson, Jones & Neath, Stat. Sci. 28(3) 2013, 360–375 (V): "if one of the component samplers is uniformly ergodic, then P_mix will be uniformly ergodic" (non-adaptive). Gåsemyr, Scand. J. Stat. 30(1) 2003 (V, abstract). Craiu, Rosenthal & Yang, JASA 104(488) 2009 (V; INCA parallel chains "independently conditional on {γ_m}", product kernel; compact support assumption). Craiu, Gray, Łatuszyński, Madras, Roberts & Rosenthal, AAP 25(6) 2015, arXiv:1403.3950 (V; fixed kernel *outside* a compact set, dual construction). Gilks, Roberts & Sahu, JASA 93 (1998) 1045–1054 (V, citation; adaptation at regeneration times — with a prior step, regeneration times are identifiable).
- **Samsonov, Lagutin, Gabrié, Durmus, Naumov & Moulines, NeurIPS 2022, arXiv:2111.02702** (V): i-SIR (N-proposal independence step) is uniformly geometrically ergodic if ‖w‖_∞ < ∞ (Andrieu, Lee & Vihola, Bernoulli 2018, U), with rate 1 − (N−1)/(2L + N − 2), L = ‖w‖_∞/λ(w). With the prior as proposal, w = likelihood and ‖w‖_∞ < ∞ is ℓ ≤ B₊: a multi-proposal safe step with N prior draws per step would improve δ roughly N-fold at N exact evaluations. Worth a remark.
- **Rate of the prior independence sampler**: Mengersen & Tweedie, Ann. Stat. 24(1) 1996 Thm 2.1 (U: as restated in Wang 2022); **Wang, Bernoulli 28(3) 2022, 2012–2033, doi 10.3150/21-BEJ1409, arXiv:2008.02455** (V): Thm 2 worst-case TV = (1 − 1/w*)^n *exactly*, w* = sup π/q (credits Smith & Tierney 1996, Tierney 1994 Cor. 4, Liu 1996). Cotter, Roberts, Stuart & White, Stat. Sci. 28(3) 2013 §4.4 (V): pCN with δ = 2 is the prior independence sampler, "only accepted 12 proposals out of 10⁶". Hairer, Stuart & Vollmer, AAP 24(6) 2014 (V): same remark, Meyn–Tweedie theory applies. Brown & Jones, JAP 61 (2024) 33–54 (V) is about Wasserstein rates of independence samplers; the "constant L(θ*)/Z with the prior as proposal" attribution (a search snippet pointed to a WIREs 2024 article) is **U**.
- **Laitinen & Vihola, EJP 31 (2026) 1–22, doi 10.1214/26-EJP1509, arXiv:2408.14903v1** (V, arXiv numbering): (A3) SUE; (A5) p-waning; Thm 9 (i) WLLN under weak 1-waning, (ii) SLLN under strong 1-waning, (iii) CLT under ½-waning + (A4) (deterministic limit of averaged conditional variances); Lemma 10 DA ⇒ weak 1-waning; Lemma 11 ΣE[D_k]/k^p < ∞ ⇒ strong p-waning; Lemma 16 SA updates with TV-Lipschitz kernels; Remark 15 projection onto a feasible set; Lemma 18 CLT if S_n → s_∞ a.s.; §6 Lemma 19 / **Thm 20** ((A10) ΣP(S_k ≠ S_{k−1})/k < ∞ ⇒ SLLN, no continuity needed), Example 21 (gaps ≥ c log^{1+ε} j), **Example 22** (random activation η_k ≤ c log^{−(1+ε)} k); §7 Prop. 23 WLLN for exogenous sequences under uniform Dobrushin (A11); §8.2 SUE "typically" needs a restricted parameter set. **No fixed-component remark in LV.**
- Chimisov, Łatuszyński & Roberts, "Air MCMC", arXiv:1801.09309 (V; **still a preprint**): Thm 1 (lags k^β): WLLN, SLLN for β > ½, CLT for β > 1 if γ_n → γ_∞; §3.4 Thm 4, coin w.p. p_i → 0 "can decay arbitrarily slowly", Example 1/Prop. 4 (SLLN without convergence in law). Hofstadler, Łatuszyński, Roberts & Rudolf, arXiv:2402.12122, SPA 2026 doi 10.1016/j.spa.2026.104905 (V, abstract): a.s. rates for increasingly rare adaptation under Wasserstein contraction, no DA needed. Brown & Rosenthal, JAP 62 (2025) 1318–1339, doi 10.1017/jpr.2025.4, arXiv:2406.00820 (V, abstract): Wasserstein analogue of RR2007 with a WLLN for Lipschitz functions.
- Fort, Moulines & Priouret, Ann. Stat. 39(6) 2011, arXiv:1203.3036 (V): Thm 2.7/Cor 2.8 SLLN under drift + A4, which with uniform constants and V ≡ 1 reads ΣD_k/k < ∞ a.s. Fort, Moulines, Priouret & Vandekerkhove, Bernoulli 20(2) 2014, 457–485 (V): CLT under A3(a) + A4 deterministic limit. Atchadé & Fort, Bernoulli 16(1) 2010 and 18(3) 2012 (V, abstracts). Saksman & Vihola, AAP 20(6) 2010; Vihola, SPA 121 (2011); Vihola RAM, Stat. Comput. 22(5) 2012 (V, abstracts; drift routes, super-exponential tails). Atchadé, AAP 20(3) 2010 "cautionary tale" (V, abstract): applies to kernels with π_θ ≠ π, not to π-invariant DAMH.

### 2.5 Surrogate-gradient HMC and adaptive HMC

- Rasmussen, Bayesian Statistics 7 (2003) 651–659 (V, full text): GP-driven trajectory + one exact Metropolis step; GP frozen in the sampling phase. The root of "surrogate HMC + exact correction".
- Zhang, Shahbaba & Zhao 2017 (above). Strathmann, Sejdinovic, Livingstone, Szabó & Gretton, NeurIPS 2015 (KMC), arXiv:1506.02564 (V): adaptation w.p. a_t → 0 citing RR2007 Thm 1; Prop. 3 geometric ergodicity only with adaptation stopped, d = 1, bounded surrogate gradient vanishing at infinity. **Lan, Bui-Thanh, Christie & Girolami, JCP 2016, arXiv:1507.06244** (V): GP emulator for manifold HMC; online refinement only at *regeneration times*, obtained by "systematically alternat[ing] between GPeGMC and the independence sampler" — a precedent for mixing a state-independent exact step into surrogate HMC, used for regeneration rather than minorisation. Li, Holbrook, Shahbaba & Baldi, Comput. Stat. 34(1) 2019, 281–299 (V): NN approximates the gradient field directly (non-conservative), leapfrog still volume-preserving and reversible; frozen after burn-in. Levy, Hoffman & Sohl-Dickstein (L2HMC, ICLR 2018), Hoffman et al. (NeuTra, arXiv:1903.03704): learned kernels/transports trained before sampling (V, abstracts). Deveney et al. 2023 (above).
- **Wang, Mohamed & de Freitas, ICML 2013, PMLR 28:1462–1470, arXiv:1302.6182** (V): adaptive HMC by Bayesian optimisation over (ε, L), adaptation probability p_i → 0; Prop. 3 ergodicity assumes **compact support**, π bounded above and below, **finite parameter set**; SUE from the MALA component P_γ ≥ (1/L)Q_MALA; "Without restricting the state space to be compact, it is unlikely that HMC is uniformly ergodic." Same template as Thm 5.1 + Section 6.1 with both restrictions removed by the prior step.
- Practice: Hoffman & Gelman (NUTS, JMLR 15, 2014): "adapt any tunable MCMC parameters during the burn-in phase, and freeze"; ChEES (Hoffman, Radul & Sountsov, AISTATS 2021): "only adapt trajectory length during the initial warmup phase to avoid bias"; SNAPER (arXiv:2110.11576); Stan's windowed warm-up (V). Exact self-tuning alternatives: GIST (Bou-Rabee, Carpenter & Marsden, Stat. Surveys 20, 2026, arXiv:2404.15253; reversibility only), autoMALA (AISTATS 2024; irreducibility left open), dynamic HMC/NUTS theory (Durmus, Gruffaz, Kailas, Saksman & Vihola, AAP 36(3) 2026, arXiv:2307.03460). Marshall & Roberts, Stat. Comput. 22(5) 2012 (V, working paper CRiSM 09-04): adaptive MALA, Γ adapted at times with gaps → ∞, step increments → 0, uniform drift. **Atchadé, MCAP 8(2) 2006, 235–254** (V, abstract): adaptive MALA with *truncated* (bounded) drift — the device that makes a Langevin kernel's dependence on its parameters controllable uniformly in x; natural HMC analogue: clip ∇ℓ̃_θ.
- HMC ergodicity: **Durmus, Moulines & Saksman, Ann. Stat. 48(6) 2020, 3545–3564, arXiv:1705.00166** (V): irreducibility needs conditions on U (degree theory); "a mixture of Markov kernels is irreducible as soon as one component of the mixture is irreducible"; reviews Cancès, Legoll & Stoltz (M2AN 2007, bounded U and ∇U). **Livingstone, Betancourt, Byrne & Girolami, Bernoulli 25(4A) 2019, arXiv:1601.08057** (V): geometric ergodicity fails for light and heavy tails; irreducibility via P[L = 1] > 0; counterexample Gaussian target, L = 2, ε = √2 ⇒ proposal = δ_x. Bou-Rabee & Sanz-Serna (AAP 2017), Bou-Rabee, Eberle & Zimmer (AAP 2020), Chen, Dwivedi, Wainwright & Yu (JMLR 2020), Mangoubi & Smith (AAP 2021) (V, abstracts).
- Perturbation theory (for the open item): Rudolf & Schweizer, Bernoulli 24(4A) 2018; **Johndrow & Mattingly, arXiv:1711.05382** ("singular with respect to the original kernel, so they are not close in total variation" but close in W₁); Medina-Aguayo, Rudolf & Schweizer, SPA 130(4) 2020; Pillai & Smith, arXiv:1405.0182; Negrea & Rosenthal, Adv. Appl. Probab. (arXiv:1702.07441) (V, abstracts). All take one-step closeness as a hypothesis; none supplies a Lipschitz bound of the HMC kernel in ε, M or the gradient field.

### 2.6 Limit theorems, DA efficiency and ESS estimation (Section 7 material)

- Lazy chain: **Łatuszyński & Roberts, MCAP 15(1) 2013, 237–247, doi 10.1007/s11009-011-9237-8, arXiv:1102.2171, Corollary 1**: for P reversible, ergodic with a CLT for f and P_ε = εId + (1−ε)P, σ²_{f,P_ε} = σ²_{f,P}/(1−ε) + εσ²_f/(1−ε), i.e. τ_ε = (τ + ε)/(1 − ε). Mixtures: Peskun/Tierney 1998 (V, metadata) gives σ²(wQ + (1−w)P) ≤ (σ²(P) + w Var_π f)/(1−w) when Q is π-reversible; Neal & Rosenthal, JAP 62 (2025) 188–208, arXiv:2305.18268 (V, abstract; efficiency dominance in random mixtures); Maire, Douc & Olsson, Ann. Stat. 42(4) 2014 (V, abstract; alternating kernels); Andrieu, Lee & Vihola arXiv:1312.6432 Lemma 34 (U; quoted in Banterle et al.).
- DA theory: Banterle, Grazian, Lee & Robert, FoDS 1(2) 2019, 103–128, doi 10.3934/fods.2019005 (V: Peskun var(DA) ≥ var(MH), Dirichlet-form bounds Props 1–2, Prop. 3 DA can lose geometric ergodicity); Sherlock & Lee, arXiv:1706.02142 (V, abstract: DA variance never lower than parent; variance bounding inherited if the log-density gap is bounded, e.g. independence samplers); Sherlock, Thiery & Golightly, Ann. Stat. 49(5) 2021 (V, abstract; diffusion limits); Franks & Vihola, SPA 130(10) 2020 (V, abstract); Quiroz et al. JCGS 2018; Golightly, Henderson & Sherlock, Stat. Comput. 25(5) 2015; Payne & Mallick, J. Classif. 35(1) 2018 (V, metadata).
- **Atchadé, Ann. Stat. 39(2) 2011, 990–1011, doi 10.1214/10-AOS828, arXiv:0911.1164** (V, full text): kernel (lag-window) estimators of asymptotic variance for adaptive MCMC. A1 simultaneous geometric ergodicity + drift; A2 D(θ_{n−1}, θ_n) ≤ Cγ_n V^η(X_n) with deterministic γ_n = O(n^{−α}), α > ½. The estimator converges to a *random* limit Γ²(h); n^{−1/2}Σh(X_k) → √Γ² · Z (normal mixture); consistent for σ²(h) only if θ_n → deterministic θ*; otherwise it "clearly provides a poor estimate"; multiple chains help "but this takes away some of the advantages". Randomised installation violates A2 (O(1) increments). **No other consistency theory for Geyer / Sokal / Wolff / batch means under adaptation was found.**
- Estimators: Geyer, Stat. Sci. 7 (1992) Thm 3.2 gives lim inf σ²_pos,n ≥ σ² a.s. for reversible chains (per Dai & Jones, JMVA 159 (2017), arXiv:1706.00853, V) — "asymptotically conservative"; Flegal & Jones, Ann. Stat. 38(2) 2010 (V); Vats, Flegal & Jones, Biometrika 106(2) 2019 (V; multivariate ESS); Vehtari et al., Bayesian Analysis 16(2) 2021 (V); Thompson arXiv:1011.0175 (V); Wolff, CPC 156 (2004) 143–153, erratum 176 (2007) (V); Sokal 1997, NATO ASI B 361, doi 10.1007/978-1-4899-0319-8_6 (V via secondary). Software practice: RStan docs "warmup samples should not be used for inference"; PyMC discards tuning draws by default (V).

### 2.7 The authors' own record (all V)

| Item | Content relevant to the paper |
|---|---|
| Bérešová, "Numerical realization of the Bayesian inversion accelerated using surrogate models", PANM 21 (2022) 25–36, doi 10.21136/panm.2022.03 | Introduces DAMH-SMU and the SAMPLER / COLLECTOR / SOLVERS-pool architecture (polynomial and RBF surrogates). Verbatim: "Since G̃ changes, the proposal distribution also changes and the algorithm becomes an adaptive MCMC method. Several possibilities of ensuring the ergodicity … are offered by [Bai–Roberts–Rosenthal 2011; Roberts–Rosenthal 2007]. In situations when it is not possible to prove the ergodicity of the DAMH-SMU algorithm, the validity of the sampling process can be ensured simply by stopping the adaptations at some point." |
| Bérešová (Domesová), PhD thesis, VŠB-TU Ostrava 2022, hdl 11104/0336621 | DAMH support condition supp f̃ ⊃ supp f (cited by PANM). |
| Domesová, Béreš & Blaheta, IACMAG 2021, LNCE 125, 846–853; Blaheta, Béreš, Domesová & Horák, Comput. Geosci. 24 (2020) 1911–1932; Domesová & Béreš 2017/2018; Domesová AETA 2018 | Surrogate-accelerated DAMH applications; surrogate built in a preliminary stage. |
| IPMS 2024 extended abstract (Springer 2025, doi 10.1007/978-3-031-87213-6_44); SNA'25 (Bérešová, Béreš, Luber: "DAMH-SMU adaptive algorithm with subchains", MLP retrained continuously, no ergodicity discussion); seminar talk 11 Nov 2024 (abstract claims "adaptively constructed surrogate models, while maintaining the asymptotic exactness of the Markov chain" — claimed, not proved) | NN surrogates on TSX. |
| **Bérešová, Béreš, Luber & Sysala, arXiv:2606.14743 (3 June 2026)** | Collector architecture, Hamiltonian sub-chains on the surrogate via autodiff, DA with NN; "the neural network is updated only during a burn-in phase, after which the surrogate model remains fixed"; "stepwise exactness is not enough to guarantee ergodicity"; "the standard diminishing-adaptation and containment conditions are not satisfied". Cites Deveney 2023, Conrad 2016, Cui 2011, Sherlock 2017, MLDA. |
| github.com/dom0015/surrDAMH (Version2) | "Several Markov chains run in parallel and share one surrogate". |

---

## 3. Similarity map: each claimed result against its closest prior art

| Draft result | Closest prior art | Identical | Different / added by the draft |
|---|---|---|---|
| R1 Prop. 2.3 (DAMH invariant for any π̃-reversible sub-chain incl. HMC, K > 1, atoms) | Christen & Fox 2005; Tierney 1998; MLDA 2023 (arbitrary-length sub-chains, detailed balance); Cui–Marzouk–Willcox 2015 (L-step sub-chain); Zhang et al. 2017 App. A; Deveney 2023 (K = 1 HMC); own arXiv:2606.14743 | the statement | a clean general statement; mild. **Known.** |
| R2 Thm 5.1 (prior step at fixed rate ⇒ SUE, δ = w e^{−B₊} Z) | SGH17 Alg. 1 + Assumption 1 + App. D.1; PHL20 Thm 4.1; AM06 Rem. 7; HHH09; GK10; Brofos 2022 App. N; JJN13; MT96 Thm 2.1; Wang 2022 | the mixture-with-a-globally-minorised-kernel argument; the independence-proposal remark | the *prior* as the canonical proposal, so the minorisation needs only ℓ ≤ B₊ (noise model) and Z > 0: nothing on G (measurable, may fail), the surrogate, the sub-chain (HMC with adapted ε, M, L) or the tails; unbounded space; applied to DA with K-step sub-chains; HMC irreducibility sidestepped. **Mechanism known, scope and packaging new.** |
| R3(a) Section 6.1 randomised installation | RR2007 Cor. 7/16; SGH17 (p_i → 0); Zhang 2017 (a_t → 0, HMC, network); KMC 2015; Wang 2013; Air §3.4; LV Ex. 22; Marshall–Roberts (sparse adaptation times) | the device | joint installation of network versions and HMC parameters; rigorous combination with SUE (Zhang 2017 and KMC skipped this); rate guidance Σλ_n/n < ∞ vs Σλ_n/√n. **Device known.** |
| R3(b) Section 6.2 averaged installation | GK10 eq. (9) (averaged proposal parameters); Cui 2019 Thm 3 (Lipschitz-in-index DA); RLMH 2024 Lemma 4 (clipping + ball + Lipschitz-in-weights for NN proposals); AM06 Prop. 19–20; LV Lemma 16 + Rem. 15 (projection) | the continuity route | TV-Lipschitz bound of the DAMH kernel in ℓ̃ (Lemma 6.5, "(8K+4)Δ"); perceptron Lipschitz in weights with input clipping and weight box; Polyak averaging of *surrogate* weights. **No precedent for Polyak-averaged surrogate weights found; the most novel technical piece.** Restrictions (Gaussian noise, MLP, fixed proposal density, no HMC) are real and must be stated as such. |
| R4 Thm 6.11 (ergodic, coupling bound, WLLN, SLLN, no CLT) | RR2007 Thm 5/23 (+ proof constants); HHH09 Thm 2 and Brofos Thm 4.1 (product bounds); LV Thm 9/11/20; FMP 2011 A4; Air Thm 1; RR2007 Ex. 24 (SLLN failure) | the theorems | assembly for this sampler; SLLN for an online-trained surrogate in DA (not found elsewhere: SGH17, Cui, Conrad give TV only; Davis 2022 an MSE rate under drift). **Assembly; the SLLN claim is new in this literature.** |
| R5 Section 7 (cost model, IACT, estimation under adaptation) | ŁR2013 Cor. 1 (lazy chain); Peskun bounds; Atchadé 2011 (estimation under adaptation); Geyer/Flegal–Jones/Vats; Stan/PyMC practice; Craiu 2009 INCA (shared adaptation, product space) | the estimators and the lazy-chain formula | ESS-per-exact-evaluation accounting with pre-rejections and the rate w; the explicit protocol (frozen tail / replicate runs). **Standard; cite ŁR2013 and Atchadé 2011.** |
| Discussion routes 1–2 | RR2009 §8 (drift), BRR 2011, Parno–Marzouk (SSAGE), Cui 2011 (compact), Brofos App. N (compact defensive) | – | the "bounded prior" route 2 is what Cui 2011 / Brofos do; say so. |

---

## 4. Recommendations

### 4.1 What is **not** new (state it, do not hide it)

- SUE from a mixture with a fixed, globally minorised exact kernel (SGH17; PHL20; AM06; HHH09; GK10; Brofos; folklore per AT08).
- Independence proposal dominating the target ⇒ whole space small (MT96; Tierney 1994; AM06 §7.1; SGH17 App. D.1); the prior as a heavy-tailed dominating proposal (HHH09 remark); pCN with β = 1 is the prior independence sampler (Cotter 2013; HSV14).
- Randomised adaptation with vanishing probability (RR2007; SGH17; Zhang 2017; KMC; Wang 2013; Air; LV Ex. 22).
- Ergodicity + WLLN from SUE + DA (RR2007); SLLN under Σ E[D_n]/n < ∞ (LV; FMP 2011); CLT requires converging kernels (LV Lemma 18; FMPV 2014; Air Thm 1(iv); Atchadé 2011).
- Lipschitz-in-index route to DA (AM06; Cui 2019; LV Lemma 16; RLMH 2024); averaging adaptive parameters (GK10).
- DAMH invariance with sub-chains (CF2005; MLDA; Cui–Marzouk–Willcox 2015; own 2606.14743).
- Lazy-chain IACT formula (ŁR2013 Cor. 1); Peskun ordering for DA (Banterle 2019; Sherlock & Lee).
- "Freeze after burn-in" as the valid default (NUTS; Stan; Deveney 2023; Cui–Marzouk–Willcox 2015 finite adaptation; own PANM 2022 and 2606.14743).

### 4.2 What the contribution **is** (the defensible core)

1. **A model-free safe component for Bayesian inverse problems.** For any posterior with a bounded likelihood (Gaussian noise suffices), the exact prior-proposal step at a fixed rate makes *every* adaptive family of π-invariant kernels simultaneously uniformly ergodic, on ℝ^d, with the explicit and sharp (Wang 2022) constant w Z e^{−B₊}. It answers Andrieu & Thoms' "practical choice of the non-adaptive component" objection, removes the compactness / tail / finiteness assumptions of SGH17's practice, Wang 2013, Cui 2011/2019, Conrad 2016/2018 and Parno–Marzouk, and makes HMC irreducibility (Durmus–Moulines–Saksman; Livingstone et al.) irrelevant for an NN surrogate potential whose regularity nobody can verify.
2. **Diminishing adaptation for a neural network by sampler-side rules only**, with no assumption on the network's accuracy or training: randomised installation (any sub-chain, incl. HMC with adapted ε, M, L) and averaged installation (new TV-Lipschitz bound of the DAMH kernel in ℓ̃, perceptron bound in a weight box). This repairs the one-line appeals of Zhang et al. 2017 / KMC / MLDA / Lykkegaard 2021 / Benavoli 2021 and the freeze of Deveney 2023 and own 2606.14743.
3. **The first ergodicity + WLLN + SLLN statement for a delayed-acceptance sampler whose global NN surrogate is retrained without limit**, including the shared-collector multi-chain setting (cf. Conrad 2018 for local approximations on compact sets).
4. **An honest cost analysis** of the guarantee: fraction w of unscreened exact evaluations, lag of installation, the per-ESS-per-evaluation multiplier, and a protocol for error estimation under adaptation grounded in Atchadé 2011. (Experiments pending.)
5. **Closing an open problem the authors themselves posed** (PANM 2022; 2606.14743).

### 4.3 Framing

- **Title/abstract**: keep "by algorithmic means" — it is the correct differentiator from the inexact-bound literature (Stuart–Teckentrup, Lie et al., Sprungk, Lu et al.) and from the compactness/drift proofs. Consider stating in the abstract: "the two conditions are met by two settings of the sampler, an exact prior-proposal step at a fixed rate and a rule for installing network versions, with no assumption on the forward map, the network or the posterior tails".
- **Introduction narrative** (suggested order): (i) surrogate-accelerated DA is standard and exact for a fixed surrogate (CF2005; Cui 2011; MLDA; Deveney 2023; own work); (ii) training the surrogate during sampling is what everyone wants and does (Zhang 2017; Lykkegaard 2021; Benavoli 2021; Yan–Zhou 2020; own 2606.14743), and it makes the chain adaptive (RR2007); (iii) the literature either freezes (NUTS/Stan practice; Deveney; Cui–Marzouk–Willcox; own PANM 2022), appeals to RR2007 without checking containment (Zhang 2017; KMC; MLDA remark; Lykkegaard 2021 "de facto"), or proves it under compactness / tail envelopes / drift (Cui 2011/2019; Conrad 2016/2018; Parno–Marzouk; Wang 2013), or under an assumed minorisation of a fixed kernel (SGH17); (iv) this paper: the prior step makes SGH17's assumption automatic and model-free, two installation rules give DA for a network, result: ergodic, WLLN, SLLN, quantitative coupling bound, with the costs quantified; (v) what is not given (rate, CLT, bias bound) — contrast Davis 2022.
- **Present Theorem 5.1 as a generalisation of SGH17's Theorem 1 with Assumption 1 verified by the noise model**, not as an independent discovery; keep the "value lies in what it does not ask" sentence, it is fair.
- **Fix Rem. 5.3 / intro references to RR2009**: the fixed component there yields containment by drift; the right analogues are SGH17, PHL20, AM06, HHH09, GK10.
- **Robustness message** (Rem. 6.12(ii), "a poor network at the beginning is a transient, not a permanent bias"): cite Cui–Marzouk–Willcox 2015 for the same claim in the finite-adaptation setting, and Sprungk 2020 / Lu et al. 2023 for why inexact-surrogate bias is worst precisely when data are informative.
- **Be explicit about δ**: Rem. 5.3(ii) already says the bound is vacuous; add Wang 2022 (exact rate, so no better constant exists for the pure prior step) and Cotter 2013's 12/10⁶ as evidence; present the defensive-mixture / i-SIR variants (library_notes/35; Samsonov 2022) as the way to make the exact evaluations useful — one remark, not implemented.
- **Multi-chain collector**: formalise once via the product-space device of Craiu, Rosenthal & Yang 2009 (chains conditionally independent given the shared index) and cite Conrad 2018 as the precedent; note the asynchronous MPI timing must be indexed so that (A1)/eq. (adaptive-step) holds per chain.

### 4.4 Concrete text changes (by location)

| Location | Change |
|---|---|
| Section 5 intro, Thm 5.1, Rem. 5.2 | Cite SGH17 (Alg. 1, Assumption 1, App. D.1) and PHL20 Thm 4.1 as the same argument; AM06 Rem. 7, HHH09, GK10, Brofos 2022 as proposal-level and flow analogues; AT08 p. 350 for the folklore and the "practical choice" objection. Replace the RR2009 analogy or qualify it ("a fixed proposal component used for containment via drift"). |
| Rem. 5.3(ii) | Add Wang 2022 Thm 2 (rate exact), Cotter 2013 §4.4 (12 of 10⁶), and the i-SIR / defensive-mixture remark. |
| Section 6.1, Rem. 6.3 | Add RR2007 Cor. 7/16 wording, Zhang 2017 and KMC as surrogate-HMC precedents that used the device without SUE, Wang 2013 (compact), Air §3.4 ("can decay arbitrarily slowly"), LV Ex. 22. |
| Lemma 6.5 ("no source known") | Cite Cui 2019 Thm 3 (Lipschitz-in-index DA), AM06 Prop. 19–20, LV Lemma 8/16 as the same type of estimate; the (8K+4) constant for DAMH appears to be new. |
| Section 6.2 (weight box, clipping, averaging) | Cite GK10 eq. (9) (parameter averaging in adaptive IMH), RLMH 2024 Lemma 4 (clipping + ball + Lipschitz in NN weights), LV Rem. 15 (projection). |
| Thm 6.11(iv), Rem. 6.12(iv) | Motivate Σλ_n/n < ∞ by RR2007 Example 24 (SLLN fails under DA alone with a 1/k-type randomised rule); cite LV Thm 20 directly (no continuity needed) besides Thm 9 + Lemma 11; for the CLT add FMPV 2014 A4 and Atchadé 2011 (normal *mixture* limit). |
| Section 6.4 (alternatives) | (ii) Air MCMC is still a preprint — say "preprint"; add Hofstadler et al. 2026 (a.s. rates for rare adaptation); (v) add Johndrow–Mattingly (TV vs W₁) and Brown & Rosenthal 2025 (Wasserstein adaptive theory) as the plausible route, Atchadé 2006 (truncated drift) as the Langevin precedent; add Lan et al. 2016 / Gilks–Roberts–Sahu 1998 (adaptation at regeneration times, identifiable thanks to the prior step) as alternative (vi). |
| Rem. 5.2(iii) / Section 6.5 (HMC irreducibility) | Cite Durmus–Moulines–Saksman 2020 and Livingstone et al. 2019 for why irreducibility of a Metropolised HMC sub-chain is non-trivial and for the mixture-inherits-irreducibility remark. |
| Section 6.5 (frozen references) | Cite Cui–Marzouk–Willcox 2015 (finite adaptation with sub-chains), Deveney 2023 and own 2606.14743 (freeze after warm-up/burn-in) as the practice being compared against. |
| Section 7.2 | "consistent overestimate" → "asymptotically conservative: lim inf of the estimate ≥ σ² a.s. for reversible chains (Geyer 1992, Thm 3.2)". |
| Section 7.4 and intro TODO | Cite Atchadé 2011: estimators on an adaptive run converge to a random limit; consistent only if the kernel converges to a deterministic limit with O(n^{−α}), α > ½, increments; randomised installation does not satisfy this; hence frozen tail / replicate runs. Cite RStan/PyMC practice. |
| Section 6.5(i) / 7.2 (lazy chain) | Cite ŁR2013 Cor. 1 for τ_w = (τ+w)/(1−w); note it is an upper bound for the mixture with the prior step by Peskun (Tierney 1998). |
| Introduction | Quote PANM 2022 and 2606.14743 (open problem; freeze); position vs SGH17 (generalisation), Conrad 2016/2018 (asymptotically exact, inexact per step, compact/envelope), Davis 2022 (rates, drift), Deveney 2023 (nearest method), Zhang 2017 (incomplete claim), Cui 2011/2019 (compact). |

### 4.5 Experiments that would make the paper valuable (Section 8 is still TODO)

1. **Adaptive (guaranteed) vs frozen-with-prior-step vs frozen-without-prior-step**, replicate runs, error of posterior moments against a long reference (the comparison the draft already plans). This is the paper's empirical question: does continued training beat the ≤ (1−w) multiplier of Section 6.5(i)?
2. **A robustness case**: a deliberately poor initial network (blind spot, library_notes/22) where the frozen chain is biased or stuck for the whole budget and the guaranteed adaptive chain recovers. This is the only experiment that shows the *value of the guarantee* rather than its cost, and it is what the "transient, not a permanent bias" message needs.
3. **The rate λ_n and the installation lag**: log² vs polynomial schedules; show the lag is harmless once the data set is large (the expectation stated in Section 6.4).
4. **Cost of w**: ESS per exact evaluation vs w ∈ {0, 0.01, 0.05, 0.1}, with the pre-rejection fraction reported, against the formula in Section 6.5(i).
5. **Estimation under adaptation**: compare whole-stage ESS heuristics with frozen-tail estimates and replicate-run errors (an empirical illustration of Atchadé 2011's point).
6. **HMC vs pCN sub-chains** under the same guarantee, since the HMC case is where the randomised rule is the only option.
7. Optional baselines: SGH17-style k-NN surrogate (surrDAMH has a kd-tree surrogate) under the same mixture, to show the NN/HMC gain is not the mechanism's gain; frozen "usual practice" from 2606.14743.
Report exact output directories; never fill in uncomputed numbers (CLAUDE.md rule).

### 4.6 Venue

Where the closest work appeared: SIAM/ASA JUQ (Deveney 2023; Conrad 2018; MLDA 2023; Parno–Marzouk 2018; Lie 2018; Latz 2020) — best fit for theory + algorithm + PDE inversion; Statistics and Computing (Zhang 2017; Davis 2022; Bon 2021; Marshall–Roberts 2012) — good for an algorithm/theory-centred version; JCGS (SGH17; CF2005) — natural if framed as "SGH17 generalised"; Bayesian Analysis (Buchholz 2021; Vehtari 2021); JASA (Conrad 2016) is unlikely without rates or a CLT. Recommendation: SIAM/ASA JUQ first, Statistics and Computing or JCGS second.

### 4.7 Referee objections to pre-empt

1. "Theorem 5.1 is SGH17 Thm 1 with Assumption 1 verified" — yes; say so and sell the scope (no compactness, no assumption on G, HMC, NN, SLLN).
2. "δ is astronomically small, the guarantee is vacuous" — Rem. 5.3(ii) + Wang 2022 + Cotter 12/10⁶; the content is uniformity in state and index (robustness), not speed; experiments 1–2 must show robustness, not speed.
3. "No rate, no CLT; Davis 2022 has log³T/T" — state it; explain why a CLT is unavailable for NN weights (LV Lemma 18, FMPV A4, Atchadé 2011).
4. "Averaged installation is proven under restrictive hypotheses (Gaussian noise, MLP, fixed proposal density, no HMC)" — already stated in Section 6.4; keep randomised installation as the main rule.
5. "The multi-chain asynchronous collector is not an adaptive MCMC algorithm in the RR2007 sense" — formalise via Craiu 2009 / Conrad 2018 and the filtration remark; the installation coin fixed before the step is the key.
6. "If experiments show freezing is as good, the paper's practical message is 'freeze'" — that outcome is still publishable (it quantifies the cost of a guarantee), but then the robustness experiment carries the paper.
7. "Why not GIST / NUTS-style exact self-tuning instead of adapting HMC parameters?" — note GIST proves reversibility only; the prior step would give ergodicity for those sub-chains too (one sentence).

---

## 5. Reference checklist for `refs.bib` (not added; data as seen)

Already in refs.bib (fine): RR2007, RR2004, RR2009, BRR2011, LR2014, FMP2011, HST2001, AT2008, MT1996, MT2009, Tierney 1994/1998, CF2005, Cui 2011, Banterle 2019, Stuart 2010, KS2005, Cotter 2013, Neal 2011, Lykkegaard 2021, SGH2017, LV2026, Kingma–Ba, Izmailov, Thorisson, Jarner–Hansen, Dellacherie–Meyer, Kallenberg, Chimisov 2018, Geyer, Sokal, Flegal–Jones, Vehtari.

To add (V unless marked):
- Pompe, Holmes, Łatuszyński, Ann. Stat. 48(5) 2020, 2930–2952, doi 10.1214/19-AOS1916.
- Andrieu & Moulines, Ann. Appl. Probab. 16(3) 2006, 1462–1505, doi 10.1214/105051606000000286.
- Holden, Hauge, Holden, Ann. Appl. Probab. 19(1) 2009, 395–413, doi 10.1214/08-AAP545.
- Giordani & Kohn, JCGS 19(2) 2010 (pages U), arXiv:0801.1864.
- Brofos, Gabrié, Brubaker, Lederman, AISTATS 2022, PMLR 151:5949–5986.
- Wang, G., Bernoulli 28(3) 2022, 2012–2033, doi 10.3150/21-BEJ1409.
- Łatuszyński & Roberts, Methodol. Comput. Appl. Probab. 15(1) 2013, 237–247, doi 10.1007/s11009-011-9237-8.
- Atchadé, Ann. Statist. 39(2) 2011, 990–1011, doi 10.1214/10-AOS828.
- Atchadé, Methodol. Comput. Appl. Probab. 8(2) 2006, 235–254, doi 10.1007/s11009-006-8550-0.
- Fort, Moulines, Priouret, Vandekerkhove, Bernoulli 20(2) 2014, 457–485, doi 10.3150/12-BEJ493.
- Conrad, Marzouk, Pillai, Smith, JASA 111(516) 2016, 1591–1607, doi 10.1080/01621459.2015.1096787.
- Conrad, Davis, Marzouk, Pillai, Smith, SIAM/ASA JUQ 6(1) 2018, 339–373, doi 10.1137/16M1084080.
- Davis, Marzouk, Smith, Pillai, Stat. Comput. 32:60 (2022), doi 10.1007/s11222-022-10123-0.
- Cui, Fox, O'Sullivan, IJNME 118(10) 2019, 578–605, doi 10.1002/nme.6028.
- Cui, Marzouk, Willcox, IJNME 102(5) 2015, 966–990, doi 10.1002/nme.4748.
- Lykkegaard, Dodwell, Fox, Mingas, Scheichl, SIAM/ASA JUQ 11(1) 2023, 1–30, doi 10.1137/22M1476770.
- Deveney, Müller, Shardlow, SIAM/ASA JUQ 11(3) 2023, 970–995, doi 10.1137/22M1513113.
- Zhang, Shahbaba, Zhao, Stat. Comput. 27(6) 2017, 1473–1490, doi 10.1007/s11222-016-9699-1.
- Strathmann, Sejdinovic, Livingstone, Szabó, Gretton, NeurIPS 28 (2015), arXiv:1506.02564.
- Lan, Bui-Thanh, Christie, Girolami, J. Comput. Phys. 308 (2016) 81–101 (vol./pages U), arXiv:1507.06244.
- Wang, Mohamed, de Freitas, ICML 2013, PMLR 28(3):1462–1470.
- Durmus, Moulines, Saksman, Ann. Statist. 48(6) 2020, 3545–3564, doi 10.1214/19-AOS1941.
- Livingstone, Betancourt, Byrne, Girolami, Bernoulli 25(4A) 2019, 3109–3138, doi 10.3150/18-BEJ1083.
- Hoffman & Gelman, JMLR 15 (2014) 1593–1623 (pages U).
- Parno & Marzouk, SIAM/ASA JUQ 6(2) 2018, 645–682.
- Craiu, Rosenthal, Yang, JASA 104(488) 2009, 1454–1466, doi 10.1198/jasa.2009.tm08393.
- Johndrow & Mattingly, arXiv:1711.05382 (2017). Brown & Rosenthal, J. Appl. Probab. 62 (2025) 1318–1339, doi 10.1017/jpr.2025.4. Hofstadler, Łatuszyński, Roberts, Rudolf, SPA 2026, doi 10.1016/j.spa.2026.104905, arXiv:2402.12122.
- Wang, Chen, Kanagawa, Oates, arXiv:2405.13574 (2024). Samsonov et al., NeurIPS 2022, arXiv:2111.02702.
- Sprungk, Inverse Problems 36 (2020) 055015 (no. U), arXiv:1906.07120. Stuart & Teckentrup, Math. Comp. 87(310) 2018, 721–753. Lie, Sullivan, Teckentrup, SIAM/ASA JUQ 6(4) 2018, 1600–1629.
- Dai & Jones, J. Multivariate Anal. 159 (2017) 184–199 (for Geyer's Thm 3.2 statement). Vats, Flegal, Jones, Biometrika 106(2) 2019, 321–337.
- Sherlock & Lee, arXiv:1706.02142 (venue U). Sherlock, Thiery, Golightly, Ann. Statist. 49(5) 2021, 2972–2990. Franks & Vihola, SPA 130(10) 2020, 6157–6183.
- Johnson, Jones, Neath, Statist. Sci. 28(3) 2013, 360–375. Gilks, Roberts, Sahu, JASA 93(443) 1998, 1045–1054.
- Own: Bérešová, PANM 21 (2022) 25–36, doi 10.21136/panm.2022.03; Bérešová, Béreš, Luber, Sysala, arXiv:2606.14743 (2026).

Unverified attributions to drop or check: "Brown & Jones WIREs 2024, constant L(θ*)/Z" (their 2024 paper is JAP 61, 33–54, on Wasserstein rates); Andrieu, Lee & Vihola Bernoulli 2018 Lemma 34 (quoted via Banterle); Hesterberg's defensive mixtures (Technometrics; year inconsistent across sources); Caracciolo–Pelissetto–Sokal 1990 (not reachable).

---

## 6. Searched for and not found (the gaps are real as of 2026-10-23)

- A stand-alone lemma "mixture with a fixed Doeblin kernel ⇒ SUE" in RR2007, LV or AT08 (only instance-level statements exist).
- An explicit statement of the constant Z e^{−sup ℓ} for a prior-proposal exact step in the Bayesian-inversion literature (ingredients exist; the one snippet pointing to Brown & Jones is unverified).
- Polyak-averaged surrogate weights as a DA mechanism.
- A TV- or Wasserstein-Lipschitz bound of the Metropolised HMC kernel in step size / mass matrix / gradient field, uniform in the state.
- An ergodicity proof for a global NN surrogate retrained without limit inside DA, on an unbounded space; an SLLN or CLT for any online-trained surrogate in DA.
- A consistent IACT/ESS estimator for adaptive chains beyond Atchadé 2011's conditions.
- Any 2024–2026 competitor to R2–R4 (checked: Cao et al. 2025, Zacchei et al. 2025, Jimenez-Beltran et al. 2026, Riccius et al. 2024, MINE arXiv:2603.10987, MCMC-Net arXiv:2412.16883, the 2026 survey arXiv:2603.13646 — all offline-trained or inexact).
