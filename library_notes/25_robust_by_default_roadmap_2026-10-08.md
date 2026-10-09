# 25 — Robust-by-default roadmap: two simple modes, shared safety components, and the open items they absorb (2026-10-08)

**Design note / plan**, nothing implemented. Goal set by the author (2026-10-07/08): *a user gives the forward
model, the noise model and the prior, and obtains reliable posterior samples without stagnation and without
biased chains, with no advanced settings; posteriors may have several modes that are not far apart.* Two simple
modes are to be offered: one as robust as possible, one faster (Hamiltonian on network gradients). Sources:
notes 15–24, a sweep of every open item in `library_notes/` and `CHANGELOG.md` done 2026-10-08 (§2), the
tinyDA comparison (chat, 2026-10-07), and the code at commit `c518457` plus the uncommitted tree.

---

## 1. Principle

**Both modes are exact.** The difference between them is how fast the chain mixes on a hard problem, never
whether the sampled law is right. "Fast" therefore means "may be inefficient", not "may be wrong". This is a
design rule, not a slogan: every component that protects exactness (§3 S1–S4) is in both modes; the modes
differ only in the proposal of the DAMH component (§4).

The user-facing contract: `stages=Auto(budget=..., mode="robust" | "fast")`. Everything else keeps today's
defaults. The resolved stage list is printed at start-up and written to the manifest, so any automatic run
can be re-run as an explicit stage list by an expert.

---

## 2. Review of open items (sweep of notes 00–24 and CHANGELOG, 2026-10-08)

41 open items were found. Most are either absorbed by a roadmap component below or stay in the backlog as
not relevant to this goal. Items the sweep flagged as **forgotten** are marked ★.

### 2.1 Absorbed into the roadmap

| open item (source) | goes into |
|---|---|
| DAMH sub-chain proposals adapt on the *outer* acceptance, 2–9× over-scaling at long sub-chains (10 §8(b4), 19 §4.2/§7(b), 21 §6 item 4, 16 §2) | S6 (fix) + layout rule "adaptive DAMH uses sub-chain length 1" (§5) |
| blind spot P1 audit (22 §4 P1) | S2 |
| blind spot P2 / kernel mixture (22 §4 P2, 24 §8) | S2 |
| blind spot P3 global error model (22 §4 P3, 16 §5 item 8) | S1 |
| blind spot P6 diagnostics: audit statistic, `weighted_ess`, warm-up vs final shift (22 §4 P6) | S5 |
| DAMH-SMU violates diminishing adaptation A5; fixes N1–N3, Polyak recommended (18 §6, 20) | S3 |
| ★ prior truncation vs containment on ℝ^d, decision needed to close the proof (18 §5.3, 20 §6 point 1) | S3 (decision D4) |
| "surrogate still learning" RMSE-trend diagnostic (20 §6 point 2) | S5 |
| no-move / >90 % pre-rejection warning after ~1000 iterations (06 7.5, 15 §4.3 item 4, 19 §7(a)) | S5 (stagnation flag) |
| ESS per exact evaluation / per second, `α₁`, `α₂|₁`, surrogate-call counts not reported (06 7.6, 15 §4.3, 16 §5 item 3) | S5 |
| carried adaptation state not in the manifest; a `scale=None` run is not reproducible from its manifest (06 7.6, 10 §8(b3)) | S7 |
| no section of warnings raised during the run in `write_report` (09 WS9) | S5 |
| NN surrogate: no input normalisation, no validation split, early stop on one minibatch, astronomically large finite proposals reach the training set unguarded (06 3.9/3.12, 09 WS6) | S4 |
| polynomial/RBF with `min_snapshots_to_update=0` stalls at 10⁵ snapshots (19 §4.3/§7(d)) | S4 (size-dependent default) + layout (network is the default surrogate) |
| `GaussianMixture` distribution: `grad_logpdf=0`, `logpdf=−inf` far from components, `rvs()` shape, no `mean`/`get_covariance` (06 1.11, 10 §3) | S8 (fix or refuse as prior in `Auto`) |
| Hamiltonian mass default `Σ̂⁻¹` with step re-adaptation at fixed integration time; expose `T` instead of `num_steps` (16 §5 items 4–5, CHANGELOG). **Robust estimation of the mass worked out in note 26 §10 (2026-10-09)**: identity + low-rank correction, shrinkage towards the prior, within-chain pooling, clipping, fallbacks; open decisions MM1–MM3 there | fast mode, phase 3 (§7) |
| mixture of proposals (23) | tier 2, M1 |
| generalised pCN (21 §6 item 3, §8) | tier 2, opt-in; measured 2–2.5× ESS/eval over covariance RW — not a robustness item |
| DAMH `target_rate` meaning: outer vs second-stage acceptance (06 5.8, 09 §4 item 2) | settled by S6: sub-chain proposals adapt on their own acceptance, the outer rate is reported |

Already done but still listed open in older notes (verified in code): pooling of the ranks' sufficient
statistics instead of averaging covariances (`set_pooled_state`, 2026-09-21); the frozen-DAMH-after-MH stale
evaluator (`_install_newest_evaluator`, 2026-09-30).

### 2.2 Backlog, not relevant to this goal (kept in 10 §8)

Performance items (WS8 transfer, batching, `Waitany`; Cholesky cache; torch dtype casts; post-processing
loops), protocol tests I12, TODO banners, ★ write-only flags (`_request_pending`,
`snapshot_count_since_last_update`, `PendingRequest.active`/`max_requests` — set, never read; may hide an
intended guard), legacy `Samples.html_report` deletion, ★ `mu`/`sigma` naming, ★ `paths_to_append` →
`SolverSpec`, `Configuration` importing `mpi4py`, seed last mile (`torch.manual_seed`, bare `np.random` in
`initial_training`/`tools`/`test_data` — affects reproducibility, medium), ★ unreachable code after `raise`
in the gradient-with-transform path, ★ unbounded MPI tag growth, frozen-stage 2–3× wall-time anomaly, pCN
β → 1 saturation print, GRF example's rank-2 forward map (14 §10 F3), randomised sub-chain length, ChEES.

Two of these deserve a decision soon although they are outside the goal: the write-only flags (a latent
logic gap, 15 minutes to read) and the seed last mile (reproducibility of the automatic modes).

---

## 3. Shared components (both modes)

| id | component | default | evidence | size |
|---|---|---|---|---|
| **S1** | **Global Gaussian error model** in the surrogate likelihood: `N(y; G̃(u) + ε̄, Σ_noise + Σ_surr)`, `ε̄`/`Σ_surr` from the residuals of the *current* surrogate on all exact snapshots (posterior-weighted by multiplicity), shrunk towards the held-out `TestData` residuals when snapshots are few; full `Σ_surr` with Ledoit–Wolf shrinkage; the same object in the sub-chain, the Hamiltonian potential and the DA correction | on; `error_model=False` to disable | 22 §4 P3; notebook `damh_surrogate_from_data_2d.ipynb` (global model); tinyDA/Cui et al. 2018 (same model, accepted-state recursion); Lykkegaard et al. 2021 on prior-vs-posterior residuals | ~350 (likelihood object, collector statistics, evaluator message +2 fields in all roles, tests, report) |
| **S2** | **Exact safety kernel + audit** in every DAMH stage: kernel mixture `(1−p)·DAMH + p·MH(RandomWalk())` (24), `p = 0.05`; audit of pre-rejected proposals with `p_audit = 0.05` (22 P1); surrogate scored at every exact-step proposal (audit statistic for free) | on, fixed weights | 24 §5–§7 (cost `p·r/(1−r)`: +2 % Hamiltonian scheme, +19 % long RW sub-chains at `p = 0.1`; A4 dropped from the Doeblin bound); 20 §(d) | ~300 for the minimal P2+P1 facade (24 §6.6), ~850 for the general `kernels.Mixture` (24 §4) |
| **S3** | **Diminishing adaptation of the network**: sampler-side Polyak averaging of received weights per stage (20 N1); decision D4 on prior truncation | on | 18 §6, 20 | ~150 |
| **S4** | **Surrogate hardening**: input standardisation, validation split for early stopping, refuse training points with `|u| > 8` (internal space) or non-finite outputs, size-dependent `min_snapshots_to_update` for RBF/polynomial | on | 06 3.9/3.12, 19 §4.3 | ~150 |
| **S5** | **Verdict diagnostics** in `summary.csv` and the HTML report: between-chain split R-hat and ESS per stage (per exact evaluation and per second), outer acceptance, `α₁`, `α₂|₁`, pre-rejection rate, audit statistic (`log L − log L̃` at exact-step and audited points, count above 5 nats), "surrogate still learning" RMSE trend, warm-up-vs-final posterior shift, stagnation flag (zero exact acceptance or >90 % pre-rejection over 1000 iterations, printed during the run), warnings raised during the run; one verdict line per stage | always | 22 P6, 15 §4.3, 16 §5 item 3, 20 §6, 06 7.5/7.6 | ~300 |
| **S6** | **Sub-chain adaptation fix**: a random-walk / pCN proposal inside a DAMH sub-chain adapts on the sub-chain's own acceptance against `π̃` (as the Hamiltonian dual averaging already does), the outer rate is only reported | on | 21 §6 item 4, 19 §4.2 | ~80 (+ behaviour-change evidence per CLAUDE.md) |
| **S7** | **Reproducibility of automatic runs**: resolved `Auto` layout and the carried adaptation state written to the manifest | always | 06 7.6 | ~40 |
| **S8** | **Prior support**: `GaussianMixture` fixed (log-sum-exp, gradient, `rvs` shape, `mean`/`get_covariance`) or refused by `Auto` with a message | — | 06 1.11 | ~80 |

Nothing in S1–S8 asks the user for a value. S1 and S2 change the acceptance rate of DAMH stages by design
(flag per CLAUDE.md rules); S6 changes the adapted scale inside DAMH stages (behaviour change, needs the
evidence table of 10 §2).

---

## 4. The two modes

| | **robust** | **fast** |
|---|---|---|
| warm-up | adaptive random walk, MH on the exact model | same |
| DAMH proposal | adaptive random walk (shrinkage-AM + Robbins–Monro), sub-chain length 1 | `Hamiltonian(integrator="dimension_robust")`, mass `I`, dual-averaged step, `num_steps` default 30 (19), sub-chain length 1 |
| surrogate | network (differentiability not needed; RBF/polynomial allowed by `Auto` only below a size limit, S4) | network (gradients required) |
| exact component of S2 | `RandomWalk()` with the carried covariance, adaptive on its own exact acceptance | same (a random walk, not a second Hamiltonian: 24 §6.5) |
| failure mode on a hard problem | slow mixing, flagged by S5 | slow mixing or many divergent trajectories, flagged by S5 |
| expected efficiency | note 19: RW sub-chains 34–44× MH at best | note 19: 177× MH on the GRF problem |
| mode-hopping (nearby modes) | pooled covariance across LHS-started chains spans the modes; exact RW component | same mechanisms; the Hamiltonian itself hops poorly |

Phase 3 for the fast mode (not needed for exactness): mass `Σ̂⁻¹` from the carried covariance with step
re-adaptation at fixed integration time `T` (16 §5 items 4–5), **estimated robustly as in note 26 §10**
(identity + low-rank correction with shrinkage towards the prior, informed directions only, eigenvalues
clipped to `[1e-4, 1]`, within-chain pooling that excludes stuck chains, diagonal / previous-mass fallbacks;
a dense inverse sample covariance is *not* the default), and the self-demotion rule: at a stage
boundary, if the Hamiltonian component's acceptance collapsed, divergences exceeded a threshold or the audit
statistic shows hidden mass, the next stage runs the robust proposal. A switch at a stage boundary is a new
kernel and needs no new theory.

---

## 5. Automatic layout (`Auto`): splitting the budget over stages

Inputs: `budget` = exact evaluations in total (divided by the number of sampler ranks into the per-chain
budget `B`; `Stage.max_evaluations` is per chain) **or** `time_limit` in seconds; `mode`; the dimension `d`
is known from the prior. Chains start from a Latin-hypercube design in the internal space
(`initial_sample_type="lhs"`, exists). Rule (to be tuned by §6; the warm-up fraction is the only number
that the validation should move):

| stage | algorithm / proposal | per-chain budget | why |
|---|---|---|---|
| 0 warm-up | MH, adaptive random walk, `is_excluded=True` | `n0 = clip(20·d, 0.05·B, 0.25·B)` | the chains must reach the posterior from prior-typical starts (O(d) accepted moves at acceptance ≈ 0.23) and the pooled covariance needs O(d) states; every evaluation of every chain is a snapshot, so `n0 × chains` points train the first surrogate. Kept small on purpose: the warm-up is pure burn-in (excluded), and S1/S2 make the DAMH stage self-correcting, so evaluations are better spent there, where they also train the surrogate |
| 1…K DAMH chunks | DAMH-SMU, the mode's proposal, S2 on, sub-chain length 1 | `(B − n0)/K` each, `K = 4` | equal chunks instead of one long stage, because a stage boundary is cheap (one `Allgather` + carry-over) and buys: cross-chain pooling of the adaptation (shape and scale agree across chains, important for nearby modes), a diagnostics checkpoint (between-chain R-hat, stagnation, audit statistic), the fast→robust self-demotion point (§4), and an optional freeze when the "still learning" trend says the surrogate converged (expert flag, off by default) |
| final frozen | — | none | 22 P7 / 19 §4.4: never freeze right after the warm-up; freezing saves collector work only |

Rules around the table:

* **Too small a budget**: if `B < 50·d`, `Auto` runs a single adaptive MH stage with the whole budget and
  prints why (a surrogate trained on fewer points than that does not pay, 19 §4).
* **Time budget**: the same fractions on wall time, via `Stage.time_limit` (exists), with the warm-up
  additionally floored at `20·d` evaluations; if the floor is not reached inside its time share, `Auto`
  warns that the budget is too small. Chains then end chunks at different iteration counts, which the
  stage boundary already tolerates.
* **Held-out set**: `TestData` costs exact evaluations outside the chain budget. `Auto` takes a small
  prior-draw set (`min(512, 0.05·B·chains)`, counted in the budget) *and* holds out a random 20 % of the
  warm-up snapshots from training: the prior draws give coverage for the blind-spot statistic, the held-out
  warm-up points give the posterior-weighted residuals S1 needs when the chain's own snapshots are few.
* **Adaptation across chunks**: today the carry-over hands the pooled covariance (or `β`, step size) but
  not the Welford statistics, so the adaptive random walk re-learns its shape from the carried start in
  every chunk. Extend `carry_over()` to hand `(n, mean, M2)` as well, so the shape estimate continues
  (~30 lines, S7).
* `Auto` emits the explicit stage list it resolved (printed and in the manifest, S7).

Worked example, `d = 20`, four chains, `budget = 80 000` → `B = 20 000`: warm-up `n0 = clip(400, 1000,
5000) = 1000`, then four DAMH chunks of 4 750; the first surrogate is trained on about 4 000 snapshots.

## 6. Validation before either mode is called a default

* Linear-Gaussian MPI posterior tests (`tests/mpi/test_mpi_posterior.py`) for both modes; byte-identity of
  every existing explicit-stage configuration (S1/S2 default-on must not touch explicit stage lists unless
  the author decides otherwise — decision D1).
* 2-D toy models of `damh_surrogate_from_data_2d.ipynb`, including `"bimodal"`, run through the library with
  a short warm-up started away from the posterior: both modes must reach the reference posterior; record
  exact evaluations to within 0.1 sd of the reference mean.
* GRF problem (19/22): short warm-up from a prior draw, 4 chains, both modes, `p ∈ {0, 0.05, 0.1}`,
  `p_audit ∈ {0, 0.05}`, error model on/off → choose the S2 defaults from these runs, not from a setting.
* S6 evidence table: adapted scales with and without the fix at sub-chain lengths 1, 5, 20.

---

## 7. Order of work

| phase | content | size |
|---|---|---|
| 1 | S6, S4, S7, S8, the stagnation flag of S5; decision D4 | ~400 |
| 2 | S1, S2 (minimal facade or general mixture per D2), S3, rest of S5, `Auto` with the robust mode | ~1300–1900 |
| 3 | fast mode: Hamiltonian defaults, mass (robust estimation, note 26 §10)/`T`, self-demotion; tier 2: proposal mixture (23), generalised pCN opt-in (21 §8) | ~600 |

Delegation as in CLAUDE.md: algorithm/MPI/theory pieces (S1 message path, S2 kernel, S3, S6) to opus;
diagnostics, hardening, tests, docs (S4, S5, S7, S8, `Auto` plumbing) to sonnet; every diff reviewed.

---

## 8. Decisions for the author

* **D1** S1/S2 default-on for *every* DAMH stage, or only inside `Auto`? (Default-on changes acceptance
  rates of existing explicit configurations; `Auto`-only keeps them bit-identical.)
* **D2** S2 as the minimal P2+P1 facade (`Stage.exact_step_probability`, `audit_prerejected`) first, or the
  general `kernels.Mixture` (24 §3) straight away?
* **D3** S2 default weights 0.05/0.05 as placeholders until §6 decides.
* **D4** Prior truncation to a compact set (formal proof closes) vs containment on ℝ^d (assumed) — 18 §5.3.
* **D5** `Auto` warm-up rule of §5 item 1.
* **D6** Backlog items to clear now although outside the goal: write-only flags, seed last mile.
* **D7** Hamiltonian mass estimation (fast mode): decisions MM1–MM3 of note 26 §10.8 — within-chain vs total
  pooling, eigenvalue cap at the prior variance, last-stage vs accumulated samples.
