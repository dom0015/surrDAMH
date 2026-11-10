# Experiment plan for the paper mechanisms: validation set and paper set (2026-10-23; plan, nothing run)

Context: the six mechanisms of `pdf_reports/adaptive_nn_ergodicity_paper/main.tex` §6.5 are implemented
(note 37). Section 8 of the paper is a TODO. Two sets are proposed: **A** checks that the implementation
does what the theory says (on real Bayesian inverse problems, beyond the unit tests), **B** produces the
paper's figures (validity and efficiency of the guaranteed adaptive stage against frozen references).

## 0. Problems, surrogate, common settings

| Id | Problem | d | Why |
|---|---|---|---|
| LG | linear-Gaussian toy of `tests/conftest.py` (G = A x, closed-form posterior) | 2, 10 | exactness at the standard-error level, no reference run needed |
| P1 | GRF diffusion, note 19 §1 (`toy_examples/grf_diffusion.py`, 4×4 observations, noise sd 0.01, 0.2 ms/solve) | 20 | the paper's main inverse problem; nonlinear, posterior far from Laplace |
| P1-d | same solver with `no_parameters` 50 and 100 (finer `nx` as needed) | 50, 100 | dimension scaling of the prior step and of the lag |
| BA | `solver_examples/bimodal_asym.py` (c = 0.75, r = 1.25, 10-nat barrier, far-mode mass ≈ 0.15) | 10, 30 | multimodal, likelihood-made barrier, unequal modes: what the exact step buys |

Surrogate: `NeuralNetworkUpdater(hidden_layer_sizes=(100,), activation="silu", iterations_batch=100)`,
`min_snapshots_to_update=0` (continuous retraining), `input_clip=8` where the averaged rule is used.
Chains: 8 samplers + collector (`-n 10`). Replicates: R = 5 via `Configuration.seed_offset = 100_000 k`.
Stage layout (every configuration): (S0) MH adaptive random walk, 1 000 evaluations/chain, `burn_in`;
(S1) DAMH-SMU adaptive, 3 000 evaluations/chain, `burn_in`; (S2) PRODUCTION, 20 000 evaluations/chain,
the configuration under test; (S3) FROZEN TAIL, 5 000 evaluations/chain, `surrogate_model_updates=False`,
same w and same sub-chain, used only for the autocorrelation estimate (paper §7.4). Sub-chains: Hamiltonian
`dimension_robust`, `mass="covariance"`, `integration_time=pi/2`, jitter 0.5, dual-averaged step (K ∈ {1, 5});
pCN(beta carried, fixed in S2 for the averaged rule) with K = 5. Reference posterior: 10^6 exact
evaluations pooled over 8 chains of frozen DAMH K = 1 with the random walk (note 19 §3 format), per problem.
Metrics (paper §7): ESS per exact evaluation (min over the 16 observed directions, and of the mode
indicator on BA) from the frozen tail with SE over replicates; posterior moment error vs the reference
(mean and sd per observed direction, 5 %/95 % quantiles) over replicates — the only measure valid under
adaptation; f_pre, prior-step acceptance, exact evaluations per step, surrogate test RMSE
(`surrogate_quality.csv`), wall time per stage. Existing tooling: `toy_examples/tmp_efficiency_study_2026-09-28/scripts/{analyze,plots}.py`
(ESS/eval, posterior.csv) can be adapted; new: `installations/` CSV reader for segments.

Tooling gaps to close first (small): (T1) an evaluator version id (the collector's `surrogate_update_index`
attached to the published evaluator) so a chain's received version can be mapped to the collector's
learning curve (needed by A2, B3); (T2) per-evaluation timings c_e, c_v, c_grad (stage wall time / counts
is the fallback); (T3) a reader that splits a chain into installation segments (`installations/` +
`samples/`).

## A. Validation set: does everything work as intended

| Id | What | Configuration and range | Measured | Pass |
|---|---|---|---|---|
| A1 | Prior step is exact | LG d = 2, 10; DAMH-SMU with `exact_step_proposal="prior"`, w ∈ {1, 0.5, 0.1, 0.02}; K ∈ {1, 5}; 3 replicates | mean, covariance vs closed form; SE from ESS | every moment within 3 SE; w = 1 reproduces the independence sampler's acceptance = E_pi[min(1, L(y)/L(x))] |
| A2 | Coin mechanics with a real collector | P1 d = 20, NN, 8 chains; `RandomisedInstallation(n_lambda, a_lambda)`, n_lambda ∈ {100, 1 000, 10 000}, a_lambda ∈ {0.5, 0.75, 1}; collector made FAST (`iterations_batch=20`) so versions outnumber coins | installs vs Σ_n λ_n·1{pending}; gaps between installs vs n^{a_lambda}; `versions_received` > `installations`; chains hold different versions at the same time (T1) | installs within binomial SE of the prediction; gap growth exponent within ±0.1 of a_lambda; notes = CSV counts |
| A3 | Sub-chain parameters change only at installs | P1, Hamiltonian `integration_time` (live `num_steps` column in `adaptive_stats` vs the recursion's `log_step_size`), randomised rule n_lambda = 100, a_lambda = 1 | iterations at which `num_steps` changes | exactly the iterations in `installations/`; the recursion moves every period |
| A4 | Averaged rule arithmetic and lag | P1, pCN K = 5 (beta fixed), `AveragedInstallation(j0, a_zeta, weight_box)`, j0 ∈ {1, 5, 20}, a_zeta ∈ {0.6, 1}, ω ∈ {None, 5, 20}; local runner with an instrumented provider (as `tests/unit/test_installation.py`) for the norms; MPI for the posterior | ‖θ̄_j − θ̄_{j−1}‖∞ per install; fraction of clipped weights; test RMSE of the installed average vs the raw newest (T1) | norm ≤ 2ωζ_j and decays like j^{−a_zeta}; ω = 20 never binds, ω = 5 binds rarely; moment errors as A6 |
| A5 | Input clip changes the surrogate, not the target | P1, b ∈ {None, 8, 3, 2}; frozen and adaptive stages | fraction of chain states with |x_k| > b; f_pre; moment errors | b = 8: fraction 0 and results identical to None within SE; b = 2: moment errors still within tolerance (validity) while f_pre rises (efficiency) |
| A6 | Posterior correctness of every production configuration | P1 d = 20 and BA d = 10; F0, F1, A-rand, A-avg, U (defined in B1); R = 5 | moment errors vs reference per replicate; mode mass on BA | errors within 3 replicate-SE of the frozen references; no configuration biased |
| A7 | Cost formula | from A6 runs: exact evaluations per step vs w + (1−w)(1−f_pre) | ratio | 1 ± 0.01 |
| A8 | Blind-spot escape (uniform ergodicity in practice) | BA d = 10: preliminary stages started in the NEAR mode with 500 evaluations (the network knows one mode); production F0 vs F1 (w = 0.05) vs A-rand; 10 replicates | far-mode mass, first-passage iteration to the far mode | F0: mass 0 in most replicates (stuck; valid but not mixed); F1/A-rand: mass within tolerance of 0.15, first passage finite in every replicate |
| A9 | Protocol robustness | P1, 8 chains, 10^5 steps/chain under the randomised rule; stage transitions; `continue_sampling` of a run with a rule | no hang, buffers bounded, production stage starts from the last received version, seeds shift in the continuation | run completes; manifest seeds as expected |

Already pinned by the unit/MPI tests (do not repeat): closed-form posterior under the prior step and the rule
on LG with the polynomial updater; the coin replay; local = MPI chain 0; the gain arithmetic with stub evaluators.

## B. Paper set (Section 8)

Configurations of the production stage (all with the same preliminary stages and the frozen tail):

| Code | Production stage | Guarantee |
|---|---|---|
| F0 | frozen surrogate, no exact step (usual practice) | Harris ergodic (density sub-chain), no SUE |
| F1 | frozen surrogate + prior step w | uniformly ergodic (Thm 5.1), finite adaptation |
| A-rand | prior step w + `RandomisedInstallation(n_lambda, a_lambda)`, deferred sub-chain parameters | Thm 6.11 (main) |
| A-avg | prior step w + `AveragedInstallation(j0, a_zeta, weight_box)`, pCN/RW fixed, `input_clip` | Thm 6.11 (averaged rule) |
| U | prior step w, every version installed, continuous tuning (`installation=None`, adaptive proposal) | SUE but not DA: no theorem |
| U0 | U without the prior step (what the library did before) | none |

| Id | Question (paper §) | Design and range | Figure/table |
|---|---|---|---|
| B1 | Validity and efficiency of the guaranteed stage vs freezing (§6.5 (i)–(iii), §8 headline) | P1 d = 20; the six codes; HMC K ∈ {1, 5} and pCN K = 5; w = 0.05; n_lambda = 500, a_lambda = 0.75; j0 = 5, a_zeta = 0.8, ω = 10; R = 5 | moment error (box plots over replicates) and ESS/eval ± SE per code; f_pre; test RMSE at end; one table + one figure |
| B2 | Cost of the prior step (§6.5 (i)) | F1 on P1: w ∈ {0, 0.01, 0.02, 0.05, 0.1, 0.2} × surrogate quality {good: S1 3 000 evals; poor: S1 300 evals} × K ∈ {1, 10}; R = 3 | ESS/eval multiplier vs the formula (1−w)τ/(τ+w)·(1−f_pre)/(w+(1−w)(1−f_pre)); prior-step acceptance |
| B3 | Installation lag vs learning curve (§6.4) | A-rand on P1: (n_lambda, a_lambda) ∈ {100, 1 000, 10 000} × {0.5, 0.75, 1}; production budget 20 000 and 80 000; R = 3 | test RMSE of the installed vs the newest version over the stage (T1); installs per 1 000 steps; ESS/eval and moment error vs lag |
| B4 | When does continued training beat freezing (§6.5 (ii))? | P1: preliminary budget S1 ∈ {300, 1 000, 3 000, 10 000} × production budget {5 000, 20 000, 80 000}; F0, F1, A-rand, U; R = 3 | moment error and ESS/eval vs budgets; the crossover where (ii) outweighs (i) |
| B5 | Deferred vs continuous sub-chain tuning (§6.5 (iii)) | P1, HMC K = 5: A-rand vs "A-rand with continuous tuning" (same coin, `installation=None` + manual coin not available: use U with the same w) vs F1; also n_lambda small (10) to force frequent installs | step-size trajectory, acceptance after installs (mismatch), ESS/eval |
| B6 | Averaged-rule knobs (§6.2, §6.4) | P1, pCN K = 5: j0 ∈ {1, 5, 20} × a_zeta ∈ {0.6, 0.8, 1} × ω ∈ {None, 10}; R = 3 | ESS/eval, moment error, variance of the second-stage acceptance over windows (smoothing), ω binding rate |
| B7 | Dimension (§6.5, prior-step acceptance → 0) | P1-d, d ∈ {20, 50, 100}, HMC K = 5: F0, F1, A-rand, U; w = 0.05; R = 3 | prior-step acceptance vs d (log scale), ESS/eval gap A-rand vs U vs F0, moment errors |
| B8 | Multimodal posterior: the exact step as a guaranteed mode jump (§8, Discussion) | BA d ∈ {10, 30}: F0, F1, A-rand with w ∈ {0.02, 0.05, 0.1}; HMC with and without the flattened component of note 33 (τ = 0.05); initial network from both modes (fair) and from one mode (A8's start); R = 5 | far-mode mass error, crossings per 1 000 steps, ESS of the mode indicator per evaluation |
| B9 | Cost in wall time for a cheap and an expensive model (§7.1, §7.3) | from B1: c_e measured (0.2 ms) and c_e := 1 s, 60 s (extrapolated with the cost model) | ESS per second table for the six codes under three c_e |

Reporting rules (CLAUDE.md, paper §7.4): ESS from the frozen tail only; whole-stage ESS figures labelled
heuristic; moment errors over replicates are the validity measure; exact output directories cited; no
uncomputed numbers. Suggested order: T1–T3, then A1, A2, A6, A8 (two days of runs), then B1, B2, B7, B8,
and B3–B6 as the budget allows. Rough cost on P1: one production configuration with 8 chains × 25 000
evaluations ≈ 10–20 min wall time (surrogate-bound); B1 ≈ 6 codes × 3 sub-chains × 5 replicates ≈ 15–30 h,
B2 ≈ 72 runs, B3 ≈ 54 runs, B4 ≈ 144 runs, B7 ≈ 36 runs (d = 100 slower), B8 ≈ 180 runs of 3–5 min.

## C. Range-extension rule (author, 2026-10-23)

A swept range is provisional. After every sweep, check for each parameter whether the optimum (or the
crossover / the plateau the experiment looks for) lies INSIDE the grid with at least one grid point on
each side and a response that is no longer monotone at the boundary. If not, extend the grid in that
direction by the same geometric factor as the grid's own spacing (one decade for n_lambda, j0, budgets;
×2 for w and K; +25 for d) and re-run the configurations next to the boundary, until the optimum is
interior or a hard limit (budget, memory, solver) is reached and stated. Record every extension in the
study note with the reason (which boundary, which response). Triggers per experiment:

| Experiment | Swept parameter | Extend when |
|---|---|---|
| B2, B8 | w | the best ESS/eval or mode-mass error is at w = 0.01 or 0.2 |
| B3 | n_lambda, a_lambda | ESS/eval or moment error still improves at n_lambda = 100 or 10 000, or at a_lambda = 0.5 / 1 |
| B4 | preliminary and production budgets | the crossover (continued training beats freezing) is not bracketed by the budgets |
| B6 | j0, a_zeta | the best configuration is at j0 = 1 or 20, or at a_zeta = 0.6 / 1 |
| B7 | d | the prior-step acceptance or the ESS/eval gap has not reached its asymptotic regime at d = 100 (add d = 200) |
| B1, B5 | K | the best K is 1 or 5 (add 10, 20); the step-size mismatch after installs is not resolved within the tested n_lambda (add 10, 10 000) |
| A2, A4 | n_lambda, a_lambda, j0, a_zeta, omega | the predicted behaviour is confirmed only at one end of the grid; omega binds at every tested value (add 50) or never (add 2) |

The same applies to the number of replicates: if the replicate SE is wider than the effect under study,
double R before extending any range.

## D. Efficiency measures (what is reported, how it is estimated, where it is valid)

| Measure | Definition | Estimation | Valid when | Used in |
|---|---|---|---|---|
| ESS per exact evaluation (primary) | Eff_g = ESS_g / N_exact = 1/(tau_g c_step) with c_e = 1, c_v = c_grad = 0 (paper §7.3) | tau_g by the Sokal-window estimator (`emcee.autocorr.integrated_time`, c = 5, what `post_processing.Autocorrelation` already does) on the FROZEN TAIL only; N_exact = accepted + rejected + audited of the tail (`notes`); SE over replicates, not from the estimator | frozen tail: time-homogeneous, uniformly ergodic (F1/A codes) or Harris ergodic (F0) | B1–B8 |
| Replicate-variance ESS (adaptation-valid) | ESS_rep = Var_pi(g) / Var_rep(ĝ), where ĝ is the production-stage average of g over all chains of a run and Var_rep its variance over independent replicates (different seeds and collectors); corrected 2026-10-23 (an earlier version had a spurious factor n) | Var_pi(g) from the reference run; R = 5 gives a 35 % relative SE of the variance, so R = 10 for the headline comparison B1 | always (no CLT or homogeneity needed; chains of one run share a surrogate, so the unit of replication is the run, not the chain) | B1, B4, B7 (the whole production stage, adaptation included) |
| Moment error per evaluation (validity + accuracy) | RMSE over replicates of the posterior mean and sd in the 16 observed singular directions (P1) / of the far-mode mass (BA) vs the reference, as a function of cumulative exact evaluations | from `samples/` with the selection mask, reference from the 10^6-evaluation run | always | A6, B1, B3, B4, B8 |
| Test functions g | each observed singular direction (min over them is the conservative figure), the log-likelihood, and on BA the mode indicator 1{t > c}; the min-over-coordinates rewards a stuck chain, so on BA the mode indicator is the reported one | — | — | all |
| ESS per surrogate gradient / per second | c_v, c_grad = 0 replaced by measured costs; per second from stage wall time; extrapolation with c_step = [w + (1−w)(1−f_pre)] c_e + (1−w) K (c_v + n_lf c_grad) for c_e = 1 s, 60 s | wall time per stage from the run, counts from `subchain_stats`/`notes`, n_lf from the step and integration time | — | B1, B9 |
| Prior-step cost multiplier | Eff(w)/Eff(0) vs (1−w) tau/(tau + w) · (1−f_pre)/(w + (1−w)(1−f_pre)) | tau from the w = 0 frozen tail, f_pre from `subchain_stats` | frozen stage | B2 |
| Explanatory diagnostics | f_pre (pre-rejected fraction), second-stage acceptance given a move, exact-step acceptance, exact evaluations per step, surrogate test RMSE per version (`surrogate_quality.csv`), installs per 1 000 steps and lag (T1), step-size and acceptance before/after installs, crossings per 1 000 steps and first passage (BA), weight-box binding rate | existing CSVs + `installations/` | — | all |

Not used as evidence: whole-stage tau_g under adaptation (reported only as a labelled heuristic, paper
§7.4(ii)); R-hat across chains of one run (chains share the surrogate: heuristic only); min-ESS alone on
multimodal problems. Every ESS figure comes with its SE over replicates and the number of replicates.
