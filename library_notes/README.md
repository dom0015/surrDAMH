# library_notes — entry point

This folder documents a completed medium refactor of `surrDAMH` (originally a read-only review
started 2026-09-12; the refactor it recommended was then implemented, committed by the author
across `175059e`..`9efb54b` on `working_Kuba`, and re-verified against the current code on
2026-09-18). Actualised and shortened 2026-09-18 — see "History of this folder" below.

## Current state (2026-09-18)

- **The package imports, all suites pass, 11 of 12 maintained examples run.** `sampling_TSX.py`
  fails at import (`ModuleNotFoundError: pyvista`, before even reaching its separately-missing
  mesh files) — a real, currently-unfixed defect in a maintained example; see the session
  report for the exact error, or re-run it yourself.
- **Everything in the medium-refactor plan (`09_improvement_plan.md`, WS0–WS11) is either
  DONE, decided-and-deferred, or explicitly still open** — no workstream is silently abandoned.
  Two statuses were corrected on 2026-09-18 after independent re-verification: WS2 (standalone
  runner) is PARTIAL, not DONE (`Configuration` still imports `mpi4py`); several per-workstream
  claims about which validation tests exist were wrong (only V1–V3d exist, not "V1-V3,V5-V8").
- **DAMH/DAMH-SMU correctness is validated**, not just argued: closed-form Gaussian-toy checks
  (V1–V3d), MPI posterior-vs-closed-form checks across 9 configurations (B1–B9), and a longer
  GRF-diffusion campaign (0 of 120 comparisons beyond 3 SE) — see `10_manual_review_notes.md`
  §5/§5b/§5c/§5d/§5e.
- **Two real, previously-mis-stated defects were found and corrected in this pass** (not fixed
  in code — see `06_findings_consolidated.md` 2.7 and 2.10, and `10` §8(b)): a silent
  report-content gap in pool mode, and an MPI request left un-`Wait()`-ed on sampler shutdown.

## Where things live

| Question | File |
|---|---|
| What's the architecture, beyond what `docs/` covers? | `00_overview.md` |
| Is finding X fixed? | `06_findings_consolidated.md` (status table) |
| Is test Y written? | `07_testing_plan.md` (status table) |
| Was safe-change group Z applied? | `08_safe_changes_plan.md` (status table) |
| What was decided, and why? What's still open by workstream? | `09_improvement_plan.md` §3 (decisions), §4 (future work), §1 (per-workstream status) |
| What's the evidence for behaviour change N? What do the validation numbers say? | `10_manual_review_notes.md` §2 (evidence), §5 (validation), §8 (**the** authoritative open list) |
| Output format v2 / Evaluator contract, as designed | `11_output_format_v2_spec.md`, `12_evaluator_contract_spec.md` (dated design records) |
| What dead code exists and why it's kept | `13_dead_code_report.md` (reference only — author decided not to delete dead code) |
| The GRF long-run validation campaign | `14_grf_validation_2026-09-17.md` (dated record) |
| How usable is the adaptive proposal; which knobs must be guessed; what adaptivity to add | `15_adaptivity_study_2026-09-18.md` (dated record; raw runs in `toy_examples/out_adaptivity_study_2026-09-18/`, 4 GB, off-limits) |
| Why pCN loses on the corrected GRF problem: implementation reviewed (correct), auto-tuning within 20 % of the fixed optima, the gap is the 12× posterior anisotropy (synthetic Gaussian reproduces 16×); DAMH sub-chain adaptation on the outer acceptance is a real defect (fixed β = 0.1 with a 200-step sub-chain 2.3× the adaptive); cost per effective sample incl. surrogate values/gradients: Hamiltonian DAMH 8–10× cheaper than the best pCN DAMH | `21_pcn_investigation_2026-09-29.md` (dated record; runs `P_*`/`Q_*` and `report_pcn.html` in `toy_examples/tmp_efficiency_study_2026-09-28/`) |
| **GRF solver bug (observations extrapolated from mesh cell 0, rank-2 forward map — notes 14/18 sampled that artefact; fixed 2026-09-28)** and the efficiency study on the corrected problem: adaptive random walk 12x adaptive pCN in MH; DAMH + Hamiltonian (30 leapfrog steps) on the network surrogate 177x MH; random-walk sub-chains 34–44x; pCN sub-chains ≤ 5x; cost model and slow-solver check; P2 (d = 40) confirmation | `19_efficiency_study_2026-09-28.md` (dated record; runs + figures in `toy_examples/tmp_efficiency_study_2026-09-28/`, readable; `report.html` there) |
| Which proposal / surrogate / sub-chain length pays off on the GRF example (adaptive pCN 8.6x the fixed-beta example; DAMH with a 5-step pCN sub-chain ~10x MH; k-d tree useless at 20 parameters); deadlock and frozen-stage trap found on the way | `18_sampling_schemes_grf_2026-09-22.md` (dated record; runs + figures in `toy_examples/out_scheme_study_2026-09-22/`, 4.3 GB, off-limits) |
| Design of the `Stage(proposal=RandomWalk()...)` interface, implemented 2026-09-21 | `17_stage_proposal_objects_design_2026-09-21.md` |
| What the literature prescribes for removing guessed parameters (RAM / shrinkage-AM + Robbins–Monro, sub-chain adaptation in DAMH, dual-averaging step size at fixed T, mass = Σ̂⁻¹, adaptive pCN, DA diagnostics), with references and prototype evidence | `16_adaptivity_options_research_2026-09-20.md` (dated record; reviews + prototypes in `toy_examples/out_adaptivity_research_2026-09-20/`) |
| Is DAMH-SMU (surrogate retrained while sampling) a valid MCMC algorithm? Assumptions A0–A6, full proof (reversibility, Lipschitz-in-surrogate kernel bound, uniform Doeblin, coupling), and what the NN surrogate needs (bounded outputs/weights, Polyak-averaged publication or decaying budget or randomised installation) | `18_damh_smu_validity_proof_2026-09-22.md` (theory only; refereed) |
| How to make the on-the-fly NN surrogate satisfy diminishing adaptation *per stage* without touching the collector (sampler-side Polyak averaging of received weights, or randomised installation), and whether ergodicity then holds (TV convergence + strong LLN on a compact/truncated prior; on ℝ^d only under an unproven containment); literature check (Sherlock et al. 2017 is the right reference, Laitinen–Vihola 2026 gives the SLLN) and a standalone toy experiment | `20_diminishing_adaptation_nn_2026-09-28.md` (dated record; literature review, scripts and runs in `out_diminishing_adaptation_2026-09-28/`, git-ignored, off-limits) |
| Can the first network be blind exactly where the posterior is high, so that proposals there are pre-rejected and the exact model is never evaluated there (the surrogate blind spot)? Mechanism, the optimistic/pessimistic asymmetry, why a Hamiltonian proposal on the surrogate makes it worse, literature (Christen–Fox, Conrad et al., Kaipio–Somersalo, Cui et al.), and proposals P1–P7 (audit of pre-rejected proposals, mixture kernel, global/local approximation error model, diagnostics) | `22_surrogate_blind_spot_2026-10-06.md` (design note, nothing implemented; demo `toy_examples/hmc_surrogate_error_model_2d.ipynb`) |
| How hard is a `Mixture(proposals, probabilities)` proposal (mixture of kernels)? Verdict: moderate and additive (~450–600 lines, half tests/docs); `BlockProposal` already has the mechanics (and an unenforced `Block` with two full groups is a fixed 50/50 mixture today); all four specs mixable; adaptation routed to the proposing component; open decisions: carry-over keying for two adaptive components of one type, per-component `adaptive_stats` layout | `23_mixture_proposal_feasibility_2026-10-06.md` (design note, nothing implemented) |
| Can one stage mix the library's step kernels (exact MH incl. gradient-driven, DAMH with any proposal/sub-chain, surrogate-only)? All `π`-invariant kernels are mixable, surrogate-only MH is not (wrong invariant law); proposed interface `Stage(algorithm=kernels.Mixture([DAMH(...), MH(...)], probabilities))`; ~750–900 lines general, ~250–300 for the P2 special case; one trap (stale surrogate terms on `current` after an exact move); cost `p·r/(1−r)`; the exact component is the only remedy acting in a frozen stage and removes A4 from the Doeblin bound; build with P1 | `24_p2_exact_step_in_damh_feasibility_2026-10-06.md` (design note, nothing implemented) |
| Robust-by-default roadmap: two exact simple modes (`Auto(budget, mode="robust"|"fast")`), shared safety components S1–S8 (global error model, exact safety kernel + audit, Polyak averaging, surrogate hardening, verdict diagnostics, sub-chain adaptation fix, manifest reproducibility, GaussianMixture fix), automatic layout, validation plan, phases, decisions D1–D6; includes the 2026-10-08 sweep of all open items in notes 00–24 (absorbed vs backlog, forgotten ones marked) | `25_robust_by_default_roadmap_2026-10-08.md` (plan, nothing implemented) |
| Hamiltonian proposal in DAMH, the chosen plan: dual-averaged `eps`; integration time `T` per stage estimated from a U-turn diagnostic (median, pooled across chains; NUTS/ChEES rejected); a fresh `L ~ Uniform{1..L_max}` for every trajectory (constant per sub-chain reproduces the resonance); deterministic sub-chain length `K*` from measured `c_exact / c_traj` (`argmax (1 - rho^K)/(R + K)`, `K = 1` in robust mode, user may pin); full momentum refresh (persistent momentum valid only with flip, rejected); random `K` not needed; 2-D check: fixed half-period `T` is period-2, state-dependent stop drifts; §10 (2026-10-09): robust mass estimation (identity + low-rank correction, prior shrinkage, Marchenko–Pastur cut, clipping, within-chain pooling, fallbacks, decisions MM1–MM3) | `26_hamiltonian_proposal_plan_2026-10-08.md` (plan, nothing implemented) |

For the user-facing contract (config fields, output format, how to run, how to write a solver
or surrogate) start at `docs/README.md` instead — it's maintained going forward, this folder is
a review/decision trail.

## Open items (as of 2026-09-18)

Full split into needs-a-decision vs decision-free-but-not-done: `10_manual_review_notes.md` §8.
Headline: nothing left is a correctness bug with an unknown fix — everything open is either a
deliberately deferred theory question (`HamiltonianInfinite` parametrisation; DAMH's
adaptive-target-rate semantics — `state_dependent_approximation` left this list on 2026-09-18,
when the theory check came out negative and the feature was removed, finding 1.1 / decision 26),
a performance/hardening item explicitly out of this refactor's scope (MPI buffer
size, busy-wait loops, RBF/polynomial numerical hardening), or genuinely small and unscheduled
(the pool-mode report-content gap, the sampler-shutdown MPI request, the progress-print line).

## History of this folder

Originally 15 files (~5060 lines): five per-subsystem read-only reviews (`01`–`05`, deleted
2026-09-18 — their ~100 findings are fully absorbed into `06`), a findings/testing/safe-changes
triad (`06`–`08`), the improvement plan and its running review log (`09`, `10`), two
implemented design specs (`11`, `12`), a dead-code inventory (`13`), and a validation-campaign
report (`14`). The 2026-09-18 pass: collapsed `00`–`05` into one short architecture note,
reduced `06`–`08` to status tables, corrected several stale/wrong status claims found while
re-verifying against the current code (see the ⚠-marked entries in `06`/`09`/`10`), compacted
`10` into an evidence log plus the open list, and marked `11`/`12`/`14` as dated records and
`13` as reference-only. Before/after line counts are in the session's report to the author.
