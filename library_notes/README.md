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
| What the literature prescribes for removing guessed parameters (RAM / shrinkage-AM + Robbins–Monro, sub-chain adaptation in DAMH, dual-averaging step size at fixed T, mass = Σ̂⁻¹, adaptive pCN, DA diagnostics), with references and prototype evidence | `16_adaptivity_options_research_2026-09-20.md` (dated record; reviews + prototypes in `toy_examples/out_adaptivity_research_2026-09-20/`) |

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
