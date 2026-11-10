# Paper mechanisms implemented: prior-proposal step, installation rules, deferred adaptation, input clipping (2026-10-23)

Context: `pdf_reports/adaptive_nn_ergodicity_paper/main.tex` Section 6.5 ("Realisation in surrDAMH")
described the production step as implemented, but six mechanisms were planned only (audit of
2026-10-23, memory note). The author asked for all of them. Nothing is committed; branch `working_Kuba`.

## 1. What was implemented

| Mechanism | Paper | Library | Default |
|---|---|---|---|
| Exact prior-proposal step at rate w | Thm 5.1, §6.5 step (1) | `Stage(exact_step_probability=w, exact_step_proposal="prior")`; runtime class `modules/prior_proposal.PriorIndependence` (independence sampler y ~ prior in the internal space, acceptance `min{1, L(y)/L(x)}`, nothing adapts, stream `seed0 + 5`) | `"random_walk"` (S2 unchanged) |
| Randomised installation λ_n | Def 6.1, §6.5 step (4) | `Stage(installation=RandomisedInstallation(n_lambda, a_lambda))`; coin stream `seed0 + 6` (`INSTALLATION_SEED_OFFSET`), one uniform per outer iteration i ≥ 1 with λ_{i−1}; receiving decoupled from installing (newest received kept in a buffer); `installations/<stage>/rank%04d.csv` (iteration, version); `notes` + `installations, versions_received` | `None` |
| Sub-chain parameters change only at installations | §6 preamble, §6.5 step (4) | `Proposal.set_deferred_adaptation(True)` / `apply_pending_adaptation()` on `GaussRandomWalk_adaptive`, `PCN_adaptive`, `Hamiltonian[Infinite]_adaptive`, `BlockProposal`; switched on by the randomised rule for the stage proposal and the exact step's random walk; applied at every committed install | off |
| Averaged installation with gains ζ_j | Def 6.7 | `Stage(installation=AveragedInstallation(j0, a_zeta, weight_box))`; θ̄_0 = network at stage start (projected), j-th RECEIVED version gets ζ_j = min{1,(j/j0)^−a_ζ} (ζ_1 = 1 replaces the start network); refuses adaptive proposals and the adaptive exact random walk | `None` |
| Weight box Θ_ω | Lemma 6.4, Def 6.7 | `AveragedInstallation(weight_box=omega)`: model part of every received state (not the S1 error-model entries) clipped to [−ω, ω] before averaging, the stage-start one included | `None` |
| Input clipping clip_b | Lemma 6.4 | `NeuralNetworkUpdater(input_clip=b)`: `torch.clamp` before the first layer of `PyTorchMLP`, in training and prediction; `vjp`/`jacobian` get zero gradient in a clipped coordinate; checkpoint hparam | `None` |

`polyak_averaging=True` (S3) is kept as the legacy flag: its running mean keeps the stage-start
network in (weight 1/k); `AveragedInstallation(1, 1.0)` is the paper's mean of the received versions.
The two are refused together. `Auto` is unchanged (still S2 random walk + legacy Polyak).

Section 6.5 of the paper was rewritten to this interface (option names in a footnote; the collector
buffer; the coin drawn once per step; the mass fixed per stage, the leapfrog count following the
step at installs; the installation record). Sections 6.1–6.4 needed no change: Def 6.7's θ̄_0 is the
stage-start network, as in the code after the gain-index fix below.

Tooling added the same day for the experiments (note 38, T1): `Evaluator.version` (the collector's
publication index, or the local manager's count), the third column `collector_version` of
`installations/`, and the new `versions/<stage>/rank%04d.csv` with one row per received evaluator
(both only when `Stage.installation` is set). Checked on a local run with the network updater: the
randomised rule installed 14 of 30 received versions and every row carries the publication index.

## 2. Decisions taken by the manager (author to revisit)

- The install coin is drawn at the top of every outer iteration i ≥ 1 of a DAMH stage, exact-step
  iterations included, so the kernel of every step is the mixture (an installation never depends on
  the step's own kernel choice). An install before an exact-step iteration leaves the chain state's
  surrogate terms stale until the next sub-chain re-scores them (flag `_surrogate_rescore_pending`);
  the exact step's acceptance never uses them.
- Gain index: the first implementation used ζ_{k} with k counting the stage-start evaluator as 1
  (paper's version j got ζ_{j+1}). Changed by the manager to ζ_j, j = received versions, so the code
  follows Def 6.7 exactly and the "stale first evaluator" caveat of the implementation map (§4) does
  not apply to the new rule.
- `AveragedInstallation` with a Hamiltonian proposal is accepted with a `describe()` note (the
  paper's averaged rule covers proposal-density sub-chains only).
- Manifest: `installation_seed` is listed per rank/stage, and the stage dicts carry
  `exact_step_proposal` and `installation` for every run (JSON keys added; CSVs unchanged).
- The S2 exact step with `exact_step_proposal="random_walk"` under the randomised rule defers the
  exact walk's adaptation too (ψ includes it); under the averaged rule it must be fixed.

## 3. Verification (observed)

| Command | Result |
|---|---|
| `./run_tests.sh unit` (after integration + gain fix) | 900 passed, 1 skipped, 78 deselected, 134 s |
| `/dolfinx-env/bin/python3 -m pytest tests/mpi -q -m mpi -k "s2 or paper_mechanisms or mixture or surrogate" -rxX` | 19 passed, 193 s |
| `tests/unit/test_prior_step.py`, `test_installation.py` (57 tests), `test_deferred_adaptation.py` (22), `test_input_clip.py` (10) | included above |
| Byte identity at defaults (agent 1's script, 4 configurations, 47 output files vs an untouched copy of the package) | identical except `run_manifest.json` (new keys) |
| Posterior checks: prior step p = 1 and 0.5 (local, closed form), randomised rule with the polynomial updater (local), MPI 2 chains + collector with prior step 0.2 + `RandomisedInstallation(50, 0.5)` | all within the test tolerances (4 SE / R-hat) |

Not verified: a run on the neural-network updater under the randomised rule with a real collector
where versions arrive faster than the coin rate (in the MPI test every received version was
installed because versions were rare); the averaged rule and the input clip on a real NN run beyond
unit tests; efficiency (that is Section 8's job). `paper_mechanisms` MPI test runs about 1 minute.

## 4. Where to look

Code: `surrDAMH/installation.py`, `surrDAMH/modules/prior_proposal.py`, `surrDAMH/stages.py`
(`exact_step_proposal`, `installation`, `_validate_installation`), `surrDAMH/modules/algorithms.py`
(`_receive_pending_evaluator`, `_randomised_installation_step`, `_polyak_candidate`),
`surrDAMH/modules/proposals.py` (deferred API), `surrDAMH/surrogates/torch_perceptron_minibatches.py`
(`input_clip`), `surrDAMH/modules/seeds.py` (+6). Docs: `docs/stages.md` ("Installation rules",
"Deferred adaptation", field rows), `docs/outputs.md` (`installations/`, notes columns, seed),
`docs/writing_a_surrogate.md`. Tests: `tests/unit/test_prior_step.py`, `test_installation.py`,
`test_deferred_adaptation.py`, `test_input_clip.py`, `tests/mpi/test_mpi_paper_mechanisms.py`.
