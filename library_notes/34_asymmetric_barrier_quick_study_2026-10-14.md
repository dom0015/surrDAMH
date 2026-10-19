# 34 — Quick study: tempered momentum vs flattened likelihood on the asymmetric, likelihood-made barrier (2026-10-14)

**Dated record, 30 runs in 14 minutes (08:05–08:19), author's time box of 30 min.** Raw runs, scripts, references:
`toy_examples/tmp_asym_study_2026-10-14/` (`runs/<name>/` + `.log`, `jobs/*.jsonl`, `refs/ref_AB{10,30}-10.json`,
`half_steps.json`, `logs/{time_log,batch_progress}.txt`, `scripts/` = study 33's with the `AB` problems added). Untracked.
Purpose: the test review item 23 asked for — note 33's win of the flattened component on the point-symmetric M-family
could come from the antipodal prior rotation; this problem has no such symmetry and a barrier made by the likelihood.

**Read first.** 3 repetitions at d = 10, 2 at d = 30; one setting of the tempered component (study 31's best:
f 0.35, β 0.001, T_t = 3T, step pinned at ½ the carried step: 0.0856 at d = 10, 0.0698 at d = 30) against three
flattened settings (τ ∈ {0.05, 0.2, 0.5}, T_t = T, carried step); p = 0.3 for every component; layout, budget
(4 000 evaluations per chain, measured stage 2 000), network, error model and Polyak as study 33. Mixing verdict =
split R̂ of the mode indicator < 1.05 and every chain crossed. DAMH-SMU runs are not reproducible. The posterior is
the prior in the d − 1 directions perpendicular to `e`, so the prior draw is effectively a 1-D proposal here — see §3.

## 1. Problem
`toy_examples/solver_examples/bimodal_asym.py` (2026-10-14): prior N(0, I_d), `G(u) = (u·e − c)²`, `e ∝ (1, …, d)`,
`y = r²`, c = 0.75, r = 1.25, σ = 0.3233 (`sigma_for_depth(10)`): modes at t = −0.49 (mass 0.859) and t = 1.97
(mass 0.141), saddle t = 0.78 with depths 11.8 / 10.0 nats of which ≈ 11.6 are likelihood misfit; the antipode of a
mode is 161 nats below it. Exact 1-D quadrature reference. d = 10 (network (32, 32)) and d = 30 ((64, 64)).

## 2. Results (crossings per 1 000 measured-stage iterations, mean [min, max]; mass of the far mode, ref 0.141)

| d = 10 | cross/1 000 | mixed | mass_right per run | R̂_mode | oob | mode ESS | ESS/eval | wall s |
|---|---|---|---|---|---|---|---|---|
| plain H | 0.06 [0, 0.17] | 0/3 | 0.249 0.231 0.000 | ∞ 2.29 — | 0.006 | 21 | 1.25 | 12 |
| prior draw p 0.2 | 7.9 [7.2, 9.1] | 3/3 | 0.120 0.133 0.121 | 1.01–1.03 | 0.006 | 458 | 0.098 | 13 |
| tempered ref (½ step) | **10.3** [9.2, 11.7] | 3/3 | 0.139 0.141 0.122 | 1.01–1.02 | 0.030 | **719** | 0.107 | 32 |
| flattened τ 0.05 | 9.6 [5.2, 12.2] | 3/3 | 0.141 0.129 0.122 | 1.02 | 0.004 | 657 | 0.105 | 15 |
| flattened τ 0.2 | 3.3 [3.0, 3.7] | 1/3 | 0.141 0.154 0.109 | 1.04–1.08 | 0.006 | 207 | 0.058 | 15 |
| flattened τ 0.5 | 0.2 [0.05, 0.4] | 0/3 | 0.123 0.009 0.244 | 1.1–2.3 | 0.004 | 27 | 0.96 | 13 |

| d = 30 | cross/1 000 | mixed | mass_right per run | R̂_mode | oob | mode ESS | ESS/eval | wall s |
|---|---|---|---|---|---|---|---|---|
| plain H | 0.06 [0, 0.11] | 0/2 | 0.019 0.125 | 1.20 ∞ | 0.004 | 32 | 1.31 | 13 |
| prior draw p 0.2 | 8.2 [8.0, 8.4] | 2/2 | 0.118 0.159 | 1.01 | 0.003 | 506 | 0.242 | 13 |
| tempered ref (½ step) | 3.9 [3.4, 4.4] | 2/2 | 0.175 0.111 | 1.05 1.03 | 0.043 | 254 | 0.208 | 39 |
| flattened τ 0.05 | **9.1** [8.9, 9.3] | 2/2 | 0.130 0.140 | 1.01 1.02 | 0.004 | **752** | 0.270 | 18 |
| flattened τ 0.2 | 2.9 [2.7, 3.2] | 0/2 | 0.102 0.109 | 1.06 1.07 | 0.003 | 172 | 0.306 | 16 |
| flattened τ 0.5 | 0.14 | 0/2 | 0.086 0.153 | 1.7–2.4 | 0.004 | 19 | 1.00 | 17 |

All 30 runs rc = 0, no time limit hit, no non-finite proposals. `ESS/eval` is the min-over-parameters ESS, which
rewards non-mixing (plain H: 1.3), listed for cost only.

## 3. Answers
* **The note-33 result is not an artefact of the M-family's symmetry.** On a likelihood-made, asymmetric barrier with
  unequal modes, the flattened component at τ = 0.05 mixes at both dimensions (5/5 runs), reproduces the 0.141
  far-mode mass, and its crossing rate does not fall from d = 10 to d = 30 (9.6 → 9.1).
* **The tempered-momentum component works at d = 10 (best crossings, 10.3) but loses 2.6× by d = 30** (3.9, with the
  far-mode mass 0.175 / 0.111 and R̂ at the 1.05 threshold), at 2–3× the wall time and with 3–4 % out-of-box
  proposals even at the halved step. Same trend as note 33 (20× loss from d = 3 to 30 on the M-family).
* **τ must be small**: 0.2 mixes in 1 of 5 runs, 0.5 never — as in note 33 the flattened barrier `τ·Δ_L` must be
  below a few nats.
* **The prior draw did not collapse here** (7.9 → 8.2) because the posterior equals the prior in the d − 1 perpendicular
  directions, so its acceptance is a 1-D matter; on a problem whose likelihood informs many directions it would
  (note 33: 11.8 → 0.05 on the M-family, where the modes sit at prior radius √d). This family is therefore a fair test
  of the two Hamiltonian mechanisms against each other, not of them against the prior draw.
* **Cost per crossing (d = 30, mean)**: flattened 0.05 ≈ 2 s per 1 000 iterations' worth of wall time at 9 crossings;
  tempered ≈ 39 s at 4 crossings — about 10× more seconds per crossing.

## 4. Recommendation
Keep the flattened-likelihood component as the crossing mechanism (τ = 0.05, T_t = T, carried step, p = 0.3 remains
the C6 placeholder; now supported on three problem families, two of them asymmetric). The tempered-momentum
component has not won on any problem at d ≥ 30, needs a hand-set half step and costs 2–3× the wall time; on the
evidence of studies 31, 33 and 34 it can be removed (decision for the author, item 23). What has NOT been tested for
either: a likelihood that informs many directions with a barrier between modes (the realistic case), and d > 30.
