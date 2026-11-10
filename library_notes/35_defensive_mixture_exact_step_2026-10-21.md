# Defensive-mixture independence step as the exact safety component (design note, nothing implemented)

Date: 2026-10-21. Status: idea recorded for later; the author does not plan to implement it soon.
Theory context: `pdf_reports/adaptive_nn_ergodicity_paper/main.tex`, Section 5 (prior-proposal step,
Theorem 5.2) and the Discussion section (routes to containment).

## 1. The idea

Replace the exact prior-proposal step `PCN(beta=1)` (proposal density `g = p`, the prior) by an
exact **independence sampler with the defensive-mixture proposal**

    g = (1 - a) p + a h,        a in (0, 1),

where `h` is any sampleable, normalised density approximating the posterior (Laplace fit, Gaussian
mixture at the modes found in a pilot run, normalising flow, ...). One MH step: draw `y ~ g`,
accept with the exact ratio `pi(y) g(x) / (pi(x) g(y))`, i.e. one exact evaluation per step,
no surrogate screening. Mixed into every kernel of the adaptive family with a fixed rate `w`
(`Mixture([DAMH(...), <this step>], [1 - w, w])`), it keeps the containment guarantee of the
prior step:

    pi / g <= e^{B+} / ((1 - a) Z)     for every h,

so the mixture family is simultaneously uniformly ergodic with constant `>= w (1 - a) Z e^{-B+}`
(Theorem 5.2 with `delta` multiplied by `1 - a`). No condition on the forward map, the surrogate
or the posterior tails; the only condition on `h` is that it is a proper density that can be
sampled and evaluated **with its normalising constant relative to p** (the mixture weights enter
the acceptance ratio).

## 2. Why do it

The prior-proposal step accepts with probability of the order `Z e^{-B+}` (prior mass of the
well-fitting region): for informative data essentially never. Its exact evaluations are pure
insurance, and it cannot be surrogate-screened without losing the guarantee (the screening ratio
brings the surrogate into the minorisation constant). The defensive mixture is the only lever that
makes those evaluations useful:

- where `h` approximates `pi` within a constant factor, `inf g / pi` rises from `Z e^{-B+}` towards
  `a`, so the exact step actually moves the chain;
- with `h` a mixture placed at the modes of a pilot run, the exact step makes **mode jumps with
  provable validity**, at the same cost per step as the prior step (one exact evaluation) and
  without modifying the Hamiltonian/surrogate dynamics (alternative to the tempered/flattened
  components of notes 30-34);
- if `h` is adapted during the run (`h_gamma` fitted to the surrogate posterior), the prior
  component keeps the bound uniform in `gamma` automatically; `h_gamma` is then part of the
  adapted index and must satisfy diminishing adaptation (freeze after the pilot run, or Polyak
  average its parameters).

Limits: independence samplers degrade with dimension unless `h` is very close to `pi`
(acceptance governed by the spread of `log(pi/h)`); for a non-Gaussian posterior at d = 30 a
Laplace `h` may give acceptance near zero, i.e. no gain over the prior step. Expected to pay off
in low/moderate `d` and on multimodal posteriors with known modes.

## 3. This is NOT the three-kernel mixture

Two valid constructions that are easy to confuse:

| | three-kernel mixture | defensive-mixture step |
|---|---|---|
| kernel | `(1-w_p-w_h) P_gamma + w_p P_p + w_h P_h`, `P_p` = prior-proposal step, `P_h` = independence sampler with proposal `h`, each with **its own** acceptance | `(1-w) P_gamma + w P_g`, **one** independence sampler with proposal `g = (1-a)p + a h` and **one** acceptance using `g` |
| containment | from `P_p` alone; `h` contributes nothing to the constant | from the prior component of `g`; `h` can raise the constant up to ~`a` |
| conditions on `h` | none beyond pi-invariance of `P_h` (automatic); `h` needed only up to a constant | `h` must be normalised relative to `p` |
| behaviour where `h(x) ~ 0` (region the pilot run missed) | `P_h` almost never accepts there (`h(x)/h(y)` tiny); the chain leaves via `P_gamma` | still accepts, since `g(x) >= (1-a) p(x)` keeps the ratio bounded (the classical defensive-mixture argument of importance sampling, Hesterberg 1995) |
| library status | **can be built today** from C2 `Mixture` + `PCN(beta=1)` + a new independence-proposal spec with density `h` | needs a new proposal spec with a mixture density |

For the paper the three-kernel form is the cleaner story (p for the theorem, h for mode jumps,
`P_gamma` for everything else). The defensive form is the one with the better constant and the
better behaviour off the support of `h`. Record which one an experiment uses.

## 4. What an implementation would need (estimate, not a plan)

1. A proposal spec `Independence(density)` for an exact MH step with a state-independent proposal:
   `density.rvs()`, `density.logpdf(x)`; acceptance `ell(y) - ell(x) + log p(y) - log p(x) + logpdf(x) - logpdf(y)`
   in internal coordinates. `PCN(beta=1)` is the special case `density = prior` (ratio reduces to
   the likelihood ratio). Non-adaptive; refuses `adaptive=True`.
2. A density object `DefensiveMixture(prior, h, a)` with `rvs` (coin + component draw) and
   `logpdf = logsumexp(log(1-a) + logp, log a + logpdf_h)`; `h` from the existing
   `distributions/` classes (Normal, GaussianMixture) or a fit of the pilot-run samples.
3. The C2 `Mixture` stage carries the rate `w`; the `component` column of `subchain_stats`
   attributes acceptances; `exact-step rows` convention of note 22(b) applies.
4. Tests: invariance on a Gaussian toy (closed form), acceptance equals the prior step when
   `a = 0`, mode-jump count on `bimodal_asym.py` vs `PCN(beta=1)` and vs the flattened component.
   Roughly 150-250 library lines plus tests.

Open decisions: whether `h` may adapt (then DA mechanism needed, cf. note 20); choice of `a`
(0.1-0.5; `1 - a` is the worst-case loss in the constant); fitting `h` from the surrogate
posterior vs from exact pilot samples.

## 5. References

Mengersen & Tweedie (1996) Thm 2.1 (independence sampler uniformly ergodic iff `pi/g` bounded);
Hesterberg (1995), Technometrics 37(2), 185-194 (defensive mixtures, from memory); the paper
draft Section 5 and Discussion; notes 24 (exact step in DAMH), 30-34 (barrier-crossing components).
