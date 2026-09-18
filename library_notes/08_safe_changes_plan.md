# Plan: changes that are safe to make now, without further tests

Companion to `06_findings_consolidated.md`. Scope: the smallest set of edits that (a) remove
certain crashes, (b) turn silent hangs/stalls into immediate errors, (c) fix bugs on paths no
current script reaches, and (d) remove dead code — **without changing the posterior, acceptance
rates, seeds, defaults or output formats of any configuration that works today**. Everything that
could change results is listed separately in §G (needs a decision) or §H (deferred until tests).

Safety argument used throughout: a change is "safe now" if, for every configuration that currently
runs to completion, the sequence of random draws, accepted samples and written files is bit-for-bit
identical after the change. Line numbers refer to the working tree on 2026-09-12.

Reading-only caveat: nothing here was executed. After applying §A, the single cheapest sanity check
is `python -c "import surrDAMH, surrDAMH.process_SAMPLER"`; it needs no MPI run.

---

## A. Import-time and start-up crashes (do first, ~10 lines)

| # | File / lines | Change | Why safe |
|---|---|---|---|
| A1 | `surrDAMH/core.py:152-195` | Delete the whole `temptemptemp` method. | No callers (grep: only its definition). Its body is `return`. Removing it removes the `NameError` on `Iterable`/`Any`. Do **not** instead add the imports and keep the method — it is dead. |
| A2 | `surrDAMH/process_SAMPLER.py:12` | Delete `from torch.mtia import snapshot`. | `snapshot` is never referenced in the file (grep). Only effect: torch is no longer imported by the sampler module. Surrogate modules still import torch where they need it. |
| A3 | `surrDAMH/core.py:123-124` | Replace<br>`self.surrogate_test_data = self.surrogate_test_data.as_surrogate_test_data(prior=self.prior, likelihood=self.likelihood)`<br>with<br>`td = self.surrogate_test_data`<br>`if td.log_posterior is None or td.weights is None:`<br>`    td.compute_log_posterior_and_weights(self.prior, self.likelihood)`<br>`self.surrogate_test_data = td.as_surrogate_test_data()` | The branch is only entered for a `TestData` instance, which today raises `TypeError` unconditionally. All current examples pass tuples and never enter it. `compute_log_posterior_and_weights` is the method `TestData.generate` already calls (`modules/test_data.py:54`), so the semantics match the intended workflow, and a `reuse()`-d object gets its weights recomputed instead of raising `ValueError` at `:100`. |

## B. Fail fast instead of hanging or stalling silently (no effect on successful runs)

Each guard raises on a configuration that today deadlocks, crashes later, or runs a dead chain.
A configuration that completes today never triggers any of them.

| # | File / lines | Change | Why safe |
|---|---|---|---|
| B1 | `surrDAMH/process_COLLECTOR.py:235` | Change `if no_snapshots_used > 0:` to unconditional polling: call `comm.sampler_stops()` every loop iteration (keep the inner `terminate(None)` / `terminate(evaluator_instance)` logic unchanged). | Today the stop signal is simply not read until a surrogate exists; when one exists the code path is identical. Reading it earlier only changes behaviour in runs that currently hang forever (`06` item 2.1). Both `terminate` branches already handle `evaluator_instance is None` (they send `None`; the sampler side treats `None` as "no new evaluator", `communication.py:75-77`). Collector-side only; no sampler-side message change, so the CLAUDE.md "matching changes" rule is satisfied. |
| B2 | `surrDAMH/process_SAMPLER.py`, before the stage loop (after `:71`) | Add: if `list_of_stages[0].algorithm_type == "DAMH"` or its `proposal_type` starts with `"Hamiltonian"`, and `conf.use_collector` is True, and `conf.min_snapshots_initial > 0`, print a **warning** on rank 0 that the first stage will block until the collector has a surrogate, which requires `initial_snapshots` or a pretrained updater. | Print only. The sampler cannot know whether the collector is pretrained, so a hard error would be wrong; the warning converts a silent hang into a diagnosable one. |
| B3 | `surrDAMH/process_SAMPLER.py:84-105` (Hamiltonian branches) | After constructing a Hamiltonian proposal add `assert conf.use_surrogate_gradients, "Hamiltonian proposals need use_surrogate_gradients=True (it may have been disabled by SamplingFramework, see warnings)"`. | Today this configuration reaches `_leapfrog` with no gradient functions and dies with `AttributeError` after the first stage starts (`06` item 2.9). Successful runs have the flag True. |
| B4 | `surrDAMH/modules/algorithms.py:120-123` (`_prepare_run`) | After evaluating the initial sample, add `if not np.isfinite(self.current.log_likelihood): raise RuntimeError(f"rank {self.rank_world}: initial sample has non-finite log-likelihood (solver_tag={self.current.solver_tag}); the chain cannot move")`. | Today such a chain rejects every proposal for the whole stage with no message (`06` item 1.7). A finite initial log-likelihood is required for any run that produces samples. Note the check must run on the **exact** log-likelihood only (not `_approx`), and only when observations were just evaluated (`self.current.observations is None` branch), so a carried-over sample from a previous stage is not re-checked. |
| B5 | `surrDAMH/modules/algorithms.py:95-100` (`AlgorithmBase.__init__`) | If `stage.artificial_acceptance_multiplicator != 1.0`: `warnings.warn(...)` once per stage and append the value to the `notes` row in `_finalize_run` (`:229-232`) as a **new trailing column** `acceptance_multiplicator`. | Warning only. The extra `notes` column is appended at the end; `Samples.load_notes` reads by header name (note 04), so existing readers are unaffected. If you prefer zero format change, keep only the warning. |
| B6 | `surrDAMH/process_SAMPLER.py:152-163` | Add a final `else: raise ValueError(f"unknown algorithm_type {stage.algorithm_type!r}")`. | Today an unknown value produces `NameError: alg_class` at `:166`. |
| B7 | `surrDAMH/modules/proposals.py:95-100` (`PCN.__init__`) | `assert 0 < beta <= 1, "pCN beta must be in (0, 1]"`. | `beta > 1` gives `sqrt` of a negative → NaN proposals today. All working runs satisfy the assertion. |
| B8 | `surrDAMH/configuration.py:37` and `__post_init__` | Keep the default `state_dependent_approximation = False` (decision 1, 2026-09-13). In `__post_init__`, if it is `True`: `warnings.warn("state_dependent_approximation=True is NOT verified (incorrect for subchain_max_length > 1, see library_notes/06 item 1.1); do not use it unless you know what you are doing", RuntimeWarning)`. Add the same sentence to the field comment. | Warning only; the algorithm is unchanged. Runs with the default are unaffected. |

## C. Bug fixes on paths that no current script reaches, or that only matter in error cases

| # | File / lines | Change | Why safe |
|---|---|---|---|
| C1 | `surrDAMH/process_SAMPLER.py:196` | `stages_will_use_surrogate = any(following_DAMH) or any(following_onlySurr) or any(following_hamiltonian)` where `following_hamiltonian = [list_of_stages[j].proposal_type in ("Hamiltonian", "HamiltonianInfinite") for j in range(i+1, no_stages)]`; simplify `:198` to `if stages_will_use_surrogate: pass`. | For stage lists where a later stage is DAMH, the old expression already evaluated to a non-empty list containing `True` → identical result. Only lists whose later surrogate users are `use_only_surrogate` or Hamiltonian-MH stages change, and those currently terminate the collector early (`06` item 2.6). |
| C2 | `surrDAMH/modules/algorithms.py:187` | `if not self.current.solver_tag < 0:` → `if not self.proposed.solver_tag < 0:`. | Behaviour differs only when a solver returns a negative tag. With well-behaved solvers both tags are always 0. In the error case this stops zero-filled observations (`process_CHILD.py:55`) from entering the surrogate training set. **Confirm intent with the author before merging** (open question 2 in note 01); the fix is low-risk but the original may have been deliberate. |
| C3 | `surrDAMH/modules/algorithms.py:194-195` | Return `-np.inf, self.prior.logpdf(parameters)` instead of `-np.inf, -np.inf` on solver failure. | Only reached for negative tags. The proposal is still rejected (`-inf` likelihood ratio); the prior term becomes finite, which is what the `prior_part` arithmetic expects. Combined with B4 this removes the NaN path. |
| C4 | `surrDAMH/surrogates/nearest_kdtree.py:33` | `weights = 1 / np.maximum(distances, 1e-300)` **plus** an exact-hit branch: where `distances[:, 0] == 0`, return `self.obs[indices[:, 0]]` for that row. | Only queries at zero distance change (today NaN, `06` item 3.2). `k == 1` path untouched. Nothing in `toy_examples/` uses `KDTreeUpdater` with `k > 1` (it is commented out in `typical_example.py`). |
| C5 | `surrDAMH/modules/Gaussian_process.py:48-52, 59-61` | Build `std` as a 1-D vector for both list and scalar input and use `block = np.outer(std, std) * corr` (scalar `std` broadcasts to the same result as today). | Scalar or constant-vector `std` gives an identical matrix; only non-constant vectors change, and those are asymmetric today (`06` item 1.6). The function is not called inside the package. The module is kept as a low-priority optional component (decision 7, 2026-09-13); the TSX experiments that might have used it are archived and obsolete. |
| C6 | `surrDAMH/stages.py:36` | `adaptive_corr_limit: float \| None = None`. | Turns a class attribute into a dataclass field with the same default. No script passes it today (it would have raised). Does **not** wire it to the proposal (see G3). |
| C7 | `surrDAMH/stages.py:17` | Add `"block"` to the `proposal_type` `Literal`. | Type annotation only. |
| C8 | `surrDAMH/modules/proposals.py:9-18` | Make `needs_gradients = False` and `subchain_length = 1` **class-level** defaults on `Proposal`; add `super().__init__()` to `GaussRandomWalk_adaptive.__init__` (`:140`). | The adaptive class currently lacks these attributes; `BlockProposal` reads `needs_gradients` (`:378`). Class defaults are read-only fallbacks; instances that set them keep their values. |
| C9 | `surrDAMH/modules/communication.py:351` and `surrDAMH/process_SOLVER.py:36-37` | Send `np.ascontiguousarray(parameters, dtype=np.float64)`. | The receivers are already float64 (`process_SOLVER.py:81`, `process_CHILD.py:39`). For every existing float64 sender the bytes are identical; only the float32 continuation path changes (`06` item 2.3). Sender-side only, receiver unchanged → matching rule satisfied. |
| C10 | `surrDAMH/modules/continuation.py:17,37` | Store and load `parameters` as `float64`. | Only affects files written **after** the change; `load_last_samples` still reads old float32 files (NumPy casts on assignment at `:45`). Together with C9 removes the dtype hazard. Flag: continued runs will start from the exact last sample instead of a float32-rounded one — a precision fix, not a semantic change. |
| C11 | `surrDAMH/post_processing.py:74` and the per-chain loop that sets `self.weights[i]` | Set `self.no_unique_samples[i] = len(self.weights[i])`. | Only used when `bins=None` is passed to the histogram helpers; no in-repo caller does (note 04, P2). |
| C12 | `summarize_tsx2_results.py:29, 234` | ~~Read both column names~~ **Superseded**: the TSX experiments are archived (2026-09-13); move this script and `analyze_tsx2_*.py` into `TSX_experiments_archived/` instead (needs go-ahead, see 09 §WS10). | Root-level analysis scripts of obsolete experiments; no library change. |

## D. Dead code, comments, documentation (zero behaviour change)

| # | File / lines | Change |
|---|---|---|
| D1 | `surrDAMH/modules/algorithms.py:25, 416-428, 491-503` | Remove the `evaluate_on_a_grid` import and the two commented-out debug blocks (the only uses of that import in the file; the function itself stays in `tools.py`). |
| D2 | `surrDAMH/modules/algorithms.py:448-462` | Replace the "2 lines of weird hot fix" comment and the commented alternative with the derivation: for a sub-chain kernel reversible w.r.t. the surrogate posterior, `Q(y→x)/Q(x→y) = π̃(y)/π̃(x)`, hence `log α = [log L(y) − log L(x)] − [log L̃(y) − log L̃(x)]`; the prior cancels. Add a note that this requires a **fixed** surrogate during the sub-chain (see H1). |
| D3 | `surrDAMH/modules/algorithms.py:297` | Remove `Algorithm_PARENT = AlgorithmBase` (no other reference in the repo). |
| D4 | `surrDAMH/modules/proposals.py:286-308` | Remove the commented-out old `HamiltonianInfinite.__init__`. |
| D5 | `surrDAMH/modules/proposals.py:311-331` | Add a docstring stating that the rotation is prior-preserving only for internal prior covariance `I` with mass `diag(sd²)`, and remains a valid (volume-preserving, reversible) MH proposal otherwise. |
| D6 | `surrDAMH/process_SOLVER.py:99-102` | Remove the dead `if False and all(child_can_solve): ... Probe ...` branch; keep the `Iprobe` line. |
| D7 | `surrDAMH/modules/communication.py:98-108, 130-152` and `process_COLLECTOR.py:228-234` | Add the invariant as a comment: "the collector consumes a TAG_UPDATE signal only when it simultaneously sends an evaluator; `terminate()` relies on this to decide whether a final message is owed". |
| D8 | `surrDAMH/modules/tools.py:4-9` | Remove the duplicated `import os` / `import numpy as np`. Keep the functions marked obsolete — `generate_surrogate_test_data` is still imported by `toy_examples/toy_example_hamilton.py:25` and `sampling_diffusion_grf.py:29`. |
| D9 | `surrDAMH/distributions/parent.py:38-43` | Fix the `get_covariance` docstring: 1-D return is the vector of **standard deviations** (as `Normal` and `PCN` treat it), not variances. |
| D10 | `surrDAMH/surrogates/parent.py:35-56` | Fix the `jacobian` docstring to the convention every implementation uses: single point in, returns `(J of shape (no_observations, no_parameters), evaluation)`. Docstring only; do not change code (see H). |
| D11 | `surrDAMH/configuration.py:73` | Assertion message: "use at least 3 MPI processes with collector and solvers pool (1 sampler + pool + collector); 4 recommended". |
| D12 | `toy_examples/typical_example_generic.py:6`, `sampling_TSX.py:6`, `sampling_diffusion_grf.py:9-10`, `neural_network_surrogate_copy.py:6` | Correct the filename in the run-command docstrings. `README.md:49`: `-n 4` → `-n 2` for `minimal_example.py` (or state that 2 is the minimum). |
| D13 | `toy_examples/typical_example.py:34`, `typical_example_generic.py:35` | Replace the commented `PyTorchNNOngoingUpdater` line with the current class name `NeuralNetworkUpdaterBasic` or delete it. |
| D14 | `README.md` | Add: the FEniCSx/dolfinx requirement of `sampling_TSX.py`, `wrapper.py`, `tunnel_with_subdomains.py`, `grf_diffusion.py`; the "run from `toy_examples/`" requirement (relative `solver_module_path`); a pointer to `library_notes/00_overview.md` §10 for the output layout until proper docs exist. |

## E. Packaging and repository hygiene

| # | File | Change | Note |
|---|---|---|---|
| E1 | `setup.py:34` | `install_requires=['numpy>=1.13.4', 'scipy', 'pandas', 'matplotlib', 'mpi4py', 'scikit-learn', 'torch', 'emcee']` | Matches `requirements.txt` and the unconditional imports. If torch is meant to be optional, that needs a lazy import in `surrogates/__init__.py` — a code change, so not in this plan. |
| E2 | `setup.py:41` | `python_requires='>=3.10'` (PEP 604 unions), or `>=3.12` per CLAUDE.md. | |
| E3 | `.gitignore` | Add `out_*/`, `*.npz`, `*.pt`, `*.pth`, `.venv/` and delete the ~10,000 per-file `.venv/...` lines. | Pure ignore-file change; nothing tracked is affected. Use `git check-ignore -v` to confirm no tracked file becomes ignored (tracked files are unaffected by ignore rules anyway). |
| E4 | repo root | Delete the two empty `torch_perceptron_par.csv` / `torch_perceptron_obs.csv` (0 bytes, untracked, produced by a default relative path). | **Ask before deleting** anything else at the root (`test_toy_shrinkable.html`, `loss_during_incremental_training.png`, `test_cuda.ipynb`, analysis scripts) — they may be wanted. |
| E5 | `toy_examples/out_tsx/sampling_TSX.py` | Decide whether the tracked file belongs there; not opened in this review. | Decision only. |
| E6 | `surrDAMH/surrogates/torch_perceptron*.py:387/645` | Default `save_snapshots` paths: prefix with `self.output_dir` if the updater has one, else keep CWD but print the absolute path. | Optional; prevents stray root files. Only affects a debugging helper. |

## F. Suggested order and commit grouping

1. **Commit 1 — "make the package importable"**: A1, A2, A3, D1, D3, D4. Sanity: `python -c "import surrDAMH, surrDAMH.process_SAMPLER"`.
2. **Commit 2 — "fail fast instead of hanging"**: B1–B7, C3.
3. **Commit 3 — "latent bug fixes without behaviour change on working paths"**: C1, C4–C11 (C2 after author confirmation; C12 separately since it is an analysis script).
4. **Commit 4 — "comments, docstrings, dead code"**: D2, D5–D14.
5. **Commit 5 — "packaging and hygiene"**: E1–E3 (+E4 if approved).

Per CLAUDE.md: work on `working_Kuba`, no commit/push unless asked; the grouping above is a proposal.

## G. Safe in principle, but changes the exact sample stream or results — needs a one-line decision each

> **Status 2026-09-17: G1–G6 all DECIDED ("do them as recommended") and IMPLEMENTED**, together
> with A30. Per-item evidence — which configurations change and which are bit-identical, with
> the comparisons that were run — in `10_manual_review_notes.md` §2.13; one CHANGELOG entry per
> item. Two deviations from the recommendations below, both documented in §2.13: G1's
> `adaptive_sample_limit` default is `None` (unbounded history, i.e. today's behaviour), not
> 10, so unset stages stay bit-identical; and G5's sub-proposal seeds use
> `2**31 + 1000*block_seed + index` rather than `seed + 100*(k+1)`, which collides for 11+
> samplers with 2 groups. G2 additionally makes `proposal_type="block"` with `adaptive=True`
> raise a `ValueError` in `build_proposal` instead of dying later. G7 was decided on
> 2026-09-13; G8 is still deferred.

| # | Item | What changes | Recommendation |
|---|---|---|---|
| G1 | Wire `Stage.adaptive_target_rate / adaptive_corr_limit / adaptive_sample_limit` into `GaussRandomWalk_adaptive` (`process_SAMPLER.py:114`), defaulting to the current 0.25 / 0.3 / 10 when `None`. | Scripts that leave the fields unset: identical. Scripts that set them (`TSX_complete_experiment_2/sampling_.py:450`, target 0.1) start honouring them → different acceptance rate and mixing. | Do it, and re-label past runs that set these fields as "ran with 0.25". |
| G2 | Normalise `sd_or_cov` to a 2-D covariance before the adaptive `Allreduce` (`process_SAMPLER.py:180-185`). | Removes the shape-mismatch hang, but when a rank never adapted, the next stage would receive a 2-D matrix (multivariate draws) instead of a 1-D sd vector (independent draws): same distribution, **different RNG stream**, so exact samples of such edge-case runs change. | Do it; statistically neutral. |
| G3 | `lhs_normal.py`: set `maxmin = quality` inside the `if` (`:47`). | Fixes the "best of 5" selection; changes the LHS initial samples of every `initial_sample_type="lhs"` run (different but better-spread starting points). | Do it, note it in the changelog. |
| G4 | Seed `rvs()` (`Normal`, `PriorIndependentComponents`) from a per-rank generator derived from `seed0` instead of the global RNG. | `initial_sample_type="prior"` runs become reproducible; the initial samples themselves change (they were not reproducible before). | Do it via an optional `generator` argument so external callers keep working. |
| G5 | Re-seed `BlockProposal` sub-proposals per rank. | Chains stop sharing identical proposal increments in block stages. Changes every block-proposal run. | Do it; current behaviour is a correctness defect, not a reproducible feature worth preserving. |
| G6 | Flush CSV writers (`monitoring.py:28`: `open(path, "w", buffering=1)`). | Partial runs become inspectable; slightly more I/O per row. No content change. | Do it. |
| G7 | Default flip `state_dependent_approximation: True → False` (`configuration.py:37`). | Posterior-affecting for scripts that rely on the default. | **Decided 2026-09-13: keep `False`**, warn on `True` (B8), exclude `True` from tests; state it in the commit message and changelog. |
| G8 | `raw_data` ragged rows: write a fixed observation block (NaN-filled) and a header. | Output-format change; readers in `post_processing.py` would need matching updates. | Defer until a golden-file test exists (07 §1 item 40). |

## H. Explicitly deferred — needs the validation runs in `07_testing_plan.md` or an author decision

- **H1** Sub-chain semantics (`06` items 1.1, 1.2): freezing the surrogate per sub-chain and fixing the
  state-dependent shift for `subchain_max_length > 1` change the DAMH kernel. Correct on paper, but
  they alter acceptance rates and must be validated with V2/V3/V4 before touching experiments.
- **H2** Zero-weight snapshots in NN training (3.1): a modelling decision (multiplicity weight vs
  training weight), changes surrogate accuracy.
- **H3** Collector busy-wait sleeps, 1 GiB `irecv` buffer default, snapshot batching (Tier 4): timing
  and protocol-adjacent; test under MPI first.
- **H4** RBF fallback rewrite, torch module unification, `initial_training` semantics, evaluator
  output-shape standardisation (Tier 3): touch the DAMH path or persisted files.
- **H5** Narrowing the bare `except BaseException` clauses in `post_processing.py` (P6): could expose
  errors that are currently swallowed in report generation.
- **H6** Deleting or merging duplicate examples (`neural_network_surrogate_copy.py`, the three
  post-processing demos): user's call.
