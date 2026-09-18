# Evaluator/Updater contract — specification (WS6)

**Status: IMPLEMENTED (2026-09-17).** §1 below is a historical record of the *pre-WS6* state;
§2 is the contract that now holds. Applied evidence with numbers:
`library_notes/10_manual_review_notes.md` §2.16; user-facing summary: `CHANGELOG.md` WS6;
user documentation: `docs/writing_a_surrogate.md`.

Deviations from this draft, decided by the author on 2026-09-17 and applied instead:

1. The option is called **`weighting`** (not `snapshot_weighting`) and lives on **every** updater
   via the base class, which implements the policy once (`Updater._rows_to_use` /
   `Updater._training_weights`) and exposes `supports_sample_weights`. §2's "document, don't
   wire" recommendation for the classical surrogates was **overruled**: `"multiplicity"` drops
   zero-multiplicity rows for all of them, and `PolynomialSklearnUpdater` really weights its
   regression (`sample_weight=`). RBF/kd-tree stay unweighted interpolants and say so.
2. Output normalization is **`output_normalization: Literal["identity","likelihood","manual"]`
   defaulting to `"likelihood"`** (statistics from the likelihood: observed data + per-observation
   noise sd), not the drafted opt-in `output_stats="auto"` estimated from the first `N` snapshots.
   Open question 3 of §4 is therefore moot. Delivered through a new hook
   `Updater.set_output_normalization(mean, scale)` called once by `SamplingFramework.__init__`
   and `run_local`.
3. The provenance of the statistics in effect is stored in the checkpoint's `updater_hparams`
   **as the value of `output_normalization`** (`"identity"`/`"likelihood"`/`"manual"`), with
   `output_mean`/`output_scale` set to `None` for `"identity"`. This keeps
   `SurrogateReused`'s `updater_cls(**hparams)` reconstruction exact without adding a
   restore-only constructor argument. The updater also exposes it as
   `output_normalization_provenance` (reported in the run manifest).
4. `_get_surrogate_observations` (`algorithms.py`) indexes `[0]` into the one-row batch instead of
   having its `.ravel()` deleted outright — deleting it would hand a `(1, q)` array to
   `Normal.logpdf`. Values are unchanged (finding A20's failure mode); the other four
   compensating reshapes named in §1 were deleted.
5. `initial_training` is fixed by saving/restoring the real snapshot arrays around the synthetic
   pass (§2's second suggested option), not by a scratch copy of the model parameters.
6. §4 open question 5: `neural_network_surrogate.py` **and** `sampling_TSX.py` were both migrated
   to the preset now (the author decided to keep the TSX example);
   `neural_network_surrogate_copy.py` was already deleted.

Shapes in §1 were read from the actual `__call__`/`jacobian`/`vjp` bodies as they were before
WS6, not inferred from docstrings.

## 1. Today — conformance table

`X: (n,p)` batch input, `x: (p,)` single point, `q = no_observations`.

| | Polynomial (`polynomial_sklearn.py`) | RBF (`rbf_scipy.py`) | KDTree (`nearest_kdtree.py`) | NN Basic (`torch_perceptron.py`) | NN Minibatches (`torch_perceptron_minibatches.py`) |
|---|---|---|---|---|---|
| `__call__(X (n,p))` | `(n,q)` — `model.predict` (`:24-26`) | `(n,q)` — `rbf_interpolator(X)` (`:23-26`) | `(n,q)` — `(:23-40)` | **`(n*q,)` flattened** (`:74`, `outputs.flatten()`) | **`(n*q,)` flattened** (`:95`, `outputs.flatten()`) |
| `__call__(x (p,))` | reshaped to `(1,p)` first (`:24`) → `(1,q)` | same pattern → `(1,q)` | same pattern → `(1,q)` | flattened → `(q,)` | flattened → `(q,)` |
| `jacobian(x)` | not implemented (inherits `NotImplementedError`) | not implemented | not implemented | `(J (q,p), y (q,))` tuple, reverse-mode, one `autograd.grad` per output row, float32 (`:77-106`) | `(J (q,p), y (q,))`, forward-mode `torch.func.jacfwd`, float64 (model cast, `:97-125`) |
| `vjp(x, v)` | inherited default: unpacks `jacobian(x)` — raises the same `NotImplementedError` | inherited, raises | inherited, raises | inherited default works (because `jacobian` already returns the `(J, y)` tuple the base `vjp` expects) | explicit override, reverse-mode autograd, float64 (`:127-161`) |
| `supports_gradients()` | `False` (not overridden) | `False` | `False` | `self.use_gradients` (`:163-164`, default `True`) | `self.use_gradients` (default `True`) |
| batch vs single-only | batch everywhere | batch everywhere | batch everywhere | `__call__` batch; `jacobian`/`vjp` **single point only**, raise `ValueError` on any other shape (`:82-85`,`:132-141`) | same restriction |
| `add_data` weight handling | **discards weights**: `weights = None` hard-set at `:49`, then defaults to all-ones internally, never uses the caller's array | **ignores**: signature accepts `weights` but body never reads it (`:55-59`) | **ignores**: same pattern (`:61-66`) | uses `weights` (all-ones default if `None`) but only for `train_on_added_data=True` fine-tuning; no documented weighting in the main `train()` L-BFGS path (grep confirms no `_weighted_loss`-equivalent in `torch_perceptron.py`) | uses `weights` in `_weighted_loss` (`:358-367`): per-sample weighted mean, clamps negatives, falls back to unweighted mean if the weight sum is 0 or non-finite |
| `train()` semantics | not overridden — fit happens inside `get_evaluator()`, always **refit from scratch** on the full accumulated `self.par/obs` (`:63-77`) | not overridden — refit from scratch on the full dataset inside `get_evaluator()` (`:61-73`), **O(N³)** every call | not overridden — tree rebuilt from scratch inside `get_evaluator()` (`:75-79`), O(N log N) | `train()` runs incrementally on the persistent model/optimizer state over `self.par/obs/weights` | `train()` runs `iterations_batch` optimiser **steps** (not epochs) over accumulated data with a persistent AdamW/LBFGS state (`:545 ff`) |
| `get_initial_snapshots()` | inherited `None` | inherited `None` | inherited `None` | **not overridden**, always `None`, even though `load_state` exists (S5) | overridden, returns loaded snapshots (`:315`) |
| Persistence (`save_state`/`load_state`) | not implemented (inherits `NotImplementedError`) | not implemented | not implemented | implemented but signature drops `save_optimizer`, and `data_path` is not optional in practice (S16) | implemented to the full parent signature, `save_optimizer` honoured |
| `supports_*_persistence()` | `False`/`False` (correct — nothing implemented) | `False`/`False` | `False`/`False` | reports `False`/`False` **although both are implemented** (S5) | reports `True`/`True` (`:657-661`), correct |
| `@register_updater` (for `reuse.SurrogateReused`) | not registered | not registered | not registered | **not registered** — a Basic checkpoint cannot be restored via `SurrogateReused` (S5) | registered (`torch_perceptron_minibatches.py:170`) |
| Picklable (evaluator shipped over MPI) | yes — plain sklearn `Pipeline`, no torch state | yes — plain scipy object | yes — `cKDTree` + numpy arrays | yes — `clone_model_to_cpu` rebuilds from hyperparameters + `state_dict`, no optimizer/CUDA state (`:78-87`) | same pattern, also strips optimizer state (`:78-87` equivalent) |
| WS1 unit test coverage (`tests/unit/test_surrogates.py`) | `TestUpdaterContract` (shape/finiteness, pickling), `TestPolynomialExactQuadratic` | `TestUpdaterContract`, `TestRBFDuplicatedSnapshot` | `TestUpdaterContract`, `TestKDTreeExactHit` (pins the now-fixed zero-distance guard) | `TestUpdaterContract`, `TestNNGradients`, `TestCheckpointRoundTrip`, `TestNNOutputNormalization` (has **no** output-normalization API — noted at test file line 197) | all of the above plus `TestCheckpointRoundTripMinibatchesOnly`, `TestInitialTrainingSyntheticRows` (pins S6 as an unfixed bug), `TestSnapshotWeightingAffectsTraining` (pins today's all-or-nothing zero-weight behaviour, S2) |

Base class today (`surrogates/parent.py`):
- `Evaluator.__init__` only *annotates* `self.no_parameters: int` (`:35`) without assigning it — a
  dead statement; every concrete evaluator sets its own `self.no_parameters` in its own `__init__`
  instead of calling `super().__init__()`. **There is no `no_observations` on the base class at all.**
- `Evaluator.jacobian`/`vjp` docstrings (`:47-70`) **already document the `(J, evaluation)` tuple
  convention that the code implements** — this was corrected in the working tree since
  `03_surrogates_and_distributions.md` was written (that note's **S4**, "docstring promises `(n,q,p)`
  batched, code returns a tuple", is **fixed**; flagging the disagreement per the task brief).
- `SurrogateAsSolver.get_observations` (`:19-30`) already returns `(no_observations,)` regardless of
  whether the wrapped evaluator returns `(1,q)` or a flattened `(q,)` — the A20 fix referenced in
  `10_manual_review_notes.md` §2.7 is present (`np.asarray(result).reshape(-1)`).
- `Updater.__init__` only annotates `self.no_snapshots: int` (`:90`), same dead-statement pattern;
  Polynomial/RBF/KDTree use their own counters (`num_snapshots`, or none at all) instead.

Call sites that compensate for the flattened torch output (grep, still present):
`algorithms.py:288,292` (`.ravel()` on `_get_surrogate_observations`'s single/pair results),
`algorithms.py:321` (`.reshape(-1)` before computing the surrogate gradient vector),
`process_COLLECTOR.py:80,173` (`.reshape(true_observations.shape)` in both quality-metric paths).

## 2. Proposed contract

| Element | Rule | Behaviour change? |
|---|---|---|
| `__call__(X: (n,p)) -> (n,q)` | Always 2-D output, **including `n=1`**. No flattening anywhere. Single-point convenience: **not added** — callers pass `x.reshape(1,-1)` explicitly (matches what `SurrogateAsSolver` already does at `parent.py:29`); avoids a second calling convention. | **Behaviour change for NN evaluators only** (shape, not values) — remove `.flatten()` at `torch_perceptron.py:95`, `torch_perceptron_minibatches.py:95`. Delete the compensating `.ravel()`/`.reshape()` at the 5 call sites listed above once both torch evaluators comply — same numerical result (row-major flatten/reshape round-trips exactly), pure contract fix. |
| `jacobian(x: (p,)) -> (J: (q,p), y: (q,))` | Codify what all implementations already return; single-point only (raise `ValueError` on batched input, as the torch ones already do). Non-differentiable evaluators keep raising `NotImplementedError`. | None — matches current behaviour and current docstring exactly. |
| `vjp(x: (p,), v: (q,)) -> (g: (p,), y: (q,))` | Base-class default stays "call `jacobian`, contract via `J.T @ v`" for evaluators that implement only `jacobian`. Evaluators with a cheaper reverse-mode path (NN Minibatches) keep their explicit override. | None. |
| `no_parameters` / `no_observations` on `Evaluator` | Add both as real attributes set in `Evaluator.__init__(self, no_parameters, no_observations)`, called via `super().__init__(...)` from every subclass (currently skipped everywhere). | Contract fix; also finally makes the dead annotation useful for introspection/validation (e.g. shape asserts in tests). |
| `set_use_gradients(enabled)` | Keep the existing optional-hook signature; document that evaluators without gradients silently no-op (current default `Evaluator.set_use_gradients`, `parent.py:76-78`) rather than raising — no code change, just documented. | None |
| `as_solver()` | Keep `SurrogateAsSolver` as-is (already fixed, A20) — `get_observations` reshapes to `(no_observations,)` regardless of the evaluator's `__call__` shape. Once `__call__` is standardised to `(1,q)` for `n=1`, the `reshape(-1)` in `SurrogateAsSolver.get_observations` (`parent.py:30`) becomes provably correct rather than "right by accident" (current docstring already says this). | None (already implemented) — documents that the shape standardisation removes the "by accident" caveat |
| `snapshot_weighting` (decision 3) | New `Updater.__init__` keyword, `Literal["uniform","multiplicity"] = "uniform"`. Lives on **each concrete updater that trains from weighted data** (currently only the two NN updaters actually train on a loss; classical ones ignore weights entirely — see the deletion/normalisation rows below). Policy application point: inside the updater's own loss/fit call (e.g. `NeuralNetworkUpdaterMinibatches._weighted_loss`), **not** in `process_COLLECTOR.py` — the collector keeps sending the real multiplicity weights (`algorithms.py:225`, `weight=0` for rejected) unchanged; the updater decides what to do with them. `"uniform"`: replace every incoming weight with `1.0` before it reaches the loss (rejected proposals now contribute, at weight 1, same as accepted ones) — **default**. `"multiplicity"`: use the weights as sent (today's behaviour) — docstring/start-up log must state "rejected proposals (weight 0) are not used for training" when this option is selected. | **Real behaviour change for the NN surrogates**: `"uniform"` (the new default) changes which snapshots influence the fit relative to today's un-labelled behaviour (today = `"multiplicity"` unconditionally, i.e. `S2`). This is the change the plan calls out as posterior/surrogate-accuracy relevant — flag in `CHANGELOG.md` per CLAUDE.md rules. Classical surrogates (poly/RBF/kd-tree) do not consume weights at all today; decision 3 does not force them to start doing so (see next row). |
| Classical surrogates and `weights` | Explicit choice, not silent: either (a) keep ignoring `weights` but say so in each docstring ("this updater treats every snapshot as equally informative regardless of `weight`"), or (b) wire `weights` in as per-row regression weights (`sample_weight=` in sklearn's `LinearRegression.fit`, an analogous scipy RBF option does not exist — would require a manual weighted least-squares reformulation). This spec recommends **(a) document, don't wire** for RBF/KDTree (no natural weighted variant without larger surgery) and **document only** for Polynomial too, since `Ridge`+`sample_weight` is a bigger change than WS6's mechanical scope. | (a) = no behaviour change, pure documentation. (b) would be a behaviour change, out of scope here — flagged as future work if desired (see open questions). |
| Normalisation (`normalize_inputs`, `output_stats="auto"`) | Opt-in keywords on `NeuralNetworkUpdaterMinibatches.__init__`, default preserving **today's identity behaviour**: `normalize_inputs=False` (no input scaling, ever — matches today, no such feature exists), `output_stats: Literal["identity","auto"] = "identity"` (today's default `output_mean=0, output_scale=1`, i.e. no normalisation unless the caller supplies statistics — `torch_perceptron_minibatches.py:214-221`). `output_stats="auto"`: estimate `output_mean`/`output_scale` from the first `N` snapshots (`N` a new constructor parameter, e.g. `normalization_snapshot_count`), then **freeze** and persist them in the checkpoint (extends the existing `output_mean`/`output_scale` checkpoint fields, `torch_perceptron_minibatches.py:279-311` — no new checkpoint schema needed, just a computed-vs-supplied provenance flag). | Opt-in only; default unchanged. Choosing `"auto"` is a user decision with a real effect on surrogate accuracy, must be logged at start-up like the `snapshot_weighting` choice. |
| `initial_training` (3.4) | Must not persist synthetic rows. Concretely: `NeuralNetworkUpdaterMinibatches.initial_training` (`:478-488`) currently overwrites `self.par/obs/weights` with `n=1000` synthetic `randn` rows and sets `no_snapshots=n` — this pollutes `get_training_data_arrays()`/checkpoints. Fix: run the synthetic pre-training pass in a scratch copy of the model's parameters (or simply skip storing into `self.par/obs/weights`/`no_snapshots`) and restore the real (possibly empty) snapshot arrays before returning. Basic's `initial_training` already does not persist synthetic rows (`torch_perceptron.py:207`, opposite of Minibatches) — Basic is deleted anyway (decision 4), so Minibatches' behaviour becomes the only one and must match Basic's (safer) semantics. | **Behaviour change**: today, calling `initial_training` on the Minibatches updater changes what a subsequent `get_training_data_arrays()`/checkpoint contains; after the fix it does not. Only affects users who call `initial_training` at all (grep shows this is opt-in, not called automatically by `process_COLLECTOR.py`). |
| Checkpoint/restore | Keep the current `surrogate_checkpoint.pt` + `surrogate_training_data.npz` pair (`reuse.py:12-15`) and the `@register_updater` + `updater_hparams` mechanism unchanged; only add `torch.load(..., weights_only=True)` (S22) as a hardening fix, and register whatever the Basic-replacement preset becomes (§3) so restore keeps working after the deletion. | `weights_only=True` is a hardening fix, not expected to change any existing checkpoint's ability to load (state dicts + hparams are plain tensors/scalars already) — but flag it as unverified against every historical checkpoint until tested. |

## 3. Deletion list (decision 4: delete `NeuralNetworkUpdaterBasic`)

What disappears:
- `surrDAMH/surrogates/torch_perceptron.py` in full (`PyTorchMLP`, `PyTorchNNEvaluator`,
  `NeuralNetworkUpdaterBasic` — ~250 duplicated lines per `03_surrogates_and_distributions.md`).
- The re-export in `surrDAMH/surrogates/__init__.py:4,7` (`from .torch_perceptron import
  NeuralNetworkUpdaterBasic`, and its `__all__` entry).

Grep of every reference (`NeuralNetworkUpdaterBasic\|torch_perceptron import\|from surrDAMH.surrogates.torch_perceptron `):

| File | Reference | Action needed |
|---|---|---|
| `surrDAMH/surrogates/__init__.py:4,7` | import + `__all__` entry | remove |
| `surrDAMH/surrogates/torch_perceptron.py` | the class itself | delete file |
| `tests/unit/test_surrogates.py:31,68,247` (+ doc comments at `:10,14,197`) | imports and constructs `NeuralNetworkUpdaterBasic` directly, and its own file header says it deliberately keeps testing both modules "while WS6 work is pending" | rewrite: drop the Basic-specific fixtures/tests, or repoint them at the replacement preset (§ below) so `TestUpdaterContract`'s parametrisation still covers a full-batch-like configuration |
| `toy_examples/sampling_TSX.py:40` | constructs `surrDAMH.surrogates.NeuralNetworkUpdaterBasic(...)` | TSX-specific, FEniCSx-dependent — per `09_improvement_plan.md` WS10 this file's fate ("stay as the real-solver example or join the archive") is undecided; either way it needs updating to the replacement preset if kept |
| `toy_examples/neural_network_surrogate.py:41` | same construction | update to the replacement preset (this file is in the "keep one of each" canonical set per WS10, not yet resolved which of `neural_network_surrogate.py`/`neural_network_surrogate_copy.py` survives) |
| `toy_examples/neural_network_surrogate_copy.py:74` | same construction | per WS10 this is a duplicate slated for removal anyway — resolve together |
| `toy_examples/out_tsx/sampling_TSX.py:40` | same construction | inside `out_tsx/` (an `out_*` directory) — **out of scope for this refactor per CLAUDE.md's off-limits-data rule**; not touched |
| `README.md` | no hits (grepped) | none |

Minimal replacement preset, if full-batch L-BFGS is ever wanted again (per plan WS6: "a preset of the
minibatches updater"):

```python
NeuralNetworkUpdaterMinibatches(
    no_parameters=..., no_observations=...,
    solver="lbfgs", batch_size=None,      # None => _infer_batch_size returns the full dataset size
    replay_ratio=0.0,                     # no replay buffer needed when every step already sees all data
    train_on_added_data=False,            # only the periodic train() call fits, matching Basic's train() semantics
)
```
This reproduces Basic's "full-batch L-BFGS, refit over everything" shape using existing
Minibatches machinery (`_infer_batch_size` already special-cases `solver=="lbfgs"` to return the
whole subset, `torch_perceptron_minibatches.py` around the batch-size logic) — no new code path
needed, just documenting the preset in `docs/writing_a_surrogate.md` (WS11).

## 4. Open questions for the author

1. `snapshot_weighting="uniform"` as the new default changes NN surrogate training data relative to
   every prior run (today = unconditional `"multiplicity"`). Confirm this is acceptable to land
   without a `legacy_weighting=` escape hatch (CLAUDE.md: no feature flags by default; WS3 set the
   same precedent for the sub-chain freeze).
2. Classical surrogates (poly/RBF/kd-tree): document-only for `weights` (§2 row "Classical
   surrogates"), or is a real weighted refit worth the extra scope for any of the three?
3. Should `output_stats="auto"`'s snapshot count `N` be a fixed constant, a fraction of
   `min_snapshots_initial`, or a fully separate user-set parameter?
4. `initial_training`'s fix changes Minibatches' current (buggy) persistence behaviour — confirm no
   existing script relies on the synthetic rows surviving into a checkpoint. Confirmed by grep:
   `initial_training` is never called from `process_COLLECTOR.py` nor from any `toy_examples/*.py`
   (zero hits) — it is dead/unused today, so this fix has no known caller to break.
5. Which of `neural_network_surrogate.py` / `neural_network_surrogate_copy.py` / `sampling_TSX.py`
   should be updated to the replacement preset now vs. left for WS10's broader toy-example cleanup?
