# Surrogate models and distributions — notes

Read-only review of the **working-tree** state (uncommitted changes included).
No code was executed (one exception, disclosed in *Method notes* at the end).
`out_*` directories were not opened.

## Files covered

Fully read:

- `surrDAMH/surrogates/__init__.py`, `parent.py`, `polynomial_sklearn.py`, `rbf_scipy.py`,
  `nearest_kdtree.py`, `reuse.py`, `torch_perceptron.py`, `torch_perceptron_minibatches.py`
- `surrDAMH/distributions/__init__.py`, `parent.py`, `normal.py`, `independent_components.py`,
  `gaussian_mixture.py`, `transformations.py`
- `surrDAMH/modules/Gaussian_process.py`, `lhs_normal.py`, `test_data.py`
- `surrDAMH/process_COLLECTOR.py`, `surrDAMH/modules/algorithm_interfaces_local.py`,
  `algorithm_interfaces_mpi.py`, `surrDAMH/modules/tools.py`

Skimmed for consumption patterns: `surrDAMH/core.py` (imports, `_configure_surrogate_gradients`,
`run`), `surrDAMH/configuration.py` (flags), `surrDAMH/modules/algorithms.py`
(`_get_surrogate_observations`, `_compute_surrogate_log_likelihood_gradient`, `_send_to_collector`),
`surrDAMH/modules/communication.py` (`CommEvaluator_*`), `surrDAMH/modules/proposals.py` (`PCN`),
`surrDAMH/process_SAMPLER.py`, `surrDAMH/solvers.py`.

Deleted file `surrDAMH/surrogates/torch_perceptron_outnorm.py`: **no remaining references**
(`git grep outnorm -- ':!out_*'` is empty). The deletion is clean.

## Interface contract (Updater / Evaluator) and conformance matrix

### Contract as declared in `surrDAMH/surrogates/parent.py`

`Updater` (constructed by the user, lives on the collector rank):

| method | declared contract | actual caller |
|---|---|---|
| `__init__(no_parameters, no_observations)` | sets `self.no_snapshots` (`parent.py:76`) | user script |
| `delayed_init(data)` | optional, no-op | `process_COLLECTOR.py:98` |
| `add_data(par, obs, wei)` | shapes `(N,p)`, `(N,q)`, `(N,1)` (`parent.py:85-91`) | `process_COLLECTOR.py:187`, `algorithm_interfaces_local.py:277,305` |
| `train()` | called periodically, "regardless of whether new data have been added" | `process_COLLECTOR.py:193` |
| `get_evaluator() -> Evaluator` | new evaluator, gets pickled and MPI-`isend`ed | `process_COLLECTOR.py:135,197` |
| `supports_gradients()`, `set_use_gradients(bool)` | default `False` / no-op | `core.py:70,87,107` |
| `get_initial_snapshots()` | `list[NDArray] \| None` preloaded into the collector | `core.py:127` |
| `supports_training_data_persistence()` / `supports_state_persistence()` | default `False` | **nothing** (dead) |
| `save_state(checkpoint_path, data_path=None, save_optimizer=True)` | parent signature (`parent.py:142`) | `tools.py`-adjacent experiment scripts |
| `load_state(checkpoint_path, data_path=None, load_optimizer=True)` | returns `(par, obs, wei)` or `None` | `tools.py:146` |

`Evaluator` (pickled and sent to every sampler):

| method | declared contract |
|---|---|
| `__call__(datapoints (n,p)) -> (n,q)` | `parent.py:25-32` |
| `jacobian(datapoints (n,p)) -> (n,q,p)` | `parent.py:35-42` |
| `vjp(datapoint (p,), vector (q,)) -> (grad (p,), evaluation (q,))` | `parent.py:45-56`; **default impl unpacks two values from `jacobian`**, contradicting the `jacobian` docstring |
| `supports_gradients()` | introspects whether `jacobian`/`vjp` were overridden (`parent.py:60`) |
| `set_use_gradients(bool)` | optional hook |
| `as_solver()` | wraps `__call__` in `SurrogateAsSolver` (used by `stage.use_only_surrogate`, `process_SAMPLER.py:147`) |

### Updater conformance

| | Polynomial | RBF | KDTree | NN Basic | NN Minibatches |
|---|---|---|---|---|---|
| `no_snapshots` attribute | ✗ (uses `num_snapshots`) | ✗ (none) | ✗ (none) | ✓ | ✓ |
| `add_data` | ✓ but **discards `weights`** (`polynomial_sklearn.py:49` hard-sets `weights = None`) | ✓, weights arg ignored | ✓, weights arg ignored | ✓ (extra kwarg `train_on_added_data=True`) | ✓ (extra kwarg, default `False`) |
| `train()` | ✗ not overridden — fitting happens inside `get_evaluator()` | ✗ (no-op; build in `get_evaluator`) | ✗ (no-op) | ✓ | ✓ |
| `get_evaluator()` | ✓ | ✓ | ✓ | ✓ | ✓ |
| `delayed_init` | inherited no-op | inherited | inherited | inherited | inherited |
| `supports_gradients()` | ✗ → `False` ✓ correct | ✗ → `False` ✓ | ✗ → `False` ✓ | `True` (unconditional) | `True` (unconditional) |
| `set_use_gradients` | inherited no-op | inherited | inherited | ✓ | ✓ |
| `get_initial_snapshots()` | inherited `None` | inherited | inherited | **✗ missing although `load_state` exists** | ✓ (`:315`) |
| `supports_*_persistence()` | `False` ✓ | `False` ✓ | `False` ✓ | **`False` although save/load implemented** | ✓ `True` |
| `save_state` / `load_state` | ✗ | ✗ | ✗ | ✓ but signature `(checkpoint_path, data_path)` — **no `save_optimizer`, `data_path` not optional** | same deviation |
| `@register_updater` (for `reuse.SurrogateReused`) | ✗ | ✗ | ✗ | **✗ — cannot be restored by `SurrogateReused`** | ✓ |

### Evaluator conformance

| | Polynomial | RBF | KDTree | NN Basic | NN Minibatches |
|---|---|---|---|---|---|
| `__call__` output shape | `(n,q)` ✓ | `(n,q)` ✓ | `(n,q)` ✓ | **flattened 1-D** (`torch_perceptron.py:74`) | **flattened 1-D** (`..._minibatches.py:95`) |
| dtype | float64 | float64 | float64 | float32→float64 numpy | float32 |
| `jacobian` | ✗ | ✗ | ✗ | returns **tuple `(J (q,p), evaluation (q,))` for a single point** — not the documented `(n,q,p)` | same, via `torch.func.jacfwd`, float64 |
| `vjp` | inherited (broken, see S4) | inherited | inherited | inherited (works, because `jacobian` returns the tuple the parent expects) | ✓ explicit, float64 autograd |
| `supports_gradients` | `False` ✓ | `False` ✓ | `False` ✓ | returns `self.use_gradients` ✓ | ✓ |
| batch vs single | batch | batch | batch | `__call__` batch, `jacobian`/`vjp` **single point only** (raises `ValueError` otherwise) | same |

Consumers currently tolerate the shape deviation: `algorithms.py:249,253` call `.ravel()`,
`process_COLLECTOR.py:173` and `:80` call `.reshape(true_observations.shape)`. A flattened
`(n*q,)` reshaped to `(n,q)` is row-major-correct, so batch monitoring of the NN surrogate is
numerically right — but it is right by accident, not by contract.

## Per-surrogate notes

### `polynomial_sklearn.py` — `PolynomialSklearnUpdater`

- **Degree rule** (`:65-68`): `num_terms` tracks `C(p+d, d)`; `terms_multiplicator = (p+d+1)/(d+1)`
  is the exact factor `C(p+d+1,d+1)/C(p+d,d)`. The loop raises the degree while
  `num_snapshots > (number of terms of degree d+1)`, capped by `max_degree=5`. The bookkeeping is
  correct, but `num_terms` is a float accumulated by repeated multiplication (rounding drift at
  high `p`, harmless in practice).
- **Refit policy**: refit only when `num_snapshots` grew; a new `Pipeline` is created only when the
  degree changed, so the estimator is reused but always refit from scratch (`LinearRegression.fit`).
- **Conditioning**: `PolynomialFeatures` on *raw* parameters, no centring/scaling, `LinearRegression`
  (unregularised `scipy.linalg.lstsq`). Ill-conditioned quickly for degree ≥ 3 in more than a few
  dimensions. No ridge/`alpha`. Since `lstsq` returns a minimum-norm solution, a rank-deficient
  system does not crash, it silently returns an arbitrary fit.
- **Few snapshots**: `conf.min_snapshots_initial` defaults to `1`, so the first model can be a
  degree-1 fit through a single point in `p` dimensions — hugely underdetermined. DAMH stays exact
  (delayed acceptance corrects), so this only costs efficiency, not correctness.
- **Output shape**: `model.predict` on 2-D `obs` gives `(n, q)` ✓.
- Snapshot list `self.par/obs/wei` grows without bound; `self.wei` is always all-ones dead state.

### `rbf_scipy.py` — `RBFInterpolationUpdater`

- Rebuilds a **global** `RBFInterpolator` from *all* snapshots on every `get_evaluator()`
  (`:62-66`). `neighbors=None` by default → dense solve, **O(N³) time / O(N²) memory each update**,
  and `get_evaluator()` is called at every surrogate update in the collector loop
  (`process_COLLECTOR.py:197`). **No pruning, no subsampling, no incremental update.** With
  `min_snapshots_to_update=1` (the default) this is a cubic cost per accepted snapshot.
- Defaults: `kernel="thin_plate_spline"`, `degree=None` (scipy picks the minimal degree for
  conditional positive definiteness, 1 for TPS), `smoothing=0.0` → **exact interpolation**, hence
  maximal sensitivity to duplicated/near-duplicated snapshots.
- **Duplicate handling** (`:67-74`): the `except ValueError` fallback. `numpy.linalg.LinAlgError`
  *is* a subclass of `ValueError` in the numpy in this environment, so scipy's "Singular matrix"
  from duplicate points is caught. The fallback then builds an augmented dataset
  `par ∪ (par+1) ∪ (par+2) ∪ …` (`p` extra shifted copies) with the **same observations**, and fits
  a `linear` kernel with `smoothing=1`. That is a de-conditioning hack that asserts
  `f(x) = f(x+1·𝟙) = f(x+2·𝟙) = …`, which is a strong false constraint on the surrogate. It is also
  not idempotent: exact duplicates remain duplicates inside the augmented set, so the fallback can
  raise again (uncaught).
- DAMH resends the *same* parameter point as a snapshot whenever a sample repeats
  (`algorithms.py:148` sends the current sample with its weight), so exact duplicates are expected,
  not exotic — verify how often the fallback fires in practice.

### `nearest_kdtree.py` — `KDTreeUpdater`

- `k == 1`: plain nearest-neighbour lookup, `self.obs[indices, :]` → `(n,q)` ✓.
- `k > 1`: **inverse-distance weighting with no zero guard** (`:33`, `weights = 1/distances`). A
  query point that coincides exactly with a snapshot gives `1/0 = inf`, then `inf/inf = nan` after
  normalisation → the surrogate returns NaN. This is not a corner case in DAMH: the current chain
  state is itself a snapshot and is re-evaluated by the surrogate
  (`algorithms.py:262,345`). See **S3**.
- `k > N` is handled (`:67`, `min(k, N)`), evaluated at `get_evaluator()` time so the cap tracks
  growth. `N == 0` would break, but `min_snapshots_initial ≥ 1` prevents it.
- Ties are resolved by `cKDTree` (deterministic but arbitrary); the surrogate is discontinuous.
- Tree rebuilt from scratch each `get_evaluator()`: O(N log N), acceptable.

### `torch_perceptron.py` — `NeuralNetworkUpdaterBasic` (renamed from `PyTorchNNOngoingUpdater`)

- Architecture: `Linear → act → (Linear → act)* → Linear`, activation only `relu` else **silently
  `tanh`** (`:44-49`); an unknown string does not raise.
- **No input normalisation, no output normalisation at all.** Loss is raw MSE on raw observations.
- Default solver `lbfgs`, `iterations_batch=100`, `loss_target=1e-5`.
- `add_data(..., train_on_added_data=True)` trains **only on the newly added batch**
  (`:252-268`), then `train()` (called by the collector) trains on everything. Training on a
  changing subset while keeping **LBFGS history across objective changes** is not sound — the
  L-BFGS curvature pairs refer to a different function. Catastrophic-forgetting risk from the
  per-batch fine-tuning.
- No validation split, no early stopping other than the `loss_target` threshold, no LR schedule,
  no gradient clipping, no weight decay.
- `initial_training` (`:207`) trains on synthetic `randn` parameters with constant observations and
  does **not** store them in `self.par/obs` — the opposite of the minibatches variant (see S6).
- Seeding: `torch.manual_seed(seed)` inside `PyTorchMLP.__init__` reseeds the **global** torch RNG
  and prints `SEED` unconditionally. `torch.cuda.manual_seed*` is called even on CPU.
- `device` is a plain string; `.to(self.device)` on the updater only. Evaluators are always cloned
  to CPU (`:59-66`) before being pickled — correct for MPI transport.
- Jacobian: reverse-mode, one `autograd.grad` per output row (`:95-104`) → `q` backward passes;
  float32 throughout.

### `torch_perceptron_minibatches.py` — `NeuralNetworkUpdaterMinibatches`

- Activation factory raises on unknown names ✓ (`:62`), supports silu/gelu/elu/leaky_relu/identity;
  supports `hidden_layer_sizes=()` (pure linear model).
- **Output normalisation is frozen user-supplied state**, `output_mean`/`output_scale` passed to
  `__init__` and never recomputed (`:214-225`). Applied consistently:
  - training targets normalised on ingest (`add_data:498`, `load_training_arrays:348`),
  - evaluator de-normalises on output (`:94`), in `jacobian` (`:115`) and `vjp` (`:150`),
  - `get_training_data_arrays` de-normalises before saving (`:324`),
  - checkpoint stores and validates both vectors (`:279-311`).
  Defaults are `zeros`/`ones`, i.e. **no normalisation unless the user supplies statistics** — and
  nothing in the library computes them from data.
- **No input normalisation.**
- Incremental retraining: the model and the AdamW optimiser state (moments) persist across updates;
  `train()` runs `iterations_batch` optimiser *steps* (not epochs) over all snapshots,
  `add_data(train_on_added_data=True)` runs steps over new data + a replay sample
  (`_get_replay_indices:386`, `replay_ratio=1.0`, optional `replay_max_old_samples`).
- Early stopping: only `batch_loss < loss_target`, measured on **one minibatch**
  (`_train_minibatches:444`) → noisy stopping criterion. **No validation split anywhere**, hence no
  train/val leakage, but also no generalisation signal inside the updater. Out-of-sample quality is
  measured externally by the collector on the *next* batch of snapshots
  (`process_COLLECTOR.py:168-182`) and optionally on a fixed test set.
- **No learning-rate schedule.** `weight_decay=1e-4` applies to biases as well.
- Loss: per-element MSE, mean over outputs, then **weighted by the snapshot weight**
  (`_weighted_loss:358`). Snapshot weights come from the sampler and are `0` for rejected proposals
  (`algorithms.py:188`) → see **S2**.
- Determinism: `self._rng = np.random.default_rng(seed)` drives batching/replay ✓; torch seeding is
  global and one-shot at construction. `torch.set_num_threads` is **never** called anywhere in the
  package → each MPI rank may spawn as many OMP threads as there are cores.
- Gradients: `jacobian` uses forward-mode `torch.func.jacfwd`, `vjp` uses reverse-mode autograd.
  Both temporarily cast the whole module with `self.model.double()` and restore with `.float()` in
  a `finally` (`:109-123`, `:143-159`). Inputs are first cast to float32 then promoted to float64
  (`:101`, `:131`), so the float64 pass starts from float32-rounded data. The `__call__` path stays
  float32, so the evaluation used for the likelihood and the one returned by `vjp` differ slightly.
- Pickling: `clone_model_to_cpu` rebuilds the module from `(input_size, output_size, hidden_layers,
  activation)` and `load_state_dict`s — so **only weights** travel, no optimiser state, no `.grad`
  buffers, no CUDA tensors, `eval()` set. `output_mean/scale` are pickled as small float32 arrays.
  This is the right design. Note the constructor call consumes global torch RNG (default init)
  before the state dict overwrites it.
- Checkpointing: `surrogate_checkpoint.pt` + `surrogate_training_data.npz`
  (`reuse.surrogate_state_paths`), `torch.load(..., map_location=...)` without `weights_only=True`
  (arbitrary-pickle load; acceptable for own files, worth noting).
- Duplication with the base variant: `PyTorchMLP`, `PyTorchNNEvaluator`, `_checkpoint_hparams`,
  `_validate_checkpoint_hparams`, `save/load_checkpoint`, `save/load_training_data`,
  `save/load_state`, `save_snapshots`, `get_loss_on_data` are near-copies under the **same class
  names in two modules**. ~250 duplicated lines.

### `reuse.py`

- Registry keyed by `cls.__name__`; only `NeuralNetworkUpdaterMinibatches` is registered.
- `SurrogateReused` reconstructs the updater from `checkpoint["updater_hparams"]`. Note the hparams
  dict stores `output_mean`/`output_scale` as **lists** while `_validate_checkpoint_hparams` expects
  to compare against arrays — the `np.asarray(...)` conversion handles it ✓.
- `updater.no_snapshots` is printed (`:39`) — only defined for the NN updaters.

## Distributions and transformations

Design: the chain always lives in an **internal** space; `prior.logpdf` is the density there and
`prior.transform` maps a sample to the solver's space. For `PriorIndependentComponents` the internal
prior is N(0, I) and `transform` is the per-component inverse-CDF map, so **no Jacobian correction
is needed** — the change of variables is absorbed by construction. This is correct and is used
consistently (`algorithms.py:154`, `:220`, `:237`; `test_data.py:44,50`).

- `parent.Distribution`: default `logpdf` returns `0.0` (silent improper flat prior if a subclass
  forgets it); `grad_logpdf`/`get_covariance`/`rvs` raise ✓; `self.mean = 0.0` scalar default.
- `parent.FromScipy`: binds scipy's `logpdf`/`rvs` directly (so it *includes* normalising constants,
  unlike `Normal`), never calls `super().__init__()` → **no `self.mean`**, no `grad_logpdf`,
  no `get_covariance`.
- `normal.Normal`: dispatches to uncorrelated/multivariate variants at construction. logpdf is
  `-0.5 (x-μ)ᵀΣ⁻¹(x-μ)` — **normalising constant dropped** (documented). Gradient
  `Σ⁻¹(μ-x)` ✓ matches. `calculate_logpdf_multivariate` calls `np.linalg.solve` on every
  evaluation — O(n³) per sample, no cached Cholesky. `get_covariance()` returns the **standard
  deviations** (not variances, not a covariance) in the uncorrelated branch; `PCN._set_prior_covariance`
  treats a 1-D input as sd, so the pairing is consistent, but the method name is wrong.
  `rvs` uses the global `np.random` (not a seeded generator).
- `independent_components.PriorIndependentComponents`: internal N(0,I) logpdf/grad/`get_covariance`
  (ones) / `rvs` all consistent ✓. `transform` is per-scalar and 1-D only. `self.sd_approximation`
  is dead state. Components renamed in the working tree (`Uniform → UniformComponent`, etc.) with
  backward-compatible aliases at `:134-137`; **`Normal = NormalComponent`** shadows the
  distribution-level `Normal` for anyone doing `from ...independent_components import *`, and
  `distributions/__init__.py` only gets the right `Normal` because `.normal` is imported *after*
  `.independent_components`. The aliases are not in `__all__`.
- `gaussian_mixture.GaussianMixture`: `logpdf` sums component pdfs and takes `log` — includes full
  normalising constants (inconsistent with `Normal`), and **underflows to `log(0) = -inf`** for
  samples far from every component (no log-sum-exp). `grad_logpdf` is the correct mixture score
  `Σ wᵢ Nᵢ Σᵢ⁻¹(μᵢ-x) / Σ wᵢ Nᵢ` but silently returns **zeros** when the mixture pdf underflows.
  `rvs(n=1)` returns shape `(1, d)` whereas every other `rvs` returns `(d,)`. No `mean`, no
  `transform` override (identity is inherited ✓), no `get_covariance`. `self.weights /= sum` fails
  on an integer weight list. A `multivariate_normal` object is re-created on every call in every
  loop iteration (expensive).
- `transformations.py`: forward maps are correct. `normal_to_uniform` saturates (`norm.cdf` hits
  exactly 1.0 around x ≈ 8.3), so extreme internal values map to the interval/beta boundary — the
  map loses injectivity in the far tail. `normal_to_lognormal` overflows to `inf` for large inputs.
  The inverse maps (`*_to_normal`, `beta_to_uniform`, `uniform_to_beta` apart from its use inside
  `normal_to_beta`) are **not used anywhere in the package** — public helpers/dead code; they return
  ±inf at the support boundary (`log(0)`, `ppf(0)`, `ppf(1)`).

## `Gaussian_process.py`, `lhs_normal.py`, `test_data.py`

### `Gaussian_process.assemble_covariance_matrix`

Builds a block-diagonal covariance for a zero-mean Gaussian process on a time grid (exponential or
squared-exponential correlation). **Not referenced anywhere in the package** (`git grep` finds only
its own `__main__`); it is a helper for user scripts building likelihood covariances.

Correctness issue: `block = variance * autocorr(...)` with `std` reshaped to `(1, n)` (`:49`)
multiplies **columns only**, giving `σⱼ² ρᵢⱼ` instead of `σᵢ σⱼ ρᵢⱼ`. For a **vector** `std` the
result is **not symmetric and not a valid covariance matrix**. Scalar `std` is fine (the demo in
`__main__` uses a constant vector, which is symmetric by accident). See **S8**.

### `lhs_normal.lhs_normal`

Used at `process_SAMPLER.py:58` for `initial_sample_type == "lhs"`.

- `maxmin` is initialised to 0 and **never updated** (`:35`, `:46-48`), so the "best of 5 candidate
  designs by maximin distance" selection degenerates to "keep the last candidate". Confirmed by
  reading.
- The inner loop reuses the name `i` from the outer loop (`:32` vs `:36`). The outer loop still runs
  5 times (the loop variable is reassigned per iteration), so this is only a readability trap.
- `LHS_final = np.zeros([n, n])` (`:29`) has the wrong shape (`(n, no_parameters)` intended); it is
  always overwritten, so it is latent.
- `numpy.matlib` is deprecated; `repmat` could be replaced by broadcasting. Removal in a future
  numpy would break this import — **needs verification** against the pinned numpy.
- `seed=0` is hard-coded at the call site → every run starts from the same LHS design.
- Relies on `prior.mean` having `len()`; fails for `FromScipy`/`GaussianMixture` priors.

### `test_data.TestData`

- `generate` draws test parameters from the **prior**, saves/restores the global numpy RNG state ✓,
  builds `surrogate_parameters` respecting `conf.transform_before_surrogate` ✓, and always calls the
  solver on `prior.transform(p)` ✓. So the test inputs live in surrogate space and the reference
  observations in the solver's original observation space — matching what the collector feeds the
  evaluator.
- `as_surrogate_test_data()` takes **no arguments** (`:98`) but `core.py:124` calls it with
  `prior=`/`likelihood=` → see **S1**.
- The surrogate error metric is computed in `process_COLLECTOR._compute_surrogate_quality_metrics`
  (`:78-92`): plain RMSE and max-abs-error **in original observation units, unweighted across
  output components** (so large-magnitude observations dominate), plus a self-normalised
  **posterior-weighted** RMSE / MAE using `exp(log_posterior - max)` weights.
  With prior-drawn test points and a concentrated posterior, those weights have a tiny effective
  sample size (default `size=16`), so `weighted_rmse` can be essentially the error at one point.
  No ESS is recorded. See **S10**.
- `save`/`reuse` deliberately do not persist `log_posterior`/`weights`, so a `reuse()`-d `TestData`
  raises in `as_surrogate_test_data()` until the caller recomputes; same for `join`/`reduce_size`
  (documented in comments, but easy to trip over).
- `compute_log_posterior_and_weights` uses `prior.logpdf(internal parameters)` ✓ consistent with the
  sampler; because `Normal.logpdf` drops the normalising constant, the stored `log_posterior` and
  the `max_log_posterior` column of `surrogate_quality_test.csv` are **shifted by an unknown
  additive constant** (the weights are unaffected, being shift-invariant).

## Potential bugs and risks

| ID | Sev | Location | Description | Why it matters | Confidence | How to verify |
|---|---|---|---|---|---|---|
| **S0** | **high** | `surrDAMH/core.py:154,158,163,164` | `temptemptemp` annotates parameters with `Iterable` and `Any`, neither of which is imported (`core.py:6` imports only `List, Literal`); there is no `from __future__ import annotations`, so annotations are evaluated at def time. | `import surrDAMH.core` raises `NameError` → the whole package is unimportable in the current working tree. | confirmed by reading | `python -c "import surrDAMH.core"` |
| **S1** | **high** | `core.py:124` vs `modules/test_data.py:98` | `self.surrogate_test_data.as_surrogate_test_data(prior=self.prior, likelihood=self.likelihood)` — the method signature is `(self)`. | `TypeError` on the collector rank whenever a `TestData` object is passed → run dies at start-up (only when `surrogate_test_data` is a `TestData`, not a tuple). | confirmed by reading | pass a `TestData` to `SamplingFramework` and start a 3-rank run |
| **S2** | **high** | `torch_perceptron_minibatches.py:358-367` + `algorithms.py:188` | Rejected proposals are sent to the collector with `weight=0`; `_weighted_loss` normalises by the weight sum, so zero-weight snapshots contribute **nothing** to the NN loss while still occupying memory, replay slots and minibatch capacity. | Expensive full-solver evaluations at exactly the boundary of the high-probability region are discarded from training → worse surrogate where it matters, and wasted steps. | confirmed by reading | count zero-weight rows in a saved `surrogate_training_data.npz`; compare NN error with weights clamped to `min=1` |
| **S3** | **high** | `nearest_kdtree.py:33` | `weights = 1/distances` with no zero guard; a query at an existing snapshot gives `inf`, then `inf/inf = NaN`. | Surrogate returns NaN → `log_likelihood_approx` NaN → DAMH pre-acceptance comparisons all false → chain freezes. Happens whenever `k>1` and the chain re-evaluates a point it already sent as a snapshot (`algorithms.py:262,345`). | confirmed by reading | `KDTreeUpdater(no_nearest_neighbors=5)`, query one of the training points |
| **S4** | medium | `parent.py:45-56` + `algorithms.py:284-290` | `Evaluator.vjp` default calls `self.jacobian(datapoint)` and unpacks **two** return values, while the `jacobian` docstring promises a single array of shape `(n,q,p)`. `algorithms.py` catches `NotImplementedError` from `vjp` and then calls `jacobian` again — which raises the same `NotImplementedError` uncaught. | Any new evaluator written to the documented contract breaks; the documented fallback path is dead. | confirmed by reading | implement a toy `Evaluator` with only `jacobian` per the docstring and run HMC/gradient path |
| **S5** | medium | `torch_perceptron.py` (whole file) | `NeuralNetworkUpdaterBasic` is not `@register_updater`-ed, reports `supports_state_persistence()/supports_training_data_persistence() == False` while implementing both, and does not override `get_initial_snapshots()`. | `SurrogateReused` cannot restore it; a restarted run silently starts the collector with 0 preloaded snapshots even though the updater holds the loaded data. | confirmed by reading | `SurrogateReused` on a Basic checkpoint → `ValueError: not registered` |
| **S6** | medium | `torch_perceptron_minibatches.py:478-488` | `initial_training` **overwrites** `self.par/obs/weights` with `n=1000` synthetic `randn` rows carrying a constant observation and sets `no_snapshots = n`. | Those fake rows stay in the training set forever, are replayed, are counted as snapshots, and are written verbatim into `surrogate_training_data.npz` as if they were solver evaluations. The Basic variant does the opposite (does not store them) — divergent semantics under the same name. | confirmed by reading | call `initial_training` then `get_training_data_arrays()` |
| **S7** | medium | `rbf_scipy.py:67-74` | The singular-matrix fallback duplicates the dataset at `par+1, par+2, …` with unchanged observations and refits with a linear kernel. | Imposes `f(x) = f(x+k·𝟙)` — a badly wrong constraint; the surrogate silently degrades instead of failing. The fallback can also raise again on exact duplicates. | confirmed by reading | feed duplicated snapshots and inspect the fitted values |
| **S8** | medium | `modules/Gaussian_process.py:49-61` | `variance * autocorr` with a **vector** `std` scales columns only → `σⱼ²ρᵢⱼ`, an asymmetric, non-PSD matrix. Should be `np.outer(std, std) * corr`. | If a user builds a likelihood covariance with a non-constant `std`, the likelihood is wrong → biased posterior. | confirmed by reading | `assemble_covariance_matrix` with `std=[1,2,3]`, check `C == C.T` |
| **S9** | medium | `gaussian_mixture.py:38-62`, `:24-36`, `:64-74` | No log-sum-exp (underflow → `-inf`, and `grad_logpdf` returns zeros); `rvs(n=1)` returns `(1,d)` while all callers expect `(d,)`; no `mean` attribute (no `super().__init__()`); no `get_covariance`; integer `weights` crash the in-place normalisation. | A `GaussianMixture` prior breaks `process_SAMPLER.py:58` (`lhs`), `PCN` construction (`prior.mean`), and produces a `(1,d)` initial sample under `initial_sample_type="prior"`. Zero gradients silently corrupt HMC. | confirmed by reading | use `GaussianMixture` as prior in `toy_examples/minimal_example.py` |
| **S10** | medium | `modules/lhs_normal.py:35,46-48` | `maxmin` never updated → "best of 5 maximin designs" always keeps the last candidate. | The LHS start points are worse-spread than intended; silently. | confirmed by reading | assert the returned design equals the 5th candidate for a fixed seed |
| **S11** | low-med | `torch_perceptron_minibatches.py:109,123,143,159` | `self.model.double()` / `.float()` on every `jacobian`/`vjp` call mutates the shared evaluator module. | With HMC (many leapfrog steps) the whole network is cast twice per gradient; also not re-entrant/thread-safe, and an exception path inside `finally` can leave the module float64 if the restore throws. | confirmed by reading | time `vjp` vs a float64-resident copy |
| **S12** | low-med | `torch_perceptron*.py` — `__call__` returns a flattened array | Contract says `(n, q)`. Callers `.ravel()`/`.reshape()`, so it works today — but `Evaluator.as_solver()` (used by `stage.use_only_surrogate`) returns `(n*q,)` for NN and `(1,q)` for the poly/RBF/kd evaluators, i.e. **two different shapes** to the same consumer. | Latent shape bug in the `use_only_surrogate` path. | confirmed by reading for the shapes; downstream impact **suspected** | run `use_only_surrogate` with a polynomial vs an NN surrogate and compare `Sample.observations.shape` |
| **S13** | low-med | `torch_perceptron.py:252-268` | `add_data(train_on_added_data=True)` (the default) fine-tunes on the new batch only, carrying **LBFGS** state across a changing objective. | Catastrophic forgetting and invalid L-BFGS curvature history → possible divergence. | confirmed by reading | track MSE on early snapshots across updates |
| **S14** | low-med | `polynomial_sklearn.py:49` | `add_data` hard-assigns `weights = None`, discarding the collector-supplied weights, and the `weights=None` default silently overrides the parent contract (`weights` is required in `parent.py:85`). | Dead parameter, misleading; `self.wei` is pure dead state. | confirmed by reading | read |
| **S15** | low-med | `rbf_scipy.py` (whole class) | No snapshot pruning; a global O(N³) solve at every surrogate update. | On long runs the collector becomes the bottleneck and may OOM; no warning. | confirmed by reading | time `get_evaluator()` vs N |
| **S16** | low | `parent.py:142-157` vs both NN updaters | Child `save_state(checkpoint_path, data_path)` drops `save_optimizer` and makes `data_path` mandatory. | Calling through the parent-declared API (`save_state(path)` or `save_optimizer=False`) raises `TypeError`; `save_optimizer` is honoured nowhere. | confirmed by reading | call `updater.save_state(p)` with one argument |
| **S17** | low | `torch_perceptron*.py` `PyTorchMLP.__init__` | `torch.manual_seed(seed)` reseeds the **global** torch RNG (and `torch.cuda.manual_seed*` is called on CPU-only runs); `print("SEED", seed)` is unconditional. | Reproducibility is global, not per-model; noisy logs. `clone_model_to_cpu` also consumes global RNG on every evaluator creation. | confirmed by reading | read |
| **S18** | low | package-wide | `torch.set_num_threads` is never called; no OMP thread capping. | With `mpiexec -n k`, each rank may use all cores → oversubscription, slower sampling. | confirmed by reading (grep) | `OMP_NUM_THREADS=1` comparison run |
| **S19** | low | `independent_components.py:137`, `distributions/__init__.py` | `Normal = NormalComponent` alias shadows the distribution-level `Normal`; `__init__` only resolves correctly because of import ordering. Aliases absent from `__all__`. | Confusing/fragile import surface. | confirmed by reading | reorder the imports in `__init__.py` and watch the breakage |
| **S20** | low | `normal.py:60-69` | `np.linalg.solve` on every `logpdf`/`grad_logpdf` of a correlated Normal. | O(n³) per MCMC step for the likelihood; a cached Cholesky would be a free speed-up. | confirmed by reading | profile `likelihood.logpdf` |
| **S21** | low | `modules/tools.py:1-11`, `parent.py:160-177` | Duplicated `import os`/`import numpy as np`; `closest_point_distance`, `closest_point_distance_kdtree`, `PriorIndependentComponents.sd_approximation`, `Updater.supports_*_persistence` (no caller), the `*_to_normal` transformations and `evaluate_on_a_grid` debug calls (`algorithms.py:419,493`) are dead code. | Maintenance noise; `algorithms.py:419` writes PNGs into the CWD during a sampling run. | confirmed by reading | grep |
| **S22** | low | `reuse.py:26`, both `load_checkpoint` | `torch.load` without `weights_only=True`. | Arbitrary-code execution when loading a foreign checkpoint; newer torch versions will change the default and may start refusing these files. | confirmed by reading | read |

## Suggested improvements

1. Fix **S0** and **S1** first — the working tree is, as read, not importable, and the `TestData`
   path is broken.
2. Make the `Evaluator` contract unambiguous: either declare `jacobian(datapoint) -> (J, evaluation)`
   for a single point (what the code actually does) or keep the batched docstring and add a separate
   `jacobian_and_value`. Update `Evaluator.vjp` and `algorithms.py:287-290` to match. Add a
   `no_observations` attribute to the base `Evaluator`.
3. Standardise `__call__` to return `(n, q)` in the torch evaluators and drop the `.ravel()`/
   `.reshape()` compensation in the callers (single, reviewed change — it touches the DAMH path, so
   check the acceptance rate is unchanged on a toy example before/after).
4. `KDTreeEvaluator`: clamp distances (`np.maximum(d, eps)`) or short-circuit exact hits.
5. `RBFInterpolationUpdater`: de-duplicate snapshots before fitting (round/`np.unique` on rows) or
   set a small default `smoothing`; add an optional cap on the number of snapshots (subsample or
   `neighbors=k`); replace the `par+i+1` fallback.
6. `PolynomialSklearnUpdater`: standardise inputs inside the pipeline
   (`StandardScaler → PolynomialFeatures → Ridge`) and refuse to fit until `num_snapshots ≥ num_terms`.
7. Collapse the two torch modules into one (shared `PyTorchMLP`, `PyTorchNNEvaluator`,
   checkpoint mixin), keep `NeuralNetworkUpdaterBasic` as a thin configuration of the minibatches
   updater (`batch_size=None, solver="lbfgs", replay_ratio=0`) — or delete it, it has no users.
   Register it, or drop its half-implemented persistence.
8. Give the minibatches updater an optional "estimate `output_mean`/`output_scale` from the first
   `N` snapshots and then freeze" mode, plus input normalisation — today the defaults mean *no*
   normalisation, which the file name suggests is handled.
9. Re-examine the zero-weight rule (S2): clamping weights to `max(w, 1)` for surrogate training, or
   a separate `training_weight` distinct from the MCMC multiplicity weight.
10. `GaussianMixture`: use `scipy.special.logsumexp`, precompute frozen `multivariate_normal`
    objects, add `mean`/`get_covariance`, make `rvs()` return `(d,)` for `n=1`.
11. `Gaussian_process`: `np.outer(std, std) * corr`.
12. `lhs_normal`: fix `maxmin`, rename the inner loop variable, drop `numpy.matlib`, allow a seed
    argument from `Configuration`.
13. Optional but valuable: state in the `Distribution` docstring whether `logpdf` includes
    normalising constants, and make the choice uniform — right now `Normal` drops them,
    `GaussianMixture` and `FromScipy` keep them, and `TestData` adds the two together.

## What should be tested or validated

No test for any of this exists today (`tests/` contains only `test_best_fit_ranking.py`).

**Interface / smoke**

1. `test_import_core`: `import surrDAMH` — would have caught S0.
2. Parametrised over all five updaters: build with `no_parameters=3, no_observations=2`, `add_data`
   20 random snapshots with weights, `train()`, `get_evaluator()`, then assert
   `evaluator(X).reshape(n, 2)` is finite and that `pickle.loads(pickle.dumps(evaluator))` gives
   bit-identical outputs (this is what MPI does).
3. Assert the evaluator output shape is exactly `(n, no_observations)` for every surrogate
   (will fail today for both torch variants — S12).

**Surrogates**

4. Polynomial: fit on an exact quadratic `y = x₀² + x₁`, 200 snapshots → RMSE < 1e-8; assert the
   printed degree escalation happens at the documented snapshot counts.
5. KDTree with `k=5`: query **one of the training points** → assert finite output (fails today, S3).
6. RBF with an exactly duplicated snapshot → assert the interpolator still reproduces the training
   observations at the training points (probes S7).
7. MLP normalisation round-trip: `normalize_outputs(denormalize_outputs(y)) == y` within float32
   tolerance; and with non-trivial `output_mean/scale`, fit `y = x²` on 500 points and assert the
   *de-normalised* evaluator output matches the raw targets (catches any train/eval normalisation
   asymmetry).
8. Gradient check: `evaluator.jacobian(x)[0]` and `evaluator.vjp(x, v)[0]` versus central finite
   differences of `evaluator(x)` (step 1e-4, float64 model) — relative error < 1e-4; assert
   `vjp(x,v)[0] ≈ jacobian(x)[0].T @ v`.
9. Gradient shape convention: assert `jacobian(x)[0].shape == (no_observations, no_parameters)` and
   that `jacobian` raises on a batched input.
10. Checkpoint round-trip: train, `save_state`, construct a fresh updater, `load_state`, assert the
    evaluator outputs match exactly and that `get_initial_snapshots()` returns the saved snapshots
    (fails today for the Basic updater, S5).
11. `SurrogateReused` on a minibatches checkpoint restores `no_snapshots` and hparams; on a Basic
    checkpoint it currently raises — pin whichever behaviour is intended.
12. Weighted-loss behaviour: two identical datasets, one with all weights 1 and one with half the
    weights 0 → assert the trained models differ (documents S2 rather than asserting it is right).

**Distributions**

13. `Normal.logpdf` vs `scipy.stats.multivariate_normal.logpdf` **up to an additive constant**:
    assert the *differences* between two samples agree to 1e-10, for both the sd and the cov branch.
14. `Normal.grad_logpdf` vs finite differences of `logpdf`.
15. `GaussianMixture.logpdf` vs `logsumexp` of the components including constants; and a far-away
    sample (e.g. `10³·𝟙`) → assert it is finite, not `-inf` (fails today, S9).
16. `GaussianMixture.grad_logpdf` vs finite differences, including in the underflow regime.
17. `rvs()` shape contract for every distribution: exactly `(no_parameters,)` (fails today for
    `GaussianMixture`, S9).
18. Transform round-trip for each component: `uniform_to_normal(normal_to_uniform(x)) ≈ x` for
    `x ∈ [-5, 5]`; same for lognormal and beta.
19. Empirical marginal check: 200 000 `transform(rvs())` draws through `PriorIndependentComponents`
    → Kolmogorov–Smirnov against the component's `pdf`/cdf, p > 0.01. This is the single most
    valuable test for the prior, since a wrong transform biases every posterior.
20. `PriorIndependentComponents.logpdf` is exactly the standard normal log-density (no Jacobian
    term) — pin this explicitly as the documented design, so nobody "fixes" it later.

**Helpers**

21. `assemble_covariance_matrix` with a vector `std`: assert symmetry and positive semi-definiteness
    (fails today, S8).
22. `lhs_normal`: assert one point per stratum per dimension, and that the returned design is the
    maximin-best of the 5 candidates (fails today, S10).
23. `TestData.generate` does not perturb the global numpy RNG stream; `save`→`reuse`→
    `compute_log_posterior_and_weights`→`as_surrogate_test_data` works end-to-end.
24. `as_surrogate_test_data()` called the way `core.py` calls it (catches S1).

## Open questions for the author

1. **Zero-weight snapshots (S2):** is it intended that rejected proposals (`weight=0`) contribute
   nothing at all to the NN loss? What fraction of the snapshots in the TSX runs carry weight 0?
2. **`output_mean`/`output_scale`:** these are frozen user inputs and default to identity. Where are
   the values in the TSX experiments computed, and is there a reason not to estimate them from the
   first batch of snapshots inside the updater?
3. **Input normalisation:** deliberately absent because the internal prior is already N(0, I)
   (`transform_before_surrogate=False`)? If so, it is worth a comment in the file — it also means the
   NN surrogates are effectively unusable with `transform_before_surrogate=True`.
4. **`NeuralNetworkUpdaterBasic`:** still needed, or superseded by the minibatches variant? It has no
   in-repo callers and its persistence is half-wired (S5).
5. **`jacobian` return convention:** should the docstring in `parent.py` be corrected to the
   `(J, evaluation)` tuple the code returns, or should the implementations be changed?
6. **RBF/kd-tree/polynomial:** are these still used for real experiments, or only in
   `toy_examples/`? That decides how much S3/S7/S15 matter.
7. **`initial_training` (S6):** is injecting 1000 synthetic constant-observation rows into the
   training set (and into the saved `.npz`) intentional?
8. **`Gaussian_process.assemble_covariance_matrix`:** is it used by the TSX likelihood construction
   with a non-constant `std`? If yes, S8 affects published posteriors.
9. **Fixed test set (S10 / `TestData`):** should the reported weighted RMSE carry an effective
   sample size, and should test points be drawn from something closer to the posterior than the
   prior (e.g. a pilot chain)?
10. **`float64` casting in `jacobian`/`vjp`:** the comment says it reduces chain-rule accumulation
    error. Was that measured? The input is float32-rounded first, so the benefit is limited to the
    forward accumulation.

---

### Method notes

- Everything above was obtained by reading files and by `git grep` / `git diff` on named files.
- One deviation from the read-only rule: I ran a single `python3 -c` one-liner to check
  `numpy.linalg.LinAlgError.__mro__`, which confirmed it subclasses `ValueError` in this
  environment — that is what makes the `except ValueError` fallback in `rbf_scipy.py:67` actually
  catch singular-matrix failures. `scipy` is **not installed** in this container, so nothing in
  `surrDAMH` could have been executed here anyway.
- Not verified (no execution): the exact exception type scipy's `RBFInterpolator` raises for
  duplicate/near-duplicate points and for a too-low `degree`; whether `numpy.matlib` still imports
  cleanly under the pinned numpy; and every runtime claim marked "suspected" above.
