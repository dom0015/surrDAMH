# Writing a surrogate

A surrogate approximates the forward model to make DAMH's sub-chains cheap (and,
optionally, to provide gradients for Hamiltonian-family proposals). Two roles:

- **`Updater`** owns the training data and the trainable state; runs on the collector
  rank (`process_COLLECTOR.run_COLLECTOR`) or in-process for `run_local`'s
  `LocalSurrogateManager`. Passive: the caller decides *when* to call `add_data`/
  `train`/`get_evaluator`, governed by `Configuration.min_snapshots_initial`/
  `min_snapshots_to_update`.
- **`Evaluator`** is what a sampler actually calls; produced by `Updater.get_evaluator()`
  and picklable so it can be shipped from the collector to every sampler rank over MPI
  each time it changes.

## The contract (WS6, implemented)

Every shipped updater (`PolynomialSklearnUpdater`, `RBFInterpolationUpdater`,
`KDTreeUpdater`, `NeuralNetworkUpdaterMinibatches`) obeys the following; a custom one is
expected to as well. The file:line-referenced conformance table is
`library_notes/12_evaluator_contract_spec.md`.

### `Evaluator`

| Method | Contract |
|---|---|
| `__call__(X)` | `X: (n, no_parameters)` → **`(n, no_observations)`, always 2-D, including `n == 1`**. There is no single-point overload: call `evaluator(x.reshape(1, -1))[0]`. |
| `jacobian(x)` | `x: (no_parameters,)` → `(J: (no_observations, no_parameters), y: (no_observations,))`. Single point only: a batched argument raises `ValueError`. `NotImplementedError` if the evaluator cannot differentiate. |
| `vjp(x, v)` | `→ (J(x).T @ v, y)`. The base class derives it from `jacobian`; override it if a reverse-mode path is cheaper. |
| `supports_gradients()` | Whether this instance provides derivatives. |
| `set_use_gradients(enabled)` | Optional hook; silently no-ops for evaluators without gradients. |
| `no_parameters`, `no_observations` | Real attributes, set by `Evaluator.__init__`. **Call `super().__init__(no_parameters, no_observations)`.** |

`SurrogateAsSolver` (used for `Stage.use_only_surrogate=True`) turns the `(1, q)` batch
result into the `(q,)` a `Solver` must return.

### `Updater`

| Method | Contract |
|---|---|
| `add_data(parameters, observations, multiplicity=None)` | `(n, no_parameters)`, `(n, no_observations)`, `(n, 1)`. |
| `train()` | Called periodically by the collector, regardless of whether new data arrived. Updaters that refit inside `get_evaluator()` leave it a no-op. |
| `get_evaluator()` | Returns a **picklable** `Evaluator`. Called whenever a sampler's request is served after enough new data arrived. |
| `supports_sample_weights` | Class attribute: whether the fit honours per-row weights. |
| `set_output_normalization(mean, scale)` | Optional hook, see below. |
| `save_state` / `load_state` / `supports_*_persistence()` | Checkpointing, see "Checkpoint reuse". |

### Snapshot multiplicity and the `weighting` option

Every snapshot arrives with a **multiplicity**: `1 + rejections` for a chain state that
was just left, and `0` for a rejected proposal (`modules/algorithms.py`). What the fit
does with it is a per-updater option, `weighting: Literal["uniform", "multiplicity"]`,
**default `"uniform"` for every updater including the neural networks**:

| `weighting` | Rows with multiplicity 0 | Rows with multiplicity `m > 0` |
|---|---|---|
| `"uniform"` (default) | used, weight 1 | used, weight 1 |
| `"multiplicity"` | dropped everywhere (never stored) | weighted by `m` if `supports_sample_weights`, otherwise used once |

The policy is implemented once in the base class (`Updater._rows_to_use` /
`Updater._training_weights`); a concrete updater only applies the mask in `add_data` and
hands the weights to its fit. The collector logs the chosen policy at start-up.

| Updater | `supports_sample_weights` | How `"multiplicity"` reaches the fit |
|---|---|---|
| `PolynomialSklearnUpdater` | `True` | `LinearRegression.fit(..., sample_weight=...)` |
| `NeuralNetworkUpdaterMinibatches` | `True` | per-sample weighted loss (`_weighted_loss`) |
| `RBFInterpolationUpdater` | `False` | interpolant — zero-multiplicity rows are dropped, the rest count once |
| `KDTreeUpdater` | `False` | interpolant — same |

### Output normalization (`NeuralNetworkUpdaterMinibatches`)

`output_normalization: Literal["identity", "likelihood", "manual"] = "likelihood"`:

- **`"likelihood"` (default)** — the training targets are centred on the observed data
  (`likelihood.mean`) and scaled by the per-observation noise standard deviation
  (`likelihood.sd` broadcast to `no_observations`, or `sqrt(diag(likelihood.cov))`).
  `SamplingFramework.__init__` and `run_local` call
  `Updater.set_output_normalization(mean, scale)` once for it. If the likelihood cannot
  supply usable statistics, a `RuntimeWarning` naming the likelihood class is issued and
  the updater stays at the identity.
- **`"manual"`** — uses the explicit `output_mean`/`output_scale` arguments.
- **`"identity"`** — no normalization (mean 0, scale 1); `output_mean`/`output_scale`
  must then be `None`.

`output_normalization_provenance` records what is actually in effect
(`"identity"`/`"likelihood"`/`"manual"`); it goes into the checkpoint and into the run
manifest's `surrogate` block.

## Minimal custom `Updater`

```python
class MyUpdater(surrDAMH.surrogates.parent.Updater):
    def __init__(self, no_parameters, no_observations, weighting="uniform"):
        super().__init__(no_parameters, no_observations, weighting=weighting)
        self.par = np.empty((0, no_parameters))
        self.obs = np.empty((0, no_observations))

    def add_data(self, parameters, observations, multiplicity=None):
        parameters = parameters.reshape(-1, self.no_parameters)
        observations = observations.reshape(-1, self.no_observations)
        mask = self._rows_to_use(multiplicity, parameters.shape[0])
        self.par = np.vstack((self.par, parameters[mask]))
        self.obs = np.vstack((self.obs, observations[mask]))
        self.no_snapshots = int(self.par.shape[0])

    def get_evaluator(self):
        # fit whatever you like on self.par/self.obs, return an Evaluator whose
        # __call__ maps (n, no_parameters) -> (n, no_observations)
        ...
```

`get_evaluator()` does not need to be idempotent between calls, but must return a
picklable `Evaluator`.

## Constructor reference (concrete updaters)

Every updater additionally accepts `weighting="uniform" | "multiplicity"`.

| Updater | Key constructor arguments |
|---|---|
| `PolynomialSklearnUpdater(no_parameters, no_observations, max_degree=5)` | Degree grows automatically as snapshots accumulate; supports sample weights. |
| `RBFInterpolationUpdater(no_parameters, no_observations, neighbors=None, smoothing=0.0, kernel="thin_plate_spline", epsilon=None, degree=None, verbose=False)` | Forwarded to `scipy.interpolate.RBFInterpolator`; refits from scratch every call (O(N³)); duplicated snapshot locations trigger a shifted-copy fallback that changes the fit, not just its cost. |
| `KDTreeUpdater(no_parameters, no_observations, no_nearest_neighbors)` | Inverse-distance-weighted average of `no_nearest_neighbors` (`1` = plain nearest-neighbor). |
| `NeuralNetworkUpdaterMinibatches(no_parameters, no_observations, hidden_layer_sizes=(100,), solver="adamw", activation="silu", learning_rate=1e-3, iterations_batch=100, batch_size=None, replay_ratio=1.0, train_on_added_data=False, output_normalization="likelihood", output_mean=None, output_scale=None, seed=None, ...)` | Minibatch training with a replay buffer (`replay_ratio` mixes in old snapshots); registered for checkpoint reuse (`surrogates.reuse.SurrogateReused`); the only updater that normalizes its targets. |

`NeuralNetworkUpdaterBasic` was **deleted** in WS6 (decision 4). Its full-batch L-BFGS
behaviour is available as a preset of the Minibatches updater, which is what
`toy_examples/neural_network_surrogate.py` and `toy_examples/sampling_TSX.py` now use:

```python
surrDAMH.surrogates.NeuralNetworkUpdaterMinibatches(
    no_parameters=..., no_observations=...,
    solver="lbfgs",             # one batch = the whole available subset
    batch_size=None,
    replay_ratio=0.0,           # no replay: every step already sees all data
    train_on_added_data=False,  # only the collector's periodic train() calls fit
)
```

## Checkpoint reuse

`surrDAMH.surrogates.reuse.SurrogateReused(experiment_folder)` reconstructs a registered
updater from `sampling_output/surrogate_checkpoint.pt` +
`surrogate_training_data.npz`, using the hyperparameters stored at save time (override
any of them via keyword arguments). All four shipped updaters are registered, but only
`NeuralNetworkUpdaterMinibatches` implements the persistence itself
(`supports_state_persistence()` / `supports_training_data_persistence()` report which).
Checkpoints are read with `torch.load(..., weights_only=True)`, so nothing but tensors,
scalars, strings and plain containers may be stored in them.
