import os
from typing import Literal
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from surrDAMH.surrogates.parent import Evaluator, Updater, WeightingPolicy
from surrDAMH.surrogates.reuse import register_updater

OutputNormalization = Literal["identity", "likelihood", "manual"]
OUTPUT_NORMALIZATIONS: tuple[str, ...] = ("identity", "likelihood", "manual")


class PyTorchMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation, seed):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = tuple(hidden_layers)
        self.activation = activation
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            print("SEED", seed)

        if len(self.hidden_layers) == 0:
            self.network = nn.Sequential(nn.Linear(input_size, output_size))
            return

        layers: list[nn.Module] = [
            nn.Linear(input_size, self.hidden_layers[0]),
            self.create_activation_layer(activation),
        ]
        for i in range(1, len(self.hidden_layers)):
            layers.append(nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
            layers.append(self.create_activation_layer(activation))
        layers.append(nn.Linear(self.hidden_layers[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    @staticmethod
    def create_activation_layer(activation: str) -> nn.Module:
        activation_name = activation.lower()
        if activation_name == "relu":
            return nn.ReLU()
        if activation_name == "tanh":
            return nn.Tanh()
        if activation_name in ("silu", "swish"):
            return nn.SiLU()
        if activation_name == "gelu":
            return nn.GELU()
        if activation_name == "elu":
            return nn.ELU()
        if activation_name in ("leaky_relu", "lrelu"):
            return nn.LeakyReLU()
        if activation_name in ("identity", "linear", "none"):
            return nn.Identity()
        raise ValueError(
            f"Unsupported activation {activation!r}. Supported: relu, tanh, silu, gelu, elu, leaky_relu, identity"
        )


class PyTorchNNEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, model,
                 output_mean: npt.NDArray, output_scale: npt.NDArray,
                 use_gradients: bool = True):
        super().__init__(no_parameters, no_observations)
        self.model = self.clone_model_to_cpu(model)
        self.output_mean = np.asarray(output_mean, dtype=np.float32).reshape(self.no_observations)
        self.output_scale = np.asarray(output_scale, dtype=np.float32).reshape(self.no_observations)
        self.use_gradients = use_gradients

    def clone_model_to_cpu(self, model):
        model_clone = model.__class__(
            model.input_size,
            model.output_size,
            model.hidden_layers,
            model.activation,
            seed=None,
        )
        model_clone.load_state_dict(model.state_dict())
        return model_clone.to("cpu").eval()

    def __call__(self, datapoints: npt.NDArray):
        """Evaluates the surrogate model: ``(n, no_parameters) -> (n, no_observations)``."""
        with torch.no_grad():
            datapoints_tensor = torch.tensor(datapoints, dtype=torch.float32, device="cpu").reshape(-1, self.no_parameters)
            outputs = self.model(datapoints_tensor)
            outputs = outputs.detach().cpu().numpy().reshape(-1, self.no_observations)
            outputs = outputs * self.output_scale.reshape(1, -1) + self.output_mean.reshape(1, -1) # overflow encountered
            return outputs

    def jacobian(self, datapoints: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        if not self.use_gradients:
            raise RuntimeError("Surrogate gradients are disabled for this PyTorch evaluator")

        datapoints_array = np.asarray(datapoints, dtype=np.float32)
        if datapoints_array.shape != (self.no_parameters,):
            raise ValueError(
                f"Expected datapoints with shape ({self.no_parameters},), got {datapoints_array.shape}"
            )

        # Cast model weights to float64 for Jacobian computation to reduce
        # chain-rule floating-point accumulation errors across layers.
        self.model.double()
        try:
            scale_tensor = torch.tensor(self.output_scale, dtype=torch.float64, device="cpu")
            mean_tensor = torch.tensor(self.output_mean, dtype=torch.float64, device="cpu")

            def model_fn(x: torch.Tensor) -> torch.Tensor:
                return self.model(x).reshape(self.no_observations) * scale_tensor + mean_tensor

            x = torch.tensor(datapoints_array, dtype=torch.float64, device="cpu")
            # Forward-mode AD: requires no_parameters=45 passes instead of
            # no_observations=72 passes that reverse-mode would need.
            jac = torch.func.jacfwd(model_fn)(x)  # shape: (no_observations, no_parameters)
            evaluation = model_fn(x)
        finally:
            self.model.float()

        return jac.detach().cpu().numpy(), evaluation.detach().cpu().numpy()

    def vjp(self, datapoint: npt.NDArray, vector: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        if not self.use_gradients:
            raise RuntimeError("Surrogate gradients are disabled for this PyTorch evaluator")

        datapoint_array = np.asarray(datapoint, dtype=np.float32) # overflow encountered
        if datapoint_array.shape != (self.no_parameters,):
            raise ValueError(
                f"Expected datapoint with shape ({self.no_parameters},), got {datapoint_array.shape}"
            )

        vector_array = np.asarray(vector, dtype=np.float32)
        if vector_array.shape != (self.no_observations,):
            raise ValueError(
                f"Expected vector with shape ({self.no_observations},), got {vector_array.shape}"
            )

        self.model.double()
        try:
            scale_tensor = torch.tensor(self.output_scale, dtype=torch.float64, device="cpu")
            mean_tensor = torch.tensor(self.output_mean, dtype=torch.float64, device="cpu")
            x = torch.tensor(datapoint_array, dtype=torch.float64, device="cpu", requires_grad=True)
            vector_tensor = torch.tensor(vector_array, dtype=torch.float64, device="cpu")

            evaluation = self.model(x).reshape(self.no_observations) * scale_tensor + mean_tensor
            gradient = torch.autograd.grad(
                evaluation,
                x,
                grad_outputs=vector_tensor,
                retain_graph=False,
                create_graph=False,
            )[0]
        finally:
            self.model.float()

        return gradient.detach().cpu().numpy(), evaluation.detach().cpu().numpy()

    def supports_gradients(self) -> bool:
        return self.use_gradients

    def set_use_gradients(self, enabled: bool) -> None:
        self.use_gradients = enabled


@register_updater
class NeuralNetworkUpdaterMinibatches(Updater):
    """
    Torch MLP surrogate trained with persistent optimizer state over minibatches drawn
    from all accumulated snapshots, with a replay mechanism so old data is not
    forgotten as new snapshots arrive (see ``_get_replay_indices``/``_iter_minibatches``).
    Registered with ``surrogates.reuse.register_updater`` (``surrogate_type =
    "NeuralNetworkUpdaterMinibatches"``), so checkpoints round-trip through
    ``SurrogateReused``.

    Weighting: ``supports_sample_weights = True`` -- with ``weighting="multiplicity"``,
    zero-multiplicity snapshots (rejected proposals) are dropped on arrival and the
    remaining rows enter ``_weighted_loss`` weighted by their multiplicity. The default
    ``weighting="uniform"`` trains on **every** snapshot with weight 1, rejected proposals
    included (WS6 decision 3; before WS6 this updater always behaved like
    ``"multiplicity"``).

    Output normalization: ``output_normalization="likelihood"`` (the default) centres the
    training targets on the observed data and scales them by the per-observation noise
    standard deviation, both taken from the likelihood via
    ``Updater.set_output_normalization`` (called once by ``SamplingFramework``/
    ``run_local``). ``"manual"`` uses the explicit ``output_mean``/``output_scale``
    arguments, ``"identity"`` does not normalize at all (the pre-WS6 default).

    Full-batch L-BFGS preset (replaces the deleted ``NeuralNetworkUpdaterBasic``)::

        NeuralNetworkUpdaterMinibatches(..., solver="lbfgs", batch_size=None,
                                        replay_ratio=0.0, train_on_added_data=False)
    """

    supports_sample_weights = True

    def __init__(
        self,
        no_parameters,
        no_observations,
        hidden_layer_sizes=(100,),
        solver: Literal["adamw", "adam", "lbfgs"] = "adamw",
        activation: Literal["silu", "relu", "tanh", "gelu", "elu", "leaky_relu", "identity"] = "silu",
        learning_rate=1e-3,
        iterations_batch=100,
        loss_target=1e-5,
        device: Literal["cpu", "cuda"] = "cpu",
        verbose: bool = False,
        seed: int | None = None,
        weighting: WeightingPolicy = "uniform",
        output_normalization: OutputNormalization = "likelihood",
        output_mean: np.ndarray | None = None,
        output_scale: np.ndarray | None = None,
        batch_size: int | None = None,
        replay_ratio: float = 1.0,
        replay_max_old_samples: int | None = None,
        train_on_added_data: bool = False,
        shuffle_batches: bool = True,
        gradient_clip_norm: float | None = None,
        weight_decay: float = 1e-4,
    ) -> None:
        """
        Args:
            no_parameters: dimension of the parameter space.
            no_observations: dimension of the observation space.
            hidden_layer_sizes: tuple of hidden layer widths.
            solver: torch optimizer; ``"lbfgs"`` also forces
                ``_infer_batch_size`` to use the whole available subset as one batch
                (full-batch behaviour, see ``library_notes/12_evaluator_contract_spec.md``
                §3 for the resulting "Basic-equivalent" preset).
            activation: hidden-layer activation.
            learning_rate: optimizer learning rate.
            iterations_batch: optimizer **steps** (not epochs) run per ``train()`` call.
            loss_target: early-stopping threshold on the training loss.
            device: ``"cpu"`` or ``"cuda"``.
            verbose: print extra fit diagnostics.
            seed: seeds both the torch model init and this updater's own
                ``numpy.random.default_rng`` (minibatch/replay sampling).
            weighting: snapshot-weighting policy, see ``Updater`` (default ``"uniform"``).
            output_normalization: where the per-observation normalization statistics come
                from. ``"likelihood"`` (default): observed data / noise sd, supplied by
                ``set_output_normalization``. ``"manual"``: the ``output_mean``/
                ``output_scale`` arguments. ``"identity"``: no normalization (mean 0,
                scale 1); ``output_mean``/``output_scale`` must then be ``None``.
            output_mean, output_scale: explicit per-observation normalization statistics,
                applied before the loss and undone in ``denormalize_outputs``. Required
                shape ``(no_observations,)``. Only meaningful for
                ``output_normalization="manual"`` (and accepted for ``"likelihood"``, where
                they act as already-resolved statistics -- this is how a checkpoint is
                restored).
            batch_size: fixed minibatch size; ``None`` picks a size from the current
                subset size (``_infer_batch_size``: whole subset if `<100`, else 64 or
                256), ignored when ``solver="lbfgs"``.
            replay_ratio: fraction of *old* (previously seen) snapshots mixed into each
                training call alongside the newly added ones, relative to the number of
                new snapshots; ``0`` trains on new data only.
            replay_max_old_samples: caps how many old snapshots ``replay_ratio`` may add.
            train_on_added_data: if True, ``add_data`` itself triggers a training pass
                (in addition to the collector's periodic ``train()`` calls); can be
                overridden per call via ``add_data(..., train_on_added_data=...)``.
            shuffle_batches: shuffle sample order within each training call.
            gradient_clip_norm: if given, clip the gradient norm to this value before
                each optimizer step.
            weight_decay: AdamW/Adam weight decay.

        Raises:
            ValueError: for an unknown ``weighting``/``output_normalization``, if
                ``output_mean``/``output_scale`` are given with
                ``output_normalization="identity"``, if they are non-finite or
                ``output_scale`` contains zeros, if ``replay_ratio < 0``, or if
                ``batch_size <= 0`` when given.
        """
        super().__init__(no_parameters, no_observations, weighting=weighting)
        self.hidden_layer_sizes = tuple(hidden_layer_sizes)
        self.solver_name = solver.lower()
        self.activation_name = activation
        self.learning_rate_init = learning_rate
        self.iterations_batch = iterations_batch
        self.loss_target = loss_target
        self.device = device
        self.verbose = verbose
        self.seed = seed
        self.batch_size = batch_size
        self.replay_ratio = float(replay_ratio)
        self.replay_max_old_samples = replay_max_old_samples
        self.train_on_added_data_default = train_on_added_data
        self.shuffle_batches = shuffle_batches
        self.gradient_clip_norm = gradient_clip_norm
        self.weight_decay = weight_decay

        if output_normalization not in OUTPUT_NORMALIZATIONS:
            raise ValueError(f"output_normalization must be one of {OUTPUT_NORMALIZATIONS}, "
                             f"got {output_normalization!r}")
        self.output_normalization = output_normalization
        if output_normalization == "identity" and (output_mean is not None or output_scale is not None):
            raise ValueError("output_normalization='identity' does not accept output_mean/output_scale; "
                             "use output_normalization='manual' to supply explicit statistics")
        statistics_given = output_mean is not None or output_scale is not None
        # provenance of the statistics actually in effect (what a checkpoint/manifest reports):
        if output_normalization == "manual":
            self.output_normalization_provenance = "manual"
        elif output_normalization == "likelihood" and statistics_given:
            self.output_normalization_provenance = "likelihood"
        else:
            # "identity", or "likelihood" before set_output_normalization() has been called
            self.output_normalization_provenance = "identity"
        self._set_output_statistics(output_mean, output_scale)
        if self.replay_ratio < 0.0:
            raise ValueError("replay_ratio must be non-negative")
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive when provided")

        self.use_gradients = True
        self.pretrained_ready = False
        self.training_data_loaded = False
        self.loaded_snapshot_count = 0
        self._last_added_indices = np.empty((0,), dtype=np.int64)
        self._rng = np.random.default_rng(seed)

        self.model = PyTorchMLP(no_parameters, no_observations, self.hidden_layer_sizes, activation, seed)
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.criterion = nn.MSELoss(reduction="none")
        self.criterionMSE = nn.MSELoss()

        self.par = torch.empty((0, self.no_parameters), dtype=torch.float32, device=self.device)
        self.obs = torch.empty((0, self.no_observations), dtype=torch.float32, device=self.device)
        self.multiplicity = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.no_snapshots = 0
        self.last_loss = 1.0

    def _set_output_statistics(self, output_mean, output_scale) -> None:
        """Validates and stores ``output_mean``/``output_scale`` (``None`` = identity)."""
        if output_mean is None:
            self.output_mean = np.zeros((self.no_observations,), dtype=np.float32)
        else:
            self.output_mean = np.asarray(output_mean, dtype=np.float32).reshape(self.no_observations)
        if output_scale is None:
            self.output_scale = np.ones((self.no_observations,), dtype=np.float32)
        else:
            self.output_scale = np.asarray(output_scale, dtype=np.float32).reshape(self.no_observations)
        if np.any(~np.isfinite(self.output_mean)):
            raise ValueError("output_mean must contain only finite values")
        if np.any(~np.isfinite(self.output_scale)) or np.any(self.output_scale == 0.0):
            raise ValueError("output_scale must contain only finite non-zero values")

    def set_output_normalization(self, mean: npt.NDArray, scale: npt.NDArray) -> None:
        """
        Adopts likelihood-derived normalization statistics (no-op unless this updater was
        configured with ``output_normalization="likelihood"``).

        Any snapshots already stored are re-normalized with the new statistics, so
        ``self.obs`` (which holds *normalized* observations) stays consistent.
        """
        if self.output_normalization != "likelihood":
            return
        previous_mean, previous_scale = self.output_mean, self.output_scale
        self._set_output_statistics(mean, scale)
        self.output_normalization_provenance = "likelihood"
        if self.obs.numel() > 0 and not (np.array_equal(previous_mean, self.output_mean)
                                         and np.array_equal(previous_scale, self.output_scale)):
            stored = self.obs.detach().cpu().numpy().reshape(-1, self.no_observations)
            original = stored * previous_scale + previous_mean
            renormalized = self.normalize_outputs(original).reshape(-1, self.no_observations)
            self.obs = torch.tensor(renormalized, dtype=torch.float32, device=self.device)

    def normalize_outputs(self, outputs: npt.NDArray) -> npt.NDArray:
        outputs_array = np.asarray(outputs, dtype=np.float32)
        return (outputs_array - self.output_mean) / self.output_scale

    def denormalize_outputs(self, normalized_outputs: npt.NDArray) -> npt.NDArray:
        normalized_array = np.asarray(normalized_outputs, dtype=np.float32)
        return normalized_array * self.output_scale + self.output_mean

    def _build_optimizer(self):
        if self.solver_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate_init, weight_decay=self.weight_decay)
        if self.solver_name == "lbfgs":
            return optim.LBFGS(self.model.parameters(), lr=self.learning_rate_init)
        if self.solver_name == "adam":
            return optim.Adam(self.model.parameters(), lr=self.learning_rate_init)
        raise ValueError("Unsupported solver. Use one of: adam, adamw, lbfgs")

    def _checkpoint_hparams(self) -> dict:
        """
        Constructor keyword arguments describing this updater, stored in the checkpoint and
        fed back to ``__init__`` by ``surrogates.reuse.SurrogateReused``.

        ``output_normalization`` records the *provenance* of the statistics that are in
        effect (``"identity"``/``"likelihood"``/``"manual"``), not the configured option, so
        that restoring a checkpoint restores exactly the normalization it was trained with.
        """
        identity_normalization = self.output_normalization_provenance == "identity"
        return {
            "no_parameters": self.no_parameters,
            "no_observations": self.no_observations,
            "hidden_layer_sizes": tuple(self.hidden_layer_sizes),
            "solver": self.solver_name,
            "activation": self.activation_name,
            "learning_rate": self.learning_rate_init,
            "iterations_batch": self.iterations_batch,
            "loss_target": self.loss_target,
            "device": self.device,
            "seed": self.seed,
            "weighting": self.weighting,
            "output_normalization": self.output_normalization_provenance,
            "output_mean": None if identity_normalization else self.output_mean.tolist(),
            "output_scale": None if identity_normalization else self.output_scale.tolist(),
            "batch_size": self.batch_size,
            "replay_ratio": self.replay_ratio,
            "replay_max_old_samples": self.replay_max_old_samples,
            "train_on_added_data": self.train_on_added_data_default,
            "shuffle_batches": self.shuffle_batches,
            "gradient_clip_norm": self.gradient_clip_norm,
            "weight_decay": self.weight_decay,
        }

    def _validate_checkpoint_hparams(self, checkpoint_hparams: dict) -> None:
        current = self._checkpoint_hparams()
        mismatches = []
        for key in (
            "no_parameters",
            "no_observations",
            "hidden_layer_sizes",
            "solver",
            "activation",
            "batch_size",
        ):
            if checkpoint_hparams.get(key) != current[key]:
                mismatches.append(f"{key}: checkpoint={checkpoint_hparams.get(key)!r}, current={current[key]!r}")
        checkpoint_mean_raw = checkpoint_hparams.get("output_mean", current["output_mean"])
        checkpoint_scale_raw = checkpoint_hparams.get("output_scale", current["output_scale"])
        # None means "identity normalization" in the checkpoint (see _checkpoint_hparams)
        checkpoint_mean = (np.zeros((self.no_observations,), dtype=np.float32) if checkpoint_mean_raw is None
                           else np.asarray(checkpoint_mean_raw, dtype=np.float32))
        checkpoint_scale = (np.ones((self.no_observations,), dtype=np.float32) if checkpoint_scale_raw is None
                            else np.asarray(checkpoint_scale_raw, dtype=np.float32))
        if checkpoint_mean.shape != (self.no_observations,) or checkpoint_scale.shape != (self.no_observations,):
            mismatches.append("output normalization vectors have incompatible shapes")
        else:
            if not np.allclose(checkpoint_mean, self.output_mean):
                mismatches.append("output_mean differs from checkpoint")
            if not np.allclose(checkpoint_scale, self.output_scale):
                mismatches.append("output_scale differs from checkpoint")
        if mismatches:
            raise ValueError(f"Checkpoint is incompatible with updater configuration: {'; '.join(mismatches)}")

    def get_initial_snapshots(self) -> list[npt.NDArray] | None:
        if not self.training_data_loaded:
            return None
        parameters, observations, multiplicity = self.get_training_data_arrays()
        return [parameters, observations, multiplicity]

    def get_training_data_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns the stored ``(parameters, observations, multiplicity)`` (observations denormalized)."""
        parameters = self.par.detach().cpu().numpy().reshape(-1, self.no_parameters)
        observations_normalized = self.obs.detach().cpu().numpy().reshape(-1, self.no_observations)
        observations = self.denormalize_outputs(observations_normalized)
        if self.multiplicity.numel() == 0:
            multiplicity = np.empty((0, 1), dtype=np.float32)
        else:
            multiplicity = self.multiplicity.detach().cpu().numpy().reshape(-1, 1)
        return parameters, observations, multiplicity

    def load_training_arrays(
        self,
        parameters: npt.NDArray,
        observations: npt.NDArray,
        multiplicity: npt.NDArray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = np.asarray(parameters, dtype=np.float32).reshape(-1, self.no_parameters)
        observations_original = np.asarray(observations, dtype=np.float32).reshape(-1, self.no_observations)
        if parameters.shape[0] != observations_original.shape[0]:
            raise ValueError("Training parameters and observations must contain the same number of rows")
        if multiplicity is None:
            multiplicity_array = np.ones((parameters.shape[0], 1), dtype=np.float32)
        else:
            multiplicity_array = np.asarray(multiplicity, dtype=np.float32).reshape(-1, 1)
            if multiplicity_array.shape[0] != parameters.shape[0]:
                raise ValueError("Training multiplicity must contain the same number of rows as parameters")

        observations_normalized = self.normalize_outputs(observations_original).reshape(-1, self.no_observations)
        self.par = torch.tensor(parameters, dtype=torch.float32, device=self.device)
        self.obs = torch.tensor(observations_normalized, dtype=torch.float32, device=self.device)
        self.multiplicity = torch.tensor(multiplicity_array, dtype=torch.float32, device=self.device)
        self.no_snapshots = parameters.shape[0]
        self.loaded_snapshot_count = self.no_snapshots
        self.training_data_loaded = self.no_snapshots > 0
        self._last_added_indices = np.arange(self.no_snapshots, dtype=np.int64)
        return parameters, observations_original, multiplicity_array

    def _batch_weights(self, index_tensor: torch.Tensor) -> torch.Tensor | None:
        """
        Per-sample loss weights for the selected rows, or ``None`` for an unweighted mean.

        ``weighting="uniform"`` (default) returns ``None``: every snapshot contributes
        equally, rejected proposals included. ``weighting="multiplicity"`` returns the
        stored multiplicities (zero-multiplicity rows were already dropped in ``add_data``).
        """
        if self.weighting != "multiplicity":
            return None
        return self.multiplicity.index_select(0, index_tensor)

    def _weighted_loss(self, predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None) -> torch.Tensor:
        per_output = self.criterion(predictions, targets)
        per_sample = torch.mean(per_output, dim=1, keepdim=True)
        if weights is None:
            return torch.mean(per_sample)
        weights_safe = torch.clamp(weights, min=0.0)
        weight_sum = torch.sum(weights_safe)
        if torch.isfinite(weight_sum) and weight_sum.item() > 0.0:
            return torch.sum(per_sample * weights_safe) / weight_sum
        return torch.mean(per_sample)

    def _apply_gradient_clipping(self) -> None:
        if self.gradient_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.gradient_clip_norm)

    def _infer_batch_size(self, subset_size: int) -> int:
        if subset_size <= 0:
            return 1
        if self.solver_name == "lbfgs":
            return subset_size
        if self.batch_size is not None:
            return min(self.batch_size, subset_size)
        if subset_size < 100:
            return subset_size
        if subset_size < 1000:
            return min(64, subset_size)
        return min(256, subset_size)

    def _get_replay_indices(self) -> np.ndarray:
        if self.no_snapshots == 0:
            return np.empty((0,), dtype=np.int64)
        recent_indices = np.asarray(self._last_added_indices, dtype=np.int64)
        if recent_indices.size == 0:
            return np.arange(self.no_snapshots, dtype=np.int64)
        if self.replay_ratio <= 0.0:
            return recent_indices
        old_mask = np.ones(self.no_snapshots, dtype=bool)
        old_mask[recent_indices] = False
        old_indices = np.flatnonzero(old_mask)
        if old_indices.size == 0:
            return recent_indices
        replay_count = int(np.ceil(recent_indices.size * self.replay_ratio))
        if self.replay_max_old_samples is not None:
            replay_count = min(replay_count, self.replay_max_old_samples)
        replay_count = max(1, replay_count)
        replay_count = min(replay_count, old_indices.size)
        sampled_old = self._rng.choice(old_indices, size=replay_count, replace=False)
        combined = np.concatenate([recent_indices, np.asarray(sampled_old, dtype=np.int64)])
        if self.shuffle_batches and combined.size > 1:
            self._rng.shuffle(combined)
        return combined

    def _iter_minibatches(self, indices: np.ndarray):
        if indices.size == 0:
            return
        batch_size = self._infer_batch_size(indices.size)
        ordered = indices.copy()
        if self.shuffle_batches and ordered.size > 1:
            self._rng.shuffle(ordered)
        for start in range(0, ordered.size, batch_size):
            yield ordered[start:start + batch_size]

    def _train_minibatches(self, indices: np.ndarray, max_steps: int) -> float:
        if indices.size == 0:
            return self.last_loss
        if self.solver_name == "lbfgs":
            return self._train_full_batch(indices, max_steps)
        optimizer = cast(optim.Optimizer, self.optimizer)

        steps_done = 0
        last_loss_value = self.last_loss
        while steps_done < max_steps:
            for batch_indices in self._iter_minibatches(indices):
                batch_idx_tensor = torch.tensor(batch_indices, dtype=torch.long, device=self.device)
                batch_par = self.par.index_select(0, batch_idx_tensor)
                batch_obs = self.obs.index_select(0, batch_idx_tensor)
                batch_weights = self._batch_weights(batch_idx_tensor)

                self.optimizer.zero_grad()
                batch_pred = self.model(batch_par)
                batch_loss = self._weighted_loss(batch_pred, batch_obs, batch_weights)
                batch_loss.backward()
                self._apply_gradient_clipping()
                optimizer.step()
                last_loss_value = float(batch_loss.detach().cpu().item())
                steps_done += 1
                if last_loss_value < self.loss_target or steps_done >= max_steps:
                    break
            else:
                continue
            break
        return last_loss_value

    def _train_full_batch(self, indices: np.ndarray, max_steps: int) -> float:
        if indices.size == 0:
            return self.last_loss
        index_tensor = torch.tensor(indices, dtype=torch.long, device=self.device)
        par = self.par.index_select(0, index_tensor)
        obs = self.obs.index_select(0, index_tensor)
        weights = self._batch_weights(index_tensor)

        def closure():
            self.optimizer.zero_grad()
            outputs = self.model(par)
            loss = self._weighted_loss(outputs, obs, weights)
            loss.backward()
            self._apply_gradient_clipping()
            return loss

        last_loss_value = self.last_loss
        for _ in range(max_steps):
            self.optimizer.step(closure=closure)
            with torch.no_grad():
                outputs = self.model(par)
                loss = self._weighted_loss(outputs, obs, weights)
            last_loss_value = float(loss.detach().cpu().item())
            if last_loss_value < self.loss_target:
                break
        return last_loss_value

    def initial_training(self, constant_observations: npt.NDArray, n: int = 1000, loss_target=1e-4):
        """
        Pre-trains the network to a constant output on ``n`` synthetic ``randn`` parameters.

        The synthetic rows are **not** persisted (finding 3.4, fixed in WS6): the real
        snapshot arrays are restored before returning, so ``get_training_data_arrays()``,
        ``get_initial_snapshots()`` and the checkpoint never report fabricated data.
        """
        parameters = np.random.randn(n, self.no_parameters).astype(np.float32)
        observations = np.tile(self.normalize_outputs(constant_observations), (n, 1)).astype(np.float32)
        saved = (self.par, self.obs, self.multiplicity, self.no_snapshots,
                 self._last_added_indices, self.training_data_loaded)
        try:
            self.par = torch.tensor(parameters, dtype=torch.float32, device=self.device)
            self.obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
            self.multiplicity = torch.ones((n, 1), dtype=torch.float32, device=self.device)
            self.no_snapshots = n
            self._last_added_indices = np.arange(n, dtype=np.int64)
            self.last_loss = self._train_minibatches(np.arange(n, dtype=np.int64), self.iterations_batch)
        finally:
            (self.par, self.obs, self.multiplicity, self.no_snapshots,
             self._last_added_indices, self.training_data_loaded) = saved
        if self.verbose:
            print(f"Initial training, MSE loss: {self.last_loss:.4e}", flush=True)

    def add_data(
        self,
        parameters: npt.NDArray,
        observations: npt.NDArray,
        multiplicity: npt.NDArray | None = None,
        train_on_added_data: bool | None = None,
    ):
        """Stores new snapshots; ``weighting="multiplicity"`` drops the zero-multiplicity rows."""
        loc_par_np = np.asarray(parameters, dtype=np.float32).reshape(-1, self.no_parameters)
        loc_obs_np = self.normalize_outputs(observations).reshape(-1, self.no_observations)
        if multiplicity is None:
            loc_mul_np = np.ones((loc_par_np.shape[0], 1), dtype=np.float32)
        else:
            loc_mul_np = np.asarray(multiplicity, dtype=np.float32).reshape(-1, 1)

        mask = self._rows_to_use(loc_mul_np, loc_par_np.shape[0])
        loc_par_np, loc_obs_np, loc_mul_np = loc_par_np[mask], loc_obs_np[mask], loc_mul_np[mask]

        loc_par = torch.tensor(loc_par_np, dtype=torch.float32, device=self.device)
        loc_obs = torch.tensor(loc_obs_np, dtype=torch.float32, device=self.device)
        loc_mul = torch.tensor(loc_mul_np, dtype=torch.float32, device=self.device)

        if loc_par.shape[0] != 0:
            start_idx = int(self.par.shape[0])
            if self.par.shape[0] == 0:
                self.par = loc_par
                self.obs = loc_obs
                self.multiplicity = loc_mul
            else:
                self.par = torch.concatenate([self.par, loc_par], dim=0)
                self.obs = torch.concatenate([self.obs, loc_obs], dim=0)
                self.multiplicity = torch.concatenate([self.multiplicity, loc_mul], dim=0)
            self.no_snapshots = int(self.par.shape[0])
            self.training_data_loaded = self.no_snapshots > 0
            self._last_added_indices = np.arange(start_idx, self.no_snapshots, dtype=np.int64)

        if train_on_added_data is None:
            train_on_added_data = self.train_on_added_data_default
        if train_on_added_data and self._last_added_indices.size > 0:
            replay_indices = self._get_replay_indices()
            self.last_loss = self._train_minibatches(replay_indices, self.iterations_batch)
            if self.verbose:
                print(
                    f"Training on new data + replay, loss: {self.last_loss:.4e}, steps: {self.iterations_batch}, subset: {replay_indices.size}",
                    flush=True,
                )

    def get_loss_on_data(self, parameters: npt.NDArray, observations: npt.NDArray):
        loc_par = torch.tensor(parameters, dtype=torch.float32, device=self.device)
        loc_obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            outputs_normalized = self.model(loc_par)
            loc_mean = torch.tensor(self.output_mean, dtype=torch.float32, device=self.device).reshape(1, -1)
            loc_scale = torch.tensor(self.output_scale, dtype=torch.float32, device=self.device).reshape(1, -1)
            outputs = outputs_normalized * loc_scale + loc_mean
            loss = self.criterionMSE(outputs, loc_obs)
        if self.verbose:
            print(f"MSE loss on given data: {loss.item():.4e}", flush=True)
        return float(loss.item())

    def train(self):
        if self.no_snapshots == 0:
            return
        full_indices = np.arange(self.no_snapshots, dtype=np.int64)
        self.last_loss = self._train_minibatches(full_indices, self.iterations_batch)
        if self.verbose:
            print(f"Training loss: {self.last_loss:.4e}, steps: {self.iterations_batch}", flush=True)

    def get_evaluator(self):
        return PyTorchNNEvaluator(
            self.no_parameters,
            self.no_observations,
            self.model,
            self.output_mean,
            self.output_scale,
            use_gradients=self.use_gradients,
        )

    def supports_gradients(self) -> bool:
        return True

    def set_use_gradients(self, enabled: bool) -> None:
        self.use_gradients = enabled

    def save_checkpoint(self, path: str) -> None:
        checkpoint = {
            "format_version": 1,
            "surrogate_type": type(self).__name__,
            "updater_hparams": self._checkpoint_hparams(),
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "runtime_state": {
                "last_loss": float(self.last_loss),
                "use_gradients": bool(self.use_gradients),
                "pretrained_ready": bool(self.pretrained_ready),
                "training_data_loaded": bool(self.training_data_loaded),
                "loaded_snapshot_count": int(self.loaded_snapshot_count),
            },
            "training_summary": {
                "num_snapshots": int(self.no_snapshots),
            },
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)

    def load_checkpoint(self, path: str, map_location: str | None = None, load_optimizer: bool = True) -> None:
        if map_location is None:
            map_location = self.device
        # weights_only=True (S22): the checkpoint holds only tensors, scalars, strings and
        # plain containers, so nothing has to be unpickled as arbitrary Python objects.
        checkpoint = torch.load(path, map_location=map_location, weights_only=True)
        if checkpoint.get("surrogate_type") != type(self).__name__:
            raise ValueError(
                f"Checkpoint surrogate type {checkpoint.get('surrogate_type')!r} is incompatible with {type(self).__name__}"
            )
        checkpoint_hparams = checkpoint.get("updater_hparams", {})
        self._validate_checkpoint_hparams(checkpoint_hparams)
        # the checkpoint records the provenance of the statistics it was trained with
        # (see _checkpoint_hparams); _validate_checkpoint_hparams already made sure the
        # numbers agree with this updater's
        self.output_normalization_provenance = checkpoint_hparams.get(
            "output_normalization", self.output_normalization_provenance)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        if load_optimizer and "optimizer_state_dict" in checkpoint:
            self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        runtime_state = checkpoint.get("runtime_state", {})
        self.last_loss = float(runtime_state.get("last_loss", self.last_loss))
        self.use_gradients = bool(runtime_state.get("use_gradients", self.use_gradients))
        self.pretrained_ready = True
        self.loaded_snapshot_count = int(
            runtime_state.get("loaded_snapshot_count", checkpoint.get("training_summary", {}).get("num_snapshots", 0))
        )

    def save_training_data(self, path: str) -> None:
        parameters, observations, multiplicity = self.get_training_data_arrays()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            parameters=parameters,
            observations=observations,
            multiplicity=multiplicity,
            no_parameters=self.no_parameters,
            no_observations=self.no_observations,
            num_snapshots=parameters.shape[0],
        )

    def load_training_data(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path) as loaded:
            parameters = loaded["parameters"]
            observations = loaded["observations"]
            multiplicity = loaded["multiplicity"] if "multiplicity" in loaded else None
        return self.load_training_arrays(parameters, observations, multiplicity)

    def save_state(self, checkpoint_path: str, data_path: str) -> None:
        self.save_checkpoint(checkpoint_path)
        self.save_training_data(data_path)

    def load_state(self, checkpoint_path: str, data_path: str | None = None,
                   load_optimizer: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        self.load_checkpoint(checkpoint_path, load_optimizer=load_optimizer)
        if data_path is None:
            return None
        loaded_arrays = self.load_training_data(data_path)
        self.pretrained_ready = True
        return loaded_arrays

    def save_snapshots(self, path_par='torch_perceptron_par.csv', path_obs='torch_perceptron_obs.csv'):
        """Debugging helper: writes the training snapshots as CSV. Relative paths are resolved
        against ``self.output_dir`` if the updater has one, otherwise against the CWD."""
        output_dir = getattr(self, "output_dir", None)
        if output_dir is not None:
            path_par = os.path.join(output_dir, path_par)
            path_obs = os.path.join(output_dir, path_obs)
        parameters, observations, _ = self.get_training_data_arrays()
        np.savetxt(path_par, parameters, delimiter=',')
        np.savetxt(path_obs, observations, delimiter=',')
        print(f"Snapshots saved to {os.path.abspath(path_par)} and {os.path.abspath(path_obs)}", flush=True)

    def supports_state_persistence(self) -> bool:
        return True

    def supports_training_data_persistence(self) -> bool:
        return True
    