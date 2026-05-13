import os
from typing import Literal
from typing import cast

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from surrDAMH.surrogates.parent import Evaluator, Updater


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
        self.no_parameters = no_parameters
        self.no_observations = no_observations
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
        with torch.no_grad():
            datapoints_tensor = torch.tensor(datapoints, dtype=torch.float32, device="cpu").reshape(-1, self.no_parameters)
            outputs = self.model(datapoints_tensor)
            outputs = outputs.detach().cpu().numpy().reshape(-1, self.no_observations)
            outputs = outputs * self.output_scale.reshape(1, -1) + self.output_mean.reshape(1, -1)
            return outputs.flatten()

    def jacobian(self, datapoints: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        if not self.use_gradients:
            raise RuntimeError("Surrogate gradients are disabled for this PyTorch evaluator")

        datapoints_array = np.asarray(datapoints, dtype=np.float32)
        if datapoints_array.shape != (self.no_parameters,):
            raise ValueError(
                f"Expected datapoints with shape ({self.no_parameters},), got {datapoints_array.shape}"
            )

        datapoints_tensor = torch.tensor(datapoints_array, dtype=torch.float32, device="cpu", requires_grad=True)
        outputs_normalized = self.model(datapoints_tensor).reshape(self.no_observations)
        scale_tensor = torch.tensor(self.output_scale, dtype=torch.float32, device="cpu")
        mean_tensor = torch.tensor(self.output_mean, dtype=torch.float32, device="cpu")
        outputs = outputs_normalized * scale_tensor + mean_tensor

        jacobian_rows = []
        for j in range(self.no_observations):
            grad = torch.autograd.grad(outputs[j], datapoints_tensor, retain_graph=True, create_graph=False)[0]
            jacobian_rows.append(grad)
        jacobian = torch.stack(jacobian_rows, dim=0)
        return jacobian.detach().cpu().numpy(), outputs.detach().cpu().numpy()

    def supports_gradients(self) -> bool:
        return self.use_gradients

    def set_use_gradients(self, enabled: bool) -> None:
        self.use_gradients = enabled


class PyTorchNNOngoingUpdater2(Updater):
    def __init__(
        self,
        no_parameters,
        no_observations,
        hidden_layer_sizes=(100,),
        solver: Literal["adam", "adamw", "lbfgs"] = "adam",
        activation: str = "tanh",
        learning_rate=1e-3,
        iterations_batch=100,
        loss_target=1e-5,
        device: Literal["cpu", "cuda"] = "cpu",
        verbose: bool = False,
        seed: int | None = None,
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
        self.no_parameters = no_parameters
        self.no_observations = no_observations
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
        self.weights = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.no_snapshots = 0
        self.last_loss = 1.0

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
            "output_mean": self.output_mean.tolist(),
            "output_scale": self.output_scale.tolist(),
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
        checkpoint_mean = np.asarray(checkpoint_hparams.get("output_mean", current["output_mean"]), dtype=np.float32)
        checkpoint_scale = np.asarray(checkpoint_hparams.get("output_scale", current["output_scale"]), dtype=np.float32)
        if checkpoint_mean.shape != (self.no_observations,) or checkpoint_scale.shape != (self.no_observations,):
            mismatches.append("output normalization vectors have incompatible shapes")
        else:
            if not np.allclose(checkpoint_mean, self.output_mean):
                mismatches.append("output_mean differs from checkpoint")
            if not np.allclose(checkpoint_scale, self.output_scale):
                mismatches.append("output_scale differs from checkpoint")
        if mismatches:
            raise ValueError(f"Checkpoint is incompatible with updater configuration: {'; '.join(mismatches)}")

    def get_training_data_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = self.par.detach().cpu().numpy().reshape(-1, self.no_parameters)
        observations_normalized = self.obs.detach().cpu().numpy().reshape(-1, self.no_observations)
        observations = self.denormalize_outputs(observations_normalized)
        if self.weights.numel() == 0:
            weights = np.empty((0, 1), dtype=np.float32)
        else:
            weights = self.weights.detach().cpu().numpy().reshape(-1, 1)
        return parameters, observations, weights

    def load_training_arrays(
        self,
        parameters: npt.NDArray,
        observations: npt.NDArray,
        weights: npt.NDArray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = np.asarray(parameters, dtype=np.float32).reshape(-1, self.no_parameters)
        observations_original = np.asarray(observations, dtype=np.float32).reshape(-1, self.no_observations)
        if parameters.shape[0] != observations_original.shape[0]:
            raise ValueError("Training parameters and observations must contain the same number of rows")
        if weights is None:
            weights_array = np.ones((parameters.shape[0], 1), dtype=np.float32)
        else:
            weights_array = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
            if weights_array.shape[0] != parameters.shape[0]:
                raise ValueError("Training weights must contain the same number of rows as parameters")

        observations_normalized = self.normalize_outputs(observations_original).reshape(-1, self.no_observations)
        self.par = torch.tensor(parameters, dtype=torch.float32, device=self.device)
        self.obs = torch.tensor(observations_normalized, dtype=torch.float32, device=self.device)
        self.weights = torch.tensor(weights_array, dtype=torch.float32, device=self.device)
        self.no_snapshots = parameters.shape[0]
        self.loaded_snapshot_count = self.no_snapshots
        self.training_data_loaded = self.no_snapshots > 0
        self._last_added_indices = np.arange(self.no_snapshots, dtype=np.int64)
        return parameters, observations_original, weights_array

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
                batch_weights = self.weights.index_select(0, batch_idx_tensor)

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
        weights = self.weights.index_select(0, index_tensor)

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
        parameters = np.random.randn(n, self.no_parameters).astype(np.float32)
        observations = np.tile(self.normalize_outputs(constant_observations), (n, 1)).astype(np.float32)
        self.par = torch.tensor(parameters, dtype=torch.float32, device=self.device)
        self.obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        self.weights = torch.ones((n, 1), dtype=torch.float32, device=self.device)
        self.no_snapshots = n
        self._last_added_indices = np.arange(n, dtype=np.int64)
        self.last_loss = self._train_minibatches(np.arange(n, dtype=np.int64), self.iterations_batch)
        if self.verbose:
            print(f"Initial training, MSE loss: {self.last_loss:.4e}", flush=True)

    def add_data(
        self,
        parameters: npt.NDArray,
        observations: npt.NDArray,
        weights: npt.NDArray | None = None,
        train_on_added_data: bool | None = None,
    ):
        loc_par_np = np.asarray(parameters, dtype=np.float32).reshape(-1, self.no_parameters)
        loc_obs_np = self.normalize_outputs(observations).reshape(-1, self.no_observations)
        if weights is None:
            loc_weights_np = np.ones((loc_par_np.shape[0], 1), dtype=np.float32)
        else:
            loc_weights_np = np.asarray(weights, dtype=np.float32).reshape(-1, 1)
        loc_par = torch.tensor(loc_par_np, dtype=torch.float32, device=self.device)
        loc_obs = torch.tensor(loc_obs_np, dtype=torch.float32, device=self.device)
        loc_weights = torch.tensor(loc_weights_np, dtype=torch.float32, device=self.device)

        if loc_par.shape[0] != 0:
            start_idx = int(self.par.shape[0])
            if self.par.shape[0] == 0:
                self.par = loc_par
                self.obs = loc_obs
                self.weights = loc_weights
            else:
                self.par = torch.concatenate([self.par, loc_par], dim=0)
                self.obs = torch.concatenate([self.obs, loc_obs], dim=0)
                self.weights = torch.concatenate([self.weights, loc_weights], dim=0)
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
        checkpoint = torch.load(path, map_location=map_location)
        if checkpoint.get("surrogate_type") != type(self).__name__:
            raise ValueError(
                f"Checkpoint surrogate type {checkpoint.get('surrogate_type')!r} is incompatible with {type(self).__name__}"
            )
        self._validate_checkpoint_hparams(checkpoint.get("updater_hparams", {}))
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
        parameters, observations, weights = self.get_training_data_arrays()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        np.savez(
            path,
            parameters=parameters,
            observations=observations,
            weights=weights,
            no_parameters=self.no_parameters,
            no_observations=self.no_observations,
            num_snapshots=parameters.shape[0],
        )

    def load_training_data(self, path: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        with np.load(path) as loaded:
            parameters = loaded["parameters"]
            observations = loaded["observations"]
            weights = loaded["weights"] if "weights" in loaded else None
        return self.load_training_arrays(parameters, observations, weights)

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
        parameters, observations, _ = self.get_training_data_arrays()
        np.savetxt(path_par, parameters, delimiter=',')
        np.savetxt(path_obs, observations, delimiter=',')
