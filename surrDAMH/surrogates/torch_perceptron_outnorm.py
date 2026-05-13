import os
from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from surrDAMH.surrogates.parent import Evaluator, Updater


class PyTorchMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation, seed):
        super(PyTorchMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        activation_layer = self.create_activation_layer(activation)
        if seed is not None:
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)  # if using multi-GPU setups
            print("SEED", seed)
        # if seed is not None:
        #     if self.device == "cuda":
        #         torch.cuda.manual_seed(seed)
        #         # torch.cuda.manual_seed_all(seed)  # if using multi-GPU setups
        #     else:
        #         torch.manual_seed(seed)
        #         print("SEED", seed)
        layers = [nn.Linear(input_size, hidden_layers[0]), activation_layer]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            activation_layer = self.create_activation_layer(activation)
            layers.append(activation_layer)
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def create_activation_layer(self, activation):
        if activation == 'relu':
            activation_layer = nn.ReLU()
        else:
            activation_layer = nn.Tanh()
        return activation_layer


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
        # Create a new instance of the same class as the original model
        model_clone = model.__class__(model.input_size, model.output_size, model.hidden_layers, model.activation, seed=None)
        # Load the state dict from the original model
        model_clone.load_state_dict(model.state_dict())
        # Move the cloned model to CPU
        model_clone = model_clone.to('cpu').eval()
        return model_clone

    def __call__(self, datapoints: npt.NDArray):
        with torch.no_grad():
            datapoints_tensor = torch.tensor(datapoints, dtype=torch.float32,
                                             device="cpu").reshape(-1, self.no_parameters)
            outputs = self.model(datapoints_tensor)
            outputs = outputs.detach().numpy().reshape(-1, self.no_observations)
            outputs = outputs * self.output_scale.reshape(1, -1) + self.output_mean.reshape(1, -1)
            outputs = outputs.flatten()
            return outputs

    def jacobian(self, datapoints: npt.NDArray) -> tuple[npt.NDArray, npt.NDArray]:
        if not self.use_gradients:
            raise RuntimeError("Surrogate gradients are disabled for this PyTorch evaluator")

        datapoints_array = np.asarray(datapoints, dtype=np.float32)
        if datapoints_array.shape != (self.no_parameters,):
            raise ValueError(
                f"Expected datapoints with shape ({self.no_parameters},), got {datapoints_array.shape}"
            )

        datapoints_tensor = torch.tensor(
            datapoints_array,
            dtype=torch.float32,
            device="cpu",
            requires_grad=True,
        )
        outputs_normalized = self.model(datapoints_tensor).reshape(self.no_observations)
        scale_tensor = torch.tensor(self.output_scale, dtype=torch.float32, device="cpu")
        mean_tensor = torch.tensor(self.output_mean, dtype=torch.float32, device="cpu")
        outputs = outputs_normalized * scale_tensor + mean_tensor

        jacobian_rows = []
        for j in range(self.no_observations):
            grad = torch.autograd.grad(
                outputs[j],
                datapoints_tensor,
                retain_graph=True,
                create_graph=False,
            )[0]
            jacobian_rows.append(grad)
        jacobian = torch.stack(jacobian_rows, dim=0)

        return jacobian.detach().cpu().numpy(), outputs.detach().cpu().numpy()

    def supports_gradients(self) -> bool:
        return self.use_gradients

    def set_use_gradients(self, enabled: bool) -> None:
        self.use_gradients = enabled


class PyTorchNNOngoingUpdater(Updater):
    def __init__(self, no_parameters, no_observations, hidden_layer_sizes=(100,), solver: Literal["adam", "adamw", "lbfgs"] = "adam",
                 activation='tanh', learning_rate=1e-3, iterations_batch=100, loss_target=1e-5,
                 device: Literal["cpu", "cuda"] = "cpu", verbose: bool = False, seed: int | None = None,
                 output_mean: np.ndarray | None = None, output_scale: np.ndarray | None = None) -> None:
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.solver_name = solver
        self.activation_name = activation
        self.learning_rate_init = learning_rate
        self.iterations_batch = iterations_batch
        self.loss_target = loss_target
        self.device = device
        self.verbose = verbose
        self.seed = seed

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

        self.use_gradients = True
        self.pretrained_ready = False
        self.training_data_loaded = False
        self.loaded_snapshot_count = 0

        self.model = PyTorchMLP(no_parameters, no_observations, hidden_layer_sizes, activation, seed)
        self.model.to(self.device)
        self.optimizer = self._build_optimizer()
        self.criterion = nn.MSELoss()  # nn.MSELoss() or nn.L1Loss()
        self.criterionMSE = nn.MSELoss()

        self.par = torch.empty((0, self.no_parameters), dtype=torch.float32, device=self.device)
        self.obs = torch.empty((0, self.no_observations), dtype=torch.float32, device=self.device)
        self.weights = torch.empty((0, 1), dtype=torch.float32, device=self.device)
        self.no_snapshots = 0
        self.last_loss = 1

    def normalize_outputs(self, outputs: npt.NDArray) -> npt.NDArray:
        outputs_array = np.asarray(outputs, dtype=np.float32)
        return (outputs_array - self.output_mean) / self.output_scale
    
    def denormalize_outputs(self, normalized_outputs: npt.NDArray) -> npt.NDArray:
        normalized_array = np.asarray(normalized_outputs, dtype=np.float32)
        return normalized_array * self.output_scale + self.output_mean

    def _build_optimizer(self):
        if self.solver_name == "adamw":
            return optim.AdamW(self.model.parameters(), lr=self.learning_rate_init, weight_decay=1e-4)
        elif self.solver_name == "lbfgs":
            return optim.LBFGS(self.model.parameters(), lr=self.learning_rate_init)
        else:
            return optim.Adam(self.model.parameters(), lr=self.learning_rate_init)
    

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
        }

    def _validate_checkpoint_hparams(self, checkpoint_hparams: dict) -> None:
        current = self._checkpoint_hparams()
        mismatches = []
        for key in ("no_parameters", "no_observations", "hidden_layer_sizes", "solver", "activation"):
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
            mismatch_message = "; ".join(mismatches)
            raise ValueError(f"Checkpoint is incompatible with updater configuration: {mismatch_message}")

    def get_training_data_arrays(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        parameters = self.par.detach().cpu().numpy().reshape(-1, self.no_parameters)
        observations_normalized = self.obs.detach().cpu().numpy().reshape(-1, self.no_observations)
        observations = self.denormalize_outputs(observations_normalized)
        if self.weights.numel() == 0:
            weights = np.empty((0, 1), dtype=np.float32)
        else:
            weights = self.weights.detach().cpu().numpy().reshape(-1, 1)
        return parameters, observations, weights

    def load_training_arrays(self, parameters: npt.NDArray, observations: npt.NDArray,
                             weights: npt.NDArray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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
        return parameters, observations_original, weights_array

    def initial_training(self, constant_observations: npt.NDArray, n: int = 1000, loss_target=1e-4):
        parameters = np.random.randn(n, self.no_parameters)
        observations = np.tile(self.normalize_outputs(constant_observations), (n, 1))
        par = torch.tensor(parameters, dtype=torch.float32, device=self.device)
        obs = torch.tensor(observations, dtype=torch.float32, device=self.device)

        def closure1():
            self.optimizer.zero_grad()
            outputs = self.model(par)
            loss = self.criterion(outputs, obs)
            loss.backward()
            return loss

        if par.shape[0] > 0:
            for iter in range(self.iterations_batch):
                self.optimizer.step(closure=closure1)
                with torch.no_grad():
                    outputs = self.model(par)
                    loss1 = self.criterionMSE(outputs, obs)
                if loss1.item() < loss_target / 2:
                    break
            if self.verbose:
                print(f"Initial training, MSE loss: {loss1.item():.4e}, I: {iter + 1}", flush=True)

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray, weights: npt.NDArray | None = None,
                 train_on_added_data: bool = True):
        loc_par = torch.tensor(parameters, dtype=torch.float32, device=self.device)
        normalized_observations = self.normalize_outputs(observations).reshape(-1, self.no_observations)
        loc_obs = torch.tensor(normalized_observations, dtype=torch.float32, device=self.device)
        if weights is None:
            loc_weights = torch.ones((loc_par.shape[0], 1), dtype=torch.float32, device=self.device)
        else:
            loc_weights = torch.tensor(weights, dtype=torch.float32, device=self.device).reshape(-1, 1)
        if parameters.shape[0] != 0:
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

        if train_on_added_data:
            def closure1():
                self.optimizer.zero_grad()
                outputs = self.model(loc_par)
                loss = self.criterion(outputs, loc_obs)
                loss.backward()
                return loss

            if loc_par.shape[0] > 0:
                for iter1 in range(self.iterations_batch):
                    self.optimizer.step(closure=closure1)
                    with torch.no_grad():
                        outputs = self.model(loc_par)
                        loss1 = self.criterionMSE(outputs, loc_obs)
                    if loss1.item() < self.loss_target / 2:
                        break
                if self.verbose:
                    print(f"Training on added data, MSE loss: {loss1.item():.4e}, I: {iter1 + 1}", flush=True)

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
            print(F"MSE loss on given data: {loss.item():.4e}", flush=True)
        return loss.item()

    def train(self):
        def closure():
            self.optimizer.zero_grad()
            outputs = self.model(self.par)
            loss = self.criterion(outputs, self.obs)
            loss.backward()
            return loss

        for iter in range(self.iterations_batch):
            self.optimizer.step(closure=closure)
            with torch.no_grad():
                outputs = self.model(self.par)
                loss = self.criterionMSE(outputs, self.obs)
            if loss.item() < self.loss_target:
                break
        if self.verbose:
            print(f"MSE loss: {loss.item():.4e}, I: {iter + 1}", flush=True)
        self.last_loss = loss.item()

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
        self.loaded_snapshot_count = int(runtime_state.get("loaded_snapshot_count", checkpoint.get("training_summary", {}).get("num_snapshots", 0)))

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

    def save_snapshots(self, path_par='torch_perceptron_par.csv',
                       path_obs='torch_perceptron_obs.csv'):
        parameters, observations, _ = self.get_training_data_arrays()
        np.savetxt(path_par, parameters, delimiter=',')
        np.savetxt(path_obs, observations, delimiter=',')
