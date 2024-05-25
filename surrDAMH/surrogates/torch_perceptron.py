from typing import Literal

import numpy as np
import numpy.typing as npt
import torch
import torch.nn as nn
import torch.optim as optim

from surrDAMH.surrogates.parent import Evaluator, Updater


class PyTorchMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers, activation):
        super(PyTorchMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.activation = activation
        activation_layer = self.create_activation_layer(activation)
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
    def __init__(self, no_parameters, no_observations, model):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.model = self.clone_model_to_cpu(model)

    def clone_model_to_cpu(self, model):
        # Create a new instance of the same class as the original model
        model_clone = model.__class__(model.input_size, model.output_size, model.hidden_layers, model.activation)
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
            outputs = outputs.flatten()
            return outputs


class PyTorchNNOngoingUpdater(Updater):
    def __init__(self, no_parameters, no_observations, hidden_layer_sizes=(100,), solver: Literal["adam", "lbfgs"] = "lbfgs",
                 activation='tanh', learning_rate=1e-3, iterations_batch=100, loss_target=1e-5,
                 device: Literal["cpu", "cuda"] = "cpu", verbose: bool = False) -> None:
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate
        self.iterations_batch = iterations_batch
        self.loss_target = loss_target
        self.device = device
        self.verbose = verbose

        self.model = PyTorchMLP(no_parameters, no_observations, hidden_layer_sizes, activation)
        self.model.to(self.device)
        if solver == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init)  # lr=1e-3
        else:
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=self.learning_rate_init)  # lr=0.5
        self.criterion = nn.MSELoss()  # nn.MSELoss() or nn.L1Loss()
        self.criterionMSE = nn.MSELoss()

        self.par = torch.empty((0, self.no_parameters), dtype=torch.float32, device=self.device)
        self.obs = torch.empty((0, self.no_observations), dtype=torch.float32, device=self.device)
        self.no_snapshots = 0
        self.last_loss = 1

    def initial_training(self, constant_observations: npt.NDArray, n: int = 1000, loss_target=1e-4):
        parameters = np.random.randn(n, self.no_parameters)
        observations = np.tile(constant_observations, (n, 1))
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
        loc_obs = torch.tensor(observations, dtype=torch.float32, device=self.device)
        if parameters.shape[0] != 0:
            if self.par.shape[0] == 0:
                self.par = loc_par
                self.obs = loc_obs
            else:
                self.par = torch.concatenate([self.par, loc_par], dim=0)
                self.obs = torch.concatenate([self.obs, loc_obs], dim=0)

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
            outputs = self.model(loc_par)
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
        return PyTorchNNEvaluator(self.no_parameters, self.no_observations, self.model)
