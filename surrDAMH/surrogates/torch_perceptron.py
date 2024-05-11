import numpy.typing as npt
from typing import Literal
import torch
import torch.nn as nn
import torch.optim as optim
from surrDAMH.surrogates.parent import Updater, Evaluator

DEVICE = "cpu"


class PyTorchMLP(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers):
        super(PyTorchMLP, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        layers = [nn.Linear(input_size, hidden_layers[0]), nn.Tanh()]
        for i in range(1, len(hidden_layers)):
            layers.append(nn.Linear(hidden_layers[i - 1], hidden_layers[i]))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_layers[-1], output_size))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)


class PyTorchNNEvaluator(Evaluator):
    def __init__(self, no_parameters, no_observations, model):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.model = self.clone_model_to_cpu(model)

    def clone_model_to_cpu(self, model):
        # Create a new instance of the same class as the original model
        model_clone = model.__class__(model.input_size, model.output_size, model.hidden_layers)
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
            return outputs.detach().numpy().reshape(-1, self.no_observations) * 275.0


class PyTorchNNOngoingUpdater(Updater):
    def __init__(self, no_parameters, no_observations, hidden_layer_sizes=(100,), solver: Literal["adam", "lbfgs"] = "lbfgs",
                 activation='relu', learning_rate_init=1e-3, iterations_batch=100, loss_target=1e-4):
        self.no_parameters = no_parameters
        self.no_observations = no_observations
        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate_init = learning_rate_init
        self.iterations_batch = iterations_batch
        self.model = PyTorchMLP(no_parameters, no_observations, hidden_layer_sizes)
        self.model.to(DEVICE)
        if solver == "adam":
            self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate_init)
        else:
            self.optimizer = optim.LBFGS(self.model.parameters(), lr=0.5)
        self.criterion = nn.L1Loss()
        self.criterion2 = nn.MSELoss()

        self.par = torch.empty((0, self.no_parameters), dtype=torch.float32, device=DEVICE)
        self.obs = torch.empty((0, self.no_observations), dtype=torch.float32, device=DEVICE)
        self.no_snapshots = 0
        self.loss_target = loss_target
        self.last_loss = 1

    def add_data(self, parameters: npt.NDArray, observations: npt.NDArray, weights: npt.NDArray | None = None):
        loc_par = torch.tensor(parameters, dtype=torch.float32, device=DEVICE)
        loc_obs = torch.tensor(observations / 275.0, dtype=torch.float32, device=DEVICE)
        if parameters.shape[0] != 0:
            if self.par.shape[0] == 0:
                self.par = loc_par
                self.obs = loc_obs
            else:
                self.par = torch.concatenate([self.par, loc_par], dim=0)
                self.obs = torch.concatenate([self.obs, loc_obs], dim=0)

        def closure1():
            self.optimizer.zero_grad()
            outputs = self.model(loc_par)
            loss = self.criterion(outputs, loc_obs)
            loss.backward()
            return loss

        if loc_par.shape[0] > 0:
            for iter1 in range(self.iterations_batch):
                # self.optimizer.zero_grad()
                # outputs = self.model(loc_par)
                # loss1 = self.criterion(outputs, loc_obs)
                # loss1.backward()
                self.optimizer.step(closure=closure1)
                with torch.no_grad():
                    outputs = self.model(loc_par)
                    loss1 = self.criterion2(outputs, loc_obs)
                if loss1.item() < self.loss_target / 2:
                    break
            print(f"Training on added data, L: {loss1.item():.4e}, I: {iter1 + 1}", flush=True)
        # else:
        #     if self.last_loss < loss_target:
        #         return

    def train(self):
        def closure():
            self.optimizer.zero_grad()
            outputs = self.model(self.par)
            loss = self.criterion(outputs, self.obs)
            loss.backward()
            return loss

        for iter in range(self.iterations_batch):
            # self.optimizer.zero_grad()
            # outputs = self.model(self.par)
            # loss = self.criterion(outputs, self.obs)
            # loss.backward()
            self.optimizer.step(closure=closure)
            with torch.no_grad():
                outputs = self.model(self.par)
                loss = self.criterion2(outputs, self.obs)
            if loss.item() < self.loss_target:
                break
        # if loc_par.shape[0] > 0:
        #     print(f"L: {loss1.item():.4e}/{loss.item():.4e}, I: {iter1 + 1}/{iter + 1}")
        # else:
        print(f"L: {loss.item():.4e}, I: {iter + 1}", flush=True)
        self.last_loss = loss.item()

    def get_evaluator(self):
        return PyTorchNNEvaluator(self.no_parameters, self.no_observations, self.model)
