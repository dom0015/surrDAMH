# FORWARD MODEL: 
# What parameters should be chosen:
# 1. Grid resolution (nx, ny): Higher resolution provides more detail but increases computational cost.
# 2. Covariance function type and parameters (length_scale, sigma, nu): These control the smoothness and variability of the random field.
# 3. Positive transformation factor: This affects the range of the diffusion coefficient and can influence the solution behavior.
# Solve -div(k grad(u)) = 1 with Dirichlet data on the vertical boundaries.

from typing import Literal

from surrDAMH.solvers import Solver
import numpy as np
import numpy.typing as npt
import time
import matplotlib.pyplot as plt

from mpi4py import MPI
# use fenics to solve a 2d stationary diffusion equation
import dolfinx as dfx
# use regular grid for the mesh
from dolfinx.mesh import create_rectangle
from dolfinx.fem import Function
from dolfinx.fem import Constant, dirichletbc, locate_dofs_geometrical
from dolfinx.fem.petsc import LinearProblem
from ufl import TestFunction, TrialFunction, dx, grad, inner
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, gamma


class Solver_diffusion_GRF(Solver):
    def __init__(self, xa=-1.0, xb=1.0, ya=-1.0, yb=1.0, nx=20, ny=20, solver_id=0, output_dir=None, sleep_time=0.0,
                 covariance_type: Literal['Matern', 'squared_exponential', 'exponential'] ='Matern', 
                 length_scale=0.7, sigma=0.1, nu=1.0, no_parameters=10, observations_per_dim=3,
                 positivity_transform_factor=1.0, u_left=1.0, u_right=0.0, source_strength=1.0):

        # grid parameters:
        self.xa = xa
        self.xb = xb
        self.ya = ya
        self.yb = yb
        self.nx = nx
        self.ny = ny
        self.no_parameters = no_parameters
        self.observations_per_dim = observations_per_dim
        self.positivity_transform_factor = positivity_transform_factor
        self.observations_per_dim = observations_per_dim
        self.no_observations = self.observations_per_dim ** 2
        self.sleep_time = sleep_time

        # preparepoints of measurements:
        x_values = np.linspace(xa, xb, self.observations_per_dim + 2)
        x_values = x_values[1:-1]  # exclude the boundary points
        y_values = np.linspace(ya, yb, self.observations_per_dim + 2)
        y_values = y_values[1:-1]  # exclude the boundary points
        x_coords, y_coords = np.meshgrid(x_values, y_values)
        measurement_points = np.column_stack((x_coords.ravel(), y_coords.ravel()))
        # add third coordinate zero:
        self.measurement_points = np.column_stack((measurement_points, np.zeros(measurement_points.shape[0])))

        # prepare the mesh
        self.mesh = create_rectangle(MPI.COMM_SELF, [np.array([self.xa, self.ya]), np.array([self.xb, self.yb])], [self.nx-1, self.ny-1], cell_type=dfx.mesh.CellType.quadrilateral)
        self.coords = self.mesh.geometry.x
        # prepare the function space
        self.V = dfx.fem.functionspace(self.mesh, ("CG", 1))
        # prepare the diffusion coefficient as a Function in the function space
        self.diffusion_coefficient = Function(self.V)

        # prepare the grf covariance matrix and KL expansion
        if covariance_type == 'Matern':
            self.cov_func = Matern
            self.cov_params = {'length_scale': length_scale, 'sigma': sigma, 'nu': nu}
        elif covariance_type == 'squared_exponential':
            self.cov_func = squared_exponential
            self.cov_params = {'length_scale': length_scale, 'sigma': sigma}
        elif covariance_type == 'exponential':
            self.cov_func = exponential
            self.cov_params = {'length_scale': length_scale, 'sigma': sigma}
        else:
            raise ValueError(f"Unknown covariance type: {covariance_type}")
        self.covariance_matrix = build_covariance_matrix(self.coords, self.cov_func, **self.cov_params)
        self.eigenvalues, self.eigenvectors = KL_expansion(self.covariance_matrix, n_terms=self.no_parameters)

        # prepare dolfinx problem

        def on_left_boundary(x):
            return np.isclose(x[0], xa)

        def on_right_boundary(x):
            return np.isclose(x[0], xb)

        left_dofs = locate_dofs_geometrical(self.V, on_left_boundary)
        right_dofs = locate_dofs_geometrical(self.V, on_right_boundary)

        left_bc = dirichletbc(np.array(u_left, dtype=np.float64), left_dofs, self.V)
        right_bc = dirichletbc(np.array(u_right, dtype=np.float64), right_dofs, self.V)
        boundary_conditions = [left_bc, right_bc]

        u = TrialFunction(self.V)
        v = TestFunction(self.V)
        source_term = Constant(self.mesh, dfx.default_scalar_type(source_strength))

        bilinear_form = inner(self.diffusion_coefficient * grad(u), grad(v)) * dx
        linear_form = source_term * v * dx

        self.problem = LinearProblem(
            bilinear_form,
            linear_form,
            bcs=boundary_conditions,
            petsc_options_prefix="diffusion_problem",
            petsc_options={
                "ksp_type": "preonly",
                "pc_type": "lu",
            },
        )

    def set_parameters(self, parameters: npt.NDArray):
        self.parameters = parameters

    def get_observations(self):
        latent_truncated_field = self.eigenvectors @ (np.sqrt(self.eigenvalues) * self.parameters)
        transformed_field = positivity_transform(latent_truncated_field, factor=self.positivity_transform_factor)
        # assign the values of the diffusion coefficient field to the Function
        self.diffusion_coefficient.x.array[:] = transformed_field.ravel()
        self.solution = self.problem.solve()

        # solution in measurement points:
        cells = np.zeros(self.measurement_points.shape[0], dtype=np.int32)  # dummy cell indices
        solution_at_measurement_points = self.solution.eval(self.measurement_points, cells)

        time.sleep(self.sleep_time)
        return np.array([solution_at_measurement_points], dtype=np.float64).ravel()
    
    def field_builder(self, parameters: npt.NDArray):
        latent_truncated_field = self.eigenvectors @ (np.sqrt(self.eigenvalues) * parameters)
        transformed_field = positivity_transform(latent_truncated_field, factor=self.positivity_transform_factor)
        return transformed_field
    
    def generate_artificial_observations(self, seed=11):
        # fix seed:
        np.random.seed(seed)
        z = np.random.normal(size=self.no_parameters)
        self.set_parameters(z)
        return self.get_observations()
    
    def plot_measurement_points(self, ax=None):
        if ax is None:
            ax = plt.gca()
        ax.scatter(self.measurement_points[:, 0], self.measurement_points[:, 1], c='red', marker='o', label='Measurement Points')

    def visualize_solution(self, show=False):
        solution_grid_x, solution_grid_y, solution_grid = reshape_function_values(self.solution, self.V)
        coefficient_grid_x, coefficient_grid_y, coefficient_grid = reshape_function_values(self.diffusion_coefficient, self.V)
        outputs = []

        # Visualize the coefficient field and the corresponding diffusion solution.
        fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)

        coefficient_image = axes[0].imshow(
            coefficient_grid,
            extent=(self.xa, self.xb, self.ya, self.yb),
            origin="lower",
            cmap="viridis",
            aspect="auto",
        )
        axes[0].set_title("Diffusion Coefficient")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(coefficient_image, ax=axes[0], label="k(x)")
        # add measurement points to the coefficient plot using "plot_measurement_points" method
        self.plot_measurement_points(axes[0])

        solution_image = axes[1].imshow(
            solution_grid,
            extent=(self.xa, self.xb, self.ya, self.yb),
            origin="lower",
            cmap="plasma",
            aspect="auto",
        )
        axes[1].set_title("Solution of the Diffusion Equation")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(solution_image, ax=axes[1], label="u(x)")
        # add measurement points to the coefficient plot using "plot_measurement_points" method
        self.plot_measurement_points(axes[1])

        outputs.append((fig, axes))
        if show:
            plt.show()

        # Visualize the diffusive flux q = -k grad(u) on the regular plotting grid.
        dx_grid = solution_grid_x[1] - solution_grid_x[0]
        dy_grid = solution_grid_y[1] - solution_grid_y[0]

        solution_grad_y, solution_grad_x = np.gradient(
            solution_grid,
            dy_grid,
            dx_grid,
            edge_order=2,
            )
        flux_x = -coefficient_grid * solution_grad_x
        flux_y = -coefficient_grid * solution_grad_y
        flux_magnitude = np.hypot(flux_x, flux_y)

        fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)

        flux_x_image = axes[0].imshow(
            flux_x,
            extent=(self.xa, self.xb, self.ya, self.yb),
            origin="lower",
            cmap="coolwarm",
            aspect="auto",
        )
        axes[0].set_title("Flux x-component")
        axes[0].set_xlabel("x")
        axes[0].set_ylabel("y")
        fig.colorbar(flux_x_image, ax=axes[0], label=r"$q_x$")

        flux_y_image = axes[1].imshow(
            flux_y,
            extent=(self.xa, self.xb, self.ya, self.yb),
            origin="lower",
            cmap="coolwarm",
            aspect="auto",
        )
        axes[1].set_title("Flux y-component")
        axes[1].set_xlabel("x")
        axes[1].set_ylabel("y")
        fig.colorbar(flux_y_image, ax=axes[1], label=r"$q_y$")

        magnitude_image = axes[2].imshow(
            flux_magnitude,
            extent=(self.xa, self.xb, self.ya, self.yb),
            origin="lower",
            cmap="magma",
            aspect="auto",
        )
        stride = max(1, self.nx // 12)
        axes[2].quiver(
            solution_grid_x[::stride],
            solution_grid_y[::stride],
            flux_x[::stride, ::stride],
            flux_y[::stride, ::stride],
            color="white",
            pivot="mid",
            scale=None,
            angles="xy",
        )
        axes[2].set_title("Flux magnitude and direction")
        axes[2].set_xlabel("x")
        axes[2].set_ylabel("y")
        fig.colorbar(magnitude_image, ax=axes[2], label=r"$|q|$")

        print(f"flux_x range: [{flux_x.min():.4f}, {flux_x.max():.4f}]")
        print(f"flux_y range: [{flux_y.min():.4f}, {flux_y.max():.4f}]")
        print(f"flux magnitude range: [{flux_magnitude.min():.4f}, {flux_magnitude.max():.4f}]")

        outputs.append((fig, axes))
        if show:
            plt.show()

        return outputs


# covariance functions:
# squared exponential:
def squared_exponential(r, length_scale=0.7, sigma=0.1):
    return sigma**2 * np.exp(-0.5 * (r / length_scale)**2)

# Matern covariance function:
def Matern(r, length_scale=0.7, sigma=0.1, nu=1.0):
    if r == 0.0:
        return sigma**2
    else:
        factor = (2**(1.0 - nu)) / gamma(nu)
        return sigma**2 * factor * (r / length_scale)**nu * kv(nu, r / length_scale)

def exponential(r, length_scale=0.7, sigma=0.1):
    return sigma**2 * np.exp(-r / length_scale)

def Matern32(r, length_scale=0.7, sigma=0.1):
    return Matern(r, length_scale=length_scale, sigma=sigma, nu=1.5)

# Linear algebra helpers for Gaussian random field sampling.
def build_covariance_matrix(points, cov_func, **kwargs):
    pairwise_differences = points[:, None, :] - points[None, :, :]
    pairwise_distances = np.linalg.norm(pairwise_differences, axis=-1)
    covariance_values = np.vectorize(lambda r: cov_func(float(r), **kwargs), otypes=[float])(pairwise_distances)
    return covariance_values

def cholesky_factorization(cov_matrix):
    return np.linalg.cholesky(cov_matrix)

def KL_expansion(cov_matrix, n_terms):
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
    ordering = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[ordering], 0.0)
    eigenvectors = eigenvectors[:, ordering]
    return eigenvalues[:n_terms], eigenvectors[:, :n_terms]

def positivity_transform(latent_field_sample, factor=1.0):
    return factor*np.exp(latent_field_sample)

def reshape_function_values(function, V, decimals=12):
    dof_coordinates = V.tabulate_dof_coordinates()[:, :2]
    rounded_x = np.round(dof_coordinates[:, 0], decimals=decimals)
    rounded_y = np.round(dof_coordinates[:, 1], decimals=decimals)
    x_coords = np.unique(rounded_x)
    y_coords = np.unique(rounded_y)
    x_indices = np.searchsorted(x_coords, rounded_x)
    y_indices = np.searchsorted(y_coords, rounded_y)
    values_on_grid = np.empty((len(y_coords), len(x_coords)))
    values_on_grid[y_indices, x_indices] = function.x.array
    return x_coords, y_coords, values_on_grid