"""
TSX poroelastic solver — GRF-based permeability via truncated KL expansion.

Self-contained module (no subdomain imports needed).  Key features:
  1. Permeability field built from KL expansion: k(x) = exp(sum xi_i sqrt(lam_i) phi_i(x))
  2. Other material parameters (storativity, Young, Poisson, alpha) are homogeneous scalars.
  3. Variable time step support (unchanged from v2).
  4. Returns (cumulative_times, pressure_values) for interpolation to observation times.
"""

import numpy as np
import h5py
from math import sin, cos

import ufl
from ufl import grad, inner, div, dx
from basix.ufl import element, mixed_element

from mpi4py import MPI
from petsc4py import PETSc
import dolfinx as dfx
import dolfinx.fem.petsc
from dolfinx.fem import Constant

# ============================================================================
# Physical constants
# ============================================================================
TUNNEL_X_HALF_AXIS = 4.375 / 2  # 2.1875 m
TUNNEL_Y_HALF_AXIS = 3.5 / 2    # 1.75 m
SEC_IN_DAY = 24 * 60 * 60
AMBIENT_PORE_PRESSURE = 4.1e6   # Pa, Chandler (2002)


# ============================================================================
# Mesh I/O
# ============================================================================

def load_mesh_and_domain_tags(path_to_mesh):
    """Load mesh and tags from XDMF files (same convention as model1)."""
    with dfx.io.XDMFFile(MPI.COMM_SELF, f'{path_to_mesh}.xdmf', 'r') as f:
        mesh = f.read_mesh(name='Grid')
        cell_tags = f.read_meshtags(mesh, name='Grid')
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)
    with dfx.io.XDMFFile(MPI.COMM_SELF, f'{path_to_mesh}_boundary.xdmf', 'r') as f:
        facet_tags = f.read_meshtags(mesh, name='Grid')
    return mesh, cell_tags, facet_tags


# ============================================================================
# KL expansion utilities
# ============================================================================

def load_kl_data(kl_filepath):
    """Load precomputed KL eigenvalues and eigenvectors from HDF5.

    Returns
    -------
    eigenvalues : np.ndarray, shape (n_modes,)
    eigenvectors : np.ndarray, shape (n_vertices, n_modes)
    """
    with h5py.File(kl_filepath, 'r') as f:
        eigenvalues = f['eigenvalues'][:]
        eigenvectors = f['eigenvectors'][:]
    return eigenvalues, eigenvectors


def kl_expansion_at_vertices(xi, eigenvalues, eigenvectors):
    """Compute the KL field at mesh vertices (zero-mean log-space).

    Parameters
    ----------
    xi : array-like, shape (n_modes,)
        KL coefficients (the inferred parameters).
    eigenvalues : np.ndarray, shape (n_modes,)
    eigenvectors : np.ndarray, shape (n_vertices, n_modes)

    Returns
    -------
    field : np.ndarray, shape (n_vertices,)
        The random field Z(x) = sum_i xi_i * sqrt(lambda_i) * phi_i(x)
    """
    xi = np.asarray(xi)
    n_modes = len(xi)
    # Z(x) = sum_i  xi_i * sqrt(lambda_i) * phi_i(x)
    weights = xi * np.sqrt(eigenvalues[:n_modes])        # shape (n_modes,)
    field = eigenvectors[:, :n_modes] @ weights           # shape (n_vertices,)
    return field


def build_sigmoid_field_dg0(mesh, inner_val, outer_val, k=6.0, x0=0.7):
    """Build a DG-0 field with smooth sigmoid transition from inner to outer
    value, based on elliptical distance from the tunnel boundary.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    inner_val : float
        Value near the tunnel (distance ≈ 0).
    outer_val : float
        Value far from the tunnel.
    k : float
        Steepness of the logistic transition.
    x0 : float
        Midpoint distance (from ellipse surface) of the transition.

    Returns
    -------
    f_dg0 : dolfinx.fem.Function on DG-0 space
    """
    from scipy.spatial.distance import cdist

    F1 = np.array([-1.3125, 0.0])   # focal points of the tunnel ellipse
    F2 = np.array([ 1.3125, 0.0])

    coords = mesh.geometry.x[:, :2]
    dist_from_foci = np.sum(cdist(coords, [F1, F2]), axis=1)
    dist_from_ellipse = (dist_from_foci - 2 * TUNNEL_X_HALF_AXIS) * 0.5

    # Logistic transition: inner_val near tunnel, outer_val far away
    field_values = (inner_val
                    + (outer_val - inner_val)
                      / (1.0 + np.exp(-k * (dist_from_ellipse - x0))))

    # Build CG-1 function, then interpolate to DG-0
    Q_cg1 = dfx.fem.functionspace(mesh, ('CG', 1))
    f_cg1 = dfx.fem.Function(Q_cg1)
    f_cg1.x.array[:] = field_values

    Q_dg0 = dfx.fem.functionspace(mesh, ('DG', 0))
    f_dg0 = dfx.fem.Function(Q_dg0)
    f_dg0.interpolate(f_cg1)
    return f_dg0


def build_permeability_field(mesh, xi, eigenvalues, eigenvectors, transform=None):
    """Build a DG-0 permeability function from KL coefficients.

    The KL eigenvectors live at mesh vertices (CG-1), so we first
    build a CG-1 function and then interpolate to DG-0.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    xi : array-like, shape (n_modes,)
    eigenvalues, eigenvectors : from load_kl_data
    transform : callable or None
        Function applied element-wise to the interpolated GRF values
        to produce the final field.  Receives and returns a numpy array.
        Default (None) uses ``np.exp``.

    Returns
    -------
    k_dg0 : dolfinx.fem.Function on DG-0 space
    """
    grf_values = kl_expansion_at_vertices(xi, eigenvalues, eigenvectors)

    # Build CG-1 function with GRF values at vertices
    Q_cg1 = dfx.fem.functionspace(mesh, ('CG', 1))
    grf_cg1 = dfx.fem.Function(Q_cg1)
    grf_cg1.x.array[:] = grf_values

    # Interpolate to DG-0 and apply transform
    Q_dg0 = dfx.fem.functionspace(mesh, ('DG', 0))
    grf_dg0 = dfx.fem.Function(Q_dg0)
    grf_dg0.interpolate(grf_cg1)

    k_dg0 = dfx.fem.Function(Q_dg0)
    if transform is None:
        k_dg0.x.array[:] = np.exp(grf_dg0.x.array)
    else:
        k_dg0.x.array[:] = transform(grf_dg0.x.array)

    return k_dg0


# ============================================================================
# Homogeneous coefficient functions (no subdomains)
# ============================================================================

def prepare_homogeneous_coefficients(mesh, lmbda_val, mu_val, alpha_val, cpp_val, k_function):
    """Create DG-0 coefficient functions with uniform scalar values,
    except permeability which is provided as an already-built DG-0 function.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    lmbda_val, mu_val, alpha_val, cpp_val : float
        Homogeneous scalar material constants.
    k_function : dolfinx.fem.Function
        Permeability field (DG-0), e.g. from build_permeability_field.

    Returns
    -------
    lmbda, mu, alpha, cpp, k : dolfinx.fem.Function  (all DG-0)
    """
    Q = dfx.fem.functionspace(mesh, ('DG', 0))

    lmbda = dfx.fem.Function(Q)
    lmbda.x.array[:] = lmbda_val

    mu = dfx.fem.Function(Q)
    mu.x.array[:] = mu_val

    alpha = dfx.fem.Function(Q)
    alpha.x.array[:] = alpha_val

    cpp = dfx.fem.Function(Q)
    cpp.x.array[:] = cpp_val

    # k is already a DG-0 function; return it directly
    return lmbda, mu, alpha, cpp, k_function


# ============================================================================
# Evaluation points and boundary conditions
# ============================================================================

def prepare_evaluation_points(mesh, point_coordinates):
    """Find cells containing evaluation points (borehole sensors)."""
    bb_tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)
    cells_t = []
    points_on_proc_t = []
    cell_candidates_t = dfx.geometry.compute_collisions_points(bb_tree, point_coordinates.T)
    colliding_cells_t = dfx.geometry.compute_colliding_cells(mesh, cell_candidates_t, point_coordinates.T)
    for i, point in enumerate(point_coordinates.T):
        if len(colliding_cells_t.links(i)) > 0:
            points_on_proc_t.append(point)
            cells_t.append(colliding_cells_t.links(i)[0])
    return np.array(points_on_proc_t, dtype=np.float64), cells_t


def boundary_inner(x):
    return np.isclose(x[0]**2 / TUNNEL_X_HALF_AXIS**2 + x[1]**2 / TUNNEL_Y_HALF_AXIS**2, 1.0)


def generate_dirichlet_bc_tsx(mesh, V, pressure_expression, pressure_outer):
    """Generate Dirichlet BCs for the TSX problem."""
    def boundary_outer(x):
        return np.logical_or(
            np.logical_or(np.isclose(x[0], -50), np.isclose(x[0], 50)),
            np.logical_or(np.isclose(x[1], -50), np.isclose(x[1], 50)))

    def boundary_outer_lr(x):
        return np.logical_or(np.isclose(x[0], -50), np.isclose(x[0], 50))

    def boundary_outer_bt(x):
        return np.logical_or(np.isclose(x[1], -50), np.isclose(x[1], 50))

    boundary_conditions = {
        'elastic_lf': {
            'marker_function': boundary_outer_lr,
            'prescribed_expression': Constant(mesh, 0.0),
            'function_space': V.sub(0).sub(0)
        },
        'elastic_bt': {
            'marker_function': boundary_outer_bt,
            'prescribed_expression': Constant(mesh, 0.0),
            'function_space': V.sub(0).sub(1)
        },
        'pressure_outer': {
            'marker_function': boundary_outer,
            'prescribed_expression': Constant(mesh, pressure_outer),
            'function_space': V.sub(1)
        },
        'pressure_inner': {
            'marker_function': boundary_inner,
            'prescribed_expression': pressure_expression,
            'function_space': V.sub(1)
        }
    }

    bcs = []
    for bc in boundary_conditions:
        edges = dfx.mesh.locate_entities_boundary(mesh, 1, boundary_conditions[bc]['marker_function'])
        dofs_on_edges = dfx.fem.locate_dofs_topological(boundary_conditions[bc]['function_space'], 1, edges)
        bcs.append(dfx.fem.dirichletbc(boundary_conditions[bc]['prescribed_expression'], dofs_on_edges,
                                       boundary_conditions[bc]['function_space']))
    return bcs


def epsilon(u):
    return ufl.sym(ufl.nabla_grad(u))


# ============================================================================
# Time step schedule helpers
# ============================================================================

def uniform_time_steps(dt, n_steps):
    """Return an array of *n_steps* identical time step sizes *dt* (seconds)."""
    return np.full(n_steps, dt)


def piecewise_uniform_time_steps(phases):
    """Build a step array from a list of ``(dt, n_steps)`` pairs."""
    return np.concatenate([np.full(n, dt) for dt, n in phases])


def geometric_time_steps(dt_start, ratio, n_steps, dt_max=None):
    """Geometrically growing time steps: ``dt_i = dt_start * ratio**i``."""
    steps = dt_start * ratio ** np.arange(n_steps)
    if dt_max is not None:
        steps = np.minimum(steps, dt_max)
    return steps


# ============================================================================
# Main solver function (variable time step)
# ============================================================================

def tsx_setup_and_computation_v2(mesh,
                                 lmbda, mu, alpha, cpp, k,
                                 time_steps,
                                 sigma_xx=-45e6,
                                 sigma_yy=-12.8e6,
                                 sigma_angle=8 * np.pi / 180):
    """Solve the 2D Biot poroelasticity problem with variable time steps.

    Parameters
    ----------
    mesh : dolfinx.mesh.Mesh
    lmbda, mu, alpha, cpp, k : dolfinx.fem.Function (DG-0)
        Material coefficient functions (possibly heterogeneous).
    time_steps : array-like of float
        Sequence of time step sizes in seconds.
    sigma_xx, sigma_yy : float
        In-situ stress components in Pa (compression is negative).
    sigma_angle : float
        Rotation of the stress tensor, in radians.

    Returns
    -------
    cumulative_times : np.ndarray, shape (len(time_steps) + 1,)
    pressure_values : list of np.ndarray
    """
    time_steps = np.asarray(time_steps, dtype=float)
    n_steps = len(time_steps)
    cumulative_times = np.concatenate([[0.0], np.cumsum(time_steps)])

    tau = Constant(mesh, time_steps[0])

    # Function spaces: P2 displacement, P1 pressure (Taylor-Hood)
    P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    P1 = element("Lagrange", mesh.basix_cell(), 1)
    V_element = mixed_element([P2, P1])
    V = dfx.fem.functionspace(mesh, V_element)
    u, p = ufl.TrialFunctions(V)
    w, q = ufl.TestFunctions(V)
    x_h = dfx.fem.Function(V)

    # Evaluation points (4 borehole sensors)
    evaluation_points = np.zeros((3, 4))
    evaluation_points[0, :] = [0, 0,
                               TUNNEL_X_HALF_AXIS + 4.0,
                               TUNNEL_X_HALF_AXIS + 1.5]
    evaluation_points[1, :] = [TUNNEL_Y_HALF_AXIS + 1.5,
                               TUNNEL_Y_HALF_AXIS + 4.0,
                               0, 0]
    ready_eval_points, eval_cells = prepare_evaluation_points(mesh, evaluation_points)

    # Boundary conditions
    pressure_init = AMBIENT_PORE_PRESSURE
    pbc_expression = Constant(
        mesh, pressure_init * max(0.0, 1 - time_steps[0] / (17 * SEC_IN_DAY)))
    bcs = generate_dirichlet_bc_tsx(mesh, V, pbc_expression, pressure_outer=pressure_init)

    # In-situ stress
    rotation = np.array([[cos(sigma_angle), -sin(sigma_angle)],
                         [sin(sigma_angle),  cos(sigma_angle)]])
    sigma_init = Constant(
        mesh, rotation.T @ np.array([[sigma_xx, 0], [0, sigma_yy]]) @ rotation)
    sigma_expression = Constant(mesh, 0.0)

    # Bilinear form (LHS)
    ff_term = cpp / tau * p * q * dx + k * inner(grad(p), grad(q)) * dx
    a = dfx.fem.form(
        2 * mu * inner(epsilon(u), epsilon(w)) * dx
        + lmbda * div(u) * div(w) * dx
        - alpha * p * div(w) * dx
        + alpha / tau * q * div(u) * dx
        + ff_term
    )

    f = Constant(mesh, (0.0, 0.0))
    g = Constant(mesh, 0.0)

    # Initial conditions
    u_h, p_h = x_h.split()
    u_h.x.array[:] = 0
    p_h.x.array[:] = pressure_init

    # Assemble LHS and set up solver
    A = dfx.fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A, A)
    solver.setType('preonly')
    solver.getPC().setType('lu')
    opts = PETSc.Options()
    opts['pc_factor_mat_solver_type'] = 'mumps'
    solver.setFromOptions()

    # Time stepping
    current_time = 0.0
    current_dt = time_steps[0]

    pressure_values = [p_h.eval(ready_eval_points, eval_cells)]

    for step_idx in range(n_steps):
        dt = time_steps[step_idx]
        current_time += dt

        if dt != current_dt:
            tau.value = dt
            A.zeroEntries()
            dfx.fem.petsc.assemble_matrix(A, a, bcs=bcs)
            A.assemble()
            solver.setOperators(A, A)
            current_dt = dt

        sigma_expression.value = min(1.0, current_time / (17 * SEC_IN_DAY))
        pbc_expression.value = pressure_init * max(0.0, 1 - current_time / (17 * SEC_IN_DAY))

        L = dfx.fem.form(
            inner(f, w) * dx + g * q * dx
            + alpha / tau * div(u_h) * q * dx
            + cpp / tau * p_h * q * dx
            - sigma_expression * inner(sigma_init, epsilon(w)) * dx
        )
        b = dfx.fem.petsc.assemble_vector(L)
        dfx.fem.petsc.apply_lifting(b, [a], [bcs])
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dfx.fem.set_bc(b, bcs)

        solver.solve(b, x_h.x.petsc_vec)
        x_h.x.scatter_forward()
        u_h, p_h = x_h.split()

        pressure_values.append(p_h.eval(ready_eval_points, eval_cells))

    return cumulative_times, pressure_values
