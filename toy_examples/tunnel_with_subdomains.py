import os
from math import sin, cos

import numpy as np
import pyvista

import ufl
from ufl import grad, inner, div, dx
from basix.ufl import element, mixed_element

from dolfinx.io import VTKFile
from mpi4py import MPI
from petsc4py import PETSc

import dolfinx as dfx
import dolfinx.fem.petsc  # needed to carry out the init :(
from dolfinx.fem import Constant
from dolfinx import plot

# from ellipses_regions_generator import generate_tsx_mesh_with_regions, Ellipse


# hardcoded values for boundary conditions
TUNNEL_X_HALF_AXIS = 4.375 / 2
TUNNEL_Y_HALF_AXIS = 3.5 / 2
SEC_IN_DAY = 24*60*60


# def provide_tsx_mesh(path_to_mesh):
#     """
#     Check if there is a file on path_to_mesh. If there is not create a tsx mesh.
#     """
#     # TODO: better input and checking if dirs exists, creating them if needed
#     suffixes = ['.xdmf', '.h5', '_boundary.xdmf', '_boundary.h5']
#     if all([os.path.isfile(f'{path_to_mesh}{suffix}') for suffix in suffixes]):
#         return None

#     # domain is given by [-50, 50] x [-50, 50] with hole with
#     x_axis = 4.375 / 2
#     y_axis = 3.5 / 2
#     # specify internal subdomains
#     number_of_rays = 5
#     ellipses_list = [Ellipse(x_axis * k, y_axis * k) for k in (1, 1.3, 1.7, 2)]

#     generate_tsx_mesh_with_regions(path_to_mesh, number_of_rays, ellipses_list)
#     return None


def load_mesh_and_domain_tags(path_to_mesh):
    """
    Taken from:
    https://github.com/jorgensd/dolfinx-tutorial/blob/v0.8.0/chapter3/subdomains.ipynb
    """

    with dfx.io.XDMFFile(MPI.COMM_SELF, f'{path_to_mesh}.xdmf', 'r') as mesh_file:
        mesh = mesh_file.read_mesh(name='Grid')
        cell_tags = mesh_file.read_meshtags(mesh, name='Grid')

    # taken from the link, no idea what it does
    mesh.topology.create_connectivity(mesh.topology.dim, mesh.topology.dim - 1)

    with dfx.io.XDMFFile(MPI.COMM_SELF, f'{path_to_mesh}_boundary.xdmf', 'r') as boundary_file:
        facet_tags = boundary_file.read_meshtags(mesh, name='Grid')

    return mesh, cell_tags, facet_tags


def prepare_coefficient_functions(mesh, cells_tags,
                                  lmbda_list, mu_list, alpha_list, cpp_list, k_list):
    # prepare space and functions
    Q = dfx.fem.functionspace(mesh, ('DG', 0))
    lmbda = dfx.fem.Function(Q)
    mu = dfx.fem.Function(Q)
    alpha = dfx.fem.Function(Q)
    cpp = dfx.fem.Function(Q)
    k = dfx.fem.Function(Q)

    list_of_lists = [lmbda_list, mu_list, alpha_list, cpp_list, k_list]
    list_of_functions = [lmbda, mu, alpha, cpp, k]

    # effectively min(set(cell_tags.values)), most of the time, depends on gmsh code
    marker_start = 1

    # create dict to store map of domains to actual cells
    tag_to_cells = {marker: cells_tags.find(marker) for marker in set(cells_tags.values)}

    for parameter_function, list_of_values in zip(list_of_functions, list_of_lists):
        for marker, value in enumerate(list_of_values, start=marker_start):
            cell_numbers = tag_to_cells[marker]
            parameter_function.x.array[cell_numbers] = value

    return lmbda, mu, alpha, cpp, k


def prepare_evaluation_points(mesh, point_coordinates):
    bb_tree = dfx.geometry.bb_tree(mesh, mesh.topology.dim)
    cells_t = []
    points_on_proc_t = []
    cell_candidates_t = dfx.geometry.compute_collisions_points(bb_tree, point_coordinates.T)
    colliding_cells_t = dfx.geometry.compute_colliding_cells(mesh, cell_candidates_t, point_coordinates.T)
    for i, point in enumerate(point_coordinates.T):
        if len(colliding_cells_t.links(i)) > 0:
            points_on_proc_t.append(point)
            cells_t.append(colliding_cells_t.links(i)[0])

    ready_eval_points = np.array(points_on_proc_t, dtype=np.float64)
    return ready_eval_points, cells_t


def boundary_inner(x):
    return np.isclose(x[0]**2/TUNNEL_X_HALF_AXIS**2 + x[1]**2/TUNNEL_Y_HALF_AXIS**2, 1.0)


# TODO: use facet tags instead of functions
def generate_dirichlet_bc_tsx(mesh, V, pressure_expression, pressure_outer):
    def boundary_outer(x):
        return np.logical_or(np.logical_or(np.isclose(x[0], -50), np.isclose(x[0], 50)),
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


def tsx_setup_and_computation(mesh,
                              lmbda, mu, alpha, cpp, k,
                              tau_f, t_steps_num,
                              sigma_xx=-45e6, sigma_yy=-11e6, sigma_angle=0):
    # time step parameter
    tau = Constant(mesh, tau_f)

    # Spaces and functions
    P2 = element("Lagrange", mesh.basix_cell(), 2, shape=(mesh.geometry.dim,))
    P1 = element("Lagrange", mesh.basix_cell(), 1)
    V_element = mixed_element([P2, P1])
    V = dfx.fem.functionspace(mesh, V_element)
    u, p = ufl.TrialFunctions(V)
    w, q = ufl.TestFunctions(V)
    x_h = dfx.fem.Function(V)  # function for the fem solution

    # prepare inputs for pressure evaluation at HGT-something points
    evaluation_points = np.zeros((3, 4))  # must be 3D, for some reason
    evaluation_points[0, :] = [0, 0, 0 + TUNNEL_X_HALF_AXIS + 4.0, 0 + TUNNEL_X_HALF_AXIS + 1.5]
    evaluation_points[1, :] = [0 + TUNNEL_Y_HALF_AXIS + 1.5, 0 + TUNNEL_Y_HALF_AXIS + 4.0, 0, 0]

    ready_eval_points, eval_cells = prepare_evaluation_points(mesh, evaluation_points)

    # bc -> zero normal displacements, 3e6 outer pressure and 3e6 to zero pressure in tunnel
    # Dirichlet
    pressure_init = 3e6  # water pressure in the massive, initial and outer condition for pressure
    pbc_expression = Constant(mesh, pressure_init * max(0.0, 1 - tau_f / (17 * SEC_IN_DAY)))
    # TODO: use generated domains
    bcs = generate_dirichlet_bc_tsx(mesh, V, pbc_expression, pressure_outer=pressure_init)

    rotation = np.array([[cos(sigma_angle), -sin(sigma_angle)],
                         [sin(sigma_angle), cos(sigma_angle)]])
    sigma_init = rotation.T @ np.array([[sigma_xx, 0], [0, sigma_yy]]) @ rotation
    sigma_init = Constant(mesh, sigma_init)
    sigma_expression = Constant(mesh, min(1.0, tau_f / (17 * SEC_IN_DAY)))

    # variational form construction
    ff_term = cpp / tau * p * q * dx  # flux-flux term
    ff_term += k * inner(grad(p), grad(q)) * dx

    # antisymmetric but coercive formulation
    a = dfx.fem.form(2*mu*inner(epsilon(u), epsilon(w))*dx + lmbda*div(u)*div(w)*dx - alpha*p*div(w)*dx +
                     alpha/tau*q*div(u)*dx + ff_term)

    # volume forces
    f = Constant(mesh, (0.0, 0.0))  # elastic volume force
    g = Constant(mesh, 0.0)  # pressure volume force
    # initial conditions
    u_h, p_h = x_h.split()
    u_h.x.array[:] = 0
    p_h.x.array[:] = pressure_init

    # assembly and set bcs
    A = dfx.fem.petsc.assemble_matrix(a, bcs=bcs)
    A.assemble()

    # PETSc4py section with solver setup
    solver = PETSc.KSP().create(mesh.comm)
    solver.setOperators(A, A)
    solver.setType('preonly')
    solver.getPC().setType('lu')
    opts = PETSc.Options()
    opts['pc_factor_mat_solver_type'] = 'mumps'
    solver.setFromOptions()

    # time stepping
    current_time = 0.0

    pressure_values = [p_h.eval(ready_eval_points, eval_cells)]
    for _ in range(1, t_steps_num):
        current_time += tau_f
        sigma_expression.value = min(1.0, current_time/(17*SEC_IN_DAY))
        pbc_expression.value = pressure_init*max(0.0, 1 - current_time/(17*SEC_IN_DAY))
        L = dfx.fem.form(inner(f, w)*dx + g*q*dx +
                         alpha/tau*div(u_h)*q*dx + cpp/tau*p_h*q*dx -
                         sigma_expression*inner(sigma_init, epsilon(w))*dx)

        b = dfx.fem.petsc.assemble_vector(L)
        dfx.fem.petsc.apply_lifting(b, [a], [bcs])  # ???
        b.ghostUpdate(addv=PETSc.InsertMode.ADD_VALUES, mode=PETSc.ScatterMode.REVERSE)
        dfx.fem.set_bc(b, bcs)
        solver.solve(b, x_h.x.petsc_vec)  # .x.petsc_vec instead of .vector
        x_h.x.scatter_forward()
        u_h, p_h = x_h.split()
        pressure_values.append(p_h.eval(ready_eval_points, eval_cells))

    return pressure_values


def plot_pressures(data):
    data_fp = np.zeros((4, len(data)))
    for i, _ in enumerate(data):
        data_fp[:, i] = [value[0] for value in data[i]]

    names = ['HGT 1-5', 'HGT 1-4', 'HGT 2-5', 'HGT 2-4']
    colors = ['red', 'green', 'blue', 'violet']
    for i, timeline in enumerate(data_fp):
        plt.plot(timeline, label=names[i], color=colors[i])

    # NOTE: hardcoded for 400 days
    plt.xticks(range(0, 801, 100), range(0, 401, 50))
    plt.legend()
    plt.ylabel('Pressure [Pa]')
    plt.xlabel('Days')
    plt.title('Pressure in control points')
    plt.show()


def pyvista_plot(dolfinx_function):
    mesh = dolfinx_function.function_space.mesh
    topology, cell_types, geometry = plot.vtk_mesh(mesh, 2)
    grid = pyvista.UnstructuredGrid(topology, cell_types, geometry)
    grid.cell_data["my_function"] = dolfinx_function.x.array.real
    plotter = pyvista.Plotter()
    plotter.title = 'Coefficients'
    plotter.add_mesh(grid, show_edges=True, scalar_bar_args={'vertical': True})
    plotter.view_xy()
    plotter.show()


def write_func_to_vtk(func, filnename):
    mesh = func.function_space.mesh
    with VTKFile(mesh.comm, filnename, 'w') as f:
        f.write_mesh(mesh)
        f.write_function(func)


"""
if __name__ == '__main__':
    import matplotlib.pyplot as plt

    SEC_IN_DAY = 24 * 60 * 60
    plot_the_data = False
    visualize_parameters = True
    vtk_output = False

    mesh_name = 'tsx_ellipses_regions'
    mesh_dir = '.'
    os.makedirs(mesh_dir, exist_ok=True)
    mesh_prefix = f'{mesh_dir}/{mesh_name}'  # TODO: nicer logic and names
    provide_tsx_mesh(mesh_prefix)

    young_e = 6e10
    poisson_nu = 0.2
    mu = young_e / (2 * (1 + poisson_nu))
    lmbda = young_e * poisson_nu / ((1 + poisson_nu) * (1 - 2 * poisson_nu))
    alpha = 0.2
    cpp = 7.712e-12
    timestep = SEC_IN_DAY / 2

    tsx_mesh, cell_tags, _ = load_mesh_and_domain_tags(mesh_prefix)

    # NOTE: following assumes that domains are label consecutively from 1, gmsh does this for now
    number_of_subdomains = max(set(cell_tags.values))

    # example data to test functionality
    lmbda_values = [lmbda + _ for _ in range(number_of_subdomains)]
    mu_values = [mu + _ for _ in range(number_of_subdomains)]
    alpha_values = [alpha - 0.00001*_ for _ in range(number_of_subdomains)]
    cpp_values = [cpp + _*1.0e-14 for _ in range(number_of_subdomains)]
    k_values = [6.0e-19 + _*1.0e-20 for _ in range(number_of_subdomains)]

    # converts lists to dolfinx functions on respective subdomains
    lmbda_func, mu_func, alpha_func, cpp_func, k_func = prepare_coefficient_functions(tsx_mesh, cell_tags,
                                                                                      lmbda_values, mu_values, alpha_values, cpp_values, k_values)

    # the actual computation
    pressure_data = tsx_setup_and_computation(tsx_mesh,
                                              lmbda_func, mu_func, alpha_func, cpp_func, k_func,
                                              tau_f=SEC_IN_DAY/2, t_steps_num=100)

    print(pressure_data)

    if plot_the_data:
        plot_pressures(pressure_data)

    if visualize_parameters:
        pyvista_plot(k_func)

    if vtk_output:
        func_name = 'k'
        write_func_to_vtk(k_func, f'{mesh_dir}/{func_name}.vtk')
"""
