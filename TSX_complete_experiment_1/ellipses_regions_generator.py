import gmsh
import meshio
from numpy import sin, cos, pi, sqrt
from itertools import product
import os
from pathlib import Path


class Ellipse:
    """
    Ellipse as x**2/a**2 + y**2/b**2 = 1
    """

    def __init__(self, a, b):
        self.a = a
        self.b = b
        if a > b:
            self.major_point_coo = (a, 0, 0)
        else:
            self.major_point_coo = (0, b, 0)

    def __lt__(self, other):
        """
        Presumes that the ellipses have no intersections and same major axes!
        """
        return self.a < other.a


class Ray:
    """
    Ray from origin given by angle from half-axis (x = t, y = 0) t > 0.
    """

    def __init__(self, phi):
        self.phi = phi

    def __eq__(self, other):
        return self.phi == other.phi

    def __lt__(self, other):
        return self.phi < other.phi

    def __str__(self):
        return f'Ray from origin, angle: {self.phi}'

    def __repr__(self):
        return repr(self.phi)


def ray_and_ellipse_intersection(ray, ellipse):
    """
    Compute the single intersection of a ray and an ellipse.
    :param ray:
    :param ellipse:
    :return: intersection_point as tuple
    """
    # easy peasy on computation on paper
    t = ellipse.a*ellipse.b/sqrt(ellipse.b**2*cos(ray.phi)**2 + ellipse.a**2*sin(ray.phi)**2)
    p_x = t*cos(ray.phi)
    p_y = t*sin(ray.phi)
    p_z = 0.0
    return p_x, p_y, p_z


def compute_region_data(ellipses_list, rays_list):
    """
    Compute points that defines the borders between regions.
    This pressumes aditional metainformation about the setup, suitable for out numerical experiments.
    :param ellipses_list: list of ellipses, each ellipse given by halfaxes, center is in 0,0, x and y axis
    :param rays_list: list of rays, given by angle taken from x > 0 axis
    :return: list of tuples of 4-points that define the regions and tuple with points on major axis
    """
    ellipses_list.sort()
    rays_list.sort()

    borders = []
    for closer_ellipse, farther_ellipse in zip(ellipses_list, ellipses_list[1:]):
        major_axis_smaller = closer_ellipse.major_point_coo
        major_axis_bigger = farther_ellipse.major_point_coo
        for i in range(len(rays_list)):
            ray1, ray2 = rays_list[i], rays_list[(i+1) % len(rays_list)]
            border_points = []
            for ray, ellipse in product((ray1, ray2), (closer_ellipse, farther_ellipse)):
                border_points.append(ray_and_ellipse_intersection(ray, ellipse))
            borders.append((border_points, (major_axis_smaller, major_axis_bigger)))
    # border points are ordered in the way what first two and second two are on the same ray
    # they are norm-wise ordered in the pairs, first smaller than bigger, in the same way as points on major axes
    return borders


def create_mesh(rays_list, ellipses_list, corner_points, resolution_outer=10, resolution_inner=0.5,
                resolution_inner_boundary=None, filename=None, return_model=False):
    """
    Creates square mesh with subdomains determined by rays and ellipses
    Due to ellipse construction at least 3 rays are needed.
    Note, that gmsh allows to set resolutions for individual points, might be worth to either:
    a) use more granular approach and assign resolution to each ellipse
    b) or use some different interface to set mesh size

    resolution_inner_boundary: if provided, used for points on the innermost ellipse;
                               otherwise falls back to resolution_inner.

    Function has an option to either return the model for future tinkering or just finalize gmsh.

    Known issue: Ellipse arcs will fail when the start and end points can be written as
    (x, y), (x, -y) or (x, y), (-x, y).
    """

    if len(rays_list) < 3:
        raise ValueError('Due to the currently used ellipse construction at least 3 rays are needed.')

    if len(ellipses_list) < 2:
        raise ValueError('Internal subdomains need at least 2 ellipses.')

    # sort ellipses to get the smallest, NOTE: would be sufficient only to find the smallest
    ellipses_list.sort()

    # geometry init
    gmsh.initialize()
    model = gmsh.model.occ  # NOTE: "model" might be a confusing variable name :(

    # center point needed for ellipses
    center_point = model.add_point(*(0, 0, 0))

    # resolve resolution for inner boundary
    res_inner_bd = resolution_inner_boundary if resolution_inner_boundary is not None else resolution_inner

    # adds distinct regions, gradually filling the hole
    outer_arcs = []  # hack-ish solution to hanging nodes
    inner_arcs = []  # hack-ish solution for the inner boundary tags TODO: rewrite the logic completely
    borders = compute_region_data(ellipses_list, rays_list)
    n_rays = len(rays_list)
    counter = 0
    for border, axis_coo in borders:
        # For the first n_rays borders the closer ellipse is the innermost one.
        # border indices 0,2 lie on the closer (inner) ellipse; 1,3 on the farther one.
        if counter < n_rays:
            resolutions = [res_inner_bd, resolution_inner, res_inner_bd, resolution_inner]
            axis_resolutions = [res_inner_bd, resolution_inner]
        else:
            resolutions = [resolution_inner] * 4
            axis_resolutions = [resolution_inner] * 2

        # points to define the lines and ellipses
        aux_points = [model.add_point(*p, meshSize=res) for p, res in zip(border, resolutions)]
        # points on major axis needed by the engine to construct ellipse arcs
        axis_coo_p = [model.add_point(*p, meshSize=res) for p, res in zip(axis_coo, axis_resolutions)]
        # arcs and lines defining the border
        arc1 = model.add_ellipse_arc(startTag=aux_points[2], endTag=aux_points[0], centerTag=center_point,
                                     majorTag=axis_coo_p[0])
        l0 = model.add_line(aux_points[0], aux_points[1])
        arc2 = model.add_ellipse_arc(startTag=aux_points[1], endTag=aux_points[3], centerTag=center_point,
                                     majorTag=axis_coo_p[1])
        l1 = model.add_line(aux_points[3], aux_points[2])
        cc = model.add_curve_loop([arc1, l0, arc2, l1])
        region = model.add_plane_surface((cc,))
        outer_arcs.append(arc2)
        inner_arcs.append(arc1)

        # synchronize is needed before adding groups, or using a different interface
        # https://gitlab.onelab.info/gmsh/gmsh/-/issues/2574
        model.synchronize()
        region_tag = gmsh.model.add_physical_group(dim=2, tags=[region])
        gmsh.model.set_physical_name(dim=2, tag=region_tag, name=f'Region {counter}')

        counter += 1

    outer_most_arcs = outer_arcs[-len(rays_list):]  # outer ellipse is defined by the last big arcs
    inner_most_arcs = inner_arcs[:len(rays_list)]  # hack-ish way to find inner ellipse arcs

    # construction of the outer box and region between outer border and outermost ellipse
    border_points = [model.add_point(*point, meshSize=resolution_outer) for point in corner_points]
    lines_outer = [model.add_line(border_points[i], border_points[i+1]) for i in range(-1, len(corner_points)-1)]
    outer_loop = model.add_curve_loop(lines_outer)
    inner_loop = model.add_curve_loop(outer_most_arcs)

    plane_surface = model.add_plane_surface((outer_loop, inner_loop))

    model.synchronize()
    outer_domain = gmsh.model.add_physical_group(dim=2, tags=[plane_surface])
    gmsh.model.set_physical_name(dim=2, tag=outer_domain, name='Outer')

    # physical groups for boundaries, supplied tags are tags of the lines and arcs that define the boundary
    model.synchronize()
    inner_loop_physical = gmsh.model.add_physical_group(dim=1, tags=inner_most_arcs)
    gmsh.model.set_physical_name(dim=1, tag=inner_loop_physical, name='Inner boundary')
    model.synchronize()
    outer_loop_physical = gmsh.model.add_physical_group(dim=1, tags=lines_outer)
    gmsh.model.set_physical_name(dim=1, tag=outer_loop_physical, name='Outer boundary')

    model.synchronize()
    model.removeAllDuplicates()

    gmsh.model.mesh.generate(dim=2)
    gmsh.model.mesh.removeDuplicateNodes()

    if filename is not None:
        gmsh.write(filename)
    else:
        current_directory = os.getcwd()
        path = Path(current_directory) / 'mesh_output'
        if not path.exists():
            path.mkdir()
        gmsh.write('mesh_output/test_out.msh')

    print(f'Inner bd tag: {inner_loop_physical}, outer bd tag: {outer_loop_physical}')

    if return_model:
        return gmsh.model
    else:
        gmsh.finalize()


def create_xdmf_mesh_from_msh_file(filename):
    """
    Function converts triangle mesh from msh to fenicx prefered xdmf. Assumes that the mesh is 2D
    and prunes the extra z-coordinate.
    Assumes that the mesh has no physical groups! TODO: fix this and unite with the other mesh saving function.
    """
    # based on
    # https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html#read-in-msh-files-with-dolfinx
    # and
    # https://github.com/jorgensd/dolfinx-tutorial/blob/v0.8.0/chapter3/subdomains.ipynb
    mesh_from_file = meshio.read(f"{filename}.msh")
    cells = mesh_from_file.get_cells_type("triangle")
    points = mesh_from_file.points[:, :2]  # removes the z-coordinate
    out_mesh = meshio.Mesh(points=points, cells={"triangle": cells})
    meshio.write(f"{filename}.xdmf", out_mesh)
    print(f"Mesh written to files: {filename}.xdmf, {filename}.h5")


def create_xdmf_mesh_from_msh_with_cell_data(filename):
    # based on
    # https://jsdokken.com/dolfinx-tutorial/chapter3/subdomains.html#read-in-msh-files-with-dolfinx
    # and
    # https://github.com/jorgensd/dolfinx-tutorial/blob/v0.8.0/chapter3/subdomains.ipynb

    mesh_in = meshio.read(f"{filename}.msh")

    cell_types = ["triangle", "line"]
    filename_suffixes = {"triangle": '', "line": "boundary"}
    points = mesh_in.points[:, :2]  # cut the z-coordinates for 2d mesh TODO: use as parameter
    for cell_type in cell_types:
        cell_data = mesh_in.get_cell_data("gmsh:physical", cell_type)
        cells = mesh_in.get_cells_type(cell_type)
        out_mesh = meshio.Mesh(points=points, cells={cell_type: cells}, cell_data={"Domains": [cell_data]})
        name = f"{filename}_{filename_suffixes[cell_type]}" if filename_suffixes[cell_type] else f"{filename}"
        meshio.write(f"{name}.xdmf", out_mesh)
        print(f"Mesh written to files: {name}.xdmf, {name}.h5")


def generate_tsx_mesh_with_regions(filename='tsx_ellipses_regions_coarse', no_of_rays=4, ellipses_list=None):
    # TODO: think through what are suitable inputs
    if not ellipses_list:
        x_axis = 4.375 / 2
        y_axis = 3.5 / 2
        ellipses_list = [Ellipse(x_axis * k, y_axis * k) for k in [1, 2.3, 3.5]]
        # ellipses_list.append(Ellipse(x_axis+1.5+1.25, y_axis+1.5+1.25))
        # ellipses_list.append(Ellipse(x_axis+4.0+1.25, y_axis+4.0+1.25))

    # hack-ish shift in angle to avoid bad ellipses
    magical_constant = 0.001

    corners = [(50, -50, 0), (50, 50, 0), (-50, 50, 0), (-50, -50, 0)]
    rays_list = [Ray(2.0*pi*n/no_of_rays + magical_constant + pi/4) for n in range(no_of_rays)]

    # TODO: Should these two be the actual input?
    _ = create_mesh(rays_list, ellipses_list, corners, filename=f'{filename}.msh')
    create_xdmf_mesh_from_msh_with_cell_data(filename)

def generate_tsx_mesh_very_coarse(filename='tsx_ellipses_very_coarse', no_of_rays=4, ellipses_list=None):
    # TODO: think through what are suitable inputs
    if not ellipses_list:
        x_axis = 4.375 / 2
        y_axis = 3.5 / 2
        ellipses_list = [Ellipse(x_axis + k, y_axis + k) for k in [0.0, 0.7, 6.0]]
        # ellipses_list.append(Ellipse(x_axis+1.5+1.25, y_axis+1.5+1.25))
        # ellipses_list.append(Ellipse(x_axis+4.0+1.25, y_axis+4.0+1.25))

    # hack-ish shift in angle to avoid bad ellipses
    magical_constant = 0.001

    resolution_0 = 0.2  # around inner boundary (ellipse)
    resolution_1 = 0.5 # should be around all the other ellipses
    resolution_2 = 10  # ok, around outer boundary (square)

    corners = [(50, -50, 0), (50, 50, 0), (-50, 50, 0), (-50, -50, 0)]
    rays_list = [Ray(2.0*pi*n/no_of_rays + magical_constant + pi/4) for n in range(no_of_rays)]

    # TODO: Should these two be the actual input?
    _ = create_mesh(rays_list, ellipses_list, corners, resolution_2, resolution_1,
                    resolution_inner_boundary=resolution_0, filename=f'{filename}.msh')
    create_xdmf_mesh_from_msh_with_cell_data(filename)

def test():
    from math import e

    magical_constant = pi / e / 10  # to offset problematic symmetries
    no_of_rays = 5
    rays = [Ray(2.0 * pi * n / no_of_rays + magical_constant) for n in range(no_of_rays)]
    ellipses = [Ellipse(4.0 * k, 3.0 * k) for k in (2, 4, 5, 6, 7)]
    corners = [(50, -50, 0), (50, 50, 0), (-50, 50, 0), (-50, -50, 0)]

    model = create_mesh(rays, ellipses, corners)
    create_xdmf_mesh_from_msh_with_cell_data('mesh_output/test_out')


if __name__ == "__main__":
    #generate_tsx_mesh_with_regions()
    generate_tsx_mesh_very_coarse()