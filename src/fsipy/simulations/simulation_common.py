import json
from typing import List, Union, Tuple
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi
from dolfin import Mesh, assemble, Constant, MPI, HDF5File, Measure, inner, MeshFunction, FunctionSpace, \
    Function, sqrt, Expression, TrialFunction, TestFunction, LocalSolver, dx


def load_mesh_and_data(mesh_path: Union[str, Path]) -> Tuple[Mesh, MeshFunction, MeshFunction]:
    """
    Load mesh, boundary data, and domain data from an HDF5 file.

    Args:
        mesh_path (str or Path): Path to the HDF5 file containing mesh and data.

    Returns:
        Tuple[Mesh, MeshFunction, MeshFunction]:
            A tuple containing:
            - mesh (dolfin.Mesh): Loaded mesh.
            - boundaries (dolfin.MeshFunction): Loaded boundary data.
            - domains (dolfin.MeshFunction): Loaded domain data.
    """
    mesh_path = Path(mesh_path)

    # Initialize an empty mesh
    mesh = Mesh()

    # Open the HDF5 file in read-only mode
    hdf5 = HDF5File(mesh.mpi_comm(), str(mesh_path), "r")

    # Read mesh data
    hdf5.read(mesh, "/mesh", False)

    # Create MeshFunction objects for boundaries and domains
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    domains = MeshFunction("size_t", mesh, mesh.topology().dim())

    # Read boundary and domain data
    hdf5.read(boundaries, "/boundaries")
    hdf5.read(domains, "/domains")

    return mesh, boundaries, domains


def print_mesh_summary(mesh: Mesh) -> None:
    """
    Print a summary of geometric information about the volumetric mesh.

    Args:
        mesh (dolfin.Mesh): Volumetric mesh object.
    """
    # Check if the input mesh is of the correct type
    if not isinstance(mesh, Mesh):
        raise ValueError("Invalid mesh object provided.")

    # Extract local x, y, and z coordinates from the mesh
    local_x_coords = mesh.coordinates()[:, 0]
    local_y_coords = mesh.coordinates()[:, 1]
    local_z_coords = mesh.coordinates()[:, 2]

    # Create a dictionary to store local geometric information
    local_info = {
        "x_min": local_x_coords.min(),
        "x_max": local_x_coords.max(),
        "y_min": local_y_coords.min(),
        "y_max": local_y_coords.max(),
        "z_min": local_z_coords.min(),
        "z_max": local_z_coords.max(),
        "num_cells": mesh.num_cells(),
        "num_edges": mesh.num_edges(),
        "num_faces": mesh.num_faces(),
        "num_facets": mesh.num_facets(),
        "num_vertices": mesh.num_vertices()
    }

    # Gather local information from all processors to processor 0
    comm = mesh.mpi_comm()
    gathered_info = comm.gather(local_info, 0)
    num_cells_per_processor = comm.gather(local_info["num_cells"], 0)

    # Compute the volume of the mesh
    dx = Measure("dx", domain=mesh)
    volume = assemble(Constant(1) * dx)

    # Print the mesh information summary only on processor 0
    if MPI.rank(comm) == 0:
        # Combine gathered information to get global information
        combined_info = {key: sum(info[key] for info in gathered_info) for key in gathered_info[0]}

        # Print various mesh statistics
        print("=== Mesh Information Summary ===")
        print(f"X range: {combined_info['x_min']} to {combined_info['x_max']} "
              f"(delta: {combined_info['x_max'] - combined_info['x_min']:.4f})")
        print(f"Y range: {combined_info['y_min']} to {combined_info['y_max']} "
              f"(delta: {combined_info['y_max'] - combined_info['y_min']:.4f})")
        print(f"Z range: {combined_info['z_min']} to {combined_info['z_max']} "
              f"(delta: {combined_info['z_max'] - combined_info['z_min']:.4f})")
        print(f"Number of cells: {combined_info['num_cells']}")
        print(f"Number of cells per processor: {int(np.mean(num_cells_per_processor))}")
        print(f"Number of edges: {combined_info['num_edges']}")
        print(f"Number of faces: {combined_info['num_faces']}")
        print(f"Number of facets: {combined_info['num_facets']}")
        print(f"Number of vertices: {combined_info['num_vertices']}")
        print(f"Volume: {volume}")
        print(f"Number of cells per volume: {combined_info['num_cells'] / volume}\n")


def load_mesh_info(mesh_path: Union[str, Path]) -> Tuple[List[int], List[int], int, float, List[float], List[float]]:
    """
    Load and process mesh information from a JSON file.

    Args:
        mesh_path (str or Path): The path to the mesh file.

    Returns:
        Tuple[List[int], List[int], int, float, List[float], List[float]]:
            A tuple containing:
            - id_in (List[int]): List of inlet IDs.
            - id_out (List[int]): List of outlet IDs.
            - id_wall (int): Computed wall ID.
            - Q_mean (float): Mean flow rate.
            - area_ratio (List[float]): List of area ratios.
            - area_inlet (List[float]): List of inlet areas.

    Raises:
        FileNotFoundError: If the info file is not found.
    """
    info_file = Path(mesh_path).with_name(Path(mesh_path).stem + "_info.json")

    try:
        # Read and parse the JSON info file
        with open(info_file) as f:
            info = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Info file '{info_file}' not found.")

    # Extract information from the JSON data
    id_in = info['inlet_id']
    id_out = info['outlet_ids']
    id_wall = min(id_in + id_out) - 1
    Q_mean = info['mean_flow_rate']
    area_ratio = info['area_ratio']
    area_inlet = info['inlet_area']

    return id_in, id_out, id_wall, Q_mean, area_ratio, area_inlet


def load_probe_points(mesh_path: Union[str, Path]) -> np.ndarray:
    """
    Load probe points from a corresponding file based on the mesh file's path.

    Args:
        mesh_path (str or Path): The path to the mesh file.

    Returns:
        np.ndarray: An array containing the loaded probe points.
    """
    mesh_path = Path(mesh_path)
    rel_path = mesh_path.stem + "_probe_point"
    probe_points_path = mesh_path.parent / rel_path
    probe_points = np.load(probe_points_path, encoding='latin1', fix_imports=True, allow_pickle=True)

    return probe_points


def print_probe_points(v: Function, p: Function, probe_points: List[Union[float, np.ndarray]]) -> None:
    """
    Print velocity and pressure at probe points.

    Args:
        v (dolfin.Function): Velocity function with components v.sub(0), v.sub(1), and v.sub(2).
        p (dolfin.Function): Pressure function.
        probe_points (list): List of probe points.

    Returns:
        None
    """
    for i, point in enumerate(probe_points):
        # Extract components of velocity and pressure at the probe point
        uu = peval(v.sub(0), point)
        vv = peval(v.sub(1), point)
        ww = peval(v.sub(2), point)
        pp = peval(p, point)

        if MPI.rank(MPI.comm_world) == 0:
            print(f"Probe Point {i}: Velocity: ({uu}, {vv}, {ww}) | Pressure: {pp}")


def peval(f: Function, x: Union[float, np.ndarray]) -> np.ndarray:
    """
    Parallel synchronized evaluation of a function.

    Args:
        f (dolfin.Function): Function to be evaluated.
        x (Union[float, np.ndarray]): Input value for the function.

    Returns:
        np.ndarray: Evaluated function values.
    """
    try:
        yloc = f(x)
    except RuntimeError:
        yloc = np.inf * np.ones(f.value_shape())

    comm = MPI.comm_world
    yglob = np.zeros_like(yloc)
    comm.Allreduce(yloc, yglob, op=mpi.MIN)

    return yglob


def local_project(f: Function, V: FunctionSpace) -> Function:
    """
    Project a given function 'f' onto a finite element function space 'V' in a reusable way.

    Args:
        f (Function): The function to be projected onto 'V'.
        V (FunctionSpace): The finite element function space.

    Returns:
        Function: The projected solution in 'V'.
    """
    u = TrialFunction(V)
    v = TestFunction(V)
    a_proj = inner(u, v) * dx
    b_proj = inner(f, v) * dx
    solver = LocalSolver(a_proj, b_proj)
    solver.factorize()
    projected_u = Function(V)
    solver.solve_local_rhs(projected_u)

    return projected_u


def calculate_and_print_flow_properties(dt: float, mesh: Mesh, v: Function, inlet_area: float, mu_f: float,
                                        n: Expression, dsi: Measure) -> None:
    """
    Calculate and print flow properties.

    Args:
        dt (float): Time step size.
        mesh (dolfin.Mesh): Mesh object.
        v (dolfin.Function): Velocity field.
        inlet_area (float): Inlet area.
        mu_f (float): Fluid dynamic viscosity.
        n (dolfin.Expression): FacetNormal expression.
        dsi (dolfin.Measure): Measure for inlet boundary.

    Returns:
        None
    """
    # Calculate the DG vector of velocity magnitudes
    DG = FunctionSpace(mesh, "DG", 0)
    V_vector = local_project(sqrt(inner(v, v)), DG).vector().get_local()
    h = mesh.hmin()

    # Calculate flow rate at the inlet
    flow_rate_inlet = abs(assemble(inner(v, n) * dsi))

    # Calculate local mean, min, and max velocities
    local_V_mean = V_vector.mean()
    local_V_min = V_vector.min()
    local_V_max = V_vector.max()

    comm = mesh.mpi_comm()
    # Gather data from all processes
    V_mean = comm.gather(local_V_mean, 0)
    V_min = comm.gather(local_V_min, 0)
    V_max = comm.gather(local_V_max, 0)

    if MPI.rank(comm) == 0:
        # Calculate mean, min, and max velocities
        v_mean = np.mean(V_mean)
        v_min = min(V_min)
        v_max = max(V_max)

        # Calculate diameter at the inlet and Reynolds numbers
        diam_inlet = np.sqrt(4 * inlet_area / np.pi)
        Re_mean = v_mean * diam_inlet / mu_f
        Re_min = v_min * diam_inlet / mu_f
        Re_max = v_max * diam_inlet / mu_f

        # Calculate CFL numbers
        CFL_mean = v_mean * dt / h
        CFL_min = v_min * dt / h
        CFL_max = v_max * dt / h

        # Print the flow properties
        print("Flow Properties:")
        print(f"  Flow Rate at Inlet: {flow_rate_inlet}")
        print(f"  Velocity (mean, min, max): {v_mean}, {v_min}, {v_max}")
        print(f"  CFL (mean, min, max): {CFL_mean}, {CFL_min}, {CFL_max}")
        print(f"  Reynolds Numbers (mean, min, max): {Re_mean}, {Re_min}, {Re_max}")
