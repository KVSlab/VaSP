import json
from typing import List, Union, Tuple, NamedTuple
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi
from dolfin import Mesh, assemble, MPI, HDF5File, Measure, inner, MeshFunction, FunctionSpace, \
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


class MeshInfo(NamedTuple):
    """
    Represents mesh information.

    Attributes:
        id_in (List[int]): List of inlet IDs.
        id_out (List[int]): List of outlet IDs.
        id_wall (int): Computed wall ID.
        Q_mean (float): Mean flow rate.
        area_ratio (List[float]): List of area ratios.
        area_inlet (List[float]): List of inlet areas.
        solid_side_wall_id (int): ID for solid side wall.
        interface_fsi_id (int): ID for the FSI interface.
        interface_outer_id (int): ID for the outer interface.
        volume_id_fluid (int): ID for the fluid volume.
        volume_id_solid (int): ID for the solid volume.
        branch_ids_offset (int): Offset solid mesh IDs when extracting a branch.
    """
    id_in: List[int]
    id_out: List[int]
    id_wall: int
    Q_mean: float
    area_ratio: List[float]
    area_inlet: List[float]
    solid_side_wall_id: int
    interface_fsi_id: int
    solid_outer_wall_id: int
    fluid_volume_id: int
    solid_volume_id: int
    branch_ids_offset: int


def load_mesh_info(mesh_path: Union[str, Path]) -> MeshInfo:
    """
    Load and process mesh information from a JSON file.

    Args:
        mesh_path (str or Path): The path to the mesh file.

    Returns:
        MeshInfo: A named tuple containing mesh information.
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
    solid_side_wall_id = info["solid_side_wall_id"]
    interface_fsi_id = info["interface_fsi_id"]
    solid_outer_wall_id = info["solid_outer_wall_id"]
    fluid_volume_id = info["fluid_volume_id"]
    solid_volume_id = info["solid_volume_id"]
    branch_ids_offset = info["branch_ids_offset"]

    return MeshInfo(id_in, id_out, id_wall, Q_mean, area_ratio, area_inlet, solid_side_wall_id, interface_fsi_id,
                    solid_outer_wall_id, fluid_volume_id, solid_volume_id, branch_ids_offset)


def load_probe_points(mesh_path: Union[str, Path]) -> np.ndarray:
    """
    Load probe points from a corresponding file based on the mesh file's path.

    Args:
        mesh_path (str or Path): The path to the mesh file.

    Returns:
        np.ndarray: An array containing the loaded probe points.
    """
    mesh_path = Path(mesh_path)
    rel_path = mesh_path.stem + "_probe_point.json"
    probe_points_path = mesh_path.parent / rel_path
    with open(probe_points_path) as f:
        probe_points = np.array(json.load(f))

    return probe_points


def load_solid_probe_points(mesh_path: Union[str, Path]) -> np.ndarray:
    """
    Load solid probe points from a corresponding file based on the mesh file's path.

    Args:
        mesh_path (str or Path): The path to the mesh file.

    Returns:
        np.ndarray: An array containing the loaded probe points.
    """
    mesh_path = Path(mesh_path)
    solid_probe_file_name = mesh_path.stem + "_solid_probe.json"
    solid_probe_path = mesh_path.parent / solid_probe_file_name
    with open(solid_probe_path) as f:
        solid_probe_points = np.array(json.load(f))

    return solid_probe_points


def print_probe_points(v: Function, p: Function, probe_points: List[np.ndarray]) -> None:
    """
    Print velocity and pressure at probe points.

    Args:
        v (dolfin.Function): Velocity function with components v.sub(0), v.sub(1), and v.sub(2).
        p (dolfin.Function): Pressure function.
        probe_points (list): List of probe points.

    Returns:
        None
    """
    # make sure that extrapolation is not allowed
    if v.get_allow_extrapolation():
        v.set_allow_extrapolation(False)

    if p.get_allow_extrapolation():
        p.set_allow_extrapolation(False)

    for i, point in enumerate(probe_points):
        u_eval = peval(v, point.tolist())
        pp = peval(p, point.tolist())

        if MPI.rank(MPI.comm_world) == 0:
            print(f"Probe Point {i}: Velocity: ({u_eval[0]}, {u_eval[1]}, {u_eval[2]}) | Pressure: {pp}")


def print_solid_probe_points(d: Function, probe_points: List[np.ndarray]) -> None:
    """
    Print displacement at probe points.

    Args:
        d (dolfin.Function): Displacement function with components d.sub(0), d.sub(1), and d.sub(2).
        probe_points (list): List of probe points.
    """
    # make sure that extrapolation is not allowed
    if d.get_allow_extrapolation():
        d.set_allow_extrapolation(False)

    for i, point in enumerate(probe_points):
        d_eval = peval(d, point.tolist())
        if MPI.rank(MPI.comm_world) == 0:
            print(f"Probe Point {i}: Displacement: {d_eval[0], d_eval[1], d_eval[2]}")


def peval(f: Function, x: Union[float, List[float]]) -> np.ndarray:
    """
    Parallel synchronized evaluation of a function.

    Args:
        f (dolfin.Function): Function to be evaluated.
        x (List[float]): Point at which to evaluate the function.

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


def local_project(f: Function, V: FunctionSpace, local_rhs: bool = False) -> Function:
    """
    Project a given function 'f' onto a finite element function space 'V' in a reusable way.

    Args:
        f (Function): The function to be projected onto 'V'.
        V (FunctionSpace): The finite element function space.
        local_rhs (bool, optional): If True, solve using a local right-hand side assembly.
            If False (default), solve using a global right-hand side assembly.

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
    if local_rhs:
        solver.solve_local_rhs(projected_u)
    else:
        solver.solve_global_rhs(projected_u)

    return projected_u


def calculate_and_print_flow_properties(dt: float, mesh: Mesh, v: Function, inlet_area: float, mu_f: float,
                                        rho_f: float, n: Expression, dsi: Measure, local_rhs: bool = False) -> None:
    """
    Calculate and print flow properties.

    Args:
        dt (float): Time step size.
        mesh (dolfin.Mesh): Mesh object.
        v (dolfin.Function): Velocity field.
        inlet_area (float): Inlet area.
        mu_f (float): Fluid dynamic viscosity.
        rho_f (float): Fluid density.
        n (dolfin.Expression): FacetNormal expression.
        dsi (dolfin.Measure): Measure for inlet boundary.
        local_rhs (bool, optional): If True, solve using a local right-hand side assembly.
            If False (default), solve using a global right-hand side assembly.

    Returns:
        None
    """
    # Calculate the DG vector of velocity magnitudes
    DG = FunctionSpace(mesh, "DG", 0)
    V_vector = local_project(sqrt(inner(v, v)), DG, local_rhs).vector().get_local()

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

    # compute the minimum cell diameter in the mesh
    h = mesh.hmin()
    h_min = MPI.min(MPI.comm_world, h)
    
    if MPI.rank(comm) == 0:
        # Calculate mean, min, and max velocities
        v_mean = np.mean(V_mean)
        v_min = min(V_min)
        v_max = max(V_max)

        # Calculate diameter at the inlet and Reynolds numbers
        diam_inlet = np.sqrt(4 * inlet_area / np.pi)
        Re_mean = rho_f * v_mean * diam_inlet / mu_f
        Re_min = rho_f * v_min * diam_inlet / mu_f
        Re_max = rho_f * v_max * diam_inlet / mu_f

        # Calculate CFL numbers
        CFL_mean = v_mean * dt / h_min * v.ufl_element().degree()
        CFL_min = v_min * dt / h_min * v.ufl_element().degree()
        CFL_max = v_max * dt / h_min * v.ufl_element().degree()

        # Print the flow properties
        print("Flow Properties:")
        print(f"  Flow Rate at Inlet: {flow_rate_inlet}")
        print(f"  Velocity (mean, min, max): {v_mean}, {v_min}, {v_max}")
        print(f"  CFL (mean, min, max): {CFL_mean}, {CFL_min}, {CFL_max}")
        print(f"  Reynolds Numbers (mean, min, max): {Re_mean}, {Re_min}, {Re_max}")
    
    exit(0)