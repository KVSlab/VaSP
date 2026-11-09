import json
import math
from typing import List, Union, Tuple, NamedTuple
from pathlib import Path

import numpy as np
from mpi4py import MPI as mpi
from dolfin import Mesh, assemble, MPI, HDF5File, Measure, inner, MeshFunction, FunctionSpace, \
    Function, sqrt, Expression, TrialFunction, TestFunction, LocalSolver, dx, UserExpression
from turtleFSI.modules.common import J_


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
        u_eval = peval(v, [float(x) for x in point.tolist()])
        pp = peval(p, [float(x) for x in point.tolist()])

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
        d_eval = peval(d, [float(x) for x in point.tolist()])
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


def compute_minimum_jacobian(mesh: Mesh, d: Function, local_rhs: bool = False) -> float:
    """
    Compute the minimum Jacobian of the mesh. The minimum Jacobian is a measure of the mesh distortion.
    The value 1 indicates no distortion, while a value less than 0 indicates mesh entanglement.

    Args:
        mesh (dolfin.Mesh): The mesh object.
        d (dolfin.Function): The displacement function.
        local_rhs (bool, optional): If True, solve using a local right-hand side assembly.

    Returns:
        float: The minimum Jacobian of the displacement.
    """
    # Create a function space for the Jacobian
    DG = FunctionSpace(mesh, "DG", 0)
    # compute the Jacobian of the displacement
    jacobian = J_(d)
    # project the Jacobian to the DG function space
    projected_jacobian = local_project(jacobian, DG, local_rhs)
    min_jacobian = np.min(projected_jacobian.vector().get_local())
    global_min_jacobian = MPI.min(MPI.comm_world, min_jacobian)

    if MPI.rank(MPI.comm_world) == 0:
        print(f"Minimum Jacobian: {global_min_jacobian}")

        if global_min_jacobian <= 0:
            print("Warning: Negative Jacobian detected.")

    return min_jacobian


class InterfacePressure(UserExpression):
    """
    An expression for the fluid-solid interface pressure based on Fourier coefficients.
    """
    def __init__(self, t, t_ramp_start, t_ramp_end, An, Bn, period, P_mean, **kwargs):
        """
        Initialize the interface pressure expression.
        """
        self.t = t
        self.t_ramp_start = t_ramp_start
        self.t_ramp_end = t_ramp_end
        self.An = An
        self.Bn = Bn
        self.omega = (2.0 * np.pi / period)
        self.P_mean = P_mean
        self.p_0 = 0.0  # Initial pressure
        self.P = self.p_0  # Apply initial pressure to inner pressure variable
        super().__init__(**kwargs)

    def update(self, t):
        """
        Update the interface pressure at a given time.
        """
        self.t = t
        # apply a sigmoid ramp to the pressure
        if self.t < self.t_ramp_start:
            ramp_factor = 0.0
        if self.t_ramp_start <= self.t < self.t_ramp_end:
            ramp_factor = -0.5 * np.cos(np.pi * (self.t - self.t_ramp_start) / (self.t_ramp_end - self.t_ramp_start)) \
                + 0.5
        if self.t >= self.t_ramp_end:
            ramp_factor = 1.0
        if MPI.rank(MPI.comm_world) == 0:
            print("ramp_factor = {} m^3/s".format(ramp_factor))

        # Calculate Pn (normalized pressure) from Fourier Coefficients
        Pn = 0 + 0j
        for i in range(len(self.An)):
            Pn = Pn + (self.An[i] - self.Bn[i] * 1j) * np.exp(1j * i * self.omega * self.t)
        Pn = abs(Pn)

        # Multiply by mean pressure and ramp factor
        self.P = ramp_factor * Pn * self.P_mean
        if MPI.rank(MPI.comm_world) == 0:
            print("Instantaneous normal stress prescribed at the FSI interface {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P

    def value_shape(self):
        return ()


# Parameters required by each solid material model of turtleFSI (see Piola1 in turtleFSI/modules/common.py)
SOLID_MODEL_PARAMETERS = {
    "StVenantKirchoff": ("rho_s", "mu_s", "lambda_s"),
    "StVenantKirchoffEnergy": ("rho_s", "mu_s", "lambda_s"),
    "NeoHookean": ("rho_s", "mu_s", "lambda_s"),
    "MooneyRivlin": ("rho_s", "mu_s", "lambda_s", "C01", "C10", "C11"),
    "Gent": ("rho_s", "mu_s", "Jm"),
}

# Parameters required by the viscoelastic forms of turtleFSI (see Svisc_D in turtleFSI/modules/common.py)
VISCOELASTIC_FORMS = ("Form1", "Form2")
VISCOELASTIC_PARAMETERS = ("mu_visc_s", "lambda_visc_s")

# Parameters that must be strictly positive
POSITIVE_PARAMETERS = ("rho_s", "mu_s", "Jm")


def as_list(value) -> list:
    """
    Return value as a list: None -> [], a list or tuple -> list, anything else -> [value].
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def _is_number(value) -> bool:
    return (isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(value, bool)
            and math.isfinite(value))


def check_solid_properties(solid_properties: Union[dict, list], dx_s_id: Union[int, list]) -> None:
    """
    Check the solid properties of each solid region, and raise an error listing all the problems found.

    turtleFSI pairs solid_properties[i] with dx_s_id[i] by position, and only fails with a KeyError
    (or not at all) when a parameter is missing. This function makes these problems explicit.

    Args:
        solid_properties (dict or list): Solid properties of each solid region, as used by turtleFSI.
        dx_s_id (int or list): ID(s) of the solid region(s).

    Raises:
        ValueError: If a material model is missing or unknown, if a required parameter is missing or
            invalid, or if solid_properties and dx_s_id do not match.
    """
    errors = solid_properties_errors(solid_properties, dx_s_id)
    if errors:
        raise ValueError("Invalid solid material parameters:\n  - " + "\n  - ".join(errors))


def solid_properties_errors(solid_properties: Union[dict, list], dx_s_id: Union[int, list]) -> List[str]:
    """
    Same checks as check_solid_properties, but return the list of problems found instead of raising.
    """
    properties = as_list(solid_properties)
    ids = as_list(dx_s_id)
    errors = []

    if len(properties) != len(ids):
        errors.append(f"{len(properties)} entries in solid_properties but {len(ids)} solid region(s) in "
                      f"dx_s_id={dx_s_id}: there must be one entry per solid region")

    for i, region_properties in enumerate(properties):
        if not isinstance(region_properties, dict):
            errors.append(f"solid_properties[{i}]={region_properties!r} is not a dictionary")
            continue
        region = ids[i] if i < len(ids) else region_properties.get("dx_s_id", f"#{i}")
        prefix = f"solid region dx_s_id={region}"

        if "dx_s_id" in region_properties and i < len(ids) and region_properties["dx_s_id"] != ids[i]:
            errors.append(f"{prefix}: solid_properties[{i}] has dx_s_id={region_properties['dx_s_id']}, but turtleFSI "
                          f"pairs solid_properties and dx_s_id by position (dx_s_id[{i}]={ids[i]})")

        model = region_properties.get("material_model")
        if model is None:
            errors.append(f"{prefix}: material_model is not defined")
            continue
        if model not in SOLID_MODEL_PARAMETERS:
            errors.append(f"{prefix}: unknown material_model '{model}', choose from {list(SOLID_MODEL_PARAMETERS)}")
            continue

        required = list(SOLID_MODEL_PARAMETERS[model])
        viscoelasticity = region_properties.get("viscoelasticity")
        if viscoelasticity in VISCOELASTIC_FORMS:
            required += VISCOELASTIC_PARAMETERS
        elif viscoelasticity not in (None, "None"):
            errors.append(f"{prefix}: unknown viscoelasticity '{viscoelasticity}', choose from "
                          f"{list(VISCOELASTIC_FORMS)} or leave it out")

        missing = [key for key in required if key not in region_properties]
        if missing:
            errors.append(f"{prefix} ({model}): missing {', '.join(missing)}")

        for key in required:
            if key not in region_properties:
                continue
            value = region_properties[key]
            if not _is_number(value):
                errors.append(f"{prefix} ({model}): {key}={value!r} is not a finite number")
            elif key in POSITIVE_PARAMETERS and value <= 0:
                errors.append(f"{prefix} ({model}): {key}={value} must be positive")

    return errors


def check_robin_parameters(robin_bc: bool, k_s, c_s, ds_s_id) -> None:
    """
    Check the parameters of the Robin boundary condition of the solid, if it is used.

    Without ds_s_id, turtleFSI applies the Robin condition on the whole exterior boundary of the mesh,
    including the fluid inlets and outlets, without any error.

    Args:
        robin_bc (bool): True if the Robin boundary condition is used.
        k_s (list): Elastic coefficient for each boundary in ds_s_id.
        c_s (list): Viscous coefficient for each boundary in ds_s_id.
        ds_s_id (int or list): ID(s) of the boundaries where the Robin condition is applied.

    Raises:
        ValueError: If robin_bc is True and the parameters are missing or inconsistent.
    """
    errors = robin_parameters_errors(robin_bc, k_s, c_s, ds_s_id)
    if errors:
        raise ValueError("Invalid Robin boundary condition parameters:\n  - " + "\n  - ".join(errors))


def robin_parameters_errors(robin_bc: bool, k_s, c_s, ds_s_id) -> List[str]:
    """
    Same checks as check_robin_parameters, but return the list of problems found instead of raising.
    A value of None means that the parameter is not defined.
    """
    if not robin_bc:
        return []

    errors = []
    if ds_s_id is None:
        errors.append("robin_bc=True but ds_s_id is not defined: give the ID(s) of the boundaries where the "
                      "Robin condition is applied (otherwise turtleFSI applies it on the whole exterior boundary, "
                      "including the fluid inlets and outlets)")
    n_ids = len(as_list(ds_s_id))

    for name, value in (("k_s", k_s), ("c_s", c_s)):
        if value is None:
            errors.append(f"robin_bc=True but {name} is not defined: give one value per ID in ds_s_id")
            continue
        if not isinstance(value, (list, tuple)):
            errors.append(f"{name}={value!r} must be a list with one value per ID in ds_s_id")
            continue
        if ds_s_id is not None and len(value) != n_ids:
            errors.append(f"{name} has {len(value)} value(s) but ds_s_id={ds_s_id} has {n_ids} ID(s)")
        not_numbers = [v for v in value if not _is_number(v)]
        if not_numbers:
            errors.append(f"{name} contains values that are not finite numbers: {not_numbers}")

    return errors


def check_material_parameters(solid_properties: list, dx_s_id_list: list, robin_bc: bool = False, k_s=None,
                              c_s=None, ds_s_id=None, **namespace) -> None:
    """
    Check the solid material parameters and the Robin boundary condition parameters used in the simulation.
    Meant to be called from the initiate() function of a problem file, i.e. after turtleFSI has assigned the
    properties to each region: initiate(**namespace) -> check_material_parameters(**namespace).
    The material models used are printed by turtleFSI (print_material_models in turtleFSI/modules/domain.py).
    """
    check_solid_properties(solid_properties, dx_s_id_list)
    check_robin_parameters(robin_bc, k_s, c_s, ds_s_id)
