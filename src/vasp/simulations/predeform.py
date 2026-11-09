# Copyright (c) 2025 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Problem file for predeforming mesh, i.e approximating a zero-pressure geometry.
Here, we inflate the geometry to P_final and v_max final, representing the cycle-average
pressure and centerline velocity in a pulsatile simulation. We then apply the reverse of
this deformation as an approximate "zero-pressure" geometry, using "predeform_mesh.py"

The predeform takes its parameters from the problem file of the main simulation, so that the
predeform and the main simulation use the same material models and parameters:

    turtleFSI -p predeform --main_problem=aneurysm

--main_problem is the name of the main problem file (looked up in the current directory, then in
vasp.simulations and turtleFSI.problems), or a path to it, e.g. --main_problem=path/to/my_problem.py

Taken from the main problem file (arguments given on the command line still have the last word):
  - solid: material_model (mandatory), solid_properties or rho_s/mu_s/lambda_s, dx_s_id
  - fluid: fluid_properties or rho_f/mu_f, dx_f_id, nonnewtonian, cross_params
  - Robin boundary condition, if used: robin_bc, k_s, c_s, ds_s_id
  - predeform load: P_final and v_max_final (mandatory)
  - boundary IDs, as int or list: inlet_id and fsi_id (mandatory), rigid_id and inlet_outlet_s_id (optional)
  - get_mesh_domain_and_boundaries, if the main problem file defines it, together with the parameters
    it takes (e.g. mesh_path, fsi_region), so that the regions created in the code are the same
  - folder: the results are stored in <folder>/predeform_results
Defined in this file: time stepping, solver settings, load ramps and output settings.
"""
import copy as _copy
import importlib as _importlib
import importlib.util as _importlib_util
import inspect as _inspect
import sys as _sys
from pathlib import Path as _Path

import numpy as np
from turtleFSI.problems import *
import turtleFSI.problems as _turtle_problems
from dolfin import HDF5File, Mesh, MeshFunction, assemble, UserExpression, FacetNormal, ds, \
    DirichletBC, Measure, inner, parameters, SpatialCoordinate, Constant

from vasp.simulations.simulation_common import calculate_and_print_flow_properties, as_list, \
    solid_properties_errors, robin_parameters_errors, check_material_parameters

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])

# Parameters taken from the main problem file
_SOLID_KEYS = ("material_model", "solid_properties", "rho_s", "mu_s", "lambda_s", "nu_s", "dx_s_id")
_FLUID_KEYS = ("fluid_properties", "rho_f", "mu_f", "dx_f_id", "nonnewtonian", "cross_params")
_ROBIN_KEYS = ("robin_bc", "k_s", "c_s", "ds_s_id")
_LOAD_KEYS = ("P_final", "v_max_final")
_BOUNDARY_ID_KEYS = ("inlet_id", "fsi_id", "rigid_id", "inlet_outlet_s_id")
_INHERITED_KEYS = _SOLID_KEYS + _FLUID_KEYS + _ROBIN_KEYS + _LOAD_KEYS + _BOUNDARY_ID_KEYS

# Parameters that belong to the predeform, and are never taken from the main problem file
_PREDEFORM_KEYS = ("T", "dt", "theta", "save_step", "checkpoint_step", "linear_solver", "atol", "rtol",
                   "recompute", "recompute_tstep", "lmbda", "t_start_v", "t_end_v", "t_start_p", "t_end_p",
                   "extrapolation", "extrapolation_sub_type", "save_deg", "folder", "sub_folder",
                   "restart_folder", "main_problem")

# Sub folder of the main problem folder where the predeform results are stored
_PREDEFORM_SUB_FOLDER = "predeform_results"


class _NotSet:
    """Marker for the parameters that are not defined in the main problem file."""

    def __repr__(self):
        return "<not set>"


_NOT_SET = _NotSet()

# get_mesh_domain_and_boundaries of the main problem file, set in set_problem_parameters
_main_problem: dict = {}


def set_problem_parameters(default_variables, **namespace):
    # Copy of the turtleFSI default values, before they are changed below
    turtle_defaults = _copy.deepcopy(default_variables)

    default_variables.update(
        dict(
            # Temporal parameters
            T=1.0,  # Simulation end time
            dt=0.01,  # Time step size
            theta=1.0,  # backward Euler time integration
            save_step=10,  # Save frequency of files for visualization
            checkpoint_step=50,  # Save frequency of checkpoint files
            # Linear solver parameters
            linear_solver="mumps",
            atol=1e-6,  # Absolute tolerance in the Newton solver
            rtol=1e-6,  # Relative tolerance in the Newton solver
            recompute=20,  # Recompute the Jacobian matrix within time steps
            recompute_tstep=20,  # Recompute the Jacobian matrix over time steps
            lmbda=0.5,  # Damping parameter for the Newton solver
            # Pre-deform load ramps (P_final and v_max_final are taken from the main problem file)
            t_start_v=0.0,  # Start time for ramping up velocity
            t_end_v=0.2,  # End time for ramping up velocity
            t_start_p=0.2,  # Start time for ramping up pressure
            t_end_p=0.9,  # End time for ramping up pressure (should be earlier than simulation end time)
            # mesh lifting parameters (see turtleFSI for options)
            extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
            extrapolation_sub_type="constant",  # ["constant","small_constant","volume","volume_change","bc1","bc2"]
            save_deg=1,  # NOTE: save_deg=1 is required for predeform simulations
        )
    )

    # Parameters of the main simulation
    args = namespace.get("args")
    main_problem = getattr(args, "main_problem", None) or default_variables.get("main_problem")
    if main_problem is None:
        raise ValueError("The predeform needs the problem file of the main simulation, e.g.\n"
                         "    turtleFSI -p predeform --main_problem=aneurysm\n"
                         "(with '=' and an underscore: --main_problem=<name of the problem file or path to it>)")
    default_variables.update(_parameters_from_main_problem(main_problem, turtle_defaults, args))

    return default_variables


def _load_main_problem(main_problem):
    """
    Import the problem file of the main simulation, given as a path to a .py file, or as a module name looked
    up in the current directory (as turtleFSI does), then in vasp.simulations and turtleFSI.problems.
    """
    name = str(main_problem)
    path = _Path(name)

    if path.stem == "predeform":
        raise ValueError("--main_problem must be the problem file of the main simulation, not the predeform itself.")

    if path.suffix == ".py":
        if not path.is_file():
            raise FileNotFoundError(f"Main problem file '{path}' not found.")
        # Allow the main problem file to import modules next to it
        if str(path.resolve().parent) not in _sys.path:
            _sys.path.insert(0, str(path.resolve().parent))
        spec = _importlib_util.spec_from_file_location(path.stem, str(path))
        module = _importlib_util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    candidates = [name] if "." in name else [name, "vasp.simulations." + name, "turtleFSI.problems." + name]
    for candidate in candidates:
        try:
            return _importlib.import_module(candidate)
        except ModuleNotFoundError as error:
            # Try the next location only if the problem file itself was not found (not one of its imports)
            if error.name is None or not (candidate == error.name or candidate.startswith(error.name + ".")):
                raise
    raise ModuleNotFoundError(f"Main problem file '{name}' not found in the current directory, in vasp.simulations "
                              f"or in turtleFSI.problems. Give its name or the path to the .py file.")


def _solid_properties_from_scalars(dx_s_id, material_model, rho_s, mu_s, lambda_s):
    """
    Build solid_properties from the scalar parameters, in the same way as turtleFSI (modules/domain.py)
    when solid_properties is not given. Only used to check the parameters before the simulation starts.
    """
    if isinstance(dx_s_id, list):
        models = material_model if isinstance(material_model, list) else [material_model] * len(dx_s_id)
        return [{"dx_s_id": region, "material_model": models[i], "rho_s": rho_s[i], "mu_s": mu_s[i],
                 "lambda_s": lambda_s[i]} for i, region in enumerate(dx_s_id)]
    return [{"dx_s_id": dx_s_id, "material_model": material_model, "rho_s": rho_s, "mu_s": mu_s,
             "lambda_s": lambda_s}]


def _parameters_from_main_problem(main_problem, turtle_defaults, args):
    """
    Take the material, Robin, load and boundary parameters, and the mesh function, from the main problem file.
    All the problems found are reported together, and the simulation does not start.
    """
    module = _load_main_problem(main_problem)
    source = getattr(module, "__file__", None) or str(main_problem)

    # Run set_problem_parameters of the main problem on a copy of the turtleFSI defaults, where the inherited
    # parameters are marked as not set: what is still marked afterwards is not defined in the main problem file,
    # so that the turtleFSI default values are never used silently.
    probe = _copy.deepcopy(turtle_defaults)
    for key in _INHERITED_KEYS:
        probe[key] = _NOT_SET
    returned = module.set_problem_parameters(default_variables=probe)
    if isinstance(returned, dict) and returned is not probe:
        probe.update(returned)

    def get(key):
        """Value from the command line if given there, otherwise from the main problem file, or _NOT_SET."""
        value = getattr(args, key, None) if args is not None else None
        return probe.get(key, _NOT_SET) if value is None else value

    def is_set(key):
        return get(key) is not _NOT_SET

    inherited = {}
    errors = []

    # Solid: material_model is mandatory, and the parameters required by each model must be defined
    dx_s_id = get("dx_s_id") if is_set("dx_s_id") else turtle_defaults["dx_s_id"]
    solid_properties = get("solid_properties")
    if solid_properties is not _NOT_SET and len(as_list(solid_properties)) > 0:
        inherited["solid_properties"] = solid_properties
        errors += solid_properties_errors(solid_properties, dx_s_id)
    else:
        solid_errors = []
        if not is_set("material_model"):
            solid_errors.append('material_model is not defined (it is mandatory), '
                                'e.g. material_model="StVenantKirchoff"')
        missing = [key for key in ("rho_s", "mu_s", "lambda_s") if not is_set(key)]
        if missing:
            solid_errors.append(f"solid parameters not defined: {', '.join(missing)} (or define solid_properties)")
        for model in as_list(get("material_model")) if is_set("material_model") else []:
            if model in ("MooneyRivlin", "Gent"):
                solid_errors.append(f"material_model '{model}' needs its own parameters, which can only be given "
                                    f"in solid_properties (one dictionary per solid region)")
        if not solid_errors:
            try:
                resolved = _solid_properties_from_scalars(dx_s_id, get("material_model"), get("rho_s"),
                                                          get("mu_s"), get("lambda_s"))
                solid_errors += solid_properties_errors(resolved, dx_s_id)
            except (IndexError, TypeError, KeyError):
                solid_errors.append(f"with dx_s_id={dx_s_id}, rho_s, mu_s and lambda_s must have one value per "
                                    f"solid region")
        errors += solid_errors
        inherited["solid_properties"] = []
        for key in ("material_model", "rho_s", "mu_s", "lambda_s"):
            if is_set(key):
                inherited[key] = get(key)
    for key in ("nu_s", "dx_s_id"):
        if is_set(key):
            inherited[key] = get(key)

    # Fluid: density and viscosity must be defined (the turtleFSI default mu_f is 1 Pa.s)
    fluid_properties = get("fluid_properties")
    if fluid_properties is not _NOT_SET and len(as_list(fluid_properties)) > 0:
        inherited["fluid_properties"] = fluid_properties
        for i, region_properties in enumerate(as_list(fluid_properties)):
            missing = [key for key in ("rho_f", "mu_f") if key not in region_properties]
            if missing:
                errors.append(f"fluid_properties[{i}]: missing {', '.join(missing)}")
    else:
        missing = [key for key in ("rho_f", "mu_f") if not is_set(key)]
        if missing:
            errors.append(f"fluid parameters not defined: {', '.join(missing)} (or define fluid_properties)")
        inherited["fluid_properties"] = []
        for key in ("rho_f", "mu_f"):
            if is_set(key):
                inherited[key] = get(key)
    for key in ("dx_f_id", "nonnewtonian", "cross_params"):
        if is_set(key):
            inherited[key] = get(key)

    # Robin boundary condition: optional, but if it is used its boundaries and coefficients must be defined
    robin_bc = get("robin_bc") if is_set("robin_bc") else False
    inherited["robin_bc"] = robin_bc
    if robin_bc:
        robin = {key: (get(key) if is_set(key) else None) for key in ("k_s", "c_s", "ds_s_id")}
        errors += robin_parameters_errors(robin_bc, robin["k_s"], robin["c_s"], robin["ds_s_id"])
        inherited.update({key: value for key, value in robin.items() if value is not None})

    # Predeform load: mandatory
    descriptions = {"P_final": "cycle-averaged pressure applied to the wall [Pa]",
                    "v_max_final": "centerline velocity at the inlet [m/s], one value or one per inlet_id"}
    for key in _LOAD_KEYS:
        if is_set(key):
            inherited[key] = get(key)
        else:
            errors.append(f"{key} is not defined (mandatory for the predeform: {descriptions[key]})")

    # Boundary IDs used by the predeform boundary conditions
    for key in ("inlet_id", "fsi_id"):
        if is_set(key):
            inherited[key] = get(key)
        else:
            errors.append(f"{key} is not defined (needed by the predeform boundary conditions; it can also be "
                          f"given on the command line, e.g. --{key}=[3,2])")
    for key in ("rigid_id", "inlet_outlet_s_id"):
        inherited[key] = get(key) if is_set(key) else None
    if is_set("inlet_id") and isinstance(get("v_max_final"), (list, tuple)) \
            and len(get("v_max_final")) != len(as_list(get("inlet_id"))):
        errors.append(f"v_max_final has {len(get('v_max_final'))} values but inlet_id={get('inlet_id')} has "
                      f"{len(as_list(get('inlet_id')))} inlet(s)")

    # Mesh, domains and boundaries: use the function of the main problem file if it defines one,
    # together with the parameters it takes
    get_mesh = getattr(module, "get_mesh_domain_and_boundaries", None)
    if get_mesh is None or get_mesh is _turtle_problems.get_mesh_domain_and_boundaries:
        _main_problem["get_mesh_domain_and_boundaries"] = None
        if is_set("mesh_path"):
            inherited["mesh_path"] = get("mesh_path")
        else:
            errors.append("the main problem file defines neither get_mesh_domain_and_boundaries nor mesh_path")
    else:
        _main_problem["get_mesh_domain_and_boundaries"] = get_mesh
        for parameter in _inspect.signature(get_mesh).parameters.values():
            key = parameter.name
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD) \
                    or key in _PREDEFORM_KEYS or key in inherited:
                continue
            if is_set(key):
                inherited[key] = get(key)
            elif parameter.default is parameter.empty:
                errors.append(f"get_mesh_domain_and_boundaries of the main problem file needs '{key}', "
                              f"which is not defined")

    if errors:
        raise ValueError(f"Predeform: problems with the parameters of the main problem file {source}:\n  - "
                         + "\n  - ".join(errors))

    # Store the results in the folder of the main simulation
    folder = probe.get("folder", turtle_defaults["folder"])
    inherited["folder"] = folder
    inherited["sub_folder"] = _PREDEFORM_SUB_FOLDER

    if MPI.rank(MPI.comm_world) == 0:
        mesh_source = ("main problem file" if _main_problem["get_mesh_domain_and_boundaries"]
                       else "predeform (mesh_path)")
        taken = [key for key, value in inherited.items()
                 if key not in ("folder", "sub_folder") and value is not None and value != []]
        # --folder and --sub-folder given on the command line override the values above
        results_folder = _Path(str(getattr(args, "folder", None) or folder)) / \
            str(getattr(args, "sub_folder", None) or _PREDEFORM_SUB_FOLDER)
        print(f"Predeform: parameters taken from the main problem file {source}:\n"
              f"    {', '.join(taken)}\n"
              f"    mesh, domains and boundaries read by: {mesh_source}\n"
              f"    results stored in: {results_folder}", flush=True)

    return inherited


def get_mesh_domain_and_boundaries(**namespace):
    # Use the function of the main problem file, so that the regions created there are the same
    get_mesh = _main_problem.get("get_mesh_domain_and_boundaries")
    if get_mesh is not None:
        return get_mesh(**namespace)

    # Otherwise, only read the mesh, domains and boundaries
    mesh_path = namespace["mesh_path"]
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    return mesh, domains, boundaries


class VelInPara(UserExpression):
    def __init__(self, t, t_start, t_end, v_max_final, n, dsi, mesh, **kwargs):
        self.t = t
        self.t_start = t_start
        self.t_end = t_end
        self.v_max_final = v_max_final
        self.v = 0.0
        self.n = n
        self.dsi = dsi
        self.d = mesh.geometry().dim()
        self.x = SpatialCoordinate(mesh)
        # Compute area of boundary tesselation by integrating 1.0 over all facets
        self.A = assemble(Constant(1.0, name="one") * self.dsi)
        # Compute barycenter by integrating x components over all facets
        self.c = [assemble(self.x[i] * self.dsi) / self.A for i in range(self.d)]
        # Compute radius by taking max radius of boundary points
        self.r = np.sqrt(self.A / np.pi)
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        # apply a sigmoid ramp to the pressure
        if self.t < self.t_start:
            ramp_factor = 0.0
        elif self.t < self.t_end and self.t > self.t_start:
            ramp_factor = -0.5 * np.cos(np.pi * (self.t - self.t_start) / (self.t_end - self.t_start)) + 0.5
        else:
            ramp_factor = 1.0
        self.v = ramp_factor * self.v_max_final

        if MPI.rank(MPI.comm_world) == 0:
            print("v (centerline, at inlet) = {} m/s".format(self.v))

    def eval(self, value, x):
        r2 = (
            (x[0] - self.c[0]) ** 2 + (x[1] - self.c[1]) ** 2 + (x[2] - self.c[2]) ** 2
        )  # radius**2
        fact_r = 1 - (r2 / self.r**2)

        value[0] = -self.n[0] * (self.v) * fact_r  # *self.t
        value[1] = -self.n[1] * (self.v) * fact_r  # *self.t
        value[2] = -self.n[2] * (self.v) * fact_r  # *self.t

    def value_shape(self):
        return (3,)


class InnerP(UserExpression):
    def __init__(self, t, t_start, t_end, P_final, **kwargs):
        self.t = t
        self.t_start = t_start
        self.t_end = t_end
        self.P_final = P_final
        self.P = 0.0
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        # apply a sigmoid ramp to the pressure
        if self.t < self.t_start:
            ramp_factor = 0.0
        elif self.t < self.t_end and self.t > self.t_start:
            ramp_factor = -0.5 * np.cos(np.pi * (self.t - self.t_start) / (self.t_end - self.t_start)) + 0.5
        else:
            ramp_factor = 1.0
        self.P = ramp_factor * self.P_final

        if MPI.rank(MPI.comm_world) == 0:
            print("P = {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P

    def value_shape(self):
        return ()


def initiate(**namespace):
    # Check the material and Robin parameters (the material models are printed by turtleFSI)
    check_material_parameters(**namespace)
    return {}


def create_bcs(DVP, mesh, boundaries, t_start_v, t_end_v, t_start_p, t_end_p, P_final,
               v_max_final, fsi_id, inlet_id, inlet_outlet_s_id, rigid_id, psi, F_solid_linear, **namespace):
    # The boundary IDs can be given as int or list (e.g. several inlets or FSI interfaces)
    inlet_ids = as_list(inlet_id)
    v_max_values = as_list(v_max_final)
    if len(v_max_values) == 1:
        v_max_values = v_max_values * len(inlet_ids)
    if len(v_max_values) != len(inlet_ids):
        raise ValueError(f"v_max_final={v_max_final} must be one value, or one value per inlet in inlet_id={inlet_id}")
    zero = tuple([0.0] * mesh.geometry().dim())

    # Apply pressure at the fsi interface by modifying the variational form
    p_out_bc_val = InnerP(t=0.0, t_start=t_start_p, t_end=t_end_p, P_final=P_final, degree=2)
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    # defined on the reference domain
    # NOTE: ('+') implicitly assumes that the solid domain has a higher domain ID than the fluid domain
    for interface_id in as_list(fsi_id):
        F_solid_linear += p_out_bc_val * inner(n("+"), psi("+")) * dSS(interface_id)

    # Fluid velocity BCs: parabolic inlet velocity profile at each inlet, and fixed inlet displacement
    bcs = []
    u_inflow_exp = []
    inlet_dsi = []
    inlet_areas = []
    ndim = mesh.geometry().dim()
    for inlet, v_max in zip(inlet_ids, v_max_values):
        dsi = ds(inlet, domain=mesh, subdomain_data=boundaries)
        ni = np.array([assemble(n[i] * dsi) for i in range(ndim)])
        n_len = np.sqrt(sum([ni[i] ** 2 for i in range(ndim)]))  # Should always be 1!?
        normal = ni / n_len

        # compute inlet area
        inlet_area = assemble(1 * dsi)
        if MPI.rank(MPI.comm_world) == 0:
            print("Inlet area = ", inlet_area)

        expression = VelInPara(t=0.0, t_start=t_start_v, t_end=t_end_v, v_max_final=v_max,
                               n=normal, dsi=dsi, mesh=mesh, degree=3)
        bcs.append(DirichletBC(DVP.sub(1), expression, boundaries, inlet))
        bcs.append(DirichletBC(DVP.sub(0), zero, boundaries, inlet))
        u_inflow_exp.append(expression)
        inlet_dsi.append(dsi)
        inlet_areas.append(inlet_area)

    # Fixed solid boundaries (zero velocity and displacement)
    for solid_id in as_list(inlet_outlet_s_id):
        bcs.append(DirichletBC(DVP.sub(1), zero, boundaries, solid_id))
        bcs.append(DirichletBC(DVP.sub(0), zero, boundaries, solid_id))

    # Rigid walls (zero displacement)
    for wall_id in as_list(rigid_id):
        bcs.append(DirichletBC(DVP.sub(0), zero, boundaries, wall_id))

    # The flow properties are printed at the first inlet
    return dict(bcs=bcs, u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val,
                F_solid_linear=F_solid_linear, inlet_area=inlet_areas[0], n=n, dsi=inlet_dsi[0])


def pre_solve(t, u_inflow_exp, p_out_bc_val, mu_eff=None, **namespace):
    # Update the time variable used for the inlet boundary condition
    for expression in u_inflow_exp:
        expression.update(t)
    p_out_bc_val.update(t)

    # Non-Newtonian fluid: update the effective viscosity once per time step
    if mu_eff is not None:
        mu_eff.update()

    return dict(u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val)


def post_solve(dvp_, n, dsi, dt, mesh, inlet_area, fluid_properties, **namespace):

    v = dvp_["n"].sub(1, deepcopy=True)
    # Reference density and viscosity of the first fluid region
    mu_f = fluid_properties[0]["mu_f"]
    rho_f = fluid_properties[0]["rho_f"]
    calculate_and_print_flow_properties(dt, mesh, v, inlet_area, mu_f, rho_f, n, dsi)
