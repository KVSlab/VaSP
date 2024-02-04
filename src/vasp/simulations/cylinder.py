"""
Problem file for tiny cylinder FSI simulation
"""
import numpy as np
from turtleFSI.problems import *
from dolfin import HDF5File, Mesh, MeshFunction, assemble, UserExpression, FacetNormal, ds, \
    DirichletBC, Measure, inner, parameters, SpatialCoordinate, Constant

from vasp.simulations.simulation_common import calculate_and_print_flow_properties

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val = 1e6  # Young modulus (Pa)
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2.0 * mu_s_val / (1.0 - 2.0 * nu_s_val)

    default_variables.update(
        dict(
            # Temporal parameters
            T=0.1,  # Simulation end time
            dt=0.001,  # Time step size
            theta=0.501,  # Theta scheme (implicit/explicit time stepping)
            save_step=1,  # Save frequency of files for visualisation
            checkpoint_step=50,  # Save frequency of checkpoint files
            # Linear solver parameters
            linear_solver="mumps",
            atol=1e-6,  # Absolute tolerance in the Newton solver
            rtol=1e-6,  # Relative tolerance in the Newton solver
            recompute=20,  # Recompute the Jacobian matix within time steps
            recompute_tstep=20,  # Recompute the Jacobian matix over time steps
            # boundary condition parameters
            mesh_path="mesh/cylinder.h5",
            inlet_id=2,  # inlet id for the fluid
            inlet_outlet_s_id=11,  # inlet and outlet id for solid
            fsi_id=22,  # id for fsi interface
            rigid_id=11,  # "rigid wall" id for the fluid and mesh problem
            outer_wall_id=33,  # id for the outer surface of the solid
            # Fluid parameters
            rho_f=1.025e3,  # Fluid density [kg/m3]
            mu_f=3.5e-3,  # Fluid dynamic viscosity [Pa.s]
            dx_f_id=1,  # ID of marker in the fluid domain. When reading the mesh, the fuid domain is assigned with a 1.
            v_max_final=0.75,  # Steady state, maximum centerline velocity of parabolic profile
            P_final=10000,  # Steady State pressure applied to wall
            # Solid parameters
            rho_s=1.0e3,  # Solid density [kg/m3]
            mu_s=mu_s_val,  # Solid shear modulus or 2nd Lame Coef. [Pa]
            nu_s=nu_s_val,  # Solid Poisson ratio [-]
            lambda_s=lambda_s_val,  # Solid 1st Lame Coef. [Pa]
            dx_s_id=2,  # ID of marker in the solid domain
            # mesh lifting parameters (see turtleFSI for options)
            extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
            extrapolation_sub_type="constant",  # ["constant","small_constant","volume","volume_change","bc1","bc2"]
            folder="cylinder_results",  # output folder generated for simulation
            save_deg=1  # save_deg=1 saves corner nodes only, save_deg=2 saves corner + mid-point nodes for viz
        )
    )

    return default_variables


def get_mesh_domain_and_boundaries(mesh_path, **namespace):
    if MPI.rank(MPI.comm_world) == 0:
        print("Obtaining mesh, domains and boundaries...")
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    return mesh, domains, boundaries


class VelInPara(UserExpression):
    def __init__(self, t, t_ramp, v_max_final, n, dsi, mesh, **kwargs):
        self.t = t
        self.t_ramp = t_ramp
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
        if self.t < self.t_ramp:
            ramp_factor = -0.5 * np.cos(np.pi * self.t / self.t_ramp) + 0.5
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
    def __init__(self, t, t_ramp, P_final, **kwargs):
        self.t = t
        self.t_ramp = t_ramp
        self.P_final = P_final
        self.P = 0.0
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        # apply a sigmoid ramp to the pressure
        if self.t < self.t_ramp:
            ramp_factor = -0.5 * np.cos(np.pi * self.t / self.t_ramp) + 0.5
        else:
            ramp_factor = 1.0
        self.P = ramp_factor * self.P_final

        if MPI.rank(MPI.comm_world) == 0:
            print("P = {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P

    def value_shape(self):
        return ()


def create_bcs(DVP, mesh, boundaries, P_final, v_max_final, fsi_id, inlet_id,
               inlet_outlet_s_id, rigid_id, psi, F_solid_linear, **namespace):

    # Apply pressure at the fsi interface by modifying the variational form
    p_out_bc_val = InnerP(t=0.0, t_ramp=0.1, P_final=P_final, degree=2)
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    # defined on the reference domain
    F_solid_linear += (p_out_bc_val * inner(n("+"), psi("+")) * dSS(fsi_id))

    # Fluid velocity BCs
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni = np.array([assemble(n[i] * dsi) for i in range(ndim)])
    n_len = np.sqrt(sum([ni[i] ** 2 for i in range(ndim)]))  # Should always be 1!?
    normal = ni / n_len

    # Parabolic Inlet Velocity Profile
    u_inflow_exp = VelInPara(t=0.0, t_ramp=0.1, v_max_final=v_max_final,
                             n=normal, dsi=dsi, mesh=mesh, degree=3)
    u_inlet = DirichletBC(DVP.sub(1), u_inflow_exp, boundaries, inlet_id)
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)

    # Solid Displacement BCs
    d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_id)
    d_inlet_s = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)
    d_rigid = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, rigid_id)

    # Assemble boundary conditions
    bcs = [u_inlet, d_inlet, u_inlet_s, d_inlet_s, d_rigid]

    # Create inlet subdomain for computing the flow rate inside post_solve
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    inlet_area = assemble(1.0 * dsi)

    return dict(bcs=bcs, u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val,
                F_solid_linear=F_solid_linear, dsi=dsi, inlet_area=inlet_area, n=n)


def pre_solve(t, u_inflow_exp, p_out_bc_val, **namespace):
    # Update the time variable used for the inlet boundary condition
    u_inflow_exp.update(t)
    p_out_bc_val.update(t)
    return dict(u_inflow_exp=u_inflow_exp, p_out_bc_val=p_out_bc_val)


def post_solve(dvp_, dt, mesh, inlet_area, mu_f, rho_f, n, dsi, **namespace):

    v = dvp_["n"].sub(1, deepcopy=True)
    calculate_and_print_flow_properties(dt, mesh, v, inlet_area, mu_f, rho_f, n, dsi)
