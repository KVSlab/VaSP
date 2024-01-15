"""
Problem file for offset stenosis FSI simulation
"""
import os
import numpy as np

from vampy.simulation.Womersley import make_womersley_bcs, compute_boundary_geometry_acrn
from turtleFSI.problems import *
from dolfin import HDF5File, Mesh, MeshFunction, facets, cells, UserExpression, FacetNormal, ds, \
    DirichletBC, Measure, inner, parameters, assemble

from fsipy.simulations.simulation_common import load_probe_points, print_probe_points, print_mesh_summary, \
    calculate_and_print_flow_properties

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):

    # Compute some solid parameters
    # Need to stay here since mus_s and lambda_s are functions of nu_s and E_s
    E_s_val = 1E6
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    default_variables.update(dict(
        # Temporal parameters
        T=0.951,  # Simulation end time
        dt=0.001,  # Timne step size
        theta=0.501,  # Theta scheme parameter
        save_step=1,  # Save frequency of files for visualisation
        checkpoint_step=50,  # Save frequency of checkpoint files
        # Linear solver parameters
        linear_solver="mumps",
        atol=1e-6,  # Absolute tolerance in the Newton solver
        rtol=1e-6,  # Relative tolerance in the Newton solver
        recompute=20,  # Recompute the Jacobian matix within time steps
        recompute_tstep=20,  # Recompute the Jacobian matix over time steps
        # boundary condition parameters
        inlet_id=3,  # inlet id for the fluid
        inlet_outlet_s_id=11,  # inlet and outlet id for solid
        fsi_id=22,  # id for fsi surface
        rigid_id=11,  # "rigid wall" id for the fluid
        outer_id=33,  # id for the outer surface of the solid
        # Fluid parameters
        Q_mean=2.5E-06,
        P_mean=11200,
        T_Cycle=0.951,  # Used to define length of flow waveform
        rho_f=[1.000E3, 1.000E3],  # Fluid density [kg/m3]
        mu_f=[1.5E-3, 1.0E-2],  # Fluid dynamic viscosity [Pa.s]
        dx_f_id=[1, 1001],  # ID of marker in the fluid domain
        # mesh lifting parameters (see turtleFSI for options)
        extrapolation="laplace",
        extrapolation_sub_type="constant",
        # Solid parameters
        rho_s=1.0E3,  # Solid density [kg/m3]
        mu_s=mu_s_val,  # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,  # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid Young's modulus [Pa]
        dx_s_id=2,  # ID of marker in the solid domain
        # FSI parameters
        fsi_x_range=[-0.0002, 0.016],  # x-range of FSI region
        fsi_y_range=[-0.0035, 0.0035],  # y-range of FSI region
        fsi_z_range=[-0.0035, 0.0035],  # z-range of FSI region
        # Simulation parameters
        folder="offset_stenosis_results_box",  # Folder name generated for the simulation
        mesh_path="mesh/file_stenosis.h5",
        FC_file="FC_MCA_10",  # File name containing the Fourier coefficients for the flow waveform
        P_FC_File="FC_Pressure",  # File name containing the Fourier coefficients for the pressure waveform
        compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)
        save_deg=2,  # Degree of the functions saved for visualisation
    ))

    return default_variables


def get_mesh_domain_and_boundaries(mesh_path, fsi_x_range, fsi_y_range, fsi_z_range, dx_f_id, fsi_id, rigid_id, outer_id, **namespace):

    # Read mesh
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    print_mesh_summary(mesh)

    # NOTE: Instead of using a sphere, we can also use a box to define the FSI region
    # Only consider FSI in domain within fsi_x_range, fsi_y_range, fsi_z_range
    i = 0
    for submesh_facet in facets(mesh):
        idx_facet = boundaries.array()[i]
        if idx_facet == fsi_id or idx_facet == outer_id:
            mid = submesh_facet.midpoint()
            if mid.x() < fsi_x_range[0] or mid.x() > fsi_x_range[1]:
                boundaries.array()[i] = rigid_id  # changed "fsi" id to "rigid wall" id
            elif mid.y() < fsi_y_range[0] or mid.y() > fsi_y_range[1]:
                boundaries.array()[i] = rigid_id
            elif mid.z() < fsi_z_range[0] or mid.z() > fsi_z_range[1]:
                boundaries.array()[i] = rigid_id
        
        i += 1
    
    # In this region, make fluid more viscous
    x_min = 0.024
    i = 0
    for cell in cells(mesh):
        idx_cell = domains.array()[i]
        if idx_cell == dx_f_id[0]:
            mid = cell.midpoint()
            if mid.x() > x_min:
                domains.array()[i] = dx_f_id[1]
        i += 1


    return mesh, domains, boundaries


class InnerP(UserExpression):
    def __init__(self, t, t_ramp, An, Bn, period, P_mean, **kwargs):
        self.t = t
        self.t_ramp = t_ramp
        self.An = An
        self.Bn = Bn
        self.omega = (2.0 * np.pi / period)
        self.P_mean = P_mean
        self.p_0 = 0.0  # Initial pressure
        self.P = self.p_0  # Apply initial pressure to inner pressure variable
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        # apply a sigmoid ramp to the pressure
        if self.t < self.t_ramp:
            ramp_factor = -0.5 * np.cos(np.pi * self.t / self.t_ramp) + 0.5
        else:
            ramp_factor = 1.0
        if MPI.rank(MPI.comm_world) == 0:
            print("ramp_factor = {} m^3/s".format(ramp_factor))

        # Caclulate Pn (normalized pressure)from Fourier Coefficients
        Pn = 0 + 0j
        for i in range(len(self.An)):
            Pn = Pn + (self.An[i] - self.Bn[i] * 1j) * np.exp(1j * i * self.omega * self.t)
        Pn = abs(Pn)

        # Multiply by mean pressure and ramp factor
        self.P = ramp_factor * Pn * self.P_mean
        if MPI.rank(MPI.comm_world) == 0:
            print("P = {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P

    def value_shape(self):
        return ()


def initiate(mesh_path, **namespace):

    probe_points = load_probe_points(mesh_path)

    return dict(probe_points=probe_points)


def create_bcs(t, DVP, mesh, boundaries, mu_f,
               fsi_id, inlet_id, inlet_outlet_s_id,
               rigid_id, psi, F_solid_linear, p_deg, FC_file,
               Q_mean, P_FC_File, P_mean, T_Cycle, **namespace):

    # Load Fourier coefficients for the velocity and scale by flow rate
    An, Bn = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), FC_file)).T
    # Convert to complex fourier coefficients
    Cn = (An - Bn * 1j) * Q_mean
    _, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, inlet_id, boundaries)

    # Create Womersley boundary condition at inlet
    tmp_element = DVP.sub(1).sub(0).ufl_element()
    inlet = make_womersley_bcs(T_Cycle, None, mu_f[0], tmp_center, tmp_radius, tmp_normal, tmp_element, Cn=Cn)
    # Initialize inlet expressions with initial time
    for uc in inlet:
        uc.set_t(t)

    # Create Boundary conditions for the velocity
    u_inlet = [DirichletBC(DVP.sub(1).sub(i), inlet[i], boundaries, inlet_id) for i in range(3)]
    u_inlet_s = DirichletBC(DVP.sub(1), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)

    # Solid Displacement BCs
    d_inlet = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, inlet_id)
    d_inlet_s = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, inlet_outlet_s_id)
    d_rigid = DirichletBC(DVP.sub(0), (0.0, 0.0, 0.0), boundaries, rigid_id)

    # Assemble boundary conditions
    bcs = u_inlet + [d_inlet, u_inlet_s, d_inlet_s, d_rigid]

    # Load Fourier coefficients for the pressure and scale by flow rate
    An_P, Bn_P = np.loadtxt(os.path.join(os.path.dirname(os.path.abspath(__file__)), P_FC_File)).T

    # Apply pulsatile pressure at the fsi interface by modifying the variational form
    n = FacetNormal(mesh)
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    p_out_bc_val = InnerP(t=0.0, t_ramp=0.2, An=An_P, Bn=Bn_P, period=T_Cycle, P_mean=P_mean, degree=p_deg)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+')) * dSS(fsi_id)

    # Create inlet subdomain for computing the flow rate inside post_solve
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    inlet_area = assemble(1.0 * dsi)
    return dict(bcs=bcs, inlet=inlet, p_out_bc_val=p_out_bc_val, F_solid_linear=F_solid_linear, n=n, dsi=dsi,
                inlet_area=inlet_area)


def pre_solve(t, inlet, p_out_bc_val, **namespace):
    for uc in inlet:
        # Update the time variable used for the inlet boundary condition
        uc.set_t(t)

        # Multiply by cosine function to ramp up smoothly over time interval 0-250 ms
        if t < 0.25:
            uc.scale_value = -0.5 * np.cos(np.pi * t / 0.25) + 0.5
        else:
            uc.scale_value = 1.0

    # Update pressure condition
    p_out_bc_val.update(t)

    return dict(inlet=inlet, p_out_bc_val=p_out_bc_val)


def post_solve(probe_points, dvp_, dt, mesh, inlet_area, dsi, mu_f, rho_f, n, **namespace):

    v = dvp_["n"].sub(1, deepcopy=True)
    p = dvp_["n"].sub(2, deepcopy=True)

    print_probe_points(v, p, probe_points)
    calculate_and_print_flow_properties(dt, mesh, v, inlet_area, mu_f[0], rho_f[0], n, dsi)
