from dolfin import *
import os, math
from turtleFSI.problems import *
from utils.Womersley import make_womersley_bcs, compute_boundary_geometry_acrn
import numpy as np
from numpy import genfromtxt
from os import path, makedirs, getcwd
from pprint import pprint
import configparser
# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):

    # Find and open config file in current directory. Should only have 1 config file in directory
    config_file = [_ for _ in os.listdir(os.getcwd()) if _.endswith(".config")][0]
    config = configparser.ConfigParser()
    with open(config_file) as stream:
        config.read_string("[Case_Specific_Variables]\n" + stream.read()) 

    # Read problem-specific variables from config file
    x_sphere = float(config.get("Case_Specific_Variables", "x_sphere"))
    y_sphere = float(config.get("Case_Specific_Variables", "y_sphere"))
    z_sphere = float(config.get("Case_Specific_Variables", "z_sphere"))
    r_sphere = float(config.get("Case_Specific_Variables", "r_sphere"))
    dt = float(config.get("Case_Specific_Variables", "dt"))
    mesh_path = config.get("Case_Specific_Variables", "mesh_path").strip("\"").strip("\'")
    save_deg_sim = int(config.get("Case_Specific_Variables", "save_deg_sim"))
    Q_mean = float(config.get("Case_Specific_Variables", "Q_mean"))
    T = float(config.get("Case_Specific_Variables", "T"))

    # Overwrite default values
    E_s_val = 1E6
    nu_s_val = 0.45
    mu_s_val = E_s_val/(2*(1+nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val*2.*mu_s_val/(1. - 2.*nu_s_val)

    default_variables.update(dict(
        T=T, # Simulation end time
        dt=dt,#0.00033964286, # Timne step size
        atol=1e-5, # Absolute tolerance in the Newton solver
        rtol=1e-5,# Relative tolerance in the Newton solver
        inlet_id=2,  # inlet
        inlet_outlet_s_id=11,  # also the "rigid wall" id for the stucture problem
        recompute=30, # Recompute the Jacobian matix within time steps                                                                                    
        recompute_tstep=5, # Recompute the Jacobian matix over time steps (dangerous!)  
        fsi_id=22,  # fsi surface
        rigid_id=11,  # "rigid wall" id for the fluid and mesh problem
        outer_id=33,  # outer surface
        folder=mesh_path,#"file_case9_el047",
        mesh_file=mesh_path,#"file_case9_el047",
        Q_file="MCA_10", # This is the location of CFD results used to prescribe the inlet velocity profile
        Q_mean=Q_mean,#1.9275E-06, # Problem specific
        theta=0.50+dt, # Theta scheme (implicit/explicit time stepping)
        rho_f=1.000E3,    # Fluid density [kg/m3]
        mu_f=3.5E-3,       # Fluid dynamic viscosity [Pa.s]
        rho_s=1.0E3,    # Solid density [kg/m3]
        mu_s=mu_s_val,     # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,      # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid Young's modulus [Pa]
        dx_f_id=1,      # ID of marker in the fluid domain
        dx_s_id=2,      # ID of marker in the solid domain
        extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
        extrapolation_sub_type="constant",  # ["constant", "small_constant", "volume", "volume_change", "bc1", "bc2"]
        compiler_parameters=_compiler_parameters,  # Update the defaul values of the compiler arguments (FEniCS)
        linear_solver="mumps",  # use list_linear_solvers() to check alternatives
        checkpoint_step=5, # CHANGE
        save_step=1, # Save frequency of files for visualisation
        save_deg=save_deg_sim,          # Degree of the functions saved for visualisation '1' '2' '3' etc... (high value can slow down simulation significantly!)
        fsi_region=[x_sphere,y_sphere,z_sphere,r_sphere], # X, Y, and Z coordinate of FSI region center, radius of spherical deformable region (outside this region the walls are rigid)
        killtime=80000 # 40 mins # CHANGE 23:25 in seconds, after this time start dumping checkpoints every timestep
    ))

    return default_variables

def get_mesh_domain_and_boundaries(mesh_file,fsi_region, fsi_id, rigid_id, outer_id, folder, **namespace):
    # Read mesh
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "mesh/" + mesh_file + ".h5", "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    # Only considere FSI in domain within this sphere BC1
    sph_x = fsi_region[0]
    sph_y = fsi_region[1]
    sph_z = fsi_region[2]
    sph_rad = fsi_region[3]

    i = 0
    for submesh_facet in facets(mesh):
        idx_facet = boundaries.array()[i]
        if idx_facet == fsi_id or idx_facet == outer_id:
            vert = submesh_facet.entities(0)
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x()-sph_x)**2 + (mid.y()-sph_y)**2 + (mid.z()-sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id  # changed "fsi" idx to "rigid wall" idx
        i += 1

    # # Checking boundaries and domains
    # f = File('toto.pvd')
    # f << boundaries
    # f << domains

    return mesh, domains, boundaries



class InnerP(UserExpression):
    def __init__(self, t, t_ramp, t_p, p_PA, **kwargs):
        self.t = t
        self.t_ramp = t_ramp
        self.t_p = t_p
        self.p_PA = p_PA    
        self.p_0 = 0.0 # Initial pressure
        self.P = self.p_0 # Apply initial pressure to inner pressure variable
        super().__init__(**kwargs)

    def update(self, t):
        self.t = t
        # apply a sigmoid ramp to the pressure 
        if self.t < self.t_ramp:
            ramp_factor = (-1/2)*np.cos(3.14159*self.t/self.t_ramp) + 1/2
        else:
            ramp_factor = 1.0
        if MPI.rank(MPI.comm_world) == 0:
            print("ramp_factor = {} m^3/s".format(ramp_factor))
        # Caclulate P as resistance boundary condition multiplied by ramp factor
        self.P = ramp_factor * (np.interp(self.t, self.t_p, self.p_PA))
        if MPI.rank(MPI.comm_world) == 0:
            print("P = {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P 

    def value_shape(self):
        return ()


def create_bcs(t, v_, DVP, mesh, boundaries, domains, mu_f,
               fsi_id, inlet_id, inlet_outlet_s_id,
               rigid_id, psi, F_solid_linear, p_deg, Q_file, Q_mean, **namespace):

    # Fluid velocity BCs
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni = np.array([assemble(n[i]*dsi) for i in range(ndim)])
    n_len = np.sqrt(sum([ni[i]**2 for i in range(ndim)]))  # Should always be 1!?
    normal = ni/n_len

    # Load normalized time and flow rate values
    t_values, Q_ = np.loadtxt(path.join(path.dirname(path.abspath(__file__)), Q_file)).T
    Q_values = Q_mean * Q_  # Specific flow rate = Normalized flow wave form * Prescribed flow rate
    tmp_area, tmp_center, tmp_radius, tmp_normal = compute_boundary_geometry_acrn(mesh, inlet_id, boundaries)

    # Create Womersley boundary condition at inlet
    inlet = make_womersley_bcs(t_values, Q_values, mesh, mu_f, tmp_area, tmp_center, tmp_radius, tmp_normal, DVP.sub(1).sub(0).ufl_element())

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

    # Define the pressure condition (apply to inner surface, numerical instability results from applying to outlet)
    #dso = ds(outlet_id1, domain=mesh, subdomain_data=boundaries) # Outlet surface # Maybe try applying to all outlets???
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
    p_t_file = genfromtxt('p_t.csv', delimiter=',')
    t_pressure=p_t_file[1:,0]
    pressure_PA=p_t_file[1:,1]
    p_out_bc_val = InnerP(t=0.0, t_p=t_pressure, t_ramp=0.2, p_PA=pressure_PA, degree=p_deg)
    F_solid_linear += p_out_bc_val * inner(n('+'), psi('+'))*dSS(fsi_id)  # defined on the reference domain

    return dict(bcs=bcs, inlet=inlet, p_out_bc_val=p_out_bc_val, F_solid_linear=F_solid_linear)


def pre_solve(t, v_, DVP, inlet, p_out_bc_val, **namespace):
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

def finished(**namespace):
    with open("finished", mode='a'): pass



