from dolfin import *
import os, math
from turtleFSI.problems import *
from turtleFSI.utils import make_womersley_bcs, compute_boundary_geometry_acrn
import numpy as np
from os import path, makedirs, getcwd
#from fenicstools import Probes
from Probe import Probes
from pprint import pprint

# BM will look at the IDs. Oringinal drawing :) Discuss drawing as well...
# Thangam will clean up terminology, innerP, prestress and that everything works. Make sure that the code prints logically and chronologically. BCs in sep file@
# David B, add stress code and xdmf for viz

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen.
parameters["form_compiler"]["optimize"] = True
# The "ghost_mode" has to do with the assembly of form containing the facet
# normals n('+') within interior boundaries (dS). for 3D mesh the value should
# be "shared_vertex", for 2D mesh "shared_facet", the default value is "none"
parameters["ghost_mode"] = "shared_vertex"
_compiler_parameters = dict(parameters["form_compiler"])


def set_problem_parameters(default_variables, **namespace):
    # Overwrite default values
    E_s_val = 1E6  # Young modulus (Pa)
    nu_s_val = 0.45
    mu_s_val = E_s_val / (2 * (1 + nu_s_val))  # 0.345E6
    lambda_s_val = nu_s_val * 2. * mu_s_val / (1. - 2. * nu_s_val)

    default_variables.update(dict(
        T=0.2, # Simulation end time
        dt=0.001, # Timne step size
        theta=0.501, # Theta scheme (implicit/explicit time stepping)
        atol=1e-5, # Absolute tolerance in the Newton solver
        rtol=1e-5,# Relative tolerance in the Newton solver
        #Q_mean=5.045e-6, # Problem specific (FIX) ICA
        Q_mean=2.5e-6, # Problem specific (FIX) MCA
        mesh_file="artery_coarse_rescaled", # duh
        inlet_id=2,  # inlet id
        outlet_id1=3,  # outlet nb
        inlet_outlet_s_id=11,  # also the "rigid wall" id for the stucture problem. MB: Clarify :)
        fsi_id=22,  # fsi Interface
        rigid_id=11,  # "rigid wall" id for the fluid and mesh problem
        outer_wall_id=33,  # outer surface / external id
        rho_f=1.025E3,    # Fluid density [kg/m3]
        mu_f=3.5E-3,       # Fluid dynamic viscosity [Pa.s]
        rho_s=1.0E3,    # Solid density [kg/m3]
        mu_s=mu_s_val,     # Solid shear modulus or 2nd Lame Coef. [Pa]
        nu_s=nu_s_val,      # Solid Poisson ratio [-]
        lambda_s=lambda_s_val,  # Solid 1rst Lamé coef. [Pa]
#        dx_f_id=1,      # ID of marker in the fluid domain. When reading the mesh, the fuid domain is assigned with a 1. Old crap :)
#        dx_s_id=2,      # ID of marker in the solid domain
        extrapolation="laplace",  # laplace, elastic, biharmonic, no-extrapolation
        # ["constant", "small_constant", "volume", "volume_change", "bc1", "bc2"]
        extrapolation_sub_type="constant",
        recompute=3000, # Number of iterations before recompute Jacobian.
        recompute_tstep=1, # Number of time steps before recompute Jacobian.
        save_step=1, # Save frequency of files for visualisation
        folder="TNT_parabolicPulse_scaledCyl_savedeg2",
        checkpoint_step=50, # checkpoint frequency
        kill_time=100000, # in seconds, after this time start dumping checkpoints every timestep
        save_deg=2, # Default could be 1. 1 saves the nodal values only while 2 takes full advantage of the mide side nodes available in the P2 solution. P2 f0r nice visualisations :)
        dump_probe_frequency=100,  # Dump frequency for sampling velocity & pressure at probes along the centerline
        probe_points=[[0.000391, -0.001916, 0.000150], # List of probe location within the domain where one wants to dump time varying values of DVP
                      [0.001286, -0.000529, -0.001027],
                      [-0.000170, 0.001041, -0.000218],
                      [8.373301e-05, 0.002781, 0.0001315]],
    ))

    if MPI.rank(MPI.comm_world) == 0:
        info_red("=== Starting simulation ===")
        info_blue("Running with the following parameters:")
        pprint(default_variables)

    return default_variables


def get_mesh_domain_and_boundaries(mesh_file, fsi_id, rigid_id, outer_wall_id, folder, **namespace):
    info_red("Obtaining mesh, domains and boundaries...")
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), "mesh/" + mesh_file + ".h5", "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    # Only consider FSI in domain within this sphere
    sph_x = 1.8124701455235481e-06
    sph_y = -6.299931555986404e-06
    sph_z = -7.300404831767082e-07
    sph_rad = 0.0034
    info_green("The coordinates of the FSI sphere are [{}, {}, {}] of radius: {}".format(sph_x, sph_y, sph_z, sph_rad))

    i = 0
    for submesh_facet in facets(mesh):
        idx_facet = boundaries.array()[i]
        if idx_facet == fsi_id:
            vert = submesh_facet.entities(0)
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x() - sph_x)**2 + (mid.y() - sph_y)**2 + (mid.z() - sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id
        if idx_facet == outer_wall_id:
            vert = submesh_facet.entities(0)
            mid = submesh_facet.midpoint()
            dist_sph_center = sqrt((mid.x() - sph_x)**2 + (mid.y() - sph_y)**2 + (mid.z() - sph_z)**2)
            if dist_sph_center > sph_rad:
                boundaries.array()[i] = rigid_id
        i += 1

    print_mesh_information(mesh)

    return mesh, domains, boundaries


def initiate(mesh, DVP, probe_points, results_folder, **namespace):
    # Function space
    DG = FunctionSpace(mesh, 'DG', 0)

    # Mesh info
    h = CellDiameter(mesh)
    characteristic_edge_length = project(h, DG)

    # CFL
    CFL = Function(DG)

    # Convert probe_points to numpy array if given as a list
    if isinstance(probe_points, list):
        probe_points = np.array(probe_points)

    # Store points file in checkpoint
    if MPI.rank(MPI.comm_world) == 0:
        probe_points.dump(path.join(results_folder, "Checkpoint", "points"))

    # Create dict for evaluation of probe points
    eval_dict = {}
    eval_dict["centerline_u_x_probes"] = Probes(probe_points.flatten(), DVP.sub(1).sub(0))
    eval_dict["centerline_u_y_probes"] = Probes(probe_points.flatten(), DVP.sub(1).sub(1))
    eval_dict["centerline_u_z_probes"] = Probes(probe_points.flatten(), DVP.sub(1).sub(2))
    eval_dict["centerline_p_probes"] = Probes(probe_points.flatten(), DVP.sub(2))

    return dict(DG=DG, CFL=CFL, characteristic_edge_length=characteristic_edge_length, eval_dict=eval_dict, probe_points=probe_points)


class InnerP(UserExpression):
    def __init__(self, t, n, u, dsi, resistance, p_0, **kwargs):
        self.t = t
        self.n = n # Mesh Normal
        self.u = u
        self.p_0 = p_0 # Initial pressure
        self.P = p_0 # Apply initial pressure to inner pressure variable
        self.dsi = dsi
        self.resistance = resistance
        super().__init__(**kwargs)

    def update(self, t, u):
        self.t = t
        self.u = u

        # Caclulate flow rate
        Q = np.abs(assemble(inner(self.u, self.n) * self.dsi))
        info_green("Instantaneous flow rate Q = {} m^3/s".format(Q))

        # Caclulate P as resistance boundary condition
        self.P = self.p_0 + self.resistance * Q
        info_green("Instantaneous normal stress prescribed at the FSI interface {} Pa".format(self.P))

    def eval(self, value, x):
        value[0] = self.P

    def value_shape(self):
        return ()


def create_bcs(t, v_, DVP, mesh, boundaries, domains, mu_f,
               fsi_id, outlet_id1, inlet_id, inlet_outlet_s_id,
               rigid_id, psi, F_solid_linear, p_deg, Q_mean, **namespace):
    info_red("Creating boundary conditions")

    # Fluid velocity BCs
    dsi = ds(inlet_id, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)
    ndim = mesh.geometry().dim()
    ni = np.array([assemble(n[i]*dsi) for i in range(ndim)])
    n_len = np.sqrt(sum([ni[i]**2 for i in range(ndim)]))  # Should always be 1!?
    normal = ni/n_len

    # Load normalized time and flow rate values
    t_values, Q_ = np.loadtxt(path.join(path.dirname(path.abspath(__file__)), "ICA_values")).T
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
    d_inlet = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_id)
    d_inlet_s = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, inlet_outlet_s_id)
    d_rigid = DirichletBC(DVP.sub(0), ((0.0, 0.0, 0.0)), boundaries, rigid_id)

    # Assemble boundary conditions
    bcs = u_inlet + [d_inlet, u_inlet_s, d_inlet_s, d_rigid]

    # Define the pressure condition (apply to inner surface, numerical instability results from applying to outlet, or using outlet flow rate)
    p_out_bc_val = InnerP(t=0.0, n=n, u=v_["n"], dsi=dsi, resistance=1e10, p_0=0.0, degree=p_deg)
    dSS = Measure("dS", domain=mesh, subdomain_data=boundaries)
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
    p_out_bc_val.update(t, v_["n"])

    return dict(inlet=inlet, p_out_bc_val=p_out_bc_val)


def post_solve(t, dt, dvp_, verbose, results_folder, eval_dict, counter, dump_probe_frequency,
               DG, CFL, characteristic_edge_length, **namespace):
    # Get deformation, velocity and pressure
    d, v, p = dvp_["n"].split(deepcopy=True)

    # Sample velocity and pressure in points/probes
    eval_dict["centerline_u_x_probes"](v.sub(0))
    eval_dict["centerline_u_y_probes"](v.sub(1))
    eval_dict["centerline_u_z_probes"](v.sub(2))
    eval_dict["centerline_p_probes"](p)

    # Store sampled velocity and pressure
    if counter > 0 and counter % dump_probe_frequency == 0:
        # Save variables along the centerline for CFD simulation
        # diagnostics and light-weight post processing
        filepath = path.join(results_folder, "Probes")
        if MPI.rank(MPI.comm_world) == 0:
            if not path.exists(filepath):
                makedirs(filepath)

        arr_u_x = eval_dict["centerline_u_x_probes"].array()
        arr_u_y = eval_dict["centerline_u_y_probes"].array()
        arr_u_z = eval_dict["centerline_u_z_probes"].array()
        arr_p = eval_dict["centerline_p_probes"].array()

        # Dump stats
        if MPI.rank(MPI.comm_world) == 0:
            arr_u_x.dump(path.join(filepath, "u_x_%s.probes" % str(counter)))
            arr_u_y.dump(path.join(filepath, "u_y_%s.probes" % str(counter)))
            arr_u_z.dump(path.join(filepath, "u_z_%s.probes" % str(counter)))
            arr_p.dump(path.join(filepath, "p_%s.probes" % str(counter)))

        # Clear stats
        MPI.barrier(MPI.comm_world)
        eval_dict["centerline_u_x_probes"].clear()
        eval_dict["centerline_u_y_probes"].clear()
        eval_dict["centerline_u_z_probes"].clear()
        eval_dict["centerline_p_probes"].clear()

    # Compute CFL
    v_mag = project(sqrt(inner(v, v)), DG)
    CFL.vector().set_local(v_mag.vector().get_local() / characteristic_edge_length.vector().get_local() * dt)
    CFL.vector().apply("insert")

    vec = CFL.vector()
    info_red("CFL --> min: {:e}, mean: {:e}, max: {:e}".format(vec.min(), np.mean(vec.get_local()), vec.max()))

def print_mesh_information(mesh):
    comm = MPI.comm_world
    local_xmin = mesh.coordinates()[:, 0].min()
    local_xmax = mesh.coordinates()[:, 0].max()
    local_ymin = mesh.coordinates()[:, 1].min()
    local_ymax = mesh.coordinates()[:, 1].max()
    local_zmin = mesh.coordinates()[:, 2].min()
    local_zmax = mesh.coordinates()[:, 2].max()
    xmin = comm.gather(local_xmin, 0)
    xmax = comm.gather(local_xmax, 0)
    ymin = comm.gather(local_ymin, 0)
    ymax = comm.gather(local_ymax, 0)
    zmin = comm.gather(local_zmin, 0)
    zmax = comm.gather(local_zmax, 0)

    local_num_cells = mesh.num_cells()
    local_num_edges = mesh.num_edges()
    local_num_faces = mesh.num_faces()
    local_num_facets = mesh.num_facets()
    local_num_vertices = mesh.num_vertices()
    num_cells = comm.gather(local_num_cells, 0)
    num_edges = comm.gather(local_num_edges, 0)
    num_faces = comm.gather(local_num_faces, 0)
    num_facets = comm.gather(local_num_facets, 0)
    num_vertices = comm.gather(local_num_vertices, 0)
    volume = assemble(Constant(1) * dx(mesh))

    if MPI.rank(MPI.comm_world) == 0:
        print("=== Mesh information ===")
        print("X range: {} to {} (delta: {:.4f})".format(min(xmin), max(xmax), max(xmax) - min(xmin)))
        print("Y range: {} to {} (delta: {:.4f})".format(min(ymin), max(ymax), max(ymax) - min(ymin)))
        print("Z range: {} to {} (delta: {:.4f})".format(min(zmin), max(zmax), max(zmax) - min(zmin)))
        print("Number of cells: {}".format(sum(num_cells)))
        print("Number of cells per processor: {}".format(int(np.mean(num_cells))))
        print("Number of edges: {}".format(sum(num_edges)))
        print("Number of faces: {}".format(sum(num_faces)))
        print("Number of facets: {}".format(sum(num_facets)))
        print("Number of vertices: {}".format(sum(num_vertices)))
        print("Volume: {:.4f}".format(volume))
        print("Number of cells per volume: {:.4f}".format(sum(num_cells) / volume))
