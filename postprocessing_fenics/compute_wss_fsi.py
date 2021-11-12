from pathlib import Path

import numpy as np
import h5py
from dolfin import *
import os

from postprocessing_common import STRESS, read_command_line

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 
parameters["reorder_dofs_serial"] = False


def compute_wss(case_path,mesh_name, nu, dt, stride, save_deg):

    """
    Removed dabla c++ function that was seemingly only used used to calculate magnitude. Made the code hard to read and didn't work in parallel on Niagara.
    Loads velocity fields from completed CFD simulation,
    and computes and saves the following hemodynamic quantities:
    (1) WSS - Wall shear stress
    (2) TWSSG - Temporal wall shear stress gradient
    (3) OSI - Oscillatory shear index
    (4) RRT - Relative residence time

    Args:
        mesh_name: Name of the input mesh for the simulation. This function will find the fluid only mesh based on this name
        case_path (Path): Path to results from simulation
        nu (float): Viscosity
        dt (float): Actual ime step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (v.h5/d.h5 in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)
    """
    # File paths

    for file in os.listdir(case_path):
        file_path = os.path.join(case_path, file)
        if os.path.exists(os.path.join(file_path, "1")):
            visualization_separate_domain_path = os.path.join(file_path, "1/Visualization_separate_domain")
        elif os.path.exists(os.path.join(file_path, "Visualization")):
            visualization_separate_domain_path = os.path.join(file_path, "Visualization_separate_domain")
        elif os.path.exists(os.path.join(file_path, "Visualization_separate_domain")):
            visualization_separate_domain_path = os.path.join(file_path, "Visualization_separate_domain")
            
    visualization_separate_domain_path = Path(visualization_separate_domain_path)
    file_path_u = visualization_separate_domain_path / "v.h5"
    WSS_ts_path = (visualization_separate_domain_path / "WSS_ts.xdmf").__str__()

    # get fluid-only version of the mesh
    mesh_name = mesh_name + ".h5"
    mesh_name = mesh_name.replace(".h5","_fluid_only.h5")
    mesh_path = os.path.join(case_path, "mesh", mesh_name)

    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg == 1:
        mesh_path_viz = mesh_path
    else:
        mesh_path_viz = mesh_path.replace("_fluid_only.h5","_refined_fluid_only.h5")

    mesh_path = Path(mesh_path)
    mesh_path_viz = Path(mesh_path_viz)

    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    # Read refined mesh saved as HDF5 format
    mesh_viz = Mesh()
    with HDF5File(MPI.comm_world, mesh_path_viz.__str__(), "r") as mesh_file:
        mesh_file.read(mesh_viz, "mesh", False)
    
    # Load mesh
    bm = BoundaryMesh(mesh, 'exterior')
    
    if MPI.rank(MPI.comm_world) == 0:
        print("Define function spaces")
    V_b1 = VectorFunctionSpace(bm, "CG", 1)
    U_b1 = FunctionSpace(bm, "CG", 1)

    # Create visualization function space for d, v 
    dve_viz = VectorElement('CG', mesh_viz.ufl_cell(), 1)
    FSdv_viz = FunctionSpace(mesh_viz, dve_viz)   # Visualisation FunctionSpace for d and v

    # Create higher-order function space for d, v and p
    dve = VectorElement('CG', mesh.ufl_cell(), save_deg)
    FSdv = FunctionSpace(mesh, dve)   # Higher degree FunctionSpace for d and v

    if MPI.rank(MPI.comm_world) == 0:
        print("Define functions")

    # Create higher-order function on unrefined mesh for post-processing calculations
    u = Function(FSdv)

    # Create lower-order function for visualization on refined mesh
    u_viz = Function(FSdv_viz)

    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    dv_trans = PETScDMCollection.create_transfer_matrix(FSdv_viz,FSdv)
    
    # RRT
    RRT = Function(U_b1)

    # OSI
    OSI = Function(U_b1)

    # WSS_mean
    WSS_mean = Function(V_b1)
    wss_mean = Function(U_b1)

    # WSS_abs
    WSS_abs = Function(U_b1)
    #wss_abs = Function(U_b1)

    # TWSSG
    TWSSG = Function(U_b1)
    #twssg_ = Function(U_b1)
    twssg = Function(V_b1)
    tau_prev = Function(V_b1)

    stress = STRESS(u, 0.0, nu, mesh)

    WSS_file = XDMFFile(MPI.comm_world, WSS_ts_path)

    WSS_file.parameters["flush_output"] = True
    WSS_file.parameters["functions_share_mesh"] = True
    WSS_file.parameters["rewrite_function_mesh"] = False

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    file_counter = 0 # Index of first time step
    file_1 = 1 # Index of second time step

    f = HDF5File(MPI.comm_world, file_path_u.__str__(), "r")
    vec_name = "/velocity/vector_%d" % file_counter
    t_0 = f.attributes(vec_name)["timestamp"]
    vec_name = "/velocity/vector_%d" % file_1
    t_1 = f.attributes(vec_name)["timestamp"]  
    time_between_files = t_1 - t_0
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation

    
    while True:
        # Read in velocity solution to vector function u
        try:
            f = HDF5File(MPI.comm_world, file_path_u.__str__(), "r")
            vec_name = "/velocity/vector_%d" % file_counter
            t = f.attributes(vec_name)["timestamp"]
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Timestep: {}".format(t), "=" * 10)
            f.read(u_viz, vec_name)
        except:
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break   

        # Calculate v in P2 based on visualization refined P1
        u.vector()[:] = dv_trans*u_viz.vector()

        # Compute WSS
        if MPI.rank(MPI.comm_world) == 0:
            print("Compute WSS (mean)")
        tau = stress()     

        # tau.vector()[:] = tau.vector()[:] * 1000 # Removed this line, presumably a unit conversion
        WSS_mean.vector().axpy(1, tau.vector())

        if MPI.rank(MPI.comm_world) == 0:
            print("Compute WSS (absolute value)")

        wss_abs = project(inner(tau,tau)**(1/2),U_b1) # Calculate magnitude of Tau (wss_abs)
        WSS_abs.vector().axpy(1, wss_abs.vector())  # WSS_abs (cumulative, added together)
        # axpy : Add multiple of given matrix (AXPY operation)

        # Name functions
        wss_abs.rename("Wall Shear Stress", "WSS_abs")

        # Write results
        WSS_file.write(wss_abs, t)

        # Compute TWSSG
        if MPI.rank(MPI.comm_world) == 0:
            print("Compute TWSSG")
        twssg.vector().set_local((tau.vector().get_local() - tau_prev.vector().get_local()) / dt) # CHECK if this needs to be the time between files or the timestep of the simulation...
        twssg.vector().apply("insert")
        twssg_ = project(inner(twssg,twssg)**(1/2),U_b1) # Calculate magnitude of TWSSG vector
        TWSSG.vector().axpy(1, twssg_.vector())

        # Update tau
        if MPI.rank(MPI.comm_world) == 0:
            print("Update WSS \n")
        tau_prev.vector().zero()
        tau_prev.vector().axpy(1, tau.vector())

        # Update file_counter
        file_counter += stride

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Saving hemodynamic indices", "=" * 10)

    n = (file_counter) // stride
    TWSSG.vector()[:] = TWSSG.vector()[:] / n
    WSS_abs.vector()[:] = WSS_abs.vector()[:] / n
    WSS_mean.vector()[:] = WSS_mean.vector()[:] / n

    WSS_abs.rename("WSS", "WSS")
    TWSSG.rename("TWSSG", "TWSSG")

    try:
        wss_mean = project(inner(WSS_mean,WSS_mean)**(1/2),U_b1) # Calculate magnitude of WSS_mean vector
        wss_mean_vec = wss_mean.vector().get_local()
        wss_abs_vec = WSS_abs.vector().get_local()

        # Compute RRT and OSI based on mean and absolute WSS
        RRT.vector().set_local(1 / wss_mean_vec)
        RRT.vector().apply("insert")
        RRT.rename("RRT", "RRT")

        OSI.vector().set_local(0.5 * (1 - wss_mean_vec / wss_abs_vec))
        OSI.vector().apply("insert")
        OSI.rename("OSI", "OSI")
        save = True
    except:
        if MPI.rank(MPI.comm_world) == 0:
            print("Failed to compute OSI and RRT")
        save = False

    if save:
        # Save OSI and RRT
        rrt_path = (visualization_separate_domain_path / "RRT.xdmf").__str__()
        osi_path = (visualization_separate_domain_path / "OSI.xdmf").__str__()

        rrt = XDMFFile(MPI.comm_world, rrt_path)
        osi = XDMFFile(MPI.comm_world, osi_path)

        for f in [rrt, osi]:
            f.parameters["flush_output"] = True
            f.parameters["functions_share_mesh"] = True
            f.parameters["rewrite_function_mesh"] = False

        rrt.write(RRT)
        osi.write(OSI)

    # Save WSS and TWSSG
    wss_path = (visualization_separate_domain_path / "WSS.xdmf").__str__()
    twssg_path = (visualization_separate_domain_path / "TWSSG.xdmf").__str__()

    wss = XDMFFile(MPI.comm_world, wss_path)
    twssg = XDMFFile(MPI.comm_world, twssg_path)

    for f in [wss, twssg]:
        f.parameters["flush_output"] = True
        f.parameters["functions_share_mesh"] = True
        f.parameters["rewrite_function_mesh"] = False

    wss.write(WSS_abs)
    twssg.write(TWSSG)

if __name__ == '__main__':
    folder, mesh, nu, dt, stride, save_deg = read_command_line()
    compute_wss(folder, mesh, nu, dt, stride, save_deg)

