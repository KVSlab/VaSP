import numpy as np
import h5py
from dolfin import *
import os
from postprocessing_common import read_command_line_stress, get_time_between_files
import stress_strain
from pathlib import Path

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 
parameters["reorder_dofs_serial"] = False


def format_output_data(case_path, mesh_name, dt, stride, save_deg):

    """
    Loads displacement and velocity data from domain specific xdmf/h5 outputs and reformats the data so that it can be read easily in fenics. 
    I couldn't figure out how to do this step in parallel, so this script must be run in serial while the more intensive operations 
    like wss and stress calculations can then be run in parallel

    Args:
        case_path (Path): Path to results from simulation
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        dt (float): Time step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (Separate Domain Visualization in this script)
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

    file_path_d = visualization_separate_domain_path / "displacement.h5"
    d_path_in = str(visualization_separate_domain_path.joinpath("d.h5")) 
    file_path_v = visualization_separate_domain_path / "velocity.h5"
    v_path_in = str(visualization_separate_domain_path.joinpath("v.h5")) 
    xdmf_file = visualization_separate_domain_path / "velocity.xdmf"


    if os.path.exists(file_path_d):
        if MPI.rank(MPI.comm_world) == 0:
            print("found displacement path")
        sim_type = "fsi"
    else:
        if MPI.rank(MPI.comm_world) == 0:
            print("could not find displacement path, postprocessing velocity only")
        sim_type = "cfd"


    # get fluid-only version of the mesh
    mesh_name = mesh_name + ".h5"
    mesh_name_fluid = mesh_name.replace(".h5","_fluid_only.h5")
    mesh_path_fluid = os.path.join(case_path, "mesh", mesh_name_fluid)
    
    if sim_type == "fsi": # get solid-only version of the mesh
        mesh_name_solid = mesh_name.replace(".h5","_solid_only.h5")
        mesh_path_solid = os.path.join(case_path, "mesh", mesh_name_solid)

    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg == 1:
        print("Warning, stress results are compromised by using save_deg = 1, especially using a coarse mesh. Recommend using save_deg = 2 instead for computing stress")
        if sim_type == "fsi":
            mesh_path_viz_solid = mesh_path_solid
        mesh_path_viz_fluid = mesh_path_fluid

    else:
        if sim_type == "fsi":
            mesh_path_viz_solid = mesh_path_solid.replace("_solid_only.h5","_refined_solid_only.h5")
        mesh_path_viz_fluid = mesh_path_fluid.replace("_fluid_only.h5","_refined_fluid_only.h5")

    # Read refined mesh saved as HDF5 format
    mesh_path_viz_fluid = Path(mesh_path_viz_fluid)
    mesh_viz_fluid = Mesh()
    with HDF5File(MPI.comm_world, mesh_path_viz_fluid.__str__(), "r") as mesh_file:
        mesh_file.read(mesh_viz_fluid, "mesh", False)

    # Create visualization function space for v
    ve_viz = VectorElement('CG', mesh_viz_fluid.ufl_cell(), 1)
    FSv_viz = FunctionSpace(mesh_viz_fluid, ve_viz)   # Visualisation FunctionSpace for v

    # Create lower-order function for visualization on refined mesh
    v_viz = Function(FSv_viz)
    

    if sim_type == "fsi": # If this is an FSI simulation, we also read in displacement
        # Read refined solid mesh saved as HDF5 format
        mesh_path_viz_solid = Path(mesh_path_viz_solid)
        mesh_viz_solid = Mesh()
        with HDF5File(MPI.comm_world, mesh_path_viz_solid.__str__(), "r") as mesh_file:
            mesh_file.read(mesh_viz_solid, "mesh", False)

        # Create visualization function space for d
        de_viz = VectorElement('CG', mesh_viz_solid.ufl_cell(), 1)
        FSd_viz = FunctionSpace(mesh_viz_solid, de_viz)   # Visualisation FunctionSpace for d 
        d_viz = Function(FSd_viz)

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # Start file counter
    file_counter = 0 
    t_0, time_between_files = get_time_between_files(xdmf_file)
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation
    t = t_0 # Initial time of simulation

    while True:

        try:
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Timestep: {}".format(t), "=" * 10)
    
            # Read in solution to vector function 
            vel_file = h5py.File(file_path_v, 'r')
            vec_name = "VisualisationVector/" + str(file_counter)
            vector_np = vel_file.get(vec_name)[()]
            vector_np_flat = vector_np.flatten('F')
            v_viz.vector().set_local(vector_np_flat)  # Set u vector

            if sim_type == "fsi": # If this is an FSI simulation, we also read in displacement
                # Read in solution to vector function 
                disp_file = h5py.File(file_path_d, 'r')
                vec_name = "VisualisationVector/" + str(file_counter)
                vector_np = disp_file.get(vec_name)[()]
                vector_np_flat = vector_np.flatten('F')
                d_viz.vector().set_local(vector_np_flat)  # Set d vector

        except:
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break

        file_mode = "w" if file_counter == 0 else "a"

        # Save velocity
        viz_v_file = HDF5File(MPI.comm_world, v_path_in, file_mode=file_mode)
        viz_v_file.write(v_viz, "/velocity", t)
        viz_v_file.close()

        if sim_type == "fsi": # If this is an FSI simulation, we also write displacement
            # Save displacment
            viz_d_file = HDF5File(MPI.comm_world, d_path_in, file_mode=file_mode)
            viz_d_file.write(d_viz, "/displacement", t)
            viz_d_file.close()

        # Update file_counter
        file_counter += stride
        t += time_between_files*stride



if __name__ == '__main__':
    folder, mesh, E_s, nu_s, dt, stride, save_deg = read_command_line_stress()
    format_output_data(folder,mesh, dt, stride, save_deg)
