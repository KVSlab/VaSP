import matplotlib as mpl
mpl.use('Agg')
import numpy as np
import h5py
from dolfin import *
import os
from postprocessing_common import read_command_line_stress, get_time_between_files
import stress_strain
from pathlib import Path
import matplotlib.pyplot as plt


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
    cycle_length = 0.951
    compare_cycle=2 # compare 3rd and 4th cycle to cycle #2
    # File paths

    for file in os.listdir(case_path):
        file_path = os.path.join(case_path, file)
        if os.path.exists(os.path.join(file_path, "1")):
            visualization_separate_domain_path = os.path.join(file_path, "1/Visualization_separate_domain")
        elif os.path.exists(os.path.join(file_path, "Visualization")):
            visualization_separate_domain_path = os.path.join(file_path, "Visualization_separate_domain")
        elif os.path.exists(os.path.join(file_path, "Visualization_separate_domain")):
            visualization_separate_domain_path = os.path.join(file_path, "Visualization_separate_domain") 
    
    imageFolder = os.path.join(visualization_separate_domain_path, "../Images") 
    if not os.path.exists(imageFolder):
        print("creating image folder")
        os.makedirs(imageFolder)

    file_path_d = Path(os.path.join(visualization_separate_domain_path, "displacement_save_deg_"+str(save_deg)+'.h5'))
    file_path_v = Path(os.path.join(visualization_separate_domain_path, "velocity_save_deg_"+str(save_deg)+'.h5')) 
    xdmf_file = Path(os.path.join(visualization_separate_domain_path, "velocity_save_deg_"+str(save_deg)+'.xdmf')) 


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
    v_viz_cycle_0 = Function(FSv_viz)
    
    Difference_v=Function(FSv_viz)
    Difference_v_vec = Difference_v.vector() 
    norm_v_l2_list = []
    norm_v_linf_list = []
    t_list = []

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
        d_viz_cycle_0 = Function(FSd_viz)
        Difference_d=Function(FSd_viz)
        norm_d_l2_list = []
        norm_d_linf_list = []

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # Start file counter
    file_counter = 0 
    t_0, time_between_files = get_time_between_files(xdmf_file)
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation
    t = t_0 # Initial time of simulation
    files_per_cycle = round(cycle_length/(save_step*dt))
    print(files_per_cycle)

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

        # Get the first cycle same timestep
        
#        if t>=cycle_length*compare_cycle:
        if t>=cycle_length:
            # Read in solution to vector function 
            vel_file = h5py.File(file_path_v, 'r')
            #vec_name = "VisualisationVector/" + str(file_counter % files_per_cycle + files_per_cycle*(compare_cycle-1)) # Get the first cycle same timestep
            vec_name = "VisualisationVector/" + str(file_counter - files_per_cycle) # Get the first cycle same timestep
            vector_np = vel_file.get(vec_name)[()]
            vector_np_flat = vector_np.flatten('F')
            v_viz_cycle_0.vector().set_local(vector_np_flat)  # Set u vector

            if sim_type == "fsi": # If this is an FSI simulation, we also read in displacement
                # Read in solution to vector function 
                disp_file = h5py.File(file_path_d, 'r')
                #vec_name = "VisualisationVector/" + str(file_counter % files_per_cycle + files_per_cycle*(compare_cycle-1)) # Get the first cycle same timestep
                vec_name = "VisualisationVector/" + str(file_counter - files_per_cycle) # Get the first cycle same timestep
                vector_np = disp_file.get(vec_name)[()] 
                vector_np_flat = vector_np.flatten('F')
                d_viz_cycle_0.vector().set_local(vector_np_flat)  # Set d vector

            # Compute Norms
            Difference_v.vector()[:] = v_viz.vector() - v_viz_cycle_0.vector()
            norm_v_l2 = norm(Difference_v,'l2')
            #norm_v_linf = norm(Difference_v,'linf') # Linf doesnt work in dolfin 2018 it seems

            print("{} is the L2 norm of the difference between step {} and {}".format(norm_v_l2, file_counter, file_counter - files_per_cycle)) #file_counter % files_per_cycle))
            Difference_d.vector()[:] = d_viz.vector() - d_viz_cycle_0.vector()
            norm_d_l2 = norm(Difference_d,'l2')
            #norm_d_linf = norm(Difference_d,'linf')
            print("{} is the L2 norm of the difference between step {} and {}".format(norm_d_l2, file_counter, file_counter - files_per_cycle)) #file_counter % files_per_cycle))

            norm_v_l2_list.append(norm_v_l2)
            #norm_v_linf_list.append(norm_v_linf)
            norm_d_l2_list.append(norm_d_l2)
            #norm_d_linf_list.append(norm_d_linf)
            t_list.append(t)
 

        # Update file_counter
        file_counter += stride
        t += time_between_files*stride


    # Plot and Save 
    plt.plot(t_list,norm_v_l2_list)
    plt.ylabel('L2 Norm - Velocity')
    plt.xlabel('Simulation Time (s)')
    imagePath=imageFolder+'/v_L2_norm_compare_to_prev_cycle_log.png'
    plt.yscale("log")
    plt.savefig(imagePath)  
    plt.close()
    csvPath = imagePath.replace(".png",".csv")
    np.savetxt(csvPath, np.transpose([t_list[:-1],norm_d_l2_list[:-1]]), delimiter=",")

    # Plot and Save
    plt.plot(t_list,norm_d_l2_list)
    plt.ylabel('L2 Norm - Displacement')
    plt.xlabel('Simulation Time (s)')
    imagePath=imageFolder+'/d_L2_norm_compare_to_prev_cycle_log.png'
    plt.yscale("log")
    plt.savefig(imagePath)  
    plt.close()
    csvPath = imagePath.replace(".png",".csv")
    np.savetxt(csvPath, np.transpose([t_list[:-1],norm_v_l2_list[:-1]]), delimiter=",")

    # Normalize Data and plot
    norm_v_l2_list_normalized = norm_v_l2_list/np.max(norm_v_l2_list)
    norm_d_l2_list_normalized = norm_d_l2_list/np.max(norm_d_l2_list)

    # Plot and Save
    plt.plot(t_list,norm_v_l2_list_normalized)
    plt.ylabel('L2 Norm - Velocity')
    plt.xlabel('Simulation Time (s)')
    imagePath=imageFolder+'/v_L2_norm_compare_to_prev_cycle_log_normalized.png'
    plt.yscale("log")
    plt.savefig(imagePath)  
    plt.close()
    csvPath = imagePath.replace(".png",".csv")
    np.savetxt(csvPath, np.transpose([t_list[:-1],norm_v_l2_list_normalized[:-1]]), delimiter=",")

    # Plot and Save
    plt.plot(t_list,norm_d_l2_list_normalized)
    plt.ylabel('L2 Norm - Displacement')
    plt.xlabel('Simulation Time (s)')
    imagePath=imageFolder+'/d_L2_norm_compare_to_prev_cycle_log_normalized.png'
    plt.yscale("log")
    plt.savefig(imagePath)  
    plt.close()
    csvPath = imagePath.replace(".png",".csv")
    np.savetxt(csvPath, np.transpose([t_list[:-1],norm_d_l2_list_normalized[:-1]]), delimiter=",")

if __name__ == '__main__':
    folder, mesh, E_s, nu_s, dt, stride, save_deg = read_command_line_stress()
    format_output_data(folder,mesh, dt, stride, save_deg)
