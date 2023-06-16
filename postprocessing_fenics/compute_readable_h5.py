import numpy as np
import h5py
from dolfin import *
import os
import re

from postprocessing_common import read_command_line, get_time_between_files
import stress_strain
from pathlib import Path

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 
parameters["reorder_dofs_serial"] = False



def format_output_data(case_path, mesh_name, dt, stride, save_deg, start_t, end_t):

    """
    Loads displacement and velocity data directly from turtleFSI output (Visualization/displacement.h5, Visualization/velocity.h5, ) 
    and reformats the data so that it can be read easily in fenics (Visualization_Separate_Domain/d.h5 and Visualization_Separate_Domain/v.h5). 
    This script works with restarts as well, by using the .xdmf file to point to the correct h5 file at the correct time. 
    This script must be run in serial while the more compuattionally intensive operations (wss and stress calculations) can then be run in parallel.

    Args:
        case_path (Path): Path to results from simulation
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        dt (float): Time step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (Separate Domain Visualization in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)
        start_t (float): desired start time for the output file
        end_t (float): desired end time for the output file

    """

    # Create directory paths 
    for file in os.listdir(case_path):
        file_path = os.path.join(case_path, file)
        if os.path.exists(os.path.join(file_path, "1")):
            visualization_separate_domain_path = os.path.join(file_path, "1/Visualization_separate_domain")
            visualization_path = os.path.join(file_path, "1/Visualization")

    # Create output directory if needed
    if os.path.exists(visualization_separate_domain_path):
        print('Path exists')
    if not os.path.exists(visualization_separate_domain_path):
        os.makedirs(visualization_separate_domain_path)

    d_path_in = str(os.path.join(visualization_separate_domain_path,"d.h5")) 
    v_path_in = str(os.path.join(visualization_separate_domain_path,"v.h5")) 


    # get fluid+solid (FSI) mesh
    mesh_name = mesh_name + ".h5"
    mesh_path = os.path.join(case_path, "mesh", mesh_name)
    # get fluid-only version of the mesh
    mesh_name_fluid = mesh_name.replace(".h5","_fluid_only.h5")
    mesh_path_fluid = os.path.join(case_path, "mesh", mesh_name_fluid)
    # get solid-only version of the mesh
    mesh_name_solid = mesh_name.replace(".h5","_solid_only.h5")
    mesh_path_solid = os.path.join(case_path, "mesh", mesh_name_solid)

    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg == 1:
        print("Warning, stress results are compromised by using save_deg = 1, especially using a coarse mesh. Recommend using save_deg = 2 instead for computing stress")
        mesh_path_viz_solid = mesh_path_solid
        mesh_path_viz_fluid = mesh_path_fluid

    else:
        mesh_path_viz_solid = mesh_path_solid.replace("_solid_only.h5","_refined_solid_only.h5")
        mesh_path_viz_fluid = mesh_path_fluid.replace("_fluid_only.h5","_refined_fluid_only.h5")
        mesh_path = mesh_path.replace(".h5","_refined.h5")


    # Get data from input mesh, .h5 and .xdmf files
    fluidIDs, solidIDs, allIDs = get_domain_ids(mesh_path) # Get list of all nodes in fluid, solid domains
    xdmf_file_v = os.path.join(visualization_path, 'velocity.xdmf') # Use velocity xdmf to determine which h5 file contains each timestep
    xdmf_file_d = os.path.join(visualization_path, 'displacement.xdmf') # Use displacement xdmf to determine which h5 file contains each timestep
    h5_ts, time_ts, index_ts = output_file_lists(xdmf_file_v) # Get list of h5 files containing each timestep, and corresponding indices for each timestep
    h5_ts_d, time_ts_d, index_ts_d = output_file_lists(xdmf_file_d)

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

    # Initialize variables
    tol = 1e-8  # temporal spacing tolerance, if this tolerance is exceeded, a warning flag will indicate that the data has uneven temporal spacing
    h5_file_prev = ""
    h5_file_prev_d = ""

    # Start file counter
    file_counter = 0 
    t_0, time_between_files = get_time_between_files(xdmf_file_v)
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation

    # Open up the first velocity.h5 file to get the number of timesteps and nodes for the output data
    file = visualization_path + '/'+  h5_ts[0]
    vectorData = h5py.File(file) 
    vectorArray = vectorData['VisualisationVector/0'][fluidIDs,:] 
    # Open up the first displacement.h5 file to get the number of timesteps and nodes for the output data
    file_d = visualization_path + '/'+  h5_ts_d[0]
    vectorData_d = h5py.File(file_d) 
    vectorArray_d = vectorData['VisualisationVector/0'][solidIDs,:] 

    while True:

        try:

            time_file = time_ts[file_counter] # Current time
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Timestep: {}".format(time_file), "=" * 10)
            
            if file_counter>0:
                if np.abs(time_file-time_ts[file_counter-1] - time_between_files) > tol: # if the spacing between files is not equal to the intended timestep
                    print('Warning: Uenven temporal spacing detected!!')
            
            if time_file>=start_t and time_file <= end_t: # For desired time range:

                # Open input velocity h5 file
                h5_file = visualization_path + '/'+h5_ts[file_counter]
                if h5_file != h5_file_prev: # If the h5 file is different than for the previous timestep, open the h5 file for the current timestep
                    vectorData.close()
                    vectorData = h5py.File(h5_file) 
                h5_file_prev = h5_file # Record h5 file name for this step
    
                # Open input displacement h5 file
                h5_file_d = visualization_path + '/'+h5_ts_d[file_counter]
                if h5_file_d != h5_file_prev_d: # If the h5 file is different than for the previous timestep, open the h5 file for the current timestep
                    vectorData_d.close()
                    vectorData_d = h5py.File(h5_file_d) 
                h5_file_prev_d = h5_file_d # Record h5 file name for this step
    
                # Open up Vector Arrays from h5 file
                ArrayName = 'VisualisationVector/' + str((index_ts[file_counter]))    
                vectorArrayFull = vectorData[ArrayName][:,:] # Important not to take slices of this array, slows code considerably... 
                ArrayName_d = 'VisualisationVector/' + str((index_ts_d[file_counter]))    
                vectorArrayFull_d = vectorData[ArrayName_d][:,:] # Important not to take slices of this array, slows code considerably... 
                # instead make a copy (VectorArrayFull) and slice that.
                
                vectorArray = vectorArrayFull[fluidIDs,:]    
                vectorArray_d = vectorArrayFull_d[solidIDs,:]    
                
                # Velocity
                vector_np_flat = vectorArray.flatten('F')
                v_viz.vector().set_local(vector_np_flat)  # Set u vector
                if MPI.rank(MPI.comm_world) == 0:
                    print("Saved data in v.h5")
    
                # Displacement
                vector_np_flat = vectorArray_d.flatten('F')
                d_viz.vector().set_local(vector_np_flat)  # Set d vector
                if MPI.rank(MPI.comm_world) == 0:
                    print("Saved data in d.h5")
    
                file_mode = "w" if not os.path.exists(v_path_in) else "a"
        
                # Save velocity
                viz_v_file = HDF5File(MPI.comm_world, v_path_in, file_mode=file_mode)
                viz_v_file.write(v_viz, "/velocity", time_file)
                viz_v_file.close()
        
                # Save displacment
                viz_d_file = HDF5File(MPI.comm_world, d_path_in, file_mode=file_mode)
                viz_d_file.write(d_viz, "/displacement", time_file)
                viz_d_file.close()


        except Exception as error:
            print("An exception occurred:", error) # An exception occurred: division by zero
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break

        # Update file_counter
        file_counter += stride
        #t += time_between_files*stride

# Helper Functions

def get_domain_topology(meshFile):
    # This function obtains the topology for the fluid, solid, and all elements of the input mesh
    # Importantly, it is ASSUMED that the fluid domain is labeled 1 and the solid domain is labeled 2 (and above, if multiple solid regions) 
    vectorData = h5py.File(meshFile,"r")
    domainsLoc = 'domains/values'
    domains = vectorData[domainsLoc][:] # Open domain array
    id_wall = (domains>1).nonzero() # domain = 2 and above is the solid
    id_fluid = (domains==1).nonzero() # domain = 1 is the fluid

    topologyLoc = 'domains/topology'
    allTopology = vectorData[topologyLoc][:,:] 
    wallTopology=allTopology[id_wall,:] 
    fluidTopology=allTopology[id_fluid,:]

    return fluidTopology, wallTopology, allTopology

def get_domain_ids(meshFile):
    # This function obtains a list of the node IDs for the fluid, solid, and all elements of the input mesh

    # Get topology of fluid, solid and whole mesh
    fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    solidIDs = np.unique(wallTopology) # find the unique node ids in the wall topology, sorted in ascending order
    fluidIDs = np.unique(fluidTopology) # find the unique node ids in the fluid topology, sorted in ascending order
    allIDs = np.unique(allTopology) 
    return fluidIDs, solidIDs, allIDs

def output_file_lists(xdmf_file):
    # If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
    # This loop determines the file names from the xdmf output file
    file1 = open(xdmf_file, 'r') 
    Lines = file1.readlines() 
    h5_ts=[]
    time_ts=[]
    index_ts=[]
    
    # This loop goes through the xdmf output file and gets the time value (time_ts), associated 
    # .h5 file (h5_ts) and index of each timestep inthe corresponding h5 file (index_ts)
    for line in Lines: 
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            time_ts.append(time)

        elif 'VisualisationVector' in line:
            #print(line)
            h5_pattern = '"HDF">(.+?):/'
            h5_str = re.findall(h5_pattern, line)
            h5_ts.append(h5_str[0])

            index_pattern = "VisualisationVector/(.+?)</DataItem>"
            index_str = re.findall(index_pattern, line)
            index = int(index_str[0])
            index_ts.append(index)
    time_increment_between_files = time_ts[2] - time_ts[1] # Calculate the time between files from xdmf file

    return h5_ts, time_ts, index_ts

if __name__ == '__main__':
    folder, mesh, _, _, _, dt, stride, save_deg, start_t, end_t = read_command_line()
    format_output_data(folder, mesh, dt, stride, save_deg, start_t, end_t)
