import os
from glob import glob
import numpy as np
import postprocessing_common_h5py
#import spectrograms as spec
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd 

"""
This script takes the visualization files from TurtleFSI and outputs the wall displacement for the wall only, and velocity and pressure for the 
fluid only. This way the simualtion can be postprocessed as a CFD simulation and a Solid Mechanics Element simulation separately. A "Transformed
Matrix" is created as well, which stores the output data in a format that can be opened quickly when we want to create spectrograms.

Args:
    mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    case_path (Path): Path to results from simulation
    stride: reduce output frequncy by this factor
    save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    dt (float): Actual time step of simulation
    start_t: Desired start time of the output files 
    end_t:  Desired end time of the output files 


Example: --dt 0.000679285714286 --mesh file_case16_el06 --end_t 0.951

"""


# Get Command Line Arguments and Input File Paths
case_path, mesh_name, save_deg, stride, ts, start_t, end_t, dvp = postprocessing_common_h5py.read_command_line()
case_name = os.path.basename(os.path.normpath(case_path)) # obtains only last folder in case_path
visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)

# Get Mesh
if save_deg == 1:
    mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
else: 
    mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 
mesh_path_sd1 = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh

# Get Command Line Arguments and Input File Paths
formatted_data_folder = "res_"+case_name+'_stride_'+str(stride)+"t"+str(start_t)+"_to_"+str(end_t)+"save_deg_"+str(save_deg)
visualization_separate_domain_folder = os.path.join(visualization_path,"../Visualization_separate_domain")
visualization_hi_pass_folder = os.path.join(visualization_path,"../visualization_hi_pass")
#visualization_sd1_folder = os.path.join(visualization_path,"../visualization_save_deg_1")

lowcut=25 # low cut frequency in Hz, for hi-pass displacement and velocity

# For pressure, displacement or velocity

# Create output folder and filenames
output_file_name = case_name+"_"+ dvp+"_mag.npz" 
formatted_data_path = formatted_data_folder+"/"+output_file_name

# If the output file exists, don't re-make it
if os.path.exists(formatted_data_path):
    print('path found!')
    time_between_output_files = ts*stride
else: 
	# Make the output h5 files with dvp magnitudes
    time_between_input_files = postprocessing_common_h5py.create_transformed_matrix(visualization_path, formatted_data_folder, mesh_path, case_name, start_t,end_t,dvp,stride)
    
    # Get the desired time between output files (reduce output frequency by "stride")
    time_between_output_files = time_between_input_files*stride 

if dvp == "v" or dvp == "d":
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,25,highcut=100,magnitude=True)
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,25,highcut=100)
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,100,highcut=150,magnitude=True)
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,100,highcut=150)
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,150,highcut=175,magnitude=True)
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,150,highcut=175)
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,175,highcut=200,magnitude=True)
    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_output_files,dvp,175,highcut=200)
#if dvp == "d":
#    postprocessing_common_h5py.create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,time_between_files,dvp,lowcut)
#

