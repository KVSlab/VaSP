'''
Gather all the names of Owais's cases on Niagara
'''
import os
import numpy as np
import postprocessing_common_h5py
import spectrograms as spec
import pandas as pd
import matplotlib.pyplot as plt
import pandas as pd 
from numpy.fft import fftfreq, fft, ifft


"""
This script calculates SPI for wss, displacement, velocity and pressure.

Args:
    mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    case_path (Path): Path to results from simulation
    stride: reduce output frequncy by this factor
    save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    start_t: Desired start time of the output files 
    end_t:  Desired end time of the output files 
    cut: Choose 'low' or 'high', low cuts frequencies below 'thresh', high cuts frequencies above 'thresh'.
    thresh: SPI frequency threshold in Hz"


Example: 
python postprocessing_h5py/create_spi.py --case /media/db_ubuntu/T7/Simulations/5_4_Verify_WSS/TFSI_rigid/_results/Case16_m06_FSI_rigid --save_deg 2 --mesh file_case16_el06 --end_t 0.951

"""

def compute_spi(case_path, mesh_name, save_deg, stride, start_t, end_t, dvp, bands):
    case_name = os.path.basename(os.path.normpath(case_path)) # obtains only last folder in case_path
    visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)
    
    bands_list = bands.split(",")
    num_bands = int(len(bands_list)/2)
    lower_freq = np.zeros(num_bands)
    higher_freq = np.zeros(num_bands)
    for i in range(num_bands):
        lower_freq[i] = float(bands_list[2*i])
        higher_freq[i] = float(bands_list[2*i+1])
    
    if save_deg == 1:
        if "/art" in visualization_path and "_cy" in visualization_path and "_ts" in visualization_path:
            mesh_path = case_path + "/data/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
            mesh_path_fluid = mesh_path
        else:
            mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
            mesh_path_fluid = mesh_path.replace(".h5","_fluid_only.h5")
    
    else: 
        mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 
        mesh_path_fluid = mesh_path.replace(".h5","_fluid_only.h5")
    
    formatted_data_folder_name = "res_"+case_name+'_stride_'+str(stride)+"t"+str(start_t)+"_to_"+str(end_t)+"save_deg_"+str(save_deg)
    formatted_data_folder = os.path.join(case_path,formatted_data_folder_name)
    visualization_separate_domain_folder = os.path.join(visualization_path,"../Visualization_separate_domain")
    
    # For wall shear stress, pressure, displacement or velocity
    
    # Create output folder and filenames
    output_file_name = case_name+"_"+ dvp+"_mag.npz" 
    formatted_data_path = formatted_data_folder+"/"+output_file_name
    
    # If the output file exists, don't re-make it
    if os.path.exists(formatted_data_path):
        print('path found!')
    elif dvp == "wss":
        time_between_input_files = postprocessing_common_h5py.create_transformed_matrix(visualization_separate_domain_folder, formatted_data_folder, mesh_path_fluid, case_name, start_t,end_t,dvp,stride)
    else: 
        # Make the output h5 files with dvp magnitudes
        time_between_input_files = postprocessing_common_h5py.create_transformed_matrix(visualization_path, formatted_data_folder, mesh_path, case_name, start_t,end_t,dvp,stride)
       
    
    # For spectrograms, we only want the magnitude
    component = dvp+"_mag"
    df = postprocessing_common_h5py.read_npz_files(formatted_data_path)
    for i in range(num_bands):
        postprocessing_common_h5py.calculate_spi(case_name, df, visualization_separate_domain_folder, mesh_path, start_t, end_t,lower_freq[i],higher_freq[i], dvp)
        print("low cut = {}, high cut = {}".format(lower_freq[i],higher_freq[i]))


if __name__ == '__main__':
    case_path, mesh_name, save_deg, stride, start_t, end_t, dvp, bands = postprocessing_common_h5py.read_command_line_spi()
    compute_spi(case_path, mesh_name, save_deg, stride, start_t, end_t, dvp, bands)

