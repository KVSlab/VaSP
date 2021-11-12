import os
import numpy as np
import postprocessing_common_h5py
import spectrograms as spec
import pandas as pd
#import matplotlib.pyplot as plt
import pandas as pd 


"""
This script creates spectrograms from formatted matrices, created in "compute_domain_specific_viz.py"

Args:
    mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    case_path (Path): Path to results from simulation
    stride: reduce output frequncy by this factor
    save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    start_t: Desired start time of the output files 
    end_t:  Desired end time of the output files 
    lowcut: High pass filter cutoff frequency (Hz)
    ylim: y limit of spectrogram graph")
    r_sphere: Sphere in which to include points for spectrogram, this is the sphere radius
    x_sphere: Sphere in which to include points for spectrogram, this is the x coordinate of the center of the sphere (in m)
    y_sphere: Sphere in which to include points for spectrogram, this is the y coordinate of the center of the sphere (in m)
    z_sphere: Sphere in which to include points for spectrogram, this is the z coordinate of the center of the sphere (in m)

"""

case_path, mesh_name, save_deg, stride,  start_t, end_t, lowcut, ylim, r_sphere, x_sphere, y_sphere, z_sphere, dvp = postprocessing_common_h5py.read_command_line_spec()
case_name = os.path.basename(os.path.normpath(case_path)) # obtains only last folder in case_path
visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)


# built-in spectrogram parameters
overlapFrac = 0.75 # Window overlap for spectrogram
window = 'blackmanharris' # Window type for spectrogram
nWindow_per_sec = 4 # Number of windows for spectrogram, should be roughly 7 windows/sec
nWindow = np.round(nWindow_per_sec*(end_t-start_t))+3
thresh=0.25 # New thresholding parameter - defines the cutoff threshold as a fraction of the total power range, not absolute values 
n_samples = 100 # number of points to sample for spectrogram

ident = "New_Threshold_Method"

if save_deg == 1:
    mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
else: 
    mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 
mesh_path_fluid = mesh_path.replace(".h5","_fluid_only.h5") # needed for formatting SPI data

formatted_data_folder = "res_"+case_name+'_stride_'+str(stride)+"t"+str(start_t)+"_to_"+str(end_t)+"save_deg_"+str(save_deg)
visualization_separate_domain_folder = os.path.join(visualization_path,"../Visualization_separate_domain")
imageFolder = os.path.join(visualization_path,"../Spectrograms")
if not os.path.exists(imageFolder):
    os.makedirs(imageFolder)

# For pressure, displacement, wss or velocity

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
    print('run create_domain_specific_viz.py before running spectrogram script!')

# For spectrograms, we only want the magnitude
component = dvp+"_mag"
df = postprocessing_common_h5py.read_npz_files(formatted_data_path)

# Get sampling constants
T, nsamples, fs = spec.get_sampling_constants(df,start_t,end_t)

# Get wall and fluid ids
fluidIDs, wallIDs, allIDs = postprocessing_common_h5py.get_domain_ids(mesh_path)

# We want to find the points in the sac, so we use a sphere to roughly define the sac.
sac_center = np.array([x_sphere, y_sphere, z_sphere])  
if dvp == "wss":
    outFile = os.path.join(visualization_separate_domain_folder,"WSS_ts.h5")
    surfaceElements, coords = postprocessing_common_h5py.get_surface_topology_coords(outFile)
else:
    coords = postprocessing_common_h5py.get_coords(mesh_path)


sphereIDs = spec.find_points_in_sphere(sac_center,r_sphere,coords)
# Get nodes in sac only
allIDs=list(set(sphereIDs).intersection(allIDs))
fluidIDs=list(set(sphereIDs).intersection(fluidIDs))
wallIDs=list(set(sphereIDs).intersection(wallIDs))

if dvp == "d":
    df = df.iloc[wallIDs]    # For displacement spectrogram, we need to take only the wall IDs
elif dvp == "wss":
    df = df.iloc[sphereIDs]  # for wss spectrogram, we use all the nodes within the sphere because the input df only includes the wall
else:
    df = df.iloc[fluidIDs]   # For velocity spectrogram, we need to take only the fluid IDs

df = df.sample(n=n_samples)
df = spec.filter_time_data(df,fs,lowcut=lowcut,highcut=15000.0,order=6,btype='highpass')

# Compute spectrogram 
bins, freqs, Pxx = spec.compute_average_spectrogram(df, fs, nWindow,overlapFrac,window,start_t,end_t,thresh)
bins = bins+start_t # Need to shift bins so that spectrogram timing is correct

fullname = case_name + '_'+str(nWindow)+'_windows_'+'_'+dvp+"_sac"+str(ident)+"thresh"+str(thresh)
path_to_fig = os.path.join(imageFolder, fullname + '.png')

spec.plot_spectrogram(bins,freqs,Pxx,case_name,min(bins),max(bins),ylim,path_to_fig)

# Old spectrogram method
if dvp == "d":
    df = df*1000    # Scale by 1000
thresh_val = -20
bins, freqs, Pxx = spec.compute_average_spectrogram(df, fs, nWindow,overlapFrac,window,start_t,end_t,thresh_val,filter_data=False,thresh_method="old")
bins = bins+start_t # Need to shift bins so that spectrogram timing is correct
fullname = case_name + '_'+str(nWindow)+'_windows_'+'_'+dvp+"_sac"+str(ident)+"thresh"+str(thresh_val)+"_old_method"
path_to_fig = os.path.join(imageFolder, fullname + '.png')
spec.plot_spectrogram(bins,freqs,Pxx,case_name,min(bins),max(bins),ylim,path_to_fig)

del(df)