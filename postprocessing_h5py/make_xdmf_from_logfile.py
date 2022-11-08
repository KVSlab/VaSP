import matplotlib as mpl
mpl.use('Agg')
import os
from glob import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import re
import postprocessing_common_h5py
import h5py

"""
This script plots the compute time of the simulation graphically, from the logfiles generated on Scinet.

Args:
    --case: case_path (Path): Path to results from simulation
    --mesh: mesh_name: Name of the non-refined input mesh for the simulation. This function will find the refined mesh based on this name
    --save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only). If we input save_deg = 1 for a simulation 
       that was run in TurtleFSI with save_deg = 2, the output from this script will be save_deg = 1, i.e only the corner nodes will be output
    --dt (float): Actual time step of simulation
    --dvp (str): d, v or p, indicating whether to postprocess displacement, velocity or pressure
Example: python postprocessing_h5py/make_xdmf_from_logfile.py --case $SCRATCH/7_0_Surgical/Pulsatile_Ramp_Cases/Case11_predeformed_slow_inflation --mesh file_case11 --dt 0.00033964286 --dvp v --save_deg 2

"""

# Get input path, mesh name save_deg and dt from command line
case_path = postprocessing_common_h5py.read_command_line()[0] 
mesh_name = postprocessing_common_h5py.read_command_line()[1] 
save_deg = postprocessing_common_h5py.read_command_line()[2]
dt = postprocessing_common_h5py.read_command_line()[4]
dvp = postprocessing_common_h5py.read_command_line()[7]
# Get Mesh
if save_deg == 1:
    mesh_path = case_path + "/mesh/" + mesh_name +".h5" # Mesh path. Points to the corner-node input mesh
else: 
    mesh_path = case_path + "/mesh/" + mesh_name +"_refined.h5" # Mesh path. Points to the visualization mesh with intermediate nodes 

#read in the fsi mesh:
fsi_mesh = h5py.File(mesh_path,'r')

# Count fluid and total nodes
coordArray= fsi_mesh['mesh/coordinates'][:,:]
topoArray= fsi_mesh['mesh/topology'][:,:]
nNodes = coordArray.shape[0]
nElements = topoArray.shape[0]

case_name = os.path.basename(os.path.normpath(case_path)) # obtains only last folder in case_path
visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)

# Get output path
imageFolder = os.path.join(visualization_path,'..',"Images")
if not os.path.exists(imageFolder):
    os.makedirs(imageFolder)


# find all logfiles in simulaation folder (file name must contain the word "logfile")
outLog=[file for file in os.listdir(case_path) if 'logfile' in file]

modified_lines=[]
time_sim=[]
compute_time_step=[]
output_file_num=[]
n_outfiles = len(outLog)

print("Fixing logfile for case: " + case_path)
print("Found {} output log files".format(n_outfiles))
if n_outfiles == 0:
    print("Found no output files - ensure the word 'logfile' is in the output text file name")

for idx in range(0,n_outfiles):
    # Open log file
    outLogPath=os.path.join(case_path,outLog[idx])
    #print(outLog[idx])
    restart_number = int(re.findall("\d+", outLog[idx])[0]) # get restart number (run # - 1) from log file name
    run_number = restart_number + 1 # get restart number (run # - 1) from log file name
    #print(run_number)

    file1 = open(outLogPath, 'r') 
    Lines = file1.readlines() 

    # Open log file get compute time and simulation time from that logfile 
    compute_time_total=0
    for line in Lines: 
        if 'Solved for timestep' in line:
            modified_lines.append(line)
            numb = re.findall("\d*\.?\d+", line) # take numbers from string
            time_sim.append(float(numb[1])) # This is the simulation time
            output_file_num.append(run_number) # This is the run number of the output h5 file
            #print(numb[1])

## convert to numpy
time_file_data=np.array([time_sim,output_file_num]).T

time_file_data = time_file_data[time_file_data[:, 1].argsort()]  # sort run number 
time_file_data = time_file_data[time_file_data[:, 0].argsort(kind='mergesort')]  # sort by simulation time while maintaining previous sort

#np.savetxt(imageFolder + "/sim_file_by_timestep_with_duplicates.csv",np.transpose([time_file_data[:-1,0],time_file_data[:-1,1]]),delimiter=',',header='t (s), run #')

# remove duplicate time steps
duplicates = []#np.zeros(len(time_sim))
for i in range(len(time_file_data[:,0])):
    try:
        if np.abs(time_file_data[i,0]-time_file_data[i+1,0]) < 0.000001:
            duplicates.append(i)
    except:
        break

time_file_data = np.delete(time_file_data,duplicates,axis=0)


if dvp == "p":
    attType = "Scalar"
    viz_type = "Pressure"
elif dvp == "v":
    attType = "Vector"
    viz_type = "Velocity"
elif dvp == "d":
    viz_type = "Displacement"
    attType = "Vector"
else:
    print("ERROR, input dvp")



# Generate precise simulation time
accurate_time_sim = np.zeros(len(time_file_data[:,0]))
h5_file_list = []
for i in range(len(time_file_data[:,0])):
    accurate_time_sim[i] = dt*(i+1)
    h5_file_name = viz_type.lower() + '_run_' + str(int(round(time_file_data[i,1])))
    if os.path.exists(visualization_path + "/" + h5_file_name + ".h5") == 0:
        h5_file_name = viz_type.lower()
    h5_file_list.append(h5_file_name)
    #print(accurate_time_sim[i]-time_file_data[:,0])

# Save desired simulation file and time data
# np.savetxt(imageFolder + "/sim_file_by_timestep.csv",np.transpose([time_file_data[:-1,0],accurate_time_sim[:-1],time_file_data[:-1,1]]),delimiter=',',header='t (s), corrected t (s), run #')


postprocessing_common_h5py.create_fixed_xdmf_file(accurate_time_sim,nElements,nNodes,attType,viz_type,h5_file_list,visualization_path)
