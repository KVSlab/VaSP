import os
from glob import glob
import numpy as np
import re
import postprocessing_common_h5py
import sys

"""
This script checks the temporal spacing in the xdmf files

Args:
    --case: case_path (Path): Path to results from simulation
    --dvp (str): postprocess d, v, or p (dicplacement velocity or pressure)
    --dt (float): Actual time step of simulation
    --start_t: Desired start time of the output files 
    --end_t:  Desired end time of the output files 
Example: python postprocessing_h5py/make_xdmf_from_logfile.py --case $SCRATCH/7_0_Surgical/Pulsatile_Ramp_Cases/Case11_predeformed_slow_inflation --mesh file_case11 --dt 0.00033964286 --dvp v --save_deg 2

"""

# Get input path, mesh name save_deg and dt from command line
case_path = postprocessing_common_h5py.read_command_line()[0] 
dvp = postprocessing_common_h5py.read_command_line()[7]
dt = postprocessing_common_h5py.read_command_line()[4]
start_t = postprocessing_common_h5py.read_command_line()[5]
end_t = postprocessing_common_h5py.read_command_line()[6]


visualization_path = postprocessing_common_h5py.get_visualization_path(case_path)

# Get name of xdmf file
if dvp == 'd':
    xdmf_file = 'displacement.xdmf' # 
elif dvp == 'v':
    xdmf_file = 'velocity.xdmf' # 
elif dvp == 'p':
    xdmf_file = 'pressure.xdmf' #  

# If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
# This loop goes through all xdmf files and determines whether they start and end at th correct time and have even temporal spacing



file1 = open(os.path.join(visualization_path,xdmf_file), 'r') 
Lines = file1.readlines() 
h5_ts=[]
time_ts=[]
index_ts=[]
time_prev=0.0
tol = 1e-5
broken_xdmf=0

# This loop goes through the xdmf output file and gets the time value (time_ts), associated 
# .h5 file (h5_ts) and index of each timestep inthe corresponding h5 file (index_ts)
for line in Lines: 
    if '<Time Value' in line:
        time_pattern = '<Time Value="(.+?)"'
        time_str = re.findall(time_pattern, line)
        time = float(time_str[0])
        time_ts.append(time)

expected_temporal_spacing = time_ts[2] - time_ts[1] # Calculate the time between files from xdmf file, assuming that the spacing between the 2nd and third timestep is correct

for time in time_ts:
    if time_prev > 0.0 + tol:
        temporal_spacing = time-time_prev
        if temporal_spacing > expected_temporal_spacing + tol:
            print("Temporal spacing ({}) greater than expected ({}) for the current step at time = {}".format(temporal_spacing,expected_temporal_spacing,time))
            broken_xdmf = 1
        elif temporal_spacing < expected_temporal_spacing - tol:
            print("Temporal spacing ({}) less than expected ({}) for the current step at time = {}".format(temporal_spacing,expected_temporal_spacing,time))
            broken_xdmf = 1
    time_prev = time

if broken_xdmf == 1:
    print("WARNING " + xdmf_file + " has UNEVEN temporal spacing, please fix.")
else:
    pass
    #print(xdmf_file + " has even temporal spacing.")
if time_ts[0] > start_t + expected_temporal_spacing + tol:
    print("WARNING " + xdmf_file + " starts after t = {}s at {}s, please fix.".format(start_t,time_ts[0]))
    broken_xdmf = 1
elif time_ts[-1] <  end_t - expected_temporal_spacing - tol:
    print("WARNING " + xdmf_file + " ends before t = {}s at {}s, please fix.".format(end_t,time_ts[-1]))
    broken_xdmf = 1
else:
    pass
    #print(xdmf_file + " starts at {}s, and ends at {}s".format(start_t,end_t))

#if broken_xdmf == 1:
#    print(case_path + ": fix "  + xdmf_file)
#    sys.stdout.write(dvp)
#    sys.exit(0)