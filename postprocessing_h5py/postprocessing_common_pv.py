import sys
import os
import numpy as np
from glob import glob
import h5py
import re
import shutil
from numpy import linalg as LA
from tempfile import mkdtemp
import pandas as pd 
from argparse import ArgumentParser
from numpy.fft import fftfreq, fft, ifft
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import time
import vtk
import pyvista as pv

"""
This script contains a number of helper functions to create visualizations outsside of fenics.
"""
def vtk_taubin_smooth(mesh, pass_band=0.1, feature_angle=60.0, iterations=20):
    """ Smooth mesh using Taubin method. """
    smoother = vtk.vtkWindowedSincPolyDataFilter()
    smoother.SetInputData(mesh) 
    smoother.SetNumberOfIterations(iterations)
    smoother.BoundarySmoothingOff()
    smoother.FeatureEdgeSmoothingOff() 
    smoother.SetFeatureAngle(feature_angle)
    smoother.SetPassBand(pass_band)
    smoother.NonManifoldSmoothingOn()
    smoother.NormalizeCoordinatesOn()
    smoother.Update()
    return pv.wrap(smoother.GetOutput())
def get_visualization_path(case_path):

    # Finds the visualization path for TurtleFSI simulations
    for file in os.listdir(case_path):
        file_path = os.path.join(case_path, file)
        if os.path.exists(os.path.join(file_path, "1")):
            visualization_path = os.path.join(file_path, "1/Visualization")
        elif os.path.exists(os.path.join(file_path, "Visualization")):
            visualization_path = os.path.join(file_path, "Visualization")
    
    return visualization_path

def q_criterion_nd(mesh):
    """ Compute non-dimensionalized q criterion.
    
    Based on "Automated Grid Refinement Using Feature Detection", Kamkar 2009.

    Args:
        mesh (vtkUnstructuredGrid): domain with velocity data

    Returns:
        vtkUnstructuredGrid: domain with non-dim Q
    """
    # if 'gradient' not in mesh.point_arrays:
    mesh = mesh.compute_derivative(
        scalars="u", gradient=True, qcriterion=False, faster=False)

    # Compute non-dim q
    J = mesh.point_arrays['gradient'].reshape(-1, 3, 3)
    S = 0.5 * (J + np.transpose(J, axes=(0,2,1)))
    A = 0.5 * (J - np.transpose(J, axes=(0,2,1)))

    S_norm = np.linalg.norm(S, axis=(1,2))
    A_norm = np.linalg.norm(A, axis=(1,2))

    Q = 0.5 * (A_norm**2 - S_norm**2)
    Q_nd = Q / S_norm**2

    mesh.point_arrays['qcriterion'] = Q
    mesh.point_arrays['qcriterion_nd'] = Q_nd

    return mesh

def assemble_mesh(mesh_file):
    """ Create UnstructuredGrid from h5 mesh file. """
   
    with h5py.File(mesh_file, 'r') as hf:
        points = np.array(hf['mesh/coordinates'][:,:])
        cells = np.array(hf['domains/topology'][:,:])
        celltypes = np.empty(cells.shape[0], dtype=np.uint8)
        celltypes[:] = vtk.VTK_TETRA
        cell_type = np.ones((cells.shape[0], 1), dtype=int) * 4
        cells = np.concatenate([cell_type, cells], axis = 1)
        mesh = pv.UnstructuredGrid(cells.ravel(), celltypes, points)
        surf = mesh.extract_surface()

    return mesh, surf

def get_data_at_idx(out_file, idx):
    vectorData = h5py.File(out_file) 
    viz_array_name = 'VisualisationVector/' + str(idx)
    vectorArray = vectorData[viz_array_name][:,:] 
    return vectorArray

def get_domain_ids(meshFile):
    # This function obtains a list of the node IDs for the fluid, solid, and all elements of the input mesh

    # Get topology of fluid, solid and whole mesh
    fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    wallIDs = np.unique(wallTopology) # find the unique node ids in the wall topology, sorted in ascending order
    fluidIDs = np.unique(fluidTopology) # find the unique node ids in the fluid topology, sorted in ascending order
    allIDs = np.unique(allTopology) 
    return fluidIDs, wallIDs, allIDs

def create_transformed_matrix(input_path, output_folder,meshFile, case_name, start_t,end_t,dvp,stride=1):
    # Create name for case, define output path
    print('Creating matrix for case {}...'.format(case_name))
    output_folder = output_folder

    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get node ids from input mesh. If save_deg = 2, you can supply the original mesh to get the data for the 
    # corner nodes, or supply a refined mesh to get the data for all nodes (very computationally intensive)
    if dvp == "d" or dvp == "v" or dvp == "p":
        fluidIDs, wallIDs, allIDs = get_domain_ids(meshFile)
        ids = allIDs

    # Get name of xdmf file
    if dvp == 'd':
        xdmf_file = input_path + '/displacement.xdmf' # Change
    elif dvp == 'v':
        xdmf_file = input_path + '/velocity.xdmf' # Change
    elif dvp == 'p':
        xdmf_file = input_path + '/pressure.xdmf' # Change
    elif dvp == 'wss':
        xdmf_file = input_path + '/WSS_ts.xdmf' # Change
    elif dvp == 'mps':
        xdmf_file = input_path + '/MaxPrincipalStrain.xdmf' # Change
    elif dvp == 'strain':
        xdmf_file = input_path + '/InfinitesimalStrain.xdmf' # Change
    else:
        print('input d, v, p, mps, strain or wss for dvp')

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
    print(index_ts)
    time_between_files = time_ts[2] - time_ts[1] # Calculate the time between files from xdmf file

    # Open up the first h5 file to get the number of timesteps and nodes for the output data
    file = input_path + '/'+  h5_ts[0]
    vectorData = h5py.File(file) 
    if dvp == "wss" or dvp == "mps" or dvp == "strain":
        ids = list(range(len(vectorData['VisualisationVector/0'][:])))
    vectorArray = vectorData['VisualisationVector/0'][ids,:] 

    num_ts = int(len(time_ts))  # Total amount of timesteps in original file

    # Get shape of output data
    num_rows = vectorArray.shape[0]
    num_cols = int((end_t-start_t)/(time_between_files*stride))-1 

    # Pre-allocate the arrays for the formatted data
    if dvp == "v" or dvp == "d":
        dvp_x = np.zeros((num_rows, num_cols))
        dvp_y = np.zeros((num_rows, num_cols))
        dvp_z = np.zeros((num_rows, num_cols))
    elif dvp == "strain":
        dvp_11 = np.zeros((num_rows, num_cols))
        dvp_12 = np.zeros((num_rows, num_cols))
        dvp_22 = np.zeros((num_rows, num_cols))
        dvp_23 = np.zeros((num_rows, num_cols))
        dvp_33 = np.zeros((num_rows, num_cols))
        dvp_31 = np.zeros((num_rows, num_cols))

    dvp_magnitude = np.zeros((num_rows, num_cols))

    # Initialize variables
    tol = 1e-8  # temporal spacing tolerance, if this tolerance is exceeded, a warning flag will indicate that the data has uneven spacing
    idx_zeroed = 0 # Output index for formatted data
    h5_file_prev = ""
    for i in range(0,num_ts):
        time_file=time_ts[i]
        if i>0:
            if np.abs(time_file-time_ts[i-1] - time_between_files) > tol: # if the spacing between files is not equal to the intended timestep
                print('Warning: Uenven temporal spacing detected!!')

        # Open input h5 file
        h5_file = input_path + '/'+h5_ts[i]
        if h5_file != h5_file_prev: # If the h5 file is different than for the previous timestep, open the h5 file for the current timestep
            vectorData.close()
            vectorData = h5py.File(h5_file) 
        h5_file_prev = h5_file # Record h5 file name for this step

        # If the timestep falls within the desired timeframe and has the correct stride
        if time_file>=start_t and time_file <= end_t and i%stride == 0:

            # Open up Vector Array from h5 file
            ArrayName = 'VisualisationVector/' + str((index_ts[i]))    
            vectorArrayFull = vectorData[ArrayName][:,:] # Important not to take slices of this array, slows code considerably... 
            # instead make a copy (VectorArrayFull) and slice that.
            
            try:
                # Get required data depending on whether pressure, displacement or velocity
                if dvp == "p" or dvp == "wss" or dvp =="mps":
                    dvp_magnitude[:,idx_zeroed] = vectorArrayFull[ids,0] # Slice VectorArrayFull
                elif dvp == "strain":
                    vectorArray = vectorArrayFull[ids,:]    
                    dvp_11[:,idx_zeroed] = vectorArray[:,0]
                    dvp_12[:,idx_zeroed] = vectorArray[:,1]
                    dvp_22[:,idx_zeroed] = vectorArray[:,4]
                    dvp_23[:,idx_zeroed] = vectorArray[:,5]
                    dvp_33[:,idx_zeroed] = vectorArray[:,8]
                    dvp_31[:,idx_zeroed] = vectorArray[:,6]
                else:
                    vectorArray = vectorArrayFull[ids,:]    
                    dvp_x[:,idx_zeroed] = vectorArray[:,0]
                    dvp_y[:,idx_zeroed] = vectorArray[:,1]
                    dvp_z[:,idx_zeroed] = vectorArray[:,2]
                    dvp_magnitude[:,idx_zeroed] = LA.norm(vectorArray, axis=1) 

            except:
                print("Finished reading solutions")
                break

            print('Transferred timestep number {} at time: '.format(index_ts[i])+ str(time_ts[i]) +' from file: '+ h5_ts[i])
            idx_zeroed+=1 # Move to the next index of the output h5 file
    
    vectorData.close()

    # Create output h5 file

    # Remove blank columns
    if dvp == "d" or dvp == "v":
        formatted_data = [dvp_magnitude,dvp_x,dvp_y,dvp_z]
        component_names = ["mag","x","y","z"]
    elif dvp == "strain":
        formatted_data = [dvp_11,dvp_12,dvp_22,dvp_23,dvp_33,dvp_31]
        component_names = ["11","12","22","23","33","31"]
    else:
        component_names = ["mag"]

    for i in range(len(component_names)):

        # Create output path
        component = dvp+"_"+component_names[i]
        output_file_name = case_name+"_"+ component+'.npz'  
        output_path = os.path.join(output_folder, output_file_name) 

        # Remove old file path
        if os.path.exists(output_path):
            print('File path exists; rewriting')
            os.remove(output_path)

        # Store output in npz file
        if dvp == "v" or dvp =="d" or dvp =="strain":
            np.savez_compressed(output_path, component=formatted_data[i])
        else:
            np.savez_compressed(output_path, component=dvp_magnitude)

    return time_between_files



if __name__ == "__main__":
    print('See functions.')
