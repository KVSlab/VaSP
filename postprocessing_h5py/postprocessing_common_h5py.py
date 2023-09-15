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
import spectrograms as spec
from argparse import ArgumentParser
from numpy.fft import fftfreq, fft, ifft
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from scipy import signal
import time

"""
This script contains a number of helper functions to create visualizations outsside of fenics.

All functions authored by David Bruneau unless specified

"""

def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--case', type=str, default="cyl_test", help="Path to simulation results",
                        metavar="PATH")
    parser.add_argument('--mesh', type=str, default="artery_coarse_rescaled", help="Mesh File Name",
                        metavar="PATH")
    parser.add_argument('--save_deg', type=int, default=2, help="Input save_deg of simulation, i.e whether the intermediate P2 nodes were saved. Entering save_deg = 1 when the simulation was run with save_deg = 2 will result in only the corner nodes being used in postprocessing")
    parser.add_argument('--stride', type=int, default=1, help="Desired frequency of output data (i.e to output every second step, stride = 2)")    

    parser.add_argument('--dt', type=float, default=0.001, help="Time step of simulation (s)")

    parser.add_argument('--start_t', type=float, default=0.0, help="Start time of simulation (s)")
    parser.add_argument('--end_t', type=float, default=0.05, help="End time of simulation (s)")
    parser.add_argument('--dvp', type=str, default="v", help="Quantity to postprocess, input v for velocity, d for sisplacement, p for pressure, or wss for wall shear stress")
    parser.add_argument('--bands', default="25,100000", help="input lower then upper band for Band-pass filtered displacement, in a list of pairs. for example: --bands '100 150 175 200' gives you band-pass filtered visualization for the band between 100 and 150, and another visualization for the band between 175 and 200")
    parser.add_argument('--points', default="0,1", help="input list of points")

    args = parser.parse_args()

    return args.case, args.mesh, args.save_deg, args.stride, args.dt, args.start_t, args.end_t, args.dvp, args.bands, args.points


def get_visualization_path(case_path):

    if os.path.exists(os.path.join(case_path,"results")):
        # Finds the visualization path for BSL Solver format
        file_path = os.path.join(case_path,"results")
        files = sorted(glob(file_path + '/art*'))
        visualization_path = files[0]

    else:
        # Finds the visualization path for TurtleFSI simulations
        for file in os.listdir(case_path):
            file_path = os.path.join(case_path, file)
            if os.path.exists(os.path.join(file_path, "1")):
                visualization_path = os.path.join(file_path, "1/Visualization")
            elif os.path.exists(os.path.join(file_path, "Visualization")):
                visualization_path = os.path.join(file_path, "Visualization")
    
    return visualization_path

def get_coords(meshFile):
    mesh = h5py.File(meshFile,"r")
    coords = mesh['mesh/coordinates'][:,:] 
    return coords

def get_surface_topology_coords(outFile):
    mesh = h5py.File(outFile,"r")
    topology = mesh["Mesh/0/mesh/topology"][:,:]
    coords = mesh["Mesh/0/mesh/geometry"][:,:]
    return topology, coords

def get_domain_topology(meshFile):
    # This function obtains the topology for the fluid, solid, and all elements of the input mesh
    vectorData = h5py.File(meshFile,"r")
    domainsLoc = 'domains/values'
    domains = vectorData[domainsLoc][:] # Open domain array
    id_wall = (domains>1).nonzero() # domain = 2 is the solid
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
    wallIDs = np.unique(wallTopology) # find the unique node ids in the wall topology, sorted in ascending order
    fluidIDs = np.unique(fluidTopology) # find the unique node ids in the fluid topology, sorted in ascending order
    allIDs = np.unique(allTopology) 
    return fluidIDs, wallIDs, allIDs

def get_domain_ids_specified_region(meshFile,fluid_sampling_domain_ID,solid_sampling_domain_ID):
    # This function obtains the topology for the fluid, solid, and all elements of the input mesh
    vectorData = h5py.File(meshFile,"r")
    domainsLoc = 'domains/values'
    domains = vectorData[domainsLoc][:] # Open domain array
    id_wall = (domains==solid_sampling_domain_ID).nonzero() # domain = 2 is the solid
    id_fluid = (domains==fluid_sampling_domain_ID).nonzero() # domain = 1 is the fluid

    topologyLoc = 'domains/topology'
    allTopology = vectorData[topologyLoc][:,:] 
    wallTopology=allTopology[id_wall,:] 
    fluidTopology=allTopology[id_fluid,:]

    wallIDs = np.unique(wallTopology) # find the unique node ids in the wall topology, sorted in ascending order
    fluidIDs = np.unique(fluidTopology) # find the unique node ids in the fluid topology, sorted in ascending order
    allIDs = np.unique(allTopology) 
    
    return fluidIDs, wallIDs, allIDs

def get_domain_ids(meshFile):
    # This function obtains a list of the node IDs for the fluid, solid, and all elements of the input mesh

    # Get topology of fluid, solid and whole mesh
    fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    wallIDs = np.unique(wallTopology) # find the unique node ids in the wall topology, sorted in ascending order
    fluidIDs = np.unique(fluidTopology) # find the unique node ids in the fluid topology, sorted in ascending order
    allIDs = np.unique(allTopology) 
    return fluidIDs, wallIDs, allIDs

def get_interface_ids(meshFile):
    fluidIDs, wallIDs, allIDs = get_domain_ids(meshFile)
    interfaceIDs_set = set(fluidIDs) - (set(fluidIDs) - set(wallIDs))
    interfaceIDs = np.array(list(interfaceIDs_set))
    return interfaceIDs

def get_sampling_constants(df,start_t,end_t):
    '''
    Author: Daniel Macdonald
    T = period, in seconds, 
    nsamples = samples per cycle
    fs = sample rate
    '''
    T = end_t - start_t
    nsamples = df.shape[1]
    fs = nsamples/T 
    return T, nsamples, fs 

def read_npz_files(filepath):

    data = np.load(filepath)['component']
    print('making df...')
    df = pd.DataFrame(data,copy=False)
    df.index.names = ['Ids']
    print('ready')
    return df

def read_csv_file(filepath):

    print('making df...')
    df = pd.read_csv(filepath).T
    return df

def filter_SPI(U, W_low_cut, tag):
    """
    Author: Mehdi Najafi

    """
    if tag=="withmean":
        U_fft = fft(U)
    else:
        U_fft = fft(U-np.mean(U))
    # filter any amplitude corresponding frequency equal to 0Hz
    U_fft[W_low_cut[0]] = 0
    # filter any amplitude corresponding frequency lower to 25Hz
    U_fft_25Hz = U_fft.copy()
    #print(U_fft_25Hz)

    U_fft_25Hz[W_low_cut[1]] = 0 # This line won't work if W is not the right shape/size!!!
    # compute the absolute values

    Power_25Hz = np.sum ( np.power( np.absolute(U_fft_25Hz),2))
    Power_0Hz  = np.sum ( np.power( np.absolute(U_fft     ),2))


    if Power_0Hz < 1e-8:
        return 0
    return Power_25Hz/Power_0Hz


def filter_SPI_print(U, W_low_cut, tag):
   """
    Author: Mehdi Najafi

    """
    if tag=="withmean":
        U_fft = fft(U)
    else:
        U_fft = fft(U-np.mean(U))
    #print(U_fft)
    # filter any amplitude corresponding frequency equal to 0Hz
    U_fft[W_low_cut[0]] = 0
    # filter any amplitude corresponding frequency lower to 25Hz
    U_fft_25Hz = U_fft.copy()
    #print(U_fft_25Hz)

    U_fft_25Hz[W_low_cut[1]] = 0 # This line won't work if W is not the right shape/size!!!
    # compute the absolute values

    Power_25Hz = np.sum ( np.power( np.absolute(U_fft_25Hz),2))
    Power_0Hz  = np.sum ( np.power( np.absolute(U_fft     ),2))


    if Power_0Hz < 1e-8:
        return 0
    return Power_25Hz/Power_0Hz

def calculate_spi(case_name, df, output_folder, meshFile,start_t,end_t,low_cut,high_cut, dvp):
    """
    Author: Mehdi Najafi and David Bruneau

    """
    # cut, thresh
    # Get wall and fluid ids
    fluidIDs, wallIDs, allIDs = get_domain_ids(meshFile)
    fluidElements, wallElements, allElements = get_domain_topology(meshFile)

    # For displacement spectrogram, we need to take only the wall IDs, filter the data and scale it. 
    if dvp == "wss":
        outFile = os.path.join(output_folder,"WSS_ts.h5")
        surfaceElements, coord = get_surface_topology_coords(outFile)
        elems=np.squeeze(surfaceElements)
        IDs = list(range(len(coord)))

    elif dvp == "d":
        IDs = wallIDs
        elems=np.squeeze(wallElements)
        Coords = get_coords(meshFile)
        coord = Coords[IDs,:]
    else:
        IDs = fluidIDs
        elems=np.squeeze(fluidElements)
        Coords = get_coords(meshFile)
        coord = Coords[IDs,:]    

    df_spec = df.iloc[IDs]          
    #ids_out = [0,1,2,3,4,5,6,7,8]

    T, num_ts, fs = spec.get_sampling_constants(df_spec,start_t,end_t)
    time_between_files = 1/fs
    W = fftfreq(num_ts, d=time_between_files)

    # Cut low and high frequencies
    mask = np.logical_or(np.abs(W) < low_cut, np.abs(W) > high_cut )
    W_cut = np.where( np.abs(W) == 0 )  + np.where(mask)

    #W_cut = np.where( np.abs(W) == 0 )  + np.where( np.abs(W) < low_cut )  # only low cut
    
    #print("array of W (high and low cut)")
    #print(len(W_cut))
    #print(W_cut)

    number_of_points = len(IDs)
    SPI = np.zeros([number_of_points])
    
    #print(df_spec.iloc[6])
    #print(filter_SPI_print(df_spec.iloc[6],W_cut,"withoutmean"))
    #print(df_spec.iloc[7])
    #print(filter_SPI_print(df_spec.iloc[7],W_cut,"withoutmean"))
    #print(df_spec)
    for i in range(len(IDs)):
        SPI[i] = filter_SPI(df_spec.iloc[i],W_cut,"withoutmean")
        #if SPI[i] > 0.01:
        #    ids_out.append(i)
                            
    output_filename = output_folder+'/'+case_name+'_spi_'+str(low_cut)+'_to_'+str(high_cut)+"_t"+str(start_t)+"_to_"+str(end_t)+"_"+dvp+'.tec'
    #df_out = df_spec.iloc[ids_out]
    #df_out = df_out.T              
    #df_out.to_csv(path_or_buf=output_folder+"/df_short.csv")

     
 
    for j in range(len(IDs)):
        elems[elems == IDs[j]] = j

    outfile = open(output_filename, 'w')
    if dvp == "wss":
        outfile.write('VARIABLES = X,Y,Z,SPI\nZONE N=%d,E=%d,F=FEPOINT,ET=TRIANGLE\n'%(coord.shape[0], elems.shape[0]))
    else:
        outfile.write('VARIABLES = X,Y,Z,SPI\nZONE N=%d,E=%d,F=FEPOINT,ET=TETRAHEDRON\n'%(coord.shape[0], elems.shape[0])) # Make sure to change to Triangle for WSS!!!
    for i in range(coord.shape[0]):
        outfile.write('% 16.12f % 16.12f % 16.12f % 16.12f\n'%
                        (coord[i,0],coord[i,1],coord[i,2],SPI[i]))
    for i in range(elems.shape[0]):
        c = elems[i]
        if dvp == "wss":
            outfile.write('\n%d %d %d'%(c[0]+1,c[1]+1,c[2]+1))
        else:
            outfile.write('\n%d %d %d %d'%(c[0]+1,c[1]+1,c[2]+1,c[3]+1)) # Need to add 1 because tecplot starts node numbering at 1
    outfile.close()   

def create_point_trace(formatted_data_folder, output_folder, point_ids,save_deg,time_between_files,start_t,dvp):

    # Get input data
    components_data = []
    component_names = ["mag","x","y","z"]
    for i in range(len(component_names)):
        if dvp == "p" and i>0:
            break
        file_str = dvp+"_"+component_names[i]+".npz"
        print(file_str)
        component_file = [file for file in os.listdir(formatted_data_folder) if file_str in file]
        component_data = np.load(formatted_data_folder+"/"+component_file[0])['component']
        components_data.append(component_data)


    # Create name for output file, define output path

    if dvp == "v":
        viz_type = 'velocity'
    elif dvp == "d":
        viz_type = 'displacement'
    elif dvp == "p":
        viz_type = 'pressure'
    else:
        print("Input d, v or p for dvp")

    num_ts = components_data[0].shape[1]   
    time_plot = np.arange(0.0, num_ts*time_between_files, time_between_files) 


    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists!')
    if not os.path.exists(output_folder):
        print("creating output folder")
        os.makedirs(output_folder)


    for point_id in point_ids:

        output_string = viz_type+"_point_id_"+str(point_id) # Base filename
        if dvp != "p":
            output_data = np.zeros((num_ts, 5))
        else:
            output_data = np.zeros((num_ts, 2))

        output_data[:,0] = time_plot
        output_data[:,1] = components_data[0][point_id,:]
        if dvp != "p":
            output_data[:,2] = components_data[1][point_id,:]
            output_data[:,3] = components_data[2][point_id,:]
            output_data[:,4] = components_data[3][point_id,:]

        point_trace_file = output_folder+'/'+output_string+'.csv' # file name for point trace
        point_trace_graph_file = output_folder+'/'+output_string+'.png' 

        if dvp != "p":
            np.savetxt(point_trace_file, output_data, delimiter=",", header="time (s), Magnitude, X Component, Y Component, Z Component")
        else:
            np.savetxt(point_trace_file, output_data, delimiter=",", header="time (s), Magnitude")

        # Plot and Save
        plt.plot(output_data[:,0],output_data[:,1],label="Mag")
        if dvp != "p":
            plt.plot(output_data[:,0],output_data[:,2],label="X")
            plt.plot(output_data[:,0],output_data[:,3],label="Y")
            plt.plot(output_data[:,0],output_data[:,4],label="Z")
        plt.title('Point # '+ str(point_id))
        if dvp == "p":
            plt.ylabel("Pressure (Pa) Not including 80-120 perfusion pressure")
        elif dvp == "v":
            plt.ylabel("Velocity (m/s)")
        elif dvp == "d":
            plt.ylabel("Displacement (m)")

        plt.xlabel('Simulation Time (s)')
        plt.legend()
        plt.savefig(point_trace_graph_file)  
        plt.close()


def create_domain_specific_viz(formatted_data_folder, output_folder, meshFile,save_deg,time_between_files,start_t,dvp,overwrite=False):

    # Get input data
    components_data = []
    component_names = ["mag","x","y","z"]
    for i in range(len(component_names)):
        if dvp == "p" and i>0:
            break
        file_str = dvp+"_"+component_names[i]+".npz"
        print(file_str)
        component_file = [file for file in os.listdir(formatted_data_folder) if file_str in file]
        component_data = np.load(formatted_data_folder+"/"+component_file[0])['component']
        components_data.append(component_data)


    # Create name for output file, define output path

    if dvp == "v":
        viz_type = 'velocity'
    elif dvp == "d":
        viz_type = 'displacement'
    elif dvp == "p":
        viz_type = 'pressure'
    else:
        print("Input d, v or p for dvp")

    viz_type = viz_type+"_save_deg_"+str(save_deg)
    output_file_name = viz_type+'.h5'  
    output_path = os.path.join(output_folder, output_file_name)  

    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists!')
    if not os.path.exists(output_folder):
        print("creating output folder")
        os.makedirs(output_folder)

    #read in the fsi mesh:
    fsi_mesh = h5py.File(meshFile,'r')

    # Count fluid and total nodes
    coordArrayFSI= fsi_mesh['mesh/coordinates'][:,:]
    topoArrayFSI= fsi_mesh['mesh/topology'][:,:]
    nNodesFSI = coordArrayFSI.shape[0]
    nElementsFSI = topoArrayFSI.shape[0]

    # Get fluid only topology
    fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    fluidIDs, wallIDs, allIDs = get_domain_ids(meshFile)
    coordArrayFluid= fsi_mesh['mesh/coordinates'][fluidIDs,:]
    nNodesFluid = len(fluidIDs)
    nElementsFluid = fluidTopology.shape[1]

    coordArraySolid= fsi_mesh['mesh/coordinates'][wallIDs,:]
    nNodesSolid = len(wallIDs)
    nElementsSolid = wallTopology.shape[1]
    # Get number of timesteps
    num_ts = components_data[0].shape[1]    

    if os.path.exists(output_path) and overwrite == False:
            print('File path {} exists; not overwriting. set overwrite = True to overwrite this file.'.format(output_path))

    else:
        # Remove old file path
        if os.path.exists(output_path):
            print('File path exists; rewriting')
            os.remove(output_path)

        # Create H5 file
        vectorData = h5py.File(output_path,'a')

        # Create mesh arrays
        # 1. update so that the fluid only nodes are used
        # Easiest way is just inputting the fluid-only mesh
        # harder way is modifying the topology of the mesh.. if an element contains a node that is in the solid, then don't include it? 
        # for save_deg = 2, maybe we can use fenics to create refined mesh with the fluid and solid elements noted?
        # hopefully that approach will yield the same node numbering as turtleFSI


        if dvp == "d":
            geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesSolid,3))
            geoArray[...] = coordArraySolid
            topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsSolid,4), dtype='i')

            # Fix Wall topology (need to renumber nodes consecutively so that dolfin can read the mesh)
            for node_id in range(nNodesSolid):
                wallTopology = np.where(wallTopology == wallIDs[node_id], node_id, wallTopology)
            topoArray[...] = wallTopology
            #print(wallTopology)

        else:
            geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesFluid,3))
            geoArray[...] = coordArrayFluid
            topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsFluid,4), dtype='i')

            # Fix Fluid topology
            for node_id in range(len(fluidIDs)):
                fluidTopology = np.where(fluidTopology == fluidIDs[node_id], node_id, fluidTopology)
            topoArray[...] = fluidTopology

        # 2. loop through elements and load in the df
        for idx in range(num_ts):
            ArrayName = 'VisualisationVector/' + str(idx)
            if dvp == "p":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFluid,1))
                v_array[:,0] = components_data[0][fluidIDs,idx]
                attType = "Scalar"

            elif dvp == "v":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFluid,3))
                v_array[:,0] = components_data[1][fluidIDs,idx]
                v_array[:,1] = components_data[2][fluidIDs,idx]
                v_array[:,2] = components_data[3][fluidIDs,idx]
                attType = "Vector"

            elif dvp == "d":
                v_array = vectorData.create_dataset(ArrayName, (nNodesSolid,3))
                v_array[:,0] = components_data[1][wallIDs,idx]
                v_array[:,1] = components_data[2][wallIDs,idx]
                v_array[:,2] = components_data[3][wallIDs,idx]
                attType = "Vector"

            else:
                print("ERROR, input dvp")

        vectorData.close() 

        # 3 create xdmf so that we can visualize
        if dvp == "d":
            create_xdmf_file(num_ts,time_between_files,start_t,nElementsSolid,nNodesSolid,attType,viz_type,output_folder)

        else:
            create_xdmf_file(num_ts,time_between_files,start_t,nElementsFluid,nNodesFluid,attType,viz_type,output_folder)

def reduce_save_deg_viz(formatted_data_folder, output_folder, meshFile,save_deg,time_between_files,start_t,dvp,overwrite=False):

    # Get input data
    components_data = []
    component_names = ["mag","x","y","z"]
    for i in range(len(component_names)):
        if dvp == "p" and i>0:
            break
        file_str = dvp+"_"+component_names[i]+".npz"
        print(file_str)
        component_file = [file for file in os.listdir(formatted_data_folder) if file_str in file]
        component_data = np.load(formatted_data_folder+"/"+component_file[0])['component']
        components_data.append(component_data)


    # Create name for output file, define output path

    if dvp == "v":
        viz_type = 'velocity'
    elif dvp == "d":
        viz_type = 'displacement'
    elif dvp == "p":
        viz_type = 'pressure'
    else:
        print("Input d, v or p for dvp")

    viz_type = viz_type+"_save_deg_"+str(save_deg)
    output_file_name = viz_type+'.h5'  
    output_path = os.path.join(output_folder, output_file_name)  

    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists!')
    if not os.path.exists(output_folder):
        print("creating output folder")
        os.makedirs(output_folder)

    #read in the fsi mesh:
    fsi_mesh = h5py.File(meshFile,'r')

    # Count fluid and total nodes
    coordArrayFSI= fsi_mesh['mesh/coordinates'][:,:]
    topoArrayFSI= fsi_mesh['mesh/topology'][:,:]
    nNodesFSI = coordArrayFSI.shape[0]
    nElementsFSI = topoArrayFSI.shape[0]

    # Get fluid only topology
    fluidIDs, wallIDs, allIDs = get_domain_ids(meshFile)

    # Get number of timesteps
    num_ts = components_data[0].shape[1]    

    if os.path.exists(output_path) and overwrite == False:
            print('File path {} exists; not overwriting. set overwrite = True to overwrite this file.'.format(output_path))

    else:
        # Remove old file path
        if os.path.exists(output_path):
            print('File path exists; rewriting')
            os.remove(output_path)
        # Create H5 file
        vectorData = h5py.File(output_path,'a')
    
        # Create mesh arrays
        geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesFSI,3))
        geoArray[...] = coordArrayFSI
        topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsFSI,4), dtype='i')
        topoArray[...] = topoArrayFSI
    
    
        # 2. loop through elements and load in the df
        for idx in range(num_ts):
            ArrayName = 'VisualisationVector/' + str(idx)
            if dvp == "p":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,1))
                v_array[:,0] = components_data[0][allIDs,idx]
                attType = "Scalar"
    
            else:
                v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,3))
                v_array[:,0] = components_data[1][allIDs,idx]
                v_array[:,1] = components_data[2][allIDs,idx]
                v_array[:,2] = components_data[3][allIDs,idx]
                attType = "Vector"
    
    
        vectorData.close() 
    
        # 3 create xdmf so that we can visualize
        create_xdmf_file(num_ts,time_between_files,start_t,nElementsFSI,nNodesFSI,attType,viz_type,output_folder)



def create_hi_pass_viz(formatted_data_folder, output_folder, meshFile,time_between_files,start_t,dvp,lowcut=0,highcut=100000,amplitude=False,filter_type="bandpass",pass_stop_list=[],overwrite=False):

    # Get input data
    components_data = []

    if dvp == "d" or dvp == "v":
        component_names = ["mag","x","y","z"]
    elif dvp == "strain":
        component_names = ["11","12","22","23","33","31"]
    else:
        component_names = ["mag"]

    for i in range(len(component_names)):
        file_str = dvp+"_"+component_names[i]+".npz"
        print("Opened file: " + file_str)
        component_file = [file for file in os.listdir(formatted_data_folder) if file_str in file]
        component_data = np.load(formatted_data_folder+"/"+component_file[0])['component']
        components_data.append(component_data)


    # Create name for output file, define output path

    if dvp == "v":
        viz_type = 'velocity'
    elif dvp == "d":
        viz_type = 'displacement'
    elif dvp == "p":
        viz_type = 'pressure'
    elif dvp == "wss":
        viz_type = 'WallShearStress'
    elif dvp == "mps":
        viz_type = 'MaxPrincipalStrain'
    elif dvp == 'strain':
        viz_type = 'InfinitesimalStrain' # Change
    else:
        print('input d, v, p, mps, strain or wss for dvp')

    if amplitude==True:
        viz_type=viz_type+"_amplitude"

    if filter_type=="multiband":
        for lowfreq, highfreq, pass_stop in zip(lowcut,highcut,pass_stop_list):
            viz_type = viz_type+"_"+pass_stop+"_"+str(int(np.rint(lowfreq)))+"_to_"+str(int(np.rint(highfreq)))

    else:
        viz_type = viz_type+"_"+str(int(np.rint(lowcut)))+"_to_"+str(int(np.rint(highcut)))
    output_file_name = viz_type+'.h5'  
    output_path = os.path.join(output_folder, output_file_name)  

    # Create output directory
    if os.path.exists(output_folder):
        print('Path exists!')
    if not os.path.exists(output_folder):
        print("creating output folder")
        os.makedirs(output_folder)

    #read in the mesh (for mps this needs to be the wall only mesh):
    fsi_mesh = h5py.File(meshFile,'r')

    # Count fluid and total nodes
    coordArrayFSI= fsi_mesh['mesh/coordinates'][:,:]
    topoArrayFSI= fsi_mesh['mesh/topology'][:,:]

    nNodesFSI = coordArrayFSI.shape[0]
    nElementsFSI = topoArrayFSI.shape[0]

    ## Get fluid only topology
    #fluidTopology, wallTopology, allTopology = get_domain_topology(meshFile)
    #fluidIDs, wallIDs, allIDs = get_domain_ids(meshFile)
    #coordArrayFluid= fsi_mesh['mesh/coordinates'][fluidIDs,:]
    #print(allTopology)
    #print(topoArrayFSI)

    # Get number of timesteps
    num_ts = components_data[0].shape[1]    

    if os.path.exists(output_path) and overwrite == False:
            print('File path {} exists; not overwriting. set overwrite = True to overwrite this file.'.format(output_path))

    else:
        # Remove old file path
        if os.path.exists(output_path):
            print('File path exists; rewriting')
            os.remove(output_path)

        # Create H5 file
        vectorData = h5py.File(output_path,'a')
    
        # Create mesh arrays
        # 1. update so that the fluid only nodes are used
        # Easiest way is just inputting the fluid-only mesh
        # harder way is modifying the topology of the mesh.. if an element contains a node that is in the solid, then don't include it? 
        # for save_deg = 2, maybe we can use fenics to create refined mesh with the fluid and solid elements noted?
        # hopefully that approach will yield the same node numbering as turtleFSI
    
        geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesFSI,3))
        geoArray[...] = coordArrayFSI
        topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsFSI,4), dtype='i')
        topoArray[...] = topoArrayFSI
    
        print("Filtering data...")
        for idy in range(nNodesFSI):
            if idy%1000 == 0:
                print("... {} filtering".format(filter_type))
            
            if filter_type=="multiband":  # loop through the bands and either bandpass or bandstop filter them
                for lowfreq, highfreq, pass_stop in zip(lowcut,highcut,pass_stop_list):   
                    for ic in range(len(component_names)):
                        components_data[ic][idy,:] = spec.butter_bandpass_filter(components_data[ic][idy,:], lowcut=lowfreq, highcut=highfreq, fs=int(1/time_between_files)-1,btype=pass_stop)
    
            else:
                f_crit = int(1/time_between_files)/2 - 1
                if highcut >=f_crit:
                    highcut = f_crit
                if lowcut < 0.1:
                    btype="lowpass"
                else:
                    btype="bandpass"
                for ic in range(len(component_names)):
                    components_data[ic][idy,:] = spec.butter_bandpass_filter(components_data[ic][idy,:], lowcut=lowcut, highcut=highcut, fs=int(1/time_between_files)-1,btype=btype)
    
        # if amplitude is selected, calculate moving RMS amplitude for the results                
        if amplitude==True:
            RMS_Magnitude = np.zeros((nNodesFSI,num_ts))
            #window_size = int((1/lowcut)/time_between_files + 1) # this is the initial guess
            window_size = 250 # this is ~1/4 the value used in the spectrograms (992) ...     
            for idy in range(nNodesFSI):
                if idy%1000 == 0:
                    print("... calculating amplitude")
                for ic in range(len(component_names)):
                    components_data[ic][idy,:] = window_rms(components_data[ic][idy,:],window_size) # For pressure
    
    
        # 2. loop through elements and load in the df
        for idx in range(num_ts):
            ArrayName = 'VisualisationVector/' + str(idx)
            if dvp == "p" or dvp =="wss" or dvp =="mps":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,1))
                v_array[:,0] = components_data[0][:,idx]
                attType = "Scalar"
                if amplitude==True:
                    RMS_Magnitude[:,idx] = components_data[0][:,idx] # Pressure is a scalar
    
            elif dvp == "strain":
                v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,9))
                v_array[:,0] = components_data[0][:,idx] # 11
                v_array[:,1] = components_data[1][:,idx] # 12
                v_array[:,2] = components_data[5][:,idx] # 31
                v_array[:,3] = components_data[1][:,idx] # 12
                v_array[:,4] = components_data[2][:,idx] # 22
                v_array[:,5] = components_data[3][:,idx] # 23
                v_array[:,6] = components_data[5][:,idx] # 31
                v_array[:,7] = components_data[3][:,idx] # 23
                v_array[:,8] = components_data[4][:,idx] # 33
                attType = "Tensor"
                if amplitude==True:
                    # Just print out dummy number for now.
                    #RMS_Magnitude[:,idx] = components_data[0][:,idx] 
    
                    print("calculating eigenvalues for ts #" + str(idx))
                    for iel in range(nNodesFSI):
                        #print("Before creating Strain Tensor: " + str(time.perf_counter()))
                        Strain_Tensor = np.array([[components_data[0][iel,idx],components_data[1][iel,idx],components_data[5][iel,idx]], [components_data[1][iel,idx],components_data[2][iel,idx],components_data[3][iel,idx]] ,[components_data[5][iel,idx],components_data[3][iel,idx],components_data[4][iel,idx]]])
                        #print("After creating Strain Tensor: " + str(time.perf_counter()))
                        if (np.abs(Strain_Tensor) < 1e-8).all():  # This is a shortcut to avoid taking eignevalues if the Strain tensor is all zeroes (outside the FSI region)
                            MPS = 0.0
                        else:
                            #print(Strain_Tensor)
                            MPS = get_eig(Strain_Tensor) # Instead of Magnitude, we take Maximum Principal Strain
                        #print(MPS)
                        #print("After calculating eigenvalues: " + str(time.perf_counter()))
                        RMS_Magnitude[iel,idx] = MPS  
                        #print("After assigning MPS: " + str(time.perf_counter()))
    
            else:
                v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,3))
                v_array[:,0] = components_data[1][:,idx]
                v_array[:,1] = components_data[2][:,idx]
                v_array[:,2] = components_data[3][:,idx]
                attType = "Vector"
                if amplitude==True:
                    RMS_Magnitude[:,idx] = LA.norm(v_array, axis=1)  # Take magnitude of RMS Amplitude, this way you don't lose any directional changes
    
        vectorData.close() 
    
        # 3 create xdmf so that we can visualize
        create_xdmf_file(num_ts,time_between_files,start_t,nElementsFSI,nNodesFSI,attType,viz_type,output_folder)

        # if amplitude is selected, save the percentiles of magnitude of RMS amplitude to file              
        if amplitude==True:
            # 3 save average amplitude, 95th percentile amplitude, max amplitude
            output_amplitudes = np.zeros((num_ts, 13))
            for idx in range(num_ts):
                output_amplitudes[idx,0] = idx*time_between_files
                output_amplitudes[idx,1] = np.percentile(RMS_Magnitude[:,idx],95)
                output_amplitudes[idx,2] = np.percentile(RMS_Magnitude[:,idx],5)
                output_amplitudes[idx,3] = np.percentile(RMS_Magnitude[:,idx],100)
                output_amplitudes[idx,4] = np.percentile(RMS_Magnitude[:,idx],0)
                output_amplitudes[idx,5] = np.percentile(RMS_Magnitude[:,idx],50)
                output_amplitudes[idx,6] = np.percentile(RMS_Magnitude[:,idx],90)
                output_amplitudes[idx,7] = np.percentile(RMS_Magnitude[:,idx],10)
                output_amplitudes[idx,8] = np.percentile(RMS_Magnitude[:,idx],97.5)
                output_amplitudes[idx,9] = np.percentile(RMS_Magnitude[:,idx],2.5)
                output_amplitudes[idx,10] = np.percentile(RMS_Magnitude[:,idx],99)
                output_amplitudes[idx,11] = np.percentile(RMS_Magnitude[:,idx],1)
                output_amplitudes[idx,12] = np.argmax(RMS_Magnitude[:,idx])
    
            amp_file = output_folder+'/'+viz_type+'.csv' # file name for amplitudes
            amp_graph_file = output_folder+'/'+viz_type+'.png' # file name for amplitudes
  
            np.savetxt(amp_file, output_amplitudes, delimiter=",", header="time (s), 95th percentile amplitude, 5th percentile amplitude, maximum amplitude, minimum amplitude, average amplitude, 90th percentile amplitude, 10th percentile amplitude, 97.5th percentile amplitude, 2.5th percentile amplitude, 99th percentile amplitude, 1st percentile amplitude, ID of node with max amplitude")
            # Plot and Save
            plt.plot(output_amplitudes[:,0],output_amplitudes[:,3],label="Maximum amplitude")
            plt.plot(output_amplitudes[:,0],output_amplitudes[:,1],label="95th percentile amplitude")
            plt.plot(output_amplitudes[:,0],output_amplitudes[:,5],label="50th percentile amplitude")
            plt.title('Amplitude Percentiles')
            plt.ylabel('Amplitude (units depend on d, v or p)')
            plt.xlabel('Simulation Time (s)')
            plt.legend()
            plt.savefig(amp_graph_file)  
            plt.close()
    
     
            if attType == "Tensor":
    
                # Save MPS amplitude to file
                # Remove old file path
                viz_type = viz_type.replace("InfinitesimalStrain","MaxPrincipalHiPassStrain")
                output_file_name = viz_type+'.h5'  
                output_path = os.path.join(output_folder, output_file_name) 
                if os.path.exists(output_path):
                    print('File path exists; rewriting')
                    os.remove(output_path)
    
                # Create H5 file
                vectorData = h5py.File(output_path,'a')
                geoArray = vectorData.create_dataset("Mesh/0/mesh/geometry", (nNodesFSI,3))
                geoArray[...] = coordArrayFSI
                topoArray = vectorData.create_dataset("Mesh/0/mesh/topology", (nElementsFSI,4), dtype='i')
                topoArray[...] = topoArrayFSI
    
                # 2. loop through elements and load in the df
                for idx in range(num_ts):
                    ArrayName = 'VisualisationVector/' + str(idx)
                    v_array = vectorData.create_dataset(ArrayName, (nNodesFSI,1))
                    v_array[:,0] = RMS_Magnitude[:,idx]
                    attType = "Scalar"
                    
                vectorData.close() 
                
                # 3 create xdmf so that we can visualize
                create_xdmf_file(num_ts,time_between_files,start_t,nElementsFSI,nNodesFSI,attType,viz_type,output_folder)



def window_rms(a, window_size,window_type="flat"): # Changed from "flat" to "blackmanharris"

    # This function takes the windowed RMS amplitude of a signal (a), to estimate the varying amplitude over time. Window size is given in number of timesteps
    # https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
    a2 = np.power(a,2)
    if window_type == "flat":
        window = np.ones(window_size)/float(window_size)
    elif window_type == "tukey":
        window = signal.windows.tukey(window_size)/float(window_size)
    elif window_type == "hann":
        window = signal.windows.hann(window_size)/float(window_size)
    elif window_type == "blackmanharris":
        window = signal.windows.blackmanharris(window_size)/float(window_size)
    elif window_type == "flattop":
        window = signal.windows.blackmanharris(window_size)/float(window_size)
    else:
        print('Did not recognize window type, using flat')
        window = np.ones(window_size)/float(window_size)
    # https://stackoverflow.com/questions/47484899/moving-average-produces-array-of-different-length
    RMS = np.sqrt(np.convolve(a2, window,mode="valid"))
    len_RMS = len(RMS)
    len_a2 = len(a2)
    pad_length = int((len_a2-len_RMS)/2)
    RMS_padded = np.zeros(len_a2)
    for i in range(len_a2):
        if i>=pad_length and i < len_RMS+pad_length:
            RMS_padded[i] = RMS[i-pad_length]

    #print("used new 2 amplitude method")
    #print(len_RMS)
    #print(len_a2)

    #np.sqrt(np.convolve(a2, window, mode='same'))
    return RMS_padded #np.sqrt(np.convolve(a2, window, mode='same'))


def create_xdmf_file(num_ts,time_between_files,start_t,nElements,nNodes,attType,viz_type,output_folder):

    # Create strings
    num_el = str(nElements)
    num_nodes = str(nNodes)
    if attType == "Scalar":
        nDim = '1'  
    elif attType == "Tensor":
        nDim = '9'  
    elif attType == "Vector":
        nDim = '3'
    else:
        print("Attribute type (Scalar, Vector or Tensor) not given! can't make xdmf file")  
    
    # Write lines of xdmf file
    lines = []
    lines.append('<?xml version="1.0"?>\n')
    lines.append('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
    lines.append('<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
    lines.append('  <Domain>\n')
    lines.append('    <Grid Name="TimeSeries_'+viz_type+'" GridType="Collection" CollectionType="Temporal">\n')
    lines.append('      <Grid Name="mesh" GridType="Uniform">\n')
    lines.append('        <Topology NumberOfElements="'+num_el+'" TopologyType="Tetrahedron" NodesPerElement="4">\n')
    lines.append('          <DataItem Dimensions="'+num_el+' 4" NumberType="UInt" Format="HDF">'+viz_type+'.h5:/Mesh/0/mesh/topology</DataItem>\n')
    lines.append('        </Topology>\n')
    lines.append('        <Geometry GeometryType="XYZ">\n')
    lines.append('          <DataItem Dimensions="'+num_nodes+' 3" Format="HDF">'+viz_type+'.h5:/Mesh/0/mesh/geometry</DataItem>\n')
    lines.append('        </Geometry>\n')

    for idx in range(num_ts):
        time_value = str(idx*time_between_files+start_t)
        lines.append('        <Time Value="'+time_value+'" />\n')
        lines.append('        <Attribute Name="'+viz_type+'" AttributeType="'+attType+'" Center="Node">\n')
        lines.append('          <DataItem Dimensions="'+num_nodes+' '+nDim+'" Format="HDF">'+viz_type+'.h5:/VisualisationVector/'+str(idx)+'</DataItem>\n')
        lines.append('        </Attribute>\n')
        lines.append('      </Grid>\n')
        if idx == num_ts-1:
            break
        lines.append('      <Grid> \n')
        lines.append('        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_'+viz_type+'&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />\n')

    lines.append('    </Grid>\n')
    lines.append('  </Domain>\n')
    lines.append('</Xdmf>\n')

    # writing lines to file 
    xdmf_path = output_folder+'/'+viz_type+'.xdmf'

    # Remove old file path
    if os.path.exists(xdmf_path):
        print('File path exists; rewriting')
        os.remove(xdmf_path)

    xdmf_file = open(xdmf_path, 'w') 
    xdmf_file.writelines(lines) 
    xdmf_file.close() 

def create_fixed_xdmf_file(time_values,nElements,nNodes,attType,viz_type,h5_file_list,output_folder):

    # Create strings
    num_el = str(nElements)
    num_nodes = str(nNodes)
    nDim = '1' if attType == "Scalar" else '3'
    
    # Write lines of xdmf file
    lines = []
    lines.append('<?xml version="1.0"?>\n')
    lines.append('<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>\n')
    lines.append('<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">\n')
    lines.append('  <Domain>\n')
    lines.append('    <Grid Name="TimeSeries_'+viz_type+'" GridType="Collection" CollectionType="Temporal">\n')
    lines.append('      <Grid Name="mesh" GridType="Uniform">\n')
    lines.append('        <Topology NumberOfElements="'+num_el+'" TopologyType="Tetrahedron" NodesPerElement="4">\n')
    lines.append('          <DataItem Dimensions="'+num_el+' 4" NumberType="UInt" Format="HDF">'+h5_file_list[0]+'.h5:/Mesh/0/mesh/topology</DataItem>\n')
    lines.append('        </Topology>\n')
    lines.append('        <Geometry GeometryType="XYZ">\n')
    lines.append('          <DataItem Dimensions="'+num_nodes+' 3" Format="HDF">'+h5_file_list[0]+'.h5:/Mesh/0/mesh/geometry</DataItem>\n')
    lines.append('        </Geometry>\n')

    h5_array_index = 0
    for idx, time_value in enumerate(time_values):
        # Zero the h5 array index if a timesteps come from the next h5 file
        if h5_file_list[idx] != h5_file_list[idx-1]:
            h5_array_index = 0
        lines.append('        <Time Value="'+str(time_value)+'" />\n')
        lines.append('        <Attribute Name="'+viz_type+'" AttributeType="'+attType+'" Center="Node">\n')
        lines.append('          <DataItem Dimensions="'+num_nodes+' '+nDim+'" Format="HDF">'+h5_file_list[idx]+'.h5:/VisualisationVector/'+str(h5_array_index)+'</DataItem>\n')
        lines.append('        </Attribute>\n')
        lines.append('      </Grid>\n')
        if idx == len(time_values)-1:
            break
        lines.append('      <Grid> \n')
        if attType == "Scalar":
            #lines.append('        <ns0:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_'+viz_type+'&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />\n')
            lines.append('        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_'+viz_type+'&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />\n')
        else:
            lines.append('        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_'+viz_type+'&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />\n')
        h5_array_index += 1

    lines.append('    </Grid>\n')
    lines.append('  </Domain>\n')
    lines.append('</Xdmf>\n')

    # writing lines to file 
    xdmf_path = output_folder+'/'+viz_type.lower()+'_fixed.xdmf'

    # Remove old file path
    if os.path.exists(xdmf_path):
        print('File path exists; rewriting')
        os.remove(xdmf_path)

    xdmf_file = open(xdmf_path, 'w') 
    xdmf_file.writelines(lines) 
    xdmf_file.close() 


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

def get_time_between_files(input_path, output_folder,meshFile, case_name, dvp,stride=1):
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
    time_between_files = time_ts[2] - time_ts[1] # Calculate the time between files from xdmf file

    return time_between_files



def get_eig(T):
########################################################################
# Method for the analytical calculation of eigenvalues for 3D-Problems #
# from: https://fenicsproject.discourse.group/t/hyperelastic-model-problems-on-plotting-stresses/3130/6
########################################################################
    '''
    Analytically calculate eigenvalues for a three-dimensional tensor T with a
    characteristic polynomial equation of the form

                lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0   .

    Since the characteristic polynomial is in its normal form , the eigenvalues
    can be determined using Cardanos formula. This algorithm is based on:
    "Efficient numerical diagonalization of hermitian 3 by 3 matrices" by
    J.Kopp, eqn 21-34, with coefficients: c2=-I1, c1 = I2, c3 = -I3).

    NOTE:
    The method implemented here, implicitly assumes that the polynomial has
    only real roots, since imaginary ones should not occur in this use case.

    In order to ensure eigenvalues with algebraic multiplicity of 1, the idea
    of numerical perturbations is adopted from "Computation of isotropic tensor
    functions" by C. Miehe (1993). Since direct comparisons with conditionals
    have proven to be very slow, not the eigenvalues but the coefficients
    occuring during the calculation of them are perturbated to get distinct
    values.
    '''

    # determine perturbation from tolerance
    #print("Begin Eig Loop: " + str(time.perf_counter()))
    tol1 = 1e-16
    pert1 = 2*tol1
    tol2 = 1e-24
    pert2 = 2*tol2
    tol3 = 1e-40
    pert3 = 2*tol3
    # get required invariants
    I1 = np.trace(T)   
    #print("I1 finished: " + str(time.perf_counter()))
                                                            # trace of tensor
    #print("I1 = "+str(I1))  
    I2 = 0.5*(np.trace(T)**2-np.tensordot(T,T))                                        # 2nd invariant of tensor
    #I2 = 0.5*(np.trace(T)**2-np.trace(np.dot(T,T.T)))                                 # 2nd invariant of tensor (equivalent)
    #print("I2 finished: " + str(time.perf_counter()))

    #print("I2 = "+str(I2))  
    I3 = np.linalg.det(T)                                                              # determinant of tensor
    #print("I3 = "+str(I3))  
    #print("I3 finished: " + str(time.perf_counter()))

    # determine terms p and q according to the paper
    # -> Follow the argumentation within the paper, to see why p must be
    # -> positive. Additionally ensure non-zero denominators to avoid problems
    # -> during the automatic differentiation
    p = I1**2 - 3*I2                                    # preliminary value for p
    if p < tol1:
        print("perturbation applied to p: p = "+str(p))
        p = np.abs(p)+pert1                                 # add numerical perturbation to p, if close to zero; ensure positiveness of p

    q = 27/2*I3 + I1**3 - 9/2*I1*I2                     # preliminary value for q
    if abs(q) < tol2:
        print("perturbation applied to q: q = "+str(q))
        q = q+np.sign(q)*pert2                           # add numerical perturbation (with sign) to value of q, if close to zero
    # sign returns -1 or +1 depending on sign of q

    # determine angle phi for calculation of roots
    phiNom2 =  27*( 1/4*I2**2*(p-I2) + I3*(27/4*I3-q) )  # preliminary value for squared nominator of expression for angle phi
    if phiNom2 < tol3: 
        print("perturbation applied to phiNom2: phiNom2 = "+str(phiNom2))

        phiNom2 = np.abs(phiNom2)+pert3                             # add numerical perturbation to ensure non-zero nominator expression for angle phi

    phi = 1/3*np.arctan2(np.sqrt(phiNom2),q)             # calculate angle phi
    #print("Conditional statements finished: " + str(time.perf_counter()))

    # calculate polynomial roots
    lambda1 = 1/3*(np.sqrt(p)*2*np.cos(phi)+I1)
    #lambda3 = 1/3*(-np.sqrt(p)*(np.cos(phi)+np.sqrt(3)*np.sin(phi))+I1)
    #lambda2 = 1/3*(-np.sqrt(p)*(np.cos(phi)-np.sqrt(3)*np.sin(phi))+I1)
    #print("all eigenvalues calculated finished: " + str(time.perf_counter()))

    # return polynomial roots (eigenvalues)
    #eig = as_tensor([[lambda1 ,0 ,0],[0 ,lambda2 ,0],[0 ,0 ,lambda3]])
    return lambda1 #, lambda2, lambda3



if __name__ == "__main__":
    print('See functions.')
