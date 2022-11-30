from argparse import ArgumentParser
import numpy as np
import h5py
import shutil
import os

def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--case', type=str, default="cyl_test", help="Path to simulation results",
                        metavar="PATH")
    parser.add_argument('--mesh', type=str, default="artery_coarse_rescaled", help="Mesh File Name",
                        metavar="PATH")

    args = parser.parse_args()

    return args.case, args.mesh

def get_domain_cells(meshFile):
    # This function obtains the cells for the fluid and solid in the input mesh
    vectorData = h5py.File(meshFile)
    domainsLoc = 'domains/values'
    domains = vectorData[domainsLoc][:] # Open domain array
    id_wall = np.array((domains>1).nonzero()).astype(int) # domain = 2 is the solid
    id_fluid = np.array((domains==1).nonzero()).astype(int) # domain = 1 is the fluid


    return id_wall, id_fluid

def get_output_file_cells(outFile,meshFile):
    # This function obtains the cells for the fluid and solid in the output file 
    fluidIDs, wallIDs, allIDs = get_domain_ids(meshFile) # Nodal IDs
    solidOnlyIDs = array3 = [ele for ele in wallIDs if ele not in fluidIDs] # solid nodal IDs not including interface
    fluidOnlyIDs = array3 = [ele for ele in fluidIDs if ele not in wallIDs] # fluid nodal IDs not including interface

    vectorData = h5py.File(outFile)
    domainsLoc = 'Mesh/0/mesh/topology'
    domains = vectorData[domainsLoc][:] # Open domain array

    solid_cells_bool = np.in1d(domains,solidOnlyIDs).reshape(domains.shape).any(axis=1)
    fluid_cells_bool = np.in1d(domains,fluidOnlyIDs).reshape(domains.shape).any(axis=1)

    all_cell_IDs = np.arange(domains.shape[0])
    fluid_cell_IDs = all_cell_IDs[~fluid_cells_bool]
    solid_cell_IDs = all_cell_IDs[~solid_cells_bool]



    return solid_cell_IDs, fluid_cell_IDs

def get_domain_topology(meshFile):
    # This function obtains the topology for the fluid, solid, and all elements of the input mesh
    vectorData = h5py.File(meshFile)
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

def fix_fluid_only_mesh(meshFile):
    # This function fixes the node numbering so that the numbers start at 0 and are continuous integers

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

    # Copy mesh file to new "fixed" file
    fluid_mesh_path =  meshFile.replace(".h5","_fluid_only.h5")    
    fluid_mesh_path_fixed =  meshFile.replace(".h5","_fluid_only_fixed.h5")    

    # Fix Fluid topology
    for node_id in range(len(fluidIDs)):
        fluidTopology = np.where(fluidTopology == fluidIDs[node_id], node_id, fluidTopology)

    shutil.copyfile(fluid_mesh_path, fluid_mesh_path_fixed)

    # Replace all the arrays in the "fixed" file with the correct node numbering and topology
    vectorData = h5py.File(fluid_mesh_path_fixed,'a')

    geoArray = vectorData["mesh/coordinates"]
    geoArray[...] = coordArrayFluid
    topoArray = vectorData["mesh/topology"]
    topoArray[...] = fluidTopology
    vectorData.close() 

    os.remove(fluid_mesh_path)
    os.rename(fluid_mesh_path_fixed,fluid_mesh_path)

def fix_solid_only_mesh(meshFile):


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
    coordArraySolid= fsi_mesh['mesh/coordinates'][wallIDs,:]

    # Copy mesh file to new "fixed" file
    solid_mesh_path =  meshFile.replace(".h5","_solid_only.h5")    
    solid_mesh_path_fixed =  meshFile.replace(".h5","_solid_only_fixed.h5")    

    # Fix Wall topology
    for node_id in range(len(wallIDs)):
        wallTopology = np.where(wallTopology == wallIDs[node_id], node_id, wallTopology)

    shutil.copyfile(solid_mesh_path, solid_mesh_path_fixed)

    # Replace all the arrays in the "fixed" file with the correct node numbering and topology
    vectorData = h5py.File(solid_mesh_path_fixed,'a')

    geoArray = vectorData["mesh/coordinates"]
    geoArray[...] = coordArraySolid
    topoArray = vectorData["mesh/topology"]
    topoArray[...] = wallTopology
    vectorData.close() 

    os.remove(solid_mesh_path)
    os.rename(solid_mesh_path_fixed,solid_mesh_path)