import numpy as np
import h5py
import shutil
import os 

import common_meshing

from dolfin import *

# This is required to get the boundary refinement to work
parameters["refinement_algorithm"] = "plaza_with_parent_facets"

"""
This script creates a refined mesh with domain markers from a specified mesh. However, the node numbering may not be the 
same as the output file with save_deg = 2, so the node numbering is corrected to match the output file
the output files. Currently, it only runs in serial (not parallel) due to the "adapt" function used in fenics. 
This mesh is later used in  the "postprocessing_h5" and "postprocessing_fenics" scripts.
See: https://fenicsproject.discourse.group/t/why-are-boundary-and-surface-markers-not-carried-over-to-the-refined-mesh/5822/2
   TO DO:
   -Add boundary creation other meshing scripts (look into "adapt()" for boundaries)
   -Add domain creation in other meshing scripts

folder, mesh_name = common_meshing.read_command_line()
mesh_path = os.path.join(folder,"mesh",mesh_name +".h5")
"""

def main():

    # find the visualization path (used to determine correct node numbering)
    for file in os.listdir(folder):
        file_path = os.path.join(folder, file)
        if os.path.exists(os.path.join(file_path, "1")):
            visualization_path = os.path.join(file_path, "1/Visualization")
        elif os.path.exists(os.path.join(file_path, "Visualization")):
            visualization_path = os.path.join(file_path, "Visualization")

    # Read in original FSI mesh
    mesh = Mesh()
    hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
    hdf.read(mesh, "/mesh", False)

    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains") 
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")

    # Refine Mesh and domains
    mesh_refine = refine(mesh)
    domains_refine = adapt(domains,mesh_refine)
    boundaries_refine = adapt(boundaries,mesh_refine) # This doesnt wrk for some reason... 

    # Create path for refined mesh
    refined_mesh_path = mesh_path.replace(".h5","_refined.h5")

    # Save refined mesh
    hdf = HDF5File(mesh_refine.mpi_comm(), refined_mesh_path, "w")
    hdf.write(mesh_refine, "/mesh")
    hdf.write(domains_refine, "/domains")
    hdf.write(boundaries_refine, "/boundaries")

    print("Refined mesh saved to:")
    print(refined_mesh_path)    

    hdf.close()


    """ -----------------------------------------------------------
    ----------------2. Correct node numbering-----------------------
    ---------------------------------------------------------------
    """

    # Define mesh Paths (The refined mesh with domains but incorrect numbering is called "wrongNumberMesh", 
    # the mesh contained in the output velocity.h5 file but no domains is "correctNumberMesh")
    wrongNumberMeshPath = mesh_path.replace(".h5","_refined.h5")
    correctNumberMeshPath = os.path.join(visualization_path, 'velocity.h5')

    # Open the mesh files using h5py
    wrongNumberMesh = h5py.File(wrongNumberMeshPath)
    correctNumberMesh = h5py.File(correctNumberMeshPath)

    # Read in nodal coordinates
    wrongNumberNodes = wrongNumberMesh['mesh/coordinates'][:]
    correctNumberNodes = correctNumberMesh['Mesh/0/mesh/geometry'][:]

    # Define an array of indices
    index = np.zeros((wrongNumberNodes.shape[0],1))
    for i in range(wrongNumberNodes.shape[0]):
        index[i,0] = i

    # Append index to node array
    wrongNumberNodes=np.append(index, wrongNumberNodes, axis=1)
    correctNumberNodes=np.append(index, correctNumberNodes, axis=1)

    # Sort both nodal arrays by all 3 nodal coordinates. This gives us the mapping between both the wrong and correct node numbering scheme
    indWrong = np.lexsort((wrongNumberNodes[:, 1],wrongNumberNodes[:, 2],wrongNumberNodes[:, 3]))
    indCorrect = np.lexsort((correctNumberNodes[:, 1],correctNumberNodes[:, 2],correctNumberNodes[:, 3]))

    orederedWrongNodes = wrongNumberNodes[indWrong]
    orederedCorrectNodes =correctNumberNodes[indCorrect]
    
    # orderedIndexMap is an array with the nodal index mapping, sorted by the "wrong" node numbering scheme
    indexMap = np.append(orederedWrongNodes,orederedCorrectNodes, axis=1)
    orderedIndexMap = indexMap[indexMap[:, 0].argsort()]
    orderedIndexMap = orderedIndexMap[:,[0,4]]

    # wrongNumberTopology is the topology from the "wrong" numbered mesh. We will modify this array to change it to the correct node numbering scheme
    wrongNumberTopology = wrongNumberMesh['mesh/topology'][:]

    # this loop replaces the node numbers in the topology array one by one
    for row in range(wrongNumberTopology.shape[0]):
        for column in range(wrongNumberTopology.shape[1]):
            wrongNumberTopology[row,column] = np.rint(orderedIndexMap[wrongNumberTopology[row,column],1])

    # wrongNumberBdTopology is the boundary topology from the "wrong" numbered mesh. We will modify this array to change it to the correct node numbering scheme
    wrongNumberBdTopology = wrongNumberMesh['boundaries/topology'][:]

    # this loop replaces the node numbers in the boundaries topology array one by one
    for row in range(wrongNumberBdTopology.shape[0]):
        for column in range(wrongNumberBdTopology.shape[1]):
            wrongNumberBdTopology[row,column] = np.rint(orderedIndexMap[wrongNumberBdTopology[row,column],1])

    # Fix boundary values (set any spurious boundary numbers to 0)
    wrongNumberBdValues= wrongNumberMesh['boundaries/values'][:]
    for row in range(wrongNumberBdValues.shape[0]):
        if wrongNumberBdValues[row] > 33:
            wrongNumberBdValues[row] = 0

    #wrongNumberMesh.close() 

    # Copy mesh file to new "fixed" file
    output_path =  mesh_path.replace(".h5","_refined_fixed.h5")    
    shutil.copyfile(wrongNumberMeshPath, output_path)

    # Replace all the arrays in the "fixed" file with the correct node numbering and topology
    vectorData = h5py.File(output_path,'a')
    vectorArray = vectorData["domains/coordinates"]
    vectorArray[...] = correctNumberNodes[:,[1,2,3]]
    vectorArray = vectorData["boundaries/coordinates"]
    vectorArray[...] = correctNumberNodes[:,[1,2,3]]
    vectorArray = vectorData["mesh/coordinates"]
    vectorArray[...] = correctNumberNodes[:,[1,2,3]]
    vectorArray = vectorData["domains/topology"]
    vectorArray[...] = wrongNumberTopology
    vectorArray = vectorData["boundaries/topology"]
    vectorArray[...] = wrongNumberBdTopology
    vectorArray = vectorData["mesh/topology"]
    vectorArray[...] = wrongNumberTopology
    vectorArray = vectorData["boundaries/values"]
    vectorArray[...] = wrongNumberBdValues
    vectorData.close() 

    os.remove(refined_mesh_path)
    os.rename(output_path,refined_mesh_path)

    print("Fixed refined mesh wih correct node numbering. Mesh saved to:")
    print(refined_mesh_path)

if __name__ == "__main__":
    main()