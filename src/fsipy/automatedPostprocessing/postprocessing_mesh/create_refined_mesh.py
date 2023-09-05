# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script creates a refined mesh with domain markers from a specified mesh. However, the node numbering may not be the
same as the output file with save_deg = 2, so the node numbering is corrected to match the output files.
Currently, it only runs in serial (not parallel) due to the "adapt" function used in fenics.
This mesh is later used in  the "postprocessing_h5" and "postprocessing_fenics" scripts.
See:
https://fenicsproject.discourse.group/t/why-are-boundary-and-surface-markers-not-carried-over-to-the-refined-mesh/5822/2
   TO DO:
   -Add boundary creation other meshing scripts (look into "adapt()" for boundaries)
   -Add domain creation in other meshing scripts
"""

import numpy as np
import h5py
from pathlib import Path

from fsipy.automatedPostprocessing.postprocessing_mesh import postprocessing_mesh_common

from dolfin import MPI, Mesh, MeshFunction, HDF5File, refine, adapt, parameters

# This is required to get the boundary refinement to work
parameters["refinement_algorithm"] = "plaza_with_parent_facets"


def create_refined_mesh(folder_path: Path, mesh_path: Path) -> None:
    """
    args:
        folder_path (Path): Path to the simulation results folder.
        mesh_path (Path): Path to the mesh file.

    Returns:
        None
    """
    # Read in original FSI mesh
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(mesh_path), "r") as hdf:
        hdf.read(mesh, "/mesh", False)
        domains = MeshFunction("size_t", mesh, mesh.topology().dim())
        hdf.read(domains, "/domains")
        boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
        hdf.read(boundaries, "/boundaries")

    # Create path for refined mesh
    refined_mesh_path = mesh_path.with_name(mesh_path.stem + "_refined.h5")

    if refined_mesh_path.exists():
        print(f"--- Refined mesh already exists: {refined_mesh_path} \n")
        print("--- Skipping mesh refinement \n")
    else:
        # Refine mesh and carry over domain and boundary markers using "adapt()" function
        refined_mesh = refine(mesh)
        refined_domains = adapt(domains, refined_mesh)
        # This doesnt wrk for some reason... comment by David Bruneau
        refined_boundaries = adapt(boundaries, refined_mesh)

        # Save refined mesh
        with HDF5File(refined_mesh.mpi_comm(), str(refined_mesh_path), "w") as hdf:
            hdf.write(refined_mesh, "/mesh")
            hdf.write(refined_domains, "/domains")
            hdf.write(refined_boundaries, "/boundaries")

        print(f"--- Refined mesh saved to: {refined_mesh_path} \n")

    print("--- Correcting node numbering in refined mesh \n")
    # Define mesh Paths (The refined mesh with domains but incorrect numbering is called "wrongNumberMesh",
    # the mesh contained in the output velocity.h5 file but no domains is "correctNumberMesh")
    correctNumberMeshPath = folder_path / "Visualization" / "velocity.h5"

    # Open the mesh files using h5py
    with h5py.File(refined_mesh_path, "r") as wrongNumberMesh, \
            h5py.File(correctNumberMeshPath, "r") as correctNumberMesh:

        # Read in nodal coordinates
        wrongNumberNodes = wrongNumberMesh['mesh/coordinates'][:]
        correctNumberNodes = correctNumberMesh['Mesh/0/mesh/geometry'][:]

        if (wrongNumberNodes == correctNumberNodes).all():
            print('--- Node numbering is already correct. Exiting ... \n')
            return
        else:
            print('--- Node numbering is incorrect between the refined mesh and the output velocity.h5 file \n')

        # add index to the node coordinates
        wrongNumberNodes = np.hstack((np.arange(len(wrongNumberNodes), dtype=int).reshape(-1, 1),
                                      wrongNumberNodes))
        correctNumberNodes = np.hstack((np.arange(len(correctNumberNodes), dtype=int).reshape(-1, 1),
                                        correctNumberNodes))

        # Sort both nodal arrays by all 3 nodal coordinates. (if x is unique, sort by x, else sort by x, y, z)
        # This gives us the mapping between both the wrong and correct node numbering scheme
        print('--- Sorting node coordinates \n')
        if correctNumberNodes[:, 1].size == np.unique(correctNumberNodes[:, 1]).size:
            print('x coordinate is unique and sort based on x coordinate only \n')
            indWrong = np.argsort(wrongNumberNodes[:, 1])
            indCorrect = np.argsort(correctNumberNodes[:, 1])
        else:
            print('x coordinate is not unique and sort based on x, y, and z coordinates \n')
            indWrong = np.lexsort((wrongNumberNodes[:, 1], wrongNumberNodes[:, 2], wrongNumberNodes[:, 3]))
            indCorrect = np.lexsort((correctNumberNodes[:, 1], correctNumberNodes[:, 2], correctNumberNodes[:, 3]))

        orederedWrongNodes = wrongNumberNodes[indWrong]
        orederedCorrectNodes = correctNumberNodes[indCorrect]

        # orderedIndexMap is an array with the nodal index mapping, sorted by the "wrong" node numbering scheme
        indexMap = np.append(orederedWrongNodes, orederedCorrectNodes, axis=1)
        orderedIndexMap = indexMap[indexMap[:, 0].argsort()]
        orderedIndexMap = orderedIndexMap[:, [0, 4]]

        # wrongNumberTopology is the topology from the "wrong" numbered mesh.
        # We will modify this array to change it to the correct node numbering scheme
        wrongNumberTopology = wrongNumberMesh['mesh/topology'][:]

        # This loop replaces the node numbers in the topology array one by one
        print('--- Correcting node numbering of the topology array in the refined mesh \n')
        for row in range(wrongNumberTopology.shape[0]):
            for column in range(wrongNumberTopology.shape[1]):
                wrongNumberTopology[row, column] = np.rint(orderedIndexMap[wrongNumberTopology[row, column], 1])

        # wrongNumberBdTopology is the boundary topology from the "wrong" numbered mesh.
        # We will modify this array to change it to the correct node numbering scheme
        wrongNumberBdTopology = wrongNumberMesh['boundaries/topology'][:]

        # this loop replaces the node numbers in the boundaries topology array one by one
        print('--- Correcting node numbering of the boundary topology array in the refined mesh \n')
        for row in range(wrongNumberBdTopology.shape[0]):
            for column in range(wrongNumberBdTopology.shape[1]):
                wrongNumberBdTopology[row, column] = np.rint(orderedIndexMap[wrongNumberBdTopology[row, column], 1])

        # Fix boundary values (set any spurious boundary numbers to 0)
        print('--- Correcting boundary values in the refined mesh \n')
        wrongNumberBdValues = wrongNumberMesh['boundaries/values'][:]
        for row in range(wrongNumberBdValues.shape[0]):
            if wrongNumberBdValues[row] > 33:
                wrongNumberBdValues[row] = 0

        # Copy mesh file to new "fixed" file
        output_path = mesh_path.with_name(mesh_path.stem + "_refined_fixed.h5")
        output_path.write_bytes(refined_mesh_path.read_bytes())

        # Replace all the arrays in the "fixed" file with the correct node numbering and topology
        print("--- Saving the corrected node numbering to the refined mesh \n")
        with h5py.File(output_path, 'a') as vectorData:
            array_name_list = ["mesh", "domains", "boundaries"]

            for name in array_name_list:
                vectorArray = vectorData[name + "/coordinates"]
                vectorArray[...] = correctNumberNodes[:, [1, 2, 3]]
                vectorArray = vectorData[name + "/topology"]
                vectorArray[...] = wrongNumberBdTopology if name == "boundaries" else wrongNumberTopology

            vectorArray = vectorData["boundaries/values"]
            vectorArray[...] = wrongNumberBdValues

        # Delete the old "wrong" mesh file and rename the new "fixed" mesh file
        refined_mesh_path.unlink()
        output_path.rename(refined_mesh_path)

        print("The node numbering in the refined mesh has been corrected to match the output velocity.h5 file")


def main() -> None:

    assert MPI.size(MPI.comm_world) == 1, "This script only runs in serial."

    args = postprocessing_mesh_common.parse_arguments()

    folder_path = Path(args.folder)
    if args.mesh_path is None:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
    else:
        mesh_path = Path(args.mesh_path)

    create_refined_mesh(folder_path, mesh_path)


if __name__ == "__main__":
    main()
