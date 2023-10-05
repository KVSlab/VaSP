# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

import json
import numpy as np
import h5py
from pathlib import Path

from fsipy.automatedPostprocessing.postprocessing_mesh import postprocessing_mesh_common

from dolfin import MPI, Mesh, MeshFunction, HDF5File, SubMesh, File


def separate_mesh(mesh_path: Path, fluid_domain_id: int, solid_domain_id: int, view: bool = False) -> None:
    """
    Given a mesh file that contains fluid and solid domains, this function separates the domains and saves them as
    separate mesh files. These domain specific mesh files are later used in the other postprocessing scripts.

    args:
        mesh_path (Path): Path to the mesh file.
        fluid_domain_id (int): Domain ID for fluid domain.
        solid_domain_id (int): Domain ID for solid domain.

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

    for domain_id, domain_name in zip([fluid_domain_id, solid_domain_id], ["fluid", "solid"]):

        domain_of_interest = SubMesh(mesh, domains, domain_id)
        domain_of_interest_path = mesh_path.with_name(mesh_path.stem + f"_{domain_name}.h5")
        print(f" --- Saving {domain_name} domain to {domain_of_interest_path} \n")
        with HDF5File(domain_of_interest.mpi_comm(), str(domain_of_interest_path), "w") as hdf:
            hdf.write(domain_of_interest, "/mesh")

        # Save for viewing in paraview
        if view:
            domain_of_interest_pvd_path = domain_of_interest_path.with_suffix(".pvd")
            File(str(domain_of_interest_pvd_path)) << domain_of_interest

    print(" --- Done separating domains \n")

    with h5py.File(mesh_path) as vectorData:
        domain_values = vectorData['domains/values'][:]
        domain_topology = vectorData['domains/topology'][:, :]
        domain_coordinates = vectorData['mesh/coordinates'][:, :]

    for domain_id, domain_name in zip([fluid_domain_id, solid_domain_id], ["fluid", "solid"]):
        # non-zero is used to find the indices of the domain of interest
        domain_of_interest_index = (domain_values == domain_id).nonzero()
        # extract the topology of the domain of interest
        domain_of_interest_topology = domain_topology[domain_of_interest_index[0], :]
        # Here, we want to extract the coordinates of the domain of interest
        # We can do this by extracting the unique node IDs of the domain of interest from the topology
        unique_node_ids = np.unique(domain_of_interest_topology)
        domain_of_interest_coordinates = domain_coordinates[unique_node_ids, :]
        # Fix topology of the domain of interest
        # This is necessary because the node numbering may not be continuous
        # while there is one to one correspondence between the node IDs and the coordinates
        if not np.all(np.diff(unique_node_ids) == 1):
            print(f" --- Fixing topology of {domain_name} domain \n")
            # Create a mapping from old node IDs to new continuous node IDs
            node_id_mapping = {old_id: new_id for new_id, old_id in enumerate(unique_node_ids)}
            # Replace old node IDs with new continuous node IDs in the topology array
            domain_of_interest_topology = np.vectorize(node_id_mapping.get)(domain_of_interest_topology)
        else:
            print(f" --- {domain_name} topology does not need to be fixed \n")

        print("--- Saving the fixed mesh file \n")
        domain_of_interest_path = mesh_path.with_name(mesh_path.stem + f"_{domain_name}.h5")
        domain_of_interest_fixed_path = mesh_path.with_name(mesh_path.stem + f"_{domain_name}_fixed.h5")
        domain_of_interest_fixed_path.write_bytes(domain_of_interest_path.read_bytes())

        with h5py.File(domain_of_interest_fixed_path, "r+") as vectorData:
            coordinate_array = vectorData["mesh/coordinates"]
            coordinate_array[...] = domain_of_interest_coordinates
            topology_array = vectorData["mesh/topology"]
            topology_array[...] = domain_of_interest_topology

        # remove the original mesh file and rename the fixed mesh file
        domain_of_interest_path.unlink()
        domain_of_interest_fixed_path.rename(domain_of_interest_path)


def main() -> None:

    assert MPI.size(MPI.comm_world) == 1, "This script only runs in serial."

    args = postprocessing_mesh_common.parse_arguments()

    folder_path = Path(args.folder)
    if args.mesh_path is None:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
    else:
        mesh_path = Path(args.mesh_path)

    # First, check if the domain specific mesh files already exist
    fluid_domain_path = mesh_path.with_name(mesh_path.stem + "_fluid.h5")
    solid_domain_path = mesh_path.with_name(mesh_path.stem + "_solid.h5")

    if fluid_domain_path.exists() and solid_domain_path.exists():
        print(" --- Domain specific mesh files already exist. Exiting ... \n")
        return
    else:
        print(" --- Separating fluid and solid domains using domain IDs \n")

        parameter_path = folder_path / "Checkpoint" / "default_variables.json"
        with open(parameter_path, "r") as f:
            parameters = json.load(f)
            fluid_domain_id = parameters["dx_f_id"]
            solid_domain_id = parameters["dx_s_id"]

            if type(fluid_domain_id) is not int:
                fluid_domain_id = fluid_domain_id[0]
                print("fluid_domain_id is not int, using first element of list \n")
            if type(solid_domain_id) is not int:
                solid_domain_id = solid_domain_id[0]
                print("solid_domain_id is not int, using first element of list \n")

        print(f" --- Fluid domain ID: {fluid_domain_id} and Solid domain ID: {solid_domain_id} \n")

        separate_mesh(mesh_path, fluid_domain_id, solid_domain_id)

        # Check if refined mesh exists
        refined_mesh_path = mesh_path.with_name(mesh_path.stem + "_refined.h5")
        if refined_mesh_path.exists():
            print(" --- Refined mesh exists, separating domains for refined mesh \n")
            separate_mesh(refined_mesh_path, fluid_domain_id, solid_domain_id, view=args.view)
        else:
            print(" --- Refined mesh does not exist \n")


if __name__ == "__main__":
    main()
