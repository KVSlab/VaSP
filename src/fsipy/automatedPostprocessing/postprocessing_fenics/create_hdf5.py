import numpy as np
import h5py
import re
from pathlib import Path
import json
import logging

from postprocessing_common import read_command_line
from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters


# set compiler arguments
parameters["reorder_dofs_serial"] = False


def create_hdf5(visualization_path, mesh_path, save_time_step, stride, start_t, end_t, extract_solid_only,
                fluid_domain_id, solid_domain_id):

    """
    Loads displacement/velocity data from turtleFSI output and reformats the data so that it can be read in fenics.

    Args:
        visualization_path (Path): Path to the folder containing the visualization files (displacement/velocity.h5)
        mesh_path (Path): Path to the mesh file (mesh.h5 or mesh_refined.h5 depending on save_deg)
        stride: stride of the time steps to be saved
        start_t (float): desired start time for the output file
        end_t (float): desired end time for the output file
        extracct_solid_only (bool): If True, only the solid domain is extracted for displacement.
                                    If False, both the fluid and solid domains are extracted.
        fluid_domain_id (int): ID of the fluid domain
        solid_domain_id (int): ID of the solid domain
    """

    # Define mesh path related variables
    fluid_domain_path = mesh_path.with_name(mesh_path.stem + "_fluid.h5")
    if extract_solid_only:
        solid_domain_path = mesh_path.with_name(mesh_path.stem + "_solid.h5")
    else:
        solid_domain_path = mesh_path

    # Check if the input mesh exists
    if not fluid_domain_path.exists() or not solid_domain_path.exists():
        raise ValueError("Mesh file not found.")

    # Read fluid and solid mesh
    logging.info("--- Reading fluid and solid mesh files \n")
    mesh_fluid = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_domain_path), "r") as mesh_file:
        mesh_file.read(mesh_fluid, "mesh", False)

    messh_solid = Mesh()
    with HDF5File(MPI.comm_world, str(solid_domain_path), "r") as mesh_file:
        mesh_file.read(messh_solid, "mesh", False)

    # Define function spaces and functions
    logging.info("--- Defining function spaces and functions \n")
    Vf = VectorFunctionSpace(mesh_fluid, "CG", 1)
    Vs = VectorFunctionSpace(messh_solid, "CG", 1)
    u = Function(Vf)
    d = Function(Vs)

    # Define paths for velocity and displacement files
    xdmf_file_v = visualization_path / "velocity.xdmf"
    xdmf_file_d = visualization_path / "displacement.xdmf"

    # Get information about h5 files associated with xdmf files and also information about the timesteps
    logging.info("--- Getting information about h5 files \n")
    h5file_name_list, timevalue_list, index_list = output_file_lists(xdmf_file_v)
    h5file_name_list_d, _, index_list_d = output_file_lists(xdmf_file_d)

    fluidIDs, solidIDs, allIDs = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)

    # Remove this if statement since it can be done when we are using d_ids
    if extract_solid_only:
        logging.info("--- Extracting solid domain only \n")
        d_ids = solidIDs
    else:
        logging.info("--- Extracting both fluid and solid domains for displacement \n")
        d_ids = allIDs

    # Open up the first velocity.h5 file to get the number of timesteps and nodes for the output data
    file = visualization_path / h5file_name_list[0]
    vectorData = h5py.File(str(file))
    vectorArray = vectorData['VisualisationVector/0'][fluidIDs, :]

    # Open up the first displacement.h5 file to get the number of timesteps and nodes for the output data
    file_d = visualization_path / h5file_name_list_d[0]
    vectorData_d = h5py.File(str(file_d))
    vectorArray_d = vectorData['VisualisationVector/0'][d_ids, :]

    # Deinfe path to the output files
    u_output_path = visualization_path / "u.h5"
    d_output_path = visualization_path / "d.h5"

    # Initialize h5 file names that might differ during the loop
    h5_file_prev = ""
    h5_file_prev_d = ""

    # Start file counter
    file_counter = 0
    while True:

        try:

            time = timevalue_list[file_counter]
            print("=" * 10, "Timestep: {}".format(time), "=" * 10)

            if file_counter > 0:
                if np.abs(time - timevalue_list[file_counter - 1] - save_time_step) > 1e-8:
                    print("WARNING : Uenven temporal spacing detected")

            if start_t <= time <= end_t:

                # Open input velocity h5 file
                h5_file = visualization_path / h5file_name_list[file_counter]
                if h5_file != h5_file_prev:
                    vectorData.close()
                    vectorData = h5py.File(str(h5_file))
                h5_file_prev = h5_file

                # Open input displacement h5 file
                h5_file_d = visualization_path / h5file_name_list_d[file_counter]
                if h5_file_d != h5_file_prev_d:
                    vectorData_d.close()
                    vectorData_d = h5py.File(str(h5_file_d))
                h5_file_prev_d = h5_file_d

                # Open up Vector Arrays from h5 file
                ArrayName = 'VisualisationVector/' + str((index_list[file_counter]))
                vectorArrayFull = vectorData[ArrayName][:, :]
                ArrayName_d = 'VisualisationVector/' + str((index_list_d[file_counter]))
                vectorArrayFull_d = vectorData_d[ArrayName_d][:, :]

                vectorArray = vectorArrayFull[fluidIDs, :]
                vectorArray_d = vectorArrayFull_d[d_ids, :]

                # Flatten the vector array and insert into the function
                vector_np_flat = vectorArray.flatten('F')
                u.vector().set_local(vector_np_flat)
                print("Saved data in u.h5")

                # Flatten the vector array and insert into the function
                vector_np_flat_d = vectorArray_d.flatten('F')
                d.vector().set_local(vector_np_flat_d)
                print("Saved data in d.h5")

                file_mode = "a" if file_counter > 0 else "w"

                # Save velocity
                viz_u_file = HDF5File(MPI.comm_world, str(u_output_path), file_mode=file_mode)
                viz_u_file.write(u, "/velocity", time)
                viz_u_file.close()

                # Save displacment
                viz_d_file = HDF5File(MPI.comm_world, str(d_output_path), file_mode=file_mode)
                viz_d_file.write(d, "/displacement", time)
                viz_d_file.close()

        except Exception as error:
            print("An exception occurred:", error)
            print("=" * 10, "Finished reading solutions", "=" * 10)
            break

        # Update file_counter
        file_counter += stride


def get_domain_ids(mesh_path, fluid_domain_id=1, solid_domain_id=2):
    """
    Given a mesh file, this function returns the IDs of the fluid and solid domains

    Args:
        mesh_path (Path): Path to the mesh file that contains the fluid and solid domains
        fluid_domain_id (int): ID of the fluid domain
        solid_domain_id (int): ID of the solid domain
    
    Returns:
        fluidIDs (list): List of IDs of the fluid domain
        solidIDs (list): List of IDs of the solid domain
        allIDs (list): List of IDs of the whole mesh
    """
    with h5py.File(mesh_path) as vectorData:
        domains = vectorData['domains/values'][:]
        topology = vectorData['domains/topology'][:, :]
    
        id_solid = (domains == solid_domain_id).nonzero()
        id_fluid = (domains == fluid_domain_id).nonzero()

        wallTopology = topology[id_solid, :]
        fluidTopology = topology[id_fluid, :]

        # Get topology of fluid, solid and whole mesh
        solidIDs = np.unique(wallTopology)
        fluidIDs = np.unique(fluidTopology)
        allIDs = np.unique(topology)

    return fluidIDs, solidIDs, allIDs


def output_file_lists(xdmf_file):
    """
    If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
    This loop determines the file names from the xdmf output file

    Args:
        xdmf_file (Path): Path to xdmf file

    Returns:
        h5file_name_list (list): List of names of h5 files associated with each timestep
        timevalue_list (list): List of time values in xdmf file
        index_list (list): List of indices of each timestp in the corresponding h5 file
    """

    file1 = open(xdmf_file, 'r')
    Lines = file1.readlines()
    h5file_name_list = []
    timevalue_list = []
    index_list = []

    # This loop goes through the xdmf output file and gets the time value (timevalue_list), associated
    # with .h5 file (h5file_name_list) and index of each timestep in the corresponding h5 file (index_list)
    for line in Lines:
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            timevalue_list.append(time)

        elif 'VisualisationVector' in line:
            h5_pattern = '"HDF">(.+?):/'
            h5_str = re.findall(h5_pattern, line)
            h5file_name_list.append(h5_str[0])

            index_pattern = "VisualisationVector/(.+?)</DataItem>"
            index_str = re.findall(index_pattern, line)
            index = int(index_str[0])
            index_list.append(index)

    return h5file_name_list, timevalue_list, index_list


def main() -> None:

    assert MPI.size(MPI.comm_world) == 1, "This script only runs in serial."

    args = read_command_line()

    logging.basicConfig(level=20, format="%(message)s")

    # Define paths for visulization and mesh files
    folder_path = Path(args.folder)
    visualization_path = folder_path / "Visualization"

    # Read parameters from default_variables.json
    parameter_path = folder_path / "Checkpoint" / "default_variables.json"
    with open(parameter_path, "r") as f:
        parameters = json.load(f)
        save_deg = parameters["save_deg"]
        dt = parameters["dt"]
        save_step = parameters["save_step"]
        save_time_step = dt * save_step
        print(f"save_time_step: {save_time_step} \n")
        fluid_domain_id = parameters["dx_f_id"]
        solid_domain_id = parameters["dx_s_id"]

        if type(fluid_domain_id) is not int:
            fluid_domain_id = fluid_domain_id[0]
            print("fluid_domain_id is not int, using first element of list \n")
        if type(solid_domain_id) is not int:
            solid_domain_id = solid_domain_id[0]
            print("solid_domain_id is not int, using first element of list \n")

        print(f"--- Fluid domain ID: {fluid_domain_id} and Solid domain ID: {solid_domain_id} \n")

    if save_deg == 2:
        mesh_path = folder_path / "Mesh" / "mesh_refined.h5"
        logging.info("--- Using refined mesh \n")
        assert mesh_path.exists(), "Mesh file not found."
    else:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        logging.info("--- Using non-refined mesh \n")
        assert mesh_path.exists(), "Mesh file not found."

    create_hdf5(visualization_path, mesh_path, save_time_step, args.stride,
                args.start_t, args.end_t, args.extract_solid_only, fluid_domain_id, solid_domain_id)


if __name__ == '__main__':
    main()
