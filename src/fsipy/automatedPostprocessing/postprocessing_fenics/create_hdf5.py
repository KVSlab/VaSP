# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import h5py
import re
from pathlib import Path
import json
import logging

from fsipy.automatedPostprocessing.postprocessing_fenics import postprocessing_fenics_common
from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters


# set compiler arguments
parameters["reorder_dofs_serial"] = False


def create_hdf5(visualization_path, mesh_path, save_time_step, stride, start_time, end_time, extract_solid_only,
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
        fluid_domain_id (int or list): ID of the fluid domain
        solid_domain_id (int or list): ID of the solid domain
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

    mesh_solid = Mesh()
    with HDF5File(MPI.comm_world, str(solid_domain_path), "r") as mesh_file:
        mesh_file.read(mesh_solid, "mesh", False)

    # Define function spaces and functions
    logging.info("--- Defining function spaces and functions \n")
    Vf = VectorFunctionSpace(mesh_fluid, "CG", 1)
    Vs = VectorFunctionSpace(mesh_solid, "CG", 1)
    u = Function(Vf)
    d = Function(Vs)

    # Define paths for velocity and displacement files
    xdmf_file_v = visualization_path / "velocity.xdmf"
    xdmf_file_d = visualization_path / "displacement.xdmf"

    # Get information about h5 files associated with xdmf files and also information about the timesteps
    logging.info("--- Getting information about h5 files \n")
    h5file_name_list, timevalue_list, index_list = output_file_lists(xdmf_file_v)
    h5file_name_list_d, _, index_list_d = output_file_lists(xdmf_file_d)

    fluid_ids, solid_ids, all_ids = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)

    # Remove this if statement since it can be done when we are using d_ids
    if extract_solid_only:
        logging.info("--- Displacement will be extracted for the solid domain only \n")
        d_ids = solid_ids
    else:
        logging.info("--- Displacement will be extracted for both the fluid and solid domains \n")
        d_ids = all_ids

    # Open up the first velocity.h5 file to get the number of timesteps and nodes for the output data
    file = visualization_path / h5file_name_list[0]
    vector_data = h5py.File(str(file))
    vector_array = vector_data['VisualisationVector/0'][fluid_ids, :]

    # Open up the first displacement.h5 file to get the number of timesteps and nodes for the output data
    file_d = visualization_path / h5file_name_list_d[0]
    vector_data_d = h5py.File(str(file_d))
    vector_array_d = vector_data['VisualisationVector/0'][d_ids, :]

    # Deinfe path to the output files
    visualization_separate_domain_folder = visualization_path.parent / "Visualization_separate_domain"
    u_output_path = visualization_separate_domain_folder / "u.h5"
    d_output_path = visualization_separate_domain_folder / "d_solid.h5" if extract_solid_only \
        else visualization_separate_domain_folder / "d.h5"

    # Initialize h5 file names that might differ during the loop
    h5_file_prev = None
    h5_file_prev_d = None

    # Define start and end time and indices for the loop
    start_time = start_time if start_time is not None else timevalue_list[0]

    if end_time is not None:
        assert end_time > start_time, "end_time must be greater than start_time"
        assert end_time <= timevalue_list[-1], "end_time must be less than the last time step"

    end_time = end_time if end_time is not None else timevalue_list[-1]

    start_time_index = int(start_time / save_time_step) - 1
    end_time_index = int(end_time / save_time_step) + 1

    for file_counter in range(start_time_index, end_time_index, stride):

        time = timevalue_list[file_counter]
        logging.info(f"--- Reading data at time: {time}")

        if file_counter > start_time_index:
            if np.abs(time - timevalue_list[file_counter - 1] - save_time_step) > 1e-8:
                logging.warning("WARNING : Uenven temporal spacing detected")

        # Open input velocity h5 file
        h5_file = visualization_path / h5file_name_list[file_counter]
        if h5_file != h5_file_prev:
            vector_data.close()
            vector_data = h5py.File(str(h5_file))
        h5_file_prev = h5_file

        # Open input displacement h5 file
        h5_file_d = visualization_path / h5file_name_list_d[file_counter]
        if h5_file_d != h5_file_prev_d:
            vector_data_d.close()
            vector_data_d = h5py.File(str(h5_file_d))
        h5_file_prev_d = h5_file_d

        # Open up Vector Arrays from h5 file
        array_name = 'VisualisationVector/' + str((index_list[file_counter]))
        vector_array_all = vector_data[array_name][:, :]
        array_name_d = 'VisualisationVector/' + str((index_list_d[file_counter]))
        vector_array_all_d = vector_data_d[array_name_d][:, :]

        vector_array = vector_array_all[fluid_ids, :]
        vector_array_d = vector_array_all_d[d_ids, :]

        # Flatten the vector array and insert into the function
        vector_np_flat = vector_array.flatten('F')
        u.vector().set_local(vector_np_flat)
        logging.info("Saved data in u.h5")

        # Flatten the vector array and insert into the function
        vector_np_flat_d = vector_array_d.flatten('F')
        d.vector().set_local(vector_np_flat_d)
        logging.info("Saved data in d.h5")

        file_mode = "a" if file_counter > start_time_index else "w"

        # Save velocity
        viz_u_file = HDF5File(MPI.comm_world, str(u_output_path), file_mode=file_mode)
        viz_u_file.write(u, "/velocity", time)
        viz_u_file.close()

        # Save displacment
        viz_d_file = HDF5File(MPI.comm_world, str(d_output_path), file_mode=file_mode)
        viz_d_file.write(d, "/displacement", time)
        viz_d_file.close()

    logging.info("--- Finished reading solutions")
    logging.info(f"--- Saved u.h5 and d.h5 in {visualization_separate_domain_folder.absolute()}")


def get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id):
    """
    Given a mesh file, this function returns the IDs of the fluid and solid domains

    Args:
        mesh_path (Path): Path to the mesh file that contains the fluid and solid domains
        fluid_domain_id (int): ID of the fluid domain
        solid_domain_id (int): ID of the solid domain

    Returns:
        fluid_ids (list): List of IDs of the fluid domain
        solid_ids (list): List of IDs of the solid domain
        all_ids (list): List of IDs of the whole mesh
    """
    with h5py.File(mesh_path) as vector_data:
        domains = vector_data['domains/values'][:]
        topology = vector_data['domains/topology'][:, :]

        if isinstance(fluid_domain_id, list):
            id_fluid = np.where((domains == fluid_domain_id[0]) | (domains == fluid_domain_id[1]))
        else:
            id_fluid = np.where(domains == fluid_domain_id)

        if isinstance(solid_domain_id, list):
            id_solid = np.where((domains == solid_domain_id[0]) | (domains == solid_domain_id[1]))
        else:
            id_solid = np.where(domains == solid_domain_id)

        wall_topology = topology[id_solid, :]
        fluid_topology = topology[id_fluid, :]

        # Get topology of fluid, solid and whole mesh
        solid_ids = np.unique(wall_topology)
        fluid_ids = np.unique(fluid_topology)
        all_ids = np.unique(topology)

    return fluid_ids, solid_ids, all_ids


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

    args = postprocessing_fenics_common.parse_arguments()

    logging.basicConfig(level=args.log_level, format="%(message)s")

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
        logging.info(f"save_time_step: {save_time_step} \n")
        fluid_domain_id = parameters["dx_f_id"]
        solid_domain_id = parameters["dx_s_id"]

        logging.info(f"--- Fluid domain ID: {fluid_domain_id} and Solid domain ID: {solid_domain_id} \n")

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        logging.info("--- Using user-defined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    elif save_deg == 2:
        mesh_path = folder_path / "Mesh" / "mesh_refined.h5"
        logging.info("--- Using refined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    else:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        logging.info("--- Using non-refined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    create_hdf5(visualization_path, mesh_path, save_time_step, args.stride,
                args.start_time, args.end_time, args.extract_solid_only, fluid_domain_id, solid_domain_id)


if __name__ == '__main__':
    main()
