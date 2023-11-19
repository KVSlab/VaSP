# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

import numpy as np
import h5py
import re
from pathlib import Path
import json
import logging

from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names
from fsipy.automatedPostprocessing.postprocessing_fenics import postprocessing_fenics_common
from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters, XDMFFile


# set compiler arguments
parameters["reorder_dofs_serial"] = False


def create_separate_domain_visualiation(visualization_path, mesh_path, save_time_step, stride, start_time, end_time, extract_solid_only,
                fluid_domain_id, solid_domain_id):

    """
    Loads displacement/velocity data from create_hdf5 and convert them into xdmf for visualization in Paraview

    Args:
        visualization_path (Path): Path to the folder containing the visualization files (d.h5/v.h5)
        mesh_path (Path): Path to the mesh file (mesh.h5 or mesh_refined.h5 depending on save_deg)
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


    fluid_ids, solid_ids, all_ids = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)

    # Remove this if statement since it can be done when we are using d_ids
    if extract_solid_only:
        logging.info("--- Displacement will be extracted for the solid domain only \n")
        d_ids = solid_ids
    else:
        logging.info("--- Displacement will be extracted for both the fluid and solid domains \n")
        d_ids = all_ids

    # Deinfe path to the output files
    u_iutput_path = visualization_path / "u.h5"
    d_iutput_path = visualization_path / "d.h5"

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


def compute_velocity_and_pressure(visualization_path, mesh_path, extract_solid_only):
    """
    Loads velocity and pressure from compressed .h5 CFD solution and
    converts and saves to .xdmf format for visualization (in e.g. ParaView).

    Args:
        folder (str): Path to results from simulation
        dt (float): Time step of simulation
        save_frequency (int): Frequency that velocity and pressure has been stored
        velocity_degree (int): Finite element degree of velocity
        pressure_degree (int): Finite element degree of pressure
        step (int): Step size determining number of times data is sampled
    """
    # File paths
    file_path_d = visualization_path / "d.h5"
    file_path_u = visualization_path / "u.h5"
    assert file_path_d.exists(), f"Displacement file {file_path_d} not found."
    assert file_path_u.exists(), f"Velocity file {file_path_u} not found."

    # Define HDF5Files
    file_d = HDF5File(MPI.comm_world, str(file_path_d), "r")
    file_u = HDF5File(MPI.comm_world, str(file_path_u), "r")
    
    # Read in datasets
    dataset_d = get_dataset_names(file_d, step=1, vector_filename="/displacement/vector_%d")
    dataset_u = get_dataset_names(file_u, step=1, vector_filename="/velocity/vector_%d")

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

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("Define function spaces")
    # Define function spaces and functions
    logging.info("--- Defining function spaces and functions \n")
    Vf = VectorFunctionSpace(mesh_fluid, "CG", 1)
    Vs = VectorFunctionSpace(mesh_solid, "CG", 1)

    u = Function(Vf)
    d = Function(Vs)


    # Create writer for velocity and pressure
    d_path = visualization_path / "displacement_solid.xdmf" if extract_solid_only else visualization_path / "displacement_whole.xdmf"
    u_path = visualization_path / "velocity_fluid.xdmf"

    d_writer = XDMFFile(MPI.comm_world, str(d_path))
    u_writer = XDMFFile(MPI.comm_world, str(u_path))

    for writer in [d_writer, u_writer]:
        writer.parameters["flush_output"] = True
        writer.parameters["functions_share_mesh"] = False
        writer.parameters["rewrite_function_mesh"] = False

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # Open up h5 file to get dt and start_time
    h5_file = visualization_path / "u.h5"
    vector_data = h5py.File(str(h5_file))
    first_time_step_data = vector_data['velocity/vector_0']
    second_time_step_data = vector_data['velocity/vector_1']
    start_time = first_time_step_data.attrs['timestamp']
    dt = second_time_step_data.attrs['timestamp'] - first_time_step_data.attrs['timestamp']
    vector_data.close()

    counter = 1
    for i in range(len(dataset_u)):
        # Set physical time (in [ms])
        t = dt * counter + start_time

        file_d.read(d, dataset_d[i])
        file_u.read(u, dataset_u[i])

        if MPI.rank(MPI.comm_world) == 0:
            timestamp = file_u.attributes(dataset_u[i])["timestamp"]
            print("=" * 10, "Timestep: {}".format(timestamp), "=" * 10)

        # Store velocity
        u.rename("velocity", "velocity")
        u_writer.write(u, t)

        # Store pressure
        d.rename("displacement", "displacement")
        d_writer.write(d, t)

    
        # Update file_counter
        counter += 1

    print("========== Post processing finished ==========")
