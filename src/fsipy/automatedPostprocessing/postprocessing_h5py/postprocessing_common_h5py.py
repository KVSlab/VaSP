# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#   2023 Daniel Macdonald
#   2023 Mehdi Najafi

"""
This file contains helper functions for creating visualizations outside of FEniCS.
"""

import logging
from pathlib import Path
from typing import Tuple, Union

import h5py
import pandas as pd
import configargparse
import numpy as np
from numpy import linalg as LA
from scipy.io import wavfile
from tqdm import tqdm

from fsipy.automatedPostprocessing.postprocessing_h5py import spectrograms as spec
from fsipy.automatedPostprocessing.postprocessing_common import get_domain_ids, output_file_lists, \
    read_parameters_from_file


def read_command_line() -> configargparse.Namespace:
    """
    Read arguments from the command line.

    Returns:
        Namespace: Parsed command line arguments.
    """
    parser = configargparse.ArgumentParser()

    parser.add_argument('--folder', type=Path, required=True, default=None,
                        help="Path to simulation results")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file (default: <folder_path>/Mesh/mesh.h5)")
    parser.add_argument('--save-deg', type=int, default=2,
                        help="Specify the save_deg used during the simulation, i.e. whether the intermediate P2 nodes "
                             "were saved. Entering save_deg=1 when the simulation was run with save_deg=2 will result "
                             "in using only the corner nodes in postprocessing.")
    parser.add_argument('--stride', type=int, default=1,
                        help="Desired frequency of output data (i.e. to output every second step, use stride=2)")
    parser.add_argument('--dt', type=float, default=0.001,
                        help="Time step of simulation (s)")
    parser.add_argument('--start-time', type=float, default=0.0,
                        help="Start time of simulation (s)")
    parser.add_argument('--end-time', type=float, default=0.05,
                        help="End time of simulation (s)")
    parser.add_argument('-q', '--quantity', type=str, default="v",
                        help="Quantity to postprocess. Choose 'v' for velocity, 'd' for displacement, 'p' for pressure,"
                             " or 'wss' for wall shear stress.")
    parser.add_argument('--bands', default="25,100000",
                        help="Input lower and upper band for band-pass filtered displacement, in a list of pairs. For "
                             "example: --bands '100 150 175 200' gives you band-pass filtered visualization for the "
                             "band between 100 and 150, and another visualization for the band between 175 and 200.")
    parser.add_argument('--points', default="0,1",
                        help="Input list of points")

    args = parser.parse_args()

    # Set default mesh path if not provided
    args.mesh_path = args.folder / "Mesh" / "mesh.h5" if args.mesh_path is None else args.mesh_path

    return args


def get_coords(mesh_path: Union[str, Path]) -> np.ndarray:
    """
    Get coordinates from a mesh file.

    Args:
        mesh_path (str or Path): Path to the mesh file.

    Returns:
        np.ndarray: Array containing the coordinates.
    """
    with h5py.File(mesh_path, "r") as mesh:
        coords = mesh['mesh/coordinates'][:, :]
    return coords


def get_surface_topology_coords(out_file: Union[str, Path]) -> tuple:
    """
    Get surface topology and coordinates from an output file.

    Args:
        out_file (str or Path): Path to the output file.

    Returns:
        tuple: Tuple containing the surface topology and coordinates.
    """
    with h5py.File(out_file, "r") as mesh:
        topology = mesh["Mesh/0/mesh/topology"][:, :]
        coords = mesh["Mesh/0/mesh/geometry"][:, :]
    return topology, coords


def get_domain_ids_specified_region(mesh_path: Union[str, Path], fluid_sampling_domain_id: int,
                                    solid_sampling_domain_id: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Obtain node IDs for the fluid, solid, and all elements within specified regions of the input mesh.

    Args:
        mesh_path (str or Path): The file path of the input mesh.
        fluid_sampling_domain_id (int): Domain ID for the fluid region to be sampled.
        solid_sampling_domain_id (int): Domain ID for the solid region to be sampled.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]: Node IDs for fluid, solid, and all elements.
    """
    with h5py.File(mesh_path, "r") as vector_data:
        domains_loc = 'domains/values'
        domains = vector_data[domains_loc][:]  # Open domain array
        id_solid = np.nonzero(domains == solid_sampling_domain_id)  # domain = 2 is the solid
        id_fluid = np.nonzero(domains == fluid_sampling_domain_id)  # domain = 1 is the fluid

        topology_loc = 'domains/topology'
        all_topology = vector_data[topology_loc][:, :]
        solid_topology = all_topology[id_solid, :]
        fluid_topology = all_topology[id_fluid, :]

        solid_ids = np.unique(solid_topology)  # Unique node ID's in the solid topology, sorted in ascending order
        fluid_ids = np.unique(fluid_topology)  # Unique node ID's in the fluid topology, sorted in ascending order
        all_ids = np.unique(all_topology)

    return fluid_ids, solid_ids, all_ids


def get_interface_ids(mesh_path: Union[str, Path], fluid_domain_id: Union[int, list[int]],
                      solid_domain_id: Union[int, list[int]]) -> np.ndarray:
    """
    Get the interface node ID's between fluid and solid domains from the given mesh file.

    Args:
        mesh_path (str or Path): Path to the mesh file.
        fluid_domain_id (int or list): ID of the fluid domain
        solid_domain_id (int or list): ID of the solid domain

    Returns:
        np.ndarray: Array containing the interface node IDs.
    """
    fluid_ids, solid_ids, _ = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)

    # Find the intersection of fluid and solid node ID's
    interface_ids_set = set(fluid_ids) & set(solid_ids)

    # Convert the set to a NumPy array
    interface_ids = np.array(list(interface_ids_set))

    return interface_ids


def get_sampling_constants(df: pd.DataFrame, start_t: float, end_t: float) -> Tuple[float, int, float]:
    """
    Calculate sampling constants from a DataFrame.

    Args:
        df (DataFrame): The input DataFrame.
        start_t (float): The start time of the data.
        end_t (float): The end time of the data.

    Returns:
        Tuple[float, int, float]: A tuple containing the period (T), number of samples per cycle (nsamples),
        and sample rate (fs).

    Author:
        Daniel Macdonald
    """
    T = end_t - start_t
    nsamples = df.shape[1]
    fs = nsamples / T
    return T, nsamples, fs


def read_npz_files(filepath: Union[str, Path]) -> pd.DataFrame:
    """
    Read data from an npz file and return it as a DataFrame.

    Args:
        filepath (str or Path): Path to the npz file.

    Returns:
        pd.DataFrame: DataFrame containing the data.
    """
    logging.info(f'--- Reading data from: {filepath}')
    data = np.load(filepath)['component']
    df = pd.DataFrame(data, copy=False)
    df.index.names = ['Ids']
    logging.info('--- DataFrame creation complete.')
    return df


def create_transformed_matrix(input_path: Union[str, Path], output_folder: Union[str, Path],
                              mesh_path: Union[str, Path], case_name: str, start_t: float, end_t: float, quantity: str,
                              fluid_domain_id: Union[int, list[int]], solid_domain_id: Union[int, list[int]],
                              stride: int = 1) -> float:
    """
    Create a transformed matrix from simulation data.

    Args:
        input_path (str or Path): Path to the input simulation data.
        output_folder (str or Path): Path to the output folder where the transformed matrix will be stored.
        mesh_path (str or Path): Path to the input mesh data.
        case_name (str): Name of the simulation case.
        start_t (float): Start time for extracting data.
        end_t (float): End time for extracting data.
        quantity (str): Quantity to extract (e.g., 'd' for displacement, 'v' for velocity).
        fluid_domain_id (int or list): ID of the fluid domain
        solid_domain_id (int or list): ID of the solid domain
        stride (int): Stride for selecting timesteps.

    Returns:
        float: Time between simulation output files.
    """
    logging.info(f"--- Creating matrix for case {case_name}...")
    input_path = Path(input_path)
    output_folder = Path(output_folder)

    # Get parameters
    parameters = read_parameters_from_file(input_path.parent)

    # Create output directory if it doesn't exist
    if not output_folder.exists():
        output_folder.mkdir(parents=True)
        logging.info(f'--- Output directory created: {output_folder}')
    else:
        logging.info(f'--- Output directory already exists: {output_folder}')

    # Get node ID's from input mesh. If save_deg=2, you can supply the original mesh to get the data for the
    # corner nodes, or supply a refined mesh to get the data for all nodes (very computationally intensive)
    if quantity in {"d", "v", "p"}:
        fluid_ids, solid_ids, all_ids = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)
        ids = all_ids

    # Get name of xdmf file
    xdmf_files = {
        'd': 'displacement.xdmf',
        'v': 'velocity.xdmf',
        'p': 'pressure.xdmf',
        'wss': 'WSS_ts.xdmf',
        'mps': 'MaxPrincipalStrain.xdmf',
        'strain': 'InfinitesimalStrain.xdmf'
    }

    if quantity in xdmf_files:
        xdmf_path = input_path / xdmf_files[quantity]
    else:
        raise ValueError("Invalid value for quantity. Please use 'd', 'v', 'p', 'wss', 'mps', or 'strain'.")

    # Get information about h5 files associated with xdmf file and also information about the timesteps
    logging.info("--- Getting information about h5 files")
    h5_ts, time_ts, index_ts = output_file_lists(xdmf_path)

    # Calculate the time between files from the xdmf file
    time_between_files = time_ts[2] - time_ts[1]

    # Open up the first h5 file to get the number of timesteps and nodes for the output data
    first_h5_file = input_path / h5_ts[0]
    vector_data = h5py.File(first_h5_file, 'r')

    if quantity in {"wss", "mps", "strain"}:
        ids = np.arange(len(vector_data['VisualisationVector/0'][:]))

    vector_array_all = vector_data['VisualisationVector/0'][:, :]
    vector_array = vector_array_all[ids, :]

    num_ts = int(len(time_ts))  # Total amount of timesteps in original file

    logging.info(f"--- Total number of timesteps: {num_ts}")

    # Get shape of output data
    num_rows = vector_array.shape[0]
    num_cols = int((end_t - start_t) / (time_between_files * stride)) - 1

    # Pre-allocate the arrays for the formatted data
    if quantity in {"v", "d"}:
        quantity_x, quantity_y, quantity_z = [np.zeros((num_rows, num_cols)) for _ in range(3)]
    elif quantity == "strain":
        quantity_11, quantity_12, quantity_22, quantity_23, quantity_33, quantity_31 = \
            [np.zeros((num_rows, num_cols)) for _ in range(6)]

    quantity_magnitude = np.zeros((num_rows, num_cols))

    # Initialize variables
    tol = 1e-8  # temporal spacing tolerance
    idx_zeroed = 0  # Output index for formatted data
    h5_file_prev = ""

    # Set start and stop timesteps
    if parameters is not None:
        dt = parameters["dt"]
        start = round(start_t / dt)
        stop = round(end_t / dt) - 2
    else:
        start = 0
        stop = num_ts

    # Initialize tqdm with the total number of iterations
    progress_bar = tqdm(total=stop - start, desc="--- Transferring timestep", unit="step")

    for i in range(start, stop):
        time_file = time_ts[i]

        # Check if the spacing between files is not equal to the intended timestep
        if i > 0 and np.abs(time_file - time_ts[i - 1] - time_between_files) > tol:
            logging.warning('WARNING: Uneven temporal spacing detected!!')

        # Open input h5 file
        h5_file = input_path / h5_ts[i]

        if h5_file != h5_file_prev:
            vector_data.close()
            vector_data = h5py.File(h5_file, 'r')

        h5_file_prev = h5_file

        # If the timestep falls within the desired timeframe and has the correct stride
        if start_t <= time_file <= end_t and i % stride == 0:
            # Open Vector Array from h5 file
            array_name = 'VisualisationVector/' + str(index_ts[i])
            vector_array_full = vector_data[array_name][:, :]

            try:
                # Get required data depending on whether pressure, displacement, or velocity
                if quantity in {"p", "wss", "mps"}:
                    quantity_magnitude[:, idx_zeroed] = vector_array_full[ids, 0]
                elif quantity == "strain":
                    vector_array = vector_array_full[ids, :]
                    quantity_11[:, idx_zeroed] = vector_array[:, 0]
                    quantity_12[:, idx_zeroed] = vector_array[:, 1]
                    quantity_22[:, idx_zeroed] = vector_array[:, 4]
                    quantity_23[:, idx_zeroed] = vector_array[:, 5]
                    quantity_33[:, idx_zeroed] = vector_array[:, 8]
                    quantity_31[:, idx_zeroed] = vector_array[:, 6]
                else:
                    vector_array = vector_array_full[ids, :]
                    quantity_x[:, idx_zeroed] = vector_array[:, 0]
                    quantity_y[:, idx_zeroed] = vector_array[:, 1]
                    quantity_z[:, idx_zeroed] = vector_array[:, 2]
                    quantity_magnitude[:, idx_zeroed] = LA.norm(vector_array, axis=1)

            except Exception as e:
                logging.error(f"ERROR: An unexpected error occurred - {e}")
                break

            # Update the information in the progress bar
            progress_bar.set_postfix({"Timestep": index_ts[i], "Time": time_ts[i], "File": h5_ts[i]})
            idx_zeroed += 1  # Move to the next index of the output h5 file

        progress_bar.update()

    progress_bar.close()

    vector_data.close()
    logging.info("--- Finished reading h5 files")

    # Create output h5 file
    if quantity in {"d", "v"}:
        formatted_data = [quantity_magnitude, quantity_x, quantity_y, quantity_z]
        component_names = ["mag", "x", "y", "z"]
    elif quantity == "strain":
        formatted_data = [quantity_11, quantity_12, quantity_22, quantity_23, quantity_33, quantity_31]
        component_names = ["11", "12", "22", "23", "33", "31"]
    else:
        component_names = ["mag"]

    for i, component_name in enumerate(tqdm(component_names, desc="--- Writing component files", unit="component")):
        # Create output path
        output_file_name = f"{quantity}_{component_name}.npz"
        output_path = output_folder / output_file_name

        # Remove old file path
        if output_path.exists():
            output_path.unlink()

        # Store output in npz file
        if quantity in {"v", "d", "strain"}:
            np.savez_compressed(output_path, component=formatted_data[i])
        else:
            np.savez_compressed(output_path, component=quantity_magnitude)

    logging.info("--- Finished writing component files\n")

    return time_between_files


def sonify_point(case_name: str, quantity: str, df, start_t: float, end_t: float, overlap_frac: float, lowcut: float,
                 image_folder: str) -> None:
    """
    Sonify a point in the dataframe and save the resulting audio as a WAV file.

    Args:
        case_name (str): Name of the case.
        quantity (str): Type of data to be sonified.
        df (pd.DataFrame): Input DataFrame containing relevant data.
        start_t (float): Start time for sonification.
        end_t (float): End time for sonification.
        overlap_frac (float): Fraction of overlap between consecutive segments.
        lowcut (float): Cutoff frequency for the high-pass filter.
        image_folder (str): Folder to save the sonified audio file.

    Returns:
        None: Saves the sonified audio file in WAV format.
    """
    # Get sampling constants
    T, _, fs = spec.get_sampling_constants(df, start_t, end_t)

    # High-pass filter dataframe for spectrogram
    df_filtered = spec.filter_time_data(df, fs, lowcut=lowcut, highcut=15000.0, order=6, btype='highpass')

    y2 = df_filtered.iloc[0] / np.max(df_filtered.iloc[0])

    sound_filename = f"{quantity}_sound_{y2.name}_{case_name}.wav"
    path_to_sound = Path(image_folder) / sound_filename

    wavfile.write(path_to_sound, int(fs), y2)
