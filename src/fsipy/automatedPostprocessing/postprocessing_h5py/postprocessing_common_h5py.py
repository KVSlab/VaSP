# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#   2023 Daniel Macdonald
#   2023 Mehdi Najafi

"""
This file contains helper functions for creating visualizations outside of FEniCS.
"""

import os
import logging
from pathlib import Path
from typing import Tuple, Union, List

import h5py
import pandas as pd
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
from scipy import signal
import matplotlib.pyplot as plt

from fsipy.automatedPostprocessing.postprocessing_common import get_domain_ids, output_file_lists, \
    read_parameters_from_file


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


def create_point_trace(formatted_data_folder: str, output_folder: str, point_ids: List[int], save_deg: bool,
                       time_between_files: float, start_t: float, dvp: str) -> None:
    """
    Create point traces for specified point IDs and save the results in CSV and PNG files.

    Args:
        formatted_data_folder (str): Path to the folder containing formatted data.
        output_folder (str): Path to the folder where output files will be saved.
        point_ids (List[int]): List of point IDs for which traces will be created.
        save_deg (bool): A boolean indicating whether to save the degree symbol in the plot title.
        time_between_files (float): Time between files in seconds.
        start_t (float): Start time of the simulation.
        dvp (str): Type of visualization ('v' for velocity, 'd' for displacement, 'p' for pressure).

    Returns:
        None
    """

    # Get input data
    components_data = []
    component_names = ["mag", "x", "y", "z"]

    for i, component_name in enumerate(tqdm(component_names, desc="--- Loading data")):
        if dvp == "p" and i > 0:
            break

        file_str = f"{dvp}_{component_name}.npz"
        matching_files = [file for file in os.listdir(formatted_data_folder) if file_str in file]

        if not matching_files:
            raise FileNotFoundError(f"No file found for {file_str}")

        component_file = matching_files[0]
        component_data = np.load(Path(formatted_data_folder) / component_file)['component']
        components_data.append(component_data)

    # Create name for output file, define output path
    if dvp == "v":
        viz_type = 'velocity'
    elif dvp == "d":
        viz_type = 'displacement'
    elif dvp == "p":
        viz_type = 'pressure'
    else:
        raise ValueError("Input 'd', 'v' or 'p' for dvp")

    num_ts = components_data[0].shape[1]
    time_plot = np.arange(start_t, num_ts * time_between_files + start_t, time_between_files)

    # Create output directory
    output_path = Path(output_folder)

    if output_path.exists():
        logging.debug(f"--- Output folder '{output_path}' already exists.")
    else:
        output_path.mkdir(parents=True, exist_ok=True)
        logging.debug(f"--- Output folder '{output_path}' created.")

    print(f"--- Processing point IDs: {point_ids}")
    for point_id in point_ids:
        output_string = f"{viz_type}_point_id_{point_id}"  # Base filename
        num_columns = 2 if dvp == "p" else 5
        output_data = np.zeros((num_ts, num_columns))
        output_data[:, 0] = time_plot

        for i in range(1, num_columns):
            output_data[:, i] = components_data[i - 1][point_id, :]

        point_trace_file = output_path / f"{output_string}.csv"
        point_trace_graph_file = output_path / f"{output_string}.png"

        header = "time (s), Magnitude" if dvp == "p" else "time (s), Magnitude, X Component, Y Component, Z Component"
        np.savetxt(point_trace_file, output_data, delimiter=",", header=header)

        # Plot and Save
        plt.plot(output_data[:, 0], output_data[:, 1], label="Mag")
        if dvp != "p":
            plt.plot(output_data[:, 0], output_data[:, 2], label="X")
            plt.plot(output_data[:, 0], output_data[:, 3], label="Y")
            plt.plot(output_data[:, 0], output_data[:, 4], label="Z")

        plt.title(f'Point # {point_id}')
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

    print(f"--- Point traces saved at: {output_folder}")


def create_xdmf_file(num_ts: int, time_between_files: float, start_t: float, n_elements: int,
                     n_nodes: int, att_type: str, viz_type: str, output_folder: Path) -> None:
    """
    Create an XDMF file for a time series visualization.

    Args:
        num_ts (int): Number of time steps.
        time_between_files (float): Time interval between files.
        start_t (float): Starting time.
        n_elements (int): Number of elements.
        n_nodes (int): Number of nodes.
        att_type (str): Type of attribute - "Scalar", "Vector", or "Tensor".
        viz_type (str): Visualization type.
        output_folder (Path): Path to the output folder.

    Returns:
        None

    Raises:
        ValueError: If an unsupported attribute type is provided.
    """
    # Create strings
    num_el = str(n_elements)
    num_nodes = str(n_nodes)
    if att_type == "Scalar":
        n_dim = '1'
    elif att_type == "Tensor":
        n_dim = '9'
    elif att_type == "Vector":
        n_dim = '3'
    else:
        raise ValueError("Attribute type must be one of 'Scalar', 'Vector', or 'Tensor'.")

    # Write lines of xdmf file
    lines = f'''<?xml version="1.0"?>
<!DOCTYPE Xdmf SYSTEM "Xdmf.dtd" []>
<Xdmf Version="3.0" xmlns:xi="http://www.w3.org/2001/XInclude">
  <Domain>
    <Grid Name="TimeSeries_{viz_type}" GridType="Collection" CollectionType="Temporal">
      <Grid Name="mesh" GridType="Uniform">
        <Topology NumberOfElements="{num_el}" TopologyType="Tetrahedron" NodesPerElement="4">
          <DataItem Dimensions="{num_el} 4" NumberType="UInt" Format="HDF">{viz_type}.h5:/Mesh/0/mesh/topology</DataItem>
        </Topology>
        <Geometry GeometryType="XYZ">
          <DataItem Dimensions="{num_nodes} 3" Format="HDF">{viz_type}.h5:/Mesh/0/mesh/geometry</DataItem>
        </Geometry>
'''  # noqa

    for idx in range(num_ts):
        time_value = str(idx * time_between_files + start_t)
        lines += f'''\
        <Time Value="{time_value}" />
        <Attribute Name="{viz_type}" AttributeType="{att_type}" Center="Node">
          <DataItem Dimensions="{num_nodes} {n_dim}" Format="HDF">{viz_type}.h5:/VisualisationVector/{idx}</DataItem>
        </Attribute>
      </Grid>
'''

        if idx == num_ts - 1:
            break

        lines += f'''\
      <Grid>
        <xi:include xpointer="xpointer(//Grid[@Name=&quot;TimeSeries_{viz_type}&quot;]/Grid[1]/*[self::Topology or self::Geometry])" />
'''  # noqa

    lines += '''\
    </Grid>
  </Domain>
</Xdmf>
'''

    # Writing lines to file
    xdmf_path = output_folder / f'{viz_type}.xdmf'

    # Remove old file if it exists
    if xdmf_path.exists():
        logging.debug(f'--- Removing existing file at: {xdmf_path}')
        xdmf_path.unlink()

    with open(xdmf_path, 'w') as xdmf_file:
        logging.debug(f'--- Writing XDMF file: {xdmf_path}')
        xdmf_file.write(lines)


def calculate_windowed_rms(signal_array: np.ndarray, window_size: int, window_type: str = "flat") -> np.ndarray:
    """
    Calculate the windowed root mean squared (RMS) amplitude of a signal.

    Args:
        signal_array (numpy.ndarray): Input signal.
        window_size (int): Size of the window in number of timesteps.
        window_type (str, optional): Type of window function to use. Default is "flat".
            Supported window types: "flat", "tukey", "hann", "blackmanharris", "flattop".

    Returns:
        numpy.ndarray: Windowed RMS amplitudes.

    References:
         https://stackoverflow.com/questions/8245687/numpy-root-mean-squared-rms-smoothing-of-a-signal
         https://stackoverflow.com/questions/47484899/moving-average-produces-array-of-different-length
    """
    # Calculate squared amplitudes of the signal
    signal_squared = np.power(signal_array, 2)

    # Define window functions for different types
    window_functions = {
        "flat": np.ones(window_size) / float(window_size),
        "tukey": signal.windows.tukey(window_size) / float(window_size),
        "hann": signal.windows.hann(window_size) / float(window_size),
        "blackmanharris": signal.windows.blackmanharris(window_size) / float(window_size),
        "flattop": signal.windows.flattop(window_size) / float(window_size),
    }

    # Select the appropriate window function based on the specified window type
    window = window_functions.get(window_type, np.ones(window_size) / float(window_size))

    # Calculate the RMS using convolution
    RMS = np.sqrt(np.convolve(signal_squared, window, mode="valid"))

    len_RMS = len(RMS)
    len_signal_squared = len(signal_squared)
    pad_length = int((len_signal_squared - len_RMS) / 2)

    # Pad the RMS array to match the length of the original signal
    RMS_padded = np.zeros(len_signal_squared)
    for i in range(len_signal_squared):
        if pad_length <= i < len_RMS + pad_length:
            RMS_padded[i] = RMS[i - pad_length]

    return RMS_padded


def get_eig(T: np.ndarray) -> float:
    """
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

    Args:
        T (np.ndarray): Three-dimensional tensor.

    Returns:
        float: First eigenvalue.

    References:
        https://fenicsproject.discourse.group/t/hyperelastic-model-problems-on-plotting-stresses/3130/6
    """
    # Determine perturbation from tolerance
    tol1 = 1e-16
    pert1 = 2 * tol1
    tol2 = 1e-24
    pert2 = 2 * tol2
    tol3 = 1e-40
    pert3 = 2 * tol3

    # Get required invariants
    I1 = np.trace(T)
    I2 = 0.5 * (np.trace(T) ** 2 - np.tensordot(T, T))
    I3 = np.linalg.det(T)

    # Determine terms p and q according to the paper
    p = I1 ** 2 - 3 * I2
    if p < tol1:
        logging.info(f"--- perturbation applied to p: p = {p}")
        p = np.abs(p) + pert1

    q = 27 / 2 * I3 + I1 ** 3 - 9 / 2 * I1 * I2
    if abs(q) < tol2:
        logging.info(f"--- perturbation applied to q: q = {q}")
        q = q + np.sign(q) * pert2

    # Determine angle phi for calculation of roots
    phi_nom2 = 27 * (1 / 4 * I2 ** 2 * (p - I2) + I3 * (27 / 4 * I3 - q))
    if phi_nom2 < tol3:
        logging.info(f"--- perturbation applied to phi_nom2: phi_nom2 = {phi_nom2}")
        phi_nom2 = np.abs(phi_nom2) + pert3

    phi = 1 / 3 * np.arctan2(np.sqrt(phi_nom2), q)

    # Calculate polynomial roots
    lambda1 = 1 / 3 * (np.sqrt(p) * 2 * np.cos(phi) + I1)

    # Return polynomial roots (eigenvalues)
    return lambda1
