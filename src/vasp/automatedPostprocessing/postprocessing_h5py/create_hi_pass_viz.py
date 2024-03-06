# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script processes visualization files from a turtleFSI simulation, generating Band-Pass filtered visualizations for
displacement (d), velocity (v), pressure (p), and strain. Additionally, it creates a "Transformed Matrix" to efficiently
store output data, optimized for quick access when generating spectrograms.
"""

import sys
import logging
from typing import List, Union, Tuple
from pathlib import Path
import pickle

import configargparse
import h5py
import numpy as np
from numpy import linalg as LA
from tqdm import tqdm
import matplotlib.pyplot as plt

from vasp.automatedPostprocessing.postprocessing_common import read_parameters_from_file
from vasp.automatedPostprocessing.postprocessing_h5py.spectrograms import butter_bandpass_filter
from vasp.automatedPostprocessing.postprocessing_h5py.postprocessing_common_h5py import create_transformed_matrix, \
    create_point_trace, create_xdmf_file, calculate_windowed_rms, get_eig, create_checkpoint_xdmf_file


def create_hi_pass_viz(formatted_data_folder: Path, output_folder: Path, mesh_path: Path, time_between_files: float,
                       dof_info: Union[dict, None], dof_info_amplitude: Union[dict, None], start_t: float,
                       quantity: str, lowcut: Union[float, List[float]] = 0,
                       highcut: Union[float, List[float]] = 100000, amplitude: bool = False,
                       filter_type: str = "bandpass", pass_stop_list: List[str] = [], overwrite: bool = False) -> None:
    """
    Create high-pass visualization data.

    Parameters:
        formatted_data_folder (Path): Path to the folder containing formatted data.
        output_folder (Path): Path to the folder where the output will be saved.
        mesh_path (Path): Path to the mesh file.
        time_between_files (float): Time between files.
        dof_info (Union[dict, None]): Dictionary containing the information about the degrees of freedom.
        dof_info_amplitude (Union[dict, None]): Dictionary containing the information about the degrees of freedom.
                                                Specifically for amplitude (scalar) data.
        start_t (float): Start time.
        quantity (str): Type of data (e.g., 'd', 'v', 'strain').
        lowcut (Union[int, List[int]]): Low-cut frequency or list of low-cut frequencies for multi-band filtering.
        highcut (Union[int, List[int]]): High-cut frequency or list of high-cut frequencies for multi-band filtering.
        amplitude (bool): Flag indicating whether to include amplitude in the visualization type.
        filter_type (str): Type of filter ('bandpass' or 'multiband').
        pass_stop_list (List[str]): List of strings indicating the pass or stop status for multi-band filtering.
        overwrite (bool): Flag indicating whether to overwrite existing files.

    Returns:
        None
    """
    # Convert single values to lists
    lowcut_list = [lowcut] if isinstance(lowcut, float) else lowcut
    highcut_list = [highcut] if isinstance(highcut, float) else highcut

    # Determine component names based on the quantity of interest

    if quantity in {"d", "v"}:
        component_names = ["mag", "x", "y", "z"]
    elif quantity == "strain":
        component_names = ["11", "12", "22", "23", "33", "31"]
    else:
        component_names = ["mag"]

    logging.info("--- Loading component data...")

    progress_bar = tqdm(total=len(component_names), desc="--- Loading components", unit=" component")

    components_data = []
    for component_name in component_names:
        file_pattern = f"{quantity}_{component_name}.npz"
        matching_files = [file for file in Path(formatted_data_folder).iterdir() if file_pattern in file.name]
        component_data = np.load(matching_files[0])["component"]
        components_data.append(component_data)

        progress_bar.set_postfix({"Component": component_name})
        progress_bar.update()
    progress_bar.close()

    # Create name for output file, define output path
    if quantity == "v":
        viz_type = "velocity"
    elif quantity == "d":
        viz_type = "displacement"
    elif quantity == "p":
        viz_type = "pressure"
    elif quantity == "strain":
        viz_type = "GreenLagrangeStrain"
    else:
        raise ValueError("Input 'd', 'v', 'p', 'strain', or for quantity")

    if filter_type == "multiband":
        assert len(lowcut_list) > 1 and len(highcut_list) > 1 and len(pass_stop_list) > 1, \
            "For multiband filtering, lowcut and highcut must be lists of length > 1."
        for low_freq, high_freq, pass_stop in zip(lowcut_list, highcut_list, pass_stop_list):
            viz_type = f"{viz_type}_{pass_stop}_{int(np.rint(low_freq))}_to_{int(np.rint(high_freq))}"
    else:
        viz_type = f"{viz_type}_{int(np.rint(lowcut))}_to_{int(np.rint(highcut))}"

    output_file_name = f"{viz_type}.h5"
    output_path = Path(output_folder) / output_file_name

    if amplitude and quantity != "strain":
        viz_type_amplitude = f"{viz_type}_amplitude"
        output_file_name_amplitude = f"{viz_type_amplitude}.h5"
        output_path_amplitude = Path(output_folder) / output_file_name_amplitude
    elif amplitude and quantity == "strain":
        viz_type_amplitude = f"{viz_type}_amplitude"
        viz_type_magnitude = f"{viz_type}_max_principal_amplitude"
        output_file_name_amplitude = f"{viz_type_amplitude}.h5"
        output_path_amplitude = Path(output_folder) / output_file_name_amplitude
        output_file_name_magnitude = f"{viz_type_magnitude}.h5"
        output_path_magnitude = Path(output_folder) / output_file_name_magnitude

    if output_folder.exists():
        logging.debug(f"--- The output path '{output_folder}' already exists.")
    else:
        logging.debug(f"--- Creating the output folder at '{output_folder}'.")
        output_folder.mkdir(parents=True, exist_ok=True)

    # Read in the mesh (for mps, this needs to be the wall only mesh):
    with h5py.File(mesh_path, "r") as fsi_mesh:
        # Count fluid and total nodes
        coord_array_fsi = fsi_mesh["mesh/coordinates"][:, :]
        topo_array_fsi = fsi_mesh["mesh/topology"][:, :]

        n_nodes_fsi = coord_array_fsi.shape[0]
        n_elements_fsi = topo_array_fsi.shape[0]
        n_cells_fsi = int(n_elements_fsi / 8)

        # Get number of timesteps
        num_ts = components_data[0].shape[1]
    logging.debug(f"--- Nodes: {n_nodes_fsi}, Elements: {n_elements_fsi}, Timesteps: {num_ts}")

    if output_path.exists() and not overwrite:
        logging.info(f"--- The file at {output_path} already exists; not overwriting. "
                     "Set overwrite=True to overwrite this file.")
        return

    # Remove old file if it exists
    if output_path.exists():
        logging.debug(f"--- The file at {output_path} already exists; overwriting.")
        output_path.unlink()

    if amplitude:
        if output_path_amplitude.exists() and not overwrite:
            logging.info(f"--- The file at {output_path_amplitude} already exists; not overwriting. "
                         "Set overwrite=True to overwrite this file.")
            return
        if output_path_amplitude.exists():
            logging.debug(f"--- The file at {output_path_amplitude} already exists; overwriting.")
            output_path_amplitude.unlink()

    # Create H5 file
    vector_data = h5py.File(output_path, "a")
    vector_data_amplitude = h5py.File(output_path_amplitude, "a") if amplitude else None
    vector_data_mps = h5py.File(output_path_magnitude, "a") if quantity == "strain" and amplitude else None

    logging.info("--- Creating mesh arrays for visualization...")

    # 1. Use fluid-only nodes for mesh creation.
    #    - Easiest method: Input the fluid-only mesh directly.
    #    - Alternative method: Modify the mesh topology, excluding nodes associated with solid elements.
    #    - For save_deg=2, consider using FEniCS to create a refined mesh with fluid and solid elements marked.
    #      This approach aims to maintain consistent node numbering with turtleFSI.
    if quantity in {"d", "v", "p"}:
        geo_array = vector_data.create_dataset("Mesh/0/mesh/geometry", (n_nodes_fsi, 3))
        geo_array[...] = coord_array_fsi
        topo_array = vector_data.create_dataset("Mesh/0/mesh/topology", (n_elements_fsi, 4), dtype="i")
        topo_array[...] = topo_array_fsi

    if quantity in {"d", "v", "p"} and amplitude:
        geo_array = vector_data_amplitude.create_dataset("Mesh/0/mesh/geometry", (n_nodes_fsi, 3))
        geo_array[...] = coord_array_fsi
        topo_array = vector_data_amplitude.create_dataset("Mesh/0/mesh/topology", (n_elements_fsi, 4), dtype="i")
        topo_array[...] = topo_array_fsi
    for idy in tqdm(range(components_data[0].shape[0]), desc="--- Filtering nodes", unit=" node"):
        if filter_type == "multiband":
            # Loop through the bands and either bandpass or bandstop filter them
            for low_freq, high_freq, filter_type_band in zip(lowcut_list, highcut_list, pass_stop_list):
                for component_index, component_data in enumerate(components_data):
                    # Apply butterworth bandpass or bandstop filter
                    components_data[component_index][idy, :] = \
                        butter_bandpass_filter(component_data[idy, :], lowcut=low_freq, highcut=high_freq,
                                               fs=int(1 / time_between_files) - 1, btype=filter_type_band)
        else:
            critical_freq = int(1 / time_between_files) / 2 - 1

            if isinstance(highcut, list):
                highcut = highcut[0]
            highcut = critical_freq if highcut >= critical_freq else highcut

            # Determine filter type based on lowcut value
            if isinstance(lowcut, list):
                lowcut = lowcut[0]
            filter_type_single = "lowpass" if lowcut < 0.1 else "bandpass"

            for component_index, component_data in enumerate(components_data):
                # Apply butterworth bandpass or bandstop filter
                components_data[component_index][idy, :] = \
                    butter_bandpass_filter(component_data[idy, :], lowcut=lowcut, highcut=highcut,
                                           fs=int(1 / time_between_files) - 1, btype=filter_type_single)

    # If amplitude is selected, calculate moving RMS amplitude for the results
    if amplitude:
        rms_magnitude = np.zeros((n_nodes_fsi, num_ts)) if quantity != "strain" else \
            np.zeros((int(n_cells_fsi * 4), num_ts))
        components_data_amplitude = np.zeros_like(components_data)
        # NOTE: Fixing the window size to 250 for now. It would be better to make this a parameter.
        window_size = 250  # This is approximately 1/4th of the value used in the spectrograms (992)
        for idy in tqdm(range(components_data[0].shape[0]), desc="--- Calculating amplitude", unit=" node"):
            for component_index, component_data in enumerate(components_data):
                components_data_amplitude[component_index][idy, :] = \
                    calculate_windowed_rms(component_data[idy, :], window_size)

    # 2. Loop through elements and load in the data
    for idx in tqdm(range(num_ts), desc="--- Saving data", unit=" timestep"):
        array_name = f"VisualisationVector/{idx}" if quantity in {"d", "v"} else f"{viz_type}/{viz_type}_{idx}"
        if quantity == "p":
            v_array = vector_data.create_dataset(array_name, (n_nodes_fsi, 1))
            v_array[:, 0] = components_data[0][:, idx]
            att_type = "Scalar"

            if amplitude:
                v_array_amplitude = vector_data_amplitude.create_dataset(array_name, (n_nodes_fsi, 1))
                v_array_amplitude[:, 0] = components_data_amplitude[0][:, idx]
                rms_magnitude[:, idx] = components_data[0][:, idx]

        elif quantity == "strain":
            # first open dof_info (dict)
            assert dof_info is not None
            for name, data in dof_info.items():
                dof_array = vector_data.create_dataset(f"{array_name}/{name}", data=data)
                dof_array[:] = data

            v_array = np.zeros((int(n_cells_fsi * 4), 9))
            v_array[:, 0] = components_data[0][:, idx]  # 11
            v_array[:, 1] = components_data[1][:, idx]  # 12
            v_array[:, 2] = components_data[5][:, idx]  # 31
            v_array[:, 3] = components_data[1][:, idx]  # 12
            v_array[:, 4] = components_data[2][:, idx]  # 22
            v_array[:, 5] = components_data[3][:, idx]  # 23
            v_array[:, 6] = components_data[5][:, idx]  # 31
            v_array[:, 7] = components_data[3][:, idx]  # 23
            v_array[:, 8] = components_data[4][:, idx]  # 33
            # flatten the array because strain is saved with `write_checkpoint` as one-dimensional array
            v_array_flat = vector_data.create_dataset(f"{array_name}/vector", (int(n_cells_fsi * 4 * 9), 1))
            v_array_flat[:, 0] = v_array.flatten()
            att_type = "Tensor"

            if amplitude:
                for name, data in dof_info.items():
                    array_name = f"{viz_type_amplitude}/{viz_type_amplitude}_{idx}"
                    dof_array = vector_data_amplitude.create_dataset(f"{array_name}/{name}", data=data)
                    dof_array[:] = data

                v_array_amplitude = np.zeros((int(n_cells_fsi * 4), 9))
                v_array_amplitude[:, 0] = components_data_amplitude[0][:, idx]  # 11
                v_array_amplitude[:, 1] = components_data_amplitude[1][:, idx]  # 12
                v_array_amplitude[:, 2] = components_data_amplitude[5][:, idx]  # 31
                v_array_amplitude[:, 3] = components_data_amplitude[1][:, idx]  # 12
                v_array_amplitude[:, 4] = components_data_amplitude[2][:, idx]  # 22
                v_array_amplitude[:, 5] = components_data_amplitude[3][:, idx]  # 23
                v_array_amplitude[:, 6] = components_data_amplitude[5][:, idx]  # 31
                v_array_amplitude[:, 7] = components_data_amplitude[3][:, idx]  # 23
                v_array_amplitude[:, 8] = components_data_amplitude[4][:, idx]  # 33
                # flatten the array because strain is saved with `write_checkpoint` as one-dimensional array
                v_array_flat_amplitude = vector_data_amplitude.create_dataset(f"{array_name}/vector",
                                                                              (int(n_cells_fsi * 4 * 9), 1))
                v_array_flat_amplitude[:, 0] = v_array_amplitude.flatten()
                att_type = "Tensor"

                # logging.info(f"--- Calculating eigenvalues for timestep #{idx}...")
                for iel in range(int(n_cells_fsi * 4)):
                    # Create the strain tensor
                    strain_tensor = np.array([
                        [components_data_amplitude[0][iel, idx], components_data_amplitude[1][iel, idx],
                         components_data_amplitude[5][iel, idx]],
                        [components_data_amplitude[1][iel, idx], components_data_amplitude[2][iel, idx],
                         components_data_amplitude[3][iel, idx]],
                        [components_data_amplitude[5][iel, idx], components_data_amplitude[3][iel, idx],
                         components_data_amplitude[4][iel, idx]]
                    ])
                    # Check if the strain tensor is all zeros. This is a shortcut to avoid taking eignevalues if
                    # the Strain tensor is all zeroes (outside the FSI region).
                    if np.all(np.abs(strain_tensor) < 1e-8):
                        MPS = 0.0
                    else:
                        # Calculate Maximum Principal Strain (MPS) for filtered strain tensor
                        MPS = get_eig(strain_tensor)

                    # Assign MPS to rms_magnitude
                    rms_magnitude[iel, idx] = MPS

                array_name = f"{viz_type_magnitude}/{viz_type_magnitude}_{idx}"
                assert dof_info_amplitude is not None
                for name, data in dof_info_amplitude.items():
                    dof_array = vector_data_mps.create_dataset(f"{array_name}/{name}", data=data)
                    dof_array[:] = data
                v_array_mps = vector_data_mps.create_dataset(f"{array_name}/vector",
                                                             (int(n_cells_fsi * 4), 1))
                v_array_mps[:, 0] = rms_magnitude[:, idx]

        else:
            v_array = vector_data.create_dataset(array_name, (n_nodes_fsi, 3))
            v_array[:, 0] = components_data[1][:, idx]
            v_array[:, 1] = components_data[2][:, idx]
            v_array[:, 2] = components_data[3][:, idx]
            att_type = "Vector"

            if amplitude:
                v_array_amplitude = vector_data_amplitude.create_dataset(array_name, (n_nodes_fsi, 3))
                v_array_amplitude[:, 0] = components_data_amplitude[1][:, idx]
                v_array_amplitude[:, 1] = components_data_amplitude[2][:, idx]
                v_array_amplitude[:, 2] = components_data_amplitude[3][:, idx]
                # Take magnitude of RMS Amplitude, this way you don't lose any directional changes
                rms_magnitude[:, idx] = LA.norm(v_array_amplitude, axis=1)

    vector_data.close()

    # 3. Create xdmf file for visualization
    if quantity in {"d", "v", "p"}:
        create_xdmf_file(num_ts, time_between_files, start_t, n_elements_fsi,
                         n_nodes_fsi, att_type, viz_type, output_folder)
    elif quantity == "strain":
        assert dof_info is not None
        n_nodes = dof_info["mesh/geometry"].shape[0]
        create_checkpoint_xdmf_file(num_ts, time_between_files, start_t, n_elements_fsi,
                                    n_nodes, att_type, viz_type, output_folder)
    else:
        NotImplementedError(f"Quantity {quantity} not implemented.")

    # If amplitude is selected, create xdmf file for visualization
    if amplitude:
        if quantity in {"d", "v", "p"}:
            create_xdmf_file(num_ts, time_between_files, start_t, n_elements_fsi,
                             n_nodes_fsi, att_type, viz_type_amplitude, output_folder)
        elif quantity == "strain":
            assert dof_info is not None
            n_nodes = dof_info["mesh/geometry"].shape[0]
            create_checkpoint_xdmf_file(num_ts, time_between_files, start_t, n_elements_fsi,
                                        n_nodes, att_type, viz_type_amplitude, output_folder)
        else:
            NotImplementedError(f"Quantity {quantity} not implemented.")

    # If amplitude is selected, save the percentiles of magnitude of RMS amplitude to file
    if amplitude:
        logging.info("--- Saving amplitude percentiles to file...")

        # Save average amplitude, 95th percentile amplitude, and maximum amplitude
        output_amplitudes = np.zeros((num_ts, 13))

        for idx in tqdm(range(num_ts), desc="--- Saving data", unit=" timestep"):
            output_amplitudes[idx, 0] = idx * time_between_files
            output_amplitudes[idx, 1] = np.percentile(rms_magnitude[:, idx], 95)
            output_amplitudes[idx, 2] = np.percentile(rms_magnitude[:, idx], 5)
            output_amplitudes[idx, 3] = np.percentile(rms_magnitude[:, idx], 100)
            output_amplitudes[idx, 4] = np.percentile(rms_magnitude[:, idx], 0)
            output_amplitudes[idx, 5] = np.percentile(rms_magnitude[:, idx], 50)
            output_amplitudes[idx, 6] = np.percentile(rms_magnitude[:, idx], 90)
            output_amplitudes[idx, 7] = np.percentile(rms_magnitude[:, idx], 10)
            output_amplitudes[idx, 8] = np.percentile(rms_magnitude[:, idx], 97.5)
            output_amplitudes[idx, 9] = np.percentile(rms_magnitude[:, idx], 2.5)
            output_amplitudes[idx, 10] = np.percentile(rms_magnitude[:, idx], 99)
            output_amplitudes[idx, 11] = np.percentile(rms_magnitude[:, idx], 1)
            output_amplitudes[idx, 12] = np.argmax(rms_magnitude[:, idx])

        # File names and paths
        amp_file = Path(output_folder) / f"{viz_type}.csv"
        amp_graph_file = amp_file.with_suffix(".png")

        # Header for the CSV file
        header = "time (s), 95th percentile amplitude, 5th percentile amplitude, maximum amplitude, " + \
                 "minimum amplitude, average amplitude, 90th percentile amplitude, 10th percentile amplitude, " + \
                 "97.5th percentile amplitude, 2.5th percentile amplitude, 99th percentile amplitude, " + \
                 "1st percentile amplitude, ID of node with max amplitude"

        # Save the data to a CSV file
        np.savetxt(amp_file, output_amplitudes, delimiter=",", header=header)

        # Plot and save figure
        plt.plot(output_amplitudes[:, 0], output_amplitudes[:, 3], label="Maximum amplitude")
        plt.plot(output_amplitudes[:, 0], output_amplitudes[:, 1], label="95th percentile amplitude")
        plt.plot(output_amplitudes[:, 0], output_amplitudes[:, 5], label="50th percentile amplitude")
        plt.title("Amplitude Percentiles")
        plt.ylabel("Amplitude (units depend on d, v or p)")
        plt.xlabel("Simulation Time (s) - Start Time (s)")
        plt.legend()

        logging.info(f"--- Saving amplitude figure at: {amp_graph_file}")
        plt.savefig(amp_graph_file)
        plt.close()

        if quantity == "strain":
            logging.info("--- Saving MPS amplitude to file...")
            assert dof_info is not None
            n_nodes = dof_info["mesh/geometry"].shape[0]
            att_type = "Scalar"
            # NOTE: this is for max-principal strain
            create_checkpoint_xdmf_file(num_ts, time_between_files, start_t, n_elements_fsi,
                                        n_nodes, att_type, viz_type_magnitude, output_folder)


def parse_command_line_args() -> Tuple[Path, Path, int, int, float, float, str,
                                       List[int], List[int], str, bool, bool, int]:
    """
    Parse arguments from the command line.

    Returns:
        Tuple[Path, Path, int, int, float, float, str, List[int], List[int], str, bool, int]:
        Parsed command line arguments in the following order:
        (folder, mesh_path, save_deg, stride, start_time, end_time, quantity, bands, point_ids,
        filter_type, amplitude, overwrite, log_level)
    """
    parser = configargparse.ArgumentParser(description=__doc__,
                                           formatter_class=configargparse.RawDescriptionHelpFormatter)

    parser.add_argument("--folder", type=Path, required=True, default=None,
                        help="Path to simulation results. This argument is required.")
    parser.add_argument("--mesh-path", type=Path, default=None,
                        help="Path to the mesh file (default: <folder>/Mesh/mesh.h5)")
    parser.add_argument("-c", "--config", is_config_file=True,
                        help="Path to configuration file.")
    parser.add_argument("--save-deg", type=int, default=None,
                        help="Specify the save_deg used during the simulation, i.e., whether the intermediate P2 nodes "
                             "were saved. Entering save_deg=1 when the simulation was run with save_deg=2 will result "
                             "in using only the corner nodes in postprocessing. If not provided, the default value "
                             "will be read from the parameters file in the simulation folder. For quantity 'p' "
                             "(pressure), the default will be 1.")
    parser.add_argument("--stride", type=int, default=1,
                        help="Desired frequency of output data (i.e. to output every second step, use stride=2). "
                             "Default is 1.")
    parser.add_argument("--start-time", type=float, default=0.0,
                        help="Start time of simulation (in seconds). Default is 0. For strain, do not provide a "
                             "start time since the user has already specified the start time when creating the "
                             "h5 file for the displacement.")
    parser.add_argument("--end-time", type=float, default=None,
                        help="End time of simulation (in seconds). Default is to end at the last time step."
                             "For strain, do not provide an end time since the user has already specified the end "
                             "time when creating the h5 file for the displacement.")
    parser.add_argument("-q", "--quantity", type=str, default="v",
                        help="Quantity to postprocess. Choose 'v' for velocity, 'd' for displacement, 'p' for pressure,"
                             " or 'strain' for strain. Default is 'v'.")
    parser.add_argument("--bands", nargs="+", type=int, default=[25, 100000],
                        help="Input lower and upper band for band-pass filtered displacement, in a list of pairs. For "
                             "example: --bands 100 150 175 200, gives you band-pass filtered visualization for the "
                             "band between 100 and 150, and another visualization for the band between 175 and 200."
                             "Default is [25, 100000].")
    parser.add_argument("--point-ids", nargs="+", type=int, default=[0, 1],
                        help="Input list of points IDs. Default is [0, 1].")
    parser.add_argument("--filter-type", type=str, default="bandpass",
                        help="Type of filter ('bandpass' or 'multiband'). Default is 'bandpass'.")
    parser.add_argument("--amplitude", action="store_true",
                        help="Flag indicating whether to compute the amplitude of the filtered results.")
    parser.add_argument("--overwrite", action="store_true",
                        help="Flag indicating whether to overwrite existing files.")
    parser.add_argument("--log-level", type=int, default=20,
                        help="Specify the log level (default is 20, which is INFO)")

    args = parser.parse_args()

    # Check if the specified folder exists
    if not args.folder.exists():
        logging.error(f"ERROR: The specified folder '{args.folder}' does not exist.")
        sys.exit(-1)

    # Set default mesh path if not provided
    args.mesh_path = args.folder / "Mesh" / "mesh.h5" if args.mesh_path is None else args.mesh_path

    # Check if the specified mesh path exists
    if not args.mesh_path.exists():
        logging.error(f"ERROR: The specified mesh path '{args.mesh_path}' does not exist.")
        sys.exit(-1)

    # Check if the length of bands is even
    if len(args.bands) % 2 != 0:
        logging.error("ERROR: The 'bands' argument should contain pairs of lower and upper bands.")
        sys.exit(-1)

    return args.folder, args.mesh_path, args.save_deg, args.stride, args.start_time, args.end_time, args.quantity, \
        args.bands, args.point_ids, args.filter_type, args.amplitude, args.overwrite, args.log_level


def main():
    folder, mesh_path, save_deg, stride, start_time, end_time, quantity, bands, point_ids, filter_type, amplitude, \
        overwrite, log_level = parse_command_line_args()

    # Create logger and set log level
    logging.basicConfig(level=log_level, format="%(message)s")

    # Load parameters from default_parameters.json
    parameters = read_parameters_from_file(folder)

    # Extract parameters
    dt = parameters["dt"]
    end_time = end_time if end_time is not None else parameters["T"]
    save_deg = save_deg if save_deg is not None else (1 if quantity == 'p' else parameters["save_deg"])
    fluid_domain_id = parameters["dx_f_id"]
    solid_domain_id = parameters["dx_s_id"]

    case_name = folder.parent.name
    visualization_path = folder / "Visualization"

    num_bands = int(len(bands) / 2)
    lower_freq = np.zeros(num_bands)
    higher_freq = np.zeros(num_bands)

    # Initialize pass_stop_list to determine band-pass or band-stop for each band.
    # Default is to let the main high-frequency band pass and remove narrow bands from "rocking modes".
    pass_stop_list = []

    # Iterate over bands and determine pass or stop based on frequency range
    for i in range(num_bands):
        lower_freq[i] = float(bands[2 * i])
        higher_freq[i] = float(bands[2 * i + 1])

        # Check if the frequency range is greater than 1000 for band-pass, else band-stop
        if higher_freq[i] - lower_freq[i] > 1000:
            pass_stop_list.append("pass")  # Let all high frequencies pass initially for multiband
        else:
            pass_stop_list.append("stop")  # Stop the specified narrowbands

    mesh_name_suffix = "" if save_deg == 1 else "_refined"

    # Original mesh path
    original_mesh_path = mesh_path

    # Updated mesh paths
    mesh_path = mesh_path.with_name(f"{mesh_path.stem}{mesh_name_suffix}{mesh_path.suffix}")
    mesh_path_solid = mesh_path.with_name(f"{mesh_path.stem}_solid.h5")  # Needed for strain

    # Paths for corner-node input mesh (save_deg=1)
    mesh_path_sd1 = original_mesh_path
    mesh_path_solid_sd1 = mesh_path_sd1.with_name(f"{mesh_path_sd1.stem}_solid.h5")  # Needed for strain

    # Create a formatted data folder name based on parameters
    formatted_data_folder_name = f"npz_{start_time}s_to_{end_time}s_stride_{stride}_save_deg_{save_deg}"
    formatted_data_folder = folder / formatted_data_folder_name

    # Visualization folder for separate domains
    visualization_separate_domain_folder = folder / "Visualization_separate_domain"

    try:
        file_path_d = visualization_separate_domain_folder / "d_solid.h5"
        assert file_path_d.exists(), f"Displacement file {file_path_d} not found."
        logging.info("--- displacement is for the solid domain only \n")
    except AssertionError:
        file_path_d = visualization_separate_domain_folder / "d.h5"
        assert file_path_d.exists(), f"Displacement file {file_path_d} not found."
        logging.info("--- displacement is for the entire domain \n")
        mesh_path_solid = mesh_path

    # Visualization folder for stress and strain
    visualization_stress_strain_folder = folder / "StressStrain"

    # Visualization folder for high-pass filtered results
    visualization_hi_pass_folder = folder / "Visualization_hi_pass"

    # Create output folder and filenames
    output_file_name = f"{quantity}_mag.npz" if quantity != "strain" else f"{quantity}_11.npz"
    formatted_data_path = formatted_data_folder / output_file_name

    logging.info("--- Creating high-pass visualizations...")
    logging.info(f"--- Start time: {start_time}; End time: {end_time}\n")

    logging.info("--- Preparing data...")
    dof_info = None
    dof_info_amplitude = None
    if formatted_data_path.exists() and quantity != "strain":
        logging.info(f"--- Formatted data already exists at: {formatted_data_path}\n")
    elif formatted_data_path.exists() and quantity == "strain":
        logging.info(f"--- Formatted data already exists at: {formatted_data_path}\n")
        dof_info_path = formatted_data_folder / "dof_info.pkl"
        with open(dof_info_path, "rb") as f:
            dof_info = pickle.load(f)
        dof_info_amplitude_path = formatted_data_folder / "dof_info_amplitude.pkl"
        with open(dof_info_amplitude_path, "rb") as f:
            dof_info_amplitude = pickle.load(f)
    else:
        # Determine mesh paths based on save_deg
        if save_deg == 1:
            mesh_path_solid = mesh_path_solid_sd1
        if quantity == "strain":
            _, dof_info, dof_info_amplitude = create_transformed_matrix(visualization_stress_strain_folder,
                                                                        formatted_data_folder, mesh_path_solid,
                                                                        case_name, start_time,
                                                                        end_time, quantity,
                                                                        fluid_domain_id, solid_domain_id, stride)
        else:
            # Make the output h5 files with quantity magnitudes
            create_transformed_matrix(visualization_path, formatted_data_folder, mesh_path, case_name, start_time,
                                      end_time, quantity, fluid_domain_id, solid_domain_id, stride)

    # Get the desired time between output files (reduce output frequency by "stride")
    # time_between_output_files = time_between_input_files * stride
    time_between_output_files = dt * stride  # FIXME: Is it okay to use dt here instead of time_between_input_files?

    if quantity != "strain":
        logging.info("--- Creating point traces...")
        try:
            create_point_trace(str(formatted_data_folder), str(visualization_separate_domain_folder), point_ids,
                               time_between_output_files, start_time, quantity)
        except Exception as e:
            logging.error(f"ERROR: Failed to create point traces: {e}")

        # Create high-pass visualizations for each frequency band
        for low_freq, high_freq in zip(lower_freq, higher_freq):
            logging.info(f"\n--- Creating high-pass visualization {low_freq}-{high_freq} with amplitude...")
            create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,
                               time_between_output_files, dof_info, dof_info_amplitude, start_time, quantity,
                               low_freq, high_freq, amplitude=amplitude, overwrite=overwrite)

        # Create multiband high-pass visualizations
        if filter_type == "multiband":
            logging.info("\n--- Creating multiband high-pass visualization with amplitude...")
            create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path,
                               time_between_output_files, dof_info, dof_info_amplitude, start_time,
                               quantity, lower_freq, higher_freq, amplitude=amplitude, filter_type="multiband",
                               pass_stop_list=pass_stop_list, overwrite=overwrite)
    elif quantity == "strain":
        logging.info(f"--- Creating high-pass visualizations for {quantity}...")
        for i in range(len(lower_freq)):
            create_hi_pass_viz(formatted_data_folder, visualization_hi_pass_folder, mesh_path_solid,
                               time_between_output_files, dof_info, dof_info_amplitude, start_time,
                               quantity, lower_freq[i], higher_freq[i], amplitude=amplitude, overwrite=overwrite)

    logging.info(f"\n--- High-pass visualizations saved at: {visualization_hi_pass_folder}\n")


if __name__ == '__main__':
    main()
