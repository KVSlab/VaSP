# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#   2023 Daniel Macdonald

"""
This file contains helper functions for creating spectrograms.
"""

import sys
import configargparse
import logging
from pathlib import Path
from typing import Union, Optional, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, periodogram
from scipy.interpolate import RectBivariateSpline
from scipy.io import wavfile
from tqdm import tqdm

from vasp.automatedPostprocessing.postprocessing_h5py.chroma_filters import normalize, chroma_filterbank
from vasp.automatedPostprocessing.postprocessing_h5py.postprocessing_common_h5py import create_transformed_matrix, \
    read_npz_files, get_surface_topology_coords, get_coords, get_interface_ids, \
    get_domain_ids_specified_region
from vasp.automatedPostprocessing.postprocessing_common import get_domain_ids


def read_command_line_spec() -> configargparse.Namespace:
    """
    Read arguments from the command line using ConfigArgParse.

    Returns:
        Namespace: Parsed command-line arguments.
    """
    parser = configargparse.ArgumentParser(formatter_class=configargparse.RawDescriptionHelpFormatter)

    parser.add_argument("--folder", type=Path, required=True, default=None,
                        help="Path to simulation results")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file (default: <folder_path>/Mesh/mesh.h5)")
    parser.add_argument('-c', '--config', is_config_file=True,
                        help='Path to configuration file')
    parser.add_argument('--save-deg', type=int, default=None,
                        help="Specify the save_deg used during the simulation, i.e., whether the intermediate P2 nodes "
                             "were saved. Entering save_deg=1 when the simulation was run with save_deg=2 will result "
                             "in using only the corner nodes in postprocessing.")
    parser.add_argument('--stride', type=int, default=1,
                        help="Desired frequency of output data (i.e. to output every second step, use stride=2)")
    parser.add_argument('--start-time', type=float, default=0.0,
                        help="Start time of the simulation (in seconds).")
    parser.add_argument('--end-time', type=float, default=None,
                        help="End time of the simulation (in seconds).")
    parser.add_argument('--lowcut', type=float, default=25,
                        help="Cutoff frequency (Hz) for the high-pass filter.")
    parser.add_argument('--ylim', type=float, default=None,
                        help="Set the y-limit of the spectrogram graph (Hz).")
    parser.add_argument('--sampling-region', type=str, default="sphere",
                        help="Specify the sampling region. Choose 'sphere' to sample within a sphere, 'domain' to "
                             "sample within a specified domain or 'box' to sample within a box.")
    parser.add_argument('--fluid-sampling-domain-id', type=int, default=1,
                        help="Domain ID for the fluid region to be sampled. Input a labelled mesh with this ID. Used "
                             "only when sampling region is 'domain'.")
    parser.add_argument('--solid-sampling-domain-id', type=int, default=2,
                        help="Domain ID for the solid region to be sampled. Input a labelled mesh with this ID. Used "
                             "only when sampling region is 'domain'.")
    parser.add_argument('-q', '--quantity', type=str, default="v",
                        help="Quantity to postprocess. Choose 'v' for velocity, 'd' for displacement, 'p' for "
                             "pressure, or 'wss' for wall shear stress.")
    parser.add_argument('--interface-only', action='store_true',
                        help="Generate spectrogram only for the fluid-solid interface. If present, interface-only "
                             "spectrogram will be generated; otherwise, the volumetric spectrogram will include all "
                             "fluid in the sac or all nodes through the wall.")
    parser.add_argument('--component', type=str, default="mag",
                        help="Component of the data to visualize. Choose 'x', 'y', 'z', 'mag' (magnitude) or 'all' "
                             "(to combine all components).")
    parser.add_argument('--sampling-method', type=str, default="RandomPoint",
                        help="Sampling method for spectrogram generation. Choose from 'RandomPoint' (random nodes), "
                             "'PointList' (list of points specified by '--point-ids'), or 'Spatial' (ensures uniform "
                             "spatial sampling, e.g., in the case of fluid boundary layer, the sampling will not bias "
                             "towards the boundary layer).")
    parser.add_argument('--n-samples', type=int, default=10000,
                        help="Number of samples to generate spectrogram data (ignored for PointList sampling).")
    parser.add_argument("--point-ids", nargs="+", type=int, default=[-1000000],
                        help="Input list of points for spectrograms a list. For "
                             "example: --point-ids 1003 1112 17560, gives an average spectrogram for those points"
                             "Default is [-1000000].")
    parser.add_argument('--overlap-frac', type=float, default=0.75,
                        help="Fraction of overlap between adjacent windows.")
    parser.add_argument('--window', type=str, default="blackmanharris",
                        help="Window function to be used for spectrogram computation. "
                             "Choose from window types available at "
                             "https://docs.scipy.org/doc/scipy/reference/signal.windows.html. "
                             "Default is 'blackmanharris'.")
    parser.add_argument('--num-windows-per-sec', type=int, default=4,
                        help="Number of windows per second for spectrogram computation.")
    parser.add_argument('--min-color', type=int, default=None,
                        help="Minimum color value for plotting the spectrogram. Default is determined based on the "
                             "'quantity' argument: if 'd', default is -42; if 'v', default is -20; if 'p', default is "
                             "-5; if 'wss', default is -18.")
    parser.add_argument('--max-color', type=int, default=None,
                        help="Maximum color value for plotting the spectrogram. Default is determined based on the "
                             "'quantity' argument: if 'd', default is -30; if 'v', default is -7; if 'p', default is "
                             "5; if 'wss', default is 0.")
    parser.add_argument('--amplitude-file-name', type=Path, default=None,
                        help="Name of the file containing displacement amplitude data.")
    parser.add_argument('--flow-rate-file-name', type=Path, default="MCA_10",
                        help="Name of the file containing flow rate data. Default is 'MCA_10'.")
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

    # Set default min_color, max_color and amplitude_file_name based in the quantity argument
    if args.quantity == "d":
        args.min_color = args.min_color if args.min_color is not None else -42
        args.max_color = args.max_color if args.max_color is not None else -30
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"displacement_amplitude_{args.lowcut}_to_100000.csv"
    elif args.quantity == "v":
        args.min_color = args.min_color if args.min_color is not None else -20
        args.max_color = args.max_color if args.max_color is not None else -7
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"velocity_amplitude_{args.lowcut}_to_100000.csv"
    elif args.quantity == "p":
        args.min_color = args.min_color if args.min_color is not None else -5
        args.max_color = args.max_color if args.max_color is not None else 5
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"pressure_amplitude_{args.lowcut}_to_100000.csv"
    elif args.quantity == "wss":
        args.min_color = args.min_color if args.min_color is not None else -18
        args.max_color = args.max_color if args.max_color is not None else 0
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"wss_amplitude_{args.lowcut}_to_100000.csv"
    else:
        logging.error(f"ERROR: Invalid value for quantity - {args.quantity}. Please use 'd', 'v', 'p', or 'wss'.")
        sys.exit(-1)

    return args


def read_spectrogram_data(folder: Union[str, Path], mesh_path: Union[str, Path], save_deg: int, stride: int,
                          start_t: float, end_t: float, n_samples: int, sampling_region: str,
                          fluid_sampling_domain_id: int, solid_sampling_domain_id: int, fsi_region: list[float],
                          quantity: str, interface_only: bool, component: str, point_ids: list[int],
                          fluid_domain_id: Union[int, list[int]], solid_domain_id: Union[int, list[int]],
                          sampling_method: str = "RandomPoint"):
    """
    Read spectrogram data and perform processing steps.

    Args:
        folder (str or Path): Path to simulation results.
        mesh_path (str or Path): Path to the mesh file.
        save_deg (int): Degree of mesh refinement.
        stride (int): Desired frequency of output data.
        start_t (float): Start time for data processing.
        end_t (float): End time for data processing.
        n_samples (int): Number of samples.
        sampling_region (str): Region for sampling data ("sphere", "domain" or "box").
        fluid_sampling_domain_id (int): Domain ID for fluid sampling (used when sampling_region="domain").
        solid_sampling_domain_id (int): Domain ID for solid sampling (used when sampling_region="domain").
        fsi_region (list): x, y, and z coordinates of sphere center and radius of the sphere (used when
            sampling_region="sphere"). In case of sampling_region="box", the list should contain [x_min, x_max,
            y_min, y_max, z_min, z_max]. The box is defined by the minimum and maximum values of x, y, and z.
        quantity (str): Quantity to postprocess.
        interface_only (bool): Whether to include only interface ID's.
        component (str): Component of the data to be visualized.
        point_ids (int): List of Point IDs (used when sampling_method="PointList").
        sampling_method (str): Method for sampling data ("RandomPoint", "PointList", or "Spatial").
        fluid_domain_id (int or list): ID of the fluid domain
        solid_domain_id (int or list): ID of the solid domain

    Returns:
        tuple: (Processed data type, DataFrame, Case name, Image folder, Hi-pass visualization folder).
    """
    folder_path = Path(folder)
    case_name = folder_path.parent.name
    visualization_path = folder_path / "Visualization"

    logging.info(f"--- Processing folder path {folder_path}\n")

    mesh_name_suffix = "" if save_deg == 1 else "_refined"
    mesh_path = Path(mesh_path)
    mesh_path = mesh_path.with_name(f"{mesh_path.stem}{mesh_name_suffix}{mesh_path.suffix}")
    mesh_path_fluid = mesh_path.with_name(f"{mesh_path.stem}_fluid.h5")  # Needed for formatting SPI data

    formatted_data_folder_name = f"npz_{start_t}s_to_{end_t}s_stride_{stride}_save_deg_{save_deg}"
    formatted_data_folder = folder_path / formatted_data_folder_name
    visualization_separate_domain_folder = folder_path / "Visualization_separate_domain"
    visualization_hi_pass_folder = folder_path / "Visualization_hi_pass"

    image_folder = folder_path / "Spectrograms"
    image_folder.mkdir(parents=True, exist_ok=True)

    logging.info("\n--- Processing data and getting ID's")

    if quantity == "wss":
        wss_output_file = visualization_separate_domain_folder / "WSS_ts.h5"
        surface_elements, coords = get_surface_topology_coords(wss_output_file)
    else:
        coords = get_coords(mesh_path)

    if sampling_region == "sphere":
        # We want to find the points in the sac, so we use a sphere to roughly define the sac.
        x_sphere, y_sphere, z_sphere, r_sphere = fsi_region
        sac_center = np.array([x_sphere, y_sphere, z_sphere])
        # Get solid and fluid ID's
        fluid_ids, solid_ids, all_ids = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)
        interface_ids = get_interface_ids(mesh_path, fluid_domain_id, solid_domain_id)
        sphere_ids = find_points_in_sphere(sac_center, r_sphere, coords)

        # Get nodes in sac only
        all_ids = np.intersect1d(sphere_ids, all_ids)
        fluid_ids = np.intersect1d(sphere_ids, fluid_ids)
        solid_ids = np.intersect1d(sphere_ids, solid_ids)
        interface_ids = np.intersect1d(sphere_ids, interface_ids)
    elif sampling_region == "domain":
        # To use this option, input a mesh with domain markers and indicate which domain represents the desired fluid
        # region for the spectrogram (fluid_sampling_domain_id) and which domain represents the desired solid region
        # (solid_sampling_domain_id).
        fluid_ids, solid_ids, all_ids = \
            get_domain_ids_specified_region(mesh_path, fluid_sampling_domain_id, solid_sampling_domain_id)
        interface_ids = np.intersect1d(fluid_ids, solid_ids)
    elif sampling_region == "box":
        x_min, x_max, y_min, y_max, z_min, z_max = fsi_region
        fluid_ids, solid_ids, all_ids = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)
        box_ids = find_points_in_box(x_min, x_max, y_min, y_max, z_min, z_max, coords)

        # Get nodes in box only
        all_ids = np.intersect1d(box_ids, all_ids)
        fluid_ids = np.intersect1d(box_ids, fluid_ids)
        solid_ids = np.intersect1d(box_ids, solid_ids)

    else:
        raise ValueError(f"Invalid sampling method '{sampling_region}'. Please specify 'sphere', 'domain' or 'box'.")

    if quantity == "wss":
        # For wss spectrogram, we use all the nodes within the sphere because the input df only includes the wall
        region_ids = sphere_ids
    elif interface_only:
        # Use only the interface IDs
        region_ids = interface_ids
    elif quantity == "d":
        # For displacement spectrogram, we need to take only the solid IDs
        region_ids = solid_ids
    else:
        # For pressure and velocity spectrogram, we need to take only the fluid IDs
        region_ids = fluid_ids

    logging.info(f"\n--- Sampling data using '{sampling_method}' sampling method")

    if sampling_method == "RandomPoint":
        idx_sampled = np.random.choice(region_ids, n_samples)
        quantity_component_name = f"{quantity}_{component}_n_samples_{n_samples}"
    elif sampling_method == "PointList":
        idx_sampled = np.array(point_ids)
        case_name = f"{case_name}_{sampling_method}_{point_ids}"
        quantity_component_name = f"{quantity}_{component}"
        logging.info(f"--- Single Point spectrogram for point: {point_ids}")
    elif sampling_method == "Spatial":
        # See old code for implementation if needed
        raise NotImplementedError("Spatial sampling method is not implemented.")
    else:
        raise ValueError(f"Invalid sampling method: {sampling_method}. Please choose from 'RandomPoint', "
                         "'PointList', or 'Spatial'.")

    logging.info("--- Obtained sample point IDs\n")

    # for "all" components, we read in each component from npz file (creating this file if it doesnt exist)
    # then we append all component dataframes together to create a spectrogram representing all components
    if component == "all":
        component_list = ["x", "y", "z"]
    else:
        component_list = [component]  # if only one component selected (mag, x, y, or z)

    for id_comp, component_name in enumerate(component_list):

        output_file_name = f"{quantity}_{component_name}.npz"
        formatted_data_path = formatted_data_folder / output_file_name

        logging.info("--- Preparing data")

        # If the output file exists, don't re-make it
        if formatted_data_path.exists():
            logging.info(f'--- Formatted data already exists at: {formatted_data_path}\n')
        else:
            if quantity == "wss":
                create_transformed_matrix(visualization_separate_domain_folder, formatted_data_folder, mesh_path_fluid,
                                          case_name, start_t, end_t, quantity, fluid_domain_id, solid_domain_id, stride)
            else:
                # Make the output h5 files with quantity magnitudes
                create_transformed_matrix(visualization_path, formatted_data_folder, mesh_path,
                                          case_name, start_t, end_t, quantity, fluid_domain_id, solid_domain_id, stride)

        logging.info("--- Reading data")

        # Read in data for selected component
        df = read_npz_files(formatted_data_path)
        df = df.iloc[idx_sampled]

        # for first component
        if id_comp == 0:
            df_selected_components = df.copy()
        else:  # if "all" components selected
            df_selected_components = df_selected_components._append(df)

    return quantity_component_name, df_selected_components, case_name, image_folder, visualization_hi_pass_folder


def find_points_in_sphere(center: np.ndarray, radius: float, coords: np.ndarray) -> np.ndarray:
    """
    Find points within a sphere defined by its center and radius.

    Args:
        center (np.ndarray): The center of the sphere as a 1D NumPy array with shape (3,).
        radius (float): The radius of the sphere.
        coords (np.ndarray): The coordinates of mesh nodes as a 2D NumPy array with shape (n, 3).

    Returns:
        np.ndarray: Indices of points within the sphere.
    """
    # Calculate vector from center to each node in the mesh
    vector_point = coords - center

    # Calculate distance from each mesh node to center
    radius_nodes = np.linalg.norm(vector_point, axis=1)

    # Get all points in the sphere
    points_in_sphere = np.where(radius_nodes < radius)[0]

    return points_in_sphere


def find_points_in_box(x_min: float, x_max: float, y_min: float, y_max: float, z_min: float, z_max: float,
                       coords: np.ndarray) -> np.ndarray:
    """
    Find points within a box defined by its minimum and maximum x, y, and z coordinates.

    Args:
        x_min (float): Minimum x coordinate of the box.
        x_max (float): Maximum x coordinate of the box.
        y_min (float): Minimum y coordinate of the box.
        y_max (float): Maximum y coordinate of the box.
        z_min (float): Minimum z coordinate of the box.
        z_max (float): Maximum z coordinate of the box.
        coords (np.ndarray): The coordinates of mesh nodes as a 2D NumPy array with shape (n, 3).

    Returns:
        np.ndarray: Indices of points within the box.
    """
    # Get all points in the box
    points_in_box = np.where((coords[:, 0] > x_min) & (coords[:, 0] < x_max) &
                             (coords[:, 1] > y_min) & (coords[:, 1] < y_max) &
                             (coords[:, 2] > z_min) & (coords[:, 2] < z_max))[0]

    return points_in_box


def shift_bit_length(x: int) -> int:
    """
    Round up to the nearest power of 2.

    Args:
        x (int): Input integer.

    Returns:
        int: The smallest power of 2 greater than or equal to the input.

    Author:
        Daniel Macdonald
    """
    return 1 << (x - 1).bit_length()


def get_psd(dfNearest: pd.DataFrame, fsamp: float, scaling: str = "density") -> tuple:
    """
    Calculate the Power Spectral Density (PSD) of a DataFrame of signals.

    Args:
        dfNearest (pd.DataFrame): DataFrame containing signals in rows.
        fsamp (float): Sampling frequency of the signals.
        scaling (str, optional): Scaling applied to the PSD. Default is "density".

    Returns:
        tuple: A tuple containing the mean PSD matrix and the corresponding frequency values.
    """
    if dfNearest.shape[0] > 1:
        Pxx_matrix = np.zeros_like(periodogram(dfNearest.iloc[0], fs=fsamp, window='blackmanharris')[1])

        for each in tqdm(range(dfNearest.shape[0]), desc="--- Calculating PSD", unit="row"):
            row = dfNearest.iloc[each]
            f, Pxx = periodogram(row, fs=fsamp, window='blackmanharris', scaling=scaling)
            Pxx_matrix += Pxx

        Pxx_mean = Pxx_matrix / dfNearest.shape[0]
    else:
        f, Pxx_mean = periodogram(dfNearest.iloc[0], fs=fsamp, window='blackmanharris')

    return Pxx_mean, f


def get_spectrogram(dfNearest: pd.DataFrame, fsamp: float, nWindow: int, overlapFrac: float, window: str,
                    start_t: float, end_t: float, scaling: str = 'spectrum', interpolate: bool = False) -> tuple:
    """
    Calculates spectrogram.

    Args:
        dfNearest (pd.DataFrame): DataFrame of shape (num_points, num_timesteps).
        fsamp (float): Sampling frequency.
        nWindow (int): Number of samples per window.
        overlapFrac (float): Fraction of overlap between windows.
        window (str): Window function for spectrogram.
        start_t (float): Start time for the spectrogram.
        end_t (float): End time for the spectrogram.
        scaling (str, optional): Scaling for the spectrogram ('density' or 'spectrum'). Default is 'spectrum'.
        interpolate (bool, optional): Perform interpolation. Default is False.

    Returns:
        tuple: Power Spectral Density matrix (Pxx_mean), frequency values (freqs), and time bins (bins).

    Author:
        Daniel Macdonald
    """
    NFFT = shift_bit_length(int(dfNearest.shape[1] / nWindow))

    if dfNearest.shape[0] > 1:
        Pxx_matrix = None  # Initialize Pxx_matrix
        for each in tqdm(range(dfNearest.shape[0]), desc="--- Calculating spectrogram", unit="row"):
            row = dfNearest.iloc[each]
            freqs, bins, Pxx = spectrogram(row, fs=fsamp, nperseg=NFFT, noverlap=int(overlapFrac * NFFT),
                                           nfft=2 * NFFT, window=window, scaling=scaling)
            if Pxx_matrix is None:
                Pxx_matrix = Pxx
            else:
                Pxx_matrix = Pxx_matrix + Pxx
    else:
        freqs, bins, Pxx_matrix = spectrogram(dfNearest.iloc[0], fs=fsamp, nperseg=NFFT,
                                              noverlap=int(overlapFrac * NFFT), nfft=2 * NFFT,
                                              window=window, scaling=scaling)

    Pxx_mean = Pxx_matrix / dfNearest.shape[0] if dfNearest.shape[0] > 1 else Pxx_matrix

    if interpolate:
        interp_spline = RectBivariateSpline(freqs, bins, Pxx_mean, kx=3, ky=3)
        bins = np.linspace(start_t, end_t, 100)
        Pxx_mean = interp_spline(freqs, bins)

    assert Pxx_mean is not None, "Pxx_mean is None"
    Pxx_mean[Pxx_mean < 0] = 1e-16

    return Pxx_mean, freqs, bins


def spectrogram_scaling(Pxx_mean: np.ndarray, lower_thresh: float) -> tuple:
    """
    Scale a spectrogram.

    Args:
        Pxx_mean (np.ndarray): Power Spectral Density matrix.
        lower_thresh (float): Threshold value for scaling.

    Returns:
        tuple: Scaled Power Spectral Density matrix (Pxx_scaled), maximum value after scaling (max_val),
            minimum value after scaling (min_val), and the lower threshold used for scaling.

    Author:
        Daniel Macdonald
    """
    Pxx_scaled = np.log(Pxx_mean)
    max_val = np.max(Pxx_scaled)
    min_val = np.min(Pxx_scaled)
    logging.info(f"--- Spectrogram Scaling: Max: {max_val}, Min: {min_val}, Threshold: {lower_thresh}")

    Pxx_threshold_indices = Pxx_scaled < lower_thresh
    Pxx_scaled[Pxx_threshold_indices] = lower_thresh

    return Pxx_scaled, max_val, min_val, lower_thresh


def butter_bandpass(lowcut: float, highcut: float, fs: float, order: int = 5, btype: str = 'band') -> tuple:
    """
    Design a Butterworth bandpass, bandstop, highpass, or lowpass filter.

    Args:
        lowcut (float): Cutoff frequency for low cut.
        highcut (float): Cutoff frequency for high cut (ignored if btype is 'highpass').
        fs (float): Sampling frequency in samples per second.
        order (int, optional): Order of the filter. Default is 5.
        btype (str, optional): Type of filter ('band', 'stop', 'highpass', 'lowpass', or 'bandpass'). Default is 'band'.

    Returns:
        tuple: Numerator (b) and denominator (a) coefficients of the filter.
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq

    if btype == 'band':
        b, a = butter(order, [low, high], btype='band')
    elif btype == 'stop':
        b, a = butter(order, [low, high], btype='bandstop')
    elif btype == 'highpass':
        b, a = butter(order, low, btype='highpass')
    elif btype == 'lowpass':
        b, a = butter(order, high, btype='lowpass')
    elif 'pass' in btype:
        b, a = butter(order, [low, high], btype='bandpass')

    return b, a


def butter_bandpass_filter(data: np.ndarray, lowcut: float = 25.0, highcut: float = 15000.0, fs: float = 2500.0,
                           order: int = 5, btype: str = 'band') -> np.ndarray:
    """
    Apply a Butterworth bandpass, bandstop, highpass, or lowpass filter to the input data.

    Args:
        data (np.ndarray): Input data to filter.
        lowcut (float, optional): Low cutoff frequency. Default is 25.0.
        highcut (float, optional): High cutoff frequency. Default is 15000.0.
        fs (float, optional): Sampling frequency. Default is 2500.0.
        order (int, optional): Order of the filter. Default is 5.
        btype (str, optional): Type of filter ('band', 'stop', 'highpass', 'lowpass'). Default is 'band'.

    Returns:
        np.ndarray: Filtered data.

    Author:
        Daniel Macdonald
    """
    b, a = butter_bandpass(lowcut, highcut, fs, order=order, btype=btype)
    y = filtfilt(b, a, data)
    return y


def filter_time_data(df: pd.DataFrame, fs: float, lowcut: float = 25.0, highcut: float = 15000.0,
                     order: int = 6, btype: str = 'highpass') -> pd.DataFrame:
    """
    Apply a Butterworth highpass, lowpass, bandpass, or bandstop filter to the time series data in a DataFrame.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        fs (float): Sampling frequency.
        lowcut (float, optional): Low cutoff frequency. Default is 25.0.
        highcut (float, optional): High cutoff frequency. Default is 15000.0.
        order (int, optional): Order of the filter. Default is 6.
        btype (str, optional): Type of filter ('highpass', 'lowpass', 'bandpass', 'bandstop'). Default is 'highpass'.

    Returns:
        pd.DataFrame: DataFrame containing filtered time series data.

    Author:
        Daniel Macdonald
    """
    df_filtered = df.copy()

    for row in tqdm(range(df.shape[0]), desc="--- Filtering rows", unit="row"):
        df_filtered.iloc[row] = butter_bandpass_filter(df.iloc[row], lowcut=lowcut, highcut=highcut, fs=fs,
                                                       order=order, btype=btype)

    return df_filtered


def compute_average_spectrogram(df: pd.DataFrame, fs: float, nWindow: int, overlapFrac: float, window: str,
                                start_t: float, end_t: float, thresh: float, scaling: str = "spectrum",
                                filter_data: bool = False, thresh_method: str = "new") -> tuple:
    """
    Compute the average spectrogram for a DataFrame of time series data.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        fs (float): Sampling frequency.
        nWindow (int): Number of samples per window.
        overlapFrac (float): Fraction of overlap between windows.
        window (str): Window function for spectrogram.
        start_t (float): Start time for the spectrogram.
        end_t (float): End time for the spectrogram.
        thresh (float): Threshold value for scaling.
        scaling (str, optional): Scaling for the spectrogram ('density' or 'spectrum'). Default is 'spectrum'.
        filter_data (bool, optional): Apply a Butterworth highpass filter to the data. Default is False.
        thresh_method (str, optional): Method for thresholding ('old', 'log_only', 'new'). Default is 'new'.

    Returns:
        tuple: Time bins (bins), frequency values (freqs), scaled Power Spectral Density matrix (Pxx_scaled),
            maximum value after scaling (max_val), minimum value after scaling (min_val), and the lower threshold
            used for scaling.

    Author:
        Daniel Macdonald
    """
    if filter_data:
        df = filter_time_data(df, fs)

    Pxx_mean, freqs, bins = get_spectrogram(df, fs, nWindow, overlapFrac, window, start_t, end_t, scaling)

    if thresh_method == "old":
        Pxx_scaled, max_val, min_val, lower_thresh = spectrogram_scaling(Pxx_mean, thresh)
    elif thresh_method == "log_only":
        Pxx_scaled = np.log(Pxx_mean)
        max_val = np.max(Pxx_scaled)
        min_val = np.min(Pxx_scaled)
        lower_thresh = "None"
    else:
        Pxx_scaled = Pxx_mean
        max_val = np.max(Pxx_scaled)
        min_val = np.min(Pxx_scaled)
        lower_thresh = "None"

    return bins, freqs, Pxx_scaled, max_val, min_val, lower_thresh


def plot_spectrogram(fig1: plt.Figure, ax1: plt.Axes, bins: np.ndarray, freqs: np.ndarray, Pxx: np.ndarray,
                     ylim: Optional[float] = None, title: Optional[str] = None, convert_a: float = 0.0,
                     convert_b: float = 0.0, x_label: Optional[str] = None,
                     color_range: Optional[list[float]] = None) -> None:
    """
    Plot a spectrogram.

    Args:
        fig1 (plt.Figure): Matplotlib figure to plot on.
        ax1 (plt.Axes): Matplotlib axes to plot on.
        bins (np.ndarray): Time bins.
        freqs (np.ndarray): Frequency values.
        Pxx (np.ndarray): Power spectral density values.
        ylim (float, optional): Maximum frequency to display on the y-axis.
        title (str, optional): Title of the plot. Default is None.
        convert_a (float, optional): Conversion factor for the x-axis. Default is 0.0.
        convert_b (float, optional): Offset for the x-axis conversion. Default is 0.0.
        x_label (str, optional): Label for the x-axis. Default is None.
        color_range (list[float], optional): Range for the color scale. Default is None.

    Returns:
        None
    """
    if color_range is None:
        im = ax1.pcolormesh(bins, freqs, Pxx, shading='gouraud')
    else:
        im = ax1.pcolormesh(bins, freqs, Pxx, shading='gouraud', vmin=color_range[0], vmax=color_range[1])

    fig1.colorbar(im, ax=ax1)

    if title is not None:
        ax1.set_title('{}'.format(title), y=1.08)
    if x_label is not None:
        ax1.set_xlabel(x_label)
    ax1.set_ylabel('Frequency [Hz]')
    if ylim is not None:
        ax1.set_ylim((0, ylim))

    if convert_a > 0.000001 or convert_b > 0.000001:
        ax2 = ax1.twiny()
        ax2.set_xlim(ax1.get_xlim())

        def time_convert(x):
            return np.round(x * convert_a + convert_b, decimals=2)

        ax2.set_xticks(ax1.get_xticks())
        ax2.set_xticklabels(time_convert(ax1.get_xticks()))
        if x_label is not None:
            ax2.set_xlabel(x_label)


def chromagram_from_spectrogram(Pxx: np.ndarray, fs: float, n_fft: int, n_chroma: int = 24,
                                norm: Union[bool, str] = True) -> np.ndarray:
    """
    Calculate chromagram from a spectrogram.

    Args:
        Pxx (np.ndarray): Input spectrogram.
        fs (float): Sampling frequency.
        n_fft (int): Number of FFT points.
        n_chroma (int, optional): Number of chroma bins. Default is 24.
        norm (bool, optional): Normalize chroma. Options are 'max', 'sum', or False for no normalization.
           Default is True.

    Returns:
        np.ndarray: Chromagram.

    Author:
        Daniel Macdonald
    """
    # Calculate chroma filterbank
    chromafb = chroma_filterbank(
        sr=fs,
        n_fft=n_fft,
        tuning=0.0,
        n_chroma=n_chroma,
        ctroct=5,
        octwidth=2,
    )

    # Calculate chroma
    chroma = np.dot(chromafb, Pxx)

    # Normalize
    if norm == "max":
        # Normalize chroma so that the maximum value is 1 in each column
        chroma = normalize(chroma, norm=np.inf, axis=0)
    elif norm == "sum":
        # Normalize chroma so that each column sums to 1
        chroma = (chroma / np.sum(chroma, axis=0))  # Chroma must sum to one for entropy fuction to work
    else:
        logging.info("Raw chroma selected")

    return chroma


def calc_chroma_entropy(chroma: np.ndarray, n_chroma: int) -> np.ndarray:
    """
    Calculate chroma entropy.

    Args:
        chroma (np.ndarray): Chromagram.
        n_chroma (int): Number of chroma bins.

    Returns:
        np.ndarray: Chroma entropy.

    Author:
        Daniel Macdonald
    """
    chroma_entropy = -np.sum(chroma * np.log(chroma), axis=0) / np.log(n_chroma)
    return 1 - chroma_entropy


def plot_chromagram(fig1: plt.Figure, ax1: plt.Axes, bins: np.ndarray, chroma: np.ndarray, title: Optional[str] = None,
                    path: Optional[Union[str, Path]] = None, convert_a: float = 1.0, convert_b: float = 0.0,
                    x_label: Optional[str] = None,
                    shading: Optional[Literal['flat', 'nearest', 'gouraud', 'auto']] = 'gouraud',
                    color_range: Optional[list[float]] = None) -> None:
    """
    Plot a chromagram.

    Args:
        fig1 (plt.Figure): Matplotlib figure to plot on.
        ax1 (plt.Axes): Matplotlib axes to plot on.
        bins (np.ndarray): Time bins.
        chroma (np.ndarray): Chromagram values.
        title (str, optional): Title of the plot. Default is None.
        path (str, optional): Path to save the figure. Default is None.
        convert_a (float, optional): Conversion factor for the x-axis. Default is 1.0.
        convert_b (float, optional): Offset for the x-axis conversion. Default is 0.0.
        x_label (str, optional): Label for the x-axis. Default is None.
        shading (str, optional): Shading style for the plot. Default is 'gouraud'.
        color_range (list[float], optional): Range for the color scale. Default is None.

    Returns:
        None
    """
    bins = bins * convert_a + convert_b

    chroma_y = np.linspace(0, 1, chroma.shape[0])

    if color_range is None:
        im = ax1.pcolormesh(bins, chroma_y, chroma, shading=shading)
    else:
        im = ax1.pcolormesh(bins, chroma_y, chroma, shading=shading, vmin=color_range[0], vmax=color_range[1])

    fig1.colorbar(im, ax=ax1)

    ax1.set_ylabel('Chroma')

    if title is not None:
        ax1.set_title('{}'.format(title))
    if x_label is not None:
        ax1.set_xlabel(x_label)

    if path is not None:
        fig1.savefig(path)
        path_csv = Path(path).with_suffix(".csv")
        np.savetxt(path_csv, chroma, delimiter=",")


def get_sampling_constants(df: pd.DataFrame, start_t: float, end_t: float) -> tuple:
    """
    Get sampling constants such as period, number of samples, and sample rate.

    Args:
        df (pd.DataFrame): Input DataFrame containing time series data.
        start_t (float): Start time of the data.
        end_t (float): End time of the data.

    Returns:
        tuple: Period, number of samples, and sample rate.

    Author:
        Daniel Macdonald
    """
    T = end_t - start_t
    nsamples = df.shape[1]
    fs = nsamples / T
    return T, nsamples, fs


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
    T, _, fs = get_sampling_constants(df, start_t, end_t)

    # High-pass filter dataframe for spectrogram
    df_filtered = filter_time_data(df, fs, lowcut=lowcut, highcut=15000.0, order=6, btype='highpass')

    num_points = df_filtered.shape[0]
    max_val_df = np.max(df_filtered)
    y2 = np.zeros(df_filtered.shape[1])
    for i in range(num_points):
        y2 += df_filtered.iloc[i] / max_val_df  # Add waveforms for each point together, normalized by overall max value

    y2 = y2 / num_points  # Normalize by number of points

    sound_filename = f"{quantity}_sound_{case_name}.wav"
    path_to_sound = Path(image_folder) / sound_filename

    wavfile.write(path_to_sound, int(fs), y2)
