# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#   2023 Daniel Macdonald

"""
This file contains helper functions for creating spectrograms.
"""

import sys
import timeit
import configargparse
import logging
from pathlib import Path
from typing import Union, Optional, Literal

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, spectrogram, periodogram
from scipy.interpolate import RectBivariateSpline

from fsipy.automatedPostprocessing.postprocessing_h5py.chroma_filters import normalize, chroma_filterbank
from fsipy.automatedPostprocessing.postprocessing_h5py.postprocessing_common_h5py import create_transformed_matrix, \
    read_npz_files, get_surface_topology_coords, get_coords, get_interface_ids, \
    get_domain_ids_specified_region
from fsipy.automatedPostprocessing.postprocessing_common import get_domain_ids


def read_command_line_spec() -> configargparse.Namespace:
    """
    Read arguments from the command line using ConfigArgParse.

    Returns:
        ArgumentParser: Parsed command-line arguments.
    """
    parser = configargparse.ArgumentParser(formatter_class=configargparse.RawDescriptionHelpFormatter)

    parser.add_argument("--folder", type=Path, required=True, default=None,
                        help="Path to simulation results")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file (default: <folder_path>/Mesh/mesh.h5)")
    parser.add_argument('--save-deg', type=int, default=None,
                        help="Specify the save_deg used during the simulation, i.e., whether the intermediate P2 nodes "
                             "were saved. Entering save_deg=1 when the simulation was run with save_deg=2 will result "
                             "in using only the corner nodes in postprocessing.")
    parser.add_argument('--stride', type=int, default=1,
                        help="Desired frequency of output data (i.e. to output every second step, use stride=2)")
    parser.add_argument('--start-time', type=float, default=0.0,
                        help="Start time of the simulation (in seconds).")
    parser.add_argument('--end-time', type=float, default=0.05,
                        help="End time of the simulation (in seconds).")
    parser.add_argument('--lowcut', type=float, default=25,
                        help="Cutoff frequency (Hz) for the high-pass filter.")
    parser.add_argument('--ylim', type=float, default=800,
                        help="Set the y-limit of the spectrogram graph.")
    parser.add_argument('--sampling-region', type=str, default="sphere",
                        help="Specify the sampling region. Choose 'sphere' to sample within a sphere or 'domain' to "
                             "sample within a specified domain.")
    parser.add_argument('--fluid-sampling-domain-id', type=int, default=1,
                        help="Domain ID for the fluid region to be sampled. Input a labelled mesh with this ID.")
    parser.add_argument('--solid-sampling-domain-id', type=int, default=2,
                        help="Domain ID for the solid region to be sampled. Input a labelled mesh with this ID.")
    parser.add_argument('--r-sphere', type=float, default=1000000,
                        help="Radius of the sphere used to include points for spectrogram.")
    parser.add_argument('--x-sphere', type=float, default=0.0,
                        help="X-coordinate of the center of the sphere used to include points for spectrogram (in "
                             "meters).")
    parser.add_argument('--y-sphere', type=float, default=0.0,
                        help="Y-coordinate of the center of the sphere used to include points for spectrogram (in "
                             "meters).")
    parser.add_argument('--z-sphere', type=float, default=0.0,
                        help="Z-coordinate of the center of the sphere used to include points for spectrogram (in "
                             "meters).")
    parser.add_argument('--dvp', type=str, default="v",
                        help="Quantity to postprocess. Choose 'v' for velocity, 'd' for displacement, 'p' for "
                             "pressure, or 'wss' for wall shear stress.")
    parser.add_argument('--Re_a', type=float, default=0.0,
                        help="Assuming linearly increasing Reynolds number: Re(t) = Re_a*t + Re_b. If both Re_a and "
                             "Re_b are 0, the plot won't be against Reynolds number.")
    parser.add_argument('--Re_b', type=float, default=0.0,
                        help="Assuming linearly increasing Reynolds number: Re(t) = Re_a*t + Re_b. If both Re_a and "
                             "Re_b are 0, the plot won't be against Reynolds number.")
    parser.add_argument('--interface-only', action='store_true',
                        help="Generate spectrogram only for the fluid-solid interface. If present, interface-only "
                             "spectrogram will be generated; otherwise, the volumetric spectrogram will include all "
                             "fluid in the sac or all nodes through the wall.")
    parser.add_argument('--component', type=str, default="mag",
                        help="Component of the data to visualize. Choose 'x', 'y', 'z', or 'mag' (magnitude).")
    parser.add_argument('--sampling-method', type=str, default="RandomPoint",
                        help="Sampling method for spectrogram generation. Choose from 'RandomPoint' (random nodes), "
                             "'SinglePoint' (single point specified by 'point_id'), or 'Spatial' (ensures uniform "
                             "spatial sampling, e.g., in the case of fluid boundary layer, the sampling will not bias "
                             "towards the boundary layer).")
    parser.add_argument('--n-samples', type=int, default=10000,
                        help="Number of samples to generate spectrogram data (ignored for SinglePoint sampling).")
    parser.add_argument('--point-id', type=int, default=-1000000,
                        help="Point ID for SinglePoint sampling. Ignored for other sampling methods.")

    parser.add_argument('--overlap-frac', type=float, default=0.75,
                        help="Fraction of overlap between adjacent windows.")
    parser.add_argument('--window', type=str, default="blackmanharris",
                        help="Window function to be used for spectrogram computation.")
    parser.add_argument('--num-windows-per-sec', type=int, default=4,
                        help="Number of windows per second for spectrogram computation.")
    parser.add_argument('--thresh-val', type=int, default=None,
                        help="Threshold value for the spectrogram. Default is determined based on the 'dvp' argument: "
                             "if 'd', default is -42; if 'v', default is -20; if 'p', default is -5; if 'wss', "
                             "default is -18.")
    parser.add_argument('--max-plot', type=int, default=None,
                        help="Maximum value for plotting the spectrogram. Default is determined based on the 'dvp' "
                             "argument: if 'd', default is -30; if 'v', default is -7; if 'p', default is 5; if 'wss', "
                             ", default is 0.")
    parser.add_argument('--amplitude-file-name', type=Path, default=None,
                        help="Name of the file containing displacement amplitude data.")
    parser.add_argument('--flow-rate-file-name', type=Path, default="MCA_10",
                        help="Name of the file containing flow rate data. Default is 'MCA_10'.")
    parser.add_argument("--log-level", type=int, default=20,
                        help="Specify the log level (default is 20, which is INFO)")

    args = parser.parse_args()

    # Set default mesh path if not provided
    args.mesh_path = args.folder / "Mesh" / "mesh.h5" if args.mesh_path is None else args.mesh_path

    # Set default thresh_val, max_plot and amplitude_file_name based in the dvp argument
    if args.dvp == "d":
        args.thresh_val = args.thresh_val if args.thresh_val is not None else -42
        args.max_plot = args.max_plot if args.max_plot is not None else -30
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"displacement_amplitude_{args.lowcut}_to_100000.csv"
    elif args.dvp == "v":
        args.thresh_val = args.thresh_val if args.thresh_val is not None else -20
        args.max_plot = args.max_plot if args.max_plot is not None else -7
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"velocity_amplitude_{args.lowcut}_to_100000.csv"
    elif args.dvp == "p":
        args.thresh_val = args.thresh_val if args.thresh_val is not None else -5
        args.max_plot = args.max_plot if args.max_plot is not None else 5
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"pressure_amplitude_{args.lowcut}_to_100000.csv"
    elif args.dvp == "wss":
        args.thresh_val = args.thresh_val if args.thresh_val is not None else -18
        args.max_plot = args.max_plot if args.max_plot is not None else 0
        args.amplitude_file_name = args.amplitude_file_name if args.amplitude_file_name is not None \
            else f"wss_amplitude_{args.lowcut}_to_100000.csv"
    else:
        logging.error(f"ERROR: Invalid value for dvp - {args.dvp}. Please use 'd', 'v', 'p', or 'wss'.")
        sys.exit(-1)

    return args


def read_spectrogram_data(folder: Union[str, Path], mesh_path: Union[str, Path], save_deg: int, stride: int,
                          start_t: float, end_t: float, n_samples: int, ylim: float, sampling_region: str,
                          fluid_sampling_domain_id: int, solid_sampling_domain_id: int, fsi_region: list[float],
                          dvp: str, interface_only: bool, component: str, point_id: int,
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
        ylim (float): Y-axis limit of the spectrogram graph.
        sampling_region (str): Region for sampling data ("sphere" or "domain").
        fluid_sampling_domain_id (int): Domain ID for fluid sampling (used when sampling_region="domain").
        solid_sampling_domain_id (int): Domain ID for solid sampling (used when sampling_region="domain").
        fsi_region (list): x, y, and z coordinates of sphere center and radius of the sphere (used when
            sampling_region="sphere").
        dvp (str): Type of data to be processed.
        interface_only (bool): Whether to include only interface ID's.
        component (str): Component of the data to be visualized.
        point_id (int): Point ID (used when sampling_method="SinglePoint").
        sampling_method (str): Method for sampling data ("RandomPoint", "SinglePoint", or "Spatial").
        fluid_domain_id (int or list): ID of the fluid domain
        solid_domain_id (int or list): ID of the solid domain

    Returns:
        tuple: (Processed data type, DataFrame, Case name, Image folder, Hi-pass visualization folder).
    """
    start_time = timeit.default_timer()

    folder_path = Path(folder)
    case_name = folder_path.parent.name
    visualization_path = folder_path / "Visualization"

    # 1. Get names of relevant directories, files

    mesh_name_suffix = "" if save_deg == 1 else "_refined"
    mesh_path = Path(mesh_path)
    mesh_path = mesh_path.with_name(f"{mesh_path.stem}{mesh_name_suffix}{mesh_path.suffix}")
    mesh_path_fluid = mesh_path.with_name(f"{mesh_path.stem}_fluid.h5")  # Needed for formatting SPI data

    formatted_data_folder_name = f"res_{case_name}_stride_{stride}t{start_t}_to_{end_t}save_deg_{save_deg}"
    formatted_data_folder = folder_path / formatted_data_folder_name
    visualization_separate_domain_folder = folder_path / "Visualization_separate_domain"
    visualization_hi_pass_folder = folder_path / "Visualization_hi_pass"

    image_folder = folder_path / "Spectrograms"
    image_folder.mkdir(parents=True, exist_ok=True)

    output_file_name = f"{case_name}_{dvp}_{component}.npz"
    formatted_data_path = formatted_data_folder / output_file_name

    elapsed_time = timeit.default_timer() - start_time

    # 2. Prepare data

    start_time = timeit.default_timer()

    # If the output file exists, don't re-make it
    if formatted_data_path.exists():
        logging.info(f'Formatted data already exists at: {formatted_data_path}')
    elif dvp == "wss":
        create_transformed_matrix(visualization_separate_domain_folder, formatted_data_folder, mesh_path_fluid,
                                  case_name, start_t, end_t, dvp, fluid_domain_id, solid_domain_id, stride)
    else:
        # Make the output h5 files with dvp magnitudes
        create_transformed_matrix(visualization_path, formatted_data_folder, mesh_path,
                                  case_name, start_t, end_t, dvp, fluid_domain_id, solid_domain_id, stride)

    elapsed_time = timeit.default_timer() - start_time
    logging.info(f"Made matrix in {elapsed_time:.6f} seconds")

    # 3. Read data

    start_time = timeit.default_timer()

    # For spectrograms, we only want the magnitude
    df = read_npz_files(formatted_data_path)

    elapsed_time = timeit.default_timer() - start_time
    logging.info(f"Read matrix in {elapsed_time:.6f} seconds")

    # 4. Process data and get ID's

    start_time = timeit.default_timer()

    # We want to find the points in the sac, so we use a sphere to roughly define the sac.
    x_sphere, y_sphere, z_sphere, r_sphere = fsi_region
    sac_center = np.array([x_sphere, y_sphere, z_sphere])

    if dvp == "wss":
        wss_output_file = visualization_separate_domain_folder / "WSS_ts.h5"
        surface_elements, coords = get_surface_topology_coords(wss_output_file)
    else:
        coords = get_coords(mesh_path)

    if sampling_region == "sphere":
        # Get wall and fluid ID's
        fluid_ids, wall_ids, all_ids = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)
        interface_ids = get_interface_ids(mesh_path, fluid_domain_id, solid_domain_id)
        sphere_ids = find_points_in_sphere(sac_center, r_sphere, coords)

        # Get nodes in sac only
        all_ids = np.intersect1d(sphere_ids, all_ids)
        fluid_ids = np.intersect1d(sphere_ids, fluid_ids)
        wall_ids = np.intersect1d(sphere_ids, wall_ids)
        interface_ids = np.intersect1d(sphere_ids, interface_ids)
    elif sampling_region == "domain":
        # To use this option, input a mesh with domain markers and indicate which domain represents the desired fluid
        # region for the spectrogram (fluid_sampling_domain_id) and which domain represents the desired solid region
        # (solid_sampling_domain_id).
        fluid_ids, wall_ids, all_ids = \
            get_domain_ids_specified_region(mesh_path, fluid_sampling_domain_id, solid_sampling_domain_id)
        interface_ids = np.intersect1d(fluid_ids, wall_ids)
    else:
        raise ValueError(f"Invalid sampling method '{sampling_region}'. Please specify 'sphere' or 'domain'.")

    if dvp == "wss":
        # For wss spectrogram, we use all the nodes within the sphere because the input df only includes the wall
        region_ids = sphere_ids
    elif interface_only:
        # Use only the interface IDs
        region_ids = interface_ids
        dvp = dvp + "_interface"
    elif dvp == "d":
        # For displacement spectrogram, we need to take only the wall IDs
        region_ids = wall_ids
    else:
        # For pressure and velocity spectrogram, we need to take only the fluid IDs
        region_ids = fluid_ids

    elapsed_time = timeit.default_timer() - start_time
    logging.info(f"Got ID's in {elapsed_time:.6f} seconds")

    # 5. Sample data (reduce compute time by random sampling)

    start_time = timeit.default_timer()

    if sampling_method == "RandomPoint":
        idx_sampled = np.random.choice(region_ids, n_samples)
    elif sampling_method == "SinglePoint":
        idx_sampled = np.array([point_id])
        case_name = f"{case_name}_{sampling_method}_{point_id}"
        logging.info(f"Single Point spectrogram for point: {point_id}")
    elif sampling_method == "Spatial":
        # See old code for implementation if needed
        raise NotImplementedError("Spatial sampling method is not implemented.")
    else:
        raise ValueError(f"Invalid sampling method: {sampling_method}. Please choose from 'RandomPoint', "
                         "'SinglePoint', or 'Spatial'.")

    elapsed_time = timeit.default_timer() - start_time
    logging.info(f"Obtained sample points in {elapsed_time:.6f} seconds")

    start_time = timeit.default_timer()

    df = df.iloc[idx_sampled]
    dvp = f"{dvp}_{component}_{n_samples}"

    elapsed_time = timeit.default_timer() - start_time
    logging.info(f"Sampled dataframe in {elapsed_time:.6f} seconds")

    return dvp, df, case_name, image_folder, visualization_hi_pass_folder


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

        for each in range(dfNearest.shape[0]):
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
        for each in range(dfNearest.shape[0]):
            row = dfNearest.iloc[each]
            freqs, bins, Pxx = spectrogram(row, fs=fsamp, nperseg=NFFT, noverlap=int(overlapFrac * NFFT),
                                           nfft=2 * NFFT, window=window, scaling=scaling)
            if each == 0:
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
    logging.info(f"Pxx_scaled max: {max_val}")
    logging.info(f"Pxx_scaled max: {min_val}")
    logging.info(f"Pxx threshold: {lower_thresh}")
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

    for row in range(df.shape[0]):
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
