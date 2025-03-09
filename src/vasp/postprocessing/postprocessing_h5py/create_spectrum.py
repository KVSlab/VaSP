# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script creates a power spectrum plot.
"""

import logging
from pathlib import Path
from typing import Optional, Union

import numpy as np
import matplotlib.pyplot as plt

from vasp.postprocessing.postprocessing_h5py import spectrograms as spec
from vasp.postprocessing.postprocessing_common import read_parameters_from_file


def create_spectrum(case_name: str, quantity: str, df, start_t: float, end_t: float, num_windows_per_sec: float,
                    overlap_frac: float, window: str, lowcut: float, min_color: float, max_color: float,
                    image_folder: Union[str, Path], flow_rate_file: Optional[str] = None,
                    amplitude_file: Optional[str] = None, power_scaled: bool = False) -> None:
    """
    Create a power spectrum plot and save the results as an image and CSV file.

    Args:
        case_name (str): Name of the case.
        quantity (str): Type of data to be processed.
        df: Input DataFrame containing relevant data.
        start_t (float): Desired start time of the output files.
        end_t (float): Desired end time of the output files.
        num_windows_per_sec (float): Number of windows per second.
        overlap_frac (float): Fraction of overlap between consecutive windows.
        window (str): Type of window function to use.
        lowcut (float): Cutoff frequency for the high-pass filter.
        min_color (float): Minimum value for the color range.
        max_color (float): Maximum value for the color range.
        image_folder (str or Path): Folder to save the spectrum image and CSV file.
        flow_rate_file (str): File name for flow rate data.
        amplitude_file (str): File name for amplitude data.
        power_scaled (bool): Whether to use power scaling in the PSD calculation.

    Returns:
        None: Saves the spectrum plot as an image and CSV file.
    """
    # Get sampling constants
    logging.info("--- Getting sampling constants...")
    T, _, fs = spec.get_sampling_constants(df, start_t, end_t)

    # PSD calculation
    Pxx_array, freq_array = spec.get_psd(df, fs, scaling="spectrum")

    # Plot PSD
    logging.info("\n--- Plotting PSD...")
    plt.plot(freq_array, np.log(Pxx_array))
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('input units^2/Hz')

    logging.info("--- Saving PSD plot and CSV data...")
    plot_name = f"{quantity}_psd_no_filter_{case_name}"
    path_to_fig = Path(image_folder) / f"{plot_name}.png"
    path_csv = Path(image_folder) / f"{plot_name}.csv"

    # Save the figure
    plt.savefig(path_to_fig)

    # Save CSV data
    data_csv = np.stack((freq_array, np.log(Pxx_array)), axis=1)
    np.savetxt(path_csv, data_csv, header="Freqs(Hz),spectrum", delimiter=",")

    logging.info(f"--- PSD plot saved at: {path_to_fig}")
    logging.info(f"--- CSV data saved at: {path_csv}")


def main():
    # Load in case-specific parameters
    args = spec.read_command_line_spec()

    # Create logger and set log level
    logging.basicConfig(level=args.log_level, format="%(message)s")

    # Load parameters from default_parameters.json
    parameters = read_parameters_from_file(args.folder)

    # Extract parameters
    fsi_region = parameters["fsi_region"] if args.fsi_region is None else args.fsi_region
    fluid_domain_id = parameters["dx_f_id"]
    solid_domain_id = parameters["dx_s_id"]
    end_time = args.end_time if args.end_time is not None else parameters["T"]
    save_deg = args.save_deg if args.save_deg is not None else (1 if args.quantity == 'p' else parameters["save_deg"])

    # Create or read in spectrogram dataframe
    quantity, df, case_name, image_folder, visualization_hi_pass_folder = \
        spec.read_spectrogram_data(args.folder, args.mesh_path, save_deg, args.stride, args.start_time,
                                   end_time, args.n_samples, args.sampling_region,
                                   args.fluid_sampling_domain_id, args.solid_sampling_domain_id, fsi_region,
                                   args.quantity, args.interface_only, args.component, args.point_id, fluid_domain_id,
                                   solid_domain_id, sampling_method=args.sampling_method)

    # Should these files be used?
    # amplitude_file = Path(visualization_hi_pass_folder) / args.amplitude_file_name
    # flow_rate_file = Path(args.folder) / args.flow_rate_file_name

    # Create spectrograms
    create_spectrum(case_name, quantity, df, args.start_time, end_time, args.num_windows_per_sec,
                    args.overlap_frac, args.window, args.lowcut, args.min_color, args.max_color, image_folder,
                    flow_rate_file=None, amplitude_file=None, power_scaled=False)

    if args.sampling_method == "SinglePoint":
        spec.sonify_point(case_name, quantity, df, args.start_time, args.end_time, args.overlap_frac, args.lowcut,
                          image_folder)


if __name__ == '__main__':
    main()
