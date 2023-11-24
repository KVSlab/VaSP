# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script creates a combined spectrogram from three component-wise spectrograms by averaging each component. This
avoids artifacts associated with taking the spectrogram of the magnitude when the direction reverses.
"""

import logging
from pathlib import Path
from typing import Union, Optional

import numpy as np
import matplotlib.pyplot as plt

from fsipy.automatedPostprocessing.postprocessing_h5py import spectrograms as spec


def create_spectrogram_from_components(folder: Union[str, Path], quantity: str, n_samples: int, thresh_val: float,
                                       max_plot: float, ylim: Optional[float] = None) -> None:
    """
    Create a composite spectrogram by averaging three component-wise spectrograms.

    Args:
        folder (str or Path): Path to the case folder.
        quantity (str): Type of data to be processed.
        n_samples (int): Number of samples.
        thresh_val (float): Threshold value for the color range.
        max_plot (float): Maximum value for the color range.
        ylim (float, optional): Y-axis limit for the plot.

    Returns:
        None: Saves the composite spectrogram as a PNG file and CSV file.
    """
    folder_path = Path(folder)
    image_folder = folder_path / "Spectrograms"

    # Get all csv files (make sure there is only one for each component)
    x_csv_files = list(Path(image_folder).rglob(f"{quantity}_x_*spectrogram.csv"))
    y_csv_files = list(Path(image_folder).rglob(f"{quantity}_y_*spectrogram.csv"))
    z_csv_files = list(Path(image_folder).rglob(f"{quantity}_z_*spectrogram.csv"))

    # Create the spec name
    spec_path = x_csv_files[0].with_name(x_csv_files[0].name.replace("_x_", "_combined_")).with_suffix(".png")

    # Get bins from x csv file header
    with open(x_csv_files[0], "r") as bins_file:
        bins_txt = bins_file.readline().replace(" ", "").replace("#", "")
        bins = np.fromstring(bins_txt, sep=',', dtype=float)

    # Read data
    csv_x_data = np.loadtxt(x_csv_files[0], delimiter=",")
    csv_y_data = np.loadtxt(y_csv_files[0], delimiter=",")
    csv_z_data = np.loadtxt(z_csv_files[0], delimiter=",")

    # Frequencies are the first column of the data
    freqs = csv_x_data[:, 0]

    # Average the components
    Pxx = np.mean([csv_x_data[:, 1:], csv_y_data[:, 1:], csv_z_data[:, 1:]], axis=0)

    # Create separate spectrogram figure
    fig, ax = plt.subplots()
    fig.set_size_inches(7.5, 5)
    title = f"threshold Pxx = {thresh_val}"

    # Plot and save the spectrogram
    spec.plot_spectrogram(fig, ax, bins, freqs, Pxx, ylim, title=title, x_label="Time (s)",
                          color_range=[thresh_val, max_plot])
    fig.savefig(spec_path)

    # Save data as CSV
    path_csv = spec_path.with_suffix(".csv")
    data_csv = np.append(freqs[np.newaxis].T, Pxx, axis=1)
    np.savetxt(path_csv, data_csv, header=bins_txt, delimiter=",")

    logging.info(f"Spectrogram saved at: {spec_path}")
    logging.info(f"Data CSV saved at: {path_csv}")


def main():
    # Load in case-specific parameters
    args = spec.read_command_line_spec()

    # Create logger and set log level
    logging.basicConfig(level=args.log_level, format="%(message)s")

    # Create spectrograms
    create_spectrogram_from_components(args.folder, args.quantity, args.n_samples, args.thresh_val, args.max_plot,
                                       ylim=args.ylim)


if __name__ == '__main__':
    main()
