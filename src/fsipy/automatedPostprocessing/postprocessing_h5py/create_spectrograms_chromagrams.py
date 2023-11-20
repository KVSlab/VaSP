# Copyright (c) 2023 David Bruneau
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script creates spectrograms, power spectral density and chromagrams from formatted matrices (.npz files)"
"""

from pathlib import Path
from typing import Union, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.io import wavfile

from fsipy.automatedPostprocessing.postprocessing_h5py import spectrograms as spec


def create_spectrogram_composite(case_name: str, dvp: str, df: pd.DataFrame, start_t: float, end_t: float,
                                 num_windows_per_sec: float, overlap_frac: float, window: str, lowcut: float,
                                 thresh_val: float, max_plot: float, image_folder: Union[str, Path],
                                 flow_rate_file: Optional[Union[str, Path]] = None,
                                 amplitude_file: Optional[Union[str, Path]] = None, power_scaled: bool = False,
                                 ylim: Optional[float] = None) -> None:
    """
    Create a composite spectrogram figure.

    Args:
        case_name (str): Path to simulation results.
        dvp (str): DVP identifier.
        df (pd.DataFrame): Input dataframe.
        start_t (float): Start time for analysis.
        end_t (float): End time for analysis.
        num_windows_per_sec (float): Number of windows per second.
        overlap_frac (float): Overlap fraction for windows.
        window (str): Type of window for spectrogram.
        lowcut (float): Lowcut frequency for high-pass filter.
        thresh_val (float): Threshold value.
        max_plot (float): Maximum plot value.
        image_folder (Union[str, Path]): Folder to save the images.
        flow_rate_file (Union[str, Path], optional): File containing flow rate data.
        amplitude_file (Union[str, Path], optional): File containing amplitude data.
        power_scaled (bool, optional): Whether to scale the power.
        ylim (float, optional): Y-axis limit for the plot.
    """

    # Calculate number of windows (you can adjust this equation to fit your temporal/frequency resolution needs)
    num_windows = np.round(num_windows_per_sec * (end_t - start_t)) + 3

    # Get sampling constants
    T, _, fs = spec.get_sampling_constants(df, start_t, end_t)

    # High-pass filter dataframe for spectrogram
    df_filtered = spec.filter_time_data(df, fs, lowcut=lowcut, highcut=15000.0, order=6, btype='highpass')

    # PSD calculation
    Pxx_array, freq_array = spec.get_psd(df_filtered, fs)

    # Plot PSD
    plt.plot(freq_array, Pxx_array)
    plt.xlabel('Freq. (Hz)')
    plt.ylabel('input units^2/Hz')

    # Set ylim if provided
    if ylim is not None:
        plt.ylim([0, ylim])

    psd_filename = f"{dvp}_psd_{case_name}"
    path_to_psd_figure = Path(image_folder) / (psd_filename + '.png')

    # Save the figure
    plt.savefig(path_to_psd_figure)

    # Create composite figure
    if amplitude_file and flow_rate_file:
        fig1, (ax1, ax2, ax3, ax4, ax5) = plt.subplots(5, sharex=True, gridspec_kw={'height_ratios': [1, 3, 1, 1, 1]})
    elif flow_rate_file:
        fig1, (ax1, ax2, ax3, ax4) = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [1, 3, 1, 1]})
    elif amplitude_file:
        fig1, (ax2, ax3, ax4, ax5) = plt.subplots(4, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1, 1]})
    else:
        fig1, (ax2, ax3, ax4) = plt.subplots(3, sharex=True, gridspec_kw={'height_ratios': [3, 1, 1]})

    # Spectrogram--------------------------------------------------------------
    fig1.set_size_inches(7.5, 9)

    # Specs with Reyynolds number
    bins, freqs, Pxx, max_val, min_val, lower_thresh = \
        spec.compute_average_spectrogram(df_filtered, fs, num_windows, overlap_frac, window, start_t, end_t, thresh_val,
                                         scaling="spectrum", filter_data=False, thresh_method="old")
    bins = bins + start_t  # Need to shift bins so that spectrogram timing is correct
    spec.plot_spectrogram(fig1, ax2, bins, freqs, Pxx, ylim, color_range=[thresh_val, max_plot])

    # Chromagram ------------------------------------------------------------
    n_fft = spec.shift_bit_length(int(df.shape[1] / num_windows)) * 2
    n_chroma = 24
    # Recalculate spectrogram without filtering the data
    bins_raw, freqs_raw, Pxx_raw, max_val_raw, min_val_raw, lower_thresh_raw = \
        spec.compute_average_spectrogram(df, fs, num_windows, overlap_frac, window, start_t, end_t, thresh_val,
                                         scaling="spectrum", filter_data=False, thresh_method="old")
    bins_raw = bins_raw + start_t  # Need to shift bins so that spectrogram timing is correct
    # Reverse the log of the data
    Pxx_raw = np.exp(Pxx_raw)

    # Calculate chromagram
    # Normalize so that all chroma in column sum to 1 (other option is "max", which sets the max value
    # in each column to 1)
    norm = "sum"
    chroma = spec.chromagram_from_spectrogram(Pxx_raw, fs, n_fft, n_chroma=n_chroma, norm=norm)
    if power_scaled:
        chroma_power = chroma * (Pxx.max(axis=0) - thresh_val)
        spec.plot_chromagram(fig1, ax3, bins_raw, chroma_power)
    else:
        spec.plot_chromagram(fig1, ax3, bins_raw, chroma)

    # Hack to make all the x axes of the subplots align
    divider2 = make_axes_locatable(ax4)
    cax2 = divider2.append_axes("right", size="5%", pad=0.9)
    cax2.remove()

    # Calculate SBI
    chroma_entropy = spec.calc_chroma_entropy(chroma, n_chroma)
    # Plot SBI
    if power_scaled:
        chroma_entropy_power = chroma_entropy * (Pxx.max(axis=0) - thresh_val)
        ax4.plot(bins, chroma_entropy_power)
    else:
        ax4.plot(bins, chroma_entropy)
    ax4.set_ylabel('SBI')

    # Plot Flow Rate or inlet velocity from input file
    if flow_rate_file:
        flow_rates = np.loadtxt(flow_rate_file)
        flow_rates = flow_rates[np.where((flow_rates[:, 0] > start_t) & (flow_rates[:, 0] < end_t))]
        ax1.plot(flow_rates[:, 0], flow_rates[:, 1])
        ax1.set_ylabel('Flow Rate (normalized)')
        # Hack to make all the x axes of the subplots align
        divider = make_axes_locatable(ax1)
        cax = divider.append_axes("right", size="5%", pad=0.9)
        cax.remove()

    if amplitude_file:
        # Hack to make all the x axes of the subplots align
        divider = make_axes_locatable(ax5)
        cax = divider.append_axes("right", size="5%", pad=0.9)
        cax.remove()
        output_amplitudes = np.genfromtxt(amplitude_file, delimiter=',')
        output_amplitudes = output_amplitudes[output_amplitudes[:, 0] >= start_t]
        output_amplitudes = output_amplitudes[output_amplitudes[:, 0] <= end_t]
        ax5.plot(output_amplitudes[:, 0], output_amplitudes[:, 10], label=case_name)
        ax5.set_ylabel("Amplitude")
        ax5.set_xlabel('Time (s)')
    else:
        ax4.set_xlabel('Time (s)')

    # Name composite figure and save
    composite_figure_name = f"{dvp}_{case_name}_{num_windows}_windows_thresh{thresh_val}_composite_figure"
    if power_scaled:
        composite_figure_name += "_power_scaled"

    composite_figure_path = Path(image_folder) / (composite_figure_name + '.png')
    fig1.savefig(composite_figure_path)

    # create separate spectrogram figure
    fig2, ax2_1 = plt.subplots()
    fig2.set_size_inches(7.5, 5)
    title = f"Pxx max = {max_val:.2e}, Pxx min = {min_val:.2e}, threshold Pxx = {lower_thresh}"
    spec.plot_spectrogram(fig2, ax2_1, bins, freqs, Pxx, ylim, title=title,
                          x_label="Time (s)", color_range=[thresh_val, max_plot])

    # Save data to files (spectrogram)
    spec_filename = f"{dvp}_{case_name}_{num_windows}_windows_thresh{thresh_val}_spectrogram"
    path_to_spec = Path(image_folder) / (spec_filename + '.png')
    fig2.savefig(path_to_spec)

    output_csv_path = path_to_spec.with_suffix('.csv')
    data_csv = np.append(freqs[np.newaxis].T, Pxx, axis=1)
    bins_txt = np.array2string(bins, max_line_width=10000, precision=2, separator=',').replace("[", "").replace("]", "")
    np.savetxt(output_csv_path, data_csv, header=bins_txt, delimiter=",")

    # Save data to files (chromagram)
    chroma_filename = f"{dvp}_{case_name}_{num_windows}_windows_chromagram"
    path_to_chroma = Path(image_folder) / (chroma_filename + '.png')
    chroma_y = np.linspace(0, 1, chroma.shape[0])

    output_csv_path = path_to_chroma.with_suffix('.csv')
    data_csv = np.append(chroma_y[np.newaxis].T, chroma, axis=1)
    bins_txt = \
        np.array2string(bins_raw, max_line_width=10000, precision=2, separator=',').replace("[", "").replace("]", "")
    np.savetxt(output_csv_path, data_csv, header=bins_txt, delimiter=",")

    # Save data to files (SBI)
    sbi_filename = f"{dvp}_{case_name}_{num_windows}_windows_SBI"
    path_to_sbi = Path(image_folder) / (sbi_filename + '.png')

    csv_output_path = path_to_sbi.with_suffix('.csv')
    data_csv = np.array([bins, chroma_entropy]).T
    np.savetxt(csv_output_path, data_csv, header="t (s), SBI", delimiter=",")


def sonify_point(case_name: str, dvp: str, df, start_t: float, end_t: float, overlap_frac: float, lowcut: float,
                 image_folder: str) -> None:
    """
    Sonify a point in the dataframe and save the resulting audio as a WAV file.

    Args:
        case_name (str): Name of the case.
        dvp (str): Type of data to be sonified.
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

    sound_filename = f"{dvp}_sound_{y2.name}_{case_name}.wav"
    path_to_sound = Path(image_folder) / sound_filename

    wavfile.write(path_to_sound, int(fs), y2)


def main():
    # Load in case-specific parameters
    args = spec.read_command_line_spec()

    # Create or read in spectrogram dataframe
    dvp, df, case_name, image_folder, visualization_hi_pass_folder = \
        spec.read_spectrogram_data(args.folder, args.mesh_path, args.save_deg, args.stride, args.start_time,
                                   args.end_time, args.n_samples, args.ylim, args.sampling_region,
                                   args.fluid_sampling_domain_id, args.solid_sampling_domain_id, args.r_sphere,
                                   args.x_sphere, args.y_sphere, args.z_sphere, args.dvp, args.interface_only,
                                   args.component, args.point_id, sampling_method=args.sampling_method)

    # Should these files be used?
    # amplitude_file = Path(visualization_hi_pass_folder) / args.amplitude_file_name
    # flow_rate_file = Path(args.folder) / args.flow_rate_file_name

    # Create spectrograms
    create_spectrogram_composite(case_name, dvp, df, args.start_time, args.end_time, args.num_windows_per_sec,
                                 args.overlap_frac, args.window, args.lowcut, args.thresh_val, args.max_plot,
                                 image_folder, flow_rate_file=None, amplitude_file=None, power_scaled=False)

    if args.sampling_method == "SinglePoint":
        sonify_point(case_name, dvp, df, args.start_time, args.end_time, args.overlap_frac, args.lowcut,
                     image_folder)


if __name__ == '__main__':
    main()
