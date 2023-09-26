#!/usr/bin/env python

# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later

"""
This script is a versatile Python script for parsing and visualizing
simulation log files. It extracts data from log files, offers flexible
plotting options, and helps users gain insights from simulation
results.

Example:

fsipy-log-plotter simulation.log --plot-all --save --output-directory Results --figure-size 12,8
"""

import re
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import matplotlib.pyplot as plt


def parse_log_file(log_file: str) -> Dict[str, Any]:
    """
    Parse a log file and store data in a structured dictionary with numpy arrays.

    Args:
        log_file (str): Path to the log file.

    Returns:
        dict: A dictionary containing the parsed data.
    """
    logging.info(f"--- Parsing data from '{log_file}'")

    # Initialize empty lists and dictionaries to store data
    data: Dict[str, Any] = {
        "time_step": [],
        "time": [],
        "cpu_time": [],
        "ramp_factor": [],
        "pressure": [],
        "newton_iteration": {
            "atol": [],
            "rtol": [],
        },
        "probe_points": {},
        "flow_properties": {
            "flow_rate": [],
            "velocity_mean": [],
            "velocity_min": [],
            "velocity_max": [],
            "cfl_mean": [],
            "cfl_min": [],
            "cfl_max": [],
            "reynolds_mean": [],
            "reynolds_min": [],
            "reynolds_max": [],
        }
    }

    # Define regular expressions for matching specific lines
    time_step_pattern = re.compile(r"Solved for timestep (.*), t = (.*) in (.*) s")
    ramp_factor_pattern = re.compile(r"ramp_factor = (.*) m\^3/s")
    pressure_pattern = re.compile(r"Instantaneous normal stress prescribed at the FSI interface (.*) Pa")
    newton_iteration_pattern = \
        re.compile(r'Newton iteration (.*): r \(atol\) = (.*) \(tol = .*\), r \(rel\) = (.*) \(tol = .*\)')
    probe_point_pattern = re.compile(r"Probe Point (.*): Velocity: \((.*), (.*), (.*)\) \| Pressure: (.*)")
    flow_rate_pattern = re.compile(r"\s*Flow Rate at Inlet: (.*)")
    velocity_pattern = re.compile(r"\s*Velocity \(mean, min, max\): (.*), (.*), (.*)")
    cfl_pattern = re.compile(r"\s*CFL \(mean, min, max\): (.*), (.*), (.*)")
    reynolds_pattern = re.compile(r"\s*Reynolds Numbers \(mean, min, max\): (.*), (.*), (.*)")

    # Open and read the log file line by line
    with open(log_file, 'r') as file:
        for line in file:
            match = time_step_pattern.match(line)
            if match:
                data["time_step"].append(int(match.group(1)))
                data["time"].append(float(match.group(2)))
                data["cpu_time"].append(float(match.group(3)))
                continue

            match = ramp_factor_pattern.match(line)
            if match:
                data["ramp_factor"].append(float(match.group(1)))
                continue

            match = pressure_pattern.match(line)
            if match:
                data["pressure"].append(float(match.group(1)))
                continue

            match = newton_iteration_pattern.match(line)
            if match:
                data["newton_iteration"]["atol"].append(float(match.group(2)))
                data["newton_iteration"]["rtol"].append(float(match.group(3)))
                continue

            match = probe_point_pattern.match(line)
            if match:
                probe_point = int(match.group(1))
                if probe_point not in data["probe_points"]:
                    data["probe_points"][probe_point] = {
                        "magnitude": [],
                        "pressure": []
                    }
                velocity_magnitude = \
                    np.sqrt(float(match.group(2))**2 + float(match.group(3))**2 + float(match.group(4))**2)
                data["probe_points"][probe_point]["magnitude"].append(velocity_magnitude)
                data["probe_points"][probe_point]["pressure"].append(float(match.group(5)))
                continue

            match = flow_rate_pattern.match(line)
            if match:
                data["flow_properties"]["flow_rate"].append(float(match.group(1)))
                continue

            match = velocity_pattern.match(line)
            if match:
                data["flow_properties"]["velocity_mean"].append(float(match.group(1)))
                data["flow_properties"]["velocity_min"].append(float(match.group(2)))
                data["flow_properties"]["velocity_max"].append(float(match.group(3)))
                continue

            match = cfl_pattern.match(line)
            if match:
                data["flow_properties"]["cfl_mean"].append(float(match.group(1)))
                data["flow_properties"]["cfl_min"].append(float(match.group(2)))
                data["flow_properties"]["cfl_max"].append(float(match.group(3)))
                continue

            match = reynolds_pattern.match(line)
            if match:
                data["flow_properties"]["reynolds_mean"].append(float(match.group(1)))
                data["flow_properties"]["reynolds_min"].append(float(match.group(2)))
                data["flow_properties"]["reynolds_max"].append(float(match.group(3)))

    # Convert lists to numpy arrays
    data["time_step"] = np.array(data["time_step"])
    data["time"] = np.array(data["time"])
    data["cpu_time"] = np.array(data["cpu_time"])
    data["ramp_factor"] = np.array(data["ramp_factor"])
    data["pressure"] = np.array(data["pressure"])
    data["newton_iteration"]["atol"] = np.array(data["newton_iteration"]["atol"])
    data["newton_iteration"]["rtol"] = np.array(data["newton_iteration"]["rtol"])

    for probe_point in data["probe_points"]:
        data["probe_points"][probe_point]["magnitude"] = np.array(data["probe_points"][probe_point]["magnitude"])
        data["probe_points"][probe_point]["pressure"] = np.array(data["probe_points"][probe_point]["pressure"])

    data["flow_properties"]["flow_rate"] = np.array(data['flow_properties']['flow_rate'])
    data["flow_properties"]["velocity_mean"] = np.array(data["flow_properties"]["velocity_mean"])
    data["flow_properties"]["velocity_min"] = np.array(data["flow_properties"]["velocity_min"])
    data["flow_properties"]["velocity_max"] = np.array(data["flow_properties"]["velocity_max"])
    data["flow_properties"]["cfl_mean"] = np.array(data["flow_properties"]["cfl_mean"])
    data["flow_properties"]["cfl_min"] = np.array(data["flow_properties"]["cfl_min"])
    data["flow_properties"]["cfl_max"] = np.array(data["flow_properties"]["cfl_max"])
    data["flow_properties"]["reynolds_mean"] = np.array(data["flow_properties"]["reynolds_mean"])
    data["flow_properties"]["reynolds_min"] = np.array(data["flow_properties"]["reynolds_min"])
    data["flow_properties"]["reynolds_max"] = np.array(data["flow_properties"]["reynolds_max"])

    return data


def parse_dictionary_from_log(log_file: str) -> dict:
    """
    Parse a dictionary-like content from a log file and return it as a dictionary.

    Args:
        log_file (str): Path to the log file.

    Returns:
        dict: Parsed dictionary.
    """
    logging.info(f"--- Parsing dictionary from '{log_file}'")

    parsed_dict = {}
    entry_lines = []
    in_entry = False

    with open(log_file, 'r') as file:
        for line in file:
            line = line.strip()

            # Check if the line starts a new dictionary entry
            if line.startswith('{'):
                in_entry = True
                entry_lines = [line]
            elif in_entry:
                entry_lines.append(line)

                # Check if the line ends the current dictionary entry
                if line.endswith('}'):
                    # Combine the lines and modify for JSON compatibility
                    entry_str = '\n'.join(entry_lines)
                    entry_str = (
                        entry_str.replace("'", '"')
                        .replace("None", "null")
                        .replace("True", "true")
                        .replace("False", "false")
                    )

                    # Handle PosixPath objects by converting them to strings
                    entry_str = re.sub(r'"(restart_folder)":\s+PosixPath\("([^"]+)"\)', r'"\1": "\2"', entry_str)

                    try:
                        entry_dict = json.loads(entry_str)
                        if isinstance(entry_dict, dict):
                            parsed_dict.update(entry_dict)
                            break  # Exit the loop since there is only one dictionary
                        else:
                            logging.warning(f"WARNING: Entry is not a valid dictionary: {entry_str}")
                    except json.JSONDecodeError as e:
                        logging.warning(f"WARNING: JSONDecodeError while parsing entry: {e}")

    return parsed_dict


def plot_variable_vs_time(time: np.ndarray, variable: np.ndarray, variable_name: str, save_to_file: bool = False,
                          output_directory: Optional[str] = None, figure_size: Tuple[int, int] = (10, 6),
                          start: Optional[int] = None, end: Optional[int] = None) -> None:
    """
    Plot a variable against time.

    Args:
        time (numpy.ndarray): Array containing time values.
        variable (numpy.ndarray): Array containing the variable values.
        variable_name (str): Name of the variable for labeling the plot.
        save_to_file (bool, optional): Whether to save the figure to a file (default is False).
        output_directory (str, optional): The directory where the figure will be saved when save_to_file is True.
        figure_size (tuple, optional): Figure size in inches (width, height). Default is (10, 6).
        start (int, optional): Index to start plotting data from. Default is None (start from the beginning).
        end (int, optional): Index to end plotting data at. Default is None (end at the last data point).
    """
    logging.info(f"--- Creating plot for {variable_name}")

    plt.figure(figsize=figure_size)
    plt.plot(time[start:end], variable[start:end], label=variable_name, linestyle='-', color='b')
    plt.xlabel("Time [s]")
    plt.ylabel(variable_name)
    plt.title(f"{variable_name} vs. Time")
    plt.grid(True)
    plt.legend()

    if save_to_file:
        save_plot_to_file(variable_name, output_directory)


def plot_variable_comparison(variable: np.ndarray, variable_name: str, time_steps_per_cycle: int,
                             save_to_file: bool = False, output_directory: Optional[str] = None,
                             figure_size: Tuple[int, int] = (10, 6),
                             start_cycle: Optional[int] = 1,
                             end_cycle: Optional[int] = None) -> None:
    """
    Plot comparison of a variable across multiple cycles.

    Args:
        variable (numpy.ndarray): Array containing the variable values.
        variable_name (str): Name of the variable for labeling the plot.
        time_steps_per_cycle (int): The number of time steps in each cycle.
        save_to_file (bool, optional): Whether to save the figure to a file (default is False).
        output_directory (str, optional): The directory where the figure will be saved when save_to_file is True.
        figure_size (tuple, optional): Figure size in inches (width, height). Default is (10, 6).
        start_cycle (int, optional): The cycle to start comparing from. Default is 1 (start at first cycle).
        end_cycle (int, optional): The cycle to end comparing at. Default is None (end at last cycle).
    """
    # Determine the total number of cycles
    num_cycles = round(len(variable) / time_steps_per_cycle)

    # If start_cycle is None, start at first cycle
    first_cycle = 1 if start_cycle is None else int(start_cycle)

    # If end_cycle is not provided, end at last cycle
    last_cycle = num_cycles if end_cycle is None else int(end_cycle)

    logging.info(f"--- Creating plot for {variable_name} - " +
                 f"Comparing from cycle {first_cycle} to cycle {last_cycle}")

    # Split the data into separate cycles
    split_variable_data = np.array_split(variable, num_cycles)

    plt.figure(figsize=figure_size)

    # Plot each cycle
    for cycle in range(first_cycle - 1, last_cycle):
        cycle_variable_data = split_variable_data[cycle]

        plt.plot(cycle_variable_data, label=f"Cycle {cycle + 1}")

    plt.xlabel("[-]")
    plt.ylabel(variable_name)
    plt.title(f"{variable_name} - Comparing from cycle {first_cycle} to cycle {last_cycle}")
    plt.grid(True)
    plt.legend()

    if save_to_file:
        save_plot_to_file(variable_name + "Comparison", output_directory)


def plot_multiple_variables_vs_time(time: np.ndarray, variable_mean: np.ndarray, variable_min: np.ndarray,
                                    variable_max: np.ndarray, variable_name: str, save_to_file: bool = False,
                                    output_directory: Optional[str] = None, figure_size: Tuple[int, int] = (10, 6),
                                    start: Optional[int] = None, end: Optional[int] = None) -> None:
    """
    Plot mean, min, and max variables against time on the same figure.

    Args:
        time (numpy.ndarray): Array containing time values.
        variable_mean (numpy.ndarray): Array containing the mean variable values.
        variable_min (numpy.ndarray): Array containing the min variable values.
        variable_max (numpy.ndarray): Array containing the max variable values.
        variable_name (str): Names of the variables for labeling the plot (mean, min, max).
        save_to_file (bool, optional): Whether to save the figure to a file (default is False).
        output_directory (str, optional): The directory where the figure will be saved when save_to_file is True.
        figure_size (tuple, optional): Figure size in inches (width, height). Default is (10, 6).
        start (int, optional): Index to start plotting data from. Default is None (start from the beginning).
        end (int, optional): Index to end plotting data at. Default is None (end at the last data point).
    """
    logging.info(f"--- Creating plot for {variable_name}")

    plt.figure(figsize=figure_size)
    plt.plot(time[start:end], variable_mean[start:end], label=variable_name + " (mean)", linestyle='-', color='b')
    plt.plot(time[start:end], variable_min[start:end], label=variable_name + " (min)", linestyle='-', color='g')
    plt.plot(time[start:end], variable_max[start:end], label=variable_name + " (max)", linestyle='-', color='r')
    plt.xlabel("Time [s]")
    plt.ylabel(variable_name)
    plt.title(f"{variable_name} vs. Time")
    plt.grid(True)
    plt.legend()

    if save_to_file:
        save_plot_to_file(variable_name, output_directory)


def plot_multiple_variables_comparison(variable_mean: np.ndarray, variable_min: np.ndarray, variable_max: np.ndarray,
                                       variable_name: str, time_steps_per_cycle: int, save_to_file: bool = False,
                                       output_directory: Optional[str] = None, figure_size: Tuple[int, int] = (10, 6),
                                       start_cycle: Optional[int] = 1,
                                       end_cycle: Optional[int] = None) -> None:
    """
    Plot mean, min, and max variables against time on the same figure.

    Args:
        variable_mean (numpy.ndarray): Array containing the mean variable values.
        variable_min (numpy.ndarray): Array containing the min variable values.
        variable_max (numpy.ndarray): Array containing the max variable values.
        variable_name (str): Names of the variables for labeling the plot (mean, min, max).
        time_steps_per_cycle (int): The number of time steps in each cycle.
        save_to_file (bool, optional): Whether to save the figure to a file (default is False).
        output_directory (str, optional): The directory where the figure will be saved when save_to_file is True.
        figure_size (tuple, optional): Figure size in inches (width, height). Default is (10, 6).
        start_cycle (int, optional): The cycle to start comparing from. Default is 1 (start from first cycle).
        end_cycle (int, optional): The cycle to end comparing at. Default is None (end at last cycle).
    """
    # Determine the total number of cycles
    num_cycles = round(len(variable_mean) / time_steps_per_cycle)

    # If start_cycle is None, start at first cycle
    first_cycle = 1 if start_cycle is None else int(start_cycle)

    # If end_cycle is not provided, end at last cycle
    last_cycle = num_cycles if end_cycle is None else int(end_cycle)

    logging.info(f"--- Creating plot for {variable_name} - " +
                 f"Comparing from cycle {first_cycle} to cycle {last_cycle}")

    # Split the data into separate cycles
    split_variable_mean_data = np.array_split(variable_mean, num_cycles)
    split_variable_min_data = np.array_split(variable_min, num_cycles)
    split_variable_max_data = np.array_split(variable_max, num_cycles)

    # Create subplots for mean, min, and max
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=figure_size, sharex=True)

    for cycle in range(first_cycle - 1, last_cycle):
        cycle_variable_mean_data = split_variable_mean_data[cycle]
        cycle_variable_min_data = split_variable_min_data[cycle]
        cycle_variable_max_data = split_variable_max_data[cycle]

        ax1.plot(cycle_variable_mean_data, label=f"Cycle {cycle + 1}")
        ax2.plot(cycle_variable_min_data, label=f"Cycle {cycle + 1}")
        ax3.plot(cycle_variable_max_data, label=f"Cycle {cycle + 1}")

    # Set labels and titles
    ax1.set_ylabel(f"{variable_name}")
    ax2.set_ylabel(f"{variable_name}")
    ax3.set_ylabel(f"{variable_name}")
    ax1.set_title(f"{variable_name} (mean)")
    ax2.set_title(f"{variable_name} (min)")
    ax3.set_title(f"{variable_name} (max)")
    plt.xlabel("[-]")

    # Add legend to each subplot
    ax1.legend(loc="upper right")
    ax2.legend(loc="upper right")
    ax3.legend(loc="upper right")

    # Add common title
    plt.suptitle(f"{variable_name} - Comparing from cycle {first_cycle} to cycle {last_cycle}")

    # Adjust spacing between subplots
    plt.tight_layout()

    if save_to_file:
        save_plot_to_file(variable_name + "Comparison", output_directory)


def plot_newton_iteration(variable: np.ndarray, variable_name: str, save_to_file: bool = False,
                          output_directory: Optional[str] = None, figure_size: Tuple[int, int] = (10, 6), ) -> None:
    """
    Plot Newton iteration (atol or rtol).

    Args:
        variable (numpy.ndarray): Array containing the variable values.
        variable_name (str): Name of the variable for labeling the plot.
        save_to_file (bool, optional): Whether to save the figure to a file (default is False).
        output_directory (str, optional): The directory where the figure will be saved when save_to_file is True.
        figure_size (tuple, optional): Figure size in inches (width, height). Default is (10, 6).
    """
    logging.info(f"--- Creating plot for {variable_name}")

    plt.figure(figsize=figure_size)
    plt.plot(variable, label=variable_name, linestyle='--', marker='o', color='b')
    plt.ylabel(variable_name)
    plt.title(f"{variable_name}")
    plt.grid(True)
    plt.legend()
    plt.gca().set_yscale('log')

    if save_to_file:
        save_plot_to_file(variable_name, output_directory)


def plot_probe_points(time: np.ndarray, probe_points: Dict[int, Dict[str, np.ndarray]],
                      selected_probe_points: Optional[List[int]] = None, save_to_file: bool = False,
                      output_directory: Optional[str] = None, figure_size: Tuple[int, int] = (12, 6),
                      start: Optional[int] = None, end: Optional[int] = None) -> None:
    """
    Plot velocity magnitude and pressure for probe points against time.

    Args:
        time (numpy.ndarray): Time array.
        probe_points (dict): Probe point data containing velocity magnitude and pressure arrays.
        selected_probe_points (list, optional): List of probe points to plot. Plot all probe points if not provided.
        save_to_file (bool, optional): Whether to save the figures to files (default is False).
        output_directory (str, optional): The directory where the figure will be saved when save_to_file is True.
        figure_size (tuple, optional): Figure size in inches (width, height). Default is (12, 8).
        start (int, optional): Index to start plotting data from. Default is None (start from the beginning).
        end (int, optional): Index to end plotting data at. Default is None (end at the last data point).
    """
    logging.info("--- Creating plot for velocity magnitude and pressure for probe points")

    # If selected_probe_points is not provided, plot all probe points
    if selected_probe_points is None:
        selected_probe_points = list(probe_points.keys())

    # Filter probe_points dictionary to select only the specified probe points
    selected_probe_data = \
        {probe_point: data for probe_point, data in probe_points.items() if probe_point in selected_probe_points}

    num_selected_probe_points = len(selected_probe_data)

    # Calculate the number of rows and columns for subplots
    num_rows = int(np.ceil(num_selected_probe_points / 2))
    num_cols = min(2, num_selected_probe_points)

    for probe_point in selected_probe_points:
        if probe_point not in probe_points:
            # Log a warning for probe points not found in the dictionary
            logging.warning(f"WARNING: Probe point {probe_point} not found. Skipping.")

    # Create subplots based on the number of selected probe points
    if num_rows == 1 and num_cols == 1:
        # If only one probe point is selected, create a single figure
        fig = plt.figure(figsize=figure_size)
        axs = [fig.gca()]  # Get the current axis as a list
    else:
        fig, axs = plt.subplots(num_rows, num_cols, figsize=figure_size)

    for i, (probe_point, data) in enumerate(selected_probe_data.items()):
        row = i // 2
        col = i % 2

        ax = axs[row, col] if num_rows > 1 else axs[col]  # type: ignore[call-overload]

        # Extract the data within the specified range (start:end)
        magnitude_data = data["magnitude"][start:end]
        pressure_data = data["pressure"][start:end]

        l1, = ax.plot(time[start:end], magnitude_data, color='b')
        ax.set_xlabel("Time [s]")
        ax.set_ylabel("Velocity [m/s]")
        ax.set_title(f"Probe point {probe_point}")
        ax.grid(True)
        ax.tick_params(axis='y', which='major', labelsize=12, labelcolor='b')

        ax2 = ax.twinx()
        l2, = ax2.plot(time[start:end], pressure_data, color='r')
        ax2.set_ylabel("Pressure [Pa]", color='r')
        ax2.legend([l1, l2], ["Velocity Magnitude", "Pressure"], loc="upper right")
        ax2.tick_params(axis='y', which='major', labelcolor='r')

    # Adjust spacing between subplots
    plt.tight_layout()

    if save_to_file:
        save_plot_to_file("Probe Points", output_directory)


def plot_probe_points_comparison(probe_points: Dict[int, Dict[str, np.ndarray]], time_steps_per_cycle: int,
                                 selected_probe_points: Optional[List[int]] = None, save_to_file: bool = False,
                                 output_directory: Optional[str] = None, figure_size: Tuple[int, int] = (12, 6),
                                 start_cycle: Optional[int] = 0, end_cycle: Optional[int] = None) -> None:
    """
    Plot comparison of velocity magnitude and pressure for probe points across multiple cycles.

    Args:
        probe_points (dict): Probe point data containing velocity magnitude and pressure arrays.
        selected_probe_points (list, optional): List of probe points to plot. Plot all probe points if not provided.
        time_steps_per_cycle (int): The number of time steps in each cycle.
        save_to_file (bool, optional): Whether to save the figures to files (default is False).
        output_directory (str, optional): The directory where the figure will be saved when save_to_file is True.
        figure_size (tuple, optional): Figure size in inches (width, height). Default is (12, 8).
        start_cycle (int, optional): The cycle to start comparing from. Default is 1 (start from  first cycle).
        end_cycle (int, optional): The cycle to end comparing at. Default is None (end at last cycle).
    """
    # If selected_probe_points is not provided, plot all probe points
    if selected_probe_points is None:
        selected_probe_points = list(probe_points.keys())

    # Filter probe_points dictionary to select only the specified probe points
    selected_probe_data = \
        {probe_point: data for probe_point, data in probe_points.items() if probe_point in selected_probe_points}

    for probe_point in selected_probe_points:
        if probe_point not in probe_points:
            # Log a warning for probe points not found in the dictionary
            logging.warning(f"WARNING: Probe point {probe_point} not found. Skipping.")

    # Determine the total number of cycles
    num_cycles = round(len(probe_points[selected_probe_points[0]]["magnitude"]) / time_steps_per_cycle)

    # If start_cycle is None, start at first cycle
    first_cycle = 1 if start_cycle is None else int(start_cycle)

    # If end_cycle is not provided, end at last cycle
    last_cycle = num_cycles if end_cycle is None else int(end_cycle)

    for i, (probe_point, data) in enumerate(selected_probe_data.items()):
        logging.info(f"--- Creating plot for probe point {probe_point} - " +
                     f"Comparing from cycle {first_cycle} to cycle {last_cycle}")

        # Create subplots for magnitude and pressure
        fig, axs = plt.subplots(2, 1, figsize=figure_size)

        ax, ax2 = axs

        # Split the data into separate cycles
        split_magnitude_data = np.array_split(data["magnitude"], num_cycles)
        split_pressure_data = np.array_split(data["pressure"], num_cycles)

        # Plot each cycle
        for cycle in range(first_cycle - 1, last_cycle):
            cycle_magnitude_data = split_magnitude_data[cycle]
            cycle_pressure_data = split_pressure_data[cycle]

            ax.plot(cycle_magnitude_data, label=f"Cycle {cycle + 1}")
            ax.set_xlabel("[-]")
            ax.set_ylabel("Velocity [m/s]")
            ax.set_title("Velocity Magnitude")
            ax.grid(True)
            ax.legend(loc="upper right")

            ax2.plot(cycle_pressure_data, label=f"Cycle {cycle + 1}")
            ax2.set_xlabel("[-]")
            ax2.set_ylabel("Pressure [Pa]")
            ax2.set_title("Pressure")
            ax2.grid(True)
            ax2.legend(loc="upper right")

        # Add common title
        fig.suptitle(f"Probe point {probe_point} - " +
                     f"Comparing from cycle {first_cycle} to cycle {last_cycle}")

        # Adjust spacing between subplots
        plt.tight_layout()

        if save_to_file:
            save_plot_to_file(f"Probe Points Comparison {probe_point}", output_directory)


def save_plot_to_file(variable_name: str, output_directory: Optional[str]) -> None:
    """
    Save a plot to a file.

    Args:
        variable_name (str): Name of the variable for generating the filename.
        output_directory (str, optional): The directory where the figure will be saved.
    """
    if output_directory:
        output_path = Path(output_directory)
        output_path.mkdir(parents=True, exist_ok=True)  # Create the directory if it doesn't exist
    else:
        output_path = Path.cwd()  # Use current working directory as the default

    filename = output_path / f"{variable_name.lower().replace(' ', '_')}.png"
    plt.savefig(filename)
    logging.info(f"    Plot saved to {filename}")


def compute_average_over_cycles(data: np.ndarray, time_steps_per_cycle: int) -> np.ndarray:
    """
    Compute the average over cycles in a numpy array.

    Args:
        data (np.ndarray): The input data array.
        time_steps_per_cycle (int): The number of time steps in each cycle.

    Returns:
        np.ndarray: A numpy array containing the average data over cycles.
    """
    # Determine the total number of cycles
    total_cycles = len(data) // time_steps_per_cycle

    # Reshape the data into a 3D array
    data_3d = data[:total_cycles * time_steps_per_cycle].reshape(total_cycles, time_steps_per_cycle, -1)

    # Compute the average over cycles
    average_over_cycles = np.mean(data_3d, axis=0)

    # Remove singleton dimension if present
    average_over_cycles = np.squeeze(average_over_cycles)

    return average_over_cycles


def parse_command_line_args() -> argparse.Namespace:
    """
    Parse command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("log_file", type=str, help="Path to the log file")
    parser.add_argument("--plot-all", action="store_true",
                        help="Plot all data in separate figures (default when no specific --plot options are provided)")
    parser.add_argument("--plot-cpu-time", action="store_true", help="Plot CPU time")
    parser.add_argument("--plot-ramp-factor", action="store_true", help="Plot ramp factor")
    parser.add_argument("--plot-pressure", action="store_true", help="Plot pressure")
    parser.add_argument("--plot-newton-iteration-atol", action="store_true", help="Plot Newton iteration (atol)")
    parser.add_argument("--plot-newton-iteration-rtol", action="store_true", help="Plot Newton iteration (rtol)")
    parser.add_argument("--plot-probe-points", action="store_true", help="Plot probe points")
    parser.add_argument("--plot-flow-rate", action="store_true", help="Plot flow rate")
    parser.add_argument("--plot-velocity", action="store_true", help="Plot velocity (mean, min and max)")
    parser.add_argument("--plot-cfl", action="store_true", help="Plot CFL numbers (mean, min and max)")
    parser.add_argument("--plot-reynolds", action="store_true", help="Plot Reynolds numbers (mean, min and max)")
    parser.add_argument("--probe-points", type=int, nargs="+", help="List of probe points to plot")
    parser.add_argument("--start-cycle", type=int, default=1,
                        help="Cycle to start plotting data from (default: 1, start at first cycle)")
    parser.add_argument("--end-cycle", type=int, default=None,
                        help="Cycle to end plotting data at (default: end at last cycle)")
    parser.add_argument("--compute-average", action="store_true",
                        help="Compute average over cycles. This option has no effect for " +
                             "--plot-newton-iteration-atol and --plot-newton-iteration-rtol.")
    parser.add_argument("--compare-cycles", action="store_true",
                        help="Compare data across multiple cycles. This option has no effect for " +
                             "--plot-newton-iteration-atol and --plot-newton-iteration-rtol.")
    parser.add_argument("--figure-size", type=lambda x: tuple(map(int, x.split(','))), default=(10, 6),
                        help="Figure size in inches (width, height), e.g., '12,8' (default: 10,6)")
    parser.add_argument("--save", action="store_true", help="Save the figures to files")
    parser.add_argument("--output-directory", type=str, default="Images",
                        help="Directory where plot images will be saved (default: 'Images')")
    parser.add_argument("--log-level", type=int, default=20,
                        help="Specify the log level (default is 20, which is INFO)")

    return parser.parse_args()


def main() -> None:
    args = parse_command_line_args()

    # Enable --plot-all by default if no specific --plot options are provided
    plot_types = ['cpu_time', 'ramp_factor', 'pressure', 'newton_iteration_atol', 'newton_iteration_rtol',
                  'probe_points', 'flow_rate', 'velocity', 'cfl', 'reynolds']
    args.plot_all = args.plot_all or all(not getattr(args, f'plot_{plot_type}') for plot_type in plot_types)

    # Create logger and set log level
    logging.basicConfig(level=args.log_level, format="%(message)s")

    # Log a warning if --compute-average and --compare-cycles are used together
    if args.compute_average and args.compare_cycles:
        logging.warning("WARNING: Select either --compute-average or --compare-cycles, not both.")
        args.compare_cycles = False

    # Parse log data
    parsed_dict = parse_dictionary_from_log(args.log_file)
    parsed_data = parse_log_file(args.log_file)

    # Extract end time, cycle length and time step size from the parsed dictionary
    end_time = parsed_dict.get("T", 0.951)
    cycle_length = parsed_dict.get("T_cycle", 0.951)
    dt = parsed_dict.get("dt", 0.001)

    # Calculate the number of cycles and the number of time steps per cycle
    num_cycles = int(end_time / cycle_length)
    time_steps_per_cycle = round(cycle_length / dt) + 1

    # Determine start and end range for data
    start_cycle = args.start_cycle
    end_cycle = max(args.end_cycle if args.end_cycle else num_cycles, 1)
    start = (start_cycle - 1) * time_steps_per_cycle
    end = min(end_cycle * time_steps_per_cycle, len(parsed_data.get("time", [])))

    # Extract variables from the parsed data
    time = parsed_data.get("time", [])
    cpu_time = parsed_data.get("cpu_time", [])
    ramp_factor = parsed_data.get("ramp_factor", [])
    pressure = parsed_data.get("pressure", [])
    newton_iteration_atol = parsed_data.get("newton_iteration", {}).get("atol", [])
    newton_iteration_rtol = parsed_data.get("newton_iteration", {}).get("rtol", [])
    probe_points = parsed_data.get("probe_points", {})
    flow_rate = parsed_data.get("flow_properties", {}).get("flow_rate", [])
    velocity_mean = parsed_data.get("flow_properties", {}).get("velocity_mean", [])
    velocity_min = parsed_data.get("flow_properties", {}).get("velocity_min", [])
    velocity_max = parsed_data.get("flow_properties", {}).get("velocity_max", [])
    cfl_mean = parsed_data.get("flow_properties", {}).get("cfl_mean", [])
    cfl_min = parsed_data.get("flow_properties", {}).get("cfl_min", [])
    cfl_max = parsed_data.get("flow_properties", {}).get("cfl_max", [])
    reynolds_mean = parsed_data.get("flow_properties", {}).get("reynolds_mean", [])
    reynolds_min = parsed_data.get("flow_properties", {}).get("reynolds_min", [])
    reynolds_max = parsed_data.get("flow_properties", {}).get("reynolds_max", [])

    # Compute average over cycles for all data (except Newton iteration) if enabled
    if args.compute_average:
        logging.info(f"--- Computing average over cycles (cycle {start_cycle}-{end_cycle})")
        cpu_time = compute_average_over_cycles(cpu_time[start:end], time_steps_per_cycle)
        ramp_factor = compute_average_over_cycles(ramp_factor[start:end], time_steps_per_cycle)
        pressure = compute_average_over_cycles(pressure[start:end], time_steps_per_cycle)
        flow_rate = compute_average_over_cycles(flow_rate[start:end], time_steps_per_cycle)
        velocity_mean = compute_average_over_cycles(velocity_mean[start:end], time_steps_per_cycle)
        velocity_min = compute_average_over_cycles(velocity_min[start:end], time_steps_per_cycle)
        velocity_max = compute_average_over_cycles(velocity_max[start:end], time_steps_per_cycle)
        cfl_mean = compute_average_over_cycles(cfl_mean[start:end], time_steps_per_cycle)
        cfl_min = compute_average_over_cycles(cfl_min[start:end], time_steps_per_cycle)
        cfl_max = compute_average_over_cycles(cfl_max[start:end], time_steps_per_cycle)
        reynolds_mean = compute_average_over_cycles(reynolds_mean[start:end], time_steps_per_cycle)
        reynolds_min = compute_average_over_cycles(reynolds_min[start:end], time_steps_per_cycle)
        reynolds_max = compute_average_over_cycles(reynolds_max[start:end], time_steps_per_cycle)

        for probe_point, probe_data in probe_points.items():
            probe_points[probe_point]["magnitude"] = \
                compute_average_over_cycles(probe_data["magnitude"][start:end], time_steps_per_cycle)
            probe_points[probe_point]["pressure"] = \
                compute_average_over_cycles(probe_data["pressure"][start:end], time_steps_per_cycle)

        time = time[start:start + len(cpu_time)]
        start = 0
        end = len(cpu_time)

    def check_and_warn_empty(variable_name, variable_data, condition):
        """Check if a variable's data is empty, print a warning, and set the condition to False if it's empty."""
        if len(variable_data) == 0:
            logging.warning(f"WARNING: No information about '{variable_name}' found in the log file. Skipping.")
            condition = False  # Set the condition to False if the array is empty
        return condition

    if check_and_warn_empty("CPU Time", cpu_time, args.plot_all or args.plot_cpu_time):
        if args.compare_cycles:
            # Call the plot function to plot CPU time comparison across multiple cycles
            plot_variable_comparison(cpu_time, "CPU Time", time_steps_per_cycle, save_to_file=args.save,
                                     output_directory=args.output_directory, figure_size=args.figure_size,
                                     start_cycle=start_cycle, end_cycle=end_cycle)
        else:
            # Call the plot function to plot CPU time vs. time
            plot_variable_vs_time(time, cpu_time, "CPU Time", save_to_file=args.save,
                                  output_directory=args.output_directory, figure_size=args.figure_size,
                                  start=start, end=end)

    if check_and_warn_empty("Ramp Factor", ramp_factor, args.plot_all or args.plot_ramp_factor):
        if args.compare_cycles:
            # Call the plot function to plot ramp factor comparison across multiple cycles
            plot_variable_comparison(ramp_factor, "Ramp Factor", time_steps_per_cycle, save_to_file=args.save,
                                     output_directory=args.output_directory, figure_size=args.figure_size,
                                     start_cycle=start_cycle, end_cycle=end_cycle)
        else:
            # Call the plot function to plot ramp factor vs. time
            plot_variable_vs_time(time, ramp_factor, "Ramp Factor", save_to_file=args.save,
                                  output_directory=args.output_directory, figure_size=args.figure_size,
                                  start=start, end=end)

    if check_and_warn_empty("Pressure", pressure, args.plot_all or args.plot_pressure):
        if args.compare_cycles:
            # Call the plot function to plot pressure comparison across multiple cycles
            plot_variable_comparison(pressure, "Pressure", time_steps_per_cycle, save_to_file=args.save,
                                     output_directory=args.output_directory, figure_size=args.figure_size,
                                     start_cycle=start_cycle, end_cycle=end_cycle)
        else:
            # Call the plot function to plot pressure vs. time
            plot_variable_vs_time(time, pressure, "Pressure", save_to_file=args.save,
                                  output_directory=args.output_directory, figure_size=args.figure_size,
                                  start=start, end=end)

    if check_and_warn_empty("Newton iteration (atol)", newton_iteration_atol,
                            args.plot_all or args.plot_newton_iteration_atol):
        # Call the plot function to plot Newton iteration (atol)
        plot_newton_iteration(newton_iteration_atol, "Newton iteration (atol)", save_to_file=args.save,
                              output_directory=args.output_directory, figure_size=args.figure_size)

    if check_and_warn_empty("Newton iteration (atol)", newton_iteration_rtol,
                            args.plot_all or args.plot_newton_iteration_rtol):
        # Call the plot function to plot Newton iteration (rtol)
        plot_newton_iteration(newton_iteration_rtol, "Newton iteration (rtol)", save_to_file=args.save,
                              output_directory=args.output_directory, figure_size=args.figure_size)

    if check_and_warn_empty("Flow Rate", flow_rate, args.plot_all or args.plot_flow_rate):
        if args.compare_cycles:
            # Call the plot function to plot flow rate comparison across multiple cycles
            plot_variable_comparison(flow_rate, "Flow Rate", time_steps_per_cycle, save_to_file=args.save,
                                     output_directory=args.output_directory, figure_size=args.figure_size,
                                     start_cycle=start_cycle, end_cycle=end_cycle)
        else:
            # Call the plot function to plot flow rate vs. time
            plot_variable_vs_time(time, flow_rate, "Flow Rate", save_to_file=args.save,
                                  output_directory=args.output_directory, figure_size=args.figure_size,
                                  start=start, end=end)

    if check_and_warn_empty("Velocity", velocity_mean, args.plot_all or args.plot_velocity):
        if args.compare_cycles:
            # Call the plot function to plot velocity comparison across multiple cycles
            plot_multiple_variables_comparison(velocity_mean, velocity_min, velocity_max, "Velocity",
                                               time_steps_per_cycle, save_to_file=args.save,
                                               output_directory=args.output_directory, figure_size=args.figure_size,
                                               start_cycle=start_cycle, end_cycle=end_cycle)
        else:
            # Call the plot function to plot velocity vs. time
            plot_multiple_variables_vs_time(time, velocity_mean, velocity_min, velocity_max, "Velocity",
                                            save_to_file=args.save, output_directory=args.output_directory,
                                            figure_size=args.figure_size, start=start, end=end)

    if check_and_warn_empty("CFL", cfl_mean, args.plot_all or args.plot_cfl):
        if args.compare_cycles:
            # Call the plot function to plot velocity comparison across multiple cycles
            plot_multiple_variables_comparison(cfl_mean, cfl_min, cfl_max, "CFL",
                                               time_steps_per_cycle, save_to_file=args.save,
                                               output_directory=args.output_directory, figure_size=args.figure_size,
                                               start_cycle=start_cycle, end_cycle=end_cycle)
        else:
            # Call the plot function to plot CFL vs. time
            plot_multiple_variables_vs_time(time, cfl_mean, cfl_min, cfl_max, "CFL", save_to_file=args.save,
                                            output_directory=args.output_directory, figure_size=args.figure_size,
                                            start=start, end=end)

    if check_and_warn_empty("Reynolds Numbers", reynolds_mean, args.plot_all or args.plot_reynolds):
        if args.compare_cycles:
            # Call the plot function to plot velocity comparison across multiple cycles
            plot_multiple_variables_comparison(reynolds_mean, reynolds_min, reynolds_max, "Reynolds Numbers",
                                               time_steps_per_cycle, save_to_file=args.save,
                                               output_directory=args.output_directory, figure_size=args.figure_size,
                                               start_cycle=start_cycle, end_cycle=end_cycle)
        else:
            # Call the plot function to plot Reynolds numbers vs. time
            plot_multiple_variables_vs_time(time, reynolds_mean, reynolds_min, reynolds_max, "Reynolds Numbers",
                                            save_to_file=args.save, output_directory=args.output_directory,
                                            figure_size=args.figure_size, start=start, end=end)

    if check_and_warn_empty("Probe Points", probe_points, args.plot_all or args.plot_probe_points):
        if args.compare_cycles:
            # Call the plot function to plot probe points comparison across multiple cycles
            plot_probe_points_comparison(probe_points, time_steps_per_cycle, selected_probe_points=args.probe_points,
                                         save_to_file=args.save, figure_size=args.figure_size,
                                         output_directory=args.output_directory, start_cycle=start_cycle,
                                         end_cycle=end_cycle)
        else:
            # Call the plot function to plot probe points
            plot_probe_points(time, probe_points, selected_probe_points=args.probe_points, save_to_file=args.save,
                              figure_size=args.figure_size, output_directory=args.output_directory, start=start,
                              end=end)

    if not args.save:
        logging.info("--- Showing plot(s)")
        plt.show()


if __name__ == "__main__":
    main()
