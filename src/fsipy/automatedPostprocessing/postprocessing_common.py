# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

import re
import json
import logging
from pathlib import Path
from typing import Union, Optional, Dict

import numpy as np
import h5py


def get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id):
    """
    Given a mesh file, this function returns the IDs of the fluid and solid domains

    Args:
        mesh_path (Path): Path to the mesh file that contains the fluid and solid domains
        fluid_domain_id (int): ID of the fluid domain
        solid_domain_id (int): ID of the solid domain

    Returns:
        fluid_ids (list): List of IDs of the fluid domain
        solid_ids (list): List of IDs of the solid domain
        all_ids (list): List of IDs of the whole mesh
    """
    with h5py.File(mesh_path) as vector_data:
        domains = vector_data['domains/values'][:]
        topology = vector_data['domains/topology'][:, :]

        if isinstance(fluid_domain_id, list):
            id_fluid = np.where((domains == fluid_domain_id[0]) | (domains == fluid_domain_id[1]))
        else:
            id_fluid = np.where(domains == fluid_domain_id)

        if isinstance(solid_domain_id, list):
            id_solid = np.where((domains == solid_domain_id[0]) | (domains == solid_domain_id[1]))
        else:
            id_solid = np.where(domains == solid_domain_id)

        wall_topology = topology[id_solid, :]
        fluid_topology = topology[id_fluid, :]

        # Get topology of fluid, solid and whole mesh
        solid_ids = np.unique(wall_topology)
        fluid_ids = np.unique(fluid_topology)
        all_ids = np.unique(topology)

    return fluid_ids, solid_ids, all_ids


def output_file_lists(xdmf_file):
    """
    If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
    This loop determines the file names from the xdmf output file

    Args:
        xdmf_file (Path): Path to xdmf file

    Returns:
        h5file_name_list (list): List of names of h5 files associated with each timestep
        timevalue_list (list): List of time values in xdmf file
        index_list (list): List of indices of each timestp in the corresponding h5 file
    """

    file1 = open(xdmf_file, 'r')
    Lines = file1.readlines()
    h5file_name_list = []
    timevalue_list = []
    index_list = []

    # This loop goes through the xdmf output file and gets the time value (timevalue_list), associated
    # with .h5 file (h5file_name_list) and index of each timestep in the corresponding h5 file (index_list)
    for line in Lines:
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            timevalue_list.append(time)

        elif 'VisualisationVector' in line:
            h5_pattern = '"HDF">(.+?):/'
            h5_str = re.findall(h5_pattern, line)
            h5file_name_list.append(h5_str[0])

            index_pattern = "VisualisationVector/(.+?)</DataItem>"
            index_str = re.findall(index_pattern, line)
            index = int(index_str[0])
            index_list.append(index)

    return h5file_name_list, timevalue_list, index_list


def read_parameters_from_file(folder: Union[str, Path]) -> Optional[Dict]:
    """
    Reads parameters from a JSON file located in the specified folder.

    Args:
        folder (str): The folder containing simulation results

    Returns:
        dict or None: The loaded parameters as a Python dictionary, or None if an error occurs.
    """
    file_path = Path(folder) / "Checkpoint" / "default_variables.json"

    try:
        with open(file_path, 'r') as json_file:
            parameters = json.load(json_file)
            return parameters
    except FileNotFoundError:
        logging.error(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        logging.error(f"Error parsing JSON file: {e}")
        return None
