# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

import re
import json
import logging
from pathlib import Path
from typing import Union, Optional, Dict, Tuple, List

import numpy as np
import h5py


def get_domain_ids(
        mesh_path: Path, fluid_domain_id: int, solid_domain_id: int) -> Tuple[List[int], List[int], List[int]]:
    """
    Given a mesh file, this function returns the IDs of the fluid and solid domains.
    The IDs is a list of integers that correspond to the index of the coordinates (nodes)
    in the mesh file.

    Args:
        mesh_path (Path): Path to the mesh file that contains the fluid and solid domains
        fluid_domain_id (int): ID of the fluid domain
        solid_domain_id (int): ID of the solid domain

    Returns:
        fluid_ids (list): List of IDs of the fluid domain
        solid_ids (list): List of IDs of the solid domain
        all_ids (list): List of IDs of the whole mesh
    """
    assert mesh_path.exists() and mesh_path.is_file(), f"Mesh file {mesh_path} does not exist"
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


def output_file_lists(xdmf_file: Path) -> Tuple[List[str], List[float], List[int]]:
    """
    If the simulation has been restarted, the output is stored in multiple files and may not have even temporal spacing
    This loop determines the file names from the xdmf output file

    Args:
        xdmf_file (Path): Path to xdmf file

    Returns:
        Tuple[List[str], List[float], List[int]]: A tuple containing:
            - List of names of h5 files associated with each timestep
            - List of time values in xdmf file
            - List of indices of each timestep in the corresponding h5 file
    """

    with open(xdmf_file, 'r') as file:
        lines = file.readlines()

    h5file_name_list: List[str] = []
    timevalue_list: List[float] = []
    index_list: List[int] = []
    checkpoint_data: bool = False

    for line in lines:
        if "FiniteElementFunction" in line:
            checkpoint_data = True
            break

    time_pattern: str = '<Time Value="(.+?)"'
    h5_pattern_checkpoint: str = r'"HDF">(.*?):'
    index_pattern_checkpoint: str = r'_([0-9]+)\/vector'
    h5_pattern_no_checkpoint: str = '"HDF">(.+?):/'
    index_pattern_no_checkpoint: str = "VisualisationVector/(.+?)</DataItem"

    # This loop goes through the xdmf output file and gets the time value (timevalue_list), associated
    # with .h5 file (h5file_name_list) and index of each timestep in the corresponding h5 file (index_list)
    for line in lines:
        if '<Time Value' in line:
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            timevalue_list.append(time)

        if checkpoint_data and 'vector' in line:
            h5_str = re.findall(h5_pattern_checkpoint, line)
            h5file_name_list.append(h5_str[0])

            index_str = re.findall(index_pattern_checkpoint, line)
            index = int(index_str[0])
            index_list.append(index)

        elif not checkpoint_data and 'VisualisationVector' in line:
            h5_str = re.findall(h5_pattern_no_checkpoint, line)
            h5file_name_list.append(h5_str[0])

            index_str = re.findall(index_pattern_no_checkpoint, line)
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
