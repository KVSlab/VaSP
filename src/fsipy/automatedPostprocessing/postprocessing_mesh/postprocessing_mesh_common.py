# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later
"""common functions for postprocessing-mesh scripts"""

import argparse
from pathlib import Path

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    parser.add_argument('--folder', type=Path, required=True, help="Path to simulation results")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file (default: <folder_path>/Mesh/mesh.h5)")
    parser.add_argument('-v', '--view', action='store_true', default=False,
                        help="Determine whether or not to save a pvd file for viewing in paraview")

    return parser.parse_args()
