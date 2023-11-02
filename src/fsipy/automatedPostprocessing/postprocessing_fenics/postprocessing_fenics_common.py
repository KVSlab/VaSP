# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later
"""common functions for postprocessing-fenics scripts"""
import argparse
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--folder", type=Path, help="Path to simulation results")
    parser.add_argument("--stride", type=int, default=1, help="Save frequency of simulation")
    parser.add_argument("-st", "--start_time", type=float, default=None, help="Desired start time for postprocessing")
    parser.add_argument("-et", "--end_time", type=float, default=None, help="Desired end time for postprocessing")
    parser.add_argument("--extract-solid-only", action="store_true", help="Extract solid displacement only")
    parser.add_argument("--log-level", type=int, default=20,
                        help="Specify the log level (default is 20, which is INFO)")

    return parser.parse_args()
