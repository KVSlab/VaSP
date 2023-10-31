from argparse import ArgumentParser
from pathlib import Path

def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--folder', type=Path, help="Path to simulation results")

    parser.add_argument('--stride', type=int, default=1, help="Save frequency of simulation")    

    parser.add_argument('--start_t', type=float, default=0.0, help="Desired start time for postprocessing")

    parser.add_argument('--end_t', type=float, default=0.951, help="Desired end time for postprocessing")

    parser.add_argument("--extract-solid-only", action="store_true", help="Extract solid displacement only")

    return parser.parse_args()

