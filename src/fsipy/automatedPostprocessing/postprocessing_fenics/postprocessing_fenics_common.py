import argparse
from pathlib import Path

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser()

    parser.add_argument('--folder', type=Path, help="Path to simulation results")

    parser.add_argument('--stride', type=int, default=1, help="Save frequency of simulation")    

    parser.add_argument('--start_t', type=float, default=0.0, help="Desired start time for postprocessing")

    parser.add_argument('--end_t', type=float, default=0.951, help="Desired end time for postprocessing")

    parser.add_argument("--extract-solid-only", action="store_true", help="Extract solid displacement only")

    return parser.parse_args()

