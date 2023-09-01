import argparse


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=str, required=True, help="Path to simulation results")
    parser.add_argument('--mesh-path', type=str, default=None,
                        help="Path to the mesh file (default: <folder_path>/Mesh/mesh.h5)")
    return parser.parse_args()