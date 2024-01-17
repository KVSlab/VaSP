import argparse
import h5py
from pathlib import Path
import numpy as np

from fsipy.automatedPostprocessing.postprocessing_common import get_domain_ids


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file")
    parser.add_argument("--fsi-region",
                        type=float,
                        nargs="+",
                        default=None,
                        help="list of points defining the FSI region. x_min, x_max, y_min, y_max, z_min, z_max")
    return parser.parse_args()


def generate_solid_probe(mesh_path: Path, fsi_region: list) -> None:

    fluid_domain_id = 1
    solid_domain_id = 2

    with h5py.File(mesh_path, "r") as mesh:
            coords = mesh['mesh/coordinates'][:, :]
    fluid_ids, solid_ids, all_ids = get_domain_ids(mesh_path, fluid_domain_id, solid_domain_id)
    x_min, x_max, y_min, y_max, z_min, z_max = fsi_region
    points_in_box = np.where((coords[:, 0] > x_min) & (coords[:, 0] < x_max) &
                              (coords[:, 1] > y_min) & (coords[:, 1] < y_max) &
                              (coords[:, 2] > z_min) & (coords[:, 2] < z_max))[0]
    solid_probe_ids = np.intersect1d(points_in_box, solid_ids)
    # pick 20 points from the solid probe
    solid_probe_ids = np.random.choice(solid_probe_ids, 20, replace=False)

    # get the coordinates of the solid probe
    solid_probe_coords = coords[solid_probe_ids, :]

    # save as csv file with x, y, z coordinates
    output_path = mesh_path.parent / "solid_probe.csv"
    np.savetxt(output_path, solid_probe_coords, delimiter=",")
    print("Solid probe coordinates saved in solid_probe.csv")

    return None

def main():
    args = parse_arguments()
    generate_solid_probe(args.mesh_path, args.fsi_region)

if __name__ == "__main__":
    main()
    