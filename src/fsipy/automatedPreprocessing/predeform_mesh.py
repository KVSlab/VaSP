# File under GNU GPL (v3) licence, see LICENSE file for details.
# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.

"""
This script is used to predeform the mesh for an FSI simulation. It assumes
that the simulation has already been executed, and the displacement information
is available in the 'displacement.h5' file. By applying the reverse of the
displacement to the original mesh, this script generates a predeformed mesh for
subsequent simulation steps.
"""

import argparse
import h5py
from pathlib import Path


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=str, required=True, help="Path to simulation results")
    parser.add_argument('--mesh-path', type=str, default=None,
                        help="Path to the mesh file (default: <folder_path>/Checkpoint/mesh.h5)")
    parser.add_argument('--scale-factor', type=float, default=-1,
                        help="Scale factor for mesh deformation (default: -1)")
    return parser.parse_args()


def predeform_mesh(folder_path: Path, mesh_path: Path, scale_factor: float) -> None:
    """
    Predeform the mesh for FSI simulation.

    Args:
        folder_path (Path): Path to the simulation results folder.
        mesh_path (Path): Path to the mesh file.
        scale_factor (float): Scale factor for mesh deformation.

    Returns:
        None
    """
    print("Predeforming mesh...")

    # Path to the displacement file
    disp_path = folder_path / "Visualization" / "displacement.h5"
    predeformed_mesh_path = mesh_path.with_name(mesh_path.stem + "_predeformed.h5")

    # Make a copy of the original mesh
    predeformed_mesh_path.write_bytes(mesh_path.read_bytes())

    # Read the displacement file and get the displacement from the last time step
    with h5py.File(disp_path, "r") as vectorData:
        number_of_datasets = len(vectorData["VisualisationVector"].keys())
        disp_array = vectorData[f"VisualisationVector/{number_of_datasets - 1}"][:, :]

    # Open the new mesh file in read-write mode
    with h5py.File(predeformed_mesh_path, 'r+') as vectorData:
        ArrayNames = ['mesh/coordinates', 'domains/coordinates', 'boundaries/coordinates']
        for ArrayName in ArrayNames:
            vectorArray = vectorData[ArrayName]
            modified = vectorData[ArrayName][:, :] + disp_array * scale_factor
            vectorArray[...] = modified

    print("Mesh predeformed successfully!")


def main() -> None:
    """
    Main function for parsing arguments and predeforming the mesh.

    Returns:
        None
    """
    args = parse_arguments()

    folder_path = Path(args.folder)
    if args.mesh_path is None:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
    else:
        mesh_path = Path(args.mesh_path)

    predeform_mesh(folder_path, Path(mesh_path), args.scale_factor)


if __name__ == '__main__':
    main()
