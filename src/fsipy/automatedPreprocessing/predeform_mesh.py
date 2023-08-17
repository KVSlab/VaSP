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

from argparse import ArgumentParser, RawDescriptionHelpFormatter
import h5py
from pathlib import Path

def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=str, required=True, help="Path to simulation results")
    parser.add_argument('--mesh-path', type=str, default=None,
                        help="Path to the mesh file (default: <folder_path>/Checkpoint/mesh.h5)")
    return parser.parse_args()

def predeform_mesh(folder_path: str, mesh_path: str) -> None:
    """
    Predeform the mesh for FSI simulation.

    Args:
        folder_path (str): Path to the simulation results folder.
        mesh_path (str): Path to the mesh file.

    Returns:
        None
    """
    # Path to the displacement file
    disp_path = Path(folder_path) / "Visualization" / "displacement.h5"
    if mesh_path is None:
        mesh_path = Path(folder_path) / "Checkpoint" / "mesh.h5"
    predeformed_mesh_path = mesh_path.with_name(mesh_path.stem + "_predeformed.h5")

    # Read the displacement file and get the displacement from the last time step
    with h5py.File(disp_path, "r") as vectorData:
        number_of_datasets = len(vectorData["VisualisationVector"].keys())
        disp_array = vectorData[f"VisualisationVector/{number_of_datasets - 1}"][:, :]

    # Open the new mesh file in read-write mode
    with h5py.File(predeformed_mesh_path, 'r+') as vectorData:
        # We modify the original geometry by adding the reverse of the displacement
        # Hence, scaleFactor = -1.0
        scaleFactor = -1.0

        ArrayNames = ['mesh/coordinates', 'domains/coordinates', 'boundaries/coordinates']
        for ArrayName in ArrayNames:
            vectorArray = vectorData[ArrayName]
            modified = vectorData[ArrayName][:, :] + disp_array * scaleFactor
            vectorArray[...] = modified

def main() -> None:
    """
    Main function for parsing arguments and predeforming the mesh.

    Returns:
        None
    """
    args = parse_arguments()
    predeform_mesh(args.folder, args.mesh_path)

if __name__ == '__main__':
    main()
