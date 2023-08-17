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
from os import path
import h5py
from shutil import copyfile


def parse_arguments():
    parser = ArgumentParser(description=__doc__, formatter_class=RawDescriptionHelpFormatter)
    parser.add_argument('--folder', type=str, help="Path to simulation results")
    return parser.parse_args()


def predeform_mesh():
    args = parse_arguments()

    folder_path = args.folder

    # Path to the displacement file
    disp_path = path.join(folder_path, "Visualization", "displacement.h5")
    mesh_path = path.join(folder_path, "Checkpoint", "mesh.h5")

    # Read the displacement file and get the displacement from the last time step
    vectorData = h5py.File(disp_path, "r")
    number_of_datasets = len(vectorData["VisualisationVector"].keys())
    disp_array = vectorData[f"VisualisationVector/{number_of_datasets - 1}"][:, :]

    # Create a copy of the mesh file with a new name
    predeformed_mesh_path = mesh_path.replace(".h5", "_predeformed.h5")
    copyfile(mesh_path, predeformed_mesh_path)

    # Open the new mesh file in append mode
    vectorData = h5py.File(predeformed_mesh_path, 'a')

    # We modify the original geometry by adding the reverse of the displacement
    # Hence, scaleFactor = -1.0
    scaleFactor = -1.0

    ArrayNames = ['mesh/coordinates', 'domains/coordinates', 'boundaries/coordinates']
    for ArrayName in ArrayNames:
        vectorArray = vectorData[ArrayName]
        modified = vectorData[ArrayName][:, :] + disp_array * scaleFactor
        vectorArray[...] = modified

    vectorData.close()


if __name__ == '__main__':
    predeform_mesh()
