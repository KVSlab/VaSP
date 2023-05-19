import os
import h5py
import numpy as np
from argparse import ArgumentParser

"""
Author: Kei Yamamoto <keiya@simula.no>
Last updated: 2023/05/19
When restarting a simulation in turtleFSI, the visualization files are not always correct due to different mesh partitioning.
This scripts fixes the visualization files by checking mesh in h5 files and swapping the node numbering, tpoology, and vector values.
After running this script, you can use the combine_xdmf.py script to merge the visualization files.
"""

def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--folder", help="Path to the folder containing the visualization files")
    parser.add_argument("--correct", help="Path to the correct visualization file")
    parser.add_argument("--wrong", help="Path to the wrong visualization file")
    args = parser.parse_args()
    return args

def main(folder, correct, wrong):
    """
    Args:
        folder (str): Path to the folder containing the visualization files
        correct (str): Path to the correct visualization file, usually velocity/displacement/pressure.h5 
        wrong (str): Path to the wrong visualization file, usually velocity/displacement/pressure_run_{i}.h5, i = 1, 2, 3, ...

    Returns:
        None
    """
    # Here we find the path to the visualization files 
    wrongNumberVizPath = os.path.join(folder, wrong)
    correctNumberVizPath = os.path.join(folder, correct)

    # Open the files using h5py, r+ means read and write
    wrongNumberViz = h5py.File(wrongNumberVizPath, 'r+')
    correctNumberViz = h5py.File(correctNumberVizPath, 'r+')

    # Get the mesh coordinates from the mesh
    wrongNumberNodes = wrongNumberViz['Mesh/0/mesh/geometry'][:]
    correctNumberNodes = correctNumberViz['Mesh/0/mesh/geometry'][:]

    # Here, we simply copy toplogy from the correct file to the wrong file if they are not the same
    if (correctNumberViz['Mesh/0/mesh/topology'][:] != wrongNumberViz['Mesh/0/mesh/topology'][:]).all():
        print('Topology is not the same')
        wrongNumberViz['Mesh/0/mesh/topology'][...] = correctNumberViz['Mesh/0/mesh/topology'][:]
        print('Topology is now fixed')
    else:
        print('Topology is the same')

    # Check if the node numbering is correct
    if (correctNumberNodes == wrongNumberNodes).all():
        print('Node numbering is correct')
        print('...exiting')
        exit()
    else:
        print('Node numbering is incorrect')
   
    # Create a dictionary with the index and coordinates of the correct node numbering
    index_dict =  {index: value for index, value in enumerate(correctNumberNodes)}
    map_index = []
    
    # create a list of the index for swapping the wrong node numbering with the correct node numbering
    
    for i, values in enumerate(index_dict.values()):
        if i % 1000 == 0:
            print(f"Creating map_index: Going through {i} nodes out of {len(index_dict)} nodes")
        for j in range(len(wrongNumberNodes)):
            if (values == wrongNumberNodes[j]).all():
                map_index.append(j)

    # fix the node numbering in the wrong visualization file and overwrite the wrong h5 file
    fixed_nodes = wrongNumberNodes[map_index]
    wrongNumberViz['Mesh/0/mesh/geometry'][...] = fixed_nodes

    assert (correctNumberNodes == wrongNumberViz['Mesh/0/mesh/geometry'][:]).all()
    print('Node numbering is now fixed')

    # based on the index, we can now fix the node numbering in the h5 file
    for i in range(len(wrongNumberViz["VisualisationVector"].keys())):
        if i % 1000 == 0:
            print(f"Fixing vectors: Going through {i} time steps out of {len(wrongNumberViz['VisualisationVector'].keys())} steps")
        wrongNumberViz["VisualisationVector"][str(i)][...] = np.array(wrongNumberViz["VisualisationVector"][str(i)])[map_index]

    # close the files
    wrongNumberViz.close()
    correctNumberViz.close()

if __name__ == '__main__':
    args = parse_args()
    folder = args.folder
    correct = args.correct
    wrong = args.wrong
    main(folder, correct, wrong)