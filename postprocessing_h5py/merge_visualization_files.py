#!/usr/bin/env python

import sys
from pathlib import Path
from xml.etree import ElementTree as ET


def merge_visualization_files(visualization_folder, **namesapce):
    # Gather files
    xdmf_files = list(visualization_folder.glob("*.xdmf"))
    xdmf_displacement = [f for f in xdmf_files if "displacement" in f.__str__()]
    xdmf_velocity = [f for f in xdmf_files if "velocity" in f.__str__()]
    xdmf_pressure = [f for f in xdmf_files if "pressure" in f.__str__()]

    # Merge files
    for files in [xdmf_displacement, xdmf_velocity, xdmf_pressure]:
        print(files)
        if len(files) > 1:
            merge_xml_files(files)


def merge_xml_files(files):
    # Get first timestep and trees
    first_timesteps = []
    trees = []
    for f in files:
        trees.append(ET.parse(f))
        root = trees[-1].getroot()
        first_timesteps.append(float(root[0][0][0][2].attrib["Value"]))

    # Index valued sort (bypass numpy dependency)
    first_timestep_sorted = sorted(first_timesteps)
    indexes = [first_timesteps.index(i) for i in first_timestep_sorted]

    # Get last timestep of first tree
    base_tree = trees[indexes[0]]
    last_node = base_tree.getroot()[0][0][-1]
    ind = 1 if len(list(last_node)) == 3 else 2
    last_timestep = float(last_node[ind].attrib["Value"])

    # Append
    for index in indexes[1:]:
        tree = trees[index]
        for node in list(tree.getroot()[0][0]):
            ind = 1 if len(list(node)) == 3 else 2
            if last_timestep < float(node[ind].attrib["Value"]):
                base_tree.getroot()[0][0].append(node)
                last_timestep = float(node[ind].attrib["Value"])

    # Seperate xdmf files
    new_file = [f for f in files if "_" not in f.name.__str__()]
    old_files = [f for f in files if "_" in f.name.__str__()]

    # Write new xdmf file
    base_tree.write(new_file[0], xml_declaration=True)

    # Delete xdmf file
    [f.unlink() for f in old_files]


if __name__ == '__main__':
    try:
        folder = Path(sys.argv[1])
    except:
        print("usage: {} <folder>".format(sys.argv[0]))
        sys.exit(1)

    merge_visualization_files(folder)
