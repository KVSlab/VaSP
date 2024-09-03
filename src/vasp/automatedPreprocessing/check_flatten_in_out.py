import argparse
import h5py
from pathlib import Path
from shutil import copyfile
import numpy as np
import os


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--mesh-path", type=Path, default=None, help="Path to the mesh file"
    )
    parser.add_argument(
        "--num-inlets-outlets",
        type=int,
        help="Combined number of inlets and outlets (i.e, input 2 for 1 inlet and 1 outlet)",
    )
    return parser.parse_args()


def check_flatten_boundary(num_inlets_outlets, mesh_path, threshold_stdev=0.001):
    """
    Check whether inlets and outlets are flat, then flatten them if necessary

    Returns:
        .h5 file of mesh with flattened outlets (flat_in_out_mesh_path)
    """
    flat_in_out_mesh_path = Path(str(mesh_path).replace(".h5", "_flat_outlet.h5"))
    copyfile(mesh_path, flat_in_out_mesh_path)
    delete_mesh = True  # For later, if outlets are already flat delete the mesh at
    # flat_in_out_mesh_path

    vectorData = h5py.File(flat_in_out_mesh_path, "a")
    facet_ids = np.array(vectorData["boundaries/values"])
    facet_topology = vectorData["boundaries/topology"]

    for inlet_id in range(2, 2 + num_inlets_outlets):
        inlet_facet_ids = [i for i, x in enumerate(facet_ids) if x == inlet_id]
        inlet_facet_topology = facet_topology[inlet_facet_ids, :]
        inlet_nodes = np.unique(inlet_facet_topology.flatten())
        # pre-allocate arrays
        inlet_facet_normals = np.zeros((len(inlet_facet_ids), 3))

        # From: https://stackoverflow.com/questions/53698635/
        # how-to-define-a-plane-with-3-points-and-plot-it-in-3d
        for idx, facet in enumerate(inlet_facet_topology):
            p0 = vectorData["boundaries/coordinates"][facet[0]]
            p1 = vectorData["boundaries/coordinates"][facet[1]]
            p2 = vectorData["boundaries/coordinates"][facet[2]]

            x0, y0, z0 = p0
            x1, y1, z1 = p1
            x2, y2, z2 = p2

            ux, uy, uz = [x1 - x0, y1 - y0, z1 - z0]  # Vectors
            vx, vy, vz = [x2 - x0, y2 - y0, z2 - z0]

            # cross product of vectors defines the plane normal
            u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
            normal = np.array(u_cross_v)

            # Facet unit normal vector (u_normal)
            u_normal = normal / np.sqrt(
                normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2
            )

            # check if facet unit normal vector has opposite
            # direction and reverse the vector if necessary
            if idx == 0:
                u_normal_baseline = u_normal
            else:
                angle = np.arccos(
                    np.clip(np.dot(u_normal_baseline, u_normal), -1.0, 1.0)
                )
                if angle > np.pi / 2:
                    u_normal = -u_normal

            # record u_normal
            inlet_facet_normals[idx, :] = u_normal

        # Average normal and d (we will assign this later to all facets)
        normal_avg = np.mean(inlet_facet_normals, axis=0)
        inlet_coords = np.array(vectorData["boundaries/coordinates"][inlet_nodes])
        point_avg = np.mean(inlet_coords, axis=0)
        d_avg = -point_avg.dot(normal_avg)  # plane coefficient

        # Standard deviation of components of normal vector
        normal_stdev = np.std(inlet_facet_normals, axis=0)
        if np.max(normal_stdev) > threshold_stdev:  # if surface is not flat
            print(
                "Surface with ID {} is not flat: Standard deviation of facet unit\
normals is {}, greater than threshold of {}".format(
                    inlet_id, np.max(normal_stdev), threshold_stdev
                )
            )

            # Move the inlet nodes into the average inlet plane (do same for outlets)
            ArrayNames = [
                "boundaries/coordinates",
                "mesh/coordinates",
                "domains/coordinates",
            ]
            print("Moving nodes into a flat plane")
            for ArrayName in ArrayNames:
                vectorArray = vectorData[ArrayName]
                for node_id in range(len(vectorArray)):
                    if node_id in inlet_nodes:
                        # from https://stackoverflow.com/questions/9605556/
                        # how-to-project-a-point-onto-a-plane-in-3d (bobobobo)
                        node = vectorArray[node_id, :]
                        scalar_distance = node.dot(normal_avg) + d_avg
                        node_inplane = node - scalar_distance * normal_avg
                        vectorArray[node_id, :] = node_inplane
            delete_mesh = False

    vectorData.close()
    if delete_mesh is True:
        print(
            "Outlets and Inlets are all flat (Standard deviation of facet unit\
normals is less than {})".format(
                threshold_stdev
            )
        )
        os.remove(flat_in_out_mesh_path)


def main():
    args = parse_arguments()
    check_flatten_boundary(
        args.num_inlets_outlets, args.mesh_path, threshold_stdev=0.001
    )


if __name__ == "__main__":
    main()
