# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import json

from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters, XDMFFile
from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names
from fsipy.automatedPostprocessing.postprocessing_fenics import postprocessing_fenics_common


# set compiler arguments
parameters["reorder_dofs_serial"] = False


def create_separate_domain_visualization(visualization_separate_domain_folder, mesh_path, stride=1):
    """
    Loads displacement and velocity from .h5 file given by create_hdf5.py,
    converts and saves to .xdmf format for visualization (in e.g. ParaView).
    This function works with MPI. If the displacement was saved for the entire domain,
    no additional xdmf file will be created for the displacement.
    Args:
        visualization_separate_domain_folder (Path): Path to the folder containing the .h5 files
        mesh_path (Path): Path to the mesh that contains both fluid and solid domain
        stride (int): Save frequency of visualization output (default: 1)
    """
    # Path to the input files
    try:
        file_path_d = visualization_separate_domain_folder / "d_solid.h5"
        assert file_path_d.exists()
        extract_solid_only = True
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using d_solid.h5 file \n")
    except AssertionError:
        file_path_d = visualization_separate_domain_folder / "d.h5"
        assert file_path_d.exists()
        extract_solid_only = False
        if MPI.rank(MPI.comm_world) == 0:
            print("--- displacement is for the entire domain \n")
            print("--- No additional xdmf file will be created for displacement \n")

    file_path_u = visualization_separate_domain_folder / "u.h5"
    assert file_path_u.exists(), f"Velocity file {file_path_u} not found.  Make sure to run create_hdf5.py first."

    # Define HDF5Files and get dataset names
    if extract_solid_only:
        file_d = HDF5File(MPI.comm_world, str(file_path_d), "r")
        dataset_d = get_dataset_names(file_d, step=stride, vector_filename="/displacement/vector_%d")

    file_u = HDF5File(MPI.comm_world, str(file_path_u), "r")
    dataset_u = get_dataset_names(file_u, step=stride, vector_filename="/velocity/vector_%d")

    # Define mesh path related variables
    fluid_domain_path = mesh_path.with_name(mesh_path.stem + "_fluid.h5")
    assert fluid_domain_path.exists(), f"Fluid mesh file {fluid_domain_path} not found."

    if extract_solid_only:
        solid_domain_path = mesh_path.with_name(mesh_path.stem + "_solid.h5")
        assert solid_domain_path.exists(), f"Solid mesh file {solid_domain_path} not found."

    # Read fluid and solid mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Reading fluid and solid mesh files \n")

    mesh_fluid = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_domain_path), "r") as mesh_file:
        mesh_file.read(mesh_fluid, "mesh", False)

    if extract_solid_only:
        mesh_solid = Mesh()
        with HDF5File(MPI.comm_world, str(solid_domain_path), "r") as mesh_file:
            mesh_file.read(mesh_solid, "mesh", False)

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    Vf = VectorFunctionSpace(mesh_fluid, "CG", 1)
    u = Function(Vf)
    if extract_solid_only:
        Vs = VectorFunctionSpace(mesh_solid, "CG", 1)
        d = Function(Vs)

    # Create writer for displacement and velocity
    if extract_solid_only:
        d_path = visualization_separate_domain_folder / "displacement_solid.xdmf"
        d_writer = XDMFFile(MPI.comm_world, str(d_path))
        d_writer.parameters["flush_output"] = True
        d_writer.parameters["functions_share_mesh"] = False
        d_writer.parameters["rewrite_function_mesh"] = False

    u_path = visualization_separate_domain_folder / "velocity_fluid.xdmf"
    u_writer = XDMFFile(MPI.comm_world, str(u_path))
    u_writer.parameters["flush_output"] = True
    u_writer.parameters["functions_share_mesh"] = False
    u_writer.parameters["rewrite_function_mesh"] = False

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    for i in range(len(dataset_u)):
        if extract_solid_only:
            file_d.read(d, dataset_d[i])
        file_u.read(u, dataset_u[i])

        timestamp = file_u.attributes(dataset_u[i])["timestamp"]
        if MPI.rank(MPI.comm_world) == 0:
            print("=" * 10, "Timestep: {}".format(timestamp), "=" * 10)

        # Store velocity
        u.rename("velocity", "velocity")
        u_writer.write(u, timestamp)

        # Store displacement
        if extract_solid_only:
            d.rename("displacement", "displacement")
            d_writer.write(d, timestamp)

    # Close files
    if extract_solid_only:
        d_writer.close()
    u_writer.close()

    if MPI.rank(MPI.comm_world) == 0:
        print("========== Post processing finished ========== \n")
        print(f"--- Visualization files are saved in: {visualization_separate_domain_folder.absolute()}")


def main() -> None:
    args = postprocessing_fenics_common.parse_arguments()

    # Check if unused arguments are passed
    if MPI.rank(MPI.comm_world) == 0:
        if args.start_time is not None:
            print("--- Warning: start_time is not used in this script. \n")
        if args.end_time is not None:
            print("--- Warning: end_time is not used in this script. \n")
        if args.extract_solid_only:
            print("--- Warning: extract_solid_only is not used in this script. \n")
        if args.log_level != 20:
            print("--- Warning: log_level is not used in this script. \n")

    if MPI.size(MPI.comm_world) == 1:
        print("--- Running in serial mode, you can use MPI to speed up the postprocessing. \n")

    # Define paths for visulization and mesh files
    folder_path = Path(args.folder)
    assert folder_path.exists(), f"Folder {folder_path} not found."
    visualization_separate_domain_folder = folder_path / "Visualization_separate_domain"

    # Read parameters from default_variables.json
    parameter_path = folder_path / "Checkpoint" / "default_variables.json"
    with open(parameter_path, "r") as f:
        parameters = json.load(f)
        save_deg = parameters["save_deg"]

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using user-defined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    elif save_deg == 2:
        mesh_path = folder_path / "Mesh" / "mesh_refined.h5"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using refined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    else:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using non-refined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    create_separate_domain_visualization(visualization_separate_domain_folder, mesh_path, args.stride)


if __name__ == "__main__":
    main()
