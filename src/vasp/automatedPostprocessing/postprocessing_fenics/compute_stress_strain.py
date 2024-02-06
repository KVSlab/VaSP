# Copyright (c) 2023 David Bruneau
# Modified by Kei Yamamoto 2023
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
import argparse
from dolfin import MPI, TensorFunctionSpace, VectorFunctionSpace, FunctionSpace, \
    Function, Mesh, HDF5File, Measure, MeshFunction, as_tensor, XDMFFile, PETScDMCollection, \
    TrialFunction, TestFunction, inner, LocalSolver, parameters
from ufl.form import Form
from turtleFSI.modules import common

from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names
from vasp.automatedPostprocessing.postprocessing_common import read_parameters_from_file
from vasp.automatedPostprocessing.postprocessing_fenics.postprocessing_fenics_common import project_dg

# set compiler arguments
parameters["reorder_dofs_serial"] = False
parameters["form_compiler"]["cpp_optimize"] = True
parameters["form_compiler"]["optimize"] = True
parameters["form_compiler"]["quadrature_degree"] = 6


def parse_arguments():
    """Read arguments from commandline"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("--folder", type=Path, help="Path to simulation results folder")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file. If not given (None), " +
                             "it will assume that mesh is located <folder>/Mesh/mesh.h5)")
    parser.add_argument("--stride", type=int, default=1, help="Save frequency of output data")
    args = parser.parse_args()

    return args


def solve_dg(a: Form, L: Form, T: TensorFunctionSpace) -> Function:
    """
    Solves the strain/stress form efficiently for a DG space
    Args:
        a: Bilinear form of the variational problem
        L: Linear form of the variational problem
        T: TensorFunctionSpace
    Returns:
        t: Function
    """
    assert T.ufl_element().family() == "Discontinuous Lagrange", "Function space must be DG"
    t = Function(T)
    solver = LocalSolver(a, L)
    solver.factorize()
    solver.solve_local_rhs(t)

    return t


def compute_stress(visualization_separate_domain_folder: Path, mesh_path: Path, stride: int,
                   solid_properties: list, fluid_properties: list) -> None:
    """
    Loads displacement fields from completed FSI simulation, computes and saves
    the following solid mechanical quantities:

    (1) True (Cauchy) Stress -- tensor
    (2) Green-Lagrange Strain -- tensor
    (3) Maximum Principal Stress (Cauchy/True) -- scalar
    (4) Maximum Principal Strain (Green-Lagrange) -- scalar

    Args:
        visualization_separate_domain_folder (Path): Path to the folder containing d.h5 (or d_solid.h5) file
        mesh_path (Path): Path to the mesh file (non-refined, whole domain)
        stride (int): Save frequency of output data
        solid_properties (list): List of dictionaries containing solid properties used in the simulation
        fluid_properties (list): List of dictionaries containing fluid properties used in the simulation
    """
    # find the displacement file and check if it is for the entire domain or only for the solid domain
    try:
        file_path_d = visualization_separate_domain_folder / "d_solid.h5"
        assert file_path_d.exists(), f"Displacement file {file_path_d} not found."
        solid_only = True
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using d_solid.h5 file \n")
    except AssertionError:
        file_path_d = visualization_separate_domain_folder / "d.h5"
        assert file_path_d.exists(), f"Displacement file {file_path_d} not found."
        solid_only = False
        if MPI.rank(MPI.comm_world) == 0:
            print("--- displacement is for the entire domain \n")

    file_d = HDF5File(MPI.comm_world, str(file_path_d), "r")

    with HDF5File(MPI.comm_world, str(file_path_d), "r") as f:
        dataset = get_dataset_names(f, step=stride, vector_filename="/displacement/vector_%d")

    # Read the original mesh and also the refined mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Read the original mesh and also the refined mesh \n")

    mesh_name = mesh_path.stem
    solid_mesh_path = mesh_path.parent / f"{mesh_name}_solid.h5" if solid_only else mesh_path
    mesh = Mesh()
    with HDF5File(MPI.comm_world, str(solid_mesh_path), "r") as mesh_file:
        mesh_file.read(mesh, "/mesh", False)
        domains = MeshFunction("size_t", mesh, mesh.topology().dim())

        if solid_only and len(solid_properties) == 1:
            domains.set_all(solid_properties[0]["dx_s_id"])
        elif solid_only and len(solid_properties) > 1:
            mesh_file.read(domains, "/mesh")
        else:
            mesh_file.read(domains, "/domains")

    refined_mesh_path = mesh_path.parent / f"{mesh_name}_refined_solid.h5" if solid_only else \
        mesh_path.parent / f"{mesh_name}_refined.h5"
    refined_mesh = Mesh()
    with HDF5File(MPI.comm_world, str(refined_mesh_path), "r") as mesh_file:
        mesh_file.read(refined_mesh, "mesh", False)

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    # Create function space for the displacement on the refined mesh with P1 elements
    Vv_refined = VectorFunctionSpace(refined_mesh, "CG", 1)
    d_p1 = Function(Vv_refined)
    # Create function space for the displacement on the refined mesh with P2 elements
    Vv_non_refined = VectorFunctionSpace(mesh, "CG", 2)
    d_p2 = Function(Vv_non_refined)

    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    d_transfer_matrix = PETScDMCollection.create_transfer_matrix(Vv_refined, Vv_non_refined)

    # Create function space for stress and strain
    VT = TensorFunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "DG", 1)

    # Create functions for stress and strain
    TS = Function(VT)
    GLS = Function(VT)
    MPStress = Function(V)
    MPStrain = Function(V)
    # Create test and trial functions
    v = TestFunction(VT)
    u = TrialFunction(VT)

    # Set up dx (dx_s for solid, dx_f for fluid) for each domain
    dx = Measure("dx", subdomain_data=domains)
    dx_s = {}
    dx_s_id_list = []
    a = 0
    for idx, solid_region in enumerate(solid_properties):
        dx_s_id = solid_region["dx_s_id"]
        dx_s[idx] = dx(dx_s_id, subdomain_data=domains)
        dx_s_id_list.append(dx_s_id)

    if not solid_only:
        dx_f = {}
        dx_f_id_list = []
        for idx, fluid_region in enumerate(fluid_properties):
            dx_f_id = fluid_region["dx_f_id"]
            dx_f[idx] = dx(dx_f_id, subdomain_data=domains)
            dx_f_id_list.append(dx_f_id)
    else:
        dx_f = None
        dx_f_id_list = None

    a = 0
    for solid_region in range(len(dx_s_id_list)):
        a += inner(u, v) * dx_s[solid_region]

    if not solid_only and isinstance(dx_f_id_list, list) and isinstance(dx_f, dict):
        for fluid_region in range(len(dx_f_id_list)):
            a += inner(u, v) * dx_f[fluid_region]
    else:
        pass

    # Create XDMF files for saving stress and strain
    stress_strain_path = visualization_separate_domain_folder.parent / "StressStrain"
    stress_strain_path.mkdir(parents=True, exist_ok=True)
    stress_strain_names = ["TrueStress", "GreenLagrangeStrain", "MaxPrincipalStress", "MaxPrincipalStrain"]
    stress_strain_variables = [TS, GLS, MPStress, MPStrain]
    stress_strain_dict = dict(zip(stress_strain_names, stress_strain_variables))
    xdmf_paths = [stress_strain_path / f"{name}.xdmf" for name in stress_strain_names]

    stress_strain = {}
    for index, path in zip(stress_strain_names, xdmf_paths):
        stress_strain[index] = XDMFFile(MPI.comm_world, str(path))
        stress_strain[index].parameters["rewrite_function_mesh"] = False
        stress_strain[index].parameters["flush_output"] = True
        stress_strain[index].parameters["functions_share_mesh"] = True

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    counter = 0
    for data in dataset:
        # Read diplacement data and interpolate to P2 space
        file_d.read(d_p1, data)
        d_p2.vector()[:] = d_transfer_matrix * d_p1.vector()

        t = file_d.attributes(dataset[counter])["timestamp"]
        if MPI.rank(MPI.comm_world) == 0:
            print("=" * 10, f"Calculating Stress & Strain at Timestep: {t}", "=" * 10)

        # Deformation Gradient for computing cauchy stress
        deformationF = common.F_(d_p2)

        # Compute Green-Lagrange strain tensor
        green_lagrange_strain = common.E(d_p2)

        L_sigma = 0
        L_epsilon = 0

        for solid_region in range(len(dx_s_id_list)):
            # Form for second PK stress (using specified material model)
            PiolaKirchoff2 = common.S(d_p2, solid_properties[solid_region])
            # Form for True (Cauchy) stress
            cauchy_stress = (1 / common.J_(d_p2)) * deformationF * PiolaKirchoff2 * deformationF.T
            L_sigma += inner(cauchy_stress, v) * dx_s[solid_region]
            L_epsilon += inner(green_lagrange_strain, v) * dx_s[solid_region]

        # Here, we add almost zero values to the fluid regions if displacement is for the entire domain
        if not solid_only and isinstance(dx_f_id_list, list) and isinstance(dx_f, dict):
            for fluid_region in range(len(dx_f_id_list)):
                nought_value = 1e-10
                sigma_nought = as_tensor(
                    [
                        [nought_value, nought_value, nought_value],
                        [nought_value, nought_value, nought_value],
                        [nought_value, nought_value, nought_value],
                    ]
                )

                epsilon_nought = as_tensor(
                    [
                        [nought_value, nought_value, nought_value],
                        [nought_value, nought_value, nought_value],
                        [nought_value, nought_value, nought_value],
                    ]
                )

                L_sigma += inner(sigma_nought, v) * dx_f[fluid_region]
                L_epsilon += inner(epsilon_nought, v) * dx_f[fluid_region]

        # Calculate stress and strain
        sigma = solve_dg(a, L_sigma, VT)
        epsilon = solve_dg(a, L_epsilon, VT)

        # Calculate principal stress
        stress_eigen_value11, _, _ = common.get_eig(sigma)
        strain_eigen_value11, _, _ = common.get_eig(epsilon)
        # Project to DG space
        max_principal_stress = project_dg(stress_eigen_value11, V)
        max_principal_strain = project_dg(strain_eigen_value11, V)

        # Save stress and strain
        TS.assign(sigma)
        GLS.assign(epsilon)
        MPStress.assign(max_principal_stress)
        MPStrain.assign(max_principal_strain)
        # Write indices to file
        for name, xdmf_object in stress_strain.items():
            variable = stress_strain_dict[name]
            xdmf_object.write_checkpoint(variable, name, t, XDMFFile.Encoding.HDF5, append=True)
            xdmf_object.close()

        counter += 1


def main() -> None:
    """Main function."""
    if MPI.size(MPI.comm_world) == 1:
        print("--- Running in serial mode, you can use MPI to speed up the postprocessing. \n")

    args = parse_arguments()
    folder_path = args.folder

    visualization_separate_domain_folder = args.folder / "Visualization_separate_domain"
    assert (
        visualization_separate_domain_folder.exists()
    ), f"Visualization_separate_domain folder {visualization_separate_domain_folder} not found."

    parameters = read_parameters_from_file(args.folder)
    if parameters is None:
        raise RuntimeError("Error reading parameters from file.")
    else:
        solid_properties = parameters["solid_properties"]
        fluid_properties = parameters["fluid_properties"]

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using user-defined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    else:
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using mesh from default turrtleFSI Mesh folder \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    compute_stress(visualization_separate_domain_folder, mesh_path, args.stride, solid_properties, fluid_properties)


if __name__ == "__main__":
    main()
