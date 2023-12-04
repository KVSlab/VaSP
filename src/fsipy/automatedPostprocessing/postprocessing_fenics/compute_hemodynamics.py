# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#  Kei Yamamoto 2023

"""
This script computes hemodynamic indices from the velocity field.
It is assumed that the user has already run create_hdf5.py to create the hdf5 files
and obtained u.h5 in the Visualization_separate_domain folder.
"""

import numpy as np
from pathlib import Path
import argparse

from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters, XDMFFile, TrialFunction, \
    TestFunction, inner, ds, assemble, FacetNormal, sym, project, FunctionSpace, PETScDMCollection, grad, \
    LUSolver
from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names
from fsipy.automatedPostprocessing.postprocessing_common import read_parameters_from_file
from fsipy.automatedPostprocessing.postprocessing_fenics.postprocessing_fenics_common import project_dg

# set compiler arguments
parameters["reorder_dofs_serial"] = False


def parse_arguments():
    """Read arguments from commandline"""
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--folder', type=Path, help="Path to simulation results folder")
    parser.add_argument('--mesh-path', type=Path, default=None,
                        help="Path to the mesh file. If not given (None), " +
                             "it will assume that mesh is located <folder_path>/Mesh/mesh.h5)")
    parser.add_argument("--stride", type=int, default=1, help="Save frequency of output data")
    args = parser.parse_args()

    return args


class SurfaceProjector:
    """
    Project a function contains surface integral onto a function space V
    """
    def __init__(self, V: FunctionSpace):
        """
        Initialize the surface projector

        Args:
            V (FunctionSpace): function space to project onto
        """
        u = TrialFunction(V)
        v = TestFunction(V)
        a_proj = inner(u, v) * ds
        # keep_diagonal=True & ident_zeros() are necessary for the matrix to be invertible
        self.A = assemble(a_proj, keep_diagonal=True)
        self.A.ident_zeros()
        self.u_ = Function(V)
        self.solver = LUSolver(self.A)

    def __call__(self, f: Function) -> Function:
        v = TestFunction(self.u_.function_space())
        self.b_proj = inner(f, v) * ds
        self.b = assemble(self.b_proj)
        self.solver.solve(self.u_.vector(), self.b)
        return self.u_


class Stress:
    """
    A class to compute wall shear stress from velocity field
    Here, we use cauchy stress tensor to compute WSS. Typically, cauchy stress tensor is defined as
    sigam = mu_f * (grad(u) + grad(u).T) + p * I but one can prove that the pressure term does not contribute to WSS.
    This is consitent with the other definition, tau = mu_f * grad(u) * n, which also does not contain pressure term.
    """
    def __init__(self, u: Function, mu_f: float, mesh: Mesh, velocity_degree: int) -> None:
        """
        Initialize the stress object

        Args:
            u (Function): velocity field
            mu_f (float): dynamic viscosity
            mesh (Mesh): mesh
            velocity_degree (int): degree of velocity field
        """
        self.V = VectorFunctionSpace(mesh, 'DG', velocity_degree - 1)
        self.projector = SurfaceProjector(self.V)

        sigma = (2 * mu_f * sym(grad(u)))

        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        self.Ft = F - (Fn * n)  # vector-valued

    def __call__(self) -> Function:
        """compute stress for given velocity field u"""
        self.Ftv = self.projector(self.Ft)

        return self.Ftv


def compute_hemodyanamics(visualization_separate_domain_folder: Path, mesh_path: Path,
                          mu_f: float, stride: int = 1) -> None:
    """
    Compute hemodynamic indices from velocity field
    Definition of hemodynamic indices can be found in:
        https://kvslab.github.io/VaMPy/quantities.html

    Args:
        visualization_separate_domain_folder (Path): Path to the folder containing u.h5
        mesh_path (Path): Path to the mesh folder
        mu_f (float): Dynamic viscosity
        stride (int): Save frequency of output data
    """

    file_path_u = visualization_separate_domain_folder / "u.h5"
    assert file_path_u.exists(), f"Velocity file {file_path_u} not found.  Make sure to run create_hdf5.py first."
    file_u = HDF5File(MPI.comm_world, str(file_path_u), "r")

    with HDF5File(MPI.comm_world, str(file_path_u), "r") as f:
        dataset = get_dataset_names(f, step=stride, vector_filename="/velocity/vector_%d")

    # Read the original mesh and also the refined mesh
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Read the original mesh and also the refined mesh \n")

    fluid_mesh_path = mesh_path / "mesh_fluid.h5"
    mesh = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_mesh_path), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    refined_mesh_path = mesh_path / "mesh_refined_fluid.h5"
    refined_mesh = Mesh()
    with HDF5File(MPI.comm_world, str(refined_mesh_path), "r") as mesh_file:
        mesh_file.read(refined_mesh, "mesh", False)

    # Define functionspaces and functions
    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define function spaces \n")

    # Create function space for the velocity on the refined mesh with P1 elements
    Vv_refined = VectorFunctionSpace(refined_mesh, "CG", 1)
    # Create function space for the velocity on the refined mesh with P2 elements
    Vv_non_refined = VectorFunctionSpace(mesh, "CG", 2)

    # Create function space for hemodynamic indices with DG1 elements
    Vv = VectorFunctionSpace(mesh, "DG", 1)
    V = FunctionSpace(mesh, "DG", 1)

    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define functions")

    # u_p2 is the velocity on the refined mesh with P2 elements
    u_p2 = Function(Vv_non_refined)
    # u_p1 is the velocity on the refined mesh with P1 elements
    u_p1 = Function(Vv_refined)

    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    u_transfer_matrix = PETScDMCollection.create_transfer_matrix(Vv_refined, Vv_non_refined)

    # Time-dependent wall shear stress
    WSS = Function(Vv)

    # Relative residence time
    RRT = Function(V)

    # Oscillatory shear index
    OSI = Function(V)

    # Endothelial cell activation potential
    ECAP = Function(V)

    # Time averaged wall shear stress and mean WSS magnitude
    TAWSS = Function(V)
    WSS_mean = Function(Vv)

    # Temporal wall shear stress gradient
    TWSSG = Function(V)
    twssg = Function(Vv)
    tau_prev = Function(Vv)

    # Define stress object with P2 elements and non-refined mesh
    stress = Stress(u_p2, mu_f, mesh, velocity_degree=2)

    # Create XDMF files for saving indices
    hemodynamic_indices_path = visualization_separate_domain_folder.parent / "Hemodynamic_indices"
    hemodynamic_indices_path.mkdir(parents=True, exist_ok=True)
    index_names = ["RRT", "OSI", "ECAP", "WSS", "TAWSS", "TWSSG"]
    index_variables = [RRT, OSI, ECAP, WSS, TAWSS, TWSSG]
    index_dict = dict(zip(index_names, index_variables))
    xdmf_paths = [hemodynamic_indices_path / f"{name}.xdmf" for name in index_names]

    indices = {}
    for index, path in zip(index_names, xdmf_paths):
        indices[index] = XDMFFile(MPI.comm_world, str(path))
        indices[index].parameters["rewrite_function_mesh"] = False
        indices[index].parameters["flush_output"] = True
        indices[index].parameters["functions_share_mesh"] = True

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # Get time difference between two consecutive time steps
    dt = file_u.attributes(dataset[1])["timestamp"] - file_u.attributes(dataset[0])["timestamp"]

    counter = 0
    for data in dataset:
        # Read velocity data and interpolate to P2 space
        file_u.read(u_p1, data)
        u_p2.vector()[:] = u_transfer_matrix * u_p1.vector()

        t = file_u.attributes(dataset[counter])["timestamp"]
        if MPI.rank(MPI.comm_world) == 0:
            print("=" * 10, f"Calculating WSS at Timestep: {t}", "=" * 10)

        # compute WSS and accumulate for time-averaged WSS
        tau = stress()

        # Write temporal WSS
        tau.rename("WSS", "WSS")
        indices["WSS"].write_checkpoint(tau, "WSS", t, XDMFFile.Encoding.HDF5, append=True)

        # Compute time-averaged WSS by accumulating WSS magnitude
        tawss = project(inner(tau, tau) ** (1 / 2), V)
        TAWSS.vector().axpy(1, tawss.vector())

        # Simply accumulate WSS for computing OSI and ECAP later
        WSS_mean.vector().axpy(1, tau.vector())

        # Compute TWSSG
        twssg.vector().set_local((tau.vector().get_local() - tau_prev.vector().get_local()) / dt)
        twssg.vector().apply("insert")
        twssg_ = project_dg(inner(twssg, twssg) ** (1 / 2), V)
        TWSSG.vector().axpy(1, twssg_.vector())

        # Update tau
        tau_prev.vector().zero()
        tau_prev.vector().axpy(1, tau.vector())

        counter += 1

    indices["WSS"].close()
    file_u.close()

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Saving hemodynamic indices", "=" * 10)

    index_dict['TWSSG'].vector()[:] = index_dict['TWSSG'].vector()[:] / counter
    index_dict['TAWSS'].vector()[:] = index_dict['TAWSS'].vector()[:] / counter
    WSS_mean.vector()[:] = WSS_mean.vector()[:] / counter
    wss_mean = project(inner(WSS_mean, WSS_mean) ** (1 / 2), V)
    wss_mean_vec = wss_mean.vector().get_local()
    tawss_vec = index_dict['TAWSS'].vector().get_local()

    # Compute RRT, OSI, and ECAP based on mean and absolute WSS
    # Note that we use np.divide because WSS is zero inside the domain and we want to avoid division by zero
    rrt_divided = np.divide(np.ones_like(wss_mean_vec), wss_mean_vec,
                            out=np.zeros_like(wss_mean_vec), where=wss_mean_vec != 0)
    index_dict['RRT'].vector().set_local(rrt_divided)

    wss_divied_by_tawss = np.divide(wss_mean_vec, tawss_vec, out=np.zeros_like(wss_mean_vec), where=tawss_vec != 0)
    osi = 0.5 * (1 - wss_divied_by_tawss)
    index_dict['OSI'].vector().set_local(osi)

    ecap = np.divide(index_dict['OSI'].vector().get_local(), tawss_vec,
                     out=np.zeros_like(wss_mean_vec), where=tawss_vec != 0)
    index_dict['ECAP'].vector().set_local(ecap)

    for index in ['RRT', 'OSI', 'ECAP']:
        index_dict[index].vector().apply("insert")

    # Rename displayed variable names
    for name, var in index_dict.items():
        var.rename(name, name)

    # Write indices to file
    for name, xdmf_object in indices.items():
        index = index_dict[name]
        if name == "WSS":
            pass
        else:
            indices[name].write_checkpoint(index, name, 0, XDMFFile.Encoding.HDF5, append=False)
            indices[name].close()
            if MPI.rank(MPI.comm_world) == 0:
                print(f"--- {name} is saved in {hemodynamic_indices_path}")


def main() -> None:
    if MPI.size(MPI.comm_world) == 1:
        print("--- Running in serial mode, you can use MPI to speed up the postprocessing. \n")

    args = parse_arguments()
    # Define paths for visulization and mesh files
    folder_path = Path(args.folder)
    assert folder_path.exists(), f"Folder {folder_path} not found."
    visualization_separate_domain_folder = folder_path / "Visualization_separate_domain"
    assert visualization_separate_domain_folder.exists(), f"Folder {visualization_separate_domain_folder} not found. " \
        "Please make sure to run create_hdf5.py first."

    parameters = read_parameters_from_file(args.folder)
    if parameters is None:
        raise RuntimeError("Error reading parameters from file.")
    else:
        save_deg = parameters["save_deg"]
        assert save_deg == 2, "This script only works for save_deg = 2"
        mu_f = parameters["mu_f"]

    if isinstance(mu_f, list):
        if MPI.rank(MPI.comm_world) == 0:
            print("--- two fluid regions are detected. Using the first fluid region for viscosity \n")
        mu_f = mu_f[0]

    if args.mesh_path:
        mesh_path = Path(args.mesh_path)
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using user-defined mesh \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
    else:
        mesh_path = folder_path / "Mesh"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using mesh from default turrtleFSI Mesh folder \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    compute_hemodyanamics(visualization_separate_domain_folder, mesh_path, mu_f, args.stride)


if __name__ == "__main__":
    main()
