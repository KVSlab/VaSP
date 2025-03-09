# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later
# Contributions:
#  Kei Yamamoto 2023

"""
This script computes hemodynamic indices from the velocity field.
It is assumed that the user has already run create_hdf5.py to create the hdf5 files
and obtained u.h5 in the Visualization_separate_domain folder.
"""
# following two imports need to come before dolfin
from mpi4py import MPI
from petsc4py import PETSc  # noqa: F401
import logging
import numpy as np
from pathlib import Path

from dolfin import Mesh, HDF5File, VectorFunctionSpace, Function, MPI, parameters, XDMFFile, TrialFunction, \
    TestFunction, inner, ds, assemble, FacetNormal, sym, FunctionSpace, PETScDMCollection, grad, \
    LUSolver, FunctionAssigner, BoundaryMesh  # noqa: F811

from vampy.automatedPostprocessing.postprocessing_common import get_dataset_names
from vasp.postprocessing.postprocessing_common import read_parameters_from_file
from vasp.postprocessing.postprocessing_fenics.postprocessing_fenics_common import project_dg, parse_arguments
from vasp.postprocessing.postprocessing_fenics.create_hdf5 import create_hdf5

# set compiler arguments
# this was necessary for num_sub_spaces() to work with MPI by Kei 2024
parameters["reorder_dofs_serial"] = True


class InterpolateDG:
    """
    interpolate DG function from the domain to the boundary. FEniCS built-in function interpolate does not work
    with DG function spaces. This class is a workaround for this issue. Basically, for each facet, we find the
    mapping between the dofs on the boundary and the dofs on the domain. Then, we copy the values of the dofs on the
    domain to the dofs on the boundary. This is done for each subspaces of the DG vector function space.
    """
    def __init__(self, V: VectorFunctionSpace, V_sub: VectorFunctionSpace, mesh: Mesh, boundary_mesh: Mesh) -> None:
        """
        Initialize the interpolator

        Args:
            V (VectorFunctionSpace): function space on the domain
            V_sub (VectorFunctionSpace): function space on the boundary
            mesh (Mesh): whole mesh
            boundary_mesh (Mesh): boundary mesh of the whole mesh
        """
        assert V.ufl_element().family() == "Discontinuous Lagrange", "V must be a DG space"
        assert V_sub.ufl_element().family() == "Discontinuous Lagrange", "V_sub must be a DG space"
        self.V = V
        self.v_sub = Function(V_sub)
        self.Ws = [V_sub.sub(i).collapse() for i in range(V_sub.num_sub_spaces())]
        self.ws = [Function(Wi) for Wi in self.Ws]
        self.w_sub_copy = [w_sub.vector().get_local() for w_sub in self.ws]
        self.sub_dofmaps = [W_sub.dofmap() for W_sub in self.Ws]
        self.sub_coords = [Wi.tabulate_dof_coordinates() for Wi in self.Ws]
        self.mesh = mesh
        self.sub_map = boundary_mesh.entity_map(self.mesh.topology().dim() - 1).array()
        self.mesh.init(self.mesh.topology().dim() - 1, self.mesh.topology().dim())
        self.f_to_c = self.mesh.topology()(self.mesh.topology().dim() - 1, self.mesh.topology().dim())
        self.dof_coords = V.tabulate_dof_coordinates()
        self.fa = FunctionAssigner(V_sub, self.Ws)

    def __call__(self, u_vec: np.ndarray) -> Function:
        """interpolate DG function from the domain to the boundary"""

        for k, (coords_k, vec, sub_dofmap) in enumerate(zip(self.sub_coords, self.w_sub_copy, self.sub_dofmaps)):
            for i, facet in enumerate(self.sub_map):
                cells = self.f_to_c(facet)
                # Get closure dofs on parent facet
                sub_dofs = sub_dofmap.cell_dofs(i)
                closure_dofs = self.V.sub(k).dofmap().entity_closure_dofs(
                    self.mesh, self.mesh.topology().dim(), [cells[0]])
                copy_dofs = np.empty(len(sub_dofs), dtype=np.int32)

                for dof in closure_dofs:
                    for j, sub_coord in enumerate(coords_k[sub_dofs]):
                        if np.allclose(self.dof_coords[dof], sub_coord):
                            copy_dofs[j] = dof
                            break
                sub_dofs = sub_dofmap.cell_dofs(i)
                vec[sub_dofs] = u_vec[copy_dofs]

            self.ws[k].vector().set_local(vec)

        self.fa.assign(self.v_sub, self.ws)

        return self.v_sub


class SurfaceProjector:
    """
    Project a function contains surface integral onto a function space V
    """
    def __init__(self, V: FunctionSpace) -> None:
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
    def __init__(self, u: Function, V_dg: VectorFunctionSpace, V_sub: VectorFunctionSpace, mu_f: float, mesh: Mesh,
                 boundary_mesh: Mesh) -> None:
        """
        Initialize the stress object

        Args:
            u (Function): velocity field
            mu_f (float): dynamic viscosity
            mesh (Mesh): mesh
            velocity_degree (int): degree of velocity field
        """
        assert V_dg.ufl_element().family() == "Discontinuous Lagrange", "V_dg must be a DG space"
        self.projector = SurfaceProjector(V_dg)
        self.interpolator = InterpolateDG(V_dg, V_sub, mesh, boundary_mesh)

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
        self.Ftv_bd = self.interpolator(self.Ftv.vector().get_local())

        return self.Ftv_bd


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
    mesh_name = mesh_path.stem
    fluid_mesh_path = mesh_path.parent / f"{mesh_name}_fluid.h5"

    mesh = Mesh()
    with HDF5File(MPI.comm_world, str(fluid_mesh_path), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    boundary_mesh = BoundaryMesh(mesh, "exterior")

    refined_mesh_path = mesh_path.parent / f"{mesh_name}_refined_fluid.h5"
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

    # Create function space for the boundary mesh
    Vv_boundary = VectorFunctionSpace(boundary_mesh, "DG", 1)
    V_boundary = FunctionSpace(boundary_mesh, "DG", 1)
    # Create function space for hemodynamic indices with DG1 elements
    Vv = VectorFunctionSpace(mesh, "DG", 1)

    if MPI.rank(MPI.comm_world) == 0:
        print("--- Define functions")

    # u_p2 is the velocity on the refined mesh with P2 elements
    u_p2 = Function(Vv_non_refined)
    # u_p1 is the velocity on the refined mesh with P1 elements
    u_p1 = Function(Vv_refined)

    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    u_transfer_matrix = PETScDMCollection.create_transfer_matrix(Vv_refined, Vv_non_refined)

    # Time-dependent wall shear stress
    WSS = Function(Vv_boundary)

    # Relative residence time
    RRT = Function(V_boundary)

    # Oscillatory shear index
    OSI = Function(V_boundary)

    # Endothelial cell activation potential
    ECAP = Function(V_boundary)

    # Time averaged wall shear stress and mean WSS magnitude
    TAWSS = Function(V_boundary)
    WSS_mean = Function(Vv_boundary)

    # Temporal wall shear stress gradient
    TWSSG = Function(V_boundary)
    twssg = Function(Vv_boundary)
    tau_prev = Function(Vv_boundary)

    # Define stress object with P2 elements and non-refined mesh
    stress = Stress(u=u_p2, V_dg=Vv, V_sub=Vv_boundary, mu_f=mu_f,
                    mesh=mesh, boundary_mesh=boundary_mesh)

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

        # compute the magnitude of WSS
        local_size = tau.vector()[:].size // Vv_boundary.num_sub_spaces()
        work_vec = tau.vector().get_local()
        tau_vec = work_vec.reshape(local_size, Vv_boundary.dofmap().block_size()).copy()
        tau_tmp = Function(Vv_boundary)
        work_vec[:] = 0
        # instead of using sqrt(inner(tau, tau)), we use np.linalg.norm to avoid the issue with inner(tau, tau) being
        # negative value. Here, we simply compute the magnitude of the dofs
        work_vec[::Vv_boundary.num_sub_spaces()] = np.linalg.norm(tau_vec, axis=1)
        tau_tmp.vector().set_local(work_vec)
        tau_tmp.vector().apply('insert')
        V0 = Vv_boundary.sub(0).collapse()
        tau_norm = Function(V0)
        assigner = FunctionAssigner(V0, Vv_boundary.sub(0))
        assigner.assign(tau_norm, tau_tmp.sub(0))
        TAWSS.vector()[:] += tau_norm.vector().get_local()

        # Simply accumulate WSS for computing OSI and ECAP later
        WSS_mean.vector().axpy(1, tau.vector())

        # Compute TWSSG
        twssg.vector().set_local((tau.vector().get_local() - tau_prev.vector().get_local()) / dt)
        twssg.vector().apply("insert")
        twssg_ = project_dg(inner(twssg, twssg) ** (1 / 2), V_boundary)
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
    local_size = WSS_mean.vector()[:].size // Vv_boundary.num_sub_spaces()
    work_vec = WSS_mean.vector().get_local()
    wss_mean_vec = work_vec.reshape(local_size, Vv_boundary.dofmap().block_size()).copy()
    wss_mean_mag_tmp = Function(Vv_boundary)
    work_vec[:] = 0
    work_vec[::Vv_boundary.num_sub_spaces()] = np.linalg.norm(wss_mean_vec, axis=1)
    wss_mean_mag_tmp.vector().set_local(work_vec)
    wss_mean_mag_tmp.vector().set_local(work_vec)

    wss_mean_mag = Function(V0)
    assigner = FunctionAssigner(V0, Vv_boundary.sub(0))
    assigner.assign(wss_mean_mag, wss_mean_mag_tmp.sub(0))
    tawss_vec = index_dict['TAWSS'].vector().get_local()
    wss_mean_mag = wss_mean_mag.vector().get_local()
    # Compute RRT, OSI, and ECAP based on mean and absolute WSS
    index_dict['RRT'].vector().set_local(1 / wss_mean_mag)
    index_dict['OSI'].vector().set_local(0.5 * (1 - wss_mean_mag / tawss_vec))
    index_dict['ECAP'].vector().set_local(index_dict['OSI'].vector().get_local() / tawss_vec)

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

    # assert that OSI is within 0 to 0.5
    min = index_dict['OSI'].vector().get_local().min()
    max = index_dict['OSI'].vector().get_local().max()

    tol = 1e-12
    assert -tol <= min < 0.5, "OSI min is not within 0 to 0.5"
    assert -tol < max <= 0.5 + tol, "OSI max is not within 0 to 0.5"


def main() -> None:
    if MPI.size(MPI.comm_world) == 1:
        print("--- Running in serial mode, you can use MPI to speed up the postprocessing. \n")

    args = parse_arguments()
    folder_path = Path(args.folder)

    assert folder_path.exists(), f"Folder {folder_path} not found."

    visualization_separate_domain_folder = folder_path / "Visualization_separate_domain"
    parameters = read_parameters_from_file(args.folder)
    if parameters is None:
        raise RuntimeError("Error reading parameters from file.")

    if visualization_separate_domain_folder.exists():
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Visualization_separate_domain folder found \n")
    else:
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Visualization_separate_domain folder not found \n")
            folder_path = Path(args.folder)
            visualization_path = folder_path / "Visualization"

            # extract necessary parameters
            save_deg = parameters["save_deg"]
            dt = parameters["dt"]
            save_step = parameters["save_step"]
            save_time_step = dt * save_step
            logging.info(f"save_time_step: {save_time_step} \n")
            fluid_domain_id = parameters["dx_f_id"]
            solid_domain_id = parameters["dx_s_id"]

            logging.info(f"--- Fluid domain ID: {fluid_domain_id} and Solid domain ID: {solid_domain_id} \n")

            if args.mesh_path:
                mesh_path = Path(args.mesh_path)
                logging.info("--- Using user-defined mesh \n")
                assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
            elif save_deg == 2:
                mesh_path = folder_path / "Mesh" / "mesh_refined.h5"
                logging.info("--- Using refined mesh \n")
                assert mesh_path.exists(), f"Mesh file {mesh_path} not found."
            else:
                mesh_path = folder_path / "Mesh" / "mesh.h5"
                logging.info("--- Using non-refined mesh \n")
                assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

            if args.extract_entire_domain:
                extract_solid_only = False
            else:
                extract_solid_only = True

            print(f"save_time_step: {save_time_step} \n")
            # call from single processor
            print("--- Creating HDF5 file for fluid velocity and solid displacement \n")
            create_hdf5(visualization_path, mesh_path, save_time_step, args.stride, args.start_time,
                        args.end_time, extract_solid_only, fluid_domain_id, solid_domain_id)

    MPI.comm_world.barrier()

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
        mesh_path = folder_path / "Mesh" / "mesh.h5"
        if MPI.rank(MPI.comm_world) == 0:
            print("--- Using mesh from default turrtleFSI Mesh folder \n")
        assert mesh_path.exists(), f"Mesh file {mesh_path} not found."

    compute_hemodyanamics(visualization_separate_domain_folder, mesh_path, mu_f, args.stride)


if __name__ == "__main__":
    main()
