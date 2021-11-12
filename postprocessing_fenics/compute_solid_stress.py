from pathlib import Path

import numpy as np
import h5py
from dolfin import *
import os
from postprocessing_common import read_command_line_stress
import stress_strain

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. Doesnt affect the speed
parameters["reorder_dofs_serial"] = False


def compute_stress(case_path, mesh_name, E_s, nu_s, dt, stride, save_deg):

    """
    Loads displacement fields from completed FSI simulation,
    and computes and saves the following solid mechanical quantities:
    (1) True Stress
    (2) Infinitesimal Strain
    (3) Maximum Principal Stress (True)
    (4) Maximum Principal Strain (Infinitesimal)

    Args:
        case_path (Path): Path to results from simulation
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        E_s (float): Elastic Modulus
        nu_s (float): Poisson's Ratio
        dt (float): Actual ime step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (v.h5/d.h5 in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)

    """

    # Calculate solid Lame parameters
    mu_s = E_s/(2*(1+nu_s))  # 0.345E6
    lambda_s = nu_s*2.*mu_s/(1. - 2.*nu_s)

    # File paths

    for file in os.listdir(case_path):
        file_path = os.path.join(case_path, file)
        if os.path.exists(os.path.join(file_path, "1")):
            visualization_separate_domain_path = os.path.join(file_path, "1/Visualization_separate_domain")
        elif os.path.exists(os.path.join(file_path, "Visualization")):
            visualization_separate_domain_path = os.path.join(file_path, "Visualization_separate_domain")
    
    visualization_separate_domain_path = Path(visualization_separate_domain_path)

    file_path_d = visualization_separate_domain_path / "d.h5"
    sig_path = (visualization_separate_domain_path / "TrueStress.xdmf").__str__()
    ep_path = (visualization_separate_domain_path / "InfinitesimalStrain.xdmf").__str__()
    sig_P_path = (visualization_separate_domain_path / "MaxPrincipalStress.xdmf").__str__()
    ep_P_path = (visualization_separate_domain_path / "MaxPrincipalStrain.xdmf").__str__()

    # get solid-only version of the mesh
    mesh_name = mesh_name + ".h5"
    mesh_name = mesh_name.replace(".h5","_solid_only.h5")
    mesh_path = os.path.join(case_path, "mesh", mesh_name)

    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg == 1:
        if MPI.rank(MPI.comm_world) == 0:
            print("Warning, stress results are compromised by using save_deg = 1, especially using a coarse mesh. Recommend using save_deg = 2 instead for computing stress")
        mesh_path_viz = mesh_path
    else:
        mesh_path_viz = mesh_path.replace("_solid_only.h5","_refined_solid_only.h5")
    
    mesh_path = Path(mesh_path)
    mesh_path_viz = Path(mesh_path_viz)
        
    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)

    # Read refined mesh saved as HDF5 format
    mesh_viz = Mesh()
    with HDF5File(MPI.comm_world, mesh_path_viz.__str__(), "r") as mesh_file:
        mesh_file.read(mesh_viz, "mesh", False)

    if MPI.rank(MPI.comm_world) == 0:
        print("Define function spaces and functions")

    # Create higher-order function space for d, v and p
    dve = VectorElement('CG', mesh.ufl_cell(), save_deg)
    FSdv = FunctionSpace(mesh, dve)   # Higher degree FunctionSpace for d and v

    # Create visualization function space for d, v and p
    dve_viz = VectorElement('CG', mesh_viz.ufl_cell(), 1)
    FSdv_viz = FunctionSpace(mesh_viz, dve_viz)   # Visualisation FunctionSpace for d and v

    # Create higher-order function on unrefined mesh for post-processing calculations
    d = Function(FSdv)

    # Create lower-order function for visualization on refined mesh
    d_viz = Function(FSdv_viz)
    
    # Create a transfer matrix between higher degree and lower degree (visualization) function spaces
    dv_trans = PETScDMCollection.create_transfer_matrix(FSdv_viz,FSdv)

    dx = Measure("dx")
    
    # Create tensor function space for stress and strain (this is necessary to evaluate tensor valued functions)
    '''
    Strain/stress are in L2, therefore we use a discontinuous function space with a degree of 1 for P2P1 elements
    Could also use a degree = 0 to get a constant-stress representation in each element
    For more info see the Fenics Book (P62, or P514-515), or
    https://comet-fenics.readthedocs.io/en/latest/demo/viscoelasticity/linear_viscoelasticity.html?highlight=DG#A-mixed-approach
    https://fenicsproject.org/qa/10363/what-is-the-most-accurate-way-to-recover-the-stress-tensor/
    https://fenicsproject.discourse.group/t/why-use-dg-space-to-project-stress-strain/3768
    '''
    Te = TensorElement("DG", mesh.ufl_cell(), save_deg-1) 
    Tens = FunctionSpace(mesh, Te)
    Fe = FiniteElement("DG", mesh.ufl_cell(), save_deg-1) 
    Scal = FunctionSpace(mesh, Fe)

    sig_file = XDMFFile(MPI.comm_world, sig_path)
    ep_file = XDMFFile(MPI.comm_world, ep_path)
    sig_P_file = XDMFFile(MPI.comm_world, sig_P_path)
    ep_P_file = XDMFFile(MPI.comm_world, ep_P_path)

    sig_file.parameters["rewrite_function_mesh"] = False
    ep_file.parameters["rewrite_function_mesh"] = False
    sig_P_file.parameters["rewrite_function_mesh"] = False
    ep_P_file.parameters["rewrite_function_mesh"] = False

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    file_counter = 0 # Index of first time step
    file_1 = 1 # Index of second time step

    f = HDF5File(MPI.comm_world, file_path_d.__str__(), "r")
    vec_name = "/displacement/vector_%d" % file_counter
    t_0 = f.attributes(vec_name)["timestamp"]
    vec_name = "/displacement/vector_%d" % file_1
    t_1 = f.attributes(vec_name)["timestamp"]  
    time_between_files = t_1 - t_0
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation

    while True:
        try:
            f = HDF5File(MPI.comm_world, file_path_d.__str__(), "r")
            vec_name = "/displacement/vector_%d" % file_counter
            t = f.attributes(vec_name)["timestamp"]
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Timestep: {}".format(t), "=" * 10)
            f.read(d_viz, vec_name)
        except:
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break        

        # Calculate d in P2 based on visualization refined P1
        d.vector()[:] = dv_trans*d_viz.vector()

        # Deformation Gradient and first Piola-Kirchoff stress (PK1)
        deformationF = stress_strain.F_(d) # calculate deformation gradient from displacement
        
        # Cauchy (True) Stress and Infinitesimal Strain (Only accurate for small strains, ask DB for True strain calculation...)
        epsilon = stress_strain.eps(d) # Form for Infinitesimal strain (need polar decomposition if we want to calculate logarithmic/Hencky strain)
        ep = stress_strain.project_solid(epsilon,Tens,dx) # Calculate stress tensor (this projection method is 6x faster than the built in version)
        #ep = project(epsilon,Tens) # Calculate stress tensor

        S_ = stress_strain.S(d, lambda_s, mu_s)  # Form for second PK stress (using St. Venant Kirchoff Model)
        sigma = (1/stress_strain.J_(d))*deformationF*S_*deformationF.T  # Form for Cauchy (true) stress 
    
        sig = stress_strain.project_solid(sigma,Tens,dx) # Calculate stress tensor 
        #sig =project(sigma,Tens) # Calculate stress tensor

        # Calculate eigenvalues of the stress tensor (Three eigenvalues for 3x3 tensor)
        # Eigenvalues are returned as a diagonal tensor, with the Maximum Principal stress as 1-1
        eigStress11,eigStress22,eigStress33  = stress_strain.get_eig(sigma) 
        eigStrain11,eigStrain22,eigStrain33 = stress_strain.get_eig(epsilon)

        sig_P = stress_strain.project_solid(eigStress11,Scal,dx) # Calculate Principal stress tensor
        #sig_P = project(eigStress,Tens) # Calculate Principal stress tensor
        ep_P = stress_strain.project_solid(eigStrain11,Scal,dx) # Calculate Principal stress tensor
        #ep_P = project(eigStrain,Tens) # Calculate Principal stress tensor
    
        # Name function
        ep.rename("InfinitesimalStrain", "ep")
        sig.rename("TrueStress", "sig")
        ep_P.rename("MaximumPrincipalStrain", "ep_P")
        sig_P.rename("MaximumPrincipalStress", "sig_P")

        if MPI.rank(MPI.comm_world) == 0:
            print("Writing Additional Viz Files for Stresses and Strains!")

        # Write results
        ep_file.write(ep, t)
        sig_file.write(sig, t)
        ep_P_file.write(ep_P, t)
        sig_P_file.write(sig_P, t)

        # Update file_counter
        file_counter += stride


if __name__ == '__main__':
    folder, mesh, E_s, nu_s, dt, stride, save_deg = read_command_line_stress()
    compute_stress(folder,mesh, E_s, nu_s, dt, stride, save_deg)
