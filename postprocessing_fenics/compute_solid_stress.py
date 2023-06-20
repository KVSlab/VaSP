from pathlib import Path

import numpy as np
import h5py
from dolfin import *
from turtleFSI.modules import common
import os
import re
from postprocessing_common import read_command_line

#from simulations.stenosis_FC_sd2.pulsatile_vessel import set_problem_parameters
# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. Doesnt affect the speed
parameters["reorder_dofs_serial"] = False

def read_material_properties(case_path):
    '''
    This function loads in material properties from a logfile from the simulation, or from a text file called "material_properties.txt"
    If using the text file option, each material (fluid and solid) region must be listed in the text file, otherwise stress will not be computed for that region
    Each line should be formatted like this:
    {'dx_s_id': 1001, 'material_model': 'StVenantKirchoff', 'rho_s': 1000.0, 'mu_s': 344827.5862068966, 'lambda_s': 3103448.2758620703}
    {'dx_f_id': 1, 'rho_f': 1000.0, 'mu_f': 0.0035}
    {'dx_f_id': 1, 'rho_f': 1000.0, 'mu_f': 0.0035}

    '''

    # find all logfiles in simulaation folder (file name must contain the word "logfile")
    outLog=[file for file in os.listdir(case_path) if 'logfile' in file]
    solid_properties=[]
    fluid_properties=[]
    n_outfiles = len(outLog)

    print_MPI("Found {} output log files".format(n_outfiles))
    if n_outfiles == 0:
        print_MPI("Found no output files - ensure the word 'logfile' is in the output text file name, searching for material_properties.txt")
        if os.path.exists(os.path.join(case_path, "material_properties.txt")):
            material_properties_file = os.path.join(case_path, "material_properties.txt")
            print_MPI("found material_properties.txt")
        else:
            print_MPI("failed to find material propertiesin case_path")
    else:
        print_MPI("Reading material properties from log file: {}".format(outLog[0]))
        material_properties_file = outLog[0]

    # Open log file
    outLogPath=os.path.join(case_path,material_properties_file)
    file1 = open(outLogPath, 'r') 
    Lines = file1.readlines() 

    # Open log file get compute time and simulation time from that logfile 
    for line in Lines: 
        if 'dx_s_id' in line:
            material_properties_dict = eval(line)
            solid_properties.append(material_properties_dict)
        elif 'dx_f_id' in line:
            material_properties_dict = eval(line)
            fluid_properties.append(material_properties_dict)
    
    fluid_properties = remove_duplicates(fluid_properties)
    solid_properties = remove_duplicates(solid_properties)


    return solid_properties, fluid_properties

def remove_duplicates(l):
    seen = set()
    new_l = []
    for d in l:
        t = tuple(d.items())
        if t not in seen:
            seen.add(t)
            new_l.append(d)
    
    return new_l

def print_MPI(print_string):
    if MPI.rank(MPI.comm_world) == 0:
        print(print_string)

def get_mesh_domain_and_boundaries(mesh_path):
    # Read mesh
    mesh = Mesh()
    hdf = HDF5File(MPI.comm_world, mesh_path.__str__(), "r")
    hdf.read(mesh, "/mesh", False)
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
    domains = MeshFunction("size_t", mesh, 3)
    hdf.read(domains, "/domains")

    return mesh, domains, boundaries

def project_solid(tensorForm, fxnSpace, dx_s):#,dx_s_id_list):
    #
    # This function projects a UFL tensor equation (tensorForm) using a tensor function space (fxnSpace)
    # on only the solid part of the mesh, given by the differential operator for the solid domain (dx_s)
    #
    # This is basically the same as the inner workings of the built-in "project()" function, but it
    # allows us to calculate on a specific domain rather than the whole mesh. For whatever reason, it's also 6x faster than
    # the built in project function...
    #

    v = TestFunction(fxnSpace) 
    u = TrialFunction(fxnSpace)
    tensorProjected=Function(fxnSpace) # output tensor-valued function
    #a=0
    #L=0
    #for solid_region in range(len(dx_s_id_list)):
    #    a+=inner(u,v)*dx_s[solid_region] # bilinear form
    #    L+=inner(tensorForm,v)*dx_s[solid_region] # linear form
    a=inner(u,v)*dx_s # bilinear form
    L=inner(tensorForm,v)*dx_s # linear form     
    # Alternate way that doesnt work on MPI (may be faster on PC)
    #quadDeg = 4 # Need to set quadrature degree for integration, otherwise defaults to many points and is very slow
    #solve(a==L, tensorProjected,form_compiler_parameters = {"quadrature_degree": quadDeg}) 
 
    '''
    From "Numerical Tours of Continuum Mechanics using FEniCS", the stresses can be computed using a LocalSolver 
    Since the stress function space is a DG space, element-wise projection is efficient
    '''
    solver = LocalSolver(a, L)
    solver.factorize()
    solver.solve_local_rhs(tensorProjected)

    return tensorProjected

def setup_stress_forms(tensorForm, fxnSpace, dx_s):
    #
    # This function sets up a UFL tensor equation (tensorForm) using a tensor function space (fxnSpace)
    # on only the solid part of the mesh, given by the differential operator for the solid domain (dx_s)
    #

    v = TestFunction(fxnSpace) 
    u = TrialFunction(fxnSpace)
    a=inner(u,v)*dx_s # bilinear form
    L=inner(tensorForm,v)*dx_s # linear form

    return a, L

def solve_stress_forms(a, L, fxnSpace):
    # Solves the stress form efficiently for a DG space
    tensorProjected=Function(fxnSpace) # output tensor-valued function
    solver = LocalSolver(a, L)
    solver.factorize()
    solver.solve_local_rhs(tensorProjected)

    return tensorProjected


def compute_stress(case_path, mesh_name, dt, stride, save_deg):

    """
    Loads displacement fields from completed FSI simulation,
    and computes and saves the following solid mechanical quantities:
    (1) True Stress
    (2) Infinitesimal Strain
    (3) Maximum Principal Stress (True)
    (4) Maximum Principal Strain (Infinitesimal)
    edit June 19th, 2023:  we now read material properties from a "logfile" in the simulation directory or from "material_properties.txt"
    This script can now compute stress for subdomains with different material properties and different material models (Mooney-Rivlin, for example)

    Args:
        case_path (Path): Path to results from simulation
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        dt (float): Actual ime step of simulation
        stride: reduce the output data frequency by this factor, relative to input data (v.h5/d.h5 in this script)
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)

    """


    displacement_domains = "all" # d.h5 now contains all solid AND fluid domains
    solid_properties, fluid_properties = read_material_properties(case_path) # we now read material properties from a logfile or from material_properties.txt

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
    d_out_path = (visualization_separate_domain_path / "disp_test.xdmf").__str__()

    mesh_name = mesh_name + ".h5"


    mesh_path = os.path.join(case_path, "mesh", mesh_name)
    mesh_path_solid = mesh_path.replace(".h5","_solid_only.h5")

    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg == 1:
        print_MPI("Warning, stress results are compromised by using save_deg = 1, especially using a coarse mesh. Recommend using save_deg = 2 instead for computing stress")
        if displacement_domains == "all": # for d.h5, we can choose between using the entire domain and just the solid domain
            mesh_path_viz = mesh_path
        else:
            mesh_path_viz = mesh_path_solid     
    else:
        if displacement_domains == "all": # for d.h5, we can choose between using the entire domain and just the solid domain
            mesh_path_viz = mesh_path.replace(".h5","_refined.h5")
        else:
            mesh_path_viz = mesh_path_solid.replace("_solid_only.h5","_refined_solid_only.h5")
    
    mesh_path = Path(mesh_path)
    mesh_path_viz = Path(mesh_path_viz)
        
    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)
    

    _, domains, _ = get_mesh_domain_and_boundaries(mesh_path)

    # Read refined mesh saved as HDF5 format
    mesh_viz = Mesh()
    with HDF5File(MPI.comm_world, mesh_path_viz.__str__(), "r") as mesh_file:
        mesh_file.read(mesh_viz, "mesh", False)

    print_MPI("Define function spaces and functions")

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

    dx = Measure("dx", subdomain_data=domains)

    #dx_s_id = [2,1]
    #solid_properties = [{'dx_s_id': 2, 'material_model': 'StVenantKirchoff', 'rho_s': 1000.0, 'mu_s': 344827.5862068966, 'lambda_s': 3103448.2758620703},
    #                    {'dx_s_id': 1, 'material_model': 'StVenantKirchoff', 'rho_s': 1000.0, 'mu_s': 103448200.7586206897, 'lambda_s': 931034400.82758621}]
    dx_s = {}
    dx_s_id_list = []
    for idx, solid_region in enumerate(solid_properties):
        dx_s_id = solid_region["dx_s_id"]
        dx_s[idx] = dx(dx_s_id, subdomain_data=domains) # Create dx_s for each solid region
        dx_s_id_list.append(dx_s_id)
        print_MPI(solid_region)

    dx_f = {}
    dx_f_id_list = []
    for idx, fluid_region in enumerate(fluid_properties):
        dx_f_id = fluid_region["dx_f_id"]
        dx_f[idx] = dx(dx_f_id, subdomain_data=domains) # Create dx_s for each solid region
        dx_f_id_list.append(dx_f_id)
        print_MPI(fluid_region)


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
    d_out_file = XDMFFile(MPI.comm_world, d_out_path)

    sig_file.parameters["rewrite_function_mesh"] = False
    ep_file.parameters["rewrite_function_mesh"] = False
    sig_P_file.parameters["rewrite_function_mesh"] = False
    ep_P_file.parameters["rewrite_function_mesh"] = False
    d_out_file.parameters["rewrite_function_mesh"] = False

    print_MPI("========== Start post processing ==========")

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
            print_MPI("========== Timestep: {} ==========".format(t))
            f.read(d_viz, vec_name)
        except Exception as error:
            print_MPI("An exception occurred:", error) # An exception occurred

            print_MPI("========== Finished reading solutions ==========")
            break        
        
        d_viz.rename("Displacement_test", "d_viz")

        d_out_file.write(d_viz, t)
        # Calculate d in P2 based on visualization refined P1
        d.vector()[:] = dv_trans*d_viz.vector()




        # Deformation Gradient and first Piola-Kirchoff stress (PK1)
        deformationF = common.F_(d) # calculate deformation gradient from displacement
        
        # Cauchy (True) Stress and Infinitesimal Strain (Only accurate for small strains, ask DB for True strain calculation...)
        epsilon = common.eps(d) # Form for Infinitesimal strain (need polar decomposition if we want to calculate logarithmic/Hencky strain)
        #ep = project_solid(epsilon,Tens,dx) # Calculate stress tensor (this projection method is 6x faster than the built in version)
        
        # ADD THIS BACK?
        #eigStrain11,eigStrain22,eigStrain33 = common.get_eig(epsilon)


        #ep_P = project_solid(eigStrain11,Scal,dx) # Calculate Principal stress tensor, on whole solid domain
        #ep = project(epsilon,Tens) # Calculate stress tensor
        
        a = 0 
        a_scal = 0
        L_sig = 0
        L_sig_P = 0
        L_ep = 0
        L_ep_P = 0


        v = TestFunction(Tens) 
        u = TrialFunction(Tens)
        v_scal = TestFunction(Scal) 
        u_scal = TestFunction(Scal) 
        for solid_region in range(len(dx_s_id_list)):

            PiolaKirchoff2 = common.S(d, solid_properties[solid_region]) # Form for second PK stress (using specified material model)
            sigma = (1/common.J_(d))*deformationF*PiolaKirchoff2*deformationF.T  # Form for Cauchy (true) stress 
            #eigStress11,eigStress22,eigStress33  = common.get_eig(sigma)  # Calculate principal stress
            a+=inner(u,v)*dx_s[solid_region] # bilinear form
            a_scal+=inner(u_scal,v_scal)*dx_s[solid_region] # bilinear form

            L_sig+=inner(sigma,v)*dx_s[solid_region] 
            #L_sig_P+=inner(eigStress11,v_scal)*dx_s[solid_region] 
            L_ep+=inner(epsilon,v)*dx_s[solid_region] 
            #L_ep_P+=inner(eigStrain11,v_scal)*dx_s[solid_region] 

        for fluid_region in range(len(dx_f_id_list)):

            nought_value = 1e-10
            sigma_nought = as_tensor([[nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value]]) # Add placeholder value to fluid region
            epsilon_nought = as_tensor([[nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value],
                               [nought_value ,nought_value ,nought_value]]) # Add placeholder value to fluid region            #eigStress11,eigStress22,eigStress33  = common.get_eig(sigma)  # Calculate principal stress
            a+=inner(u,v)*dx_f[fluid_region] # bilinear form
            a_scal+=inner(u_scal,v_scal)*dx_f[fluid_region] # bilinear form
            L_sig+=inner(sigma_nought,v)*dx_f[fluid_region] 
            L_ep+=inner(epsilon_nought,v)*dx_f[fluid_region] 

        sig = solve_stress_forms(a,L_sig,Tens) # Calculate stress tensor 
        #sig_P = solve_stress_forms(a_scal,L_sig_P,Scal) # Calculate stress tensor 
        ep = solve_stress_forms(a,L_ep,Tens) # Calculate stress tensor 
        #ep_P = solve_stress_forms(a_scal,L_ep_P,Scal) # Calculate stress tensor 
        eigStrain11,eigStrain22,eigStrain33 = common.get_eig(ep)
        eigStress11,eigStress22,eigStress33  = common.get_eig(sig)  # Calculate principal stress
        ep_P=project_solid(eigStrain11,Scal,dx)
        sig_P=project_solid(eigStress11,Scal,dx)

        #S_ = stress_strain.S(d, lambda_s, mu_s)  # Form for second PK stress (using St. Venant Kirchoff Model)
        #sigma = (1/stress_strain.J_(d))*deformationF*S_*deformationF.T  # Form for Cauchy (true) stress 


        #sig =project(sigma,Tens) # Calculate stress tensor

        # Calculate eigenvalues of the stress tensor (Three eigenvalues for 3x3 tensor)
        # Eigenvalues are returned as a diagonal tensor, with the Maximum Principal stress as 1-1
        #eigStress11,eigStress22,eigStress33  = stress_strain.get_eig(sigma) 

        #sig_P = project_solid(eigStress11,Scal,dx) # Calculate Principal stress tensor
        #sig_P+=project_solid(eigStress11,Scal,dx_s[1]) # Calculate Principal stress tensor

        #sig_P = project(eigStress,Tens) # Calculate Principal stress tensor

        #ep_P = project(eigStrain,Tens) # Calculate Principal stress tensor
    
        # Name function
        ep.rename("InfinitesimalStrain", "ep")
        sig.rename("TrueStress", "sig")
        ep_P.rename("MaximumPrincipalStrain", "ep_P")
        sig_P.rename("MaximumPrincipalStress", "sig_P")

        print_MPI("Writing Additional Viz Files for Stresses and Strains!")

        # Write results
        ep_file.write(ep, t)
        sig_file.write(sig, t)
        ep_P_file.write(ep_P, t)
        sig_P_file.write(sig_P, t)

        # Update file_counter
        file_counter += stride


if __name__ == '__main__':
    folder, mesh, _, _, _, dt, stride, save_deg, _, _ = read_command_line()
    compute_stress(folder,mesh, dt, stride, save_deg)
