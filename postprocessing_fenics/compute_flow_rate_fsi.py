import numpy as np
import h5py
#from dolfin import *
from dolfin import PETScDMCollection, HDF5File, Mesh, FunctionSpace, VectorElement, ds, assemble, inner, Constant, parameters, MPI, MeshFunction, Function, FacetNormal
import os
from pathlib import Path

from postprocessing_common import read_command_line, get_time_between_files

# set compiler arguments
parameters["form_compiler"]["quadrature_degree"] = 6 # Not investigated thorougly. See MSc theses of Gjertsen. 
parameters["reorder_dofs_serial"] = False

def compute_wss(case_path,mesh_name, dt, stride, save_deg):

    """
    Loads velocity fields from completed CFD simulation,
    and computes flow rate. 

    Args:
        mesh_name: Name of the input mesh for the simulation. This function will find the refined and solid only mesh based on this name
        case_path (Path): Path to results from simulation
        nu (float): Viscosity
        dt (float): Actual ime step of simulation
        save_step: desired save frequency for output stress
        save_deg (int): element degree saved from P2-P1 simulation (save_deg = 1 is corner nodes only)

    This code is "finnicky", it seems that sometimes compiler issues can arise (maybe because of PetscDMCollection?) and the flow rates or areas
    will be equal to zero, or the code will not compile. Running dijitso clean on Niagara can fix this issue. The separate inlet and outlet 
    differentials, (dsi, dso#) seem ineffective but I was unable to make the list form for this work. 
    Something like:    ds = Measure("ds", subdomain_data=boundaries) should work, but doesnt want to compile (see _SS/Flow_rate_compact.py)

    EDIT: Removed PetSCDMCollection and simply using P1 refined mesh to calculate the flow rates. This seems more reliable but may not be super accurate. 


    """
    # File paths

    for file in os.listdir(case_path):
        file_path = os.path.join(case_path, file)
        if os.path.exists(os.path.join(file_path, "1")):
            visualization_path = os.path.join(file_path, "1/Visualization")
        elif os.path.exists(os.path.join(file_path, "Visualization")):
            visualization_path = os.path.join(file_path, "Visualization")
    
    visualization_path = Path(visualization_path)
    file_path_u = visualization_path / "v.h5"
    file_path_v = visualization_path / "velocity.h5"
    file_path_flow_rate = visualization_path / "flow_rates.txt"
    xdmf_file = visualization_path / "velocity.xdmf"


    # get fluid-only version of the mesh
    mesh_name = mesh_name + ".h5"
    mesh_path = os.path.join(case_path, "mesh", mesh_name)

    # if save_deg = 1, make the refined mesh path the same (Call this mesh_viz)
    if save_deg == 1:
        mesh_path = mesh_path
    else:
        mesh_path = mesh_path.replace(".h5","_refined.h5")

    mesh_path = Path(mesh_path)

    # Read mesh saved as HDF5 format
    mesh = Mesh()
    with HDF5File(MPI.comm_world, mesh_path.__str__(), "r") as mesh_file:
        mesh_file.read(mesh, "mesh", False)
        boundaries = MeshFunction("size_t", mesh, 2)
        mesh_file.read(boundaries, "/boundaries")
    


    # Create higher-order function space for d, v and p
    dve = VectorElement('CG', mesh.ufl_cell(), 1)
    FSdv = FunctionSpace(mesh, dve)   # Higher degree FunctionSpace for d and v

    if MPI.rank(MPI.comm_world) == 0:
        print("Define functions")

    # Create higher-order function on unrefined mesh for post-processing calculations
    u = Function(FSdv)
    

    if MPI.rank(MPI.comm_world) == 0:
        print("=" * 10, "Start post processing", "=" * 10)

    # Start file counter
    file_counter = 0 
    t_0, time_between_files = get_time_between_files(xdmf_file)
    save_step = round(time_between_files/dt) # This is the output frequency of the simulation
    t = t_0 # Initial time of simulation

    flow_rate_output = []

    # Inlet/outlet differential
    dsi = ds(2, domain=mesh, subdomain_data=boundaries)
    dso3 = ds(3, domain=mesh, subdomain_data=boundaries)
    n = FacetNormal(mesh)

    area_inlet = assemble(Constant(1.0, name="one") * dsi) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
    area_outlet3 = assemble(Constant(1.0, name="one") * dso3) # Get error: ufl.log.UFLException: This integral is missing an integration domain.

    print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(2,area_inlet))
    flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(2,area_inlet))
    print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(3,area_outlet3))
    flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(3,area_outlet3))


    # Get range of inlet and outlet ids in model
    inlet_outlet_min = 2 # lower bound for inlet and outlet IDs (inlet is usually 2)
    inlet_outlet_max = 9 # upper bound for inlet and outlet IDs
    bd_ids = np.unique(boundaries.array()[:])
    inlet_outlet_ids = bd_ids[(bd_ids >= inlet_outlet_min) & (bd_ids <=inlet_outlet_max)]

    if len(inlet_outlet_ids) > 2:
        dso4 = ds(4, domain=mesh, subdomain_data=boundaries)
        area_outlet4 = assemble(Constant(1.0, name="one") * dso4) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
        print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(4,area_outlet4))
        flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(4,area_outlet4))

    if len(inlet_outlet_ids) > 3:
        dso5 = ds(5, domain=mesh, subdomain_data=boundaries)
        area_outlet5 = assemble(Constant(1.0, name="one") * dso5) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
        print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(5,area_outlet5))
        flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(5,area_outlet5))   

    if len(inlet_outlet_ids) > 4:
        dso6 = ds(6, domain=mesh, subdomain_data=boundaries)
        area_outlet6 = assemble(Constant(1.0, name="one") * dso6) # Get error: ufl.log.UFLException: This integral is missing an integration domain.
        print("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(6,area_outlet6))
        flow_rate_output.append("Inlet/Outlet area for ID# {} is: {:e} m^2\n".format(6,area_outlet6))      

    while True:

        
        if MPI.rank(MPI.comm_world) == 0:
            print("=" * 10, "Timestep: {}".format(t), "=" * 10)
        try:
            # Read in solution to vector function 
            vel_file = h5py.File(file_path_v, 'r')
            vec_name = "VisualisationVector/" + str(file_counter)
            vector_np = vel_file.get(vec_name)[()]
            vector_np_flat = vector_np.flatten('F')
            u.vector().set_local(vector_np_flat)  # Set u vector
        except:
            if MPI.rank(MPI.comm_world) == 0:
                print("=" * 10, "Finished reading solutions", "=" * 10)
            break   

        # Compute flow rate(s)
        flow_rate_output.append("============ Timestep: {} =============\n".format(t))
        flow_rate_inlet = assemble(inner(u, n)*dsi)
        flow_rate_outlet3 = assemble(inner(u, n)*dso3)
           
        flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(2,flow_rate_inlet))
        flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(3,flow_rate_outlet3))

        if MPI.rank(MPI.comm_world) == 0:
            print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(2,flow_rate_inlet))
            print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(3,flow_rate_outlet3))

        if 4 in inlet_outlet_ids:
            flow_rate_outlet4 = assemble(inner(u, n)*dso4)
            flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(4,flow_rate_outlet4))
            if MPI.rank(MPI.comm_world) == 0:
                print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(4,flow_rate_outlet4))

        if 5 in inlet_outlet_ids:
            flow_rate_outlet5 = assemble(inner(u, n)*dso5)
            flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(5,flow_rate_outlet5))
            if MPI.rank(MPI.comm_world) == 0:
                print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(5,flow_rate_outlet5))

        if 6 in inlet_outlet_ids:
            flow_rate_outlet6 = assemble(inner(u, n)*dso6)
            flow_rate_output.append("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(6,flow_rate_outlet6))
            if MPI.rank(MPI.comm_world) == 0:
                print("Inlet/Outlet flow rate for ID# {} is: {:e} m^3/s\n".format(6,flow_rate_outlet6))

        # Update file_counter
        file_counter += stride
        t += time_between_files*stride

    # Write flow rates to file
    flow_rate_file = open(file_path_flow_rate,"w")
    flow_rate_file.writelines(flow_rate_output)
    flow_rate_file.close()

if __name__ == '__main__':
    folder, mesh, nu, dt, stride, save_deg = read_command_line()
    compute_wss(folder, mesh, dt, stride, save_deg)
