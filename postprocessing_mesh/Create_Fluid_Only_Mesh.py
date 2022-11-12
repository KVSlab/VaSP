import common_meshing
import numpy as np
import os
from dolfin import * # Import order is important here for some reason... import dolfin here last

# This script creates a fluid-only mesh in h5 format from a specified mesh. Currently, it only runs in serial (not parallel)
# due to the "SubMesh" function used in fenics. This mesh is later used in  the "postprocessing_fenics" scripts
# If the input mesh is not refined (save_deg = 2 mesh) the script also creates 
# "boundaries" in the h5 file, which are used for calculating flow rates. 
#
#    TO DO:
#    -Add boundary creation in other meshing scripts
#    -Add domain creation in all meshing scripts


# These numbers are hard-coded, should be ok unless we have more than 7 inlets and outlets
wall_id = 22 # Wall id in FSI mesh
inlet_outlet_min = 2 # lower bound for inlet and outlet IDs (inlet is usually 2)
inlet_outlet_max = 9 # upper bound for inlet and outlet IDs

folder, mesh_name = common_meshing.read_command_line()
print(folder)
mesh_path = os.path.join(folder,"mesh",mesh_name +".h5")
print(mesh_path)

# Read in original FSI mesh
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
hdf.read(mesh, "/mesh", False)

domains = MeshFunction("size_t", mesh, 3)
hdf.read(domains, "/domains")
'''
# Read in boundaries, only if creating boundaries for output mesh
if "refined" not in mesh_path:
    boundaries = MeshFunction("size_t", mesh, 2)
    hdf.read(boundaries, "/boundaries")
'''
# Extract fluid part of Mesh
mesh_fluid = SubMesh(mesh,domains,1)
	
# Create path for fluid mesh
fluid_mesh_path = mesh_path.replace(".h5","_fluid_only.h5")

# Save refined mesh
hdf = HDF5File(mesh_fluid.mpi_comm(), fluid_mesh_path, "w")
hdf.write(mesh_fluid, "/mesh")

print("fluid-only mesh saved to:")
print(fluid_mesh_path)    

hdf.close()

# This created mesh may have different node numbering than the original mesh. This next line fixes the node numbering
#  so that it starts at 0 and matches the separate domain "velocity.h5" file we create later. 
common_meshing.fix_fluid_only_mesh(mesh_path)

print("Fixed fluid-only mesh wih correct node numbering. Mesh saved to:")
print(fluid_mesh_path)    
'''
# For non-refined mesh, we want to get the boundaries for performing flow rate calculations. 
if "refined" not in mesh_path:
    ## Read fixed mesh
    mesh_fluid_fixed = Mesh()
    hdf = HDF5File(mesh_fluid_fixed.mpi_comm(), fluid_mesh_path, "a")
    hdf.read(mesh_fluid_fixed, "/mesh", False)
    boundaries_fluid = MeshFunction("size_t", mesh_fluid_fixed, 2)

    # Mark outside of fluid mesh
    class Mesh_Exterior(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary

    mesh_exterior = Mesh_Exterior()
    mesh_exterior.mark(boundaries_fluid, wall_id) # Mark the entire outside of the mesh  with the wall ID

    # Iterate over the inlets and outlets of the fsi mesh and the exterior of the fluid mesh to mark te inlets and outlets of the fluid mesh
    i = 0
    tol = 1e-8
    for mesh_facet in facets(mesh): # Loop through facets of original FSI mesh
        idx_facet = boundaries.array()[i]
        if (idx_facet >= inlet_outlet_min and idx_facet <= inlet_outlet_max): # If the facet is on the inlet, or one of the outlets
            #vert = mesh_facet.entities(0) # returns array of points
            mid_fsi = mesh_facet.midpoint()
            j = 0
            for fluid_mesh_facet in facets(mesh_fluid_fixed): # Loop through facets of fluid only mesh
                if boundaries_fluid.array()[j] == wall_id: # If the facet is on the exterior of the mesh (wall ID)
                    mid_fluid = fluid_mesh_facet.midpoint()
                    dist = sqrt((mid_fsi.x() - mid_fluid.x() )**2 + (mid_fsi.y() - mid_fluid.y())**2 + (mid_fsi.z() - mid_fluid.z() )**2)
                    if dist < tol: # If the facet of the Fluid Only mesh has the same midpoint as the original FSI mesh
                        boundaries_fluid.array()[j] = idx_facet # Assign the inlet/outlet ID to Fluid Only mesh
                        print(j)
                        break
                j+=1
        i += 1
        
    hdf.write(boundaries_fluid, "/boundaries")
    
#    ## Save mesh to pvd file for viewing in paraview
#    ff = File(mesh_path.replace(".h5","_fluid_boundaries.pvd"))
#    ff << boundaries_fluid
'''