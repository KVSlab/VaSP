import common_meshing
import numpy as np
from dolfin import *  # Import order is important here for some reason... import dolfin here last

# This script creates a solid-only mesh in h5 format from a specified mesh. Currently, it only runs in serial (not parallel)
# due to the "SubMesh" function used in fenics. This mesh is later used in  the "postprocessing_fenics" scripts
#
#    TO DO:
#    -Add boundary creation in this and other meshing scripts
#    -Add domain creation in all meshing scripts


folder, mesh_name = common_meshing.read_command_line()
mesh_path = folder + "/mesh/" + mesh_name +".h5"

# Read in original FSI mesh
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
hdf.read(mesh, "/mesh", False)

domains = MeshFunction("size_t", mesh, 3)
hdf.read(domains, "/domains")

# Extract solid part of Mesh
mesh_solid = SubMesh(mesh,domains,2)

# Create path for refined mesh
solid_mesh_path = mesh_path.replace(".h5","_solid_only.h5")

# Save refined mesh
hdf = HDF5File(mesh_solid.mpi_comm(), solid_mesh_path, "w")
hdf.write(mesh_solid, "/mesh")
#hdf.write(boundaries, "/boundaries")

print("Solid mesh saved to:")
print(solid_mesh_path)    

hdf.close()

# This created mesh may have different node numbering than the original mesh. This next line fixes the node numbering
#  so that it starts at 0 and matches the separate domain "displacement.h5" file we create later. 
common_meshing.fix_solid_only_mesh(mesh_path)

print("Fixed solid-only mesh wih correct node numbering. Mesh saved to:")
print(solid_mesh_path)    

### Read fixed mesh
#mesh_solid_fixed = Mesh()
#hdf = HDF5File(mesh_solid_fixed.mpi_comm(), solid_mesh_path, "r")
#hdf.read(mesh_solid_fixed, "/mesh", False)
### Save mesh to pvd file for viewing in paraview
#ff = File(mesh_path.replace(".h5","solid_mesh_fixed.pvd"))
#ff << mesh_solid_fixed
