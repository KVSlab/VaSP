import common_meshing
import numpy as np
from dolfin import * # Import order is important here for some reason... import dolfin here last


folder, mesh_name = common_meshing.read_command_line()
mesh_path = folder + "/mesh/" + mesh_name +".h5"

# Read in original FSI mesh
mesh = Mesh()
hdf = HDF5File(mesh.mpi_comm(), mesh_path, "r")
hdf.read(mesh, "/mesh", False)

domains = MeshFunction("size_t", mesh, 3)
hdf.read(domains, "/domains")

boundaries = MeshFunction("size_t", mesh, 2)
hdf.read(boundaries, "/boundaries")

ff = File(mesh_path.replace(".h5","_boundaries.pvd"))
ff << boundaries
## Save mesh to pvd file for viewing in paraview
ff = File(mesh_path.replace(".h5","_domains.pvd"))
ff << domains
