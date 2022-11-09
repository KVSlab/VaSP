from dolfin import *




mesh_name = "offset_stenosis"
mesh_file = Mesh("meshes/"+mesh_name+"_fsi.xml")

# Rescale the mesh coordinated from [mm] to [m]
x = mesh_file.coordinates()
scaling_factor = 0.001  # from mm to m
x[:, :] *= scaling_factor
mesh_file.bounding_box_tree().build(mesh_file)

# Convert subdomains to mesh function
boundaries = MeshFunction("size_t", mesh_file, 2, mesh_file.domains())

boundaries.set_values(boundaries.array()+1)  # FIX ME, THIS IS NOT NORMAL!

ff = File("meshes/"+mesh_name+"_boundaries.pvd")
ff << boundaries

domains = MeshFunction("size_t", mesh_file, 3, mesh_file.domains())
domains.set_values(domains.array()+1)  # in order to have fluid==1 and solid==2

ff = File("meshes/"+mesh_name+"_domains.pvd")
ff << domains  

hdf = HDF5File(mesh_file.mpi_comm(), "meshes/file_"+mesh_name+".h5", "w")
hdf.write(mesh_file, "/mesh")
hdf.write(boundaries, "/boundaries")
hdf.write(domains, "/domains")

print("PASSED SERIAL MESH TREATMENT, SAVED TO:")
print("../meshes/file_"+mesh_name+".h5")
