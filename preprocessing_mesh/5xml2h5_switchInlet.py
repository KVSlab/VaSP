from dolfin import *


#mesh_name = 'case8_el042'
#inlet_id_actual = 4
mesh_name = 'stenosis'
inlet_id_actual = 3
#mesh_name = 'case16_el06'
#inlet_id_actual = 4
#mesh_name = 'case16_el06_1Layer'
#inlet_id_actual = 4
#mesh_name = 'case16_el05'
#inlet_id_actual = 3
#mesh_names = ['case3','case11','case12']
mesh_names = ['stenosis']

#mesh_names = ['case8_el042','case9_el047','case16_el06']
#inlet_ids = [3,2,4]

inlet_ids = [2]
for idx in range(len(mesh_names)):

    mesh_name = mesh_names[idx]
    inlet_id_actual = inlet_ids[idx]
    mesh_file = Mesh("meshes/"+mesh_name+"_fsi.xml")
    
    # Rescale the mesh coordinated from [mm] to [m]
    x = mesh_file.coordinates()
    scaling_factor = 0.001  # from mm to m
    x[:, :] *= scaling_factor
    mesh_file.bounding_box_tree().build(mesh_file)
    
    # Convert subdomains to mesh function
    boundaries = MeshFunction("size_t", mesh_file, 2, mesh_file.domains())
    
    boundaries.set_values(boundaries.array()+1)  # FIX ME, THIS IS NOT NORMAL!
    #print(boundaries.array())
    for i in range(0,len(boundaries.array())):
        if boundaries.array()[i]==1:
            boundaries.array()[i] = 0
        elif boundaries.array()[i]==2:
            boundaries.array()[i] = inlet_id_actual
        elif boundaries.array()[i]==inlet_id_actual:
            boundaries.array()[i] = 2 # we want the inlet to be 2s
    
    # ---------------------
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
    