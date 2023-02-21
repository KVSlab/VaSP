import argparse
from pathlib import Path
from dolfin import Mesh, MeshFunction, File, HDF5File

def commandLine():
    """
    args:
        None
    return:
        None
    """
    parser = argparse.ArgumentParser(description='Switch inlet in the mesh file')
    parser.add_argument('-folder_path', type=Path, help='Path to the folder containing the mesh file')
    parser.add_argument('-mesh_name', type=str, help='name of the mesh file (without extension)')
    parser.add_argument('-inlet_id_actual', type=int, help='id of the inlet in the mesh file')
    args = parser.parse_args()
    return args

def switchInlet(folder_path, mesh_name, inlet_id_actual):
    """
    args:
        folder_path: Path to the folder containing the mesh file
        mesh_name: name of the mesh file (without extension)
        inlet_id_actual: id of the inlet in the mesh file
    return:
        None (save the new mesh file)
    """
    mesh_name_xml = mesh_name+".xml"
    mesh = Mesh(str(folder_path / mesh_name_xml))
    # Convert subdomains to mesh function
    boundaries = MeshFunction("size_t", mesh, 2, mesh.domains())
    
    boundaries.set_values(boundaries.array()+1)  # FIX ME, THIS IS NOT NORMAL!
    
    for i in range(0,len(boundaries.array())):
        if boundaries.array()[i]==1:
            boundaries.array()[i] = 0
        elif boundaries.array()[i]==2:
            boundaries.array()[i] = inlet_id_actual
        elif boundaries.array()[i]==inlet_id_actual:
            boundaries.array()[i] = 2 # we want the inlet to be 2s
    
    boundary_file = mesh_name+"_boundaries.pvd"
    ff = File(str(folder_path / boundary_file))
    ff << boundaries
    domain_file = mesh_name+"_domains.pvd"
    domains = MeshFunction("size_t", mesh, 3, mesh.domains())
    domains.set_values(domains.array()+1)  # in order to have fluid==1 and solid==2

    ff = File(str(folder_path / domain_file))
    ff << domains   
    hd5_file = mesh_name+".h5"
    hdf = HDF5File(mesh.mpi_comm(), str(folder_path / hd5_file), "w")
    hdf.write(mesh, "/mesh")
    hdf.write(boundaries, "/boundaries")
    hdf.write(domains, "/domains")
    
    print("PASSED SERIAL MESH TREATMENT")
    
    return None

if __name__ == "__main__":
    args = commandLine()
    switchInlet(args.folder_path, args.mesh_name, args.inlet_id_actual)