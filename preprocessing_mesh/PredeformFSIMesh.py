import common_meshing
import h5py
from shutil import copyfile
from numpy import genfromtxt
import numpy as np
import common_meshing

# User inputs ----------------------------
scaleFactor=-1.0  # This is the factor used to scale the deformation
folder, mesh_name = common_meshing.read_command_line()

# -----------------------------------
disp_filepath = folder +"/"+ mesh_name +'/1/Visualization/displacement.h5'
mesh_path = folder + '/mesh/' + mesh_name +".h5"
# -----------------------------------------

dt = 0.00033964285714285700
t = 0.28

index_t = int(np.round(t/dt))
print(disp_filepath)

vectorData = h5py.File(disp_filepath, "r") 
ArrayName = 'VisualisationVector/' + str(index_t)	
deformation = vectorData[ArrayName][:,:] # Important not to take slices of this array, slows code considerably... 

predeformed_mesh_path=mesh_path.replace(".h5", "_predeformed.h5")
copyfile(mesh_path, predeformed_mesh_path)

#f = HDF5File(mpi_comm_world(),'meshes/'+mesh_name, 'r')
#mesh_file = Mesh()
#f.read(mesh, 'mesh')

vectorData = h5py.File(predeformed_mesh_path,'a')

ArrayNames = ['boundaries/coordinates','mesh/coordinates','domains/coordinates']


#hdf5_store = h5py.File("meshes/deformed_"+meshname+'.h5', "w")

for ArrayName in ArrayNames:

	vectorArray = vectorData[ArrayName]
	modified = vectorData[ArrayName][:,:] + deformation*scaleFactor
	vectorArray[...] = modified

vectorData.close() 