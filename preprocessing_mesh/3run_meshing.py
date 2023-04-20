# !/usr/bin/python

# Local imports
from vmtkmeshgeneratorfsi import *
from vmtkthreshold import *
from vmtkentityrenumber import *

from os import path 
import vmtk #import vmtkscripts
import vtk
import numpy as np


#################
# USER INPUTS
# ################################################################################
# file_name = "stenosis"
# ifile_surface = "surfaces/"+file_name+".vtp"
# #ifile_surface = "surfaces/"+file_name+"_clipped.stl"
# # FIXME: meshes/ does not exit and we need to create a folder before running this script
# ofile_mesh = "meshes/"+file_name
# TargetEdgeLength_f = 1.8  # more or less minimum edge length of the fluid mesh (These vary for each case, of course)
# TargetEdgeLength_s = 1.950  # more or less minimum edge length of the solid mesh
# Thick_solid = 0.3  # constant tickness of the solid wall
# nb_boundarylayers = 2  # number of sub-boundary layers is the solid and fluid mesh
# BoundaryLayerThicknessFactor = Thick_solid / TargetEdgeLength_f  # Wall Thickness == TargetEdgeLength*BoundaryLayerThicknessFactor
# # Refinement parameters
# seedX = (0.0, 0.0, 0.0)  # location of the seed to refine from (distance based)
# factor_scale = 5  # multiplier for max element size (orig 2)
# factor_shape = 1.0  # 1==linear scale based on distance (orig 0.1)
# rem_it = 50
# iterations = 50
clip_surface=False

#factor_scale = 2  # multiplier for max element size (orig 2)
#factor_shape = 1.0  # 1==linear scale based on distance (orig 0.1)
#rem_it = 5
#iterations = 5
# ################################################################################
# END OF USER INPUTS
########################


##################
## USER INPUTS
## ################################################################################
file_name = "dab_mesh" 
folder_name = "case9_300k/"
ifile_surface = "surfaces/"+ folder_name + file_name+".stl"
ofile_mesh = "surfaces/" + folder_name +file_name

TargetEdgeLength_f = 0.260  # more or less minimum edge length of the fluid mesh .410 is good too
TargetEdgeLength_s = 0.280 # more or less minimum edge length of the solid mesh .430 is good too
Thick_solid = 0.25  # constant tickness of the solid wall
nb_boundarylayers = 2  # number of sub-boundary layers is the solid and fluid mesh
BoundaryLayerThicknessFactor = Thick_solid / TargetEdgeLength_f  # Wall Thickness == TargetEdgeLength*BoundaryLayerThicknessFactor
# Refinement parameters
seedX = (123.099, 134.62, 64.087)  # location of the seed to refine from (distance based)
factor_scale = 5  # multiplier for max element size (orig 2)
factor_shape = 1.0  # 1==linear scale based on distance (orig 0.1)
rem_it = 200
iterations = 200


# Read vtp surface file (vtp) ##################################################
reader = vmtkscripts.vmtkSurfaceReader()
reader.InputFileName = ifile_surface
reader.Execute()
surface = reader.Surface
################################################################################
if clip_surface:
    clipp = vmtkscripts.vmtkSurfaceClipper()
    clipp.Surface = surface
    clipp.Execute()
    surface = clipp.Surface
    writ = vmtkscripts.vmtkSurfaceWriter()
    writ.Surface=surface
    writ.Format='stl'
    writ.OutputFileName = "surfaces/"+file_name+"_clipped.stl"
    writ.Execute()

N = surface.GetNumberOfPoints()
dist_array = np.zeros(N)
# Compute distance
for i in range(N):
    piX = surface.GetPoints().GetPoint(i)
    dist_array[i] = np.sqrt(np.sum((np.asarray(seedX) - np.asarray(piX))**2))
dist_array[:] = dist_array[:] - dist_array.min()  # between 0 and max
dist_array[:] = dist_array[:] / dist_array.max() + 1  # between 1 and 2
dist_array[:] = dist_array[:]**factor_shape - 1  # between 0 and 2^factor_shape
dist_array[:] = dist_array[:] / dist_array.max()  # between 0 and 1
dist_array[:] = dist_array[:]*(factor_scale-1) + 1  # between 1 and factor_scale
dist_array[:] = dist_array[:] * TargetEdgeLength_s  # Scaled TargetEdgeLength
array = vtk.vtkDoubleArray()
array.SetNumberOfComponents(1)
array.SetNumberOfTuples(N)
array.SetName("Size")
for i in range(N):
    array.SetTuple1(i, dist_array[i])
surface.GetPointData().AddArray(array)

method = "taubin"
smoother = vmtkscripts.vmtkSurfaceSmoothing()
smoother.Surface = surface
smoother.NumberOfIterations = iterations
smoother.Method = method
smoother.PassBand = 0.5
smoother.Execute()
surface = smoother.Surface

remesher = vmtkscripts.vmtkSurfaceRemeshing()
remesher.Surface = surface
remesher.Iterations = rem_it
remesher.ElementSizeMode = 'edgelength'
remesher.TargetEdgeLength = TargetEdgeLength_f
remesher.Execute()
surface = remesher.Surface

# writer = vtk.vtkXMLPolyDataWriter()  # vtp files
# writer.SetFileName('toto_surf.vtp')
# writer.SetInputData(surface)
# writer.Update()
# writer.Write()

# Create FSI mesh ##############################################################
print("--- Creating fsi mesh")
meshGenerator = vmtkMeshGeneratorFsi()
meshGenerator.Surface = surface
# for remeshing
#meshGenerator.SkipRemeshing = 1
meshGenerator.ElementSizeMode = 'edgelength'
meshGenerator.TargetEdgeLength = TargetEdgeLength_f
meshGenerator.MaxEdgeLength = 20*meshGenerator.TargetEdgeLength
meshGenerator.MinEdgeLength = 0.4*meshGenerator.TargetEdgeLength
# for boundary layer (used for both fluid boundary layer and solid domain)
meshGenerator.BoundaryLayer = 1
meshGenerator.NumberOfSubLayers = nb_boundarylayers
meshGenerator.BoundaryLayerOnCaps = 0
meshGenerator.SubLayerRatio = 0.75
meshGenerator.BoundaryLayerThicknessFactor = BoundaryLayerThicknessFactor
# mesh
meshGenerator.Tetrahedralize = 1
# Cells and walls numbering
meshGenerator.SolidSideWallId = 11
meshGenerator.InterfaceId_fsi = 22
meshGenerator.InterfaceId_outer = 33
meshGenerator.VolumeId_fluid = 0  # (keep to 0)
meshGenerator.VolumeId_solid = 1
meshGenerator.Execute()
mesh = meshGenerator.Mesh
################################################################################

# Write mesh in VTU format #####################################################
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(path.join(ofile_mesh + "_fsi.vtu"))
writer.SetInputData(mesh)
writer.Update()
writer.Write()
################################################################################

# Write mesh to FEniCS to format ###############################################
meshWriter = vmtkscripts.vmtkMeshWriter()
meshWriter.CellEntityIdsArrayName = "CellEntityIds"
meshWriter.Mesh = mesh
meshWriter.OutputFileName = path.join(ofile_mesh + "_fsi.xml")
meshWriter.WriteRegionMarkers = 1
meshWriter.Compressed = 0
meshWriter.Execute()
################################################################################
# Remove outer layers to make cfd mesh

################################################################################
renumber = VmtkEntityRenumber()
renumber.Mesh = mesh
renumber.CellEntityIdsArrayName = "CellEntityIds"
renumber.CellEntityIdOld=1
renumber.CellEntityIdNew=1001
renumber.Execute()

renumber2 = VmtkEntityRenumber()
renumber2.Mesh = mesh
renumber2.CellEntityIdsArrayName = "CellEntityIds"
renumber2.CellEntityIdOld=22
renumber2.CellEntityIdNew=1
renumber2.Execute()

thresh = vmtkThreshold()
thresh.Mesh = mesh
thresh.CellEntityIdsArrayName = "CellEntityIds"
thresh.HighThreshold = 6
thresh.Execute()


# Write mesh to FEniCS to format ###############################################
meshWriter = vmtkscripts.vmtkMeshWriter()
meshWriter.CellEntityIdsArrayName = "CellEntityIds"
meshWriter.Mesh = thresh.Mesh 
meshWriter.OutputFileName = path.join(ofile_mesh + "_cfd.xml")
meshWriter.WriteRegionMarkers = 1
meshWriter.Compressed = 0
meshWriter.Execute()
################################################################################

# Write mesh in VTU format #####################################################
writer = vtk.vtkXMLUnstructuredGridWriter()
writer.SetFileName(path.join(ofile_mesh + "_cfd.vtu"))
writer.SetInputData(thresh.Mesh)
writer.Update()
writer.Write()
################################################################################

#viewer = vmtkscripts.vmtkMeshViewer()
#viewer.Mesh = thresh.Mesh 
#viewer.Execute()
