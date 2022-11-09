"""
Convert Owais 2015 CFD mesh to stl for FSI meshing. Need to unzip xml.gz first with gzip
"""
import pyvista as pv
from vmtk import vtkvmtk, vmtkscripts
# from common import ReadPolyData, WritePolyData, surface_cleaner, triangulate_surface
import vtk 
from os import path

###############################################################################
file_name = "offset_stenosis"
ifile_mesh = "meshes/"+file_name+".xml"


###############################################################################


dolfin = pv.read(ifile_mesh)
#viewer = vmtkscripts.vmtkMeshViewer()
#viewer.Mesh = dolfin
#viewer.Self

m2s = vmtkscripts.vmtkMeshToSurface()
m2s.Mesh = dolfin
m2s.SurfaceOutputFileName = 'looooooo'
m2s.Execute()
surf_an = m2s.Surface

writ = vmtkscripts.vmtkSurfaceWriter()
writ.Surface=surf_an
writ.Format='stl'
writ.OutputFileName = "surfaces/"+file_name+".stl"
writ.Execute()
