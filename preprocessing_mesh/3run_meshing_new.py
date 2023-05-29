# !/usr/bin/python

# Local imports
from vmtkmeshgeneratorfsi import *
from vmtkthreshold import *
from vmtkentityrenumber import *

from os import path, listdir, remove
from pathlib import Path
import time
import vmtk #import vmtkscripts
import vtk
import numpy as np
import argparse

def arg_parser():
    parser = argparse.ArgumentParser(description='Run meshing')
    parser.add_argument('--case', type=str, default='case9_300k/', help='case name')
    parser.add_argument('--mesh', type=str, default='dab_mesh', help='mesh name')
    parser.add_argument('-ef', '--target_edge_length_f', type=float, default=0.280, help='target edge length for fluid mesh')
    parser.add_argument('-es','--target_edge_length_s', type=float, default=0.300, help='target edge length for solid mesh')
    parser.add_argument('--thick_solid', type=float, default=0.25, help='thick solid')
    parser.add_argument('--nb_boundarylayers', type=int, default=2, help='nb boundarylayers')
    parser.add_argument('--seedX', type=float, nargs=3, default=(123.099, 134.62, 64.087), help='seedX')
    return parser.parse_args()


def main():
    args = arg_parser()
    case = args.case
    mesh = args.mesh
    TargetEdgeLength_f = args.target_edge_length_f
    TargetEdgeLength_s = args.target_edge_length_s
    Thick_solid = args.thick_solid
    nb_boundarylayers = args.nb_boundarylayers
    seedX = args.seedX
    clip_surface=False
    
    ifile_surface = "surfaces/" + case + mesh+ ".stl"
    ofile_mesh = "surfaces/" + case + mesh
    # check if output file already exists and delete it if it does
    output_files = [ofile_mesh+"_fsi.vtu", ofile_mesh+"_fsi.xml", ofile_mesh+"_cfd.vtu", ofile_mesh+"_cfd.xml"]
    for file in output_files:
        if path.exists(file):
            remove(file)

    # Parameters ###################################################################
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

if __name__ == '__main__':
    #TODO: auto restart does not work 
    max_retries = 5
    retry_delay = 5  # Time delay in seconds between retries
    args = arg_parser()
    main()