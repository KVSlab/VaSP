#!/usr/bin/env python

# MODIFICATION OF vmtkmeshgenerator.py FILE FROM VMTK
# SUITABE FOR Fluid Structure Interaction MESH GENERATION
# THIS FILE COULD BE ADDED AS A VMTK CLASS
# MODIFICATION BY: Alban Souche, SIMULA, Fornebu (October 2018)

######################## MODIFED FROM ##########################################
## Program:   VMTK
# Module:    $RCSfile: vmtkmeshgenerator.py,v $
## Language:  Python
# Date:      $Date: 2006/02/23 09:27:52 $
# Version:   $Revision: 1.7 $

# Copyright (c) Luca Antiga, David Steinman. All rights reserved.
# See LICENCE file for details.

# This software is distributed WITHOUT ANY WARRANTY; without even
# the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
# PURPOSE.  See the above copyright notices for more information.
################################################################################

from __future__ import absolute_import  # NEEDS TO STAY AS TOP LEVEL MODULE FOR Py2-3 COMPATIBILITY
import vtk
from vmtk import vtkvmtk, vmtkscripts, pypes
import sys
import numpy as np


class vmtkMeshGeneratorFsi(pypes.pypeScript):

    def __init__(self):

        pypes.pypeScript.__init__(self)

        self.Surface = None

        self.TargetEdgeLength = 1.0
        self.TargetEdgeLengthFactor = 1.0
        self.TargetEdgeLengthArrayName = ''
        self.TargetEdgeLengthArrayNameSolid = ''
        self.MaxEdgeLength = 1E16
        self.MinEdgeLength = 0.0
        self.TriangleSplitFactor = 5.0
        self.CellEntityIdsArrayName = 'CellEntityIds'
        self.ElementSizeMode = 'edgelength'
        self.ElementSizeModeFluid = 'edgelength'
        self.ElementSizeModeSolid = 'edgelength'
        self.VolumeElementScaleFactor = 0.8
        self.CappingMethod = 'simple'
        self.SkipCapping = 0
        self.RemeshCapsOnly = 0
        self.SkipRemeshing = 0
        self.EndcapsEdgeLengthFactor = 1.0

        self.BoundaryLayer = 0
        self.NumberOfSubLayersFluid = 2
        self.NumberOfSubLayersSolid = 2
        self.SubLayerRatioFluid = 0.75
        self.SubLayerRatioSolid = 0.5
        self.BoundaryLayerThicknessFactorFluid = 0.85
        self.BoundaryLayerThicknessFactorSolid = 0.25

        self.NumberOfSubsteps = 2000
        self.Relaxation = 0.01
        self.LocalCorrectionFactor = 0.45

        self.Tetrahedralize = 0

        self.BoundaryLayerOnCaps = 1

        self.SizingFunctionArrayName = 'VolumeSizingFunction'

        self.SolidSideWallId = 11
        self.InterfaceId_fsi = 22
        self.InterfaceId_outer = 33
        self.VolumeId_fluid = 0  # (keep to 0)
        self.VolumeId_solid = 1

        self.isAVF = False
        self.VeinIdsOffset = 1000

        self.Mesh = None
        self.RemeshedSurface = None

        self.SetScriptName('vmtkmeshgenerator')
        self.SetScriptDoc('generate a mesh suitable for CFD from a surface')
        self.SetInputMembers([
            ['Surface', 'i', 'vtkPolyData', 1, '', 'the input surface', 'vmtksurfacereader'],
            ['TargetEdgeLength', 'edgelength', 'float', 1, '(0.0,)'],
            ['TargetEdgeLengthArrayName', 'edgelengtharray', 'str', 1],
            ['TargetEdgeLengthArrayNameSolid', 'edgelengtharraysolid', 'str', 1],
            ['TargetEdgeLengthFactor', 'edgelengthfactor', 'float', 1, '(0.0,)'],
            ['TriangleSplitFactor', 'trianglesplitfactor', 'float', 1, '(0.0,)'],
            ['EndcapsEdgeLengthFactor', 'endcapsedgelengthfactor', 'float', 1, '(0.0,)'],
            ['MaxEdgeLength', 'maxedgelength', 'float', 1, '(0.0,)'],
            ['MinEdgeLength', 'minedgelength', 'float', 1, '(0.0,)'],
            ['CellEntityIdsArrayName', 'entityidsarray', 'str', 1],
            ['ElementSizeMode', 'elementsizemode', 'str', 1, '["edgelength","edgelengtharray"]'],
            ['ElementSizeModeFluid', 'elementsizemodefluid', 'str', 1, '["edgelength","edgelengtharray"]'],
            ['ElementSizeModeSolid', 'elementsizemodesolid', 'str', 1, '["edgelength","edgelengtharray"]'],
            ['CappingMethod', 'cappingmethod', 'str', 1, '["simple","annular","concaveannular"]'],
            ['SkipCapping', 'skipcapping', 'bool', 1, ''],
            ['SkipRemeshing', 'skipremeshing', 'bool', 1, ''],
            ['VolumeElementScaleFactor', 'volumeelementfactor', 'float', 1, '(0.0,)'],
            ['BoundaryLayer', 'boundarylayer', 'bool', 1, ''],
            ['NumberOfSubLayersFluid', 'sublayersfluid', 'int', 1, '(0,)'],
            ['NumberOfSubLayersSolid', 'sublayerssolid', 'int', 1, '(0,)'],
            ['NumberOfSubsteps', 'substeps', 'int', 1, '(0,)'],
            ['NumberOfSubstepsSolid', 'substepssolid', 'int', 1, '(0,)'],
            ['NumberOfSubstepsFluid', 'substepsfluid', 'int', 1, '(0,)'],
            ['Relaxation', 'relaxation', 'float', 1, '(0.0,)'],
            ['RelaxationSolid', 'relaxationsolid', 'float', 1, '(0.0,)'],
            ['RelaxationFluid', 'relaxationfluid', 'float', 1, '(0.0,)'],
            ['LocalCorrectionFactor', 'localcorrection', 'float', 1, '(0.0,)'],
            ['LocalCorrectionFactorFluid', 'localcorrectionfluid', 'float', 1, '(0.0,)'],
            ['LocalCorrectionFactorSolid', 'localcorrectionsolid', 'float', 1, '(0.0,)'],
            ['SubLayerRatioFluid', 'sublayerratiofluid', 'float', 1, '(0.0,)'],
            ['SubLayerRatioSolid', 'sublayerratiosolid', 'float', 1, '(0.0,)'],
            ['BoundaryLayerThicknessFactor', 'thicknessfactor', 'float', 1, '(0.0,)'],
            ['BoundaryLayerThicknessFactorFluid', 'thicknessfactorfluid', 'float', 1, '(0.0,)'],
            ['BoundaryLayerThicknessFactorSolid', 'thicknessfactorsolid', 'float', 1, '(0.0,)'],
            ['RemeshCapsOnly', 'remeshcapsonly', 'bool', 1, ''],
            ['BoundaryLayerOnCaps', 'boundarylayeroncaps', 'bool', 1, ''],
            ['Tetrahedralize', 'tetrahedralize', 'bool', 1, ''],
            ['VeinIdsOffset', 'veinidsoffset', 'int', 1000, '(0,)'],
            ['IsAVF', 'isAVF', 'bool', 0, ''],
        ])
        self.SetOutputMembers([
            ['Mesh', 'o', 'vtkUnstructuredGrid', 1, '', 'the output mesh', 'vmtkmeshwriter'],
            ['CellEntityIdsArrayName', 'entityidsarray', 'str', 1],
            ['RemeshedSurface', 'remeshedsurface', 'vtkPolyData', 1, '', 'the output surface', 'vmtksurfacewriter'],
        ])

    def Execute(self):

        from vmtk import vmtkscripts
        if self.Surface == None:
            self.PrintError('Error: No input surface.')

        wallEntityOffset = 1

        if self.SkipCapping or not self.BoundaryLayerOnCaps:
            self.PrintLog("Not capping surface")
            surface = self.Surface
            cellEntityIdsArray = vtk.vtkIntArray()
            cellEntityIdsArray.SetName(self.CellEntityIdsArrayName)
            cellEntityIdsArray.SetNumberOfTuples(surface.GetNumberOfCells())
            cellEntityIdsArray.FillComponent(0, 0.0)
            surface.GetCellData().AddArray(cellEntityIdsArray)
        else:
            self.PrintLog("Capping surface")
            capper = vmtkscripts.vmtkSurfaceCapper()
            capper.Surface = self.Surface
            capper.Interactive = 0
            capper.Method = self.CappingMethod
            capper.TriangleOutput = 0
            capper.CellEntityIdOffset = wallEntityOffset
            capper.Execute()
            surface = capper.Surface

        if self.SkipRemeshing:
            remeshedSurface = surface
        else:
            self.PrintLog("Remeshing surface")
            remeshing = vmtkscripts.vmtkSurfaceRemeshing()
            remeshing.Surface = surface
            remeshing.CellEntityIdsArrayName = self.CellEntityIdsArrayName
            remeshing.TargetEdgeLength = self.TargetEdgeLength
            remeshing.MaxEdgeLength = self.MaxEdgeLength
            remeshing.MinEdgeLength = self.MinEdgeLength
            remeshing.TargetEdgeLengthFactor = self.TargetEdgeLengthFactor
            remeshing.TargetEdgeLengthArrayName = self.TargetEdgeLengthArrayName
            remeshing.TriangleSplitFactor = self.TriangleSplitFactor
            remeshing.ElementSizeMode = self.ElementSizeMode
            if self.RemeshCapsOnly:
                remeshing.ExcludeEntityIds = [wallEntityOffset]
            remeshing.Execute()
            remeshedSurface = remeshing.Surface

        if self.BoundaryLayer:

            projection = vmtkscripts.vmtkSurfaceProjection()
            projection.Surface = remeshedSurface
            projection.ReferenceSurface = surface
            projection.Execute()

            normals = vmtkscripts.vmtkSurfaceNormals()
            normals.Surface = projection.Surface
            normals.NormalsArrayName = 'Normals'
            normals.Execute()

            surfaceToMesh = vmtkscripts.vmtkSurfaceToMesh()
            surfaceToMesh.Surface = normals.Surface
            surfaceToMesh.Execute()

            self.PrintLog("Generating boundary layer fluid")
            placeholderCellEntityId = 9999
            boundaryLayer = vmtkscripts.vmtkBoundaryLayer()
            boundaryLayer.Mesh = surfaceToMesh.Mesh
            boundaryLayer.WarpVectorsArrayName = 'Normals'
            boundaryLayer.NegateWarpVectors = True
            boundaryLayer.ThicknessArrayName = self.TargetEdgeLengthArrayName
            if self.ElementSizeMode == 'edgelength':
                boundaryLayer.ConstantThickness = True
            else:
                boundaryLayer.ConstantThickness = False
            boundaryLayer.IncludeSurfaceCells = 0
            boundaryLayer.NumberOfSubLayers = self.NumberOfSubLayersFluid
            boundaryLayer.NumberOfSubsteps = self.NumberOfSubsteps
            boundaryLayer.Relaxation = self.Relaxation
            boundaryLayer.LocalCorrectionFactor = self.LocalCorrectionFactor
            boundaryLayer.SubLayerRatio = self.SubLayerRatioFluid
            boundaryLayer.Thickness = self.BoundaryLayerThicknessFactorFluid * self.TargetEdgeLength
            boundaryLayer.ThicknessRatio = self.BoundaryLayerThicknessFactorFluid * self.TargetEdgeLengthFactor
            boundaryLayer.MaximumThickness = self.BoundaryLayerThicknessFactorFluid * self.MaxEdgeLength
            if not self.BoundaryLayerOnCaps:
                boundaryLayer.SidewallCellEntityId = placeholderCellEntityId
                boundaryLayer.InnerSurfaceCellEntityId = wallEntityOffset
                boundaryLayer.VolumeCellEntityId = self.VolumeId_fluid
            boundaryLayer.Execute()

            self.PrintLog("Generating boundary layer solid")
            boundaryLayer2 = vmtkscripts.vmtkBoundaryLayer()
            boundaryLayer2.Mesh = surfaceToMesh.Mesh
            boundaryLayer2.WarpVectorsArrayName = 'Normals'
            boundaryLayer2.NegateWarpVectors = True
            boundaryLayer2.ThicknessArrayName = self.TargetEdgeLengthArrayNameSolid
            if self.ElementSizeModeSolid == 'edgelength':
                boundaryLayer2.ConstantThickness = True
            else:
                boundaryLayer2.ConstantThickness = False
            boundaryLayer2.IncludeSurfaceCells = 1
            boundaryLayer2.NumberOfSubLayers = self.NumberOfSubLayersSolid
            boundaryLayer2.NumberOfSubsteps = self.NumberOfSubsteps
            boundaryLayer2.Relaxation = self.Relaxation
            boundaryLayer2.LocalCorrectionFactor = self.LocalCorrectionFactor
            boundaryLayer2.SubLayerRatio = self.SubLayerRatioSolid
            boundaryLayer2.Thickness = self.BoundaryLayerThicknessFactorSolid * self.TargetEdgeLength
            boundaryLayer2.ThicknessRatio = self.BoundaryLayerThicknessFactorSolid * self.TargetEdgeLengthFactor
            boundaryLayer2.MaximumThickness = self.BoundaryLayerThicknessFactorSolid * self.MaxEdgeLength
            if not self.BoundaryLayerOnCaps:
                boundaryLayer2.SidewallCellEntityId = self.SolidSideWallId  # placeholderCellEntityId
                boundaryLayer2.InnerSurfaceCellEntityId = self.InterfaceId_fsi  # wallEntityOffset
                boundaryLayer2.OuterSurfaceCellEntityId = self.InterfaceId_outer  # wallEntityOffset
                boundaryLayer2.VolumeCellEntityId = self.VolumeId_solid
            boundaryLayer2.Execute()

            meshToSurface = vmtkscripts.vmtkMeshToSurface()
            meshToSurface.Mesh = boundaryLayer.InnerSurfaceMesh
            meshToSurface.Execute()

            innerSurface = meshToSurface.Surface

            if self.isAVF:
                self.PrintLog("Generating centerlines.")
                centerlinesExtract = vmtkscripts.vmtkCenterlines()
                centerlinesExtract.Surface = innerSurface
                centerlinesExtract.SeedSelectorName = "openprofiles"
                centerlinesExtract.Execute()
    
                extractGroups = vmtkscripts.vmtkBranchExtractor()
                extractGroups.Centerlines = centerlinesExtract.Centerlines
                extractGroups.RadiusArrayName = centerlinesExtract.RadiusArrayName
                extractGroups.GroupIdsArrayName = 'GroupIds'
                extractGroups.Execute()
    
                clipVein = vmtkscripts.vmtkMeshBranchClipper()
                clipVein.Mesh = boundaryLayer2.Mesh
                clipVein.Centerlines = extractGroups.Centerlines
                clipVein.GroupIdsArrayName = extractGroups.GroupIdsArrayName
                clipVein.RadiusArrayName = extractGroups.RadiusArrayName
                clipVein.BlankingArrayName = extractGroups.BlankingArrayName
                clipVein.Interactive = 1
                clipVein.Execute()
    
                solidCellLocator = vtk.vtkCellLocator()
                veinCellLocator = vtk.vtkCellLocator()
    
                solidCellLocator.SetDataSet(boundaryLayer2.Mesh)
                solidCellLocator.BuildLocator()
    
                veinCellLocator.SetDataSet(clipVein.Mesh)
                veinCellLocator.BuildLocator()
    
                solidCellIds = boundaryLayer2.Mesh.GetCellData().GetScalars("CellEntityIds")
    
                vtkIdList = vtk.vtkIdList()
                solidCellLocator.FindCellsWithinBounds(clipVein.Mesh.GetBounds(), vtkIdList)
                ids = np.array([vtkIdList.GetId(i) for i in range(vtkIdList.GetNumberOfIds())])
    
                cell = [0.0, 0.0, 0.0]
                cellId = vtk.mutable(0)
                subId = vtk.mutable(0)
                dist = vtk.mutable(0.0)
    
                for id in ids:
                    point = boundaryLayer2.Mesh.GetCell(id).GetPoints().GetPoint(0)
                    veinCellLocator.FindClosestPoint(point, cell, cellId, subId, dist)
                    if dist == 0:
                        solidCellIds.SetValue(id, solidCellIds.GetValue(id) + self.VeinIdsOffset)
    
                boundaryLayer2.Mesh.GetCellData().Update()
    
#                clipArtery = vmtkscripts.vmtkMeshBranchClipper()
#                clipArtery.Mesh = boundaryLayer2.Mesh
#                clipArtery.Centerlines = extractGroups.Centerlines
#                clipArtery.GroupIds = clipVein.GroupIds
#                clipArtery.GroupIdsArrayName = extractGroups.GroupIdsArrayName
#                clipArtery.RadiusArrayName = extractGroups.RadiusArrayName
#                clipArtery.BlankingArrayName = extractGroups.BlankingArrayName
#                clipArtery.InsideOut = 1
#                clipArtery.Execute()
#    
#                mergeSolid = vmtkscripts.vmtkMeshMerge()
#                mergeSolid.Mesh1 = clipVein.Mesh
#                mergeSolid.Mesh2 = clipArtery.Mesh
#                mergeSolid.CellEntityIdOffset1 = int(1000)
#                mergeSolid.CellEntityIdOffset2 = 0
#                mergeSolid.Execute()

            if not self.BoundaryLayerOnCaps:

                self.PrintLog("Capping inner surface")
                capper = vmtkscripts.vmtkSurfaceCapper()
                capper.Surface = innerSurface
                capper.Interactive = 0
                capper.Method = self.CappingMethod
                capper.TriangleOutput = 1
                capper.CellEntityIdOffset = wallEntityOffset
                capper.Execute()

                self.PrintLog("Remeshing endcaps")
                remeshing = vmtkscripts.vmtkSurfaceRemeshing()
                remeshing.Surface = capper.Surface
                remeshing.CellEntityIdsArrayName = self.CellEntityIdsArrayName
                remeshing.TargetEdgeLength = self.TargetEdgeLength * self.EndcapsEdgeLengthFactor
                remeshing.MaxEdgeLength = self.MaxEdgeLength
                remeshing.MinEdgeLength = self.MinEdgeLength
                remeshing.TargetEdgeLengthFactor = self.TargetEdgeLengthFactor * self.EndcapsEdgeLengthFactor
                remeshing.TargetEdgeLengthArrayName = self.TargetEdgeLengthArrayName
                remeshing.TriangleSplitFactor = self.TriangleSplitFactor
                remeshing.ElementSizeMode = self.ElementSizeMode
                remeshing.ExcludeEntityIds = [wallEntityOffset]  # [wallEntityOffset, InterfaceId] #[wallEntityOffset]
                remeshing.Execute()

                innerSurface = remeshing.Surface

            self.PrintLog("Computing sizing function")
            sizingFunction = vtkvmtk.vtkvmtkPolyDataSizingFunction()
            sizingFunction.SetInputData(innerSurface)
            sizingFunction.SetSizingFunctionArrayName(self.SizingFunctionArrayName)
            sizingFunction.SetScaleFactor(self.VolumeElementScaleFactor)
            sizingFunction.Update()

            surfaceToMesh2 = vmtkscripts.vmtkSurfaceToMesh()
            surfaceToMesh2.Surface = sizingFunction.GetOutput()
            surfaceToMesh2.Execute()

            self.PrintLog("Generating volume mesh")
            tetgen = vmtkscripts.vmtkTetGen()
            tetgen.Mesh = surfaceToMesh2.Mesh
            tetgen.GenerateCaps = 0
            tetgen.UseSizingFunction = 1
            tetgen.SizingFunctionArrayName = self.SizingFunctionArrayName
            tetgen.CellEntityIdsArrayName = self.CellEntityIdsArrayName
            tetgen.Order = 1
            tetgen.Quality = 1
            tetgen.PLC = 1
            tetgen.NoBoundarySplit = 1
            tetgen.RemoveSliver = 1
            tetgen.OutputSurfaceElements = 1
            tetgen.OutputVolumeElements = 1
            tetgen.RegionAttrib = 0
            tetgen.Execute()

            # ADDING CELL IDs

            #array = vtk.vtkDoubleArray()
            # array.SetNumberOfTuples(tetgen.Mesh.GetNumberOfCells())
            # array.SetNumberOfComponents(1)
            #array.FillComponent(0, 1.0)
            # array.SetName('RegionIDs')
            # tetgen.Mesh.GetCellData().AddArray(array)

            #array = vtk.vtkDoubleArray()
            # array.SetNumberOfTuples(boundaryLayer.Mesh.GetNumberOfCells())
            # array.SetNumberOfComponents(1)
            #array.FillComponent(0, 1.0)
            # array.SetName('RegionIDs')
            # boundaryLayer.Mesh.GetCellData().AddArray(array)

            #array = vtk.vtkDoubleArray()
            # array.SetNumberOfTuples(boundaryLayer2.Mesh.GetNumberOfCells())
            # array.SetNumberOfComponents(1)
            #array.FillComponent(0, 2.0)
            # array.SetName('RegionIDs')
            # boundaryLayer2.Mesh.GetCellData().AddArray(array)

            #####

            # w1 = vtk.vtkXMLUnstructuredGridWriter()
            # w1.SetInputData(boundaryLayer2.Mesh)
            # w1.SetFileName('BoundaryLayer2.vtu')
            # w1.Write()
            #
            # w2 = vtk.vtkXMLUnstructuredGridWriter()
            # w2.SetInputData(tetgen.Mesh)
            # w2.SetFileName('Tetgen.vtu')
            # w2.Write()
            #
            # w3 = vtk.vtkXMLUnstructuredGridWriter()
            # w3.SetInputData(boundaryLayer.Mesh)
            # w3.SetFileName('BoundaryLayer.vtu')
            # w3.Write()
            #
            # w4 = vtk.vtkXMLUnstructuredGridWriter()
            # w4.SetInputData(boundaryLayer.InnerSurfaceMesh)
            # w4.SetFileName('InnerLayer.vtu')
            # w4.Write()

            #from IPython import embed; embed()

            self.PrintLog("Assembling fluid mesh")
            appendFilter = vtkvmtk.vtkvmtkAppendFilter()
            appendFilter.AddInputData(boundaryLayer.Mesh)
            appendFilter.AddInputData(tetgen.Mesh)
            appendFilter.Update()
            self.Mesh = appendFilter.GetOutput()

            if not self.BoundaryLayerOnCaps:
                cellEntityIdsArray = self.Mesh.GetCellData().GetArray(self.CellEntityIdsArrayName)

                def VisitNeighbors(i, cellEntityId):
                    cellPointIds = vtk.vtkIdList()
                    self.Mesh.GetCellPoints(i, cellPointIds)
                    neighborPointIds = vtk.vtkIdList()
                    neighborPointIds.SetNumberOfIds(1)
                    pointNeighborCellIds = vtk.vtkIdList()
                    neighborCellIds = vtk.vtkIdList()

                    for j in range(cellPointIds.GetNumberOfIds()):
                        neighborPointIds.SetId(0, cellPointIds.GetId(j))
                        self.Mesh.GetCellNeighbors(i, neighborPointIds, pointNeighborCellIds)
                        for k in range(pointNeighborCellIds.GetNumberOfIds()):
                            neighborCellIds.InsertNextId(pointNeighborCellIds.GetId(k))

                    for j in range(neighborCellIds.GetNumberOfIds()):
                        cellId = neighborCellIds.GetId(j)
                        neighborCellEntityId = cellEntityIdsArray.GetTuple1(cellId)
                        neighborCellType = self.Mesh.GetCellType(cellId)
                        if neighborCellType not in [vtk.VTK_TRIANGLE, vtk.VTK_QUADRATIC_TRIANGLE, vtk.VTK_QUAD]:
                            continue
                        if neighborCellEntityId != placeholderCellEntityId:
                            continue
                        cellEntityIdsArray.SetTuple1(cellId, cellEntityId)
                        VisitNeighbors(cellId, cellEntityId)

                for i in range(self.Mesh.GetNumberOfCells()):
                    cellEntityId = cellEntityIdsArray.GetTuple1(i)
                    cellType = self.Mesh.GetCellType(i)
                    if cellType not in [vtk.VTK_TRIANGLE, vtk.VTK_QUADRATIC_TRIANGLE, vtk.VTK_QUAD]:
                        continue
                    if cellEntityId in [0, 1, placeholderCellEntityId]:
                        continue
                    VisitNeighbors(i, cellEntityId)

            self.PrintLog("Assembling final FSI mesh")
            appendFilter2 = vtkvmtk.vtkvmtkAppendFilter()
            appendFilter2.AddInputData(appendFilter.GetOutput())
            appendFilter2.AddInputData(boundaryLayer2.Mesh)
            #appendFilter2.AddInputData(mergeSolid.GetOutput())
            appendFilter2.Update()
            self.Mesh = appendFilter2.GetOutput()

        else:

            self.PrintLog("Computing sizing function")
            sizingFunction = vtkvmtk.vtkvmtkPolyDataSizingFunction()
            sizingFunction.SetInputData(remeshedSurface)
            sizingFunction.SetSizingFunctionArrayName(self.SizingFunctionArrayName)
            sizingFunction.SetScaleFactor(self.VolumeElementScaleFactor)
            sizingFunction.Update()

            self.PrintLog("Converting surface to mesh")
            surfaceToMesh = vmtkscripts.vmtkSurfaceToMesh()
            surfaceToMesh.Surface = sizingFunction.GetOutput()
            surfaceToMesh.Execute()

            self.PrintLog("Generating volume mesh")
            tetgen = vmtkscripts.vmtkTetGen()
            tetgen.Mesh = surfaceToMesh.Mesh
            tetgen.GenerateCaps = 0
            tetgen.UseSizingFunction = 1
            tetgen.SizingFunctionArrayName = self.SizingFunctionArrayName
            tetgen.CellEntityIdsArrayName = self.CellEntityIdsArrayName
            tetgen.Order = 1
            tetgen.Quality = 1
            tetgen.PLC = 1
            tetgen.NoBoundarySplit = 1
            tetgen.RemoveSliver = 1
            tetgen.OutputSurfaceElements = 1
            tetgen.OutputVolumeElements = 1

            tetgen.Execute()

            self.Mesh = tetgen.Mesh

            if self.Mesh.GetNumberOfCells() == 0 and surfaceToMesh.Mesh.GetNumberOfCells() > 0:
                self.PrintLog('An error occurred during tetrahedralization. Will only output surface mesh.')
                self.Mesh = surfaceToMesh.Mesh

        if self.Tetrahedralize:

            tetrahedralize = vtkvmtk.vtkvmtkUnstructuredGridTetraFilter()
            tetrahedralize.SetInputData(self.Mesh)
            tetrahedralize.Update()

            self.Mesh = tetrahedralize.GetOutput()

        self.RemeshedSurface = remeshedSurface


if __name__ == '__main__':

    main = pypes.pypeMain()
    main.Arguments = sys.argv
    main.Execute()
