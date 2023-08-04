from vmtkmeshgeneratorfsi import vmtkMeshGeneratorFsi
from vmtk import vmtkdistancetospheres
from morphman import vmtkscripts, write_polydata, get_point_data_array, create_vtk_array

# Global array names
distanceToSpheresArrayName = "DistanceToSpheres"
distanceToSpheresArrayNameSolid = "Thickness"


def distance_to_spheres_solid_thickness(surface, save_path, distance_offset=0, distance_scale=0.1,
                                        min_distance=0.25, max_distance=0.3):
    """
    Determines the solid thickness using vmtkdistancetospheres.

    Args:
        surface (vtkPolyData): Input surface model
        distance_offset (float): Offset added to the distances
        distance_scale (float): Scale applied to the distances
        min_distance (float): Minimum value for the distances
        max_distance (float): Maximum value for the distances

    Returns:
        surface (vtkPolyData): Processed surface model with info on solid thickness

    """
    distanceToSpheres = vmtkdistancetospheres.vmtkDistanceToSpheres()
    distanceToSpheres.Surface = surface
    distanceToSpheres.DistanceOffset = distance_offset
    distanceToSpheres.DistanceScale = distance_scale
    distanceToSpheres.MinDistance = min_distance
    distanceToSpheres.MaxDistance = max_distance
    distanceToSpheres.DistanceToSpheresArrayName = distanceToSpheresArrayNameSolid
    distanceToSpheres.Execute()
    distance_to_sphere = distanceToSpheres.Surface

    write_polydata(distance_to_sphere, save_path)

    return distance_to_sphere


def dist_sphere_spheres(surface, centerlines, region_center, misr_max, save_path, factor,
                        distance_offset=0.0, distance_scale=0.2, min_distance=0.4, max_distance=0.7):
    """
    Determines the target edge length for each cell on the surface, including
    potential refinement or coarsening of certain user specified areas.
    Level of refinement/coarseness is determined based on the distance to the spheres.

    Args:
        surface (vtkPolyData): Input surface model
        centerlines (vtkPolyData): Centerlines of input model
        region_center (list): Point representing region to refine
        misr_max (list): Maximum inscribed sphere radius in region of refinement
        save_path (str): Location to store processed surface
        factor (float): Coarsening factor, determining the level of refinement (<1) or coarsening (>1)
        distance_offset (float): Offset added to the distances
        distance_scale (float): Scale applied to the distances
        min_distance (float): Minimum value for the distances
        max_distance (float): Maximum value for the distances

    Returns:
        surface (vtkPolyData): Processed surface model with info on cell specific target edge length
    """
    distanceToSpheres = vmtkdistancetospheres.vmtkDistanceToSpheres()
    distanceToSpheres.Surface = surface
    distanceToSpheres.DistanceOffset = distance_offset
    distanceToSpheres.DistanceScale = distance_scale
    distanceToSpheres.MinDistance = min_distance
    distanceToSpheres.MaxDistance = max_distance
    distanceToSpheres.DistanceToSpheresArrayName = distanceToSpheresArrayName
    distanceToSpheres.Execute()
    distance_to_sphere = distanceToSpheres.Surface

    surfaceCurvature = vmtkscripts.vmtkSurfaceCurvature()
    surfaceCurvature.AbsoluteCurvature = 1
    surfaceCurvature.MedianFiltering = 1
    surfaceCurvature.CurvatureType = "gaussian"
    surfaceCurvature.Offset = 0.15
    surfaceCurvature.BoundedReciprocal = 1
    surfaceCurvature.Surface = distance_to_sphere
    surfaceCurvature.Execute()
    distance_to_sphere = surfaceCurvature.Surface

    surfaceArrayOperation = vmtkscripts.vmtkSurfaceArrayOperation()
    surfaceArrayOperation.Surface = distance_to_sphere
    surfaceArrayOperation.InputArrayName = "Curvature"
    surfaceArrayOperation.Input2ArrayName = distanceToSpheresArrayName
    surfaceArrayOperation.ResultArrayName = "Size"
    surfaceArrayOperation.Operation = "multiply"
    surfaceArrayOperation.Execute()
    distance_to_sphere = surfaceArrayOperation.Surface
    
    remeshing = vmtkscripts.vmtkSurfaceRemeshing()
    remeshing.Surface = distance_to_sphere
    remeshing.ElementSizeMode = "edgelengtharray"
    remeshing.TargetEdgeLengthArrayName = "Size"
    remeshing.Execute()
    distance_to_sphere = remeshing.Surface

    write_polydata(distance_to_sphere, save_path)

    return distance_to_sphere


def generate_mesh(surface, add_boundary_layer, meshing_method, solid_thickness, solid_thickness_parameters):
    """
    Generates a mesh suitable for FSI from a input surface model.

    Args:
        surface (vtkPolyData): Surface model to be meshed.

    Returns:
        mesh (vtkUnstructuredGrid): Output mesh
        remeshedsurface (vtkPolyData): Remeshed version of the input model
    """

    print("--- Creating FSI mesh")
    meshGenerator = vmtkMeshGeneratorFsi()
    meshGenerator.Surface = surface
    # for remeshing
    meshGenerator.SkipRemeshing = 1
    if meshing_method == 'distancetospheres':
        meshGenerator.ElementSizeMode = 'edgelength'
        meshGenerator.TargetEdgeLength = 0.5
    else:
        meshGenerator.ElementSizeMode = "edgelengtharray"  # Variable size mesh
        meshGenerator.TargetEdgeLengthArrayName = "Size"  # Variable size mesh
    meshGenerator.LogOn = 1
    # for boundary layer (used for both fluid boundary layer and solid domain)
    meshGenerator.BoundaryLayer = 1
    meshGenerator.NumberOfSubLayersSolid = 2
    meshGenerator.NumberOfSubLayersFluid = 2
    meshGenerator.BoundaryLayerOnCaps = 0
    meshGenerator.SubLayerRatioFluid = 0.75
    meshGenerator.SubLayerRatioSolid = 0.75
    meshGenerator.BoundaryLayerThicknessFactorFluid = 0.85
    meshGenerator.BoundaryLayerThicknessFactorSolid = 1
    if solid_thickness == 'variable':
        meshGenerator.ElementSizeModeSolid = "edgelengtharray"
        meshGenerator.TargetEdgeLengthArrayNameSolid = distanceToSpheresArrayNameSolid
    else:
        meshGenerator.ElementSizeModeSolid = "edgelength"
    # mesh
    meshGenerator.Tetrahedralize = 1
    meshGenerator.VolumeElementScaleFactor = 0.8
    meshGenerator.EndcapsEdgeLengthFactor = 1.0
    # Cells and walls numbering
    meshGenerator.SolidSideWallId = 11
    meshGenerator.InterfaceId_fsi = 22
    meshGenerator.InterfaceId_outer = 33
    meshGenerator.VolumeId_fluid = 0  # (keep to 0)
    meshGenerator.VolumeId_solid = 1

    # Mesh
    meshGenerator.Execute()

    # Remeshed surface, store for later
    remeshSurface = meshGenerator.RemeshedSurface

    # Full mesh
    mesh = meshGenerator.Mesh

    return mesh, remeshSurface
