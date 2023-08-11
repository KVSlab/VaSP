from fsipy.automatedPreprocessing.vmtkmeshgeneratorfsi import vmtkMeshGeneratorFsi
from vmtk import vmtkdistancetospheres
from morphman import vmtkscripts, write_polydata, get_point_data_array, create_vtk_array
from dolfin import Mesh, MeshFunction, File, HDF5File
from pathlib import Path

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

def convert_xml_mesh_to_hdf5(file_name_xml_mesh, scaling_factor=0.001):
    """Converts an XML mesh to an HDF5 mesh.

    Args:
        file_name_xml_mesh (str): The name of the XML mesh file.
        scaling_factor (float, optional): A scaling factor to apply to the mesh coordinates. The default value is 0.001, which converts from millimeters to meters.

    Returns:
        None

    Raises:
        FileNotFoundError: If the XML mesh file does not exist.
    """

    # Check if the XML mesh file exists
    if not Path(file_name_xml_mesh).is_file():
        raise FileNotFoundError(f"The file '{file_name_xml_mesh}' does not exist.")

    mesh = Mesh(file_name_xml_mesh)

    # Rescale the mesh coordinates
    x = mesh.coordinates()
    x[:, :] *= scaling_factor
    mesh.bounding_box_tree().build(mesh)

    # Convert subdomains to mesh function
    boundaries = MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries.set_values(boundaries.array() + 1)  # FIXME: Explain why this is necessary

    boundary_file = File(str(Path(file_name_xml_mesh).with_suffix('').with_suffix('')) + '_boundaries.pvd')
    boundary_file << boundaries

    domains = MeshFunction("size_t", mesh, 3, mesh.domains())
    domains.set_values(domains.array() + 1)  # in order to have fluid==1 and solid==2

    domain_file = File(str(Path(file_name_xml_mesh).with_suffix('').with_suffix('')) + '_domains.pvd')
    domain_file << domains

    file_name_h5_mesh = str(Path(file_name_xml_mesh).with_suffix('').with_suffix('.h5'))
    hdf = HDF5File(mesh.mpi_comm(), file_name_h5_mesh, "w")
    hdf.write(mesh, "/mesh")
    hdf.write(boundaries, "/boundaries")
    hdf.write(domains, "/domains")
