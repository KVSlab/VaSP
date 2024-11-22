# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later

from pathlib import Path
from typing import Union

import h5py
import numpy as np
import meshio
from dolfin import Mesh, MeshFunction, File, HDF5File, FunctionSpace, Function, XDMFFile, cells, Edge
from vmtk import vmtkdistancetospheres, vmtkdijkstradistancetopoints
from morphman import vmtkscripts, write_polydata

from vasp.automatedPreprocessing.vmtkmeshgeneratorfsi import vmtkMeshGeneratorFsi
from vasp.simulations.simulation_common import load_mesh_and_data

from vtk import vtkPolyData

# Global array names
distanceToSpheresArrayName = "DistanceToSpheres"
distanceToSpheresArrayNameSolid = "Thickness"
dijkstraArrayName = "DijkstraDistanceToPoints"


def distance_to_spheres_solid_thickness(surface: vtkPolyData, save_path: Union[str, Path],
                                        distance_offset: float = 0, distance_scale: float = 0.1,
                                        min_distance: float = 0.25, max_distance: float = 0.3) -> vtkPolyData:
    """
    Determines the solid thickness using vmtkdistancetospheres.
    Write the distance data to `save_path`.

    Args:
        surface (vtkPolyData): Input surface model
        save_path (Union[str, Path]): Location to store processed surface
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

    write_polydata(distance_to_sphere, str(save_path))

    return distance_to_sphere


def dist_sphere_spheres(surface: vtkPolyData, save_path: Union[str, Path],
                        distance_offset: float, distance_scale: float,
                        min_distance: float, max_distance: float,
                        distance_method: str = 'geodesic') -> vtkPolyData:
    """
    Determines the target edge length for each cell on the surface, including
    potential refinement or coarsening of certain user specified areas.
    Level of refinement/coarseness is determined based on the distance to the spheres.
    The distance computation can be either 'euclidean' or 'geodesic' (default).

    Args:
        surface (vtkPolyData): Input surface model
        save_path (Union[str, Path]): Location to store processed surface
        distance_offset (float): Offset added to the distances
        distance_scale (float): Scale applied to the distances
        min_distance (float): Minimum value for the distances
        max_distance (float): Maximum value for the distances
        distance_method (str): Method to compute distances ('euclidean' or 'geodesic')

    Returns:
        surface (vtkPolyData): Processed surface model with info on cell specific target edge length
    """
    if distance_method == 'euclidean':
        distanceToSpheres = vmtkdistancetospheres.vmtkDistanceToSpheres()
        distance_array_name = distanceToSpheresArrayName
    elif distance_method == 'geodesic':
        distanceToSpheres = vmtkdijkstradistancetopoints.vmtkDijkstraDistanceToPoints()
        distance_array_name = dijkstraArrayName
    else:
        raise ValueError("Invalid distance computation method. Choose 'euclidean' or 'geodesic'.")

    distanceToSpheres.Surface = surface
    distanceToSpheres.DistanceOffset = distance_offset
    distanceToSpheres.DistanceScale = distance_scale
    distanceToSpheres.MinDistance = min_distance
    distanceToSpheres.MaxDistance = max_distance
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
    surfaceArrayOperation.Input2ArrayName = distance_array_name
    surfaceArrayOperation.ResultArrayName = "Size"
    surfaceArrayOperation.Operation = "multiply"
    surfaceArrayOperation.Execute()
    distance_to_sphere = surfaceArrayOperation.Surface

    write_polydata(distance_to_sphere, str(save_path))

    return distance_to_sphere


def generate_mesh(surface: vtkPolyData, number_of_sublayers_fluid: int, number_of_sublayers_solid: int,
                  solid_thickness: str, solid_thickness_parameters: list, centerlines: vtkPolyData,
                  solid_side_wall_id: int = 11, interface_fsi_id: int = 22, solid_outer_wall_id: int = 33,
                  fluid_volume_id: int = 0, solid_volume_id: int = 1, no_solid: bool = False,
                  extract_branch: bool = False, branch_group_ids: list = [], branch_ids_offset: int = 1000) -> tuple:
    """
    Generates a mesh suitable for FSI from an input surface model.

    Args:
        surface (vtkPolyData): Surface model to be meshed.
        number_of_sublayers_fluid (int): Number of sublayers for fluid.
        number_of_sublayers_solid (int): Number of sublayers for solid.
        solid_thickness (str): Type of solid thickness ('variable' or 'constant').
        solid_thickness_parameters (list): List of parameters for solid thickness.
        centerlines (vtkPolyData): Centerlines of input model.
        solid_side_wall_id (int, optional): ID for solid side wall. Default is 11.
        interface_fsi_id (int, optional): ID for the FSI interface. Default is 22.
        solid_outer_wall_id (int, optional): ID for solid outer wall. Default is 33.
        fluid_volume_id (int, optional): ID for the fluid volume. Default is 0.
        solid_volume_id (int, optional): ID for the solid volume. Default is 1.
        no_solid (bool, optional): Generate mesh without solid.
        extract_branch (bool, optional): Enable extraction of a specific branch, marking solid mesh IDs with an offset.
        branch_group_ids (list, optional): Specify group IDs to extract for the branch.
        branch_ids_offset (int): Set offset for marking solid mesh IDs when extracting a branch.

    Returns:
        tuple: A tuple containing the generated mesh (vtkUnstructuredGrid) and the remeshed surface (vtkPolyData).
    """
    meshGenerator = vmtkscripts.vmtkMeshGenerator() if no_solid else vmtkMeshGeneratorFsi()
    meshGenerator.Surface = surface

    # Mesh Parameters
    meshGenerator.ElementSizeMode = "edgelengtharray"  # Variable size mesh
    meshGenerator.TargetEdgeLengthArrayName = "Size"  # Variable size mesh
    meshGenerator.LogOn = 1
    meshGenerator.ExitOnError = 0
    meshGenerator.BoundaryLayer = 1
    meshGenerator.NumberOfSubLayersSolid = number_of_sublayers_solid
    meshGenerator.NumberOfSubLayersFluid = number_of_sublayers_fluid
    meshGenerator.NumberOfSubLayers = number_of_sublayers_fluid
    meshGenerator.BoundaryLayerOnCaps = 0
    meshGenerator.SubLayerRatioFluid = 0.75
    meshGenerator.SubLayerRatioSolid = 0.75
    meshGenerator.SubLayerRatio = 0.75
    meshGenerator.BoundaryLayerThicknessFactor = 0.5
    meshGenerator.Tetrahedralize = 1
    meshGenerator.VolumeElementScaleFactor = 0.8
    meshGenerator.EndcapsEdgeLengthFactor = 1.0
    meshGenerator.Centerlines = centerlines
    meshGenerator.ExtractBranch = extract_branch
    meshGenerator.BranchGroupIds = branch_group_ids
    meshGenerator.BranchIdsOffset = branch_ids_offset
    meshGenerator.ThicknessMethod = solid_thickness

    # Solid thickness handling
    if solid_thickness in ["variable", "painted"]:
        meshGenerator.ElementSizeModeSolid = "edgelengtharray"
        meshGenerator.TargetEdgeLengthArrayNameSolid = distanceToSpheresArrayNameSolid
    else:
        meshGenerator.ElementSizeModeSolid = "edgelength"
        meshGenerator.SolidThickness = solid_thickness_parameters[0]
    # IDs
    meshGenerator.SolidSideWallId = solid_side_wall_id
    meshGenerator.InterfaceFsiId = interface_fsi_id
    meshGenerator.SolidOuterWallId = solid_outer_wall_id
    meshGenerator.FluidVolumeId = fluid_volume_id
    meshGenerator.SolidVolumeId = solid_volume_id

    # Generate mesh
    meshGenerator.Execute()
    remeshed_surface = meshGenerator.RemeshedSurface
    generated_mesh = meshGenerator.Mesh

    return generated_mesh, remeshed_surface


def convert_xml_mesh_to_hdf5(file_name_xml_mesh: Union[str, Path], scaling_factor: float = 1) -> None:
    """Converts an XML mesh to an HDF5 mesh.

    Args:
        file_name_xml_mesh (Union[str, Path]): The name of the XML mesh file.
        scaling_factor (float, optional): A scaling factor to apply to the mesh coordinates.
                                          The default value is 1 (no scaling). Note that probes
                                          and parameters inside _info.json file will not be scaled
                                          if you only scale HDF5 file.

    Returns:
        None

    Raises:
        FileNotFoundError: If the XML mesh file does not exist.
    """

    # Check if the XML mesh file exists
    xml_mesh_path = Path(file_name_xml_mesh)
    if not xml_mesh_path.is_file():
        raise FileNotFoundError(f"The file '{xml_mesh_path}' does not exist.")

    mesh = Mesh(str(xml_mesh_path))

    # Rescale the mesh coordinates
    x = mesh.coordinates()
    x[:, :] *= scaling_factor
    mesh.bounding_box_tree().build(mesh)

    # Convert subdomains to mesh function
    boundaries = MeshFunction("size_t", mesh, 2, mesh.domains())
    boundaries.set_values(boundaries.array() + 1)  # FIXME: Explain why this is necessary

    base, first_dot, rest = xml_mesh_path.name.partition('.')
    file_name_boundaries = str(xml_mesh_path.with_name(base + "_boundaries.pvd"))
    boundary_file = File(file_name_boundaries)
    boundary_file << boundaries

    domains = MeshFunction("size_t", mesh, 3, mesh.domains())
    domains.set_values(domains.array() + 1)  # in order to have fluid==1 and solid==2

    file_name_domains = str(xml_mesh_path.with_name(base + "_domains.pvd"))
    domain_file = File(file_name_domains)
    domain_file << domains

    file_name_h5_mesh = str(xml_mesh_path.with_name(base + '.h5'))
    hdf = HDF5File(mesh.mpi_comm(), file_name_h5_mesh, "w")
    hdf.write(mesh, "/mesh")
    hdf.write(boundaries, "/boundaries")
    hdf.write(domains, "/domains")


def convert_vtu_mesh_to_xdmf(file_name_vtu_mesh: Union[str, Path], file_name_xdmf_mesh: Union[str, Path]) -> None:
    """
    Convert a VTU mesh to XDMF format using meshio. This function is intended to run in serial.

    Args:
        file_name_vtu_mesh (Union[str, Path]): Path to the input VTU mesh file.
        file_name_xdmf_mesh (Union[str, Path]): Path to the output XDMF file.
    """
    # Load the VTU mesh
    vtu_mesh = meshio.read(file_name_vtu_mesh)

    # Extract cell data
    tetra_data = vtu_mesh.cell_data_dict.get("CellEntityIds", {}).get("tetra", None)
    triangle_data = vtu_mesh.cell_data_dict.get("CellEntityIds", {}).get("triangle", None)

    # Extract cell types and data
    tetra_cells = None
    triangle_cells = None
    for cell in vtu_mesh.cells:
        if cell.type == "tetra":
            tetra_cells = cell.data
        elif cell.type == "triangle":
            triangle_cells = cell.data

    # Create mesh objects
    tetra_mesh = meshio.Mesh(points=vtu_mesh.points, cells={"tetra": tetra_cells},
                             cell_data={"CellEntityIds": [tetra_data]})
    triangle_mesh = meshio.Mesh(points=vtu_mesh.points, cells=[("triangle", triangle_cells)],
                                cell_data={"CellEntityIds": [triangle_data]})

    # Define Path objects
    tetra_xdmf_path = Path(file_name_xdmf_mesh)
    triangle_xdmf_path = tetra_xdmf_path.with_name(tetra_xdmf_path.stem + '_triangle.xdmf')

    # Write the VTU mesh to XDMF format
    meshio.write(tetra_xdmf_path, tetra_mesh)
    meshio.write(triangle_xdmf_path, triangle_mesh)

    print(f"Tetra mesh XDMF file written to: {tetra_xdmf_path}")
    print(f"Triangle mesh XDMF file written to: {triangle_xdmf_path}\n")


def edge_length_evaluator(file_name_mesh: Union[str, Path], file_name_edge_length_xdmf: Union[str, Path]) -> None:
    """
    Evaluates the edge length of a mesh.

    Args:
        file_name_mesh (Union[str, Path]): Path to the XML mesh file.
        file_name_edge_length_xdmf (Union[str, Path]): Path to the output XDMF file.
    """
    file_name_mesh = Path(file_name_mesh)

    # Check if the XML mesh file exists
    if not file_name_mesh.is_file():
        raise FileNotFoundError(f"The file '{file_name_mesh}' does not exist.")
    try:
        mesh = Mesh(str(file_name_mesh))
    except RuntimeError:
        mesh = Mesh()
        with XDMFFile(str(file_name_mesh)) as xdmf:
            xdmf.read(mesh)

    mesh.init(1)
    num_cells = mesh.num_cells()
    V = FunctionSpace(mesh, "DG", 0)
    u = Function(V)
    values = np.zeros(num_cells, dtype=np.float64)

    for cell in cells(mesh):
        edges = cell.entities(1)
        value = 0
        for edge in edges:
            value += Edge(mesh, edge).length()
        values[cell.index()] = value / len(edges)

    u.vector().set_local(values)
    u.vector().apply("local")
    u.rename("edge_length", "edge_length")

    with XDMFFile(str(file_name_edge_length_xdmf)) as xdmf:
        xdmf.write_checkpoint(u, "edge_length", 0, append=False)


def check_flatten_boundary(num_inlets_outlets: int, mesh_path: Union[str, Path],
                           threshold_stdev: float = 0.001) -> None:
    """
    Check whether inlets and outlets are flat, then flatten them if necessary

    Args:
        num_inlets_outlets (int): Number of inlets and outlets in the mesh
        mesh_path (Union[str, Path]): Path to the mesh file
        threshold_stdev (float): Threshold for standard deviation of facet unit normals

    Returns:
        None
    """
    mesh_path = Path(mesh_path)
    flat_in_out_mesh_path = Path(str(mesh_path).replace(".h5", "_flat_outlet.h5"))
    # copy the mesh to a new file
    flat_in_out_mesh_path.write_bytes(mesh_path.read_bytes())

    vectorData = h5py.File(str(flat_in_out_mesh_path), "a")
    facet_ids = np.array(vectorData["boundaries/values"])
    facet_topology = vectorData["boundaries/topology"]

    fix = False
    for inlet_id in range(2, 2 + num_inlets_outlets):
        inlet_facet_ids = [i for i, x in enumerate(facet_ids) if x == inlet_id]
        inlet_facet_topology = facet_topology[inlet_facet_ids, :]
        inlet_nodes = np.unique(inlet_facet_topology.flatten())
        # pre-allocate arrays
        inlet_facet_normals = np.zeros((len(inlet_facet_ids), 3))

        # From: https://stackoverflow.com/questions/53698635/
        # how-to-define-a-plane-with-3-points-and-plot-it-in-3d
        for idx, facet in enumerate(inlet_facet_topology):
            p0 = vectorData["boundaries/coordinates"][facet[0]]
            p1 = vectorData["boundaries/coordinates"][facet[1]]
            p2 = vectorData["boundaries/coordinates"][facet[2]]

            x0, y0, z0 = p0
            x1, y1, z1 = p1
            x2, y2, z2 = p2

            ux, uy, uz = [x1 - x0, y1 - y0, z1 - z0]  # Vectors
            vx, vy, vz = [x2 - x0, y2 - y0, z2 - z0]

            # cross product of vectors defines the plane normal
            u_cross_v = [uy * vz - uz * vy, uz * vx - ux * vz, ux * vy - uy * vx]
            normal = np.array(u_cross_v)

            # Facet unit normal vector (u_normal)
            u_normal = normal / np.sqrt(
                normal[0] ** 2 + normal[1] ** 2 + normal[2] ** 2
            )

            # check if facet unit normal vector has opposite
            # direction and reverse the vector if necessary
            if idx == 0:
                u_normal_baseline = u_normal
            else:
                angle = np.arccos(
                    np.clip(np.dot(u_normal_baseline, u_normal), -1.0, 1.0)
                )
                if angle > np.pi / 2:
                    u_normal = -u_normal

            # record u_normal
            inlet_facet_normals[idx, :] = u_normal

        # Average normal and d (we will assign this later to all facets)
        normal_avg = np.mean(inlet_facet_normals, axis=0)
        inlet_coords = np.array(vectorData["boundaries/coordinates"][inlet_nodes])
        point_avg = np.mean(inlet_coords, axis=0)
        d_avg = -point_avg.dot(normal_avg)  # plane coefficient

        # Standard deviation of components of normal vector
        normal_stdev = np.std(inlet_facet_normals, axis=0)
        if np.max(normal_stdev) > threshold_stdev:  # if surface is not flat
            print(
                "Surface with ID {} is not flat: Standard deviation of facet unit\
normals is {}, greater than threshold of {}".format(
                    inlet_id, np.max(normal_stdev), threshold_stdev
                )
            )

            # Move the inlet nodes into the average inlet plane (do same for outlets)
            ArrayNames = [
                "boundaries/coordinates",
                "mesh/coordinates",
                "domains/coordinates",
            ]
            print("Moving nodes into a flat plane")
            for ArrayName in ArrayNames:
                vectorArray = vectorData[ArrayName]
                for node_id in range(len(vectorArray)):
                    if node_id in inlet_nodes:
                        # from https://stackoverflow.com/questions/9605556/
                        # how-to-project-a-point-onto-a-plane-in-3d (bobobobo)
                        node = vectorArray[node_id, :]
                        scalar_distance = node.dot(normal_avg) + d_avg
                        node_inplane = node - scalar_distance * normal_avg
                        vectorArray[node_id, :] = node_inplane
            fix = True

    if fix:
        vectorData.close()
        # Replace the original mesh file with the modified one
        mesh_path.unlink()
        mesh_path.write_bytes(flat_in_out_mesh_path.read_bytes())
        print("Changes made to the mesh file")
        flat_in_out_mesh_path.unlink()

        # overwrite Paraview files for domains and boundaries
        boundary_file = File(str(mesh_path.with_name(mesh_path.stem + "_boundaries.pvd")))
        domain_file = File(str(mesh_path.with_name(mesh_path.stem + "_domains.pvd")))

        mesh, boundaries, domains = load_mesh_and_data(mesh_path)
        boundary_file << boundaries
        domain_file << domains

    else:
        print(
            "Surface with ID {} is flat: Standard deviation of facet unit\
normals is {}, less than threshold of {}".format(
                inlet_id, np.max(normal_stdev), threshold_stdev
            ))
        vectorData.close()
        flat_in_out_mesh_path.unlink()
        print("No changes made to the mesh file")
