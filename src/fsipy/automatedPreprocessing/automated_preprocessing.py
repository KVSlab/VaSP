#!/usr/bin/env python

import argparse
import sys
from os import remove, path

import numpy as np
from morphman import get_uncapped_surface, write_polydata, get_parameters, vtk_clean_polydata, \
    vtk_triangulate_surface, write_parameters, vmtk_cap_polydata, compute_centerlines, get_centerline_tolerance, \
    get_vtk_point_locator, extract_single_line, vtk_merge_polydata, get_point_data_array, smooth_voronoi_diagram, \
    create_new_surface, compute_centers, vmtk_smooth_surface, str2bool, vmtk_compute_voronoi_diagram, \
    prepare_output_surface, vmtk_compute_geometric_features

from vampy.automatedPreprocessing import ToolRepairSTL
from vampy.automatedPreprocessing.preprocessing_common import read_polydata, get_centers_for_meshing, \
    dist_sphere_diam, dist_sphere_curvature, dist_sphere_constant, get_regions_to_refine, add_flow_extension, \
    write_mesh, mesh_alternative, find_boundaries, compute_flow_rate, setup_model_network, \
    radiusArrayName, scale_surface, get_furtest_surface_point, check_if_closed_surface
from vampy.automatedPreprocessing.simulate import run_simulation
from vampy.automatedPreprocessing.visualize import visualize_model

from fsipy.automatedPreprocessing.preprocessing_common import generate_mesh, distance_to_spheres_solid_thickness, \
    dist_sphere_spheres, convert_xml_mesh_to_hdf5, convert_vtu_mesh_to_xdmf, edge_length_evaluator


def run_pre_processing(input_model, verbose_print, smoothing_method, smoothing_factor, smoothing_iterations,
                       meshing_method, refine_region, is_atrium, add_flow_extensions, visualize, config_path,
                       coarsening_factor, inlet_flow_extension_length, outlet_flow_extension_length,
                       number_of_sublayers_fluid, number_of_sublayers_solid, edge_length,
                       region_points, compress_mesh, scale_factor, resampling_step, meshing_parameters,
                       remove_all, solid_thickness, solid_thickness_parameters, mesh_format,
                       flow_rate_factor):
    """
    Automatically generate mesh of surface model in .vtu and .xml format, including prescribed
    flow rates at inlet and outlet based on flow network model.

    Runs simulation of meshed case on a remote ssh server if server configuration is provided.

    Args:
        input_model (str): Name of case
        verbose_print (bool): Toggles verbose mode
        smoothing_method (str): Method for surface smoothing
        smoothing_factor (float): Smoothing factor of Voronoi smoothing
        smoothing_iterations (int): Number of smoothing iterations for Taubin and Laplace smoothing
        meshing_method (str): Determines what the density of the volumetric mesh depends upon
        refine_region (bool): Refines selected region of input if True
        is_atrium (bool): Determines whether this is an atrium case
        add_flow_extensions (bool): Adds flow extensions to mesh if True
        visualize (bool): Visualize resulting surface model with flow rates
        config_path (str): Path to configuration file for remote simulation
        coarsening_factor (float): Refine or coarsen the standard mesh size with given factor
        inlet_flow_extension_length (float): Factor defining length of flow extensions at the inlet(s)
        outlet_flow_extension_length (float): Factor defining length of flow extensions at the outlet(s)
        number_of_sublayers_fluid (int): Number of sublayers for fluid
        number_of_sublayers_solid (int): Number of sublayers for solid
        edge_length (float): Edge length used for meshing with constant element size
        region_points (list): User defined points to define which region to refine
        compress_mesh (bool): Compresses finalized mesh if True
        scale_factor (float): Scale input model by this factor
        resampling_step (float): Float value determining the resampling step for centerline computations, in [m]
        meshing_parameters (list): Parameters for meshing method 'distancetospheres'
        remove_all (bool): Remove mesh and all pre-processing files
        solid_thickness (str): Constant or variable mesh thickness
        solid_thickness_parameters (list): Specify parameters for solid thickness
        mesh_format (str): Specify the format for the generated mesh
        flow_rate_factor (float): Flow rate factor
    """
    # Get paths
    case_name = input_model.rsplit(path.sep, 1)[-1].rsplit('.')[0]
    dir_path = input_model.rsplit(path.sep, 1)[0]
    print("\n--- Working on case:", case_name, "\n")

    # Naming conventions
    base_path = path.join(dir_path, case_name)
    file_name_centerlines = base_path + "_centerlines.vtp"
    file_name_refine_region_centerlines = base_path + "_refine_region_centerline.vtp"
    file_name_region_centerlines = base_path + "_sac_centerline_{}.vtp"
    file_name_distance_to_sphere_diam = base_path + "_distance_to_sphere_diam.vtp"
    file_name_distance_to_sphere_const = base_path + "_distance_to_sphere_const.vtp"
    file_name_distance_to_sphere_curv = base_path + "_distance_to_sphere_curv.vtp"
    file_name_distance_to_sphere_spheres = base_path + "_distance_to_sphere_spheres.vtp"
    file_name_distance_to_sphere_solid_thickness = base_path + "_distance_to_sphere_solid_thickness.vtp"
    file_name_parameters = base_path + "_info.json"
    file_name_probe_points = base_path + "_probe_point.json"
    file_name_voronoi = base_path + "_voronoi.vtp"
    file_name_voronoi_smooth = base_path + "_voronoi_smooth.vtp"
    file_name_voronoi_surface = base_path + "_voronoi_surface.vtp"
    file_name_surface_smooth = base_path + "_smooth.vtp"
    file_name_model_flow_ext = base_path + "_flowext.vtp"
    file_name_clipped_model = base_path + "_clippedmodel.vtp"
    file_name_flow_centerlines = base_path + "_flow_cl.vtp"
    file_name_surface_name = base_path + "_remeshed_surface.vtp"
    file_name_xml_mesh = base_path + ".xml"
    file_name_vtu_mesh = base_path + ".vtu"
    file_name_xdmf_mesh = base_path + ".xdmf"
    file_name_edge_length_xdmf = base_path + "_edge_length.xdmf"
    region_centerlines = None

    if remove_all:
        print("--- Removing mesh and all pre-processing files\n")
        files_to_remove = [
            file_name_centerlines, file_name_refine_region_centerlines, file_name_region_centerlines,
            file_name_distance_to_sphere_diam, file_name_distance_to_sphere_const, file_name_distance_to_sphere_curv,
            file_name_distance_to_sphere_spheres, file_name_distance_to_sphere_solid_thickness,
            file_name_parameters, file_name_probe_points,
            file_name_voronoi, file_name_voronoi_smooth, file_name_voronoi_surface, file_name_surface_smooth,
            file_name_model_flow_ext, file_name_clipped_model, file_name_flow_centerlines, file_name_surface_name,
            file_name_xml_mesh, file_name_vtu_mesh, file_name_xdmf_mesh,
        ]
        for file in files_to_remove:
            if path.exists(file):
                remove(file)

    # Open the surface file.
    print("--- Load model file\n")
    surface = read_polydata(input_model)

    # Scale surface
    if scale_factor is not None:
        surface = scale_surface(surface, scale_factor)

    # Check if surface is closed and uncapps model if True
    is_capped = check_if_closed_surface(surface)
    if is_capped:
        if not path.isfile(file_name_clipped_model):
            print("--- Clipping the models inlets and outlets.\n")
            # Value of gradients_limit should be generally low, to detect flat surfaces corresponding
            # to closed boundaries. Area_limit will set an upper limit of the detected area, may vary between models.
            # The circleness_limit parameters determines the detected regions' similarity to a circle, often assumed
            # to be close to a circle.
            surface = get_uncapped_surface(surface, gradients_limit=0.01, area_limit=20, circleness_limit=5)
            write_polydata(surface, file_name_clipped_model)
        else:
            surface = read_polydata(file_name_clipped_model)

    # Get model parameters
    parameters = get_parameters(base_path)

    if "check_surface" not in parameters.keys():
        surface = vtk_clean_polydata(surface)
        surface = vtk_triangulate_surface(surface)

        # Check the mesh if there is redundant nodes or NaN triangles.
        ToolRepairSTL.surfaceOverview(surface)
        ToolRepairSTL.foundAndDeleteNaNTriangles(surface)
        surface = ToolRepairSTL.cleanTheSurface(surface)
        foundNaN = ToolRepairSTL.foundAndDeleteNaNTriangles(surface)
        if foundNaN:
            raise RuntimeError(("There is an issue with the surface. "
                                "Nan coordinates or some other shenanigans."))
        else:
            parameters["check_surface"] = True
            write_parameters(parameters, base_path)

    # Create a capped version of the surface
    capped_surface = vmtk_cap_polydata(surface)

    # Get centerlines
    print("--- Get centerlines\n")
    inlet, outlets = get_centers_for_meshing(surface, is_atrium, base_path)
    has_outlet = len(outlets) != 0

    # Get point the furthest away inlet when only one boundary
    if not has_outlet:
        outlets = get_furtest_surface_point(inlet, surface)

    source = outlets if is_atrium else inlet
    target = inlet if is_atrium else outlets

    centerlines, voronoi, _ = compute_centerlines(source, target, file_name_centerlines, capped_surface,
                                                  resampling=resampling_step)
    tol = get_centerline_tolerance(centerlines)

    # Get 'center' and 'radius' of the regions(s)
    region_center = []
    misr_max = []

    if refine_region:
        regions = get_regions_to_refine(capped_surface, region_points, base_path)
        for i in range(len(regions) // 3):
            print("--- Region to refine ({}): {:.3f} {:.3f} {:.3f}"
                  .format(i + 1, regions[3 * i], regions[3 * i + 1], regions[3 * i + 2]))

        centerline_region, _, _ = compute_centerlines(source, regions, file_name_refine_region_centerlines,
                                                      capped_surface, resampling=resampling_step)

        # Extract the region centerline
        refine_region_centerline = []
        info = get_parameters(base_path)
        num_anu = info["number_of_regions"]

        # Compute mean distance between points
        for i in range(num_anu):
            if not path.isfile(file_name_region_centerlines.format(i)):
                line = extract_single_line(centerline_region, i)
                locator = get_vtk_point_locator(centerlines)
                for j in range(line.GetNumberOfPoints() - 1, 0, -1):
                    point = line.GetPoints().GetPoint(j)
                    ID = locator.FindClosestPoint(point)
                    tmp_point = centerlines.GetPoints().GetPoint(ID)
                    dist = np.sqrt(np.sum((np.asarray(point) - np.asarray(tmp_point)) ** 2))
                    if dist <= tol:
                        break

                tmp = extract_single_line(line, 0, start_id=j)
                write_polydata(tmp, file_name_region_centerlines.format(i))

                # List of VtkPolyData sac(s) centerline
                refine_region_centerline.append(tmp)

            else:
                refine_region_centerline.append(read_polydata(file_name_region_centerlines.format(i)))

        # Merge the sac centerline
        region_centerlines = vtk_merge_polydata(refine_region_centerline)

        for region in refine_region_centerline:
            region_factor = 0.9 if is_atrium else 0.5
            region_center.append(region.GetPoints().GetPoint(int(region.GetNumberOfPoints() * region_factor)))
            tmp_misr = get_point_data_array(radiusArrayName, region)
            misr_max.append(tmp_misr.max())

    # Smooth surface
    if smoothing_method == "voronoi":
        print("--- Smooth surface: Voronoi smoothing\n")
        if not path.isfile(file_name_surface_smooth):
            # Get Voronoi diagram
            if not path.isfile(file_name_voronoi):
                voronoi = vmtk_compute_voronoi_diagram(capped_surface, file_name_voronoi)
                write_polydata(voronoi, file_name_voronoi)
            else:
                voronoi = read_polydata(file_name_voronoi)

            # Get smooth Voronoi diagram
            if not path.isfile(file_name_voronoi_smooth):
                voronoi_smoothed = smooth_voronoi_diagram(voronoi, centerlines, smoothing_factor,
                                                          no_smooth_cl=region_centerlines)
                write_polydata(voronoi_smoothed, file_name_voronoi_smooth)
            else:
                voronoi_smoothed = read_polydata(file_name_voronoi_smooth)

            # Create new surface from the smoothed Voronoi
            surface_smoothed = create_new_surface(voronoi_smoothed)

            # Uncapp the surface
            surface_uncapped = prepare_output_surface(surface_smoothed, surface, centerlines, file_name_voronoi_surface,
                                                      test_merge=True)

            # Check if there has been added new outlets
            num_outlets = centerlines.GetNumberOfLines()
            inlets, outlets = compute_centers(surface_uncapped)
            num_outlets_after = len(outlets) // 3

            if num_outlets != num_outlets_after:
                write_polydata(surface, file_name_surface_smooth)
                print(("ERROR: Automatic clipping failed. You have to open {} and " +
                       "manually clipp the branch which still is capped. " +
                       "Overwrite the current {} and restart the script.").format(
                    file_name_surface_smooth, file_name_surface_smooth))
                sys.exit(0)

            surface = surface_uncapped

            # Smoothing to improve the quality of the elements
            surface = vmtk_smooth_surface(surface, "laplace", iterations=200)

            # Write surface
            write_polydata(surface, file_name_surface_smooth)

        else:
            surface = read_polydata(file_name_surface_smooth)

    elif smoothing_method in ["laplace", "taubin"]:
        print("--- Smooth surface: {} smoothing\n".format(smoothing_method.capitalize()))
        if not path.isfile(file_name_surface_smooth):
            surface = vmtk_smooth_surface(surface, smoothing_method, iterations=smoothing_iterations, passband=0.1,
                                          relaxation=0.01)

            # Save the smoothed surface
            write_polydata(surface, file_name_surface_smooth)
        else:
            surface = read_polydata(file_name_surface_smooth)

    elif smoothing_method == "no_smooth" or None:
        print("--- No smoothing of surface\n")

    # Add flow extensions
    if add_flow_extensions:
        if not path.isfile(file_name_model_flow_ext):
            print("--- Adding flow extensions\n")
            # Add extension normal on boundary for atrium models
            extension = "centerlinedirection" if is_atrium else "boundarynormal"
            if is_atrium:
                # Flip lengths if model is atrium
                inlet_flow_extension_length, outlet_flow_extension_length = \
                    outlet_flow_extension_length, inlet_flow_extension_length

            # Add extensions to inlet (artery)
            surface_extended = add_flow_extension(surface, centerlines, is_inlet=True,
                                                  extension_length=inlet_flow_extension_length,
                                                  extension_mode=extension)

            # Add extensions to outlets (artery)
            surface_extended = add_flow_extension(surface_extended, centerlines, is_inlet=False,
                                                  extension_length=outlet_flow_extension_length)

            surface_extended = vmtk_smooth_surface(surface_extended, "laplace", iterations=200)
            write_polydata(surface_extended, file_name_model_flow_ext)
        else:
            surface_extended = read_polydata(file_name_model_flow_ext)
    else:
        surface_extended = surface

    # Capp surface with flow extensions
    capped_surface = vmtk_cap_polydata(surface_extended)

    # Get new centerlines with the flow extensions
    if add_flow_extensions:
        if not path.isfile(file_name_flow_centerlines):
            print("--- Compute the model centerlines with flow extension.\n")
            # Compute the centerlines.
            if has_outlet:
                inlet, outlets = get_centers_for_meshing(surface_extended, is_atrium, base_path,
                                                         use_flow_extensions=True)
            else:
                inlet, _ = get_centers_for_meshing(surface_extended, is_atrium, base_path, use_flow_extensions=True)
            # Flip outlets and inlets for atrium models
            source = outlets if is_atrium else inlet
            target = inlet if is_atrium else outlets
            centerlines, _, _ = compute_centerlines(source, target, file_name_flow_centerlines, capped_surface,
                                                    resampling=resampling_step)

        else:
            centerlines = read_polydata(file_name_flow_centerlines)

    # Clip centerline if only one inlet to avoid refining model surface
    if not has_outlet:
        line = extract_single_line(centerlines, 0)
        line = vmtk_compute_geometric_features(line, smooth=False)

        # Clip centerline where Frenet Tangent is constant
        n = get_point_data_array("FrenetTangent", line, k=3)
        n_diff = np.linalg.norm(np.cross(n[1:], n[:-1]), axis=1)
        n_id = n_diff[::-1].argmax()
        centerlines = extract_single_line(centerlines, 0, end_id=centerlines.GetNumberOfPoints() - n_id - 1)

    # Choose input for the mesh
    print("--- Computing distance to sphere\n")
    if meshing_method == "constant":
        if not path.isfile(file_name_distance_to_sphere_const):
            distance_to_sphere = dist_sphere_constant(surface_extended, centerlines, region_center, misr_max,
                                                      file_name_distance_to_sphere_const, edge_length)
        else:
            distance_to_sphere = read_polydata(file_name_distance_to_sphere_const)

    elif meshing_method == "curvature":
        if not path.isfile(file_name_distance_to_sphere_curv):
            distance_to_sphere = dist_sphere_curvature(surface_extended, centerlines, region_center, misr_max,
                                                       file_name_distance_to_sphere_curv, coarsening_factor)
        else:
            distance_to_sphere = read_polydata(file_name_distance_to_sphere_curv)
    elif meshing_method == "diameter":
        if not path.isfile(file_name_distance_to_sphere_diam):
            distance_to_sphere = dist_sphere_diam(surface_extended, centerlines, region_center, misr_max,
                                                  file_name_distance_to_sphere_diam, coarsening_factor)
        else:
            distance_to_sphere = read_polydata(file_name_distance_to_sphere_diam)
    elif meshing_method == "distancetospheres":
        if not path.isfile(file_name_distance_to_sphere_spheres):
            if len(meshing_parameters) == 4:
                distance_to_sphere = dist_sphere_spheres(surface_extended, file_name_distance_to_sphere_spheres,
                                                         *meshing_parameters)
            else:
                print("ERROR: Invalid parameters for meshing method 'distancetospheres'. This should be " +
                      "given as four parameters: 'offset', 'scale', 'min' and 'max.")
                sys.exit(-1)
        else:
            distance_to_sphere = read_polydata(file_name_distance_to_sphere_spheres)

    # Get solid thickness
    if solid_thickness == 'variable':
        if not path.isfile(file_name_distance_to_sphere_solid_thickness):
            if len(solid_thickness_parameters) == 4:
                distance_offset, distance_scale, min_distance, max_distance = solid_thickness_parameters
                distance_to_sphere = distance_to_spheres_solid_thickness(distance_to_sphere,
                                                                         file_name_distance_to_sphere_solid_thickness,
                                                                         distance_offset, distance_scale,
                                                                         min_distance, max_distance)
            elif len(solid_thickness_parameters) == 0:
                distance_to_sphere = distance_to_spheres_solid_thickness(distance_to_sphere,
                                                                         file_name_distance_to_sphere_solid_thickness)
            else:
                print("ERROR: Invalid parameters for variable solid thickness. This should be " +
                      "given as four parameters: 'offset', 'scale', 'min' and 'max.")
                sys.exit(-1)
        else:
            distance_to_sphere = read_polydata(file_name_distance_to_sphere_solid_thickness)
    else:
        if len(solid_thickness_parameters) != 1 or solid_thickness_parameters[0] <= 0:
            print("ERROR: Invalid parameter for constant solid thickness. This should be a " +
                  "single number greater than zero.")
            sys.exit(-1)

    # Compute mesh
    if not path.isfile(file_name_vtu_mesh):
        print("--- Computing mesh\n")
        try:
            mesh, remeshed_surface = generate_mesh(distance_to_sphere,
                                                   number_of_sublayers_fluid,
                                                   number_of_sublayers_solid,
                                                   solid_thickness,
                                                   solid_thickness_parameters)
        except Exception:
            distance_to_sphere = mesh_alternative(distance_to_sphere)
            mesh, remeshed_surface = generate_mesh(distance_to_sphere,
                                                   number_of_sublayers_fluid,
                                                   number_of_sublayers_solid,
                                                   solid_thickness,
                                                   solid_thickness_parameters)

        assert mesh.GetNumberOfPoints() > 0, "No points in mesh, try to remesh."
        assert remeshed_surface.GetNumberOfPoints() > 0, "No points in surface mesh, try to remesh."

        if mesh.GetNumberOfPoints() < remeshed_surface.GetNumberOfPoints():
            print("--- An error occurred during meshing. Will attempt to re-mesh \n")
            mesh, remeshed_surface = generate_mesh(distance_to_sphere,
                                                   number_of_sublayers_fluid,
                                                   number_of_sublayers_solid,
                                                   solid_thickness,
                                                   solid_thickness_parameters)

        if mesh_format in ("xml", "hdf5"):
            write_mesh(compress_mesh, file_name_surface_name, file_name_vtu_mesh, file_name_xml_mesh,
                       mesh, remeshed_surface)

            # Add .gz to XML mesh file if compressed
            if compress_mesh:
                file_name_xml_mesh = file_name_xml_mesh + ".gz"
        else:
            # Write mesh in VTU format
            write_polydata(remeshed_surface, file_name_surface_name)
            write_polydata(mesh, file_name_vtu_mesh)
    else:
        mesh = read_polydata(file_name_vtu_mesh)

    if mesh_format == "hdf5":
        print("--- Converting XML mesh to HDF5\n")
        convert_xml_mesh_to_hdf5(file_name_xml_mesh)
        # Evaluate edge length for inspection
        edge_length_evaluator(file_name_xml_mesh, file_name_edge_length_xdmf)
    elif mesh_format == "xdmf":
        convert_vtu_mesh_to_xdmf(file_name_vtu_mesh, file_name_xdmf_mesh)
        # Evaluate edge length for inspection
        edge_length_evaluator(file_name_xdmf_mesh, file_name_edge_length_xdmf)

    network, probe_points = setup_model_network(centerlines, file_name_probe_points, region_center, verbose_print)

    # Load updated parameters following meshing
    parameters = get_parameters(base_path)

    print("--- Computing flow rates and flow split, and setting boundary IDs\n")
    mean_inflow_rate = compute_flow_rate(is_atrium, inlet, parameters, flow_rate_factor)

    find_boundaries(base_path, mean_inflow_rate, network, mesh, verbose_print, is_atrium)

    # Display the flow split at the outlets, inlet flow rate, and probes.
    if visualize:
        print("--- Visualizing flow split at outlets, inlet flow rate, and probes in VTK render window. ")
        print("--- Press 'q' inside the render window to exit.")
        visualize_model(network.elements, probe_points, surface_extended, mean_inflow_rate)

    # Start simulation though ssh, without password
    if config_path is not None:
        print("--- Uploading mesh and simulation files to cluster. Queueing simulation and post-processing.")
        run_simulation(config_path, dir_path, case_name)

    print("--- Removing unused pre-processing files")
    files_to_remove = [
        file_name_centerlines, file_name_refine_region_centerlines, file_name_region_centerlines,
        file_name_distance_to_sphere_diam, file_name_distance_to_sphere_const, file_name_distance_to_sphere_curv,
        file_name_distance_to_sphere_spheres, file_name_distance_to_sphere_solid_thickness,
        file_name_voronoi, file_name_voronoi_smooth, file_name_voronoi_surface, file_name_surface_smooth,
        file_name_model_flow_ext, file_name_clipped_model, file_name_flow_centerlines, file_name_surface_name
    ]
    for file in files_to_remove:
        if path.exists(file):
            remove(file)


def read_command_line(input_path=None):
    """
    Read arguments from commandline and return all values in a dictionary.
    If input_path is not None, then do not parse command line, but
    only return default values.

        Args:
            input_path (str): Input file path, positional argument with default None.
    """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter,
                                     description="Automated pre-processing for vascular modeling.")

    # Add common arguments
    required = input_path is None
    v = parser.add_mutually_exclusive_group(required=False)
    v.add_argument('-v', '--verbosity',
                   dest='verbosity',
                   action='store_true',
                   default=False,
                   help="Activates the verbose mode.")

    parser.add_argument('-i', '--input-model',
                        type=str,
                        required=True,
                        help="Path to input model containing the 3D model. Expected format is VTK/VTP or STL.")

    parser.add_argument('-cm', '--compress-mesh',
                        type=str2bool,
                        required=False,
                        default=True,
                        help="Compress output mesh after generation.")

    parser.add_argument('-sm', '--smoothing-method',
                        type=str,
                        required=False,
                        default="no_smooth",
                        choices=["voronoi", "no_smooth", "laplace", "taubin"],
                        help="Determines smoothing method for surface smoothing. For Voronoi smoothing you can " +
                             "control the smoothing factor with --smoothing-factor (default = 0.25). For Laplace " +
                             "and Taubin smoothing, you can controll the amount of smoothing iterations with " +
                             "--smothing-iterations (default = 800).")

    parser.add_argument('-c', '--coarsening-factor',
                        type=float,
                        required=False,
                        default=1.0,
                        help="Refine or coarsen the standard mesh size. The higher the value the coarser the mesh.")

    parser.add_argument('-sf', '--smoothing-factor',
                        type=float,
                        required=False,
                        default=0.25,
                        help="Smoothing factor for Voronoi smoothing, removes all spheres which" +
                             " has a radius < MISR*(1-0.25), where MISR varying along the centerline.")

    parser.add_argument('-si', '--smoothing-iterations',
                        type=int,
                        required=False,
                        default=800,
                        help="Number of smoothing iterations for Laplace and Taubin type smoothing.")

    parser.add_argument('-m', '--meshing-method',
                        type=str,
                        choices=["diameter", "curvature", "constant", "distancetospheres"],
                        default="diameter",
                        help="Determines method of meshing. The method 'constant' is supplied with a constant edge " +
                             "length controlled by the -el argument, resulting in a constant density mesh. " +
                             "The 'curvature' method and 'diameter' method produces a variable density mesh," +
                             " based on the surface curvature and the distance from the " +
                             "centerline to the surface, respectively. The 'distancetospheres' method allows to " +
                             "place spheres where the surface is pointing by pressing 'space'. By pressing 'd', the " +
                             "surface is coloured by the distance to the spheres. By pressing 'a', a scaling " +
                             "function can be specified by four parameters: 'offset', 'scale', 'min' and 'max'. " +
                             "These parameters for the scaling function can also be controlled by the -mp argument.")

    parser.add_argument('-el', '--edge-length',
                        default=None,
                        type=float,
                        help="Characteristic edge length used for the 'constant' meshing method.")

    refine_region = parser.add_mutually_exclusive_group(required=False)
    refine_region.add_argument('-r', '--refine-region',
                               action='store_true',
                               default=False,
                               help="Determine whether or not to refine a specific region of " +
                                    "the input model")

    parser.add_argument('-rp', '--region-points',
                        type=float,
                        nargs="+",
                        default=None,
                        help="If -r or --refine-region is True, the user can provide the point(s)"
                             " which defines the regions to refine. " +
                             "Example providing the points (0.1, 5.0, -1) and (1, -5.2, 3.21):" +
                             " --region-points 0.1 5 -1 1 5.24 3.21")

    atrium = parser.add_mutually_exclusive_group(required=False)
    atrium.add_argument('-at', '--is-atrium',
                        action="store_true",
                        default=False,
                        help="Determine whether or not the model is an atrium model.")

    parser.add_argument('-f', '--add-flowextensions',
                        default=True,
                        type=str2bool,
                        help="Add flow extensions to the model.")

    parser.add_argument('-fli', '--inlet-flowextension',
                        default=5,
                        type=float,
                        help="Length of flow extensions at inlet(s).")

    parser.add_argument('-flo', '--outlet-flowextension',
                        default=5,
                        type=float,
                        help="Length of flow extensions at outlet(s).")

    parser.add_argument('-nbf', '--number-of-sublayers-fluid',
                        default=2,
                        type=int,
                        help="Number of sublayers in the fluid domain.")

    parser.add_argument('-nbs', '--number-of-sublayers-solid',
                        default=2,
                        type=int,
                        help="Number of sublayers in the solid domain.")

    parser.add_argument('-viz', '--visualize',
                        default=True,
                        type=str2bool,
                        help="Visualize surface, inlet, outlet and probes after meshing.")

    parser.add_argument('-cp', '--config-path',
                        type=str,
                        default=None,
                        help='Path to configuration file for remote simulation. ' +
                             'See ssh_config.json for details')

    parser.add_argument('-bl', '--add-boundary-layer',
                        default=True,
                        type=str2bool,
                        help="Adds boundary layers along geometry wall if true.")

    parser.add_argument('-sc', '--scale-factor',
                        default=None,
                        type=float,

                        help="Scale input model by this factor. Used to scale model to [mm].")

    parser.add_argument('-rs', '--resampling-step',
                        default=0.1,
                        type=float,
                        help="Resampling step used to resample centerline in [m].")

    parser.add_argument('-mp', '--meshing-parameters',
                        type=float,
                        nargs="+",
                        default=[0, 0.1, 0.4, 0.6],
                        help="Parameters for meshing method 'distancetospheres'. This should be given as " +
                             "four numbers for the distancetosphere scaling function: 'offset', 'scale', 'min' " +
                             "and 'max'. For example --meshing-parameters 0 0.1 0.3 0.4")

    remove_all = parser.add_mutually_exclusive_group(required=False)
    remove_all.add_argument('-ra', '--remove-all',
                            action="store_true",
                            default=False,
                            help="Remove mesh and all pre-processing files.")

    parser.add_argument('-st', '--solid-thickness',
                        type=str,
                        choices=["constant", "variable"],
                        default="constant",
                        help="Determines whether to use constant or variable thickness for the solid. " +
                             "Use --solid-thickness-parameters to adjust distancetospheres parameters " +
                             "when using variable thickness.")

    parser.add_argument('-stp', '--solid-thickness-parameters',
                        type=float,
                        nargs="+",
                        default=[0.3],
                        help="Parameters for solid thickness. For constant solid thickness, this should be " +
                             "given as a single float. For 'variable' solid thickness, this should be given as " +
                             "four floats for the distancetosphere scaling function: 'offset', 'scale', 'min' " +
                             "and 'max'. For example --solid-thickness-parameters 0 0.1 0.25 0.3")

    parser.add_argument('-mf', '--mesh-format',
                        type=str,
                        choices=["xml", "hdf5", "xdmf"],
                        default="hdf5",
                        help="Specify the format for the generated mesh. Available options: 'xml', 'hdf5', 'xdmf'.")

    parser.add_argument('-fr', '--flow-rate-factor',
                        default=0.31,
                        type=float,
                        help="Flow rate factor.")

    # Parse path to get default values
    if required:
        args = parser.parse_args()
    else:
        args = parser.parse_args(["-i" + input_path])

    if args.meshing_method == "constant" and args.edge_length is None:
        raise ValueError("ERROR: Please provide the edge length for uniform density meshing using --edge-length.")

    if args.refine_region and args.region_points is not None:
        if len(args.region_points) % 3 != 0:
            raise ValueError("ERROR: Please provide the region points as a multiple of 3.")

    if args.verbosity:
        print()
        print("--- VERBOSE MODE ACTIVATED ---")

        def verbose_print(*args):
            for arg in args:
                print(arg, end=' ')
                print()
    else:
        def verbose_print(*args):
            return None

    verbose_print(args)

    return dict(input_model=args.input_model, verbose_print=verbose_print, smoothing_method=args.smoothing_method,
                smoothing_factor=args.smoothing_factor, smoothing_iterations=args.smoothing_iterations,
                meshing_method=args.meshing_method, refine_region=args.refine_region, is_atrium=args.is_atrium,
                add_flow_extensions=args.add_flowextensions, config_path=args.config_path, edge_length=args.edge_length,
                coarsening_factor=args.coarsening_factor, inlet_flow_extension_length=args.inlet_flowextension,
                number_of_sublayers_fluid=args.number_of_sublayers_fluid,
                number_of_sublayers_solid=args.number_of_sublayers_solid, visualize=args.visualize,
                region_points=args.region_points, compress_mesh=args.compress_mesh,
                outlet_flow_extension_length=args.outlet_flowextension,
                scale_factor=args.scale_factor, resampling_step=args.resampling_step,
                meshing_parameters=args.meshing_parameters, remove_all=args.remove_all,
                solid_thickness=args.solid_thickness, solid_thickness_parameters=args.solid_thickness_parameters,
                mesh_format=args.mesh_format, flow_rate_factor=args.flow_rate_factor)


def main_meshing():
    run_pre_processing(**read_command_line())


if __name__ == "__main__":
    run_pre_processing(**read_command_line())
