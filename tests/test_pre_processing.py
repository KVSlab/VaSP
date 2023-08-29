import vtk
from pathlib import Path
from dolfin import Mesh, XDMFFile
from vampy.automatedPreprocessing.preprocessing_common import read_polydata
from fsipy.automatedPreprocessing.automated_preprocessing import read_command_line, \
    run_pre_processing


def test_mesh_model_with_one_inlet():
    """
    Test meshing procedure on a specific 3D model with only one inlet.
    """
    # Define test data paths
    model_path = Path("tests/test_data/tube/tube.stl")
    mesh_path_vtu = model_path.with_suffix(".vtu")
    mesh_path_xdmf = model_path.with_suffix(".xdmf")

    # Define expected values
    expected_num_points = 3626
    expected_num_cells = 20119

    # Get default input parameters
    common_input = read_command_line(str(model_path))
    common_input.update(
        dict(
            meshing_method="diameter",
            smoothing_method="taubin",
            refine_region=False,
            coarsening_factor=1.3,
            visualize=False,
            compress_mesh=False,
            outlet_flow_extension_length=1,
            inlet_flow_extension_length=1,
        )
    )

    # Run pre processing
    run_pre_processing(**common_input)

    # Check that mesh files are created
    assert mesh_path_vtu.is_file(), f"VTU mesh file not found at {mesh_path_vtu}"
    assert mesh_path_xdmf.is_file(), f"XDMF mesh file not found at {mesh_path_xdmf}"

    # Check that mesh files are not empty and have expected sizes
    mesh_vtu = read_polydata(str(mesh_path_vtu))
    mesh_xdmf = Mesh()
    try:
        with XDMFFile(str(mesh_path_xdmf)) as xdmf_file:
            xdmf_file.read(mesh_xdmf)
    except Exception as e:
        print(f"Error reading XDMF mesh: {e}")

    assert mesh_vtu.GetNumberOfPoints() == expected_num_points, \
        f"VTU mesh has {mesh_vtu.GetNumberOfPoints()} points, expected {expected_num_points}"
    assert mesh_xdmf.num_cells() == expected_num_cells, \
        f"XDMF mesh has {mesh_xdmf.num_cells()} cells, expected {expected_num_cells}"


def test_mesh_model_with_one_inlet_and_one_outlet():
    """
    Test meshing procedure on a specific 3D model with one inlet and one outlet.
    """
    # Define test data paths
    model_path = Path("tests/test_data/cylinder/cylinder.vtp")
    mesh_path_vtu = model_path.with_suffix(".vtu")
    mesh_path_xdmf = model_path.with_suffix(".xdmf")

    # Define expected values
    expected_num_points = 2153
    expected_num_cells = 11459

    # Get default input parameters
    common_input = read_command_line(str(model_path))
    common_input.update(
        dict(
            meshing_method="diameter",
            smoothing_method="no_smooth",
            refine_region=False,
            coarsening_factor=1.3,
            visualize=False,
            compress_mesh=False,
            outlet_flow_extension_length=1,
            inlet_flow_extension_length=1,
        )
    )

    # Run pre processing
    run_pre_processing(**common_input)

    # Check that mesh files are created
    assert mesh_path_vtu.is_file(), f"VTU mesh file not found at {mesh_path_vtu}"
    assert mesh_path_xdmf.is_file(), f"XDMF mesh file not found at {mesh_path_xdmf}"

    # Check that mesh files are not empty and have expected sizes
    mesh_vtu = read_polydata(str(mesh_path_vtu))
    mesh_xdmf = Mesh()
    try:
        with XDMFFile(str(mesh_path_xdmf)) as xdmf_file:
            xdmf_file.read(mesh_xdmf)
    except Exception as e:
        print(f"Error reading XDMF mesh: {e}")

    assert mesh_vtu.GetNumberOfPoints() == expected_num_points, \
        f"VTU mesh has {mesh_vtu.GetNumberOfPoints()} points, expected {expected_num_points}"
    assert mesh_xdmf.num_cells() == expected_num_cells, \
        f"XDMF mesh has {mesh_xdmf.num_cells()} cells, expected {expected_num_cells}"


def test_mesh_model_with_one_inlet_and_two_outlets():
    """
    Test meshing procedure on a specific 3D model with one inlet and two outlets.
    """
    # Define test data paths
    model_path = Path("tests/test_data/artery/artery.stl")
    mesh_path_vtu = model_path.with_suffix(".vtu")
    mesh_path_xdmf = model_path.with_suffix(".xdmf")

    # Define expected values
    expected_num_points = 5860
    expected_num_cells = 32283

    # Get default input parameters
    common_input = read_command_line(str(model_path))
    common_input.update(
        dict(
            meshing_method="diameter",
            smoothing_method="taubin",
            refine_region=False,
            coarsening_factor=1.3,
            visualize=False,
            compress_mesh=False,
            outlet_flow_extension_length=1,
            inlet_flow_extension_length=1,
        )
    )

    # Run pre processing
    run_pre_processing(**common_input)

    # Check that mesh files are created
    assert mesh_path_vtu.is_file(), f"VTU mesh file not found at {mesh_path_vtu}"
    assert mesh_path_xdmf.is_file(), f"XDMF mesh file not found at {mesh_path_xdmf}"

    # Check that mesh files are not empty and have expected sizes
    mesh_vtu = read_polydata(str(mesh_path_vtu))
    mesh_xdmf = Mesh()
    try:
        with XDMFFile(str(mesh_path_xdmf)) as xdmf_file:
            xdmf_file.read(mesh_xdmf)
    except Exception as e:
        print(f"Error reading XDMF mesh: {e}")

    assert mesh_vtu.GetNumberOfPoints() == expected_num_points, \
        f"VTU mesh has {mesh_vtu.GetNumberOfPoints()} points, expected {expected_num_points}"
    assert mesh_xdmf.num_cells() == expected_num_cells, \
        f"XDMF mesh has {mesh_xdmf.num_cells()} cells, expected {expected_num_cells}"


def test_mesh_model_with_variable_mesh_density():
    """
    Test meshing procedure on a specific 3D model with variable mesh density.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/artery/artery.stl")
    model_path = original_model_path.with_name(original_model_path.stem + "_variable_mesh_density.stl")
    sphere_file_path = model_path.with_name("stored_" + model_path.stem + "_distance_to_sphere_spheres.vtp")
    copied_sphere_file_path = sphere_file_path.with_name(model_path.stem + "_distance_to_sphere_spheres.vtp")

    mesh_path_vtu = model_path.with_suffix(".vtu")
    mesh_path_xdmf = model_path.with_suffix(".xdmf")

    # Make copies of the original model and sphere files using pathlib
    model_path.write_text(original_model_path.read_text())
    copied_sphere_file_path.write_text(sphere_file_path.read_text())

    # Define expected values
    expected_num_points = 18116
    expected_num_cells = 103233

    # Get default input parameters
    common_input = read_command_line(str(model_path))
    common_input.update(
        dict(
            meshing_method="distancetospheres",
            meshing_parameters=[0, 0.1, 0.3, 0.5],
            smoothing_method="taubin",
            refine_region=False,
            visualize=False,
            compress_mesh=False,
            outlet_flow_extension_length=5,
            inlet_flow_extension_length=5,
        )
    )

    # Run pre processing
    run_pre_processing(**common_input)

    # Check that mesh files are created
    assert mesh_path_vtu.is_file(), f"VTU mesh file not found at {mesh_path_vtu}"
    assert mesh_path_xdmf.is_file(), f"XDMF mesh file not found at {mesh_path_xdmf}"

    # Check that mesh files are not empty and have expected sizes
    mesh_vtu = read_polydata(str(mesh_path_vtu))
    mesh_xdmf = Mesh()
    try:
        with XDMFFile(str(mesh_path_xdmf)) as xdmf_file:
            xdmf_file.read(mesh_xdmf)
    except Exception as e:
        print(f"Error reading XDMF mesh: {e}")

    assert mesh_vtu.GetNumberOfPoints() == expected_num_points, \
        f"VTU mesh has {mesh_vtu.GetNumberOfPoints()} points, expected {expected_num_points} points"
    assert mesh_xdmf.num_cells() == expected_num_cells, \
        f"XDMF mesh has {mesh_xdmf.num_cells()} cells, expected {expected_num_cells} cells"


def compute_cylinder_diameter_at_cut(mesh, point_coords, plane_normal):
    """
    Compute the diameter of a cylinder at a cut through a mesh.
    """
    # Create a vtkPlane source
    plane = vtk.vtkPlane()
    plane.SetOrigin(point_coords)
    plane.SetNormal(plane_normal)

    # Create a vtkCutter to cut the mesh with the plane
    cutter = vtk.vtkCutter()
    cutter.SetInputData(mesh)
    cutter.SetCutFunction(plane)
    cutter.Update()
    cut_output = cutter.GetOutput()

    # Get the bounds of the cut plane
    bounds = cut_output.GetBounds()
    xmin = bounds[0]
    xmax = bounds[1]

    # Calculate the diameter using the bounds of the cut plane
    diameter = abs(xmax - xmin)

    return diameter


def test_mesh_model_with_variable_solid_thickness():
    """
    Test meshing procedure on a specific 3D model with variable solid thickness.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/cylinder/cylinder.vtp")
    model_path = original_model_path.with_name(original_model_path.stem + "_variable_solid_thickness.vtp")
    sphere_file_path = model_path.with_name("stored_" + model_path.stem + "_distance_to_sphere_solid_thickness.vtp")
    copied_sphere_file_path = sphere_file_path.with_name(model_path.stem + "_distance_to_sphere_solid_thickness.vtp")

    mesh_path_vtu = model_path.with_suffix(".vtu")
    mesh_path_xdmf = model_path.with_suffix(".xdmf")

    # Make copies of the original model and sphere files using pathlib
    model_path.write_text(original_model_path.read_text())
    copied_sphere_file_path.write_text(sphere_file_path.read_text())

    # Define expected values
    expected_num_points = 5687
    expected_num_cells = 31335
    expected_diameter_at_inlet = 1.3839144706726074
    expected_diameter_at_outlet = 1.7820959687232971

    # Get default input parameters
    common_input = read_command_line(str(model_path))
    common_input.update(
        dict(
            solid_thickness="variable",
            solid_thickness_parameters=[0, 0.1, 0.2, 0.4],
            meshing_method="diameter",
            smoothing_method="no_smooth",
            refine_region=False,
            coarsening_factor=1.3,
            visualize=False,
            compress_mesh=False,
            outlet_flow_extension_length=5,
            inlet_flow_extension_length=5,
        )
    )

    # Run pre processing
    run_pre_processing(**common_input)

    # Check that mesh files are created
    assert mesh_path_vtu.is_file(), f"VTU mesh file not found at {mesh_path_vtu}"
    assert mesh_path_xdmf.is_file(), f"XDMF mesh file not found at {mesh_path_xdmf}"

    # Check that mesh files are not empty and have expected sizes
    mesh_vtu = read_polydata(str(mesh_path_vtu))
    mesh_xdmf = Mesh()
    try:
        with XDMFFile(str(mesh_path_xdmf)) as xdmf_file:
            xdmf_file.read(mesh_xdmf)
    except Exception as e:
        print(f"Error reading XDMF mesh: {e}")

    assert mesh_vtu.GetNumberOfPoints() == expected_num_points, \
        f"VTU mesh has {mesh_vtu.GetNumberOfPoints()} points, expected {expected_num_points} points"
    assert mesh_xdmf.num_cells() == expected_num_cells, \
        f"XDMF mesh has {mesh_xdmf.num_cells()} cells, expected {expected_num_cells} cells"

    # Compute diameter at inlet and outlet
    diameter_at_inlet = compute_cylinder_diameter_at_cut(mesh_vtu, [0, -3.1, 0], [0, 1, 0])
    diameter_at_outlet = compute_cylinder_diameter_at_cut(mesh_vtu, [0, 3.1, 0], [0, 1, 0])

    assert diameter_at_inlet == expected_diameter_at_inlet, \
        f"VTU mesh has diameter {diameter_at_inlet} at inlet, expected {expected_diameter_at_inlet}"
    assert diameter_at_outlet == expected_diameter_at_outlet, \
        f"VTU mesh has diameter {diameter_at_outlet} at outlet, expected {expected_diameter_at_outlet}"


if __name__ == "__main__":
    test_mesh_model_with_one_inlet()
    test_mesh_model_with_one_inlet_and_one_outlet()
    test_mesh_model_with_one_inlet_and_two_outlets()
    test_mesh_model_with_variable_mesh_density()
    test_mesh_model_with_variable_solid_thickness()
