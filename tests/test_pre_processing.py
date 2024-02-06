import subprocess
import pytest
from pathlib import Path

import vtk
from dolfin import Mesh, HDF5File, XDMFFile, FunctionSpace, Function
from vampy.automatedPreprocessing.preprocessing_common import read_polydata

from vasp.automatedPreprocessing.automated_preprocessing import read_command_line, \
    run_pre_processing

# Define test cases for testing command line options for vasp-generate-mesh script
command_line_test_cases = [
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-ra"], "--- Removing mesh and all pre-processing files"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-f", "True"], "--- Adding flow extensions"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-f", "False"], "--- Not adding flow extensions"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-sm", "laplace"], "--- Smooth surface: Laplace smoothing"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-sm", "taubin"], "--- Smooth surface: Taubin smoothing"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-sm", "no_smooth"], "--- No smoothing of surface"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-mf", "xml"], "--- Writing Dolfin file"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-mf", "hdf5"], "--- Converting XML mesh to HDF5"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-mf", "xdmf"], "--- Converting VTU mesh to XDMF"),
    (["-i", "tests/test_data/tube/tube.stl", "-sc", "0.001", "-c", "1"], "--- Scale model by factor 0.001"),
    (["-i", "tests/test_data/artery/artery.stl", "-m", "curvature", "-c", "1.6"], "Number of cells: 135660"),
    (["-i", "tests/test_data/tube/tube.stl", "-m", "diameter"], "Number of cells: 14223"),
    (["-i", "tests/test_data/tube/tube.stl", "-m", "constant", "-el", "0.4"], "Number of cells: 39886"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-fli", "3", "-flo", "7"], "Number of cells: 11552"),
    (["-i", "tests/test_data/cylinder/cylinder.vtp", "-nbf", "1", "-nbs", "3"], "Number of cells: 11550"),
    # Add more test cases as needed
]


@pytest.mark.parametrize("args, description", command_line_test_cases)
def test_vasp_mesh(args, description, tmpdir):
    # Define test data paths
    original_model_path = Path(args[1])

    # Copy the original model to tmpdir
    model_path = copy_original_model_to_tmpdir(original_model_path, tmpdir)

    # Replace the original model path in args with the model in tmpdir
    args[1] = str(model_path)

    command = ["vasp-generate-mesh", "-viz", "False", "-c", "2"] + args

    # Run the script and capture the output and return code
    result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

    # Check if the return code is 0 (indicating success)
    assert result.returncode == 0, f"Test failed: {description}\nCaptured output:\n{result.stdout}"

    # Check if the expected description is present in the output
    assert description in result.stdout, f"Test failed: {description}\nCaptured output:\n{result.stdout}"


def copy_original_model_to_tmpdir(original_model_path, tmpdir):
    """
    Copy the original model to the tmpdir.
    """
    model_path = Path(tmpdir) / original_model_path.name
    model_path.write_text(original_model_path.read_text())
    return model_path


def run_pre_processing_with_common_input(model_path, common_input):
    """
    Run pre-processing with the given common input.
    """
    run_pre_processing(**common_input)

    # Check that mesh files are created
    mesh_path_vtu = model_path.with_suffix(".vtu")
    mesh_path_hdf5 = model_path.with_suffix(".h5")
    assert mesh_path_vtu.is_file(), f"VTU mesh file not found at {mesh_path_vtu}"
    assert mesh_path_hdf5.is_file(), f"HDF5 mesh file not found at {mesh_path_hdf5}"

    # Check that mesh files are not empty
    mesh_vtu = read_polydata(str(mesh_path_vtu))
    mesh_hdf5 = Mesh()
    try:
        hdf5_file = HDF5File(mesh_hdf5.mpi_comm(), str(mesh_path_hdf5), "r")
        hdf5_file.read(mesh_hdf5, "/mesh", False)
    except Exception as e:
        print(f"Error reading HDF5 mesh: {e}")

    return model_path, mesh_vtu, mesh_hdf5


def assert_mesh_sizes(mesh_vtu, mesh_hdf5, expected_num_points, expected_num_cells):
    """
    Assert that mesh sizes match the expected values.
    """
    assert mesh_vtu.GetNumberOfPoints() == expected_num_points, \
        f"VTU mesh has {mesh_vtu.GetNumberOfPoints()} points, expected {expected_num_points}"
    assert mesh_hdf5.num_cells() == expected_num_cells, \
        f"HDF5 mesh has {mesh_hdf5.num_cells()} cells, expected {expected_num_cells}"


def test_mesh_model_with_one_inlet(tmpdir):
    """
    Test meshing procedure on a specific 3D model with only one inlet.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/tube/tube.stl")

    # Copy the original model to tmpdir
    model_path = copy_original_model_to_tmpdir(original_model_path, tmpdir)

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

    # Run pre-processing and assert mesh sizes
    model_path, mesh_vtu, mesh_hdf5 = run_pre_processing_with_common_input(model_path, common_input)
    assert_mesh_sizes(mesh_vtu, mesh_hdf5, expected_num_points, expected_num_cells)

    # Test edge length evaluation
    edge_length_mesh_path = model_path.with_name(model_path.stem + "_edge_length.xdmf")
    assert edge_length_mesh_path.is_file(), f"Edge length mesh file not found at {edge_length_mesh_path}"

    try:
        V = FunctionSpace(mesh_hdf5, "DG", 0)
        v = Function(V)
        edge_length_file = XDMFFile(str(edge_length_mesh_path))
        edge_length_file.read_checkpoint(v, "edge_length")
    except Exception as e:
        print(f"Error reading edge length mesh: {e}")

    assert v.vector().min() > 0, "Edge length mesh has negative values"


def test_mesh_model_with_one_inlet_and_one_outlet(tmpdir):
    """
    Test meshing procedure on a specific 3D model with one inlet and one outlet.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/cylinder/cylinder.vtp")

    # Copy the original model to tmpdir
    model_path = copy_original_model_to_tmpdir(original_model_path, tmpdir)

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

    # Run pre-processing and assert mesh sizes
    model_path, mesh_vtu, mesh_hdf5 = run_pre_processing_with_common_input(model_path, common_input)
    assert_mesh_sizes(mesh_vtu, mesh_hdf5, expected_num_points, expected_num_cells)


def test_mesh_model_with_one_inlet_and_two_outlets(tmpdir):
    """
    Test meshing procedure on a specific 3D model with one inlet and two outlets.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/artery/artery.stl")

    # Copy the original model to tmpdir
    model_path = copy_original_model_to_tmpdir(original_model_path, tmpdir)

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

    # Run pre-processing and assert mesh sizes
    model_path, mesh_vtu, mesh_hdf5 = run_pre_processing_with_common_input(model_path, common_input)
    assert_mesh_sizes(mesh_vtu, mesh_hdf5, expected_num_points, expected_num_cells)


def test_mesh_model_with_variable_mesh_density(tmpdir):
    """
    Test meshing procedure on a specific 3D model with variable mesh density.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/artery/artery.stl")
    sphere_file_path = original_model_path.with_name("stored_" + original_model_path.stem +
                                                     "_variable_mesh_density_distance_to_sphere_spheres.vtp")

    # Copy the original model to tmpdir
    model_path = copy_original_model_to_tmpdir(original_model_path, tmpdir)

    # Copy the sphere file to tmpdir
    copied_sphere_file_path = model_path.with_name(model_path.stem + "_distance_to_sphere_spheres.vtp")
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

    # Run pre-processing and assert mesh sizes
    model_path, mesh_vtu, mesh_hdf5 = run_pre_processing_with_common_input(model_path, common_input)
    assert_mesh_sizes(mesh_vtu, mesh_hdf5, expected_num_points, expected_num_cells)


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


def test_mesh_model_with_variable_solid_thickness(tmpdir):
    """
    Test meshing procedure on a specific 3D model with variable solid thickness.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/cylinder/cylinder.vtp")
    sphere_file_path = original_model_path.with_name("stored_" + original_model_path.stem +
                                                     "_variable_solid_thickness_distance_to_sphere_solid_thickness.vtp")

    # Copy the original model to tmpdir
    model_path = copy_original_model_to_tmpdir(original_model_path, tmpdir)

    # Copy the sphere file to tmpdir
    copied_sphere_file_path = model_path.with_name(model_path.stem + "_distance_to_sphere_solid_thickness.vtp")
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

    # Run pre processing and assert mesh sizes
    model_path, mesh_vtu, mesh_hdf5 = run_pre_processing_with_common_input(model_path, common_input)
    assert_mesh_sizes(mesh_vtu, mesh_hdf5, expected_num_points, expected_num_cells)

    # Compute diameter at inlet and outlet
    diameter_at_inlet = compute_cylinder_diameter_at_cut(mesh_vtu, [0, -3.1, 0], [0, 1, 0])
    diameter_at_outlet = compute_cylinder_diameter_at_cut(mesh_vtu, [0, 3.1, 0], [0, 1, 0])

    assert diameter_at_inlet == expected_diameter_at_inlet, \
        f"VTU mesh has diameter {diameter_at_inlet} at inlet, expected {expected_diameter_at_inlet}"
    assert diameter_at_outlet == expected_diameter_at_outlet, \
        f"VTU mesh has diameter {diameter_at_outlet} at outlet, expected {expected_diameter_at_outlet}"


def test_xdmf_mesh_format(tmpdir):
    """
    Test meshing procedure with generated mesh in XDMF format.
    """
    # Define test data paths
    original_model_path = Path("tests/test_data/cylinder/cylinder.vtp")
    model_path = Path(tmpdir) / original_model_path.name
    mesh_path_vtu = model_path.with_suffix(".vtu")
    mesh_path_xdmf = model_path.with_suffix(".xdmf")

    # Copy the original model to tmpdir
    model_path.write_text(original_model_path.read_text())

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
            mesh_format="xdmf",
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
