from pathlib import Path

import pytest
import numpy as np
from dolfin import Mesh, cpp

from vasp.automatedPreprocessing.automated_preprocessing import read_command_line, \
    run_pre_processing
from vasp.simulations.simulation_common import load_mesh_and_data, load_mesh_info, \
    load_probe_points


@pytest.fixture(scope="function")
def temporary_hdf5_file(tmpdir):
    """
    Fixture for generating a temporary HDF5 file path with a mesh for testing purposes.
    """
    # Define the path to the generated mesh
    original_model_path = Path("tests/test_data/artery/artery.stl")
    model_path = Path(tmpdir) / original_model_path.name
    mesh_path_hdf5 = model_path.with_suffix(".h5")

    # Make a copy of the original model
    model_path.write_text(original_model_path.read_text())

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

    # Run pre processing to generate the mesh
    run_pre_processing(**common_input)

    yield mesh_path_hdf5  # Provide the temporary file path as a fixture


def test_load_mesh_and_data(temporary_hdf5_file):
    """
    Test the load_mesh_and_data function for loading mesh and data from an HDF5 file.
    """
    # Load the mesh and data from the temporary HDF5 file
    mesh, boundaries, domains = load_mesh_and_data(temporary_hdf5_file)

    # Define expected values
    expected_num_vertices = 5860
    expected_num_cells = 32283

    # Check if the loaded mesh is an instance of Mesh
    assert isinstance(mesh, Mesh), \
        f"Expected mesh to be an instance of Mesh, but got {type(mesh)}."

    # Check if the number of vertices and cells in the loaded mesh matches the expected values
    assert mesh.num_vertices() == expected_num_vertices, \
        f"Mesh has {mesh.num_vertices()} vertices, expected {expected_num_vertices}"
    assert mesh.num_cells() == expected_num_cells, \
        f"Mesh has {mesh.num_cells()} cells, expected {expected_num_cells}"

    # Check if the loaded boundaries and domains are of type dolfin.cpp.mesh.MeshFunctionSizet
    assert isinstance(boundaries, cpp.mesh.MeshFunctionSizet), \
        f"Expected boundaries to be an instance of MeshFunctionSizet, but got {type(boundaries)}."
    assert isinstance(domains, cpp.mesh.MeshFunctionSizet), \
        f"Expected domains to be an instance of MeshFunctionSizet, but got {type(domains)}."

    # Check if the loaded boundaries have non-zero data
    assert boundaries.array().max() > 0, \
        "Loaded boundaries have zero data, expected non-zero data."

    # Check if the loaded domains have non-zero data
    assert domains.array().max() > 0, \
        "Loaded domains have zero data, expected non-zero data."

    # Check if the number of boundaries and domains match the mesh topology
    expected_num_boundaries = mesh.num_faces()  # Assuming each face is a boundary
    expected_num_domains = mesh.num_cells()  # Assuming each cell is a domain

    assert boundaries.size() == expected_num_boundaries, \
        f"Number of boundaries ({boundaries.size()}) does not match mesh topology ({expected_num_boundaries})."
    assert domains.size() == expected_num_domains, \
        f"Number of domains ({domains.size()}) does not match mesh topology ({expected_num_domains})."

    # Validate the boundary and domain IDs
    known_boundary_ids = [0, 1, 2, 3, 11, 22, 33]
    known_domain_ids = [1, 2]

    # Check if the known boundary IDs exists in the loaded boundaries
    for known_boundary_id in known_boundary_ids:
        assert known_boundary_id in boundaries.array(), \
            f"Known boundary ID ({known_boundary_id}) not found in loaded boundaries."

    # Check if the known domain IDs exists in the loaded domains
    for known_domain_id in known_domain_ids:
        assert known_domain_id in domains.array(), \
            f"Known domain ID ({known_domain_id}) not found in loaded domains."


def test_load_mesh_info(temporary_hdf5_file):
    """
    Test the load_mesh_info function with specific expected values.
    """
    # Define expected values
    expected_id_in = [3]
    expected_id_out = [2, 4]
    expected_id_wall = min(expected_id_in + expected_id_out) - 1
    expected_Q_mean = 2.4817264611257612
    expected_area_ratio = [0.4124865453872114, 0.5875134546127886]
    expected_area_inlet = 8.00556922943794
    expected_solid_side_wall_id = 11
    expected_interface_fsi_id = 22
    expected_solid_outer_wall_id = 33
    expected_fluid_volume_id = 0
    expected_solid_volume_id = 1
    expected_branch_ids_offset = 1000

    # Test the load_mesh_info function with the temporary JSON info file
    mesh_info = load_mesh_info(temporary_hdf5_file)

    # Assertions using named tuple components
    assert mesh_info.id_in == expected_id_in, \
        f"Actual id_in: {mesh_info.id_in}, Expected id_in: {expected_id_in}"
    assert mesh_info.id_out == expected_id_out, \
        f"Actual id_out: {mesh_info.id_out}, Expected id_out: {expected_id_out}"
    assert mesh_info.id_wall == expected_id_wall, \
        f"Actual id_wall: {mesh_info.id_wall}, Expected id_wall: {expected_id_wall}"
    assert mesh_info.Q_mean == expected_Q_mean, \
        f"Actual Q_mean: {mesh_info.Q_mean}, Expected Q_mean: {expected_Q_mean}"
    assert mesh_info.area_ratio == expected_area_ratio, \
        f"Actual area_ratio: {mesh_info.area_ratio}, Expected area_ratio: {expected_area_ratio}"
    assert mesh_info.area_inlet == expected_area_inlet, \
        f"Actual area_inlet: {mesh_info.area_inlet}, Expected area_inlet: {expected_area_inlet}"
    assert mesh_info.solid_side_wall_id == expected_solid_side_wall_id, \
        f"Actual solid_side_wall_id: {mesh_info.solid_side_wall_id}, " \
        f"Expected solid_side_wall_id: {expected_solid_side_wall_id}"
    assert mesh_info.interface_fsi_id == expected_interface_fsi_id, \
        f"Actual interface_fsi_id: {mesh_info.interface_fsi_id}, Expected interface_fsi_id: {expected_interface_fsi_id}"
    assert mesh_info.solid_outer_wall_id == expected_solid_outer_wall_id, \
        f"Actual solid_outer_wall_id: {mesh_info.solid_outer_wall_id}, " \
        f"Expected solid_outer_wall_id: {expected_solid_outer_wall_id}"
    assert mesh_info.fluid_volume_id == expected_fluid_volume_id, \
        f"Actual fluid_volume_id: {mesh_info.fluid_volume_id}, Expected fluid_volume_id: {expected_fluid_volume_id}"
    assert mesh_info.solid_volume_id == expected_solid_volume_id, \
        f"Actual solid_volume_id: {mesh_info.solid_volume_id}, Expected solid_volume_id: {expected_solid_volume_id}"
    assert mesh_info.branch_ids_offset == expected_branch_ids_offset, \
        f"Actual branch_ids_offset: {mesh_info.branch_ids_offset}, " + \
        f"Expected branch_ids_offset: {expected_branch_ids_offset}"


def test_load_probe_points(temporary_hdf5_file):
    """
    Test the load_probe_points function by comparing loaded probe points with expected values.
    """
    # Get the mesh path from the temporary_hdf5_file fixture
    mesh_path = Path(temporary_hdf5_file)

    # Define expected probe points
    expected_probe_points = np.array([[35.66228104, 30.56293869, 39.70381927],
                                      [34.18024826, 31.10959816, 40.74102783],
                                      [32.8044548, 31.64189339, 41.15113831],
                                      [30.45511246, 31.9192028, 40.71715546],
                                      [34.03385925, 32.06142807, 42.29312515],
                                      [35.37520218, 33.24145508, 43.67529678]])

    # Test the load_probe_points function
    loaded_probe_points = load_probe_points(mesh_path)

    # Check if the loaded probe points match the expected data
    assert np.allclose(loaded_probe_points, expected_probe_points), \
        f"Loaded probe points:\n{loaded_probe_points}\n do not match expected values:\n{expected_probe_points}"
