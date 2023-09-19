from pathlib import Path
import pytest
import subprocess

from turtleFSI.problems import *

from fsipy.automatedPreprocessing.automated_preprocessing import read_command_line, \
    run_pre_processing
from fsipy.simulations.simulation_common import load_mesh_and_data

# Define the list of input geometrical data paths
input_data_paths = [
    "tests/test_data/offset_stenosis/offset_stenosis.stl"
    # Add more paths here as needed
]


@pytest.fixture(scope="function")
def temporary_hdf5_file(tmpdir, request):
    """
    Fixture for generating a temporary HDF5 file path with a mesh for testing purposes.
    """
    param_values = request.param
    original_model_path = Path(param_values)

    # Rest of your fixture code remains the same
    model_path = Path(tmpdir) / original_model_path.name
    mesh_path_hdf5 = model_path.with_suffix(".h5")

    # Make a copy of the original model
    model_path.write_bytes(original_model_path.read_bytes())

    # Get default input parameters
    common_input = read_command_line(str(model_path))
    common_input.update(
        dict(
            coarsening_factor=3.8,
            visualize=False,
            compress_mesh=False,
            outlet_flow_extension_length=4,
            inlet_flow_extension_length=0,
            edge_length=1.0,
            number_of_sublayers_fluid=1,
            number_of_sublayers_solid=1
        )
    )

    # Run pre-processing to generate the mesh
    run_pre_processing(**common_input)

    yield mesh_path_hdf5  # Provide the temporary file path as a fixture


@pytest.mark.parametrize("temporary_hdf5_file", input_data_paths, indirect=True)
def test_offset_stenosis_problem(temporary_hdf5_file):

    # cmd = ("turtleFSI --problem src.fsipy.simulations.offset_stenosis -dt 0.01 -T 0.05 --verbose True" +
    #        " --theta 0.51 --folder tmp --sub-folder 1 --mesh-path " + str(temporary_hdf5_file))
    # subprocess.run(cmd, shell=True, check=True)
    mesh, boundaries, domains = load_mesh_and_data(temporary_hdf5_file)

    assert mesh.num_vertices() > 0