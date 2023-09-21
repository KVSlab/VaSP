from pathlib import Path
import pytest
import subprocess
import re
import numpy as np

from fsipy.automatedPreprocessing.automated_preprocessing import read_command_line, \
    run_pre_processing

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
def test_offset_stenosis_problem(temporary_hdf5_file, tmpdir):
    """
    Test the offset stenosis problem.
    """
    cmd = ("turtleFSI -p offset_stenosis -dt 0.01 -T 0.04 --verbose True" +
           " --theta 0.51 --folder {} --sub-folder 1 --new-arguments mesh_path={}")
    result = subprocess.check_output(cmd.format(tmpdir, temporary_hdf5_file), shell=True, cwd="src/fsipy/simulations/")

    # Here we check the velocity and pressure at the last time step from Probe point 6
    target_probe_point = 6
    output_re = (r"Point {}: Velocity: \((-?\d+\.\d+(?:e[+-]?\d+)?), (-?\d+\.\d+(?:e[+-]?\d+)?), "
                 r"(-?\d+\.\d+(?:e[+-]?\d+)?)\) \| Pressure: (-?\d+\.\d+(?:e[+-]?\d+)?)").format(target_probe_point)
    output_match = re.findall(output_re, str(result))

    assert output_match is not None, "Regular expression did not match the output."

    expected_velocity = [2.651341658407568e-07, 1.7451668944612546e-07, 9.392417990920502e-08]
    expected_pressure = -24871.223536973383

    velocity_last_time_step = [float(output_match[-1][0]), float(output_match[-1][1]), float(output_match[-1][2])]
    pressure_last_time_step = float(output_match[-1][3])

    assert np.isclose(velocity_last_time_step, expected_velocity).all(), "Velocity does not match expected value."
    assert np.isclose(pressure_last_time_step, expected_pressure), "Pressure does not match expected value."
