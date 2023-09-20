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
    cmd = ("turtleFSI -p offset_stenosis -dt 0.01 -T 0.05 --verbose True" +
           " --theta 0.51 --folder {} --sub-folder 1 --new-arguments mesh_path={}")
    result = subprocess.run(cmd.format(tmpdir, temporary_hdf5_file), shell=True, check=True, cwd="src/fsipy/simulations/")

    # Here we check the velocity and pressure at a specific probe point at a specific time step
    target_time_step = 5
    target_probe_point = 6
    # TODO: pressusre does not need  () around it
    output_regular_expression = (
    r"Probe Point {}: Velocity: \((.*?), (.*?), (.*?)\) \| Pressure: (.*?)\n"
    r"Solved for timestep {}, t = (\d+\.\d+) in (\d+\.\d+) s".format(target_probe_point, target_time_step))

    output_match = re.search(output_regular_expression, str(result.stdout), re.MULTILINE)

    expected_velocity = [2.651341658407568e-07,  1.7451668944612546e-07, 9.392417990920502e-08]
    expected_pressure = -24871.223536973383
    if output_match:
        velocity = [float(output_match.group(5)), float(output_match.group(6)), float(output_match.group(7))]
        pressure = float(output_match.group(8))

        assert np.isclose(velocity, expected_velocity).all(), "Velocity does not match expected value."
        assert np.isclose(pressure, expected_pressure), "Pressure does not match expected value."

    else:
        raise ValueError("Could not find velocity and pressure in output.")
