import pytest
import subprocess
import re
import numpy as np


# Define the list of input geometrical data paths
# Since we will run turtleFSI from src/fsipy/simulations/, we need to go up one level
input_data_paths = [
    "../../../tests/test_data/offset_stenosis/offset_stenosis.h5"
    # Add more paths here as needed
]


@pytest.mark.parametrize("input_mesh", input_data_paths)
def test_offset_stenosis_problem(input_mesh, tmpdir):
    """
    Test the offset stenosis problem.
    """
    cmd = ("turtleFSI -p offset_stenosis -dt 0.01 -T 0.04 --verbose True" +
           " --theta 0.51 --folder {} --sub-folder 1 --new-arguments mesh_path={}")
    result = subprocess.check_output(cmd.format(tmpdir, input_mesh), shell=True, cwd="src/fsipy/simulations/")

    # Here we check the velocity and pressure at the last time step from Probe point 5
    target_probe_point = 5
    output_re = (r"Point {}: Velocity: \((-?\d+\.\d+(?:e[+-]?\d+)?), (-?\d+\.\d+(?:e[+-]?\d+)?), "
                 r"(-?\d+\.\d+(?:e[+-]?\d+)?)\) \| Pressure: (-?\d+\.\d+(?:e[+-]?\d+)?)").format(target_probe_point)
    output_match = re.findall(output_re, str(result))

    assert output_match is not None, "Regular expression did not match the output."

    expected_velocity = [-0.012555684636129378, 8.084632937234429e-06, -2.3712435710623827e-05]
    expected_pressure = 0.43014573081840823

    velocity_last_time_step = [float(output_match[-1][0]), float(output_match[-1][1]), float(output_match[-1][2])]
    pressure_last_time_step = float(output_match[-1][3])

    print("Velocity: {}".format(velocity_last_time_step))
    print("Pressure: {}".format(pressure_last_time_step))

    assert np.isclose(velocity_last_time_step, expected_velocity).all(), "Velocity does not match expected value."
    assert np.isclose(pressure_last_time_step, expected_pressure), "Pressure does not match expected value."
