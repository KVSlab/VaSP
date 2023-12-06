import pytest
import subprocess
import re
import numpy as np


# Define the list of input geometrical data paths
# Since we will run turtleFSI from src/fsipy/simulations/, we need to go up one level
input_data_paths = [
    "../../../tests/test_data/offset_stenosis/offset_stenosis.h5",
    "../../../tests/test_data/cylinder/cylinder.h5"
    # Add more paths here as needed
]


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_offset_stenosis_problem(input_mesh, tmpdir):
    """
    Test the offset stenosis problem.
    """
    cmd = ("turtleFSI -p offset_stenosis -dt 0.01 -T 0.04 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --new-arguments mesh_path={input_mesh}")
    result = subprocess.check_output(cmd, shell=True, cwd="src/fsipy/simulations/")

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


@pytest.mark.parametrize("input_mesh", [input_data_paths[1]])
def test_predeform_problem(input_mesh, tmpdir):
    """
    Test the predeform problem.
    """
    cmd = ("turtleFSI -p predeform -dt 0.01 -T 0.03 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --new-arguments mesh_path={input_mesh}")
    result = subprocess.check_output(cmd, shell=True, cwd="src/fsipy/simulations/")

    output_re = r"v \(centerline, at inlet\) = (\d+\.\d+|\d+) m/s"
    output_match = re.findall(output_re, str(result))

    assert output_match is not None, "Regular expression did not match the output."

    expected_velocity = 0.2591186271093947
    velocity_at_inlet = float(output_match[-1])

    print("Velocity: {}".format(velocity_at_inlet))

    assert np.isclose(velocity_at_inlet, expected_velocity), "Velocity does not match expected value."


@pytest.mark.parametrize("input_mesh", [input_data_paths[1]])
def test_cylinder_problem(input_mesh, tmpdir):
    """
    Test the cylinder problem.
    """
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.004 --verbose True " +
           f"--folder {tmpdir} --sub-folder 1 --new-arguments mesh_path={input_mesh}")
    result = subprocess.check_output(cmd, shell=True, cwd="src/fsipy/simulations/")
    # check flow rate at inlet
    output_flow_rate = r"Flow Rate at Inlet: (\d+(?:\.\d+)?(?:e-\d+)?)"

    output_match_flow_rate = re.findall(output_flow_rate, str(result))
    assert output_match_flow_rate is not None, "Regular expression did not match the output."

    expected_flow_rate = 1.6913532412047225e-09
    flow_rate_at_inlet = float(output_match_flow_rate[-1])

    print(f"Flow rate: {flow_rate_at_inlet}")
    assert np.isclose(flow_rate_at_inlet, expected_flow_rate), "Flow rate does not match expected value."

    # check velocity mean, min, max in the domain
    ourput_velocity = (r"Velocity \(mean, min, max\): (\d+(?:\.\d+)?(?:e-\d+)?)\s*,\s*(\d+(?:\.\d+)?(?:e-\d+)?)\s*,"
                       r"\s*(\d+(?:\.\d+)?(?:e-\d+)?)")

    output_match_velocity = re.findall(ourput_velocity, str(result))
    assert output_match_velocity is not None, "Regular expression did not match the output."

    expected_velocity_mean_min_max = [0.0015175903693111237, 2.83149082127162e-06, 0.004025814882456499]
    velocity_mean_min_max = [float(output_match_velocity[-1][0]), float(output_match_velocity[-1][1]),
                             float(output_match_velocity[-1][2])]

    print(f"Velocity mean, min, max: {velocity_mean_min_max}")
    assert np.isclose(velocity_mean_min_max, expected_velocity_mean_min_max).all(), \
        "Velocity mean, min, max does not match expected value."

    # check CFL number
    output_cfl_number = (r"CFL \(mean, min, max\): (\d+(?:\.\d+)?(?:e-\d+)?)\s*,\s*(\d+(?:\.\d+)?(?:e-\d+)?)\s*,"
                         r"\s*(\d+(?:\.\d+)?(?:e-\d+)?)")

    output_match_cfl_number = re.findall(output_cfl_number, str(result))
    assert output_match_cfl_number is not None, "Regular expression did not match the output."

    expected_cfl_number = [0.008465020210929876, 1.5793871333017697e-05, 0.022455733137609308]
    cfl_number_mean_min_max = [float(output_match_cfl_number[-1][0]), float(output_match_cfl_number[-1][1]),
                               float(output_match_cfl_number[-1][2])]

    print(f"CFL number mean, min, max: {cfl_number_mean_min_max}")
    assert np.isclose(cfl_number_mean_min_max, expected_cfl_number).all(), \
        "CFL number mean, min, max does not match expected value."

    # Check Reynolds number
    output_re_number = (r"Reynolds Numbers \(mean, min, max\): (\d+(?:\.\d+)?(?:e-\d+)?)\s*,"
                        r"\s*(\d+(?:\.\d+)?(?:e-\d+)?)\s*,\s*(\d+(?:\.\d+)?(?:e-\d+)?)")

    output_match_re_number = re.findall(output_re_number, str(result))
    assert output_match_re_number is not None, "Regular expression did not match the output."

    expected_re_number = [0.4304434011992387, 0.0008031129903162388, 1.1418663992904048]
    reynolds_number_mean_min_max = [float(output_match_re_number[-1][0]), float(output_match_re_number[-1][1]),
                                    float(output_match_re_number[-1][2])]

    print(f"Reynolds number mean, min, max: {reynolds_number_mean_min_max}")
    assert np.isclose(reynolds_number_mean_min_max, expected_re_number).all(), \
        "Reynolds number mean, min, max does not match expected value."
