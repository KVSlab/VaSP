import pytest
import subprocess
import re
import numpy as np


# Define the list of input geometrical data paths
# Since we will run turtleFSI from src/vasp/simulations/, we need to go up one level
input_data_paths = [
    "../../../tests/test_data/offset_stenosis/offset_stenosis.h5",
    "../../../tests/test_data/cylinder/cylinder.h5",
    "../../../tests/test_data/aneurysm/small_aneurysm.h5"
    # Add more paths here as needed
]


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_offset_stenosis_problem(input_mesh, tmpdir):
    """
    Test the offset stenosis problem.
    """
    cmd = ("turtleFSI -p offset_stenosis -dt 0.01 -T 0.04 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --new-arguments mesh_path={input_mesh}")
    result = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")

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

    # Here we will check the displacement from the probe point
    output_re = (r"Point {}: Displacement: \((-?\d+\.\d+(?:e[+-]?\d+)?), (-?\d+\.\d+(?:e[+-]?\d+)?), "
                 r"(-?\d+\.\d+(?:e[+-]?\d+)?)\)").format(target_probe_point)
    output_match = re.findall(output_re, str(result))

    assert output_match is not None, "Regular expression did not match the output."

    expected_displacement = [-9.431090796213597e-06, -4.33478380630615e-05, -4.655061542874265e-05]
    displacement_last_time_step = [float(output_match[-1][0]), float(output_match[-1][1]), float(output_match[-1][2])]

    print("Displacement: {}".format(displacement_last_time_step))
    assert np.isclose(displacement_last_time_step, expected_displacement).all()


@pytest.mark.parametrize("input_mesh", [input_data_paths[1]])
def test_predeform_problem(input_mesh, tmpdir):
    """
    Test the predeform problem.
    """
    cmd = ("turtleFSI -p predeform -dt 0.01 -T 0.03 --verbose True" +
           f" --folder {tmpdir} --sub-folder 1 --new-arguments mesh_path={input_mesh}")
    result = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    check_velocity_cfl_reynolds(result)


@pytest.mark.parametrize("input_mesh", [input_data_paths[1]])
def test_cylinder_problem(input_mesh, tmpdir):
    """
    Test the cylinder problem.
    """
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.004 --verbose True " +
           f"--folder {tmpdir} --sub-folder 1 --new-arguments mesh_path={input_mesh}")
    result = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    check_velocity_cfl_reynolds(result)


@pytest.mark.parametrize("input_mesh", [input_data_paths[2]])
def test_aneurysm_problem(input_mesh, tmpdir):
    """
    Test the aneurysm problem.
    """
    cmd = ("turtleFSI -p aneurysm -dt 0.001 -T 0.004 --verbose True " +
           f"--folder {tmpdir} --sub-folder 1 --new-arguments inlet_id=4 mesh_path={input_mesh}")
    result = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    check_velocity_cfl_reynolds(result)


def check_velocity_cfl_reynolds(result):
    """
    Sanity check of the velocity, CFL number, and Reynolds number in the result.
    """
    # check velocity mean, min, max in the domain
    output_velocity = (r"Velocity \(mean, min, max\): (\d+(?:\.\d+)?(?:e-\d+)?)\s*,\s*(\d+(?:\.\d+)?(?:e-\d+)?)\s*,"
                       r"\s*(\d+(?:\.\d+)?(?:e-\d+)?)")

    output_match_velocity = re.findall(output_velocity, str(result))
    assert output_match_velocity is not None, "Regular expression did not match the output."

    velocity_mean_min_max = [float(output_match_velocity[-1][0]), float(output_match_velocity[-1][1]),
                             float(output_match_velocity[-1][2])]

    print(f"Velocity mean, min, max: {velocity_mean_min_max}")
    assert all(np.isfinite(v) for v in velocity_mean_min_max), "Velocity mean, min, max should be finite values."
    assert all(v >= 0 for v in velocity_mean_min_max), "Velocity mean, min, max should be non-negative."

    # check CFL number
    output_cfl_number = (r"CFL \(mean, min, max\): (\d+(?:\.\d+)?(?:e-\d+)?)\s*,\s*(\d+(?:\.\d+)?(?:e-\d+)?)\s*,"
                         r"\s*(\d+(?:\.\d+)?(?:e-\d+)?)")

    output_match_cfl_number = re.findall(output_cfl_number, str(result))
    assert output_match_cfl_number is not None, "Regular expression did not match the output."

    cfl_number_mean_min_max = [float(output_match_cfl_number[-1][0]), float(output_match_cfl_number[-1][1]),
                               float(output_match_cfl_number[-1][2])]

    print(f"CFL number mean, min, max: {cfl_number_mean_min_max}")
    assert all(np.isfinite(cfl) for cfl in cfl_number_mean_min_max), \
        "CFL number mean, min, max should be finite values."
    assert all(cfl >= 0 for cfl in cfl_number_mean_min_max), \
        "CFL number mean, min, max should be non-negative."
