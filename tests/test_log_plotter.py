# Copyright (c) 2023 Simula Research Laboratory
# SPDX-License-Identifier: GPL-3.0-or-later

import pytest
import subprocess
import argparse
from pathlib import Path

import matplotlib.testing.compare as mpl_compare


test_cases = [
    ("offset_stenosis_5_cycles.log", ["--plot-cpu-time", "--end-cycle=3"], "cpu_time.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-ramp-factor", "--start-cycle=1"], "ramp_factor.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-pressure", "--start-cycle=2", "--end-cycle=4"], "pressure.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-newton-iteration-atol"], "newton_iteration_(atol).png"),
    ("offset_stenosis_5_cycles.log", ["--plot-newton-iteration-rtol"], "newton_iteration_(rtol).png"),
    ("offset_stenosis_5_cycles.log", ["--plot-probe-points", "--probe-points", "0", "3", "5"], "probe_points.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-probe-points-tke"], "probe_points_tke.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-flow-rate", "--figure-size=14,8"], "flow_rate.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-velocity", "--start-cycle=2", "--end-cycle=5"], "velocity.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-cfl"], "cfl.png"),
    ("offset_stenosis_5_cycles.log", ["--plot-reynolds"], "reynolds_numbers.png"),
    # Add more test cases as needed
]


@pytest.mark.parametrize("log_file, args, expected_image", test_cases)
def test_plot_options(tmpdir, log_file, args, expected_image):
    log_path = Path("tests/test_data/logs") / log_file

    command = ["fsipy-log-plotter", str(log_path), "--save", f"--output-directory={tmpdir}"] + args
    subprocess.run(command, check=True)

    # Get the paths to the generated and expected images
    generated_image_path = tmpdir / expected_image
    expected_image_path = Path("tests/test_data/reference_images") / expected_image

    # Compare the generated and expected images using matplotlib.testing.compare
    result = mpl_compare.compare_images(generated_image_path, expected_image_path, tol=0)

    # Assert that the result is None, indicating that the images are equal within the given tolerance
    assert result is None, f"Images differ: {result}"


@pytest.mark.parametrize("log_file, args, expected_image", test_cases)
def test_plot_options_compute_average(tmpdir, log_file, args, expected_image):
    if "--plot-newton-iteration-atol" in args or "--plot-newton-iteration-rtol" in args:
        pytest.skip("Skipping the test as Newton iteration plotting is specified")

    log_path = Path("tests/test_data/logs") / log_file

    command = ["fsipy-log-plotter", str(log_path), "--save", f"--output-directory={tmpdir}", "--compute-average"] + args
    subprocess.run(command, check=True)

    # Get the paths to the generated and expected images
    generated_image_path = tmpdir / expected_image
    expected_image_path = Path("tests/test_data/reference_images/test_average") / expected_image

    # Compare the generated and expected images using matplotlib.testing.compare
    result = mpl_compare.compare_images(generated_image_path, expected_image_path, tol=0)

    # Assert that the result is None, indicating that the images are equal within the given tolerance
    assert result is None, f"Images differ: {result}"


@pytest.mark.parametrize("log_file, args, expected_image", test_cases)
def test_plot_options_compare_cycles(tmpdir, log_file, args, expected_image):
    if "--plot-newton-iteration-atol" in args or "--plot-newton-iteration-rtol" in args:
        pytest.skip("Skipping the test as Newton iteration plotting is specified")

    log_path = Path("tests/test_data/logs") / log_file

    command = ["fsipy-log-plotter", str(log_path), "--save", f"--output-directory={tmpdir}", "--compare-cycles"] + args
    subprocess.run(command, check=True)

    # Adjust the expected image name to account for "_comparison.png" ending
    if "--plot-probe-points-tke" in args:
        probe_points = list(range(6))
        expected_images = [expected_image.replace(".png", f"_comparison_{point}.png") for point in probe_points]
    elif "--plot-probe-points" in args:
        probe_points_index = args.index("--probe-points")
        probe_points = args[probe_points_index + 1:]
        expected_images = [expected_image.replace(".png", f"_comparison_{point}.png") for point in probe_points]
    else:
        expected_images = [expected_image.replace(".png", "_comparison.png")]

    # Get the paths to the generated and expected images
    generated_image_paths = [tmpdir / img for img in expected_images]
    expected_image_paths = \
        [Path("tests/test_data/reference_images/test_compare_cycles") / img for img in expected_images]

    # Compare the generated and expected images using matplotlib.testing.compare
    for generated_image_path, expected_image_path in zip(generated_image_paths, expected_image_paths):
        result = mpl_compare.compare_images(generated_image_path, expected_image_path, tol=0)

        # Assert that the result is None, indicating that the images are equal within the given tolerance
        assert result is None, f"Images differ: {result}"


def test_plot_all(tmpdir):
    # Define the corresponding log file and the expected image
    log_path = Path("tests/test_data/logs/offset_stenosis_5_cycles.log")

    # Define a list of reference images for the --plot-all test
    reference_image_directory = Path("tests/test_data/reference_images/test_all")

    # List of all reference images in the subdirectory
    reference_images = list(reference_image_directory.glob("*.png"))

    # Customize the command to run the script with --plot-all
    command = ["fsipy-log-plotter", str(log_path), "--save", f"--output-directory={tmpdir}", "--plot-all"]
    subprocess.run(command, check=True)

    for reference_image in reference_images:
        expected_image = reference_image.name

        # Get the paths to the generated and expected images
        generated_image_path = tmpdir / expected_image
        expected_image_path = reference_image

        # Compare the generated and expected images using matplotlib.testing.compare
        result = mpl_compare.compare_images(generated_image_path, expected_image_path, tol=0)

        # Assert that the result is None, indicating that the images are equal within the given tolerance
        assert result is None, f"Images differ: {result}"


# Reference image generation and command-line argument handling functions


def generate_reference_images(log_file, args, output_directory):
    log_path = Path("tests/test_data/logs") / log_file

    command = ["fsipy-log-plotter", str(log_path), "--save", f"--output-directory={output_directory}"] + args

    subprocess.run(command, check=True)


def parse_command_line_args():
    parser = argparse.ArgumentParser(description="Test log plotting or generate reference images.")
    parser.add_argument("--generate-reference-images", action="store_true", help="Generate reference images.")
    parser.add_argument("--output-directory", type=str, default="tests/test_data/reference_images",
                        help="Output directory for reference images.")
    parser.add_argument("--pytest-args", nargs=argparse.REMAINDER, help="Arguments to pass to pytest.")
    return parser.parse_args()


def main():
    args = parse_command_line_args()

    if args.generate_reference_images:
        # Generate reference images specific to the "test_plot_options" test
        for log_file, test_args, _ in test_cases:
            generate_reference_images(log_file, test_args, args.output_directory)

        # Generate reference images specific to the "test_plot_options_compute_average" test
        for log_file, test_args, _ in test_cases:
            if "--plot-newton-iteration-atol" in test_args or "--plot-newton-iteration-rtol" in test_args:
                continue
            generate_reference_images(log_file, test_args + ["--compute-average"],
                                      f"{args.output_directory}/test_average")

        # Generate reference images specific to the "test_plot_options_compare_cycles" test
        for log_file, test_args, _ in test_cases:
            if "--plot-newton-iteration-atol" in test_args or "--plot-newton-iteration-rtol" in test_args:
                continue
            generate_reference_images(log_file, test_args + ["--compare-cycles"],
                                      f"{args.output_directory}/test_compare_cycles")

        # Generate reference images specific to the "plot-all" test
        generate_reference_images(log_file, ["--plot-all"], f"{args.output_directory}/test_all")

        print("Reference images generated. Be sure to visually check them for accuracy.")
    else:
        # Run tests using pytest
        pytest_args = [__file__]
        if args.pytest_args:
            pytest_args.extend(args.pytest_args)  # Add custom pytest arguments

        pytest.main(pytest_args)


if __name__ == "__main__":
    main()
