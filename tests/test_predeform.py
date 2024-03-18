import pytest
import subprocess
import h5py


# Define the list of input geometrical data paths
# Since we will run turtleFSI from src/vasp/simulations/, we need to go up one level
input_data_paths = [
    "../../../tests/test_data/cylinder/cylinder.h5"
]


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_predeform_mesh(input_mesh, tmpdir):
    """
    Test predeform_mesh function
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 1 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")

    # 2. run predeform_mesh
    cmd = f"vasp-predeform-mesh --folder {tmpdir}/1"
    _ = subprocess.check_output(cmd, shell=True)

    # 3. check the predeformed mesh by checking the coordinates
    predeformed_mesh_path = tmpdir / "1" / "Mesh" / "mesh_predeformed.h5"
    with h5py.File(predeformed_mesh_path, "r") as f:
        coordinates = f["mesh/coordinates"][:]
        first_coordinate = coordinates[0]
        expected_first_coordinate = [7.382372340085156E-5, -1.1083576098054155E-4, 4.930899508039441E-4]
        assert first_coordinate == pytest.approx(expected_first_coordinate, rel=1e-6)
