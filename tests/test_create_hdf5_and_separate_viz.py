import pytest
import subprocess
import h5py
import numpy as np

# Define the list of input geometrical data paths
# Since we will run turtleFSI from src/vasp/simulations/, we need to go up one level
input_data_paths = [
    "../../../tests/test_data/cylinder/cylinder.h5"
]


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_create_hdf5_solid_only(input_mesh, tmpdir):
    """
    Test create_hdf5 function with solid only mesh for displacement
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 2 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    # 2. run vasp-refine-mesh to refine the mesh
    cmd = (f"vasp-refine-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 3. run vasp-separate-mesh to separate the mesh
    cmd = (f"vasp-separate-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 4. run vasp-create-hdf5 to create the hdf5 file
    cmd = (f"vasp-create-hdf5 --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 5. check if the hdf5 file exists
    velocity_hdf5_path = tmpdir / "1" / "Visualization_separate_domain" / "u.h5"
    assert velocity_hdf5_path.exists()
    displacement_hdf5_path = tmpdir / "1" / "Visualization_separate_domain" / "d_solid.h5"
    assert displacement_hdf5_path.exists()

    # 6. check if the hdf5 file has the correct values
    with h5py.File(velocity_hdf5_path, "r") as f:
        first_time = f["velocity/vector_0"][0]
        assert np.isclose(first_time, 4.38261949610407E-6, atol=1e-10)
        last_time = f["velocity/vector_2"][0]
        assert np.isclose(last_time, 8.137814761280497E-6, atol=1e-10)

    with h5py.File(displacement_hdf5_path, "r") as f:
        first_time = f["displacement/vector_0"][0]
        assert np.isclose(first_time, 2.235075700301419E-9, atol=1e-10)
        last_time = f["displacement/vector_2"][0]
        assert np.isclose(last_time, 1.3776599148439903E-8, atol=1e-10)

    # 7. create separate visualization
    cmd = (f"vasp-create-separate-domain-viz --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 8. check if the separate visualization exists
    velocity_h5_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.h5"
    velocity_xdmf_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.xdmf"
    assert velocity_h5_path.exists(), "Separate visualization for velocity does not exist"
    assert velocity_xdmf_path.exists(), "Separate visualization for velocity does not exist"

    displacement_h5_path = tmpdir / "1" / "Visualization_separate_domain" / "displacement_solid.h5"
    displacement_xdmf_path = tmpdir / "1" / "Visualization_separate_domain" / "displacement_solid.xdmf"
    assert displacement_h5_path.exists(), "Separate visualization for displacement does not exist"
    assert displacement_xdmf_path.exists(), "Separate visualization for displacement does not exist"


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_create_hdf5_entire_domain(input_mesh, tmpdir):
    """
    Test create_hdf5 function with entire domain mesh for displacement
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 2 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    # 2. run vasp-refine-mesh to refine the mesh
    cmd = (f"vasp-refine-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 3. run vasp-separate-mesh to separate the mesh
    cmd = (f"vasp-separate-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 4. run vasp-create-hdf5 to create the hdf5 file
    cmd = (f"vasp-create-hdf5 --folder {tmpdir}/1/ --extract-entire-domain")
    _ = subprocess.check_output(cmd, shell=True)

    # 5. check if the hdf5 file exists, here we only check the displacement for the entire domain
    displacement_hdf5_path = tmpdir / "1" / "Visualization_separate_domain" / "d.h5"
    assert displacement_hdf5_path.exists()

    # 6. check if the hdf5 file has the correct values
    with h5py.File(displacement_hdf5_path, "r") as f:
        first_time = f["displacement/vector_0"][0]
        assert np.isclose(first_time, 2.235075700301419E-9, atol=1e-10)
        last_time = f["displacement/vector_2"][0]
        assert np.isclose(last_time, 1.3776599148439903E-8, atol=1e-10)

    # 7. create separate visualization
    cmd = (f"vasp-create-separate-domain-viz --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 8. check if the separate visualization exists (only fluid velocity)
    velocity_h5_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.h5"
    velocity_xdmf_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.xdmf"
    assert velocity_h5_path.exists(), "Separate visualization for velocity does not exist"
    assert velocity_xdmf_path.exists(), "Separate visualization for velocity does not exist"


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_create_hdf5_with_stride(input_mesh, tmpdir):
    """
    Test create_hdf5 function with stride
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 2 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    # 2. run vasp-refine-mesh to refine the mesh
    cmd = (f"vasp-refine-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 3. run vasp-separate-mesh to separate the mesh
    cmd = (f"vasp-separate-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 4. run vasp-create-hdf5 to create the hdf5 file
    cmd = (f"vasp-create-hdf5 --folder {tmpdir}/1/ --stride 2")
    _ = subprocess.check_output(cmd, shell=True)

    # 5. check if the hdf5 file exists
    velocity_hdf5_path = tmpdir / "1" / "Visualization_separate_domain" / "u.h5"
    assert velocity_hdf5_path.exists()
    displacement_hdf5_path = tmpdir / "1" / "Visualization_separate_domain" / "d_solid.h5"
    assert displacement_hdf5_path.exists()

    # 6. check if the hdf5 file has the correct values
    with h5py.File(velocity_hdf5_path, "r") as f:
        first_time = f["velocity/vector_0"][0]
        assert np.isclose(first_time, 4.38261949610407E-6, atol=1e-10)
        last_time = f["velocity/vector_1"][0]
        assert np.isclose(last_time, 8.137814761280497E-6, atol=1e-10)

    with h5py.File(displacement_hdf5_path, "r") as f:
        first_time = f["displacement/vector_0"][0]
        assert np.isclose(first_time, 2.235075700301419E-9, atol=1e-10)
        last_time = f["displacement/vector_1"][0]
        assert np.isclose(last_time, 1.3776599148439903E-8, atol=1e-10)

    # 7. create separate visualization
    cmd = (f"vasp-create-separate-domain-viz --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 8. check if the separate visualization exists
    velocity_h5_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.h5"
    velocity_xdmf_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.xdmf"
    assert velocity_h5_path.exists(), "Separate visualization for velocity does not exist"
    assert velocity_xdmf_path.exists(), "Separate visualization for velocity does not exist"

    displacement_h5_path = tmpdir / "1" / "Visualization_separate_domain" / "displacement_solid.h5"
    displacement_xdmf_path = tmpdir / "1" / "Visualization_separate_domain" / "displacement_solid.xdmf"
    assert displacement_h5_path.exists(), "Separate visualization for displacement does not exist"
    assert displacement_xdmf_path.exists(), "Separate visualization for displacement does not exist"


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_create_hdf5_with_time(input_mesh, tmpdir):
    """
    Test create_hdf5 function with time specified
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 2 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    # 2. run vasp-refine-mesh to refine the mesh
    cmd = (f"vasp-refine-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 3. run vasp-separate-mesh to separate the mesh
    cmd = (f"vasp-separate-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 4. run vasp-create-hdf5 to create the hdf5 file
    cmd = (f"vasp-create-hdf5 --folder {tmpdir}/1/ --start-time 0.001 --end-time 0.002")
    _ = subprocess.check_output(cmd, shell=True)

    # 5. check if the hdf5 file exists
    velocity_hdf5_path = tmpdir / "1" / "Visualization_separate_domain" / "u.h5"
    assert velocity_hdf5_path.exists(), "Velocity hdf5 file does not exist"
    displacement_hdf5_path = tmpdir / "1" / "Visualization_separate_domain" / "d_solid.h5"
    assert displacement_hdf5_path.exists(), "Displacement hdf5 file does not exist"

    # 6. check if the hdf5 file has the correct values
    with h5py.File(velocity_hdf5_path, "r") as f:
        first_time = f["velocity/vector_0"][0]
        assert np.isclose(first_time, 4.38261949610407E-6, atol=1e-10)
        last_time = f["velocity/vector_1"][0]
        assert np.isclose(last_time, 5.244315455211961E-6, atol=1e-10)

    with h5py.File(displacement_hdf5_path, "r") as f:
        first_time = f["displacement/vector_0"][0]
        assert np.isclose(first_time, 2.235075700301419E-9, atol=1e-10)
        last_time = f["displacement/vector_1"][0]
        assert np.isclose(last_time, 7.0569699656660426E-9, atol=1e-10)

    # 7. create separate visualization
    cmd = (f"vasp-create-separate-domain-viz --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # 8. check if the separate visualization exists
    velocity_h5_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.h5"
    velocity_xdmf_path = tmpdir / "1" / "Visualization_separate_domain" / "velocity_fluid.xdmf"
    assert velocity_h5_path.exists(), "Separate visualization for velocity does not exist"
    assert velocity_xdmf_path.exists(), "Separate visualization for velocity does not exist"

    displacement_h5_path = tmpdir / "1" / "Visualization_separate_domain" / "displacement_solid.h5"
    displacement_xdmf_path = tmpdir / "1" / "Visualization_separate_domain" / "displacement_solid.xdmf"
    assert displacement_h5_path.exists(), "Separate visualization for displacement does not exist"
    assert displacement_xdmf_path.exists(), "Separate visualization for displacement does not exist"
