import pytest
import subprocess
import h5py
import os


# Define the list of input geometrical data paths
# Since we will run turtleFSI from src/vasp/simulations/, we need to go up one level
input_data_paths = [
    "../../../tests/test_data/cylinder/cylinder.h5"
]


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_refine_mesh(input_mesh, tmpdir):
    """
    Test refine_mesh function
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 2 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    # 2. run vasp-refine-mesh to refine the mesh
    cmd = (f"vasp-refine-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # check if the refined mesh exists
    refined_mesh_path = tmpdir / "1" / "Mesh" / "mesh_refined.h5"
    assert refined_mesh_path.exists()

    # check if the refined mesh has the correct number of nodes and cells
    with h5py.File(refined_mesh_path, "r") as f:
        num_nodes = f["mesh/coordinates"].shape[0]
        assert num_nodes == 2500
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 13176

    # check if the number of refined cells is 8 times the number of original cells
    fsi_mesh_path = tmpdir / "1" / "Mesh" / "mesh.h5"
    with h5py.File(fsi_mesh_path, "r") as f:
        num_cells_fsi = f["mesh/topology"].shape[0]
        assert num_cells_fsi * 8 == num_cells


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_refine_mesh_mpi(input_mesh, tmpdir):
    """
    Test refine_mesh function where turtleFSI is run with MPI
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("mpirun -np 2 turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 2 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/", env=os.environ)
    # 2. run vasp-refine-mesh to refine the mesh
    cmd = (f"vasp-refine-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # check if the refined mesh exists
    refined_mesh_path = tmpdir / "1" / "Mesh" / "mesh_refined.h5"
    assert refined_mesh_path.exists()

    # check if the refined mesh has the correct number of nodes and cells
    with h5py.File(refined_mesh_path, "r") as f:
        num_nodes = f["mesh/coordinates"].shape[0]
        assert num_nodes == 2500
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 13176

    # check if the number of refined cells is 8 times the number of original cells
    fsi_mesh_path = tmpdir / "1" / "Mesh" / "mesh.h5"
    with h5py.File(fsi_mesh_path, "r") as f:
        num_cells_fsi = f["mesh/topology"].shape[0]
        assert num_cells_fsi * 8 == num_cells


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_separate_mesh_save_deg_1(input_mesh, tmpdir):
    """
    Test separate_mesh function with save_deg=1
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 1 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    # 2. run vasp-separate-mesh to separate the mesh
    cmd = (f"vasp-separate-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # check if the fluid/solid mesh exists
    fluid_mesh_path = tmpdir / "1" / "Mesh" / "mesh_fluid.h5"
    assert fluid_mesh_path.exists()
    solid_mesh_path = tmpdir / "1" / "Mesh" / "mesh_solid.h5"
    assert solid_mesh_path.exists()

    # check if the fluid/solid mesh has the correct number of nodes and cells
    with h5py.File(fluid_mesh_path, "r") as f:
        num_nodes = f["mesh/coordinates"].shape[0]
        assert num_nodes == 253
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 1128

    with h5py.File(solid_mesh_path, "r") as f:
        num_nodes = f["mesh/coordinates"].shape[0]
        assert num_nodes == 198
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 519

    # check if the sum of the number of cells in the fluid and solid mesh is equal to
    # the number of cells in the FSI mesh 1128 + 519 = 1647
    fsi_mesh_path = tmpdir / "1" / "Mesh" / "mesh.h5"
    with h5py.File(fsi_mesh_path, "r") as f:
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 1647


@pytest.mark.parametrize("input_mesh", [input_data_paths[0]])
def test_separate_mesh_save_deg_2(input_mesh, tmpdir):
    """
    Test separate_mesh function with save_deg=2
    """
    # 1. run turtleFSI to create some simulation results
    cmd = ("turtleFSI -p cylinder -dt 0.001 -T 0.002 --verbose True" +
           f" --theta 0.51 --folder {tmpdir} --sub-folder 1 --save-deg 2 --new-arguments mesh_path={input_mesh}")
    _ = subprocess.check_output(cmd, shell=True, cwd="src/vasp/simulations/")
    # 2. refine the mesh
    cmd = (f"vasp-refine-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)
    # 3. run vasp-separate-mesh to separate the mesh
    cmd = (f"vasp-separate-mesh --folder {tmpdir}/1/")
    _ = subprocess.check_output(cmd, shell=True)

    # check if the fluid/solid mesh exists
    fluid_mesh_path = tmpdir / "1" / "Mesh" / "mesh_refined_fluid.h5"
    assert fluid_mesh_path.exists()
    solid_mesh_path = tmpdir / "1" / "Mesh" / "mesh_refined_solid.h5"
    assert solid_mesh_path.exists()

    # check if the fluid/solid mesh has the correct number of nodes and cells
    with h5py.File(fluid_mesh_path, "r") as f:
        num_nodes = f["mesh/coordinates"].shape[0]
        assert num_nodes == 1758
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 9024

    with h5py.File(solid_mesh_path, "r") as f:
        num_nodes = f["mesh/coordinates"].shape[0]
        assert num_nodes == 1113
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 4152

    # check if the sum of the number of cells in the fluid and solid mesh is equal to
    # the number of cells in the FSI mesh 9024 + 4152 = 13176
    fsi_mesh_path = tmpdir / "1" / "Mesh" / "mesh_refined.h5"
    with h5py.File(fsi_mesh_path, "r") as f:
        num_cells = f["mesh/topology"].shape[0]
        assert num_cells == 13176
