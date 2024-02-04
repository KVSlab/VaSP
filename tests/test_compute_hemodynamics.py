from pathlib import Path
import subprocess
import shutil

from fenics import XDMFFile, Mesh, HDF5File, FunctionSpace, Function, BoundaryMesh, SubDomain, MeshFunction, assemble, \
    dx


def test_compute_hemodynamics(tmpdir):
    """
    Here we use the velocity data from Hagen–Poiseuille flow in a pipe to test the computation of hemodynamics indices.
    The analytical solution for the velociy is u = G/(4*mu) * (R^2 - r^2),
    where G is the pressure gradient divided by the pipe length, mu is the viscosity,
    R is the radius of the pipe, and r is the distance from the center of the pipe.
    In this example problem, we used G = 4, mu = 1, R = 1, and the length of the pipe is 5.
    WSS is defined as WSS = mu * du/dr |r=R, which gives WSS = G * R / 2 = 2 Pa in this case.

    We defined wall region excluding the two ends of the pipe, and then average the WSS over the wall region.
    Finally, we assert that the average WSS is within 1.95 and 2.05 Pa to pass the test.

    Ref. https://en.wikipedia.org/wiki/Hagen–Poiseuille_equation
         https://github.com/keiyamamo/turtleFSI/blob/pipe/turtleFSI/problems/pipe_laminar.py
    """
    folder_path = Path("tests/test_data/hemodynamics_data")

    # Copy the data to temporary folder
    tmpdir_path = Path(tmpdir)

    shutil.copytree(folder_path, tmpdir_path, dirs_exist_ok=True)

    cmd = (f"vasp-compute-hemo --folder {tmpdir_path}")
    _ = subprocess.check_output(cmd, shell=True)

    hemodynamics_data = tmpdir_path / "Hemodynamic_indices"

    assert hemodynamics_data.exists()

    # Read TAWSS data and check the value
    mesh_path = tmpdir_path / "Mesh" / "mesh_fluid.h5"
    tawss_data = hemodynamics_data / "TAWSS.xdmf"

    # First read mesh
    mesh = Mesh()
    with HDF5File(mesh.mpi_comm(), str(mesh_path), "r") as infile:
        infile.read(mesh, "mesh", False)

    boundary_mesh = BoundaryMesh(mesh, "exterior")

    # Define wall region excluding the two ends of the pipe
    class Wall(SubDomain):
        def inside(self, x, on_boundary):
            return x[0] > 0.1 and x[0] < 4.9

    # Mark the wall region as 1
    marker = MeshFunction("size_t", boundary_mesh, boundary_mesh.topology().dim())
    marker.set_all(0)
    Wall().mark(marker, 1)

    # Define function space and function
    V = FunctionSpace(boundary_mesh, "DG", 1)
    tawss = Function(V)

    # Read TAWSS data
    with XDMFFile(mesh.mpi_comm(), str(tawss_data)) as infile:
        infile.read_checkpoint(tawss, "TAWSS", 0)

    # Check the average value over the wall region
    surface_integral = assemble(tawss * dx(subdomain_data=marker, subdomain_id=1))
    surface_area = assemble(1 * dx(domain=boundary_mesh, subdomain_data=marker, subdomain_id=1))

    surface_average = surface_integral / surface_area

    assert 1.95 < surface_average < 2.05
