from argparse import ArgumentParser
import re
from dolfin import *

parameters["allow_extrapolation"] = True


def read_command_line():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--case', type=str, default="cyl_test", help="Path to simulation results",
                        metavar="PATH")
    parser.add_argument('--mesh', type=str, default="artery_coarse_rescaled", help="Mesh File Name",
                        metavar="PATH")
    parser.add_argument('--nu', type=float, default=3.5e-3, help="Viscosity used in simulation")
    parser.add_argument('--dt', type=float, default=0.001, help="Time step of simulation")
    parser.add_argument('--stride', type=int, default=1, help="Save frequency of simulation")    
    parser.add_argument('--save_deg', type=int, default=2, help="Input save_deg of simulation, i.e whether the intermediate P2 nodes were saved. Entering save_deg = 1 when the simulation was run with save_deg = 2 will result in only the corner nodes being used in postprocessing")
    
    args = parser.parse_args()

    return args.case, args.mesh, args.nu, args.dt, args.stride, args.save_deg

def read_command_line_stress():
    """Read arguments from commandline"""
    parser = ArgumentParser()

    parser.add_argument('--case', type=str, default="cyl_test", help="Path to simulation results",
                        metavar="PATH")
    parser.add_argument('--mesh', type=str, default="artery_coarse_rescaled", help="Mesh File Name",
                        metavar="PATH")
    parser.add_argument('--E_s', type=float, default=1e6, help="Elastic Modulus (Pascals) used in simulation")
    parser.add_argument('--nu_s', type=float, default=0.45, help="Poisson's Ratio used in simulation")
    parser.add_argument('--dt', type=float, default=0.001, help="Time step of simulation")
    parser.add_argument('--stride', type=int, default=1, help="Save frequency of simulation")    
    parser.add_argument('--save_deg', type=int, default=2, help="Input save_deg of simulation, i.e whether the intermediate P2 nodes were saved. Entering save_deg = 1 when the simulation was run with save_deg = 2 will result in only the corner nodes being used in postprocessing",
                        metavar="PATH")

    args = parser.parse_args()

    return args.case, args.mesh, args.E_s, args.nu_s, args.dt, args.stride, args.save_deg

def get_time_between_files(xdmf_file):
    # Get the time between output files from an xdmf file
    file1 = open(xdmf_file, 'r') 
    Lines = file1.readlines() 
    time_ts=[]
    
    # This loop goes through the xdmf output file and gets the time value (time_ts)
    for line in Lines: 
        if '<Time Value' in line:
            time_pattern = '<Time Value="(.+?)"'
            time_str = re.findall(time_pattern, line)
            time = float(time_str[0])
            time_ts.append(time)

    time_between_files = (time_ts[2] - time_ts[1]) # Calculate the time between files from xdmf file (in s)
    t_0 = time_ts[0]
    return t_0, time_between_files

def epsilon(u):
    """
    Computes the strain-rate tensor
    Args:
        u (Function): Velocity field

    Returns:
        epsilon (Function): Strain rate tensor of u
    """

    return 0.5 * (grad(u) + grad(u).T)


class STRESS:
    def __init__(self, u, p, nu, mesh):
        boundary_mesh = BoundaryMesh(mesh, 'exterior')
        self.bmV = VectorFunctionSpace(boundary_mesh, 'CG', 1)

        # Compute stress tensor
        sigma = (2 * nu * epsilon(u)) - (p * Identity(len(u)))

        # Compute stress on surface
        n = FacetNormal(mesh)
        F = -(sigma * n)

        # Compute normal and tangential components
        Fn = inner(F, n)  # scalar-valued
        Ft = F - (Fn * n)  # vector-valued

        # Integrate against piecewise constants on the boundary
        scalar = FunctionSpace(mesh, 'DG', 0)
        vector = VectorFunctionSpace(mesh, 'CG', 1)
        scaling = FacetArea(mesh)  # Normalise the computed stress relative to the size of the element

        v = TestFunction(scalar)
        w = TestFunction(vector)

        # Create functions
        self.Fn = Function(scalar)
        self.Ftv = Function(vector)
        self.Ft = Function(scalar)

        self.Ln = 1 / scaling * v * Fn * ds
        self.Ltv = 1 / (2 * scaling) * inner(w, Ft) * ds
        self.Lt = 1 / scaling * inner(v, self.norm_l2(self.Ftv)) * ds

    def __call__(self):
        """
        Compute stress for given velocity field u and pressure field p

        Returns:
            Ftv_mb (Function): Shear stress
        """

        # Assemble vectors
        assemble(self.Ltv, tensor=self.Ftv.vector())
        self.Ftv_bm = interpolate(self.Ftv, self.bmV)

        return self.Ftv_bm

    def norm_l2(self, u):
        """
        Compute norm of vector u in expression form
        Args:
            u (Function): Function to compute norm of

        Returns:
            norm (Power): Norm as expression
        """
        return pow(inner(u, u), 0.5)
