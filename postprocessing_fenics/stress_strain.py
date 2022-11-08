from dolfin import *
import numpy as np
import ufl  # ufl module
from os import path
#from turtleFSI.problems import *

def F_(d):
    """
    Deformation gradient tensor
    """
    return Identity(len(d)) + grad(d)


def J_(d):
    """
    Determinant of the deformation gradient
    """
    return det(F_(d))


def eps(d):
    """
    Infinitesimal strain tensor
    """
    return 0.5 * (grad(d) * inv(F_(d)) + inv(F_(d)).T * grad(d).T)

def E(d):
    """
    Green-Lagrange strain tensor
    """
    return 0.5*(F_(d).T*F_(d) - Identity(len(d)))
    
def S(d, lambda_s, mu_s):
    """
    Second Piola-Kirchhoff Stress (solid problem - Saint Venant-Kirchhoff materials)
    """
    I = Identity(len(d))
    return 2*mu_s*E(d) + lambda_s*tr(E(d))*I

def get_eig(T):
########################################################################
# Method for the analytical calculation of eigenvalues for 3D-Problems #
# from: https://fenicsproject.discourse.group/t/hyperelastic-model-problems-on-plotting-stresses/3130/6
########################################################################
    '''
    Analytically calculate eigenvalues for a three-dimensional tensor T with a
    characteristic polynomial equation of the form

                lambda^3 - I1*lambda^2 + I2*lambda - I3 = 0   .

    Since the characteristic polynomial is in its normal form , the eigenvalues
    can be determined using Cardanos formula. This algorithm is based on:
    "Efficient numerical diagonalization of hermitian 3 × 3 matrices" by
    J. Kopp (equations: 21-34, with coefficients: c2=-I1, c1=I2, c0=-I3).

    NOTE:
    The method implemented here, implicitly assumes that the polynomial has
    only real roots, since imaginary ones should not occur in this use case.
    This method only works for symmetrical (Hermetian) matrices.

    In order to ensure eigenvalues with algebraic multiplicity of 1, the idea
    of numerical perturbations is adopted from "Computation of isotropic tensor
    functions" by C. Miehe (1993). Since direct comparisons with conditionals
    have proven to be very slow, not the eigenvalues but the coefficients
    occuring during the calculation of them are perturbated to get distinct
    values.
    '''

    # determine perturbation from tolerance
    tol = 1e-8
    pert = 2*tol

    # get required invariants
    I1 = tr(T)                                                               # trace of tensor
    I2 = 0.5*(tr(T)**2-inner(T,T))                                        # 2nd invariant of tensor
    I3 = det(T)                                                              # determinant of tensor

    # determine terms p and q according to the paper
    # -> Follow the argumentation within the paper, to see why p must be
    # -> positive. Additionally ensure non-zero denominators to avoid problems
    # -> during the automatic differentiation
    p = I1**2 - 3*I2                                                            # preliminary value for p
    p = ufl.conditional(ufl.lt(p,tol),abs(p)+pert,p)                            # add numerical perturbation to p, if close to zero; ensure positiveness of p
    q = 27/2*I3 + I1**3 - 9/2*I1*I2                                             # preliminary value for q
    q = ufl.conditional(ufl.lt(abs(q),tol),q+ufl.sign(q)*pert,q)                # add numerical perturbation (with sign) to value of q, if close to zero

    # determine angle phi for calculation of roots
    phiNom2 =  27*( 1/4*I2**2*(p-I2) + I3*(27/4*I3-q) )                         # preliminary value for squared nominator of expression for angle phi
    phiNom2 = ufl.conditional(ufl.lt(phiNom2,tol),abs(phiNom2)+pert,phiNom2)    # add numerical perturbation to ensure non-zero nominator expression for angle phi
    phi = 1/3*ufl.atan_2(ufl.sqrt(phiNom2),q)                                   # calculate angle phi

    # calculate polynomial roots
    lambda1 = 1/3*(ufl.sqrt(p)*2*ufl.cos(phi)+I1)
    lambda3 = 1/3*(-ufl.sqrt(p)*(ufl.cos(phi)+ufl.sqrt(3)*ufl.sin(phi))+I1)
    lambda2 = 1/3*(-ufl.sqrt(p)*(ufl.cos(phi)-ufl.sqrt(3)*ufl.sin(phi))+I1)

    # return polynomial roots (eigenvalues)
    #eig = as_tensor([[lambda1 ,0 ,0],[0 ,lambda2 ,0],[0 ,0 ,lambda3]])
    return lambda1, lambda2, lambda3

def project_solid(tensorForm, fxnSpace, dx_s):
    #
    # This function projects a UFL tensor equation (tensorForm) using a tensor function space (fxnSpace)
    # on only the solid part of the mesh, given by the differential operator for the solid domain (dx_s)
    #
    # This is basically the same as the inner workings of the built-in "project()" function, but it
    # allows us to calculate on a specific domain rather than the whole mesh. For whatever reason, it's also 6x faster than
    # the built in project function...
    #
    v = TestFunction(fxnSpace) 
    u = TrialFunction(fxnSpace)
    a=inner(u,v)*dx_s # bilinear form
    L=inner(tensorForm,v)*dx_s # linear form
    tensorProjected=Function(fxnSpace) # output tensor-valued function
     
    # Alternate way that doesnt work on MPI (may be faster on PC)
    #quadDeg = 4 # Need to set quadrature degree for integration, otherwise defaults to many points and is very slow
    #solve(a==L, tensorProjected,form_compiler_parameters = {"quadrature_degree": quadDeg}) 
 
    '''
    From "Numerical Tours of Continuum Mechanics using FEniCS", the stresses can be computed using a LocalSolver 
    Since the stress function space is a DG space, element-wise projection is efficient
    '''
    solver = LocalSolver(a, L)
    solver.factorize()
    solver.solve_local_rhs(tensorProjected)

    return tensorProjected

def calculate_stress_strain(t, dvp_, verbose, visualization_folder,lambda_s, mu_s, mesh,dx_s, **namespace):

    # Files for storing extra outputs (stresses and strains)
    if not "ep_file" in namespace.keys():
        sig_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("TrueStress.xdmf")))
        ep_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("InfinitesimalStrain.xdmf")))
        sig_P_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("PrincipalStress.xdmf")))
        ep_P_file = XDMFFile(MPI.comm_world, str(visualization_folder.joinpath("PrincipalStrain.xdmf")))
        for tmp_t in [sig_file,ep_file,sig_P_file,ep_P_file]:
            tmp_t.parameters["flush_output"] = True
            tmp_t.parameters["rewrite_function_mesh"] = False

        return_dict = dict(ep_file=ep_file,sig_file=sig_file,ep_P_file=ep_P_file,sig_P_file=sig_P_file)
    
        namespace.update(return_dict)
    
    else:
        return_dict = {}
    
    # Split function
    d = dvp_["n"].sub(0, deepcopy=True)
    
    # Create tensor function space for stress and strain (this is necessary to evaluate tensor valued functions)
    '''
    Strain/stress are in L2, therefore we use a discontinuous function space with a degree of 1 for P2P1 elements
    Could also use a degree = 0 to get a constant-stress representation in each element
    For more info see the Fenics Book (P62, or P514-515), or
    https://comet-fenics.readthedocs.io/en/latest/demo/viscoelasticity/linear_viscoelasticity.html?highlight=DG#A-mixed-approach
    https://fenicsproject.org/qa/10363/what-is-the-most-accurate-way-to-recover-the-stress-tensor/
    https://fenicsproject.discourse.group/t/why-use-dg-space-to-project-stress-strain/3768
    '''

    Te = TensorElement("DG", mesh.ufl_cell(), 1) 
    Tens = FunctionSpace(mesh, Te)
    
    # Deformation Gradient and first Piola-Kirchoff stress (PK1)
    deformationF = F_(d) # calculate deformation gradient from displacement
    
    # Cauchy (True) Stress and Infinitesimal Strain (Only accurate for small strains, ask DB for True strain calculation...)
    epsilon = eps(d) # Form for Infinitesimal strain (need polar decomposition if we want to calculate logarithmic/Hencky strain)
    ep = project_solid(epsilon,Tens,dx_s) # Calculate stress tensor

    S_ = S(d, lambda_s, mu_s)  # Form for second PK stress (using St. Venant Kirchoff Model)
    sigma = (1/J_(d))*deformationF*S_*deformationF.T  # Form for Cauchy (true) stress 

    sig = project_solid(sigma,Tens,dx_s) # Calculate stress tensor

    # Calculate eigenvalues of the stress tensor (Three eigenvalues for 3x3 tensor)
    # Eigenvalues are returned as a diagonal tensor, with the Maximum Principal stress as 1-1
    eigStress = get_eig(sigma) 
    eigStrain = get_eig(epsilon)

    sig_P = project_solid(eigStress,Tens,dx_s) # Calculate Principal stress tensor
    ep_P = project_solid(eigStrain,Tens,dx_s) # Calculate Principal stress tensor

    # Name function
    ep.rename("InfinitesimalStrain", "ep")
    sig.rename("TrueStress", "sig")
    ep_P.rename("PrincipalInfStrain", "ep_P")
    sig_P.rename("PrincipalTrueStress", "sig_P")

    print("Writing Additional Viz Files for Stresses and Strains!")
    # Write results
    namespace["ep_file"].write(ep, t)
    namespace["sig_file"].write(sig, t)
    namespace["ep_P_file"].write(ep_P, t)
    namespace["sig_P_file"].write(sig_P, t)

    return return_dict