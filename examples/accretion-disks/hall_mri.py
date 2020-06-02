import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, ChebyshevExtremaGrid, ChebyshevTLnGrid, System
from psecas import plot_solution

"""
    Global eigenmodes for the non-ideal (ohmic and Hall diffusion) MRI in a cylindrical
    unstratified differentially rotating system.

    Adapted from a Dedalus script by Gopakumar Mohandas. Details can be found
    in the paper:
    
    Leonardo Krapp, Oliver Gressel,
    Pablo Benítez-Llambay, Turlough P. Downes, Gopakumar Mohandas,
    and Martin E. Pessah, (2018), ApJ, 865, 2
    
    https://doi.org/10.3847/1538-4357/aadcf0

    This script can also solve the problem on the domain r ∈ [0, ∞], just
    change the grid creation to use the ChebyshevTLnGrid grid.

"""

# Make a Child of the System class
class HallMRI(System):
    def __init__(self, grid, kz, variables, eigenvalue):
        # Set parameters
        self.q = 1.5
        self.eta = 0.003
        self.lh = 1
        self.h = 0.25
        self.va = 0.002
        self.kz = kz

        super().__init__(grid, variables, eigenvalue)


# Create a grid
grid = ChebyshevExtremaGrid(N=128, zmin=1, zmax=2, z='r')
# grid =  ChebyshevTLnGrid(N=128, z='r')

variables = ['rho', 'vr', 'vphi', 'vz', 'Aphi', 'bphi']

kz = 2*np.pi

# Create the system
system = HallMRI(grid, kz, variables=variables, eigenvalue='sigma')

# The linearized equations
system.add_equation("-r*sigma*rho = r*dr(vr) + vr + 1j*kz*r*vz")
system.add_equation("-r*r*sigma*vr = - 2*r**(2-q)*vphi + h**2*r*r*dr(rho) + va**2*(DrAphi)")
system.add_equation("-sigma*vphi = + (2-q)*r**(-q)*vr - va**2*1j*kz*bphi")
system.add_equation("-sigma*vz = h**2*1j*kz*rho")
system.add_equation("-r*r*sigma*Aphi = + r*r*vr - eta*(DrAphi) + lh*va*1j*kz*r*r*bphi")
system.add_equation("-r*r*sigma*bphi = - 1j*kz*r*r*vphi - 1j*kz*q*r**(2-q)*Aphi - eta*(Drbphi) - lh*va*1j*kz*(DrAphi)")

# The boundary conditions
Aphi_bound = 'r**2*dr(dr(Aphi)) + r*dr(Aphi) - Aphi = 0'
system.add_boundary('vr', 'Dirichlet', 'Dirichlet')
system.add_boundary('vphi', 'Dirichlet', 'Dirichlet')
system.add_boundary('vz', 'Neumann', 'Neumann')
system.add_boundary('Aphi', Aphi_bound, Aphi_bound)
system.add_boundary('bphi', 'Dirichlet', 'Dirichlet')

system.add_substitution('DrAphi = r*r*dr(dr(Aphi)) + r*dr(Aphi) - Aphi - kz**2*r*r*Aphi')
system.add_substitution('Drbphi = r*r*dr(dr(bphi)) + r*dr(bphi) - bphi - kz**2*r*r*bphi')


solver = Solver(grid, system)

mode = 0
Ns = np.arange(1, 32) * 32 - 1
omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-4)
phi = np.arctan(vec[2].imag / vec[2].real)
solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)

if isinstance(grid, ChebyshevTLnGrid):
    plot_solution(system, limits=[0, 1.5])
    plt.xlim(0, 1.5)
else:
    plot_solution(system)


