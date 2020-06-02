import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, System
from psecas import (
    ChebyshevExtremaGrid,
    ChebyshevRootsGrid,
    LegendreExtremaGrid,
)
from scipy import special

"""
We solve the Airy equation

    d²f/dz² = σ z f

on the domain r=[-1, 1] with the boundary condition f(±1) = 0.
The exact solutions are the Airy functions Ai(σ^{1/3} z).

See https://mathworld.wolfram.com/AiryDifferentialEquation.html

and equation 8.36 in the book *Introduction to quantum mechanics*
by David J. Griffiths, second edition.
"""

do_gen_evp = True
equation = "sigma*z*f = dz(dz(f))"


# Overwrite the default sorting method in the Solver class
class Example(Solver):
    def sorting_strategy(self, E):
        """Sorting strategy. E is a list of eigenvalues"""
        # Sort from smallest to largest eigenvalue
        # We ignore the solutions with negative σ.
        E[E.real < 0.] = np.inf
        index = np.argsort(E)
        return (E, index)


# Create grids
N = 100
zmin = -1
zmax = 1
grid1 = ChebyshevExtremaGrid(N, zmin, zmax)
grid2 = ChebyshevRootsGrid(N, zmin, zmax)
grid3 = LegendreExtremaGrid(N, zmin, zmax)

grids = list([grid1, grid2, grid3])

def get_normalizing_constant(f1, f2):
    """EVP solver return eigenmodes with arbitrary sign"""
    abs_max1_ind = np.argmax(np.abs(f1))
    abs_max2_ind = np.argmax(np.abs(f2))

    val1 = f1[abs_max1_ind] 
    val2 = f2[abs_max2_ind]
    A = val1/val2
    return A


# Create figure
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, nrows=3, ncols=2, sharex=True)

for ii in range(2):
    for jj, grid in enumerate(grids):

        # Create system
        system = System(grid, variables='f', eigenvalue='sigma')

        # Add the first (and only) equation
        system.add_equation(equation, boundary=True)

        # Create a solver object
        solver = Example(grid, system, do_gen_evp=do_gen_evp)

        z = np.linspace(grid.zmin, grid.zmax, 4000)
        E, vec = solver.solve(mode=2+15*ii, saveall=True)

        ai, aip, bi, bip = special.airy((E)**(1/3)*z)

        A = get_normalizing_constant(ai.real, system.result['f'].real)
        axes[jj, ii].plot(z, grid.interpolate(z, A*system.result['f'].real))
        axes[jj, ii].plot(z, grid.interpolate(z, A*system.result['f'].imag))
        axes[jj, ii].plot(z, ai.real, 'k--')
        axes[jj, ii].plot(z, ai.imag, 'k--')
        axes[jj, ii].set_title(type(grid).__name__)


plt.show()
