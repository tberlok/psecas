import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, FourierGrid, System

"""
The Mathieu equation is given by

    -uₓₓ + 2 q cos(2x) = σ u

where σ is the eigenvalue and q is a parameter.

We solve for u(x) and σ by using the Fourier grid.

This example is taken from the book *Spectral methods in Matlab*
by Lloyd Trefethen and this Python script reproduces the figure on
page 89.
"""

# Create grid
grid = FourierGrid(64, zmin=0, zmax=2*np.pi, z='x')

# Create the system
system = System(grid, variables='u', eigenvalue='sigma')
system.q = 1

# Add the first (and only) equation
system.add_equation("sigma*u = 2*q*np.cos(2*x)*u - dx(dx(u))")

# Create a solver object
solver = Solver(grid, system)

def sorting_strategy(E):
    """Sorting strategy. E is a list of eigenvalues"""
    # Sort from smallest to largest eigenvalue
    index = np.argsort(E)
    return (E, index)

solver.sorting_strategy = sorting_strategy

sigmas = []
for q in np.arange(0, 15, 0.2):

    # Set q in system object
    system.q = q

    # Solve
    solver.solve(saveall=True)

    # Append 11 lowest eigenvalues
    sigmas.append(solver.E[:11])

# Now plot the results
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1)
sigmas = np.array(sigmas)
for mode in range(11):
    axes.plot(np.arange(0, 15, 0.2), sigmas[:, mode].real)

axes.set_xlabel(r'$q$')
axes.set_ylabel(r'$\sigma$')
plt.show()
