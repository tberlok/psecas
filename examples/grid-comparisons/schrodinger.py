import numpy as np
import matplotlib.pyplot as plt
from freja import Solver, System
from freja import ChebyshevExtremaGrid, ChebyshevRootsGrid, LegendreExtremaGrid

"""
Solve the Schrödinger equation

    -ħ²/2m ∂²/∂x² Φ + V(x) Φ = E Φ

for the inifite well potential given by

    V(x) = 0 for 0 < x < L
    V(x) = ∞ otherwise

For this problem the eigenmodes are sinuisodal and the energies are given by

    E = n²ħ²π²/2mL²

This problem illustrates that the Gauss-Lobatto grids seem to be better at
handling problems with a boundary condition. Or alternatively, that we have a
bug in the ChebyshevRootsGrid.
"""

equation = "E*phi = hbar/(2*m)*dx(dx(phi))"


# Overwrite the default sorting method in the Solver class
class Example(Solver):
    def sorting_strategy(self, E):
        """Sorting strategy for hermite modes. E is a list of eigenvalues"""
        # Sort from smallest to largest eigenvalue
        index = np.argsort(np.abs(E))
        return (E, index)


L = 1
hbar = 1
m = 1

# Create grids
N = 128
zmin = 0
grid1 = ChebyshevExtremaGrid(N, zmin, zmax=L, z='x')
grid2 = ChebyshevRootsGrid(N, zmin, zmax=L, z='x')
grid3 = LegendreExtremaGrid(N, zmin, zmax=L, z='x')

grids = list([grid1, grid2, grid3])

# Number of solutions to plot for each grid
modes = 5

# Create figure
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=modes, nrows=3, sharey=True,
                         sharex=True)

for j, grid in enumerate(grids):
    # Create system
    system = System(grid, variables='phi', eigenvalue='E')
    system.hbar = hbar
    system.m = m

    # Add the first (and only) equation
    system.add_equation(equation, boundary=True)

    # Create a solver object
    solver = Example(grid, system)

    z = np.linspace(grid.zmin, grid.zmax, 1000)
    for mode in range(modes):
        E, vec = solver.solve(mode=mode)
        # Plottting
        axes[j, mode].set_title(r"$E/E_0 = ${:1.5f}".format(E.real/np.pi**2*2))
        axes[j, mode].plot(z, grid.interpolate(z, system.result['phi'].real))
        axes[j, mode].plot(z, grid.interpolate(z, system.result['phi'].imag))
        # axes[j, mode].set_ylim(-1, 1)
        # axes[j, mode].set_xlim(1e-2, grid.zmax)
    axes[j, 0].set_ylabel(type(grid).__name__)
plt.show()
