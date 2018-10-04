import numpy as np
import matplotlib.pyplot as plt
from freja import Solver, System, LaguerreGrid, ChebyshevTLnGrid

"""
This problem is discussed in Chapter 17 in Boyd.
See especially equation 17.41 and 17.43 and 17.64a.

The exact eigenvalues are λ = n where n > 0 is an integer.

The Laguerre polynonimals are superior for this problem because they are
solutions to the differential equation.

Our interpolation method for Laguerre polynomials is seen to have some issues
at high z (and/or low N).
"""

# Equation 17.43 in Boyd
equation = "-sigma*w = y*dy(dy(w)) + dy(w) + (-1/2 - 1/4*y)*w"

# Equation 17.64a in Boyd (changed the sign of λ in Boyd's equation)
# equation = "-sigma*w = y*dy(dy(w)) + (1 - 1/4*y)*w"


# Overwrite the default sorting method in the Solver class
class Example(Solver):
    def sorting_strategy(self, E):
        """Sorting strategy. E is a list of eigenvalues"""
        E[E.real > 100.] = 0
        E[E.real < 1e-1] = 1e5
        index = np.argsort(np.real(E))
        return (E, index)


# Create grids
N = 64
grid2 = LaguerreGrid(N=N, C=1, z='y')
grid1 = ChebyshevTLnGrid(N=N, C=32, z='y')

grids = list([grid1, grid2])

# Number of solutions to plot for each grid
modes = 4

# Create figure
plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=modes, nrows=2, sharey=True,
                         sharex=True)

for j, grid in enumerate(grids):
    # Create system
    system = System(grid, variables='w', eigenvalue='sigma')

    # Add the first (and only) equation
    system.add_equation(equation, boundary=False)

    # Create a solver object
    solver = Example(grid, system)

    z = np.logspace(np.log10(grid.zmin+1e-2), np.log10(0.5e2), 4000)
    for mode in range(modes):
        omega, vec = solver.solve(mode=mode+8)
        # Plottting
        axes[j, mode].set_title(r"$\sigma = ${:1.5f}".format(omega.real))
        axes[j, mode].semilogx(z, grid.interpolate(z, system.result['w'].real))
        axes[j, mode].semilogx(z, grid.interpolate(z, system.result['w'].imag))
        axes[j, mode].set_ylim(-1, 1)
        axes[j, mode].set_xlim(1e-2, grid.zmax)
    axes[j, 0].set_ylabel(type(grid).__name__)
plt.show()
