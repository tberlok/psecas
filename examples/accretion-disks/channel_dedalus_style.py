import numpy as np
import matplotlib.pyplot as plt
from freja import Solver, ChebyshevRationalGrid, System

# Create grid
grid = ChebyshevRationalGrid(N=199, L=1, z='r')


# Make a Child of the System class and override the make_background method
class Channel(System):
    def make_background(self):
        import numpy as np

        zg = self.grid.zg
        self.h = np.exp(-zg ** 2 / 2)


# Create the Channel system
system = Channel(grid, variables='f', eigenvalue='sigma')

# Add the first (and only) equation
system.add_equation("-h*sigma*f = 1*dr(dr(f)) +r*dr(f)", boundary=True)


# Overwrite the default sorting method in the Solver class
class ChannelSolver(Solver):
    def sorting_strategy(self, E):
        """Sorting strategy for channel modes. E is a list of eigenvalues"""
        E[E.real > 100.0] = 0
        E[E.real < -10.0] = 0
        index = np.argsort(np.real(E))[::-1]
        return (E, index)


# Create a solver object
solver = ChannelSolver(grid, system)

plt.figure(1)
plt.clf()
# Get the three fastest growing modes
modes = 3
fig, axes = plt.subplots(num=1, ncols=modes, sharey=True)
for mode in range(modes):
    # List of resolutions to try
    Ns = np.hstack((np.arange(1, 6) * 32, np.arange(2, 12) * 64))
    # Solve the system to a given tolerance
    omega, vec, err = solver.iterate_solver(
        Ns, mode=mode, verbose=True, tol=1e-6
    )
    # Plottting
    phi = np.arctan(vec[2].imag / vec[2].real)
    solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)
    axes[mode].set_title(
        r"$\sigma = ${:1.4f}".format(omega.real), fontsize=10
    )
    axes[mode].plot(grid.zg, system.result['f'].real)
    axes[mode].plot(grid.zg, system.result['f'].imag)
    axes[mode].set_xlim(-4, 4)
plt.show()
