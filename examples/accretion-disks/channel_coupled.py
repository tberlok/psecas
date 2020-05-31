import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, ChebyshevRationalGrid, SincGrid, System

"""
Here we calculate channel modes for the magneto-rotational instability.
This problem is described in the paper

*MRI channel flows in vertically stratified models of accretion discs*,
https://doi.org/10.1111/j.1365-2966.2010.16759.x,
by Henrik N. Latter, Sebastien Fromang, Oliver Gressel

We solve their equation 16 as a coupled system:

    h K F = - G'
    K G = F'

The spurious mode found in channel.py also appears here.
"""

# Create grid
grid = ChebyshevRationalGrid(N=199, C=1, z='z')
# grid = SincGrid(N=599, C=2, z='z')


# Make a Child of the System class and override the make_background method
class Channel(System):
    def make_background(self):
        import numpy as np

        zg = self.grid.zg
        self.h = np.exp(-zg ** 2 / 2)


# Create the Channel system
system = Channel(grid, variables=['F', 'G'], eigenvalue='K')

# Add  equations
system.add_equation("h*K*F = -dz(G)", boundary=True)
system.add_equation("K*G = dz(F)", boundary=True)

system.extra_binfo = [['Neumann', 'Neumann'],
                       ['Dirichlet', 'Dirichlet']]

system.extra_binfo = [['dz(F) = 0', 'dz(F) = 0'],
                       ['G = 0', 'G = 0']]


# Overwrite the default sorting method in the Solver class
class ChannelSolver(Solver):
    def sorting_strategy(self, E):
        """Sorting strategy for channel modes. E is a list of eigenvalues"""
        E[E.real > 10.0] = 0
        E[E.real < -10.0] = 0
        index = np.argsort(np.real(E))[::-1]
        return (E, index)


# Create a solver object
solver = ChannelSolver(grid, system, do_gen_evp=True)

# Get the 10 fastest growing modes
modes = 11
plt.figure(2)
plt.clf()
fig2, axes2 = plt.subplots(num=2, ncols=2, nrows=modes, sharex=True)

plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, ncols=2, nrows=modes, sharex=True,
    sharey='col')

K, vec = solver.solve(mode=0, saveall=True)
for mode in range(modes):
    # List of resolutions to try
    # Ns = np.hstack((np.arange(3, 6) * 32, np.arange(4, 12) * 64))
    # Solve the system to a given tolerance
    # K, vec, err = solver.iterate_solver(
    #     Ns, mode=mode, verbose=True, tol=1e-6, guess_tol=5e-3
    # )

    # Plottting
    K, vec = solver.E[mode], solver.v[:, mode]
    phi = np.arctan(vec[2].imag / vec[2].real)
    solver.keep_result(K, vec * np.exp(-1j * phi), mode=mode)
    axes[mode, 1].set_title(
        r"$K_n H = ${:1.4f}".format(K.real), fontsize=10
    )
    # zfine = np.linspace(-4, 4, 5000)

    # axes[mode, 0].plot(zfine, grid.interpolate(zfine, system.result['F'].real))
    A1 = np.max(np.abs(system.result['F'].real))
    A2 = np.max(np.abs(system.result['G'].real))
    axes[mode, 0].plot(grid.zg, system.result['F'].real/A1, '-')
    axes[mode, 0].set_xlim(-4, 4)
    axes[mode, 0].set_ylim(-3, 3)

    # axes[mode, 1].plot(zfine, grid.interpolate(zfine, system.result['G'].real))
    axes[mode, 1].plot(grid.zg, system.result['G'].real/A2, '-')
    axes[mode, 1].set_xlim(-4, 4)
    axes[mode, 1].set_ylim(-1.3, 1.3)

    # axes2[mode, 0].semilogy(np.abs(grid.to_coefficients(system.result['F'])))
    # axes2[mode, 1].semilogy(np.abs(grid.to_coefficients(system.result['G'])))


axes[mode, 0].set_label(r'$z$')
axes[mode, 1].set_label(r'$z$')
plt.show()
