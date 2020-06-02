import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, ChebyshevRationalGrid, SincGrid, System

"""
Here we calculate channel modes for the magneto-rotational instability.
This problem is described in the paper

*MRI channel flows in vertically stratified models of accretion discs*,
https://doi.org/10.1111/j.1365-2966.2010.16759.x,
by Henrik N. Latter, Sebastien Fromang, Oliver Gressel

We combine their equation 16 into the following equation
for G:

    G = -1/K² d/dz (1/h G') = -1/K²(z/h G' + 1/h G'')

which becomes
    
    -h K² G = z G' + G''

The results produced from this script can be compared with
Figure 1 in the paper by Latter et al.

After solving for G, F can be found from 

    F = -1/(K h) G'

The division by h gives some numerical issues, since
h → 0 as z → ± ∞. Alternatively, we can solve their
equation 17 for F directly,

    F'' + K² h F = 0

by employing a Neumann boundary condition on F. This is explored in
the script channel.py.
"""

# Create grid
grid = ChebyshevRationalGrid(N=199, C=1, z='z')
# grid = SincGrid(N=299, C=1, z='z')


# Make a Child of the System class and override the make_background method
class Channel(System):
    def make_background(self):
        import numpy as np

        zg = self.grid.zg
        self.h = np.exp(-zg ** 2 / 2)


# Create the Channel system
system = Channel(grid, variables='G', eigenvalue='K2')

# Add the first (and only) equation
system.add_equation("-h*K2*G = dz(dz(G)) +z*dz(G)", boundary=True)


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
modes = 10
fig, axes = plt.subplots(num=1, ncols=2, nrows=modes, sharex=True,
    sharey='col')
for mode in range(modes):
    # List of resolutions to try
    Ns = np.hstack((np.arange(3, 6) * 32, np.arange(4, 12) * 64))
    # Solve the system to a given tolerance
    K2, vec, err = solver.iterate_solver(
        Ns, mode=mode, verbose=True, tol=1e-8, guess_tol=1e-3
    )
    # Plottting
    phi = np.arctan(vec[2].imag / vec[2].real)
    A = np.max(np.abs(vec))
    solver.keep_result(K2, vec * np.exp(-1j * phi)/A, mode=mode)
    axes[mode, 1].set_title(
        r"$K_n H = ${:1.4f}".format(np.sqrt(K2.real)), fontsize=10
    )
    zfine = np.linspace(-4, 4, 5000)
    axes[mode, 1].plot(zfine, grid.interpolate(zfine, system.result['G'].real))
    axes[mode, 1].plot(grid.zg, system.result['G'].real, '.')
    axes[mode, 1].set_xlim(-4, 4)
    axes[mode, 1].set_ylim(-1.3, 1.3)

    F = -1/(system.h*np.sqrt(K2.real))*grid.der(system.result['G'])
    axes[mode, 0].plot(grid.zg, F.real, '-')
    axes[mode, 0].set_xlim(-4, 4)
    axes[mode, 0].set_ylim(-3, 3)

axes[mode, 0].set_label(r'$z$')
axes[mode, 1].set_label(r'$z$')
plt.show()
