from freja import Solver, ChebyshevRationalGrid
from freja.systems.kh_hydro_slab import KelvinHelmholtzHydroOnlySlab
from freja import plot_solution
import numpy as np
import matplotlib.pyplot as plt

"""
    Find the eigenmodes of the KH instability on the domain z ∈ [-∞, ∞]
    assuming that the perturbations are zero at the boundaries.

    This the pure hydro version of the Kelvin-Helmholtz instability.
    The equilibrium changes sign at z=0 and is therefore not periodic.
"""

grid = ChebyshevRationalGrid(N=32, C=0.2)

u0 = 1.0
delta = 1.0
kx = 4.0

system = KelvinHelmholtzHydroOnlySlab(grid, u0, delta, kx)
system.boundaries = [True, True, True, True]

solver = Solver(grid, system)

Ns = np.arange(1, 32) * 32
omega, v, err = solver.iterate_solver(Ns, mode=0, verbose=True)

limits = [-2, 2]
plot_solution(system, limits=limits)
plt.xlim(limits[0], limits[1])
