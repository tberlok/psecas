from evp import Solver, ChebyshevRationalGrid
from evp.systems.kh_hydro_slab import KelvinHelmholtzHydroOnlySlab
from evp import plot_solution
import numpy as np
import matplotlib.pyplot as plt

"""
    Find the eigenmodes of the KH instability on the domain z ∈ [-∞, ∞]
    assuming that the perturbations are zero at the boundaries.

    This the pure hydro version of the Kelvin-Helmholtz instability.
    The equilibrium changes sign at z=0 and is therefore not periodic.
"""

grid = ChebyshevRationalGrid(N=32, C=0.2)

u0 = 1.
delta = 1.
kx = 4.

system = KelvinHelmholtzHydroOnlySlab(grid, u0, delta, kx)
system.boundaries = [True, True, True, True]

solver = Solver(grid, system)

Ns = np.arange(1, 32)*32
omega, v, err = solver.iterate_solver(Ns, mode=0, verbose=True)

limits = [-2, 2]
plot_solution(system, limits=limits)
plt.xlim(limits[0], limits[1])
