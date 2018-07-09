from evp import Solver, ChebyshevRationalGrid
from evp.systems.kh_hydro_slab import KelvinHelmholtzHydroOnlySlab
from evp import plot_solution
import numpy as np
import matplotlib.pyplot as plt

grid = ChebyshevRationalGrid(N=32, L=0.2)

u0 = 1.
delta = 1.
kx = 4.

system = KelvinHelmholtzHydroOnlySlab(grid, u0, delta, kx)
system.boundaries = [True, True, True, True]

kh = Solver(grid, system)

omega, v, err = kh.iterate_solver(np.arange(1, 32)*32, mode=0, verbose=True)

limits = [-2, 2]
plot_solution(system, limits=limits)
plt.xlim(limits[0], limits[1])
