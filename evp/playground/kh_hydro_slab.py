from evp import Solver, ChebyshevRationalGrid
from evp.systems.kh_hydro_slab import KelvinHelmholtzHydroOnlySlab
from evp import plot_solution
import numpy as np
import matplotlib.pyplot as plt

grid = ChebyshevRationalGrid(N=32, L=0.2)

u0 = 1.
delta = 1.

system = KelvinHelmholtzHydroOnlySlab(grid, u0, delta)
system.boundaries = [True, True, True, True]

kx = 4.
kh = Solver(grid, system, kx)

omega, v, err = kh.iterate_solver(np.arange(1, 32)*32, mode=0, verbose=True)

limits = [-2, 2]
plot_solution(kh, limits=limits)
plt.xlim(limits[0], limits[1])
