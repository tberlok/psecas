import numpy as np
from evp import Solver, ChebyshevExtremaGrid
from evp.systems.hbi import HeatFluxDrivenBouyancyInstability
from evp import plot_solution

N = 64
zmin = 0
zmax = 2
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e5
Kn = 1/1500
kx = 2*np.pi

system = HeatFluxDrivenBouyancyInstability(grid, beta, Kn, kx)

solver = Solver(grid, system)

mode = 3
# Ns = np.hstack((np.arange(2, 5)*16, np.arange(3, 12)*32))
# omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
omega, vec = solver.solve(mode=mode)

plot_solution(system, smooth=True)