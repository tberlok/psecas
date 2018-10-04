import numpy as np
from freja import Solver, FourierGrid
from freja.systems.kh_uniform import KelvinHelmholtzUniform

N = 64
zmin = 0
zmax = 2
grid = FourierGrid(N, zmin, zmax)

beta = 1e4
nu = 1e-2

system = KelvinHelmholtzUniform(grid, beta, nu)

kx = 3.52615254237
kh = Solver(grid, system, kx)

# omega, v = kh.iterate_solver()

omega, v = kh.solver()

Ns = np.hstack((np.arange(1, 4)*32, np.arange(2, 12)*64))
omega, v, err = kh.iterate_solver2(Ns, tol=1e-8)
