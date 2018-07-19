import numpy as np
from evp import Solver, FourierGrid
from evp.systems.kh_uniform import KelvinHelmholtzUniform
from evp import golden_section


def f(kx, **kwargs):

    # Set up a grid
    grid = FourierGrid(N=64, zmin=0, zmax=2)
    system = KelvinHelmholtzUniform(grid, beta=1e3, nu=0, kx=kx)

    if 'nu' in kwargs.keys():
        system.nu = kwargs['nu']

    # Set up a solver
    solver = Solver(grid, system)

    # Iteratively solve
    Ns = np.hstack((np.arange(1, 5)*16, np.arange(3, 12)*32))
    omega, v, err = solver.iterate_solver(Ns, verbose=False, tol=1e-4)

    return -omega.real


(a, b) = golden_section(f, 3., 6, tol=1e-3, nu=0.0)
print(a, b, (a+b)/2, -f((a+b)/2))
(a, b) = golden_section(f, 3., 6, tol=1e-3, nu=1e-2)
print(a, b, (a+b)/2, -f((a+b)/2))
(a, b) = golden_section(f, 3., 6, tol=1e-3, nu=1e-1)
print(a, b, (a+b)/2, -f((a+b)/2))
