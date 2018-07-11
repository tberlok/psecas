import numpy as np
from evp import Solver, FourierGrid
from evp.systems.kh_uniform import KelvinHelmholtzUniform
from evp import golden_section


def f(kx):

    # Set up a grid
    grid = FourierGrid(N=64, zmin=0, zmax=2)
    system = KelvinHelmholtzUniform(grid, beta=1e3, nu=1e-2, kx=kx)

    # Set up a solver
    solver = Solver(grid, system)

    # Iteratively solve
    Ns = np.hstack((np.arange(1, 5)*32, np.arange(3, 12)*64))
    omega, v, err = solver.iterate_solver(Ns, verbose=False, tol=1e-6)

    return -omega.real


# Just a guess
a = 1
b = 7

# # with tol=0.1
# a = 4.5278640450004204
# b = 4.6180339887498949

# # with tol=0.01
# a = 4.5410196624968453
# b = 4.5491502812526283

# # with tol=0.001
# a = 4.546777798673344
# b = 4.5475109361092017

# # with tol=1e-4
# a = 4.5470169759064083
# b = 4.5470830828677604

# # with tol=1e-5
# a = 4.5470385425040591
# b = 4.5470481873797377

# Create a generator, g, which yields the interval
# Note the next(g) method
g = golden_section(f, a, b, tol=1e-5)
for (a, b) in g:
    print(a, b)
