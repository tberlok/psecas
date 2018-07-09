import numpy as np
from evp import Solver, FourierGrid, write_athena
from evp.systems.kh_uniform import KelvinHelmholtzUniform

# Set up a grid
grid = FourierGrid(N=64, zmin=0, zmax=2)

# Set up the system of equations
system = KelvinHelmholtzUniform(grid, beta=1e4, nu=1e-2, kx=2*np.pi)

# Set up a solver
solver = Solver(grid, system)

# Iteratively solve
Ns = np.hstack((np.arange(1, 5)*32, np.arange(3, 12)*64))
solver.iterate_solver(Ns, verbose=True)

# Write files for loading into Athena
write_athena(system, Nz=256, Lz=1.0)

# Write directly to the Athena directory
write_athena(system, Nz=256, Lz=1.0, path='/Users/berlok/codes/athena/bin/')
