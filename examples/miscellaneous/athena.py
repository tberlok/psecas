import numpy as np
from psecas import Solver, FourierGrid, write_athena, save_system
from psecas.systems.kh_uniform import KelvinHelmholtzUniform

"""
    This example shows how the eigenmodes can be stored in text format which
    can be loaded into the MHD code Athena (Stone, J. et al, 2008).

    It also shows how to save a system using Python's pickle package.
"""

# Set up a grid
grid = FourierGrid(N=64, zmin=0, zmax=2)

# Set up the system of equations
kx = 4.627762711864407
# kx = 2*np.pi
system = KelvinHelmholtzUniform(grid, beta=1e3, nu=1e-2, kx=kx)

# Set up a solver
solver = Solver(grid, system)

# Iteratively solve
Ns = np.hstack((np.arange(1, 5) * 32, np.arange(3, 12) * 64))
solver.iterate_solver(Ns, verbose=True, tol=1e-10)

# Write files for loading into Athena
write_athena(system, Nz=256, Lz=2.0)

# Write directly to the Athena directory
write_athena(system, Nz=256, Lz=2.0, path='/Users/berlok/codes/athena/bin/')

save_system(system, '/Users/berlok/codes/athena/bin/kh-visc.p')
