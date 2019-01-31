import numpy as np
from psecas import Solver, FourierGrid, IO
from psecas.systems.kh_uniform import KelvinHelmholtzUniform
import time
from mpi4py import MPI

comm = MPI.COMM_WORLD

"""
    This example illustrates how to use mpi4py to run a parameter study.

    The physics explored is that of the KH instability with anisotropic
    viscosity and a constant magnetic field in the x-direction.
    The equilibrium is assumed to have constant density, temperature and
    pressure. The velocity profile varies smoothly and the setup is periodic.

    More details about this specific setup can be found in

    Berlok et al, in prep.

    Another reference for the KHI with anisotric viscosity is

    Suzuki, K., Ogawa, T., Matsumoto, Y., & Matsumoto, R. (2013).
    Magnetohydrodynamic simulations of the formation of cold fronts in
    clusters of galaxies: Effects of anisotropic viscosity. Astrophysical
    Journal, 768(2). https://doi.org/10.1088/0004-637X/768/2/175
"""

directory = './data/'
kx_global = np.linspace(3, 4, 5)
kx_local = kx_global[comm.rank :: comm.size]

grid = FourierGrid(N=64, zmin=0, zmax=2)
system = KelvinHelmholtzUniform(grid, beta=1e4, nu=1e-2, kx=0)
io = IO(system, directory, __file__, len(kx_global))

solver = Solver(grid, system)

for i in range(len(kx_local)):
    t1 = time.time()
    system.kx = kx_local[i]
    omega, v = solver.solve()
    io.save_system(i)
    io.log(i, time.time() - t1, 'kx = {:1.4e}'.format(system.kx))

io.finished()
