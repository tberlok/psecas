import numpy as np
from evp import Solver, FourierGrid, IO
from evp.systems.kh_uniform import KelvinHelmholtzUniform
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

directory = './data/'
kx_global = np.linspace(3, 4, 5)
kx_local = kx_global[comm.rank::comm.size]

grid = FourierGrid(N=64, zmin=0, zmax=2)
system = KelvinHelmholtzUniform(grid, beta=1e4, nu=1e-2, kx=0)
io = IO(system, directory, __file__, len(kx_global))

solver = Solver(grid, system)

for i in range(len(kx_local)):
    t1 = time.time()
    system.kx = kx_local[i]
    omega, v = solver.solve()
    io.save_system(i)
    io.log(i, time.time()-t1, 'kx = {:1.4e}'.format(system.kx))

io.finished()
