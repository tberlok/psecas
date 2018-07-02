import numpy as np
from evp import Solver, FourierGrid
from evp.systems.kh_uniform import KelvinHelmholtzUniform
import pickle
import time
from mpi4py import MPI
comm = MPI.COMM_WORLD

base = './test'
kx_global = np.linspace(3, 4, 5)


steps = len(kx_global)
index_global = np.arange(steps)
index_local = index_global[comm.rank::comm.size]
kx_local = kx_global[comm.rank::comm.size]

grid = FourierGrid(N=64, zmin=0, zmax=2)
system = KelvinHelmholtzUniform(grid, beta=1e4, nu=1e-2)

kh = Solver(grid, system, kx_local[0])

if comm.rank == 0:
    pickle.dump(system, open('system.p', 'wb'))
    pickle.dump(grid, open('grid.p', 'wb'))

for i in range(len(kx_local)):
    t1 = time.time()
    kh.kx = kx_local[i]
    omega, v = kh.solver()
    filename = base + '-globalid-{}.p'.format(index_local[i])
    pickle.dump(kh.result, open(filename, 'wb'))
    t2 = time.time()
    f = open(base + '.log', 'a')
    msg = "Solved EVP with kx = {:1.4e} in {:1.2f} seconds. \
           Rank {} is {:2.0f}% done.\n".format(kx_local[i], t2-t1, comm.rank,
                                               (i+1)/len(kx_local)*100)
    f.write(msg)
    f.close()
