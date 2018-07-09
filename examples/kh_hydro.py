import numpy as np
from evp import Solver, FourierGrid
from evp.systems.kh_hydro import KelvinHelmholtzHydroOnly
import time

N = 64
zmin = 0
zmax = 2
grid = FourierGrid(N, zmin, zmax)

u0 = 1.5
delta = 100.0
kx = 5.

system = KelvinHelmholtzHydroOnly(grid, u0, delta, kx)

kh = Solver(grid, system)
# (omega, vec, err) = kh.iterate_solver(tol=1e-4)
# omega_bar = omega/(kx*u0)
omega_vec = []
kx_vec = np.logspace(0, 2, 10)
for kx in kx_vec:
    t1 = time.time()
    kh.system.kx = kx
    (omega, v) = kh.solver()
    omega_vec.append(omega)
    print(kx, omega)
    t2 = time.time()
    print("Solver took {} seconds".format(t2-t1))
omega_vec = np.array(omega_vec)
