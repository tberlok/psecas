import numpy as np
from evp import Solver, ChebyshevRationalGrid
from evp.systems.kh_hydro import KelvinHelmholtzHydroOnly
import time

N = 200

grid = ChebyshevRationalGrid(N, L=0.2)

u0 = 1.
delta = 0.0

system = KelvinHelmholtzHydroOnly(grid, u0, delta, z1=-0.5, z2=0.5)
system.boundaries = [True, True, True, True]

kx = 5.
kh = Solver(grid, system, kx)
# (omega, vec, err) = kh.iterate_solver(tol=1e-4)
# omega_bar = omega/(kx*u0)
omega_vec = []
kx_vec = np.logspace(-2, 2, 10)
for kx in kx_vec:
    t1 = time.time()
    kh.kx = kx
    (omega, v) = kh.solver()
    omega_vec.append(omega)
    print(kx, omega)
    t2 = time.time()
    print("Solver took {} seconds".format(t2-t1))
omega_vec = np.array(omega_vec)
