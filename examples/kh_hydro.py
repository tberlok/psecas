import numpy as np
from evp import Solver, FourierGrid
from evp.systems.kh_hydro import KelvinHelmholtzHydroOnly
import time
import matplotlib.pyplot as plt

N = 64
zmin = 0
zmax = 2
grid = FourierGrid(N, zmin, zmax)

u0 = 1.5
delta = 0.0

system = KelvinHelmholtzHydroOnly(grid, u0, delta, kx=0)

solver = Solver(grid, system)

omega_vec = []
kx_vec = np.linspace(0, 3.5, 20)
for kx in kx_vec:
    t1 = time.time()
    system.kx = kx
    (omega, v) = solver.solve()
    omega_vec.append(omega)
    print(kx, omega)
    t2 = time.time()
    print("Solver took {} seconds".format(t2-t1))
omega_vec = np.array(omega_vec)

plt.figure(1)
plt.clf()
plt.plot(kx_vec, omega_vec.real)
plt.xlabel(r"$k_x$")
plt.ylabel(r"$\omega$")
plt.show()
