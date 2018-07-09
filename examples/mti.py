import numpy as np
from evp import Solver, ChebyshevExtremaGrid
from evp.systems.mti import MagnetoThermalInstability
from evp import plot_solution

N = 64
zmin = 0
zmax = 1
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e5
Kn0 = 200
kx = 2*np.pi*2

system = MagnetoThermalInstability(grid, beta, Kn0, kx, only_interior=True)
system.boundaries = [True, True, False, True, False]

mti = Solver(grid, system)

mode = 0
Ns = np.hstack((np.arange(1, 5)*32, np.arange(3, 12)*64))
omega, vec, err = mti.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
phi = np.arctan(vec[2].imag/vec[2].real)
mti.keep_result(omega, vec*np.exp(-1j*phi), mode=mode)

plot_solution(system, smooth=False)

# import time
# betas = np.logspace(4, 7, 4)
# steps = 10
# kx_vec = 2*np.pi*np.linspace(1, 35, steps)
# for beta in betas:
#     mti.beta = beta
#     omega_vec = []
#     for kx in kx_vec:
#         t1 = time.time()
#         mti.kx = kx
#         # (omega, err) = mti.iterate_solver_simple(tol=1e-4)
#         (omega, v) = mti.solver()
#         omega_vec.append(omega)
#         print(kx, omega)
#         t2 = time.time()
#         print("Solver took {} seconds".format(t2-t1))
#     omega_vec = np.array(omega_vec)
#     plt.plot(kx_vec, omega_vec, label=r"$\beta={}$".format(beta))
#     plt.show()
