import numpy as np
import matplotlib.pyplot as plt
from freja import Solver, ChebyshevExtremaGrid
from freja.systems.mti import MagnetoThermalInstability
from scipy.interpolate import interp1d

N = 32
zmin = 0
zmax = 1
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e5
Kn0 = 200

system = MagnetoThermalInstability(grid, beta, Kn0, only_interior=True)

kx = 2 * np.pi * 2
mti = Solver(grid, system, kx)

Ns = np.hstack((np.arange(1, 4) * 32, np.arange(2, 12) * 64))
omega, v, err = mti.iterate_solver2(Ns, i=2, tol=1e-8)
# mti.solver(i=5)


def interpolate(z, f):
    from numpy.polynomial.chebyshev import chebfit, chebval

    c, res = chebfit(grid.zg, f, deg=grid.N, full=True)
    # c = chebfit(grid.zg, f, deg=grid.N, full=False)
    return chebval(z, c)


y = mti.result['drho'].real

z = np.linspace(grid.zmin, grid.zmax - 0.01, 1000)

plt.figure(1)
plt.clf()
# plt.plot(z, ak[2]*np.exp(1j*z*kxs[2]))
plt.plot(z, interp1d(grid.zg, y)(z))
plt.plot(z, interpolate(z, y), '--')
plt.plot(grid.zg, y, '+')
plt.show()
