import numpy as np
import matplotlib.pyplot as plt
from freja import Solver, FourierGrid
from freja.systems.kh_uniform import KelvinHelmholtzUniform
from scipy.interpolate import interp1d

N = 128
zmin = 0
zmax = 2
grid = FourierGrid(N, zmin, zmax)

beta = 1e4
nu = 1e-2

system = KelvinHelmholtzUniform(grid, beta, nu)

kx = 3.52615254237
kh = Solver(grid, system, kx)

omega, v = kh.solver()
result = {var: v[j*grid.NN:(j+1)*grid.NN] for j,
          var in enumerate(system.variables)}
result.update({'omega': omega, 'kx': kx, 'zg': grid.zg,
               'variables': system.variables, 'mode': 0})


def interpolate(z, f):
    ak = np.fft.rfft(f)*grid.L/grid.N
    ak[0] /= 2
    kxs = 2*np.pi*np.fft.rfftfreq(grid.N)*grid.N/grid.L

    def to_grid(z):
        return np.sum(ak*np.exp(1j*z*kxs)).real
    to_grid_v = np.vectorize(to_grid)
    return to_grid_v(z)


y = result['dvz'].real

z = np.linspace(grid.zmin, grid.zmax-0.05, 1000)

plt.figure(1)
plt.clf()
# plt.plot(z, ak[2]*np.exp(1j*z*kxs[2]))
plt.plot(z, interp1d(grid.zg, y)(z))
plt.plot(z, interpolate(z, y), '--')
plt.plot(grid.zg, y, '+')
plt.show()
