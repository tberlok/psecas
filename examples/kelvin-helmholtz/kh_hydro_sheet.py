from psecas import Solver, ChebyshevRationalGrid
from psecas.systems.kh_hydro_sheet import KelvinHelmholtzHydroOnlySlab
from psecas import plot_solution, get_2Dmap
import numpy as np
import matplotlib.pyplot as plt

"""
    Find the eigenmodes of the KH instability on the domain z ∈ [-∞, ∞]
    assuming that the perturbations are zero at the boundaries.

    This the pure hydro version of the Kelvin-Helmholtz instability.
    The equilibrium changes sign at z=0 and is therefore not periodic.
"""

a = 0.05

grid = ChebyshevRationalGrid(N=32, C=8*a)

u0 = 2.0
delta = 0.0
kx = 2*np.pi

system = KelvinHelmholtzHydroOnlySlab(grid, u0, delta, kx, a=a)
system.boundaries = [True, True, True, True]

solver = Solver(grid, system)

Ns = np.arange(1, 32) * 128 - 1
omega, vec, err = solver.iterate_solver(Ns, mode=0, tol=1e-6,
                                        verbose=True)
print(f"Converged: {solver.system.result['converged']}")

if not solver.system.result['converged']:
    raise RuntimeError(f"Not converged! err={err}")
else:
    print(f'Converged with omega={omega}, err={err}')

xmin = 0
xmax = 2 * np.pi / kx
zmin = np.max([-4, solver.grid.zg.min()])
zmax = np.min([4, solver.grid.zg.max()])

Nx = 256
Nz = 1024

plt.rc("image", origin="lower", cmap="RdBu", interpolation="None")

extent = [xmin, xmax, zmin, zmax]

phi = np.arctan(vec[2].imag / vec[2].real)
solver.keep_result(omega, vec * np.exp(-1j * phi), mode=0)

# Normalize eigenmodes
y = np.vstack(
    [
        system.result["dvx"].real,
        system.result["dvx"].imag,
        system.result["dvz"].real,
        system.result["dvz"].imag,
    ]
)

limits = [zmin, zmax]

plot_solution(system, num=1, limits=limits, smooth=True)
plt.xlim(limits[0], limits[1])

maps = {
    key: get_2Dmap(system, key, xmin, xmax, Nx, Nz, zmin=zmin, zmax=zmax)
    for key in system.variables
}

plt.figure(2)
plt.clf()
fig, axes = plt.subplots(num=2)
axes.imshow(maps["dT"] + maps["drho"], extent=extent)
