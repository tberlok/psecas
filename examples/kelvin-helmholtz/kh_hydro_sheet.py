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

grid = ChebyshevRationalGrid(N=32, C=0.4)

u0 = 1.0
delta = 1.0
kx = 5.1540899
kx = 3.5128310

system = KelvinHelmholtzHydroOnlySlab(grid, u0, delta, kx)
system.boundaries = [True, True, True, True]

solver = Solver(grid, system)

Ns = np.arange(1, 32) * 32 - 1
omega, vec, err = solver.iterate_solver(Ns, mode=0, verbose=True)

xmin = 0
xmax = 2 * np.pi / kx
zmin = -4
zmax = 4

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

limits = [-2, 2]

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
