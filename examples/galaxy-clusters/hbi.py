import numpy as np
from psecas import Solver, ChebyshevExtremaGrid
from psecas.systems.hbi import HeatFluxDrivenBuoyancyInstability
from psecas import plot_solution

"""
    The linear solution for the heat-flux-driven buoyancy instability (HBI)
    in a quasi-global setup, i.e. periodic in x and non-periodic in z.

    See the following paper for more details:

    H. N. Latter, M. W. Kunz, 2012, MNRAS, 423, 1964
    The HBI in a quasi-global model of the intracluster medium

    The script gets the unstable solution shown in figure 3 in the paper.
    Changing mode between 0, 1, 2 and 3 gives the solution shown in the four panels.
"""

N = 64
zmin = 0
zmax = 1
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e5
Kn = 1 / 1500.
kx = 250

system = HeatFluxDrivenBuoyancyInstability(grid, beta, Kn, kx)

solver = Solver(grid, system, True)

mode = 2
Ns = np.hstack((np.arange(2, 5) * 16, np.arange(3, 12) * 32))
omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
phi = np.arctan(vec[2].imag / vec[2].real)
solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)

plot_solution(system, smooth=True)


if True:
    # Plot 2D maps of the perturbations
    import matplotlib.pyplot as plt
    from psecas import get_2Dmap

    plt.rc("image", origin="lower", cmap="RdBu")
    plt.figure(2)
    plt.clf()
    fig, axes = plt.subplots(num=2, sharex=True, sharey=True, ncols=4)
    xmin = 0
    xmax = grid.zmax/10
    Nx = 512
    Nz = 1024
    extent = [xmin, xmax, system.grid.zmin, system.grid.zmax]

    dvx = get_2Dmap(system, "dvx", xmin, xmax, Nx, Nz)
    dvz = get_2Dmap(system, "dvz", xmin, xmax, Nx, Nz)
    dT = get_2Dmap(system, "dT", xmin, xmax, Nx, Nz)
    drho = get_2Dmap(system, "drho", xmin, xmax, Nx, Nz)

    axes[0].imshow(dvx, extent=extent)
    axes[0].set_title(r"$\delta v_x$")
    axes[1].imshow(dvz, extent=extent)
    axes[1].set_title(r"$\delta v_z$")
    axes[2].imshow(drho, extent=extent)
    axes[2].set_title(r"$\delta \rho/\rho$")
    axes[3].imshow(dT, extent=extent)
    axes[3].set_title(r"$\delta T/T$")

    axes[0].set_ylabel(r"$z/H_0$")
    for ax in axes:
        ax.set_xlabel(r"$x/H_0$")

    # Construct total vector potential
    dA = get_2Dmap(system, "dA", xmin, xmax, Nx, Nz)
    dx = (xmax - xmin) / Nx
    xg = (0.5 + np.arange(Nx)) * dx
    dz = (system.grid.zmax - system.grid.zmin) / Nz
    zg = (0.5 + np.arange(Nz)) * dz
    xx, zz = np.meshgrid(xg, zg)
    ampl = 5e-2
    A = ampl * dA + np.tile(xg, (Nz, 1))
    axes[0].contour(
        xx, zz, A, 32, colors="k", linestyles="solid", linewidths=0.5
    )
    plt.show()
