import numpy as np
from psecas import Solver, FourierGrid
from psecas.systems.kh_hydro import KelvinHelmholtzHydroOnly
import time
import matplotlib.pyplot as plt

"""
    The pure hydrodynamic version of the Kelvin-Helmholtz instability
    with a smoothly varying velocity profile. The equilibrium also allows
    for a density and temperature variation in z. The equilibrium is periodic
    with the velocity profile changing sign at z1 and z2 a Fourier grid
    is used to solve the EVP.

    The linearized equations for this setup are derived in

    Berlok, T. et al, in prep

    and the equilibrium has been previously been used for simulations in

    Lecoanet, D., McCourt, M., Quataert, E., Burns, K. J., Vasil, G. M.,
    Oishi, J. S., O’Leary, R. M. (2016).
    A validated non-linear kelvin-helmholtz benchmark for numerical
    hydrodynamics, MNRAS, 455(4), 4274–4288.
    https://doi.org/10.1093/mnras/stv2564
"""

# Plot omega vs kx
if True:
    grid = FourierGrid(N=64, zmin=0.0, zmax=2.0)
    system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=1.0, kx=0)
    solver = Solver(grid, system)

    omega_vec = []
    kx_vec = np.linspace(0.1, 8, 15)
    for kx in kx_vec:
        t1 = time.time()
        system.kx = kx
        (omega, v) = solver.solve()
        omega_vec.append(omega)
        print(kx, omega)
        t2 = time.time()
        print("Solver took {} seconds".format(t2 - t1))
    omega_vec = np.array(omega_vec)

    plt.figure(1)
    # plt.clf()
    plt.plot(kx_vec, omega_vec.real)
    plt.xlabel(r"$k_x$")
    plt.ylabel(r"$\omega$")
    plt.show()

# Find the kx that gives maximum growth
if False:
    from psecas import golden_section

    def f(kx):
        grid = FourierGrid(N=64, zmin=0.0, zmax=2.0)
        system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=1.0, kx=kx)
        solver = Solver(grid, system)

        Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 12) * 32))
        omega, v, err = solver.iterate_solver(Ns, verbose=False, tol=1e-6)

        return -omega.real

    (a, b) = golden_section(f, 3.512295, 3.513135, tol=1e-5)

# Create initial conditions for Athena simulation
if False:
    from psecas import write_athena, save_system

    kxmax = 3.5128286141291243
    grid = FourierGrid(N=64, zmin=0.0, zmax=2.0)
    system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=1.0, kx=kxmax)
    solver = Solver(grid, system)

    Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 12) * 32))
    omega, v, err = solver.iterate_solver(Ns, verbose=False, tol=1e-6)

    # Write files for loading into Athena
    # write_athena(system, Nz=256, Lz=2.0)

    # Write directly to the Athena directory
    write_athena(
        system, Nz=256, Lz=2.0, path='/Users/berlok/codes/athena/bin/'
    )

    save_system(system, '/Users/berlok/codes/athena/bin/kh-with-delta.p')

    plt.figure(1)
    plt.plot(kxmax, omega.real, '+')

    Lx = 2 * np.pi / system.kx
    print(Lx)
