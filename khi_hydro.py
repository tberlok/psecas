import numpy as np
from psecas import Solver, FourierGrid
from psecas.systems.kh_hydro import KelvinHelmholtzHydroOnly
from psecas import plot_solution
import matplotlib.pyplot as plt

"""
"""

# Find the kx that gives maximum growth
if False:
    from psecas import golden_section

    def f(kx):
        grid = FourierGrid(N=64, zmin=0.0, zmax=2.0)
        system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=0.0, kx=kx)
        solver = Solver(grid, system)

        Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 20) * 32))
        omega, v, err = solver.iterate_solver(Ns, verbose=False, tol=1e-8)

        return -omega.real

    (a, b) = golden_section(f, 5.148550549911674, 5.158147443539172, tol=1e-8)
    a = 5.1540899488183065
    b = 5.154089957164513

# Create initial conditions for Athena simulation
if True:
    from psecas import write_athena, save_system

    kxmax = 5.1540899
    grid = FourierGrid(N=256, zmin=0.0, zmax=2.0)
    system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=0.0, kx=kxmax)
    solver = Solver(grid, system)

    Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 20) * 32))
    omega, v, err = solver.iterate_solver(Ns, verbose=True, tol=1e-10)

    # Normalize eigenmodes
    y = np.vstack(
        [
            system.result["dvx"].real,
            system.result["dvx"].imag,
            system.result["dvz"].real,
            system.result["dvz"].imag,
        ]
    )

    val = np.max(np.abs(y))
    for key in system.variables:
        system.result[key] /= val

    # Write files for loading into Athena
    write_athena(
        system, Nz=256, Lz=2.0, path="./athena-solutions/", name="khi_hydro"
    )

    save_system(system, "./athena-solutions/khi_hydro.p")

    plt.figure(1)
    plt.plot(kxmax, omega.real, "+")

    Lx = 2 * np.pi / system.kx

    print('')
    print('Eigenvalue is:', omega)
    print('Lx should be:', Lx)

    plot_solution(system, filename='./athena-solutions/khi_hydro.pdf')
