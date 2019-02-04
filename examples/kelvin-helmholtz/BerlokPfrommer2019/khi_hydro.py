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
    from psecas import save_system

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

    # Save system pickle object
    save_system(system, "./khi_hydro.p")

    # Print out some information
    Lx = 2 * np.pi / system.kx
    print('')
    print('Eigenvalue is:', omega)
    print('Lx should be:', Lx)

    # Make a plot
    plt.figure(1)
    plt.plot(kxmax, omega.real, "+")
    plot_solution(system, filename='./khi_hydro.pdf')

    # Write files for loading into Athena
    s = system
    c_dic = {}
    for key in s.variables:
        c_dic.update({key: s.grid.to_coefficients(s.result[key])})

    perturb = []
    for key in ['drho', 'dvx', 'dvz', 'dT']:

        perturb.append(c_dic[key].real)
        perturb.append(c_dic[key].imag)

    perturb = np.transpose(perturb)
    np.savetxt(
        'khi_hydro.txt',
        perturb,
        delimiter="\t",
        newline="\n",
        fmt="%1.16e",
    )
