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
        system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=1.0, kx=kx)
        solver = Solver(grid, system)

        Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 12) * 32))
        omega, v, err = solver.iterate_solver(Ns, verbose=False, tol=1e-8)

        return -omega.real

    a = 3.512831867406509
    b = 3.512831875508205
    (a, b) = golden_section(f, a, b, tol=1e-8)

# Create initial conditions for Athena simulation
if True:
    from psecas import save_system

    kxmax = 3.5128319
    grid = FourierGrid(N=256, zmin=0.0, zmax=2.0)
    system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=1.0, kx=kxmax)
    solver = Solver(grid, system)

    Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 12) * 32))
    omega, v, err = solver.iterate_solver(Ns, verbose=True, tol=1e-8)

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
    save_system(system, "./khi_hydro_delta.p")

    from psecas import write_athena

    # Write files for loading into Athena
    write_athena(
        system,
        Nz=256,
        Lz=2.0,
        path="./",
        name="khi_hydro_delta",
    )

    # Print out some information
    Lx = 2 * np.pi / system.kx
    print('')
    print('Eigenvalue is:', omega)
    print('Lx should be:', Lx)

    # Make a plot
    plt.figure(1)
    plt.plot(kxmax, omega.real, "+")
    plot_solution(system, filename='./khi_hydro_delta.pdf')

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
        'khi_hydro_delta.txt',
        perturb,
        delimiter="\t",
        newline="\n",
        fmt="%1.16e",
    )
