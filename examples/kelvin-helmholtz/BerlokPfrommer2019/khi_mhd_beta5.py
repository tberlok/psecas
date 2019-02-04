import numpy as np
from psecas import Solver, FourierGrid
from psecas.systems.kh_uniform import KelvinHelmholtzUniform
from psecas import plot_solution
import matplotlib.pyplot as plt

"""
"""

# Find the kx that gives maximum growth
if True:
    from psecas import golden_section

    def f(kx):
        grid = FourierGrid(N=64, zmin=0.0, zmax=2.0)
        system = KelvinHelmholtzUniform(grid, beta=5, nu=0, kx=kx)
        solver = Solver(grid, system)

        Ns = np.hstack((np.arange(4, 5) * 16, np.arange(3, 16) * 32))
        omega, v, err = solver.iterate_solver(Ns, verbose=True, tol=1e-8)

        return -omega.real

    a = 1.423406443515539
    b = 6.423406450586558
    a = 5.577552015167967
    b = 5.577552023514174
    (a, b) = golden_section(f, a, b, tol=1e-8)

# Create initial conditions for Athena simulation
if True:
    from psecas import save_system

    kxmax = 5.5775520
    grid = FourierGrid(N=64, zmin=0.0, zmax=2.0)
    system = KelvinHelmholtzUniform(grid, beta=5, nu=0, kx=kxmax)
    solver = Solver(grid, system)

    Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 20) * 32))
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
    save_system(system, "./khi_mhd_beta5.p")

    # Print out some information
    Lx = 2 * np.pi / system.kx
    print('')
    print('Eigenvalue is:', omega)
    print('Lx should be:', Lx)

    # Make a plot
    plt.figure(1)
    plt.plot(kxmax, omega.real, "+")
    plot_solution(system, filename='./khi_mhd_beta5.pdf')

    # Write files for loading into Athena
    s = system
    c_dic = {}
    for key in s.variables:
        c_dic.update({key: s.grid.to_coefficients(s.result[key])})

    perturb = []
    for key in ['drho', 'dvx', 'dvz', 'dT', 'dA']:

        perturb.append(c_dic[key].real)
        perturb.append(c_dic[key].imag)

    perturb = np.transpose(perturb)
    np.savetxt(
        'khi_mhd_beta5.txt',
        perturb,
        delimiter="\t",
        newline="\n",
        fmt="%1.16e",
    )
