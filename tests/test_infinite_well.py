def test_infinite_well(show=False):

    import numpy as np
    from psecas import Solver, System
    from psecas import (
        ChebyshevExtremaGrid,
        ChebyshevRootsGrid,
        LegendreExtremaGrid,
    )

    """
    This test solves the Schrödinger equation with three different grids.
    The problem is outlined as follows:

    Solve the Schrödinger equation

        -ħ²/2m ∂²/∂x² Φ + V(x) Φ = E Φ

    for the inifite well potential given by

        V(x) = 0 for 0 < x < L
        V(x) = ∞ otherwise

    For this problem the eigenmodes are sinuisodal and the energies are given
    by

        E = n²ħ²π²/2mL²

    This problem illustrates that the Gauss-Lobatto grids are better at
    handling problems with a boundary conditions since they have grid points
    at z=zmin and z=zmax.

    The test is performed twice, the second time the equation is
    multiplied by minus 1 on both sides. In the first case, Psecas evaluates
    mat2 to be the identity matrix and solves a standard evp.
    The second time, mat2 is not the identity matrix and Psecas therefore
    solves a generalized evp. This is more time consuming, and equation 1
    is therefore the preferred form.

    Psecas does not automatically convert equation 2 to equation 1,
    but simply warns the user that a rewrite of the equations could lead to a
    speed-up.
    """

    equation1 = "E*phi = hbar/(2*m)*dx(dx(phi))"
    equation2 = "-E*phi = -hbar/(2*m)*dx(dx(phi))"

    for equation in [equation1, equation2]:

        # Overwrite the default sorting method in the Solver class
        class Example(Solver):
            def sorting_strategy(self, E):
                """
                Sorting strategy.
                E is a list of eigenvalues
                """
                # Sort from smallest to largest eigenvalue
                index = np.argsort(np.abs(E))
                return (E, index)

        L = 1
        hbar = 1
        m = 1

        # Create grids
        N = 32
        zmin = 0
        grid1 = ChebyshevExtremaGrid(N, zmin, zmax=L, z='x')
        grid2 = ChebyshevRootsGrid(2 * N, zmin, zmax=L, z='x')
        grid3 = LegendreExtremaGrid(N, zmin, zmax=L, z='x')

        grids = list([grid1, grid2, grid3])
        tols = [1e-8, 1e-3, 1e-8]

        # Number of solutions to plot for each grid
        modes = 10

        if show:
            import matplotlib.pyplot as plt
            # Create figure
            plt.figure(1)
            plt.clf()
            fig, axes = plt.subplots(
                num=1, ncols=modes, nrows=3, sharey=True, sharex=True
            )

        for j, grid in enumerate(grids):
            # Create system
            system = System(grid, variables='phi', eigenvalue='E')
            system.hbar = hbar
            system.m = m

            # Add the first (and only) equation
            system.add_equation(equation, boundary=True)

            # Create a solver object
            solver = Example(grid, system)

            z = np.linspace(grid.zmin, grid.zmax, 1000)
            for mode in range(modes):
                E, vec = solver.solve(mode=mode)

                np.testing.assert_allclose(
                    E.real / np.pi ** 2 * 2,
                    -(mode + 1) ** 2,
                    atol=tols[j],
                    rtol=tols[j],
                )
                if show:
                    # Plottting
                    axes[j, mode].set_title(
                        r"$E/E_0 = ${:1.5f}".format(E.real / np.pi ** 2 * 2)
                    )
                    axes[j, mode].plot(
                        z, grid.interpolate(z, system.result['phi'].real)
                    )
                    axes[j, mode].plot(
                        z, grid.interpolate(z, system.result['phi'].imag)
                    )
                    axes[j, 0].set_ylabel(type(grid).__name__)

            if show:
                plt.show()


if __name__ == '__main__':
    test_infinite_well(show=True)
