def test_bessel_solutions(show=False):
    import numpy as np
    from psecas import Solver, System
    from psecas import (
        ChebyshevExtremaGrid,
        ChebyshevRootsGrid,
        LegendreExtremaGrid,
    )
    from scipy.special import spherical_jn

    """
    We solve the spherical Bessel equation

        r² d²f/dr² + 2 r df/dr +(κ²r² - l(l+1)) f = 0

    on the domain r=[0, 1] with the boundary condition f(1) = 0.
    The exact solutions are the spherical bessel functions of the first kind,
    j_l(κ r).

    The problem is taken from the paper

    Tensor calculus in spherical coordinates using Jacobi polynomials
    Part-II: Implementation and Examples, https://arxiv.org/pdf/1804.09283.pdf

    by Daniel Lecoanet, Geoff Vasil, Keaton Burns, Ben Brown and Jeff Oishi,
    and the results of our script can be compared with their Figure 2.

    A key difference is that we use shifted Chebyshev and Legendre grids, and
    that we enforce the boundary f(0)=0 explicitly. Since we do not treat the
    coordinate singularity at r=0 in using Jacobi polynomials in the clever
    way that Lecoanet+ do, the r^50 scaling shown in their inset is not
    recovered in our calculation.

    Nevertheless, the solutions for the eigenvalues and eigenfunctions are in
    very good agreement with the exact solution, in particular for the extrema
    grids.
    """

    # do_gen_evp = True
    # equation = "sigma*r**2*f = -r**2*dr(dr(f)) -2*r*dr(f) +l*(l+1)*f"

    do_gen_evp = False
    equation = "sigma*f = -dr(dr(f)) -2/r*dr(f) +l*(l+1)/r**2*f"

    # Overwrite the default sorting method in the Solver class
    class Example(Solver):
        def sorting_strategy(self, E):
            """Sorting strategy. E is a list of eigenvalues"""
            # Sort from smallest to largest eigenvalue
            index = np.argsort(np.abs(E))
            return (E, index)

    # Create grids
    N = 500
    zmin = 0
    # grid = ChebyshevRootsGrid(N, zmin, zmax=1, z='r')
    grid1 = ChebyshevExtremaGrid(N, zmin, zmax=1.0, z='r')
    grid2 = ChebyshevRootsGrid(N, zmin, zmax=1.0, z='r')
    grid3 = LegendreExtremaGrid(N, zmin, zmax=1.0, z='r')

    grids = list([grid1, grid2, grid3])

    if show:
        # Create figure
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=3, ncols=2, sharex=True)

    # https://keisan.casio.com/exec/system/1180573465
    kappa_ref = 389.4203848348835221199

    tols = [1e-13, 1e-4, 1e-13]

    for jj, grid in enumerate(grids):

        # Create system
        system = System(grid, variables='f', eigenvalue='sigma')
        system.l = 50

        # Add the first (and only) equation
        system.add_equation(equation, boundary=True)

        # Create a solver object
        solver = Example(grid, system, do_gen_evp=do_gen_evp)

        r = np.linspace(grid.zmin, grid.zmax, 4000)
        E, vec = solver.solve(mode=99)

        kappa = np.sqrt(E)

        # Normalize solution to bessel function exact solution at r = 0.5
        amp = grid.interpolate(0.5, system.result['f'].real) / spherical_jn(
            50, kappa_ref * 0.5
        )

        # Test statements
        print(grid)
        np.testing.assert_allclose(
            grid.interpolate(r, system.result['f'].real / amp),
            spherical_jn(50, kappa_ref * r),
            atol=tols[jj]
        )

        np.testing.assert_allclose(
            kappa, kappa_ref, atol=tols[jj], rtol=tols[jj]
        )

        if show:
            axes[jj, 0].plot(r, grid.interpolate(r, system.result['f'].real))
            axes[jj, 0].plot(r, grid.interpolate(r, system.result['f'].imag))
            axes[jj, 0].set_title(
                type(grid).__name__
                + r", $\kappa$-error={:1.3e}".format(kappa.real - kappa_ref)
            )

            axes[jj, 1].plot(
                r,
                grid.interpolate(r, system.result['f'].real / amp)
                - spherical_jn(50, kappa_ref * r),
            )
            # axes[jj, 1].plot(z)
            axes[jj, 1].set_title(
                r'$f(r) - j_\mathcal{l}(\kappa r)$, ' + type(grid).__name__
            )

    if show:
        plt.show()


if __name__ == '__main__':
    test_bessel_solutions(show=True)
