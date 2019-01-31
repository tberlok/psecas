from psecas.grids.grid import Grid


class ChebyshevRootsGrid(Grid):
    """
    This grid uses the Chebyshev Interior or Roots grid on
    z ∈ [zmin, zmax] to discretize the system.
    Implementation follows Boyd Appendix F.9 on page 571.

    N: The number of grid points
    zmin: The z value at the lower boundary
    zmax: The z value at the upper boundary

    Optional:
    z: a string which can be set to e.g. 'x' if x is used as the
       coordinate in your linearized equations.

    """

    def __init__(self, N, zmin, zmax, z="z"):
        super().__init__(N, zmin, zmax, z=z)

    def make_grid(self):
        import numpy as np

        N = self._N
        self.NN = N + 1
        L = self.L
        N = self.NN

        factor = L / 2

        d1 = np.zeros((N, N))
        zg = np.cos(np.pi * (2 * np.arange(1, N + 1) - 1) / (2 * N))
        zg = zg[::-1]
        Q = 1 - zg ** 2

        with np.errstate(divide='ignore'):
            for jj in range(N):
                d1[:, jj] = (-1)**(np.arange(N) + jj) * \
                    np.sqrt(Q[jj] / Q) / (zg - zg[jj])

        d1[np.diag_indices(N)] = 0.5 * zg / Q

        d2 = np.dot(d1, d1)
        self.zg = (zg + 1) * L/2 + self.zmin
        self.d0 = np.eye(self.NN)
        self.d1 = d1 / factor
        self.d2 = d2 / factor ** 2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        from numpy.polynomial.chebyshev import chebfit, chebval

        c, res = chebfit(self.zg, f, deg=self.N, full=True)
        # c = chebfit(grid.zg, f, deg=grid.N, full=False)
        return chebval(z, c)


def test_chebyshev_differentation(show=False):
    """Test the differentation routine of FourierGrid"""
    import numpy as np

    N = 20
    zmin = -1
    zmax = 1
    grid = ChebyshevRootsGrid(N, zmin, zmax)

    z = grid.zg
    y = np.exp(z) * np.sin(5 * z)
    yp_exac = np.exp(z) * (np.sin(5 * z) + 5 * np.cos(5 * z))
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()
        plt.title("Differentation with matrix (ChebyshevRootsGrid)")
        plt.plot(z, yp_exac, "-")
        plt.plot(z, yp_num, "--")
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-16)

    return (yp_num, yp_exac)


def test_chebyshev_interpolation(show=False):
    """Test the inperpolation routine of ChebyshevRootsGrid"""
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval

        return hermval(x, c) * np.exp(-x ** 2 / 2)

    N = 40
    zmin = -1.1
    zmax = 1.5
    grid = ChebyshevRootsGrid(N, zmin, zmax)

    grid_fine = ChebyshevRootsGrid(N * 4, zmin, zmax)
    z = grid_fine.zg

    y = np.exp(grid.zg) * np.sin(5 * grid.zg)
    y_fine = np.exp(z) * np.sin(5 * z)
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Chebyshev")
        plt.plot(z, y_fine, "-")
        plt.plot(z, y_interpolated, "--")
        plt.plot(grid.zg, y, "+")
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)
    return (y_fine, y_interpolated)


if __name__ == "__main__":
    (yp_num, yp_exac) = test_chebyshev_differentation(show=True)
    (y_fine, y_interpolated) = test_chebyshev_interpolation(show=True)
