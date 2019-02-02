from psecas.grids.grid import Grid


class ChebyshevExtremaGrid(Grid):
    """
    This grid uses the Chebyshev extrema and endpoints grid on
    z ∈ [zmin, zmax] to discretize the system. This grid is also known as the
    Gauss-Lobatto grid. Implementation follows Boyd Appendix F.8 on page 570.

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

        factor = L / 2

        zg = np.zeros(N + 1)
        d1 = np.zeros((N + 1, N + 1))
        for ii in range(N + 1):
            zg[ii] = np.cos(np.pi * ii / N)

        p = np.ones(N + 1)
        p[0] = 2
        p[N] = 2

        with np.errstate(divide='ignore'):
            for jj in range(N+1):
                d1[:, jj] = ((-1) ** (np.arange(N+1) + jj)
                             * p / (p[jj] * (zg - zg[jj])))
            d1[np.diag_indices(N+1)] = -zg / (2 * (1 - zg ** 2))

        d1[0, 0] = (1 + 2 * N ** 2) / 6
        d1[N, N] = -(1 + 2 * N ** 2) / 6

        d2 = np.dot(d1, d1)
        self.zg = -(zg - 1) * L / 2 + self.zmin
        self.d0 = np.eye(self.NN)
        self.d1 = -d1 / factor
        self.d2 = d2 / factor ** 2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        from numpy.polynomial.chebyshev import chebfit, chebval

        c, res = chebfit(self.zg, f, deg=self.N, full=True)
        # c = chebfit(grid.zg, f, deg=grid.N, full=False)
        return chebval(z, c)
