from psecas.grids.grid import Grid


class LegendreExtremaGrid(Grid):
    """
    This grid uses the Legendre extrema and endpoints grid on
    z ∈ [zmin, zmax] to discretize the system. This grid is also known as the
    Gauss-Lobatto grid. Implementation follows Boyd Appendix F.10 on page 572.

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
        from numpy.polynomial.legendre import legder, legroots, legval

        N = self._N
        self.NN = N + 1
        L = self.L

        factor = L / 2

        d1 = np.zeros((N + 1, N + 1))

        cp = legder([0] * N + [1])
        zg = np.hstack([-1.0, legroots(cp), 1.0])

        P_N = legval(zg, [0] * N + [1])

        with np.errstate(divide='ignore'):
            d1 = P_N[:, None] / (P_N[None, :] * (zg[:, None] - zg[None, :]))

        d1[np.diag_indices(N+1)] = 0.0
        d1[0, 0] = -N * (N + 1) / 4
        d1[N, N] = +N * (N + 1) / 4

        d2 = np.dot(d1, d1)

        self.zg = (zg + 1) * L / 2 + self.zmin
        self.d0 = np.eye(self.NN)
        self.d1 = d1 / factor
        self.d2 = d2 / factor ** 2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        from numpy.polynomial.legendre import legfit, legval

        c, res = legfit(self.zg, f, deg=self.N, full=True)
        # c = chebfit(grid.zg, f, deg=grid.N, full=False)
        return legval(z, c)
