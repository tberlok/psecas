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

    def to_coefficients(self, f):
        from numpy.polynomial.chebyshev import chebfit
        import numpy as np

        # Convert grid to standard xg = [-1, 1]
        xg = (self.zg - self.zmin)/self.L * 2. - 1.

        # Get coefficients for standard Chebyshev polynomials
        c, res = chebfit(xg, f, deg=self.N, full=True)

        return c

    def interpolate(self, z, f):
        from numpy.polynomial.chebyshev import chebval
        import numpy as np

        msg = "Can't interpolate outside grid domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        c = self.to_coefficients(f)

        # Convert z-values to standard xg = [-1, 1]
        x = (z - self.zmin)/self.L * 2. - 1.
        return chebval(x, c)