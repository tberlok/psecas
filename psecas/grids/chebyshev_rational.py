from psecas.grids.grid import Grid


class ChebyshevRationalGrid(Grid):
    """
        This grid uses Rational Chebyshev functions on z ∈ [-∞, ∞],
        the TBn(z) functions, to dicretize the system (Boyd page 356 and
        Table E.5 on page 556).

        N: The number of grid points
        C: A scaling parameter which regulates the extent of the grid

        Optional:
        z: a string which can be set to e.g. 'x' if x is used as the
           coordinate in your linearized equations.

        The domain is in theory [-∞, ∞] but in practice the minimum and
        maximum values of the grid depend on both N and C.
    """

    def __init__(self, N, C=1, z="z"):
        self._observers = []

        self._N = N
        self._C = C
        self.make_grid()

        # Grid variable name
        self.z = z

    def bind_to(self, callback):
        self._observers.append(callback)

    @property
    def N(self):
        return self._N

    @N.setter
    def N(self, value):
        self._N = value
        self.make_grid()

    @property
    def zmin(self):
        return self.zg.min()

    @property
    def zmax(self):
        return self.zg.max()

    @property
    def C(self):
        return self._C

    @C.setter
    def C(self, value):
        self._C = value
        self.make_grid()

    def cheb_roots(self, N):
        import numpy as np

        d1 = np.zeros((N, N))
        zg = np.cos(np.pi * (2 * np.arange(1, N + 1) - 1) / (2 * N))
        zg = zg[::-1]
        Q = 1 - zg ** 2

        with np.errstate(divide='ignore'):
            for jj in range(N):
                d1[:, jj] = (-1)**(np.arange(N) + jj) * \
                    np.sqrt(Q[jj] / Q) / (zg - zg[jj])

        d1[np.diag_indices(N)] = 0.5 * zg / Q

        return (zg, d1)

    def make_grid(self):
        import numpy as np

        C = self.C
        self.NN = self.N + 1
        N = self.NN

        d1 = np.zeros((N, N))
        [zg_int, d1_int] = self.cheb_roots(N)

        Q = 1 - zg_int ** 2.0

        zg = C * zg_int / np.sqrt(Q)

        d1 = d1_int / C * Q[:, None]**(3/2)

        d2 = np.dot(d1, d1)
        self.zg = zg
        self.d0 = np.eye(N)
        self.d1 = d1
        self.d2 = d2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def to_coefficients(self, f):
        from numpy.polynomial.chebyshev import chebfit
        import numpy as np

        # Convert infinite grid to xg = [-1, 1]
        xg = self.zg / np.sqrt(self.C ** 2 + self.zg ** 2)

        # Get coefficients for standard Chebyshev polynomials
        c, res = chebfit(xg, f, deg=self.N, full=True)

        return c

    def interpolate(self, z, f):
        """See equations 17.37 and 17.38 in Boyd"""
        from numpy.polynomial.chebyshev import chebval
        import numpy as np

        msg = "Can't interpolate outside grid domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        # Get coefficients for standard Chebyshev polynomials
        c = self.to_coefficients(f)

        # Convert infinite grid to xg = [-1, 1]
        x = z / np.sqrt(self.C ** 2 + z ** 2)

        # Evaluate the Chebyshev polynomial
        return chebval(x, c)
