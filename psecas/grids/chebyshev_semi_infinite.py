from psecas.grids.grid import Grid


class ChebyshevTLnGrid(Grid):
    """
    This grid uses Rational Chebyshev functions on y ∈ [0, ∞], the TLn(y)
    functions, to dicretize the system (Boyd page 369 and Table E.7 p. 558)

    N: The number of grid points
    C: A scaling parameter which regulates the extent of the grid

    Optional:
    z: a string which can be set to e.g. 'x' if x is used as the
       coordinate in your linearized equations.

    The domain is in theory [0, ∞] but in practice the minimum and
    maximum values of the grid depend on both N and C.
    """

    def __init__(self, N, C=1, z="z"):
        self._observers = []

        self._N = N

        self._C = C

        # Grid variable name
        self.z = z

        self.make_grid()

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

        zg = np.cos(np.pi * (2 * np.arange(1, N + 1) - 1) / (2 * N))
        zg = zg[::-1]
        Q = 1 - zg ** 2

        with np.errstate(divide='ignore'):
            d1 = (-1)**(np.arange(N)[:, None] + np.arange(N)[None, :]) * \
                np.sqrt(Q[None, :] / Q[:, None]) / (zg[:, None] - zg[None, :])

        d1[np.diag_indices(N)] = 0.5 * zg / Q

        return (zg, d1)

    def make_grid(self):
        import numpy as np

        C = self.C
        self.NN = self.N + 1
        N = self.NN

        d1 = np.zeros((N, N))
        [zg_int, d1_int] = self.cheb_roots(N)

        zg = C * (1 + zg_int) / (1 - zg_int)

        Q = (zg_int - 1) ** 2

        d1 = Q[:, None] * d1_int / (2 * C)

        # TODO: Calculate d2 using an explicit formula. The above
        # breaks down at high N
        d2 = np.dot(d1, d1)
        self.zg = zg
        self.d0 = np.eye(N)
        self.d1 = d1
        self.d2 = d2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """See equations 17.37 and 17.38 in Boyd"""
        from numpy.polynomial.chebyshev import chebfit, chebval

        xg = (self.zg - self.C) / (self.zg + self.C)
        c, res = chebfit(xg, f, deg=self.N, full=True)
        x = (z - self.C) / (z + self.C)
        return chebval(x, c)
