from psecas.grids.grid import Grid


class SincGrid(Grid):
    """
        This grid uses Whittaker Cardinal or “Sinc” functions on z ∈ [-∞, ∞]
        to dicretize the system. See Boyd Appendix F.7 page 569.

        N: The number of grid points
        C: A scaling parameter which regulates the extent of the grid

        Optional:
        z: a string which can be set to e.g. 'x' if x is used as the
           coordinate in your linearized equations.

        The domain is in theory [-∞, ∞] but in practice the minimum and
        maximum values of the grid depend on both N and C.
    """

    def __init__(self, N, C=1, z='z'):
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

    @property
    def dz(self):
        import numpy as np

        return self.C / np.sqrt(self.N)

    def make_grid(self):
        import numpy as np
        from scipy.linalg import toeplitz

        N = self.N
        self.NN = N

        zg = self.dz * (0.5 - N / 2 + np.arange(N))
        n = np.arange(N)
        row = -np.hstack([0.0, (-1) ** (n[1:] + 1) / n[1:]])
        col = np.hstack([0.0, (-1) ** (n[1:] + 1) / n[1:]])
        col[0] = row[0]
        d1 = toeplitz(row, col)
        d1 /= self.dz

        row = np.hstack([-np.pi ** 2 / 3, -2 * (-1) ** (n[1:]) / n[1:] ** 2])
        d2 = toeplitz(row)
        d2 /= self.dz ** 2

        self.zg = zg
        self.d0 = np.eye(N)
        self.d1 = d1
        self.d2 = d2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """
        Interpolate f located at self.zg onto the grid z.

        This function uses Lagrange interpolation (eq. 4.6 in Boyd) with
        the sinc Cardinal functions (eq F.34 in Boyd)
        """
        import numpy as np

        assert len(f) == self.N

        def to_grid(z):
            return np.sum(f * np.sinc((z - self.zg) / self.dz))

        to_grid_v = np.vectorize(to_grid)

        return to_grid_v(z)
