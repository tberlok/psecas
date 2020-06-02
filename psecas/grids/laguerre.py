from psecas.grids.grid import Grid


class LaguerreGrid(Grid):
    """
        This grid uses Laguerre plynomials on y ∈ [0, ∞] to dicretize the
        system. dmsuite is used for the setup of the grid.

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

        self.maxN = 120
        msg = "It appears that dmsuite cannot handle N larger than {}"
        assert N <= self.maxN, msg.format(self.maxN)

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
        msg = "N = {} requested. Maximum allowed is {}"
        assert value <= self.maxN, msg.format(value, self.maxN)
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

    def make_grid(self):
        import numpy as np

        # from numpy.polynomial import Laguerre as H
        self.NN = self.N + 1

        from dmsuite import lagdif

        zg, D = lagdif(self.NN, 2, 1 / self.C)

        self.zg = zg
        self.d0 = np.eye(self.NN)
        self.d1 = D[0]
        self.d2 = D[1]

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """"""
        from scipy.interpolate import barycentric_interpolate
        import numpy as np

        msg = "Can't interpolate outside grid domain"
        assert np.array([z]).min() >= self.zmin, msg
        assert np.array([z]).max() <= self.zmax, msg

        return barycentric_interpolate(self.zg, f, z)
