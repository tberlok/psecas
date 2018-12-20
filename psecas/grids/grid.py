class Grid:
    """
    Base class for grids.
    """

    def __init__(self, N, zmin, zmax, z='z'):
        self._observers = []

        assert zmax > zmin

        self._N = N
        self._zmin = zmin
        self._zmax = zmax
        self.make_grid()

        # Grid variable name
        self.z = z

    @property
    def L(self):
        return self.zmax - self.zmin

    def bind_to(self, callback):
        self._observers.append(callback)

    @property
    def N(self):
        return self._N

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax

    @N.setter
    def N(self, value):
        self._N = value
        self.make_grid()

    @zmin.setter
    def zmin(self, value):
        self._zmin = value
        self.make_grid()

    @zmax.setter
    def zmax(self, value):
        self._zmax = value
        self.make_grid()

    def der(self, vec):
        """First derivative of vec defined at zg"""
        import numpy as np

        assert type(vec) is np.ndarray
        assert vec.shape[0] == self.NN
        return np.matmul(self.d1, vec)

    def dder(self, vec):
        """Second derivative of vec defined at zg"""
        import numpy as np

        assert type(vec) is np.ndarray
        assert vec.shape[0] == self.NN
        return np.matmul(self.d2, vec)
