from .grid import Grid


class FourierGrid(Grid):
    def __init__(self, N, zmin, zmax):
        super().__init__(N, zmin, zmax)

    @property
    def dz(self):
        return self.L/self.N

    def make_grid(self):
        import numpy as np
        from numpy import sin, tan, arange, pi
        from scipy.linalg import toeplitz

        N = self._N
        self.NN = N
        zmin = self.zmin
        L = self.L

        factor = L/(2.0*np.pi)

        dz = 2.0*pi/N
        zg = dz*np.arange(1, N + 1) - dz/2

        column = np.hstack([0.0, .5*(-1.0)**np.arange(1, N) /
                           tan(np.arange(1, N)*dz/2.0)])
        d1 = toeplitz(column, column[np.hstack([0, range(N-1, 0, -1)])])

        y = np.hstack([-pi**2/(3*dz**2) - 1/6, -0.5*(-1)**arange(1, N) /
                      sin(dz*arange(1, N)/2)**2])
        d2 = toeplitz(y)
        self.zg = zg*L/(2*pi) - zmin
        self.d0 = np.eye(N)
        self.d1 = d1/factor
        self.d2 = d2/factor**2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        import numpy as np

        assert len(f) == self.N

        ak = 2*np.fft.rfft(f)/self.N
        n = np.arange(self.N//2 + 1)
        # n[self.N:] = 0

        def to_grid(z):
            cos = np.sum(ak[1:].real *
                         np.cos(2.0*np.pi*n[1:]*(z-self.dz/2)/self.L))
            sin = -np.sum(ak[1:].imag *
                          np.sin(2.0*np.pi*n[1:]*(z-self.dz/2)/self.L))
            y = ak[0].real/2.0 + cos + sin
            return y

        to_grid_v = np.vectorize(to_grid)

        return to_grid_v(z)
