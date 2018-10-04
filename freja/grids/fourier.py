from freja.grids.grid import Grid


class FourierGrid(Grid):
    """
    This grid uses the Fourier grid on z ∈ [zmin, zmax] to dicretize the
    system. The grid is periodic.

    N: The number of grid points
    zmin: The z value at the lower boundary
    zmax: The z value at the upper boundary

    Optional:
    z: a string which can be set to e.g. 'x' if x is used as the
       coordinate in your linearized equations.

    """
    def __init__(self, N, zmin, zmax,  z='z'):
        super().__init__(N, zmin, zmax,  z=z)

    @property
    def dz(self):
        return self.L/self.N

    def make_grid(self):
        """
        Make the grid. We use the toeplitz implementation which is outlined
        in the book by Trefethen and the accompanying Matlab files.
        """
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


def test_fourier_differentation(show=False):
    """Test the differentation routine of FourierGrid"""
    import numpy as np
    from freja import FourierGrid

    N = 256
    zmin = 0
    zmax = 2
    grid = FourierGrid(N, zmin, zmax)

    z = grid.zg
    y = np.tanh((z-1.5)/0.05) - np.tanh((z-0.5)/0.05) + 1.0
    yp_exac = -np.tanh((z-1.5)/0.05)**2/0.05 + np.tanh((z-0.5)/0.05)**2/0.05
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        plt.title("Differentation with matrix (FourierGrid)")
        plt.plot(z, yp_exac, '-')
        plt.plot(z, yp_num, '--')
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-7)

    return (yp_num, yp_exac)


def test_fourier_interpolation(show=False):
    """Test the inperpolation routine of FourierGrid"""
    import numpy as np
    import matplotlib.pyplot as plt
    from freja import FourierGrid

    N = 16
    zmin = 0
    zmax = np.pi*np.sqrt(2)
    grid = FourierGrid(N, zmin, zmax)

    grid_fine = FourierGrid(N*4, zmin, zmax)
    z = grid_fine.zg

    y = np.sin(5*2*np.pi*grid.zg/grid.L)
    y_fine = np.sin(5*2*np.pi*z/grid.L)

    y = np.sin(5*2*np.pi*grid.zg/grid.L)*np.cos(2*np.pi*grid.zg/grid.L)**2
    y_fine = np.sin(5*2*np.pi*z/grid.L)*np.cos(2*np.pi*z/grid.L)**2

    y_interpolated = grid.interpolate(z, y)

    if show:
        plt.figure(1)
        plt.clf()
        plt.title("Interpolation with Fourier series")
        plt.plot(z, y_fine, '-')
        plt.plot(z, y_interpolated, '--')
        plt.plot(grid.zg, y, '+')
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)

    return (y_fine, y_interpolated)


if __name__ == '__main__':
    (yp_num, yp_exac) = test_fourier_differentation(show=True)
    (y_fine, y_interpolated) = test_fourier_interpolation(show=True)
