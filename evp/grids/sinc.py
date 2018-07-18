from evp.grids.grid import Grid


class SincGrid(Grid):
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
        return self.C/np.sqrt(self.N)

    def make_grid(self):
        import numpy as np
        from scipy.linalg import toeplitz

        N = self.N
        self.NN = N

        zg = self.dz*(0.5 - N/2 + np.arange(N))
        n = np.arange(N)
        row = -np.hstack([0.0, (-1)**(n[1:]+1)/n[1:]])
        col = np.hstack([0.0, (-1)**(n[1:]+1)/n[1:]])
        col[0] = row[0]
        d1 = toeplitz(row, col)
        d1 /= self.dz

        row = np.hstack([-np.pi**2/3, -2*(-1)**(n[1:])/n[1:]**2])
        d2 = toeplitz(row)
        d2 /= self.dz**2

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
            return np.sum(f*np.sinc((z-self.zg)/self.dz))

        to_grid_v = np.vectorize(to_grid)

        return to_grid_v(z)


def test_sinc_differentation(show=False):
    """Test the differentation routine of ChebyshevRationalGrid"""
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval
        return hermval(x, c)*np.exp(-x**2/2)

    def dpsi(x, c):
        """Derivative of psi"""
        from numpy.polynomial.hermite import hermval, hermder
        yp = hermval(x, hermder(c))*np.exp(-x**2/2) - x*psi(x, c)
        return yp

    def d2psi(x, c):
        """Second derivative of psi"""
        from numpy.polynomial.hermite import hermval, hermder
        yp = hermval(x, hermder(hermder(c)))*np.exp(-x**2/2)
        yp += - x*hermval(x, hermder(c))*np.exp(-x**2/2)
        yp += - psi(x, c) - x*dpsi(x, c)
        return yp

    N = 200
    grid = SincGrid(N, C=5)

    c = np.ones(4)
    y = psi(grid.zg, c)
    yp_exac = dpsi(grid.zg, c)
    ypp_exac = d2psi(grid.zg, c)
    yp_num = np.matmul(grid.d1, y)
    ypp_num = np.matmul(grid.d2, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=2)
        axes[0].set_title("Differentation with matrix (Sinc)")
        axes[0].plot(grid.zg, yp_exac, '-', grid.zg, yp_num, '--')
        axes[1].plot(grid.zg, ypp_exac, '-', grid.zg, ypp_num, '--')
        for ax in axes:
            ax.set_xlim(-15, 15)
        axes[0].set_ylim(-15, 15)
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-12)
    np.testing.assert_allclose(ypp_num, ypp_exac, atol=1e-10)

    return (yp_num, yp_exac)


def test_sinc_interpolation(show=False):
    """Test the inperpolation routine of ChebyshevRationalGrid"""
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval
        return hermval(x, c)*np.exp(-x**2/2)

    N = 200
    grid = SincGrid(N, C=5)

    grid_fine = SincGrid(N*4, C=5)
    z = grid_fine.zg

    y = psi(grid.zg, np.array(np.ones(4)))
    y_fine = psi(z, np.array(np.ones(4)))
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Sinc")
        plt.plot(z, y_fine, '-')
        plt.plot(z, y_interpolated, '--')
        plt.plot(grid.zg, y, '+')
        plt.xlim(-15, 15)
        plt.ylim(-5, 10)
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)
    return (y_fine, y_interpolated)


if __name__ == '__main__':
    test_sinc_differentation(show=True)
    test_sinc_interpolation(show=True)
