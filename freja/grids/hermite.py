from freja.grids.grid import Grid


class HermiteGrid(Grid):
    """
        This grid uses Hermite polynomials on z ∈ [-∞, ∞] to dicretize the
        system. dmsuite is used for the setup of the grid.

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

        self.maxN = 245
        msg = 'It appears that dmsuite cannot handle N larger than {}'
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
        msg = 'N = {} requested. Maximum allowed is {}'
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
        # from numpy.polynomial import Hermite as H
        self.NN = self.N

        from dmsuite import herdif

        zg, D = herdif(self.NN, 2, 1/self.C)

        self.zg = zg
        self.d0 = np.eye(self.NN)
        self.d1 = D[0]
        self.d2 = D[1]

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        """"""
        # from numpy.polynomial.hermite import hermfit, hermval
        # c, res = hermfit(self.zg, f, deg=self.N, full=True)
        # return hermval(z, c)
        from scipy.interpolate import barycentric_interpolate
        return barycentric_interpolate(self.zg, f, z)


def test_hermite_differentation(show=False):
    """Test the differentation routine of ChebyshevRationalGrid"""
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval
        return hermval(x, c)*np.exp(-x**2/2)

    def dpsi(x, c):
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

    N = 245
    grid = HermiteGrid(N)

    c = np.ones(6)
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
        axes[0].set_title("Differentation with matrix (Hermite)")
        axes[0].plot(grid.zg, yp_exac, '-', grid.zg, yp_num, '--')
        axes[1].plot(grid.zg, ypp_exac, '-', grid.zg, ypp_num, '--')
        for ax in axes:
            ax.set_xlim(-15, 15)
        axes[0].set_ylim(-15, 15)
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-12)
    np.testing.assert_allclose(ypp_num, ypp_exac, atol=1e-12)

    return (yp_num, yp_exac)


def test_hermite_interpolation(show=False):
    """Test the inperpolation routine of ChebyshevRationalGrid"""
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval
        return hermval(x, c)*np.exp(-x**2/2)

    N = 200
    grid = HermiteGrid(N)

    # grid_fine = HermiteGrid(N*2, C=0.3)
    # z = grid_fine.zg
    z = np.linspace(-5.5, 5.5, 2000)

    y = psi(grid.zg, np.array(np.ones(4)))
    y_fine = psi(z, np.array(np.ones(4)))
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Hermite")
        plt.plot(z, y_fine, '-')
        plt.plot(z, y_interpolated, '--')
        plt.plot(grid.zg, y, '+')
        plt.xlim(-15, 15)
        plt.ylim(-5, 10)
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-8)
    return (y_fine, y_interpolated)


if __name__ == '__main__':
    test_hermite_differentation(show=True)
    test_hermite_interpolation(show=True)
