from evp.grids.grid import Grid


class LaguerreGrid(Grid):
    def __init__(self, N, C=1, z='z'):
        self._observers = []

        self.maxN = 120
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
        # from numpy.polynomial import Laguerre as H
        self.NN = self.N + 1

        from dmsuite import lagdif

        zg, D = lagdif(self.NN, 2, 1/self.C)

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
        return barycentric_interpolate(self.zg, f, z)


def test_lagurre_differentation(show=False):
    """Test the differentation routine of LaguerreGrid"""
    import numpy as np

    N = 90
    grid = LaguerreGrid(N, C=3)

    y = np.exp(-grid.zg)
    yp_exac = -y
    ypp_exac = y
    yp_num = np.matmul(grid.d1, y)
    ypp_num = np.matmul(grid.d2, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=2)
        axes[0].set_title("Differentation with matrix (Laguerre)")
        axes[0].semilogx(grid.zg, yp_exac, '-+', grid.zg, yp_num, '--')
        axes[1].semilogx(grid.zg, ypp_exac, '-+', grid.zg, ypp_num, '--')
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-14)
    np.testing.assert_allclose(ypp_num, ypp_exac, atol=1e-14)

    return (yp_num, yp_exac)


def test_hermite_interpolation(show=False):
    """Test the inperpolation routine of LaguerreGrid"""
    import numpy as np

    N = 90
    grid = LaguerreGrid(N, C=3)

    z = np.linspace(0, 10, 2000)

    y = np.exp(-grid.zg)
    y_fine = np.exp(-z)
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Laguerre")
        plt.semilogx(z, y_fine, '-')
        plt.semilogx(z, y_interpolated, '--')
        plt.semilogx(grid.zg, y, '+')
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-16)
    return (y_fine, y_interpolated)


if __name__ == '__main__':
    test_lagurre_differentation(show=True)
    test_hermite_interpolation(show=True)
