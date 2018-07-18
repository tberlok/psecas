from evp.grids.grid import Grid


class ChebyshevTLnGrid(Grid):
    """
    Rational Chebyshev functions on y ∈ [0, ∞]: the TLn(y)

    See Boyd page 369.
    """
    def __init__(self, N, C=1, z='z'):
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

    def cheb_roots(self, N):
        import numpy as np
        d1 = np.zeros((N, N))
        zg = np.cos(np.pi*(2*np.arange(1, N+1)-1)/(2*N))
        Q = 1 - zg**2

        for ii in range(1, N+1):
            for jj in range(1, N+1):
                if ii == jj:
                    d1[ii-1, jj-1] = 0.5*zg[jj-1]/Q[jj-1]
                else:
                    d1[ii-1, jj-1] = (-1)**(ii+jj)*np.sqrt(Q[jj-1]/Q[ii-1]) / \
                                   (zg[ii-1] - zg[jj-1])

        return (zg, d1)

    def make_grid(self):
        import numpy as np
        C = self.C
        self.NN = self.N + 1
        N = self.NN

        d1 = np.zeros((N, N))
        [zg_int, d1_int] = self.cheb_roots(N)

        zg = C*(1 + zg_int)/(1 - zg_int)

        Q = (zg_int - 1)**2/(2*C)

        for ii in range(N):
            for jj in range(N):
                d1[ii, jj] = Q[ii]*d1_int[ii, jj]

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
        xg = (self.zg - self.C)/(self.zg + self.C)
        c, res = chebfit(xg, f, deg=self.N, full=True)
        x = (z - self.C)/(z + self.C)
        return chebval(x, c)


def test_differentation(show=False):
    """Test the differentation routine of the grid"""
    import numpy as np

    N = 200
    grid = ChebyshevTLnGrid(N, C=3)

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
        axes[0].set_title("Differentation with matrix (ChebyshevTLnGrid)")
        axes[0].semilogx(grid.zg, yp_exac, '-+', grid.zg, yp_num, '--')
        axes[1].semilogx(grid.zg, ypp_exac, '-+', grid.zg, ypp_num, '--')
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-10, rtol=1e-2)
    np.testing.assert_allclose(ypp_num, ypp_exac, atol=1e-10, rtol=1e-2)

    return (yp_num, yp_exac)


def test_interpolation(show=False):
    """Test the inperpolation routine of LaguerreGrid"""
    import numpy as np

    N = 200
    grid = ChebyshevTLnGrid(N, C=3)

    z = np.logspace(np.log10(grid.zmin), np.log10(grid.zmax), 2000)

    y = np.exp(-grid.zg)
    y_fine = np.exp(-z)
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with ChebyshevTLnGrid")
        plt.semilogx(z, y_fine, '-')
        plt.semilogx(z, y_interpolated, '--')
        plt.semilogx(grid.zg, y, '+')
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-10)
    return (y_fine, y_interpolated)


if __name__ == '__main__':
    test_differentation(show=True)
    test_interpolation(show=True)
