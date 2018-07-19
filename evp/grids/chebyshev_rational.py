from .grid import Grid


class ChebyshevRationalGrid(Grid):
    """
        This grid uses Rational Chebyshev functions on z ∈ [-∞, ∞],
        the TBn(z) functions, to dicretize the system (Boyd page 356 and
        Table E.5 on page 556).

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

        Q = 1 - zg_int**2.0

        zg = C*zg_int/np.sqrt(Q)

        for ii in range(N):
            for jj in range(N):
                d1[ii, jj] = np.sqrt(Q[ii])*Q[ii]*d1_int[ii, jj]/C

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
        import numpy as np
        xg = self.zg/np.sqrt(self.C**2 + self.zg**2)
        c, res = chebfit(xg, f, deg=self.N, full=True)
        x = z/np.sqrt(self.C**2 + z**2)
        return chebval(x, c)


def test_rational_chebyshev_differentation(show=False):
    """Test the differentation routine of ChebyshevRationalGrid"""
    import numpy as np
    from evp import ChebyshevRationalGrid

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

    N = 100
    grid = ChebyshevRationalGrid(N, C=4)

    c = np.ones(4)
    y = psi(grid.zg, c)
    yp_exac = dpsi(grid.zg, c)
    yp_num = np.matmul(grid.d1, y)
    ypp_exac = d2psi(grid.zg, c)
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


def test_rational_chebyshev_interpolation(show=False):
    """Test the inperpolation routine of ChebyshevRationalGrid"""
    import numpy as np
    from evp import ChebyshevRationalGrid

    def psi(x, c):
        from numpy.polynomial.hermite import hermval
        return hermval(x, c)*np.exp(-x**2/2)

    N = 95
    grid = ChebyshevRationalGrid(N, C=4)

    grid_fine = ChebyshevRationalGrid(N*4, C=4)
    z = grid_fine.zg

    y = psi(grid.zg, np.array(np.ones(4)))
    y_fine = psi(z, np.array(np.ones(4)))
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(3)
        plt.clf()
        plt.title("Interpolation with rational Chebyshev")
        plt.plot(z, y_fine, '-')
        plt.plot(z, y_interpolated, '--')
        plt.plot(grid.zg, y, '+')
        plt.xlim(-15, 15)
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)
    return (y_fine, y_interpolated)


if __name__ == '__main__':
    (yp_num, yp_exac) = test_rational_chebyshev_differentation(show=True)
    (y_fine, y_interpolated) = test_rational_chebyshev_interpolation(show=True)
