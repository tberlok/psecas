def test_fourier_differentation(show=False):
    """Test the differentation routine of FourierGrid"""
    import numpy as np
    from evp import FourierGrid

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


def test_chebyshev_differentation(show=False):
    """Test the differentation routine of FourierGrid"""
    import numpy as np
    from evp import ChebyshevExtremaGrid

    N = 20
    zmin = -1
    zmax = 1
    grid = ChebyshevExtremaGrid(N, zmin, zmax)

    z = grid.zg
    y = np.exp(z)*np.sin(5*z)
    yp_exac = np.exp(z)*(np.sin(5*z) + 5*np.cos(5*z))
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.title("Differentation with matrix (ChebyshevExtremaGrid)")
        plt.plot(z, yp_exac, '-')
        plt.plot(z, yp_num, '--')
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-16)

    return (yp_num, yp_exac)


def test_rational_chebyshev_interpolation(show=False):
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

    N = 100
    grid = ChebyshevRationalGrid(N, L=4)

    c = np.ones(4)
    y = psi(grid.zg, c)
    yp_exac = dpsi(grid.zg, c)
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(3)
        plt.clf()
        plt.title("Differentation with matrix (ChebyshevRationalGrid)")
        plt.plot(grid.zg, yp_exac, '-')
        plt.plot(grid.zg, yp_num, '--')
        plt.xlim(-15, 15)
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-12)

    return (yp_num, yp_exac)


if __name__ == '__main__':
    (yp_num, yp_exac) = test_fourier_differentation(show=True)
    (yp_num, yp_exac) = test_chebyshev_differentation(show=True)
    (yp_num, yp_exac) = test_rational_chebyshev_interpolation(show=True)
