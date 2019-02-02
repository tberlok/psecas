def test_rational_chebyshev_differentation(show=False):
    """Test the differentation routine of ChebyshevRationalGrid"""
    import numpy as np
    from psecas import ChebyshevRationalGrid

    def psi(x, c):
        from numpy.polynomial.hermite import hermval

        return hermval(x, c) * np.exp(-x ** 2 / 2)

    def dpsi(x, c):
        from numpy.polynomial.hermite import hermval, hermder

        yp = hermval(x, hermder(c)) * np.exp(-x ** 2 / 2) - x * psi(x, c)
        return yp

    def d2psi(x, c):
        """Second derivative of psi"""
        from numpy.polynomial.hermite import hermval, hermder

        yp = hermval(x, hermder(hermder(c))) * np.exp(-x ** 2 / 2)
        yp += -x * hermval(x, hermder(c)) * np.exp(-x ** 2 / 2)
        yp += -psi(x, c) - x * dpsi(x, c)
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
        axes[0].plot(grid.zg, yp_exac, "-", grid.zg, yp_num, "--")
        axes[1].plot(grid.zg, ypp_exac, "-", grid.zg, ypp_num, "--")
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
    from psecas import ChebyshevRationalGrid

    def psi(x, c):
        from numpy.polynomial.hermite import hermval

        return hermval(x, c) * np.exp(-x ** 2 / 2)

    N = 95
    grid = ChebyshevRationalGrid(N, C=4)

    grid_fine = ChebyshevRationalGrid(N * 4, C=4)
    z = grid_fine.zg

    y = psi(grid.zg, np.array(np.ones(4)))
    y_fine = psi(z, np.array(np.ones(4)))
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(3)
        plt.clf()
        plt.title("Interpolation with rational Chebyshev")
        plt.plot(z, y_fine, "-")
        plt.plot(z, y_interpolated, "--")
        plt.plot(grid.zg, y, "+")
        plt.xlim(-15, 15)
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)
    return (y_fine, y_interpolated)


if __name__ == "__main__":
    (yp_num, yp_exac) = test_rational_chebyshev_differentation(show=True)
    (y_fine, y_interpolated) = test_rational_chebyshev_interpolation(
        show=True
    )
