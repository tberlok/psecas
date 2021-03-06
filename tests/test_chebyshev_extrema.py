def test_chebyshev_differentation(show=False):
    """Test the differentation routine of FourierGrid"""
    from psecas import ChebyshevExtremaGrid
    import numpy as np

    N = 20
    zmin = -1
    zmax = 1
    grid = ChebyshevExtremaGrid(N, zmin, zmax)

    assert grid.zg[0] < grid.zg[-1]

    z = grid.zg
    y = np.exp(z) * np.sin(5 * z)
    yp_exac = np.exp(z) * (np.sin(5 * z) + 5 * np.cos(5 * z))
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()
        plt.title("Differentation with matrix (ChebyshevExtremaGrid)")
        plt.plot(z, yp_exac, "-")
        plt.plot(z, yp_num, "--")
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-16)

    return (yp_num, yp_exac)


def test_chebyshev_interpolation(show=False):
    """Test the inperpolation routine of ChebyshevExtremaGrid"""
    from psecas import ChebyshevExtremaGrid
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval

        return hermval(x, c) * np.exp(-x ** 2 / 2)

    N = 40
    zmin = -1.1
    zmax = 1.5
    grid = ChebyshevExtremaGrid(N, zmin, zmax)

    grid_fine = ChebyshevExtremaGrid(N * 4, zmin, zmax)
    z = grid_fine.zg

    y = np.exp(grid.zg) * np.sin(5 * grid.zg)
    y_fine = np.exp(z) * np.sin(5 * z)
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Chebyshev")
        plt.plot(z, y_fine, "-")
        plt.plot(z, y_interpolated, "--")
        plt.plot(grid.zg, y, "+")
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)
    return (y_fine, y_interpolated)


if __name__ == "__main__":
    (yp_num, yp_exac) = test_chebyshev_differentation(show=True)
    (y_fine, y_interpolated) = test_chebyshev_interpolation(show=True)
