def test_legendre_differentation(show=False):
    """Test the differentation routine of LegendreExtremaGrid"""
    from psecas import LegendreExtremaGrid
    import numpy as np

    N = 20
    zmin = -1
    zmax = 1
    grid = LegendreExtremaGrid(N, zmin, zmax)

    z = grid.zg
    y = np.exp(z) * np.sin(5 * z)
    yp_exac = np.exp(z) * (np.sin(5 * z) + 5 * np.cos(5 * z))
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()
        plt.title("Differentation with matrix (LegendreExtremaGrid)")
        plt.plot(z, yp_exac, "-")
        plt.plot(z, yp_num, "--")
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-16)

    return (yp_num, yp_exac)


def test_legendre_interpolation(show=False):
    """Test the inperpolation routine of LegendreExtremaGrid"""
    from psecas import LegendreExtremaGrid
    import numpy as np

    def psi(x, c):
        from numpy.polynomial.hermite import hermval

        return hermval(x, c) * np.exp(-x ** 2 / 2)

    N = 40
    zmin = -1.1
    zmax = 1.5
    grid = LegendreExtremaGrid(N, zmin, zmax)

    grid_fine = LegendreExtremaGrid(N * 4, zmin, zmax)
    z = grid_fine.zg

    y = np.exp(grid.zg) * np.sin(5 * grid.zg)
    y_fine = np.exp(z) * np.sin(5 * z)
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Legendre")
        plt.plot(z, y_fine, "-")
        plt.plot(z, y_interpolated, "--")
        plt.plot(grid.zg, y, "+")
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)
    return (y_fine, y_interpolated)


if __name__ == "__main__":
    (yp_num, yp_exac) = test_legendre_differentation(show=True)
    (y_fine, y_interpolated) = test_legendre_interpolation(show=True)
