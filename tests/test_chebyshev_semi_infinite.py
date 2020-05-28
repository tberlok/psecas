def test_differentation(show=False):
    """Test the differentation routine of the grid"""
    from psecas import ChebyshevTLnGrid
    import numpy as np

    N = 200
    grid = ChebyshevTLnGrid(N, C=3)
    assert grid.zg[0] < grid.zg[-1]

    grid.N = N
    assert grid.N == N
    grid.C = 3
    assert grid.C == 3

    y = np.exp(-grid.zg)
    yp_exac = -y
    ypp_exac = y
    yp_num = np.matmul(grid.d1, y)
    ypp_num = np.matmul(grid.d2, y)

    np.testing.assert_allclose(ypp_num, grid.dder(y), atol=1e-16)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()
        fig, axes = plt.subplots(num=1, nrows=2)
        axes[0].set_title("Differentation with matrix (ChebyshevTLnGrid)")
        axes[0].semilogx(grid.zg, yp_exac, "-+", grid.zg, yp_num, "--")
        axes[1].semilogx(grid.zg, ypp_exac, "-+", grid.zg, ypp_num, "--")
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-10, rtol=1e-2)
    np.testing.assert_allclose(ypp_num, ypp_exac, atol=1e-10, rtol=1e-2)

    return (yp_num, yp_exac)


def test_interpolation(show=False):
    """Test the inperpolation routine of LaguerreGrid"""
    from psecas import ChebyshevTLnGrid
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
        plt.semilogx(z, y_fine, "-")
        plt.semilogx(z, y_interpolated, "--")
        plt.semilogx(grid.zg, y, "+")
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-10)
    return (y_fine, y_interpolated)


if __name__ == "__main__":
    test_differentation(show=True)
    test_interpolation(show=True)
