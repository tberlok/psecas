def test_fourier_differentation(show=False):
    """Test the differentation routine of FourierGrid"""
    import numpy as np
    from psecas import FourierGrid

    N = 256
    zmin = 0
    zmax = 2
    grid = FourierGrid(N, zmin, zmax)

    z = grid.zg
    y = np.tanh((z - 1.5) / 0.05) - np.tanh((z - 0.5) / 0.05) + 1.0
    yp_exac = (
        -np.tanh((z - 1.5) / 0.05) ** 2 / 0.05
        + np.tanh((z - 0.5) / 0.05) ** 2 / 0.05
    )
    yp_num = np.matmul(grid.d1, y)

    if show:
        import matplotlib.pyplot as plt

        plt.figure(1)
        plt.clf()
        plt.title("Differentation with matrix (FourierGrid)")
        plt.plot(z, yp_exac, "-")
        plt.plot(z, yp_num, "--")
        plt.show()

    np.testing.assert_allclose(yp_num, yp_exac, atol=1e-7)

    return (yp_num, yp_exac)


def test_fourier_interpolation(show=False):
    """Test the inperpolation routine of FourierGrid"""
    import numpy as np
    from psecas import FourierGrid

    N = 16
    zmin = 0
    zmax = np.pi * np.sqrt(2)
    grid = FourierGrid(N, zmin, zmax)

    grid_fine = FourierGrid(N * 4, zmin, zmax)
    z = grid_fine.zg

    # y = np.sin(5 * 2 * np.pi * grid.zg / grid.L)
    # y_fine = np.sin(5 * 2 * np.pi * z / grid.L)

    y = (
        np.sin(5 * 2 * np.pi * grid.zg / grid.L)
        * np.cos(2 * np.pi * grid.zg / grid.L) ** 2
    )
    y_fine = (
        np.sin(5 * 2 * np.pi * z / grid.L)
        * np.cos(2 * np.pi * z / grid.L) ** 2
    )

    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.clf()
        plt.title("Interpolation with Fourier series")
        plt.plot(z, y_fine, "-")
        plt.plot(z, y_interpolated, "--")
        plt.plot(grid.zg, y, "+")
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)

    return (y_fine, y_interpolated)


if __name__ == "__main__":
    (yp_num, yp_exac) = test_fourier_differentation(show=True)
    (y_fine, y_interpolated) = test_fourier_interpolation(show=True)
