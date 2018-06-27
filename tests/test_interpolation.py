def test_fourier_interpolation(show=False):
    """Test the inperpolation routine of FourierGrid"""
    import numpy as np
    import matplotlib.pyplot as plt
    from evp import FourierGrid

    N = 16
    zmin = 0
    zmax = np.pi*np.sqrt(2)
    grid = FourierGrid(N, zmin, zmax)

    grid_fine = FourierGrid(N*4, zmin, zmax)
    z = grid_fine.zg

    y = np.sin(5*2*np.pi*grid.zg/grid.L)
    y_fine = np.sin(5*2*np.pi*z/grid.L)

    y = np.sin(5*2*np.pi*grid.zg/grid.L)*np.cos(2*np.pi*grid.zg/grid.L)**2
    y_fine = np.sin(5*2*np.pi*z/grid.L)*np.cos(2*np.pi*z/grid.L)**2

    y_interpolated = grid.interpolate(z, y)

    if show:
        plt.figure(1)
        plt.clf()
        plt.title("Interpolation with Fourier series")
        plt.plot(z, y_fine, '-')
        plt.plot(z, y_interpolated, '--')
        plt.plot(grid.zg, y, '+')
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)

    return (y_fine, y_interpolated)


def test_chebyshev_interpolation(show=False):
    """Test the inperpolation routine of ChebyshevExtremaGrid"""
    import numpy as np
    from evp import ChebyshevExtremaGrid

    def psi(x, c):
        from numpy.polynomial.hermite import hermval
        return hermval(x, c)*np.exp(-x**2/2)

    N = 40
    zmin = -1.1
    zmax = 1.5
    grid = ChebyshevExtremaGrid(N, zmin, zmax)

    grid_fine = ChebyshevExtremaGrid(N*4, zmin, zmax)
    z = grid_fine.zg

    y = np.exp(grid.zg)*np.sin(5*grid.zg)
    y_fine = np.exp(z)*np.sin(5*z)
    y_interpolated = grid.interpolate(z, y)

    if show:
        import matplotlib.pyplot as plt
        plt.figure(2)
        plt.clf()
        plt.title("Interpolation with Chebyshev")
        plt.plot(z, y_fine, '-')
        plt.plot(z, y_interpolated, '--')
        plt.plot(grid.zg, y, '+')
        plt.show()

    np.testing.assert_allclose(y_fine, y_interpolated, atol=1e-12)
    return (y_fine, y_interpolated)


def test_rational_chebyshev_interpolation(show=False):
    """Test the inperpolation routine of ChebyshevRationalGrid"""
    import numpy as np
    from evp import ChebyshevRationalGrid

    def psi(x, c):
        from numpy.polynomial.hermite import hermval
        return hermval(x, c)*np.exp(-x**2/2)

    N = 200
    grid = ChebyshevRationalGrid(N, L=4)

    grid_fine = ChebyshevRationalGrid(N*4, L=4)
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
    (y_fine, y_interpolated) = test_rational_chebyshev_interpolation(show=True)
    (y_fine, y_interpolated) = test_fourier_interpolation(show=True)
    (y_fine, y_interpolated) = test_chebyshev_interpolation(show=True)
