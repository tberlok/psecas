from psecas.grids.grid import Grid


class FourierGrid(Grid):
    """
    This grid uses the Fourier grid on z ∈ [zmin, zmax] to dicretize the
    system. The grid is periodic.

    N: The number of grid points
    zmin: The z value at the lower boundary
    zmax: The z value at the upper boundary

    Optional:
    z: a string which can be set to e.g. 'x' if x is used as the
       coordinate in your linearized equations.

    """

    def __init__(self, N, zmin, zmax, z="z"):
        super().__init__(N, zmin, zmax, z=z)

    @property
    def dz(self):
        return self.L / self.N

    def make_grid(self):
        """
        Make the grid. We use the toeplitz implementation which is outlined
        in the book by Trefethen and the accompanying Matlab files.
        """
        import numpy as np
        from numpy import sin, tan, arange, pi
        from scipy.linalg import toeplitz

        N = self._N
        self.NN = N
        zmin = self.zmin
        L = self.L

        factor = L / (2.0 * np.pi)

        dz = 2.0 * pi / N
        zg = dz * np.arange(1, N + 1) - dz / 2

        column = np.hstack(
            [
                0.0,
                0.5
                * (-1.0) ** np.arange(1, N)
                / tan(np.arange(1, N) * dz / 2.0),
            ]
        )
        d1 = toeplitz(column, column[np.hstack([0, range(N - 1, 0, -1)])])

        y = np.hstack(
            [
                -pi ** 2 / (3 * dz ** 2) - 1 / 6,
                -0.5 * (-1) ** arange(1, N) / sin(dz * arange(1, N) / 2) ** 2,
            ]
        )
        d2 = toeplitz(y)
        self.zg = zg * L / (2 * pi) + zmin
        self.d0 = np.eye(N)
        self.d1 = d1 / factor
        self.d2 = d2 / factor ** 2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def to_coefficients(self, f):
        """Calculate the (shifted) complex Fourier coefficients"""
        import numpy as np

        assert len(f) == self.N

        ak = np.fft.fftshift(np.fft.fft(f, norm='ortho'))

        return ak

    def interpolate(self, z, f):
        import numpy as np

        assert len(f) == self.N

        akshift = self.to_coefficients(f)

        freq = np.fft.fftfreq(self.N)
        freq_shift = np.fft.fftshift(freq)
        # n is basically np.arange(-self.N//2, self.N//2) for even N
        n = freq_shift*self.N

        def to_grid(z):
            y = np.sum(akshift*np.exp(2*np.pi*1j*n*(z-self.dz/2)/self.L))
            return y/np.sqrt(self.N)

        to_grid_v = np.vectorize(to_grid)

        # Interpolate onto z
        f_interpolated = to_grid_v(z)

        # Only return real part as original input (f) is real
        if not np.iscomplexobj(f):
            f_interpolated = f_interpolated.real

        return f_interpolated
