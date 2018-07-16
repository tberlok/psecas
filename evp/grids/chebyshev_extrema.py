from .grid import Grid


class ChebyshevExtremaGrid(Grid):
    def __init__(self, N, zmin, zmax):
        super().__init__(N, zmin, zmax)

    def make_grid(self):
        import numpy as np

        N = self._N
        self.NN = N + 1
        L = self.L

        factor = L/2

        zg = np.zeros(N+1)
        d1 = np.zeros((N+1, N+1))
        for ii in range(N+1):
            zg[ii] = np.cos(np.pi*ii/N)

        p = np.ones(N+1)
        p[0] = 2
        p[N] = 2

        for ii in range(N+1):
            for jj in range(N+1):
                if ii == jj:
                    if ii == 0:
                        d1[ii, jj] = (1+2*N**2)/6
                    elif ii == N:
                        d1[ii, jj] = -(1+2*N**2)/6
                    else:
                        d1[ii, jj] = -zg[jj]/(2*(1-zg[jj]**2))
                else:
                    d1[ii, jj] = (-1)**(ii+jj)*p[ii]/(p[jj]*(zg[ii]-zg[jj]))

        d2 = np.dot(d1, d1)
        self.zg = -(zg - 1)*L/2 + self.zmin
        self.d0 = np.eye(self.NN)
        self.d1 = -d1/factor
        self.d2 = d2/factor**2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        from numpy.polynomial.chebyshev import chebfit, chebval
        c, res = chebfit(self.zg, f, deg=self.N, full=True)
        # c = chebfit(grid.zg, f, deg=grid.N, full=False)
        return chebval(z, c)
