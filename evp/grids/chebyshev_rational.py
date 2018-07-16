from .grid import Grid


class ChebyshevRationalGrid(Grid):
    def __init__(self, N, L, z='z'):
        self._observers = []

        self._N = N
        self._L = L
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
    def L(self):
        return self._L

    @property
    def zmin(self):
        return self.zg.min()

    @property
    def zmax(self):
        return self.zg.max()

    @L.setter
    def L(self, value):
        self._L = value
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
        L = self.L
        self.NN = self.N + 1
        N = self.NN

        d1 = np.zeros((N, N))
        [zg_int, d1_int] = self.cheb_roots(N)

        Q = 1 - zg_int**2.0

        zg = L*zg_int/np.sqrt(Q)

        for ii in range(N):
            for jj in range(N):
                d1[ii, jj] = np.sqrt(Q[ii])*Q[ii]*d1_int[ii, jj]/L

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
        xg = self.zg/np.sqrt(self.L**2 + self.zg**2)
        c, res = chebfit(xg, f, deg=self.N, full=True)
        x = z/np.sqrt(self.L**2 + z**2)
        return chebval(x, c)
