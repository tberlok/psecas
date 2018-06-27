class Grid():
    def __init__(self, N, zmin, zmax):
        self._observers = []

        assert zmax > zmin

        self._N = N
        self._zmin = zmin
        self._zmax = zmax
        self.make_grid()

    @property
    def L(self):
        return self.zmax - self.zmin

    def bind_to(self, callback):
        self._observers.append(callback)

    @property
    def N(self):
        return self._N

    @property
    def zmin(self):
        return self._zmin

    @property
    def zmax(self):
        return self._zmax
    
    @N.setter
    def N(self, value):
        self._N = value
        self.make_grid()

    @zmin.setter
    def zmin(self, value):
        self._zmin = value
        self.make_grid()

    @zmax.setter
    def zmax(self, value):
        self._zmax = value
        self.make_grid()


class FourierGrid(Grid):
    def __init__(self, N, zmin, zmax):
        super().__init__(N, zmin, zmax)

    @property
    def dz(self):
        return self.L/self.N

    def make_grid(self):
        import numpy as np
        from numpy import sin, tan, arange, pi
        from scipy.linalg import toeplitz

        N = self._N
        self.NN = N
        zmin = self.zmin
        zmax = self.zmax
        L = self.L

        factor = L/(2.0*np.pi)

        dz = 2.0*pi/N
        zg = dz*np.arange(1, N + 1) - dz/2
        # zg = dz*np.arange(N)

        column = np.hstack([0.0, 
                .5*(-1.0)**np.arange(1, N)/tan(np.arange(1, N)*dz/2.0)])
        d1 = toeplitz(column, column[np.hstack([0, range(N-1, 0, -1)])])

        y = np.hstack([-pi**2/(3*dz**2) - 1/6, 
                    -0.5*(-1)**arange(1, N)/sin(dz*arange(1, N)/2)**2]) 
        d2 = toeplitz(y)
        self.zg = zg*L/(2*pi) - zmin
        self.d0 = np.eye(N)
        self.d1 = d1/factor
        self.d2 = d2/factor**2

        # Call other objects that depend on the grid
        for callback in self._observers:
            callback()

    def interpolate(self, z, f):
        import numpy as np

        assert len(f) == self.N

        ak = 2*np.fft.rfft(f)/self.N
        n = np.arange(self.N//2 + 1)
        # n[self.N:] = 0
  
        def to_grid(z):
          cos = np.sum(ak[1:].real*np.cos(2.0*np.pi*n[1:]*(z-self.dz/2)/self.L))
          sin = -np.sum(ak[1:].imag*np.sin(2.0*np.pi*n[1:]*(z-self.dz/2)/self.L))
          y = ak[0].real/2.0 + cos + sin
          return y

        # y = np.zeros(self.N)
        # for i in range(self.N):
        #   y[i] = to_grid(z[i])

        to_grid_v = np.vectorize(to_grid)

        return to_grid_v(z)


class ChebyshevExtremaGrid(Grid):
    def __init__(self, N, zmin, zmax):
        super().__init__(N, zmin, zmax)

    def make_grid(self):
        import numpy as np

        N = self._N
        self.NN = N + 1
        zmin = self._zmin
        zmax = self._zmax
        L = self.L

        factor = L/2

        zg    = np.zeros(N+1)
        d1    = np.zeros((N+1,N+1))
        for ii in range(N+1):
          zg[ii]= np.cos(np.pi*ii/N)

        p    = np.ones(N+1)
        p[0] = 2
        p[N] = 2

        for ii in range(N+1):
          for jj in range(N+1):
            if ii==jj:
              if ii==0:
                d1[ii,jj]= (1+2*N**2)/6
              elif ii==N:
                d1[ii,jj] = -(1+2*N**2)/6
              else:
                d1[ii,jj]= -zg[jj]/(2*(1-zg[jj]**2))
            else:
              d1[ii,jj]= (-1)**(ii+jj)*p[ii]/(p[jj]*(zg[ii]-zg[jj]))

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

class ChebyshevRationalGrid():
    def __init__(self, N, L):
        self._observers = []

        self._N = N
        self._L = L
        self.make_grid()

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
      d1    = np.zeros((N, N))
      zg = np.cos(np.pi*(2*np.arange(1, N+1)-1)/(2*N))
      Q = 1 - zg**2

      for ii in range(1, N+1):
        for jj in range(1, N+1):
          if ii == jj:
            d1[ii-1, jj-1] = 0.5*zg[jj-1]/Q[jj-1]
          else:
            d1[ii-1, jj-1] = (-1)**(ii+jj)*np.sqrt(Q[jj-1]/Q[ii-1])/ \
                             (zg[ii-1] - zg[jj-1])

      return (zg, d1)

    def make_grid(self):
      import numpy as np
      L = self.L
      self.NN = self.N + 1
      N = self.NN

      d1    = np.zeros((N, N))
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