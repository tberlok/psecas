# import IPython
class Solver():
    """docstring for Solver"""
    def __init__(self, grid, system, kx):
        self.grid = grid
        self.system = system
        self._kx = kx

    @property
    def kx(self):
        return self._kx

    @kx.setter
    def kx(self, value):
        self._kx = value
        self._get_matrix()

    def _set_submatrix(self, mat1, submat, eq_n, var_n, boundary):
        """Set submatrix corresponding to the term proportional to var_n
        (variable number) in eq_n (equation number). """
        NN = self.grid.NN
        N = self.grid.N
        if boundary:
            submat[0, :] = 0
            submat[N, :] = 0
            if eq_n == var_n:
                submat[0, N] = 1
                submat[N, 0] = 1
        mat1[(eq_n-1)*NN:eq_n*NN, (var_n-1)*NN:var_n*NN] = submat

    def _set_boundary(self, var_n):
        NN = self.grid.NN
        self.mat2[(var_n-1)*NN, (var_n-1)*NN] = 0.0
        self.mat2[var_n*NN-1,var_n*NN-1] = 0.0

    def _find_submatrices(self, eq, verbose=False):
      import numpy as np

      globals().update(self.grid.__dict__)
      globals().update(self.system.__dict__)

      kx = self.kx

      import re
      mats = [np.zeros((NN, NN), dtype=np.complex128) for i in range(dim)]

      if verbose:
        print('\nParsing equation:', eq)
      s = re.split(r" ", eq)

      for term in s:
          if len(term) == 0:
              s.remove('')
      for term in s:
          if verbose:
            print('\tParsing term:', term)
          for i, var in enumerate(variables):
              # No derivative
              s0 = "*" + var
              s1 = "*dz(" + var + ")"
              s2 = "*dz(dz(" + var + "))"
              if term.endswith(s0):
                  if verbose:
                    print('\t\tFound ', s0)
                  res = eval(term[:-len(s0)])
                  mats[i] += (res*d0.T).T
              # 1st derivative
              if term.endswith(s1):
                  if verbose:
                    print('\t\tFound ', s1)
                  res = eval(term[:-len(s1)])
                  mats[i] += (res*d1.T).T
              # 2nd derivative
              if term.endswith(s2):
                  if verbose:
                    print('\t\tFound ', s2)
                  res = eval(term[:-len(s2)])
                  mats[i] += (res*d2.T).T
      return mats

    def _get_matrix(self):
        import numpy as np
        dim = self.system.dim
        NN = self.grid.NN
        equations = self.system.equations
        variables = self.system.variables
        boundaries = self.system.boundaries

        # Construct matrix mat1
        mat1 = np.zeros((dim*NN,dim*NN), dtype="complex128")
        if boundaries is not None:
            self.mat2 = np.eye(dim*NN)
            for j, equation in enumerate(equations):
                if boundaries[j]:
                  self._set_boundary(j+1)


        for j, equation in enumerate(equations):
            mats = self._find_submatrices(equation)
            for i, variable in enumerate(variables):
                if boundaries is not None:
                  self._set_submatrix(mat1, mats[i], j+1, i+1, boundaries[j])
                else:
                  self._set_submatrix(mat1, mats[i], j+1, i+1, False)

        self.mat1 = mat1

    def keep_result(self, omega, vec):

      # Store result
      self.result = {var:vec[j*grid.NN:(j+1)*grid.NN] for j,
                     var in enumerate(self.system.variables)}
      self.result.update({'omega':omega, 'kx':self.kx, 'zg':self.grid.zg,
                          'variables':self.system.variables, 'm':0})
      # IPython.embed()

    def solver(self, guess=None, useOPinv=False, verbose=False, i=0):
        import numpy as np
        from scipy.linalg import eig

        boundaries = self.system.boundaries

        # Calculate matrix
        self._get_matrix()

        if guess is None:
            if boundaries is None:
              E, V = eig(self.mat1)
            else:
              # IPython.embed()
              E, V = eig(self.mat1, self.mat2)

            # Zero out large values which are most likely numerical
            E[np.abs(E.real) > 10.] = 0
            E[np.abs(E.imag) > 10.] = 0

            # Sort from largest to smallest eigenvalue
            index = np.argsort(np.real(E))[::-1]

            # Choose the largest value only
            # i = range(10)
            # Construct dictionary of perturbation variables
            omega = E[index[i]]
            v = V[:, index[i]]
            if verbose:
                print("N: {}, all eigenvalues: {}".format(self.grid.N, omega))
        else:
            from scipy.sparse.linalg import eigs
            from scipy import sparse
            smat = sparse.csr_matrix(self.mat1)
            if useOPinv:
              from numpy.linalg import pinv as inv
              if boundaries is None:
                # IPython.embed()
                OPinv = inv(self.mat1 - guess*np.eye(self.mat1.shape[0]))

              else:
                OPinv = inv(self.mat1 - guess*self.mat2)
              omega, v = eigs(smat, k=1, sigma=guess, OPinv=OPinv)
            else:
              if boundaries is None:
                omega, v = eigs(self.mat1, k=1, sigma=guess)
              else:
                omega, v = eigs(self.mat1, M=self.mat2, k=1, sigma=guess)
            omega = omega[0]
            v = np.squeeze(v)
            if verbose:
                print("N: {}, only 1 eigenvalue: {}".format(self.grid.N, omega))

        self.keep_result(omega, v)

        return (omega, v)

    def solver_only_eigenvalues(self, verbose=False):
        import numpy as np
        from scipy.linalg import eigvals

        boundaries = self.system.boundaries

        self._get_matrix()
        if boundaries is None:
          E = eigvals(self.mat1)
        else:
          E = eigvals(self.mat1, self.mat2)
        # Zero out large values which are most likely numerical
        E[E.real > 2.] = 0
        index = np.argsort(np.real(E))[::-1]

        i = 0
        omega = E[index[i]]
        if verbose:
          print(omega)
        return omega

    def iterate_solver(self, min=6, max=11, tol=1e-6, verbose=False):
        import numpy as np
        from numpy import arange
        N = 2**min
        self.grid.N = N
        (a_old, v) = self.solver(verbose=verbose)
        self.grid.N = 2*N
        (a_new, v) = self.solver(verbose=verbose)
        err = np.abs(a_old - a_new)/np.abs(a_old)

        for N in 2**arange(min+1, max):
            self.grid.N = int(N)
            # Not a good guess yet
            if err > 0.5:
                (a_new, v) = self.solver(verbose=verbose)
            # Use guess from previous iteration
            else:
                (a_new, v) = self.solver(a_old, verbose=verbose)

            err = np.abs(a_old - a_new)/np.abs(a_old)
            # Converged
            if err < tol:
                return (a_new, v, err)
            # Overwrite old with new
            a_old = np.copy(a_new)

        raise RuntimeError("Did not converge!")

    def iterate_solver2(self, Ns, i=0, tol=1e-6, verbose=False):
        import numpy as np

        self.grid.N = Ns[0]
        (a_old, v) = self.solver(i=i, verbose=verbose)
        self.grid.N = Ns[1]
        (a_new, v) = self.solver(i=i, verbose=verbose)
        err = np.abs(a_old - a_new)/np.abs(a_old)

        for i in range(2, len(Ns)):
            self.grid.N = Ns[i]
            # Not a good guess yet
            if err > 0.5:
                (a_new, v) = self.solver(i=i, verbose=verbose)
            # Use guess from previous iteration
            else:
                (a_new, v) = self.solver(a_old, i=i, verbose=verbose)

            err = np.abs(a_old - a_new)/np.abs(a_old)
            # Converged
            if err < tol:
                return (a_new, v, err)
            # Overwrite old with new
            a_old = np.copy(a_new)

        raise RuntimeError("Did not converge!")

    def iterate_solver_simple(self, min=6, max=11, tol=1e-6):
        import numpy as np
        from numpy import arange
        N = 2**min
        self.grid.N = N
        a_old = self.solver_only_eigenvalues()

        for N in 2**arange(min+1, max):
            self.grid.N = int(N)
            a_new = self.solver_only_eigenvalues()
            err = abs(a_old - a_new)/abs(a_old)
            if err < tol:
                return (a_new, err)
            # Overwrite old with new
            a_old = np.copy(a_new)

        raise RuntimeError("Did not converge!")
