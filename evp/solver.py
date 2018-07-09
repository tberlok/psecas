class Solver():
    """docstring for Solver"""
    def __init__(self, grid, system):
        self.grid = grid
        self.system = system

    def _set_submatrix(self, mat1, submat, eq_n, var_n, boundary):
        """Set submatrix corresponding to the term proportional to var_n
        (variable number) in eq_n (equation number). """
        NN = self.grid.NN
        N = self.grid.N
        if boundary:
            submat[0, :] = 0
            submat[N, :] = 0
            if eq_n == var_n:
                submat[0, 0] = 1
                submat[N, N] = 1
        mat1[(eq_n-1)*NN:eq_n*NN, (var_n-1)*NN:var_n*NN] = submat

    def _set_boundary(self, var_n):
        NN = self.grid.NN
        self.mat2[(var_n-1)*NN, (var_n-1)*NN] = 0.0
        self.mat2[var_n*NN-1, var_n*NN-1] = 0.0

    def _find_submatrices(self, eq, verbose=False):
        import numpy as np
        import re

        # This is a nasty trick
        globals().update(self.system.__dict__)

        NN = self.grid.NN
        dim = self.system.dim

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
            for i, var in enumerate(self.system.variables):
                # No derivative
                s0 = "*" + var
                s1 = "*dz(" + var + ")"
                s2 = "*dz(dz(" + var + "))"
                if term.endswith(s0):
                    if verbose:
                        print('\t\tFound ', s0)
                    res = eval(term[:-len(s0)])
                    mats[i] += (res*self.grid.d0.T).T
                # 1st derivative
                if term.endswith(s1):
                    if verbose:
                        print('\t\tFound ', s1)
                    res = eval(term[:-len(s1)])
                    mats[i] += (res*self.grid.d1.T).T
                # 2nd derivative
                if term.endswith(s2):
                    if verbose:
                        print('\t\tFound ', s2)
                    res = eval(term[:-len(s2)])
                    mats[i] += (res*self.grid.d2.T).T
        return mats

    def _get_matrix1(self):
        import numpy as np
        dim = self.system.dim
        NN = self.grid.NN
        equations = self.system.equations
        variables = self.system.variables
        boundaries = self.system.boundaries

        # Construct matrix mat1
        mat1 = np.zeros((dim*NN, dim*NN), dtype="complex128")

        for j, equation in enumerate(equations):
            mats = self._find_submatrices(equation)
            for i, variable in enumerate(variables):
                if boundaries is not None:
                    self._set_submatrix(mat1, mats[i], j+1, i+1, boundaries[j])
                else:
                    self._set_submatrix(mat1, mats[i], j+1, i+1, False)

        self.mat1 = mat1

    def _get_matrix2(self):
        import numpy as np
        dim = self.system.dim
        NN = self.grid.NN
        equations = self.system.equations
        variables = self.system.variables
        boundaries = self.system.boundaries
        try:
            # Construct matrix mat2
            rhs_equations = self.system.rhs_equations
            mat2 = np.zeros((dim*NN, dim*NN), dtype="complex128")
            for j, equation in enumerate(rhs_equations):
                mats = self._find_submatrices(equation)
                for i, variable in enumerate(variables):
                    self._set_submatrix(mat2, mats[i], j+1, i+1, False)
            self.mat2 = mat2
        except AttributeError:
            self.mat2 = np.eye(dim*NN)

        if boundaries is not None:
            for j, equation in enumerate(equations):
                if boundaries[j]:
                    self._set_boundary(j+1)

    def keep_result(self, omega, vec, mode):

        # Store result
        self.system.result = {var: vec[j*self.grid.NN:(j+1)*self.grid.NN]
                              for j, var in enumerate(self.system.variables)}
        self.system.result.update({'omega': omega, 'mode': mode})

    def solve(self, guess=None, useOPinv=True, verbose=False, mode=0):
        """
        Solve the EVP generated with the grid and parameters contained in the
        system object.

        Stores a dictionary with the result in self.system.result.

        Returns: One eigenvalue and its eigenvector.

        Optional parameters

        guess: If a guess is passed scipy's eigs method is used to find a
        single eigenvalue in the proximity of the guess.

        useOPinv (default True): If true, manually calculate OPinv instead of
        letting eigs do it.

        verbose (default False): print out information about the calculation.

        mode (default 0): mode=0 is the fastest growing, mode=1 the second
        fastest and so on.
        """
        import numpy as np
        from scipy.linalg import eig

        boundaries = self.system.boundaries

        # Calculate matrix
        self._get_matrix1()
        self._get_matrix2()

        if guess is None:
            if boundaries is None:
                E, V = eig(self.mat1)
            else:
                E, V = eig(self.mat1, self.mat2)

            # Zero out large values which are most likely numerical
            # TODO: Don't have this hardcoded
            E = self.sorting_strategy(E)

            # Sort from largest to smallest eigenvalue
            index = np.argsort(np.real(E))[::-1]

            # Choose the eigenvalue mode value only
            omega = E[index[mode]]
            v = V[:, index[mode]]
            if verbose:
                print("N: {}, all eigenvalues: {}".format(self.grid.N, omega))
        else:
            from scipy.sparse.linalg import eigs
            if useOPinv:
                # from numpy.linalg import pinv as inv
                from numpy.linalg import inv
                if boundaries is None:
                    OPinv = inv(self.mat1 - guess*np.eye(self.mat1.shape[0]))
                else:
                    OPinv = inv(self.mat1 - guess*self.mat2)
                from scipy import sparse
                smat = sparse.csr_matrix(self.mat1)
                omega, v = eigs(smat, k=1, sigma=guess, OPinv=OPinv)
            else:
                if boundaries is None:
                    omega, v = eigs(self.mat1, k=1, sigma=guess)
                else:
                    omega, v = eigs(self.mat1, M=self.mat2, k=1, sigma=guess)
            # Convert result from eigs to have same format as result from eig
            omega = omega[0]
            v = np.squeeze(v)
            if verbose:
                print("N:{}, only 1 eigenvalue:{}".format(self.grid.N, omega))

        self.keep_result(omega, v, mode)

        return (omega, v)

    def sorting_strategy(self, E):
        """A default sorting strategy"""
        import numpy as np
        E[np.abs(E.real) > 10.] = 0
        E[np.abs(E.imag) > 10.] = 0
        return E

    def solve_only_eigenvalues(self, verbose=False):
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

    def iterate_solver(self, Ns, mode=0, tol=1e-6, verbose=False):
        import numpy as np

        self.grid.N = Ns[0]
        (a_old, v) = self.solve(mode=mode, verbose=verbose)
        self.grid.N = Ns[1]
        (a_new, v) = self.solve(mode=mode, verbose=verbose)
        err = np.abs(a_old - a_new)/np.abs(a_old)

        for i in range(2, len(Ns)):
            self.grid.N = Ns[i]
            # Not a good guess yet
            if err > 0.1:
                (a_new, v) = self.solve(mode=mode, verbose=verbose)
            # Use guess from previous iteration
            else:
                (a_new, v) = self.solve(a_old, mode=mode, verbose=verbose)

            err = np.abs(a_old - a_new)/np.abs(a_old)
            # Converged
            if err < tol:
                self.system.result.update({'converged': True})
                self.system.result.update({'err': err})
                return (a_new, v, err)
            # Overwrite old with new
            a_old = np.copy(a_new)

        self.system.result.update({'converged': False})
        self.system.result.update({'err': err})

        # raise RuntimeError("Did not converge!")

    def iterate_solver_simple(self, min=6, max=11, tol=1e-6):
        import numpy as np
        from numpy import arange
        N = 2**min
        self.grid.N = N
        a_old = self.solve_only_eigenvalues()

        for N in 2**arange(min+1, max):
            self.grid.N = int(N)
            a_new = self.solve_only_eigenvalues()
            err = abs(a_old - a_new)/abs(a_old)
            if err < tol:
                return (a_new, err)
            # Overwrite old with new
            a_old = np.copy(a_new)

        raise RuntimeError("Did not converge!")
