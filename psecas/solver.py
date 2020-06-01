class Solver:
    """docstring for Solver"""

    def __init__(self, grid, system, do_gen_evp=False):
        import numpy as np

        # Grid object
        self.grid = grid

        # System object with linearized equations, parameters and equilibrium.
        self.system = system

        # do_gen_evp (default False) do the full generalized evp even though
        # boundaries suggest that an evp is sufficient
        self.do_gen_evp = do_gen_evp

        # Check if we need to solve a generalized evp
        if not self.do_gen_evp:
            # Boundaries are not all True, and not all False
            if not all(self.system.boundaries) and any(self.system.boundaries):
                self.do_gen_evp = True

        # Check that variable names are unique, i.e., that variables
        # are not a substring of another variable
        msg = """eigenmode variable names are not allowed to be substrings
                 of other eigenmode variables names"""
        for var1 in system.variables:
            tmp = np.sum([var.find(var1) for var in system.variables])
            assert tmp == 1 - system.dim, msg

        # This ensures backwards compatibility with old way of simply setting
        # True/False in boundary flag.
        if not hasattr(system, 'extra_binfo'):
            extra_binfo = []
            for boundary in system.boundaries:
                if boundary:
                    extra_binfo.append(['Dirichlet', 'Dirichlet'])
                else:
                    extra_binfo.append([None, None])
            system.extra_binfo = extra_binfo

        else:
            # In the current implementation, we always have to solve
            # the generalized evp unless all boundary conditions are Dirichlet
            # or not set
            for info in system.extra_binfo:
                if info is not None and info != 'Dirichlet':
                    self.do_gen_evp = True


    def solve(self, useOPinv=True, verbose=False, mode=0, saveall=False):
        """
        Construct and solve the (generalized) eigenvalue problem (EVP)

            M₁ v = σ M₂ v

        generated with the grid and parameters contained in the system object.

        Here σ is the eigenvalue and v is the eigenmode.
        Note that M₂ is a diagonal matrix if no boundary conditions are set.
        In that case the EVP is simply

            M₁ v = σ v

        This method stores a dictionary with the result of the calculation
        in self.system.result.

        Returns: One eigenvalue and its eigenvector.

        Optional parameters

        useOPinv (default True): If true, manually calculate OPinv instead of
        letting eigs do it.

        verbose (default False): print out information about the calculation.

        mode (default 0): mode=0 is the fastest growing, mode=1 the second
        fastest and so on.
        """
        from scipy.linalg import eig
        from scipy import sparse

        boundaries = self.system.boundaries

        # Calculate left and right-hand matrices
        self.get_matrix1()
        self.get_matrix2()

        # If mat2 is not the identity matrix, then we have to solve a generalized evp
        if not self.do_gen_evp:
            mat2_is_identity = (self.mat2 - sparse.eye(self.mat1.shape[0])).count_nonzero() == 0
            if not mat2_is_identity:
                self.do_gen_evp = True
                # Recompute matrix 1 and 2, now with boundary conditions explicitly enabled.
                # This loop is only entered on the first call
                self.get_matrix1()
                self.get_matrix2()
                diag = (self.mat2 - sparse.diags(self.mat2.diagonal())).count_nonzero() == 0
                single_val = self.mat2.diagonal().max() == self.mat2.diagonal().min()
                if diag and single_val:
                    msg = """Psecas will solve a generalized EVP but it appears that rewriting the
                    LHS of your equations could reduce the calculation to a standard EVP."""
                    print(msg)


        # Solve a generalized EVP
        if self.do_gen_evp:
            E, V = eig(self.mat1.toarray(), self.mat2.toarray())
        # Solve a standard EVP
        else:
            E, V = eig(self.mat1.toarray())

        # Sort the eigenvalues
        E, index = self.sorting_strategy(E)

        # Choose the eigenvalue mode value only
        sigma = E[index[mode]]
        v = V[:, index[mode]]

        # Save all eigenvalues and eigenvectors here
        if saveall:
            self.E = E[index]
            self.v = V[:, index]
        if verbose:
            print("N: {}, all eigenvalues: {}".format(self.grid.N, sigma))

        self.keep_result(sigma, v, mode)

        return (sigma, v)

    def solve_with_guess(self, guess, useOPinv=True, verbose=False, mode=0):
        """
        Construct and solve the (generalized) eigenvalue problem (EVP)

            M₁ v = σ M₂ v

        generated with the grid and parameters contained in the system object.

        Here σ is the eigenvalue and v is the eigenmode.
        Note that M₂ is a diagonal matrix if no boundary conditions are set.
        In that case the EVP is simply

            M₁ v = σ v

        This method stores a dictionary with the result of the calculation
        in self.system.result.

        Returns: One eigenvalue and its eigenvector.

        guess: Scipy's eigs method is used to find a
        single eigenvalue in the proximity of the guess.

        Optional parameters

        useOPinv (default True): If true, manually calculate OPinv instead of
        letting eigs do it.

        verbose (default False): print out information about the calculation.

        mode (default 0): mode=0 is the fastest growing, mode=1 the second
        fastest and so on.
        """
        import numpy as np
        from scipy import sparse
        from scipy.sparse.linalg import eigs

        boundaries = self.system.boundaries

        # Calculate matrix
        self.get_matrix1()
        self.get_matrix2()

        # If mat2 is not the identity matrix, then we have to solve a generalized evp
        if not self.do_gen_evp:
            mat2_is_identity = (self.mat2 - sparse.eye(self.mat1.shape[0])).count_nonzero() == 0
            if not mat2_is_identity:
                self.do_gen_evp = True
                # Recompute matrix 1 and 2, now with boundary conditions explicitly enabled.
                # This loop is only entered on the first call
                self.get_matrix1()
                self.get_matrix2()
                diag = (self.mat2 - sparse.diags(self.mat2.diagonal())).count_nonzero() == 0
                single_val = self.mat2.diagonal().max() == self.mat2.diagonal().min()
                if diag and single_val:
                    msg = """Psecas will solve a generalized EVP but it appears that rewriting the
                    LHS of your equations could reduce the calculation to a standard EVP."""
                    print(msg)

        # Solve a generalized EVP
        if self.do_gen_evp:
            if useOPinv:
                from numpy.linalg import inv
                OPinv = inv((self.mat1 - guess * self.mat2).toarray())
                sigma, v = eigs(self.mat1, k=1, sigma=guess, OPinv=OPinv)
            else:    
                sigma, v = eigs(self.mat1, M=self.mat2, k=1, sigma=guess)
        else:
            if useOPinv:
                from numpy.linalg import inv
                OPinv = inv(self.mat1 - guess * np.eye(self.mat1.shape[0]))
                sigma, v = eigs(self.mat1, k=1, sigma=guess, OPinv=OPinv)
            else:
                sigma, v = eigs(self.mat1, k=1, sigma=guess)
                

        # Convert result from eigs to have same format as result from eig
        sigma = sigma[0]
        v = np.squeeze(v)

        if verbose:
            print("N:{}, only 1 eigenvalue:{}".format(self.grid.N, sigma))

        self.keep_result(sigma, v, mode)

        return (sigma, v)

    def iterate_solver(
        self, Ns, mode=0, tol=1e-6, atol=1e-16, verbose=False, guess_tol=0.01,
        useOPinv=True
    ):
        """
        Iteratively call the solve method with increasing grid resolution, N.
        Returns when the relative difference in the eigenvalue is less than
        the tolerance, tol.

        Ns: list of resolutions to try, e.g. Ns = arange(32)*10

        mode: the index in the list of eigenvalues returned from solve

        tol: the target precision of the eigenvalue

        verbose (default False): print out information about the calculation.

        guess_tol: Increasing the resolution will inevitably lead to a more
        expensive computation. A speedup can however be achieved when
        searching for a single eigenvalue. This method can in this
        case use the eigenvalue from the previous calculation as a guess for
        the result of the new calculation. The parameter guess_tol makes sure
        that the guess used is a good guess. If guess_tol=0.1 the method will
        start using guesses when the relative difference to the previous
        iteration is 10 %.
        """
        import numpy as np

        self.grid.N = Ns[0]
        (sigma_old, v) = self.solve(mode=mode, verbose=verbose)
        self.grid.N = Ns[1]
        (sigma_new, v) = self.solve(mode=mode, verbose=verbose)
        a_err = np.abs(sigma_old - sigma_new)
        r_err = a_err / np.abs(sigma_old)

        for i in range(2, len(Ns)):
            self.grid.N = Ns[i]
            # Not a good guess yet
            if r_err > guess_tol:
                (sigma_new, v) = self.solve(mode=mode, verbose=verbose)
            # Use guess from previous iteration
            else:
                (sigma_new, v) = self.solve_with_guess(
                    sigma_old, mode=mode, verbose=verbose, useOPinv=useOPinv
                )

            a_err = np.abs(sigma_old - sigma_new)
            r_err = a_err / np.abs(sigma_old)
            # Converged
            if r_err < tol or a_err < atol:
                self.system.result.update({"converged": True})
                self.system.result.update({"r_err": r_err, "a_err": a_err})
                return (sigma_new, v, r_err)
            # Overwrite old with new
            sigma_old = np.copy(sigma_new)

        self.system.result.update({"converged": False})
        self.system.result.update({"r_err": r_err, "a_err": a_err})
        return (sigma_new, v, r_err)

        # raise RuntimeError("Did not converge!")

    def sorting_strategy(self, E):
        """
        A default sorting strategy.

        "Large" real and imaginary eigenvalues are removed and the eigenvalues
        are sorted from largest to smallest
        """
        import numpy as np

        E[np.abs(E.real) > 10.0] = 0
        E[np.abs(E.imag) > 10.0] = 0
        # Sort from largest to smallest eigenvalue
        index = np.argsort(np.real(E))[::-1]
        return (E, index)

    def keep_result(self, sigma, vec, mode):
        import numpy as np

        # Store result
        if all(self.system.boundaries) and not self.do_gen_evp:
            # Add zeros at both ends of the solution
            self.system.result = {
                var: np.hstack(
                    [
                        0.0,
                        vec[
                            j
                            * (self.grid.N - 1) : (j + 1)
                            * (self.grid.N - 1)
                        ],
                        0.0,
                    ]
                )
                for j, var in enumerate(self.system.variables)
            }
        else:
            self.system.result = {
                var: vec[j * self.grid.NN : (j + 1) * self.grid.NN]
                for j, var in enumerate(self.system.variables)
            }
        self.system.result.update(
            {self.system.eigenvalue: sigma, "mode": mode}
        )

    def get_matrix1(self, verbose=False):
        """
        Calculate the matrix M₂ neded in the solve method.
        """
        from scipy import sparse
        import numpy as np

        dim = self.system.dim
        grid = self.grid
        equations = self.system.equations
        boundaries = self.system.boundaries
        extra_binfo = self.system.extra_binfo

        # Construct all submatrices as sparse matrices
        rows = []
        for j, equation in enumerate(equations):
            equation = equation.split("=")[1]
            mats = self._find_submatrices(equation, verbose)
            rows.append(mats)

        # Modify according to boundary conditions
        for j in range(dim):
            for i in range(dim):
                if all((boundaries)) and not self.do_gen_evp:
                    rows[j][i] = rows[j][i][1:grid.N, 1:grid.N]
                elif any(boundaries):
                    rows[j][i] = self._modify_submatrix(rows[j][i],
                                                        j + 1, i + 1,
                                                        boundaries[j], 
                                                        extra_binfo[j], 
                                                        verbose)

        # Assemble everything
        self.mat1 = sparse.bmat(rows, format='csr')

    def get_matrix2(self, verbose=False):
        """
        Calculate the matrix M₂ neded in the solve method.
        """
        from scipy import sparse
        import numpy as np

        dim = self.system.dim
        N = self.grid.N
        sys = self.system
        equations = sys.equations
        variables = sys.variables
        boundaries = sys.boundaries
        extra_binfo = sys.extra_binfo

        # Evaluate LHS of equation
        rows = []
        for j, equation in enumerate(equations):
            equation = equation.split("=")[0]
            equation = self._var_replace(equation, sys.eigenvalue, "1.0")
            mats = self._find_submatrices(equation, verbose)
            rows.append(mats)

        # Modify according to boundary conditions
        for j in range(dim):
            for i in range(dim):
                if all((boundaries)) and not self.do_gen_evp:
                    rows[j][i] = rows[j][i][1:grid.N, 1:grid.N]
                elif any(boundaries):
                    if extra_binfo[j][0] is not None:
                        rows[j][i][0, 0] = 0
                    if extra_binfo[j][1] is not None:
                        rows[j][i][N, N] = 0

        # Assemble everything
        self.mat2 = sparse.bmat(rows, format='csr')

    def _var_replace(self, eq, var, new):
        """
        Replace all instances of string var with string new.
        This function differs from the default string replace method in
        that it only makes the replace if var is not contained inside a
        word.

        Example:
        eq = "-1j*kx*v*drho -drhodz*dvz -1.0*dz(dvz) - drho"
        var_replace(eq, 'drho', 'foo')
        returns '-1j*kx*v*foo -drhodz*dvz -1.0*dz(dvz) - foo'
        where drhodz has not been replaced.
        """
        pos = 0
        while pos != -1:
            pos = eq.find(var, pos)
            if pos != -1:
                substitute = True
                # Check if character to the left is a letter
                if pos > 0:
                    if eq[pos - 1].isalpha():
                        substitute = False
                # Check if character to the right is a letter
                if pos + len(var) < len(eq):
                    if eq[pos + len(var)].isalpha():
                        substitute = False
                if substitute:
                    eq = eq[:pos] + new + eq[pos + len(var) :]
                # Increment pos to prevent the function from repeatedly
                # finding the same occurrence of var
                else:
                    pos += len(var)
        return eq

    def _find_submatrices(self, eq, verbose=False):
        import numpy as np
        from scipy import sparse

        # This is a nasty trick
        globals().update(self.system.__dict__)
        grid = self.system.grid

        NN = self.grid.NN
        mats = []

        if verbose:
            print("\nParsing equation:", eq)

        for i, var in enumerate(self.system.variables):
            if var in eq:
                variables_t = list(np.copy(self.system.variables))
                eq_t = eq
                der = "d" + grid.z + "("
                eq_t = eq_t.replace(der + der + var + "))", "grid.d2.T")
                eq_t = eq_t.replace(der + var + ")", "grid.d1.T")
                eq_t = self._var_replace(eq_t, var, "grid.d0.T")
                eq_t = self._var_replace(eq_t, grid.z, "grid.zg")

                variables_t.remove(var)
                for var2 in variables_t:
                    eq_t = eq_t.replace(der + der + var2 + "))", "0.0")
                    eq_t = eq_t.replace(der + var2 + ")", "0.0")
                    eq_t = self._var_replace(eq_t, var2, "0.0")
                if verbose:
                    print("\nEvaluating expression:", eq_t)
                try:
                    err_msg1 = (
                        "During the parsing of:\n\n{}\n\n"
                        "Psecas tried to evaluate\n\n{}\n\n"
                        "while attempting to evaluate the terms with: {}"
                        "\nThis caused the following error to occur:\n\n"
                    )
                    # Evaluate the expression
                    submat = eval(eq_t).T
                except NameError as e:
                    strerror, = e.args
                    err_msg2 = (
                        "\n\nThis is likely because the missing variable has"
                        "\nnot been defined in your systems class or its\n"
                        "make_background method."
                    )
                    raise NameError(
                        err_msg1.format(eq, eq_t, var) + strerror + err_msg2
                    )
                except Exception as e:
                    raise Exception(err_msg1.format(eq, eq_t, var) + str(e))
                submat = np.array(submat, dtype="complex128")
            else:
                submat = np.zeros((NN, NN), dtype=np.complex128)

            # Prevent sparse.lil_matrix from changing the shape of
            # a numpy array which is all zeros.
            if np.count_nonzero(submat) == 0:
                submat = np.zeros((NN, NN), dtype=np.complex128)

            mats.append(sparse.lil_matrix(submat))

        return mats

    def _modify_submatrix(self, submat, eq_n, var_n, boundary, binfo, verbose=False):
        """
        This modifies the submatrix to incorporate boundary conditions.
    
        Dirichlet is value set to zero at boundary.
        Neumann is derivative set to zero at boundary.

        Finally, one can set a string such as

        'r**2*dr(dr(Aphi)) + r*dr(Aphi) - Aphi = 0'

        The Boundary condition on a variable cannot depend on the other independent variables.
        """
        import numpy as np

        # This is a nasty trick
        globals().update(self.system.__dict__)
        grid = self.system.grid

        N = self.grid.N
        if boundary:
            for index, bound in zip([0, N], binfo):
                if bound is not None:
                    submat[index, :] = 0
                    if eq_n == var_n:
                        if bound == 'Dirichlet':
                            submat[index, index] = 1
                        elif bound == 'Neumann':
                            submat[index, :] = grid.d1[index, :]
                        else:
                            assert '=' in bound, 'equal sign missing in boundary expression'
                            assert int(bound.split("=")[1]) == 0, 'rhs of boundary expressions must be zero'
                            var = self.system.variables[var_n-1]
                            bound_t = bound.split("=")[0]
                            der = "d" + grid.z + "("
                            mask = np.zeros(self.grid.NN)
                            mask[index] = 1
                            bound_t = bound_t.replace(der + der + var + "))", "grid.d2[{}, :]".format(index))
                            bound_t = bound_t.replace(der + var + ")", "grid.d1[{}, :]".format(index))
                            bound_t = self._var_replace(bound_t, var, "mask")
                            bound_t = self._var_replace(bound_t, grid.z, "grid.zg[{}]".format(index))
                            if verbose:
                                print("\nEvaluating expression:", bound_t)
                            try:
                                err_msg1 = (
                                    "During the parsing of:\n\n{}\n\n"
                                    "Psecas tried to evaluate\n\n{}\n\n"
                                    "while attempting to evaluate the boundary on: {}"
                                    "\nThis caused the following error to occur:\n\n"
                                )
                                # Evaluate the expression
                                submat[index, :] = eval(bound_t)
                            except NameError as e:
                                strerror, = e.args
                                err_msg2 = (
                                    "\n\nThis is likely because the missing variable has"
                                    "\nnot been defined in your systems class or its\n"
                                    "make_background method."
                                )
                                raise NameError(
                                    err_msg1.format(bound, bound_t, var) + strerror + err_msg2
                                )
                            except Exception as e:
                                raise Exception(err_msg1.format(bound, bound_t, var) + str(e))

        return submat
