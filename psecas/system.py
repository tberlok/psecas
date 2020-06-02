class System:
    """
    Dedalus style initialization of an EVP problem.
    This will be useful for people for comparing with Dedalus.
    """

    def __init__(self, grid, variables, eigenvalue):

        self.grid = grid
        # Bind grid to the make_background method.
        # This ensures that the background is always evaluated with the
        # current grid resolution
        self.grid.bind_to(self.make_background)

        if type(variables) is str:
            self.variables = list([variables])
        else:
            self.variables = list(variables)

        self.equations = ['' for ii in range(len(self.variables))]
        self.boundaries = [False for ii in range(len(self.variables))]
        self.extra_binfo = [[None, None] for ii in range(len(self.variables))]

        self.substitutions = []

        self.labels = variables

        self.eigenvalue = eigenvalue

        # Create background
        self.make_background()

    @property
    def dim(self):
        return len(self.equations)

    def add_equation(self, eq, boundary=False):
        found = False
        for ii, var in enumerate(self.variables):
            if var in eq.split('=')[0]:
                if found:
                    raise RuntimeError('Only one variable may appear on the LHS.')
                else:
                    found = True
                self.equations[ii] = eq
                if boundary:
                    self.boundaries[ii] = True
                    self.extra_binfo[ii] = ['Dirichlet', 'Dirichlet']
                else:
                    self.boundaries[ii] = False
                    self.extra_binfo[ii] = [None, None]

    def add_boundary(self, var, lower, upper):
        msg = 'Cannot set boundary on {}, as it is not found in system.variables'
        assert var in self.variables, msg.format(var)

        for ii, var2 in enumerate(self.variables):
            if var == var2:
                self.extra_binfo[ii] = [lower, upper]
                self.boundaries[ii] = True
                return

    def add_substitution(self, substitution):
        """Add equation substitution.
           There is not any symbolic manipulation,
           substitution will be done using simple text replacement.
        """
        assert '=' in substitution, 'should contain an equal sign'
        self.substitutions.append(substitution)

    def make_background(self):
        """
        Add problem parameters that depend on the grid using this fucntion.

        Example:
        Say the equations depend on rho(z) = exp(-(z/H0)**2).
        The following code would then be written:

        import numpy as np
        zg = self.grid.zg

        self.rho = np.exp(-(z/H0)**2)
        self.H0 = 0.4

        """
        pass
