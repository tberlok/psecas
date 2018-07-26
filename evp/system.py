class System():
    """
    Dedalus style initialization of an EVP problem.
    This will be useful for people for comparing with Dedalus.
    """
    def __init__(self, grid, variables, eigenvalue):
        self.equations = []
        self.boundaries = []

        self.grid = grid
        # Bind grid to the make_background method.
        # This ensures that the background is always evaluated with the
        # current grid resolution
        self.grid.bind_to(self.make_background)

        if type(variables) is str:
            self.variables = list([variables])
        else:
            self.variables = list(variables)

        self.labels = variables

        self.eigenvalue = eigenvalue

        # Create background
        self.make_background()

    @property
    def dim(self):
        return len(self.equations)

    def add_equation(self, eq, boundary=False):
        self.equations.append(eq)
        self.boundaries.append(boundary)

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
