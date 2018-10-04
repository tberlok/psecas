class Channel():
    """Linearized equations for channel mode."""

    def __init__(self, grid):
        # Problem parameters

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Variables to solve for
        self.variables = ['f']

        self.labels = [r'$f(z)$']

        # Boundary conditions
        self.boundaries = [True]

        # Do full generalized evp even though boundaries suggest that an evp
        # is sufficient
        self.do_gen_evp = True

        # Create initial background
        self.make_background()

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = 'sigma'

        # Equations
        eq1 = "-h*sigma*f = 1*dr(dr(f)) +r*dr(f)"

        self.equations = [eq1]

    def make_background(self):
        """Functing for creating the background profiles.
        Returns symbolic expressions (as a function of z) """
        import numpy as np

        zg = self.grid.zg

        # Define Background Functions
        self.h = np.exp(-zg**2/2)
