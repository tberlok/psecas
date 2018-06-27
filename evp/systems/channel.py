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

        # Create initial background
        self.make_background()

        # Number of equations in system
        self.dim = len(self.variables)

        # Equations (Careful! No space between minus and the term it
        # belongs to)
        eq1 = "1*dz(dz(f)) +z*dz(f)"

        self.equations = [eq1]

        # RHS of the eigenvalue problem
        self.rhs_equations = ['-h*f']

    def make_background(self):
        """Functing for creating the background profiles.
        Returns symbolic expressions (as a function of z) """
        import numpy as np

        zg = self.grid.zg

        # Define Background Functions
        self.h = np.exp(-zg**2/2)
        self.z = zg
