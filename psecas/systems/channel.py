class Channel:
    """The linearized equations for channel mode, equation 17 in

    *MRI channel flows in vertically stratified models of accretion discs*,
    https://doi.org/10.1111/j.1365-2966.2010.16759.x,
    by Henrik N. Latter, Sebastien Fromang, Oliver Gressel

    is 

        F'' + K² h F = 0

    and is solved  by employing a Neumann boundary condition on F.
    """

    def __init__(self, grid):
        # Problem parameters

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Variables to solve for
        self.variables = ["F"]

        self.labels = [r"$F(z)$"]

        # Boundary conditions
        self.boundaries = [True]

        # Extra info
        self.extra_binfo = [['Neumann', 'Neumann']]

        # Create initial background
        self.make_background()

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "K2"

        # Equations
        eq1 = "-h*K2*F = dz(dz(F))"

        self.equations = [eq1]

    def make_background(self):
        """Functing for creating the background profiles.
        Returns symbolic expressions (as a function of z) """
        import numpy as np

        zg = self.grid.zg

        # Define Background Functions
        self.h = np.exp(-zg ** 2 / 2)
