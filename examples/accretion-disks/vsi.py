import numpy as np
from freja import Solver, ChebyshevRationalGrid, System
from freja import plot_solution

"""
    The vertical shear instability in protoplanetary accretion disks.

    See for instance the following paper for more details:

    O. M. Umurhan,  R. P. Nelson and O. Gressel,
    Linear analysis of the vertical shear instability: outstanding issues
    and improved solutions, A&A 586, A33 (2016),
    DOI: 10.1051/0004-6361/201526494
"""


# Make a Child of the System class and override the make_background method
class VerticalShearInstability(System):
    def __init__(self, grid, variables, eigenvalue):
        self.h = 0.05
        self.p = -1.5
        self.q = -1
        self.kx = 200 * np.pi

        super().__init__(grid, variables, eigenvalue)

    def make_background(self):
        import sympy as sym
        from sympy import sqrt, exp, lambdify

        z = sym.symbols("z")

        # Constant parameters
        h = self.h
        p = self.p
        q = self.q

        self.O0 = np.sqrt(1 + (self.p + self.q) * self.h ** 2)
        O0 = self.O0

        # Define background functions
        rho_sym = exp(h ** (-2) * (1 / sqrt(1 + z ** 2 * h ** 2) - 1))
        Omg_sym = (
            1
            / O0
            * sqrt(
                1 + (p + q) * h ** 2 + q * (1 - 1 / sqrt(1 + z ** 2 * h ** 2))
            )
        )
        shr_sym = (
            -3
            / 2
            / (Omg_sym * O0)
            * (
                1
                + 1 / 3 * (2 - q) * (p + q) * h ** 2
                + q
                * (
                    1
                    - (1 + 2 / 3 * z ** 2 * h ** 2)
                    / ((1 + z ** 2 * h ** 2) ** (3 / 2))
                )
            )
        )
        drhodz_sym = sym.diff(rho_sym, z)
        domgdz_sym = sym.diff(Omg_sym, z)

        zg = self.grid.zg
        self.rho = np.ones_like(zg) * lambdify(z, rho_sym)(zg)
        self.Omg = np.ones_like(zg) * lambdify(z, Omg_sym)(zg)
        self.shr = np.ones_like(zg) * lambdify(z, shr_sym)(zg)
        self.drhodz = np.ones_like(zg) * lambdify(z, drhodz_sym)(zg)
        self.domgdz = np.ones_like(zg) * lambdify(z, domgdz_sym)(zg)


# Create a grid
grid = ChebyshevRationalGrid(N=219, C=0.3)
# grid = ChebyshevExtremaGrid(N=199, zmin=-5, zmax=5)

# Create the system
system = VerticalShearInstability(
    grid, variables=['rh', 'wx', 'wy', 'wz'], eigenvalue='sigma'
)

# The linearized equations
system.add_equation("-sigma*rh = - 1j*kx*wx - 1/h*dz(wz) - 1/h*drhodz/rho*wz")
system.add_equation("-sigma*wx = + 2*Omg*wy - 1j*kx*(h/O0)**2*rh")
system.add_equation("-sigma*wy = - (2*Omg + shr)*wx - 1/h*domgdz*wz")
system.add_equation("-sigma*wz = - h/O0**2*dz(rh)", boundary=True)

solver = Solver(grid, system)


omega, vec = solver.solve(mode=0, verbose=True)
plot_solution(system)
