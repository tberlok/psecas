class MagnetoThermalInstability:
    """
       The linear solution for the magnetothermal instability (MTI)
       in a quasi-global setup, i.e. periodic in x and non-periodic in z.

       Linearized equations for the MTI with anisotropic viscosity and heat
       conduction for a constant magnetic field in the x-direction.

       See the following paper for more details:

       T. Berlok et al, in prep.
    """

    def __init__(self, grid, beta, Kn0, kx):
        # Problem parameters

        self._beta = beta
        self._Kn0 = Kn0

        self.kx = kx

        self.mu0 = 1.0
        self.p0 = 1.0
        self.rho0 = 1.0
        self.T0 = self.p0 / self.rho0

        self.H0 = 1.0
        self.Lz = 2.0

        self.set_va_and_B0()
        self.set_nu_and_chi()

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Variables to solve for
        self.variables = ["drho", "dA", "dvx", "dvz", "dT"]

        self.labels = [
            r"$\delta \rho$",
            r"$\delta A$",
            r"$\delta v_x$",
            r"$\delta v_z$",
            r"$\delta T$",
        ]

        # Boundary conditions
        self.boundaries = [True, False, False, False, False]

        # Create initial background
        self.make_background()

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # Equations
        eq1 = "sigma*drho = -1j*kx*dvx -dlnrhodz*dvz -1.0*dz(dvz)"
        eq2 = "sigma*dA = 1.0*dvz"
        eq3 = (
            "sigma*dvx = -1j*kx*p/rho*drho -1j*kx*p/rho*dT"
            + "-nu*4/3*kx**2*dvx -nu*1j*kx*2/3*dz(dvz)"
        )
        eq4 = (
            "sigma*dvz = -T*dz(drho) -T*dz(dT) -T*dlnpdz*dT"
            + "+va**2*dz(dz(dA)) -va**2*kx**2*dA"
            + "-1/rho*1j*kx*drhonudz*2/3*dvx -1j*kx*nu*2/3*dz(dvx)"
            + "+1/rho*drhonudz*1/3*dz(dvz) +nu*1/3*dz(dz(dvz))"
        )
        eq5 = (
            "sigma*dT = -1j*kx*2/3*dvx -2/3*dz(dvz) -dlnTdz*dvz"
            + "-2/3*kx**2*kappa*dT -2/3*kx**2*kappa*dlnTdz*dA"
        )

        self.equations = [eq1, eq2, eq3, eq4, eq5]

    @property
    def beta(self):
        return self._beta

    @beta.setter
    def beta(self, value):
        self._beta = value
        self.set_va_and_B0()
        self.make_background()

    @property
    def Kn0(self):
        return self._Kn0

    @Kn0.setter
    def Kn0(self, value):
        self._Kn0 = value
        self.set_nu_and_chi()
        self.make_background()

    def set_nu_and_chi(self):
        self.nu0 = 0.48 / self._Kn0
        self.chi0 = 24.0 / self._Kn0

    def set_va_and_B0(self):
        from numpy import sqrt

        self.B0 = sqrt(2 * self.p0 / self._beta)
        self.va = self.B0 / sqrt(self.mu0 * self.rho0)

    def get_bx_and_by(self):
        """Calculate dbx and dbz. Requires a solution stored!"""
        import numpy as np

        self.grid.make_grid()
        self.result.update(
            {
                "dbx": -np.matmul(self.grid.d1, self.result["dA"]),
                "dbz": 1j * self.kx * self.result["dA"],
            }
        )

    def make_background(self):
        """Functing for creating the background profiles.
        Returns symbolic expressions (as a function of z) """
        import sympy as sym
        import numpy as np
        from sympy import exp, lambdify

        z = sym.symbols("z")

        zg = self.grid.zg

        globals().update(self.__dict__)

        # Define Background Functions
        rho_sym = rho0 * (1 - z / (3 * H0)) ** 2
        p_sym = p0 * (1 - z / (3 * H0)) ** 3

        T_sym = p_sym / rho_sym
        nu_sym = nu0 * T_sym ** (5 / 2)

        self.p = lambdify(z, p_sym)(zg)
        self.rho = lambdify(z, rho_sym)(zg)
        self.dpdz = lambdify(z, sym.diff(p_sym, z))(zg)
        self.dlnrhodz = lambdify(z, sym.diff(rho_sym, z) / rho_sym)(zg)
        self.dlnTdz = lambdify(z, sym.diff(T_sym, z) / T_sym)(zg)
        self.drhonudz = lambdify(z, sym.diff(rho_sym * nu_sym, z))(zg)

        self.T = self.p / self.rho
        self.chi = chi0 * self.T ** (5 / 2)
        self.nu = nu0 * self.T ** (5 / 2)
        self.kappa = self.chi * self.T / self.p
        self.dlnpdz = self.dpdz / self.p
