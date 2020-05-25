class HeatFluxDrivenBuoyancyInstability:
    """
       The linear equations for the heat-flux-driven buoyancy instability (HBI)
       in a quasi-global setup, i.e. periodic in x and non-periodic in z.

       See the following paper for more details:

       H. N. Latter, M. W. Kunz, 2012, MNRAS, 423, 1964
       The HBI in a quasi-global model of the intracluster medium

       The boundary conditions are not exactly the same as in this paper,
       i.e., we do not impose a boundary condition on the magnetic field.
       Improving this is left for future work.
    """

    def __init__(self, grid, beta, Kn, kx):

        # Problem parameters
        self.beta = beta
        self.Kn = Kn

        self.kx = kx

        self.grid = grid
        self.grid.bind_to(self.make_background)

        # Variables to solve for
        self.variables = ["drho", "dA", "dvx", "dvz", "dT"]

        self.labels = [
            r"$\delta \rho/\rho$",
            r"$\delta A/B$",
            r"$\delta v_x/c_s$",
            r"$\delta v_z/c_s$",
            r"$\delta T/T$",
        ]

        # Boundary conditions
        self.boundaries = [False, True, True, False, True]

        # Extra information for boundary conditions
        self.extra_binfo = [[None, None], ['Dirichlet', 'Neumann'], [None, 'Neumann'],
                            [None, None], ['Dirichlet', 'Neumann']]

        # Create initial background
        self.make_background()

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # Equations
        eq1 = "sigma*drho = -1j*kx*dvx -dlnrhodz*dvz -dz(dvz)"
        eq2 = "sigma*dA = -dvx"
        eq3 = (
            "sigma*dvx = -1j*kx*T*(drho + dT)"
            + "+2/(beta*rho)*(dA + dz(dz(dA)))"
            + "+2/Re*T**(5/2)/rho*(2/3*dz(dvz) -1/3*1j*kx*dvx)"
        )
        eq4 = (
            "sigma*dvz = -T*dz(drho) -T*dz(dT) -T*dlnpdz*dT"
            + "+2/Re*T**(5/2)/rho*(5/2*dlnTdz*(2/3*dz(dvz)-1/3*1j*kx*dvx)"
            + "+2/3*dz(dz(dvz)) -1/3*1j*kx*dz(dvx))"
        )
        eq5 = (
            "3/2*sigma*dT = -1j*kx*dvx -dz(dvz) -3/2*dlnTdz*dvz"
            + "+1/(p*Pe)*(7*T**(5/2)*dTdz*dz(dT) +35/4*dTdz**2*T**(3/2)*dT"
            + "+T**(7/2)*dz(dz(dT)) +7/2*T**(5/2)*d2Tdz*dT +1j*kx*q*dz(dA))"
        )

        self.equations = [eq1, eq2, eq3, eq4, eq5]

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
        """
        import sympy as sym
        from sympy import exp, lambdify

        z = sym.symbols("z")

        zg = self.grid.zg

        globals().update(self.__dict__)

        G = 2.0

        self.Pe = 1 / 24 * G / self.Kn
        self.Re = 1 / 0.48 / self.Kn

        zeta = 2.5 ** (7 / 2) - 1

        T_sym = (1 + zeta * z) ** (2 / 7)
        p_sym = exp(-7 / 5 * G / zeta * ((1 + zeta * z) ** (5 / 7) - 1))
        rho_sym = p_sym / T_sym

        self.T = lambdify(z, T_sym)(zg)
        self.rho = lambdify(z, rho_sym)(zg)
        self.p = lambdify(z, p_sym)(zg)
        self.dpdz = lambdify(z, sym.diff(p_sym, z))(zg)
        self.dlnTdz = lambdify(z, sym.diff(T_sym, z) / T_sym)(zg)
        self.dTdz = lambdify(z, sym.diff(T_sym, z))(zg)
        self.d2Tdz = lambdify(z, sym.diff(T_sym, z, 2))(zg)
        self.dlnrhodz = lambdify(z, sym.diff(rho_sym, z) / rho_sym)(zg)
        self.dlnpdz = self.dpdz / self.p

        self.q = -self.T ** (7 / 2) * self.dlnTdz
