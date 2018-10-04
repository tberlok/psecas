class KelvinHelmholtzUniform:
    """
       Kelvin-Helmholtz instability with anisotropic viscosity and a constant
       magnetic field in the x-direction. The equilibrium is assumed to have
       constant density, temperature and pressure. The velocity profile varies
       smoothly and the setup is periodic.

       More details about this specific setup can be found in

       Berlok et al, in prep.

       Another reference for the KHI with anisotric viscosity is

       Suzuki, K., Ogawa, T., Matsumoto, Y., & Matsumoto, R. (2013).
       Magnetohydrodynamic simulations of the formation of cold fronts in
       clusters of galaxies: Effects of anisotropic viscosity. Astrophysical
       Journal, 768(2). https://doi.org/10.1088/0004-637X/768/2/175
    """

    def __init__(self, grid, beta, nu, kx, u0=1, z1=0.5, z2=1.5, a=0.05):
        import numpy as np

        # Parameters that change (TODO: make nu, beta, and chi0 part of this)
        self._u0 = u0
        self.nu = nu
        self.beta = beta

        self.kx = kx

        self.gamma = 5.0 / 3
        self.p = 1.0
        self.rho = 1.0
        self.mu0 = 1.0

        self.B = np.sqrt(2 * self.p / beta)
        self.va = self.B / np.sqrt(self.mu0 * self.rho)

        self.grid = grid
        self.grid.bind_to(self.make_background)

        self.z1 = z1
        self.z2 = z2
        self.a = a

        # Create initial background
        self.make_background()

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
        self.boundaries = [False, False, False, False, False]

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # Equations (Careful! No space behind minus
        eq1 = "sigma*drho = -1j*kx*v*drho -1j*kx*dvx -1.0*dz(dvz)"
        eq2 = "sigma*dA = -1j*kx*v*dA +1.0*dvz"
        eq3 = "sigma*dvx = -1j*kx*v*dvx -dvdz*dvz -1j*kx*p/rho*drho -1j*kx*p/rho*dT -nu*4/3*kx**2*dvx -nu*2*kx**2*dvdz*dA -nu*1j*kx*2/3*dz(dvz)"
        eq4 = "sigma*dvz = -1j*kx*v*dvz -1/rho*p*dz(drho) -1/rho*p*dz(dT) +va**2*dz(dz(dA)) -va**2*kx**2*dA -1j*kx*nu*2/3*dz(dvx) -1j*kx*nu*d2vdz*dA -1j*kx*nu*dvdz*dz(dA) +nu*1/3*dz(dz(dvz))"
        eq5 = "sigma*dT = -1j*kx*v*dT -1j*kx*2/3*dvx -2/3*dz(dvz)"

        self.equations = [eq1, eq2, eq3, eq4, eq5]

    @property
    def u0(self):
        return self._u0

    @u0.setter
    def u0(self, value):
        self._u0 = value
        self.make_background()

    def make_background(self):
        from sympy import tanh, diff, lambdify, symbols

        z = symbols("z")

        zg = self.grid.zg

        u0 = self._u0
        z1 = self.z1
        z2 = self.z2
        a = self.a

        # Define Background Functions
        v_sym = u0 * (tanh((z - z1) / a) - tanh((z - z2) / a) - 1.0)

        dvdz_sym = diff(v_sym, z)
        d2vdz_sym = diff(dvdz_sym, z)

        self.v = lambdify(z, v_sym)(zg)
        self.dvdz = lambdify(z, dvdz_sym)(zg)
        self.d2vdz = lambdify(z, d2vdz_sym)(zg)
