class KelvinHelmholtzHydroOnly():
    """
    The pure hydrodynamic version of the Kelvin-Helmholtz instability
    with a smoothly varying velocity profile. The equilibrium also allows
    for a density and temperature variation in z. The equilibrium is periodic
    with the velocity profile changing sign at z1 and z2 a Fourier grid
    is used to solve the EVP.

    The linearized equations for this setup are derived in

    Berlok, T. et al, in prep

    and the equilibrium has been previously been used for simulations in

    Lecoanet, D., McCourt, M., Quataert, E., Burns, K. J., Vasil, G. M.,
    Oishi, J. S., O’Leary, R. M. (2016).
    A validated non-linear kelvin-helmholtz benchmark for numerical
    hydrodynamics, MNRAS, 455(4), 4274–4288.
    https://doi.org/10.1093/mnras/stv2564
    """
    def __init__(self, grid, u0, delta, kx, z1=0.5, z2=1.5, a=0.05):

        self._u0 = u0
        self._delta = delta

        self.kx = kx

        self.gamma = 5.0/3
        self.p0 = 1.0
        self.rho0 = 1.0

        self.grid = grid
        self.grid.bind_to(self.make_background)

        self.z1 = z1
        self.z2 = z2
        self.a = a

        # Create initial background
        self.make_background()

        # Variables to solve for
        self.variables = ['drho', 'dvx', 'dvz', 'dT']

        self.labels = [r'$\delta \rho$', r'$\delta v_x$',
                       r'$\delta v_z$', r'$\delta T$']

        # Boundary conditions
        self.boundaries = None

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = 'sigma'

        # Equations
        eq1 = "sigma*drho = -1j*kx*v*drho -1j*kx*dvx -dlnrhodz*dvz -1.0*dz(dvz)"
        eq2 = "sigma*dvx = -1j*kx*v*dvx -dvdz*dvz -1j*kx*p/rho*drho -1j*kx*p/rho*dT"
        eq3 = ("sigma*dvz = -1j*kx*v*dvz -1/rho*dpdz*drho -1/rho*p*dz(drho)" +
               "-1/rho*dpdz*dT -1/rho*p*dz(dT)")
        eq4 = "sigma*dT = -1j*kx*v*dT -1j*kx*2/3*dvx -dlnTdz*dvz -2/3*dz(dvz)"

        self.equations = [eq1, eq2, eq3, eq4]

        # Number of equations in system
        self.dim = len(self.variables)

    @property
    def u0(self):
        return self._u0

    @u0.setter
    def u0(self, value):
        self._u0 = value
        self.make_background()

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, value):
        self._delta = value
        self.make_background()

    def make_background(self):
        import sympy as sym
        import numpy as np
        from sympy import tanh, diff, lambdify, symbols
        z = symbols("z")

        u0 = self.u0
        delta = self.delta

        zg = self.grid.zg
        globals().update(self.__dict__)

        # Define Background Functions
        v_sym = u0*(tanh((z-z1)/a) - tanh((z-z2)/a) - 1.0)
        rho_sym = rho0*(1 + delta/2*(tanh((z-z1)/a) - tanh((z-z2)/a)))

        dvdz_sym = diff(v_sym, z)
        d2vdz_sym = diff(dvdz_sym, z)

        T0 = p0/rho0

        T_sym = T0*rho0/rho_sym
        p_sym = rho_sym*T_sym

        dvdz_sym = sym.diff(v_sym, z)
        d2vdz_sym = sym.diff(dvdz_sym, z)

        # Analytic equilibrium functions
        self.T_an = lambdify(z, T_sym)
        self.rho_an = lambdify(z, rho_sym)
        self.p_an = lambdify(z, p_sym)
        self.dpdz_an = lambdify(z, sym.diff(p_sym, z))
        self.dlnTdz_an = lambdify(z, sym.diff(T_sym, z)/T_sym)
        self.dlnrhodz_an = lambdify(z, sym.diff(rho_sym, z)/rho_sym)
        self.v_an = lambdify(z, v_sym)
        self.dvdz_an = lambdify(z, dvdz_sym)
        self.d2vdz_an = lambdify(z, d2vdz_sym)

        # Analytic equilibrium functions evaluated on the grid
        ones = np.ones_like(zg)
        self.T = ones*self.T_an(zg)
        self.rho = ones*self.rho_an(zg)
        self.p = ones*self.p_an(zg)
        self.dpdz = ones*self.dpdz_an(zg)
        self.dlnTdz = ones*self.dlnTdz_an(zg)
        self.dlnrhodz = ones*self.dlnrhodz_an(zg)
        self.v = ones*self.v_an(zg)
        self.dvdz = ones*self.dvdz_an(zg)
        self.d2vdz = ones*self.d2vdz_an(zg)
