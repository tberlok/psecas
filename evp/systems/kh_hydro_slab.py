class KelvinHelmholtzHydroOnlySlab():
    """
    The pure hydro version of the Kelvin-Helmholtz instability.

    The equilibrium changes sign at z=0 and is not periodic.
    """
    def __init__(self, grid, u0, delta, kx, a=0.147):

        self._u0 = u0
        self._delta = delta

        self.kx = kx

        self.gamma = 5.0/3
        self.p0 = 1.0
        self.rho0 = 1.0

        self.grid = grid
        self.grid.bind_to(self.make_background)

        self.a = a

        # Create initial background
        self.make_background()

        # Variables to solve for
        self.variables = ['drho', 'dvx', 'dvz', 'dT']

        self.labels = [r'$\delta \rho$', r'$\delta v_x$',
                       r'$\delta v_z$', r'$\delta T$']

        # Boundary conditions
        self.boundaries = None

        # Equations (Careful! No space behind minus)
        eq1 = "-1j*kx*v*drho -1j*kx*dvx -dlnrhodz*dvz -1.0*dz(dvz)"
        eq2 = "-1j*kx*v*dvx -dvdz*dvz -1j*kx*p/rho*drho -1j*kx*p/rho*dT"
        eq3 = "-1j*kx*v*dvz -1/rho*dpdz*drho -1/rho*p*dz(drho) -1/rho*dpdz*dT -1/rho*p*dz(dT)"
        eq4 = "-1j*kx*v*dT -1j*kx*2/3*dvx -dlnTdz*dvz -2/3*dz(dvz)"

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
        v_sym = u0*(1.0 + tanh(z/a))/2.0
        rho_sym = rho0*(1.0 + tanh(-z/a))*(delta-1)/2 + 1

        dvdz_sym = diff(v_sym, z)
        d2vdz_sym = diff(dvdz_sym, z)

        T0 = p0/rho0

        T_sym = T0*rho0/rho_sym
        p_sym = rho_sym*T_sym

        dvdz_sym = sym.diff(v_sym, z)
        d2vdz_sym = sym.diff(dvdz_sym, z)

        ones = np.ones_like(zg)
        self.T = ones*lambdify(z, T_sym)(zg)
        self.rho = ones*lambdify(z, rho_sym)(zg)
        self.p = ones*lambdify(z, p_sym)(zg)
        self.dpdz = ones*lambdify(z, sym.diff(p_sym, z))(zg)
        self.dlnTdz = ones*lambdify(z, sym.diff(T_sym, z)/T_sym)(zg)
        self.dlnrhodz = ones*lambdify(z, sym.diff(rho_sym, z)/rho_sym)(zg)
        self.v = ones*lambdify(z, v_sym)(zg)
        self.dvdz = ones*lambdify(z, dvdz_sym)(zg)
        self.d2vdz = ones*lambdify(z, d2vdz_sym)(zg)
