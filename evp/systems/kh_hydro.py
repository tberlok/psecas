class KelvinHelmholtzHydroOnly():
    """Linearized equations for KH with ansitoropic viscosity for a constant
       magnetic field in the x-direction. The equilibrium is also assumed to 
       have constant density, temperature and pressure.
    """
    def __init__(self, grid, u0, delta, z1=0.5, z2=1.5, a=0.05):
        import numpy as np
        
        self._u0 = u0
        self._delta = delta

        self.gamma = 5.0/3
        self.p0 = 1.0
        self.rho0 = 1.0

        self.grid = grid
        self.grid.bind_to(self.make_background)

        self.z1 = z1
        self.z2 = z2
        self.a  = a

        # Create initial background
        self.make_background()

        # Variables to solve for
        self.variables = ['drho', 'dvx', 'dvz', 'dT']

        self.labels = [r'$\delta \rho$', r'$\delta v_x$',
                       r'$\delta v_z$', r'$\delta T$']

        # Boundary conditions
        self.boundaries = None

        # Equations (Careful! No space between minus and the term is belongs to)
        eq1 = "-1j*kx*v*drho -1j*kx*dvx -dlnrhodz*dvz -1.0*dz(dvz)"
        eq2 = "-1j*kx*v*dvx -dvdz*dvz -1j*kx*p/rho*drho -1j*kx*p/rho*dT"
        eq3 = "-1j*kx*v*dvz -1/rho*dpdz*drho -1/rho*p*dz(drho) -1/rho*dpdz*dT -1/rho*p*dz(dT)"
        eq4 = "-1j*kx*v*dT -1j*kx*2/3*dvx -dlnTdz*dvz -2/3*dz(dvz)"
        # eq1 = "-1j*v*drho -1j*dvx -dlnrhodz/kx*dvz -1.0/kx*dz(dvz)"
        # eq2 = "-1j*v*dvx -dvdz/kx*dvz -1j*p/rho*drho -1j*p/rho*dT"
        # eq3 = "-1j*v*dvz -1/rho*dpdz/kx*drho -1/rho*p/kx*dz(drho) -1/rho*dpdz/kx*dT -1/rho*p/kx*dz(dT)"
        # eq4 = "-1j*v*dT -1j*2/3*dvx -dlnTdz/kx*dvz -2/3/kx*dz(dvz)"

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
        from sympy import sqrt, exp, tanh, diff, lambdify, symbols
        z   = symbols("z")

        u0 = self.u0
        delta = self.delta

        zg = self.grid.zg
        globals().update(self.__dict__)

        # Define Background Functions
        v_sym    = u0*(tanh((z-z1)/a) - tanh((z-z2)/a) - 1.0)
        rho_sym = rho0*(1 + delta/2*(tanh((z-z1)/a) - tanh((z-z2)/a)))

        dvdz_sym = diff(v_sym, z)
        d2vdz_sym = diff(dvdz_sym, z)

        T0 = p0/rho0

        T_sym = T0*rho0/rho_sym
        p_sym = rho_sym*T_sym

        dvdz_sym = sym.diff(v_sym, z)
        d2vdz_sym = sym.diff(dvdz_sym, z)

        self.T   = np.ones_like(zg)*lambdify(z, T_sym)(zg)
        self.rho = np.ones_like(zg)*lambdify(z, rho_sym)(zg)
        self.p = np.ones_like(zg)*lambdify(z, p_sym)(zg)
        self.dpdz = np.ones_like(zg)*lambdify(z, sym.diff(p_sym, z))(zg)
        self.dlnTdz = np.ones_like(zg)*lambdify(z, sym.diff(T_sym, z)/T_sym)(zg)
        self.dlnrhodz = np.ones_like(zg)*lambdify(z, sym.diff(rho_sym, z)/rho_sym)(zg)
        self.v    = np.ones_like(zg)*lambdify(z, v_sym)(zg)
        self.dvdz = np.ones_like(zg)*lambdify(z, dvdz_sym)(zg)
        self.d2vdz = np.ones_like(zg)*lambdify(z, d2vdz_sym)(zg)
