class KelvinHelmholtzMiura:
    """
       Kelvin-Helmholtz instability with a constant magnetic field in the
       x-direction. The equilibrium is assumed to have
       constant density, temperature and pressure. The velocity profile varies
       smoothly.

       This is a recalculation of a setup from

       Miura, A., & Pritchett, P. L. (1982). Nonlocal Stability Analysis of
       the MHD Kelvin-Helmholtz Instability in a Compressible Plasma. Journal
       of Geophysical Research, 87(A9), 7431–7444.
       https://doi.org/10.1029/JA087iA09p07431

       which was used for test simulations in

       Frank, A., Jones, T. W., Ryu, D., & Gaalaas, J. B. (1996). The
       Magnetohydrodynamic Kelvin-Helmholtz Instability: A Two-dimensional
       Numerical Study. The Astrophysical Journal, 460, 777.
       https://doi.org/10.1086/177009

       The script calculates the mode structure and growth rate for the weak
       field case listed in Table I of Frank et al and reproduces Figure 4
       in Miura & Pritchett (1982). The latter calculation requires some
       patience.

    """

    def __init__(self, grid, kx, B, z1=0.5, a=1/20.):
        import numpy as np

        self.kx = kx

        # Parameters
        self.u0 = 1.0
        self.gamma = 5/3
        self.p = 3/5
        self.rho = 1.0
        self.mu0 = 1.0

        self.B = B
        self.va = self.B / np.sqrt(self.mu0 * self.rho)

        self.grid = grid
        self.grid.bind_to(self.make_background)

        self.z1 = z1
        self.a = a

        # Create initial background
        self.make_background()

        # Variables to solve for
        # self.variables = ["drho", "dA", "dvx", "dvz", "dT"]
        # self.labels = [
        #     r"$\delta \rho$",
        #     r"$\delta A$",
        #     r"$\delta v_x$",
        #     r"$\delta v_z$",
        #     r"$\delta T$",
        # ]

        # # Boundary conditions
        # self.boundaries = [True, True, True, True, True]

        self.variables = ["dA", "dvx", "dvz", "dp"]
        self.labels = [
            r"$\delta A$",
            r"$\delta v_x$",
            r"$\delta v_z$",
            r"$\delta p/p$",
        ]

        # Boundary conditions
        self.boundaries = [False, False, True, False]

        # Number of equations in system
        self.dim = len(self.variables)

        # String used for eigenvalue (do not use lambda!)
        self.eigenvalue = "sigma"

        # # Equations
        # eq1 = "sigma*drho = -1j*kx*v*drho -1j*kx*dvx -1.0*dz(dvz)"
        # eq2 = "sigma*dA = -1j*kx*v*dA +1.0*dvz"
        # eq3 = "sigma*dvx = -1j*kx*v*dvx -dvdz*dvz -1j*kx*p/rho*drho -1j*kx*p/rho*dT"
        # eq4 = "sigma*dvz = -1j*kx*v*dvz -1/rho*p*dz(drho) -1/rho*p*dz(dT) +va**2*dz(dz(dA)) -va**2*kx**2*dA"
        # eq5 = "sigma*dT = -1j*kx*v*dT -1j*kx*2/3*dvx -2/3*dz(dvz)"

        # self.equations = [eq1, eq2, eq3, eq4, eq5]

        eq1 = "sigma*dA = -1j*kx*v*dA +1.0*dvz"
        eq2 = "sigma*dvx = -1j*kx*v*dvx -dvdz*dvz -1j*kx*p/rho*dp"
        eq3 = "sigma*dvz = -1j*kx*v*dvz -1/rho*p*dz(dp) +va**2*dz(dz(dA)) -va**2*kx**2*dA"
        eq4 = "sigma*dp = -1j*kx*v*dp -5/3*1j*kx*dvx -5/3*dz(dvz)"

        self.equations = [eq1, eq2, eq3, eq4]

    def make_background(self):
        from sympy import tanh, diff, lambdify, symbols

        z = symbols("z")

        zg = self.grid.zg

        z1 = self.z1
        a = self.a
        u0 = self.u0

        # Define Background Functions
        v_sym = -u0/2.0 * tanh((z - z1) / a)

        dvdz_sym = diff(v_sym, z)
        d2vdz_sym = diff(dvdz_sym, z)

        self.v = lambdify(z, v_sym)(zg)
        self.dvdz = lambdify(z, dvdz_sym)(zg)
        self.d2vdz = lambdify(z, d2vdz_sym)(zg)


if __name__ == '__main__':
    import numpy as np
    from psecas import Solver, LegendreExtremaGrid
    import matplotlib.pyplot as plt
    from psecas import plot_solution

    grid = LegendreExtremaGrid(N=200, zmin=-0.5, zmax=0.5)
    system = KelvinHelmholtzMiura(grid, kx=2*np.pi, B=0.2, z1=0, a=1/25.0)
    solver = Solver(grid, system)

    Ns = np.arange(1, 8) * 64
    omega, v, err = solver.iterate_solver(Ns, verbose=True, tol=1e-8)

    print('\nB={:1.2f} has gammma*2a = {:1.6f}'.format(system.B,
                                                       omega.real*system.a*2))

    plot_solution(system, num=1, filename='Ryu_and_Frank1996.pdf')

    # Make figure 4 in Miura & Pritchett (1982)
    # This calculation takes some time.
    if True:

        # The iterative solver is more precise but slow at high kx
        # Set fast=True to instead run with a fixed grid size and get a figure
        # much faster. The result looks okay but is imprecise at high k!
        fast = False

        plt.figure(2)
        plt.clf()

        for B in [1e-8, 0.2, 0.3, 0.4]:
            system = KelvinHelmholtzMiura(grid, 2*np.pi, B=B, z1=0, a=1/20.0)
            # Speed up calculation by changing boundary condition
            system.boundaries = [True, True, True, True]
            solver = Solver(grid, system)

            omega_vec = []
            kx_vec = []
            for kx in np.linspace(0.1, 16, 24):
                system.kx = kx
                if fast:
                    grid.N = 200
                    omega, v = solver.solve()
                else:
                    omega, v, err = solver.iterate_solver(Ns, verbose=False,
                                                          tol=1e-3,
                                                          atol=1e-3)
                    if not system.result['converged']:
                        break
                msg = 'kx = {:1.3f}, omega = {:1.3f}, N={}'
                print(msg.format(kx, omega.real, grid.N))
                omega_vec.append(omega)
                kx_vec.append(kx)
            omega_vec = np.array(omega_vec)
            kx_vec = np.array(kx_vec)
            plt.plot(kx_vec*system.a*2/system.u0, omega_vec.real*system.a*2,
                     label=r'$B = {:1.1f}$'.format(system.B))
        plt.xlim(0, 2)
        plt.ylim(0, 0.2)
        plt.legend()
        plt.xlabel(r"$2 k_x a$")
        plt.ylabel(r"$\gamma 2 a/V_0$")
        plt.savefig('Miura_figure4.pdf')
        plt.show()
