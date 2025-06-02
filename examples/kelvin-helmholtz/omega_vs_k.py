import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, ChebyshevRationalGrid, System

# a = 0.0001
a = 0.05

# Create grid
grid = ChebyshevRationalGrid(N=199, C=8*a, z='z')


# Make a Child of the System class and override the make_background method
class Rayleigh(System):

    def __init__(self, grid, variables, eigenvalue, k, u0=1, a=1):

        self.a = a
        self.u0 = u0
        self.k = k

        super().__init__(grid, variables, eigenvalue)

    def make_background(self):
        import sympy as sym
        import numpy as np
        from sympy import tanh, diff, lambdify, symbols

        z = symbols("z")

        zg = self.grid.zg

        u0 = self.u0
        globals().update(self.__dict__)

        # Define Background Functions
        v_sym = u0 * (tanh(z / a))
        dvdz_sym = diff(v_sym, z)
        d2vdz_sym = diff(dvdz_sym, z)

        ones = np.ones_like(zg)
        self.v = ones * lambdify(z, v_sym)(zg)
        self.dvdz = ones * lambdify(z, dvdz_sym)(zg)
        self.d2vdz = ones * lambdify(z, d2vdz_sym)(zg)


# Create the Channel system
system = Rayleigh(grid, variables='f', eigenvalue='sigma', k=2*np.pi, a=a)

# Add the first (and only) equation
system.add_equation(
    "-1j*sigma*(dz(dz(f)) - k**2*f) = v*dz(dz(f)) - v*k**2*f - d2vdz*f",
    boundary=True)


# Create a solver object
solver = Solver(grid, system, do_gen_evp=True)

# Above this line identical to the kx=2*pi script
# -------------------------------------------------------------------------
omega_vec = []
kx_vec = np.linspace(0.1, 15, 50)

for kx in kx_vec:
    Ns = np.arange(1, 24) * 93
    solver.system.k = kx
    omega, vec, err = solver.iterate_solver(
        Ns, mode=0, verbose=True, tol=1e-6, guess_tol=1e-16
    )
    omega_vec.append(omega)

    print(f"kx={kx}, omega/k={omega}, Converged: {solver.system.result['converged']}")

omega_vec = np.array(omega_vec)
plt.figure(1)
plt.clf()
plt.plot(kx_vec, omega_vec.real*kx_vec)


# Find maximum growth rate

if False:
    from psecas import golden_section

    def f(kx, **kwargs):

        system = Rayleigh(grid, variables='f', eigenvalue='sigma', k=kx, a=a)

        # Add the first (and only) equation
        system.add_equation(
            "-1j*sigma*(dz(dz(f)) - k**2*f) = v*dz(dz(f)) - v*k**2*f - d2vdz*f",
            boundary=True)

        # Create a solver object
        solver = Solver(grid, system, do_gen_evp=True)

        # Iteratively solve
        omega, vec, err = solver.iterate_solver(
            Ns, mode=0, verbose=False, tol=1e-6, guess_tol=1e-16
        )

        print(f'kx = {kx}, sigma={omega.real * kx}')

        return -omega.real * kx

    # Find maximum kx with a relative tolerance
    (a, b) = golden_section(f, 7.0, 10, tol=1e-3)
    print(a, b, (a + b) / 2, -f((a + b) / 2))
    kx_max = (a + b) / 2
    sigma_max = -f((a + b) / 2)
    print(f'kx_max = {kx_max}')
else:
    # You might want to run the script with True above.
    kx_max = 8.89880774410032
    sigma_max = 3.794042001956062

plt.plot(kx_max, sigma_max, 'rx')
plt.show()
