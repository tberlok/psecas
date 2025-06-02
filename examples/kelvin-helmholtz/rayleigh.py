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

# system.add_equation(
#     "sigma*(dz(dz(f)) - k**2*f) = v*dz(dz(f)) - v*k**2*f - d2vdz*f",
#     boundary=True)

# Overwrite the default sorting method in the Solver class
# class RayleighSolver(Solver):
#     def sorting_strategy(self, E):
#         """Sorting strategy for channel modes. E is a list of eigenvalues"""
#         E[E.real > 100.0] = 0
#         E[E.real < -10.0] = 0
#         index = np.argsort(np.real(E))[::-1]
#         return (E, index)


# Create a solver object
solver = Solver(grid, system, do_gen_evp=True)

# plt.figure(1)
# plt.clf()

# fig, axes = plt.subplots(num=1)
# # List of resolutions to try
# Ns = np.hstack((np.arange(1, 6) * 32, np.arange(2, 12) * 64))
# sigma, vec = solver.solve()
# omega, vec, err = solver.iterate_solver(
#     Ns, mode=0, verbose=True, tol=1e-6
# )
# index = np.argmax(np.abs(vec))
# phi = np.arctan(vec[index].imag / vec[index].real)
# vec *= np.exp(-1j * phi)

# plt.plot(grid.zg, system.result['f'].real)
# plt.plot(grid.zg, system.result['f'].imag)
# plt.show()

omega_vec = []
# kx_vec = np.linspace(0.002, 0.9, 60)
Ns = np.arange(1, 24) * 93
omega, vec, err = solver.iterate_solver(
    Ns, mode=0, verbose=True, tol=1e-6, guess_tol=1e-16
)

print(f"Converged: {solver.system.result['converged']}")

if not solver.system.result['converged']:
    raise RuntimeError(f"Not converged! err={err}")
else:
    print(f'Converged with omega/k={omega}, err={err}')
    print(f'Converged with omega={system.k*omega}')
