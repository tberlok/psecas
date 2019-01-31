import numpy as np
from psecas import Solver, FourierGrid
from psecas.systems.kh_uniform import KelvinHelmholtzUniform
from psecas import save_system, write_athena
from psecas import plot_solution

kxmax = 4.54704305
# omega = 1.7087545
grid = FourierGrid(N=64, zmin=0, zmax=2)
system = KelvinHelmholtzUniform(grid, beta=1e3, nu=1e-2, kx=kxmax)
# Set up a solver
solver = Solver(grid, system)
# Iteratively solve
Ns = np.hstack((np.arange(1, 5) * 16, np.arange(3, 24) * 32))
omega, vec, err = solver.iterate_solver(Ns, verbose=True, tol=1e-8)
phi = np.arctan(vec[2].imag / vec[2].real)
solver.keep_result(omega, vec * np.exp(-1j * phi), mode=0)

# Normalize eigenmodes
y = np.vstack(
    [
        system.result["dvx"].real,
        system.result["dvx"].imag,
        system.result["dvz"].real,
        system.result["dvz"].imag,
    ]
)

val = np.max(np.abs(y))
for key in system.variables:
    system.result[key] /= val

# Normalize eigenmodes
y = np.vstack(
    [
        system.result["dvx"].real,
        system.result["dvx"].imag,
        system.result["dvz"].real,
        system.result["dvz"].imag,
    ]
)

val = np.max(np.abs(y))
for key in system.variables:
    system.result[key] /= val

# Write files for loading into Athena
write_athena(system, Nz=256, Lz=2.0, path="./athena-solutions/", name="khi_nu")

save_system(system, "./athena-solutions/khi_nu.p")

Lx = 2 * np.pi / system.kx

print('')
print('Eigenvalue is:', omega)
print('Lx should be:', Lx)

plot_solution(system, filename='./athena-solutions/khi_nu.pdf')
