import numpy as np
from psecas import Solver, FourierGrid
from psecas.systems.kh_uniform import KelvinHelmholtzUniform
from psecas import save_system
from psecas import plot_solution
import matplotlib.pyplot as plt

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

# Save system pickle object
save_system(system, "./khi_nu.p")

# Print out some information
Lx = 2 * np.pi / system.kx
print('')
print('Eigenvalue is:', omega)
print('Lx should be:', Lx)

# Make a plot
plt.figure(1)
plt.plot(kxmax, omega.real, "+")
plot_solution(system, filename='./khi_nu.pdf')

# Write files for loading into Athena
s = system
c_dic = {}
for key in s.variables:
    c_dic.update({key: s.grid.to_coefficients(s.result[key])})

perturb = []
for key in ['drho', 'dvx', 'dvz', 'dT', 'dA']:

    perturb.append(c_dic[key].real)
    perturb.append(c_dic[key].imag)

perturb = np.transpose(perturb)
np.savetxt(
    'khi_nu.txt',
    perturb,
    delimiter="\t",
    newline="\n",
    fmt="%1.16e",
)
