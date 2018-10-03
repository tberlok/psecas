import numpy as np
import matplotlib.pyplot as plt
from evp import Solver, System
from evp import HermiteGrid, SincGrid, ChebyshevRationalGrid

"""
    Illustration of the behaviour of three different grids on the
    infinite domain. The example can be found in Boyd page 131-133.
    We consider the eigenvalue problem
    uₓₓ + (λ - x²) u = 0 with |u| → 0 as |x| → ∞
    which has exact solutions uⱼ(x) = exp(-x²/2)H₋j(x) where
    H₋j(x) is the jᵗʰ Hermite polynomial. The exact eigenvalues are
    λⱼ = 2 j + 1.
"""


N = 40

# Create grids
grid1 = ChebyshevRationalGrid(N=N - 1, C=2)
grid2 = SincGrid(N=N, C=2)
grid3 = HermiteGrid(N=N, C=1)

grids = list([grid1, grid2, grid3])

# Mode number
j = 8


class HermiteSolver(Solver):
    def sorting_strategy(self, E):
        """Sorting strategy for hermite modes. E is a list of eigenvalues"""
        E[E.real > 100.0] = 0
        # Ignore eigenvalues that are zero
        E[E.real == 0.0] = 1e5
        # Sort from smallest to largest eigenvalue
        index = np.argsort(np.real(E))
        return (E, index)


plt.figure(1)
plt.clf()
fig, axes = plt.subplots(num=1, nrows=3, sharex=True)

for ii, grid in enumerate(grids):
    system = System(grid, variables="u", eigenvalue="sigma")
    if isinstance(grid, ChebyshevRationalGrid):
        boundary = True
    else:
        boundary = False
    system.add_equation("sigma*u = -dz(dz(u)) + z**2*u", boundary=boundary)
    solver = HermiteSolver(grid, system)
    omega, vec = solver.solve(mode=j)
    phi = np.arctan(vec[2].imag / vec[2].real)
    vec *= np.exp(-1j * phi)
    solver.keep_result(omega, vec / np.max(np.abs(vec)), mode=j)
    msg = type(grid).__name__ + r" $\sigma = ${:1.8f}"
    axes[ii].set_title(msg.format(omega.real), fontsize=10)
    if isinstance(grid, HermiteGrid):
        z = np.linspace(3 * grid.zmin / 5, 3 * grid.zmax / 5, 5000)
    else:
        z = np.linspace(grid.zmin, grid.zmax, 5000)
    axes[ii].plot(z, grid.interpolate(z, system.result["u"].real), "C0-")
    axes[ii].plot(z, grid.interpolate(z, system.result["u"].imag), "C1-")
    axes[ii].plot(grid.zg, system.result["u"].real, "C0+")
    axes[ii].plot(grid.zg, system.result["u"].imag, "C1+")
    axes[ii].set_xlim(-15, 15)

plt.show()
