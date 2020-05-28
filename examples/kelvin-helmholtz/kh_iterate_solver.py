import numpy as np
from psecas import Solver, FourierGrid
from psecas.systems.kh_hydro import KelvinHelmholtzHydroOnly
import time
import matplotlib.pyplot as plt

# Initialize grid, system and solver for hydrodynamic KHI
kxmax = 3.5128286141291243
grid = FourierGrid(N=64, zmin=0.0, zmax=2.0)
system = KelvinHelmholtzHydroOnly(grid, u0=1.0, delta=1.0, kx=kxmax)
solver = Solver(grid, system)

# Define a sorting strategy for the eigenvalues
# This function depends on the physical system
def sorting_strategy(E):
    """
    The default sorting strategy.

    "Large" real and imaginary eigenvalues are removed and the eigenvalues
    are sorted from largest to smallest
    """
    import numpy as np

    # Set large values to zero, as they are presumably numerical artifact
    # and unphysical.
    E[np.abs(E.real) > 20.0] = 0
    E[np.abs(E.imag) > 20.0] = 0
    # Sort from largest to smallest eigenvalue
    index = np.argsort(np.real(E))[::-1]
    return (E, index)


# Overwrite the standard sorting strategy with the function defined above
solver.sorting_strategy = sorting_strategy

# Solve the full matrix problem keeping all eigenvalues and vectors
# results are stored in solver.E (a 1D array with the eigenvalues) and solver.v
# (a 2D array of the eigenvectors).
solver.solve(saveall=True)

"""
The approximation to the physical eigenvalue problem is a matrix eigenvalue
problem. The matrix has size d times N (where d is the number of linearized
equations and N is the number of grid points). There are thus d times N
eigenvalues. Obviously, the number of physical eigenvalues cannot depend on
the number of grid points. Some of the solutions found above are therefore not
physical (see e.g. the book by  Boyd).  The solutions were also calculated at
fixed number of grid  points, N. Convergent solutions are found by increasing
N and discarding the eigenvalues that are not present at higher N.
This would normally be done by manually increasing N, calling solve, looking
at the eigenvalues, increasing N, calling solve,
comparing with previous iteration and so on. This procedure is tedious.

The solver class in Psecas contains a different method, iterate_solver,
which increases N until one eigenvalue agrees with the value found at the
previous value of N (to within relative and absolute tolerances, tol & atol).
This method can only find one eigenmode at a time (unlike the solve method
which finds all eigenvalues but is unable to check for convergence and can be
very expensive to call at high N). Most often, one is only interested in the
fastest growing mode (or the first few). This can be controlled with the
input 'mode' in the call to iterate_solver.
"""

# Let us find converged solutions

# Define a range of grid resolutions to try
Ns = np.hstack((np.arange(1, 5) * 32, np.arange(3, 20) * 64))

# Solve for the fastest growing mode (mode=0)
omega, v, err = solver.iterate_solver(Ns, verbose=True, tol=1e-6, mode=0)

# Solve for the second fastest growing mode (mode=1)
omega, v, err = solver.iterate_solver(Ns, verbose=True, tol=1e-6, mode=1)

"""
The sorting_strategy function is necessary to weed out the worst of the
numerical eigenvalues. For some physical systems, infinite or nan solutions
are returned. These are discarded by the sorting_strategy function.
The solver class is initialized with a standard sorting_strategy that has
worked well on the problems I have considered so far.
One should however be careful not to remove physical eigenvalues and for
some problems it is probably necessary to define a new sorting_strategy.
"""
