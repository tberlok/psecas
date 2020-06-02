import numpy as np
from psecas import Solver, System
from psecas import ChebyshevExtremaGrid
import pytest

"""
We set up systems with errors, and see if Psecas gives a reasonable
error, i.e., a NameError.
"""


# Create grid
N = 32
grid = ChebyshevExtremaGrid(N, -1, 1)


# Create a solver object
def test_parser_findmatrices(verbose=False):
    """
    Here we add the value A
    to the equation without setting it in the system.
    This should return a NameError
    """
    
    # Create system
    system = System(grid, variables='f', eigenvalue='sigma')

    # Add the first (and only) equation
    system.add_equation("sigma*z*f = dz(dz(f)) + 2*A*f")

    with pytest.raises(NameError) as e_info:

        solver = Solver(grid, system)

    if verbose:
        print(str(e_info.value))


def test_parser_boundaries(verbose=False):
    """
    Here we add the value B
    to the boundary without setting it in the system.
    This should return a NameError
    """
    # Create system
    system = System(grid, variables='f', eigenvalue='sigma')

    # Add the first (and only) equation
    system.add_equation("sigma*z*f = dz(dz(f))")
    
    system.add_boundary('f', 'Dirichlet', 'B*f = 0')

    solver = Solver(grid, system)

    with pytest.raises(NameError) as e_info:
        # The error is found when the solve method is called
        solver.solve()

    if verbose:
        print(str(e_info.value))

if __name__ == '__main__':
    test_parser_findmatrices(True)
    test_parser_boundaries(True)
