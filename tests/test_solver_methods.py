def test_solver_methods(verbose=False):
    """Show how the solver class methods can be called directly,
       using the MTI as an example.
       This can be useful when setting up a new problem.
    """
    import numpy as np
    from psecas import Solver, ChebyshevExtremaGrid
    from psecas.systems.mti import MagnetoThermalInstability
    from scipy.linalg import eig

    grid = ChebyshevExtremaGrid(N=64, zmin=0, zmax=1)

    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)

    solver = Solver(grid, system)

    solver.get_matrix1(verbose=verbose)

    solver.get_matrix2(verbose=verbose)

    E, V = eig(solver.mat1.toarray(), solver.mat2.toarray())

    # Sort the eigenvalues
    E, index = solver.sorting_strategy(E)

    mode = 0

    # Choose the eigenvalue mode value only
    sigma = E[index[mode]]
    v = V[:, index[mode]]

    np.testing.assert_allclose(1.7814514515967603, sigma, atol=1e-8)


if __name__ == '__main__':
    test_solver_methods(True)
