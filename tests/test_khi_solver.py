def test_kh_uniform_solution(show=False, verbose=False):
    """Test eigenvalue solver using FourierGrid"""
    import numpy as np
    from psecas import Solver, FourierGrid
    from psecas.systems.kh_uniform import KelvinHelmholtzUniform

    grid = FourierGrid(N=64, zmin=0, zmax=2)

    system = KelvinHelmholtzUniform(grid, beta=1e4, nu=1e-2, kx=3.52615254237)

    solver = Solver(grid, system)

    Ns = np.hstack((np.arange(2, 16) * 32, np.arange(2, 12) * 64))
    omega, v, err = solver.iterate_solver(Ns, tol=1e-5, verbose=verbose)

    if show:
        from psecas import plot_solution

        plot_solution(system, smooth=True, num=2)

    np.testing.assert_allclose(1.66548246011, omega, atol=1e-5)
    return err


if __name__ == '__main__':
    err = test_kh_uniform_solution(show=True, verbose=True)
