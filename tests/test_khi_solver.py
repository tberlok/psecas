def test_kh_uniform_solution(show=False, verbose=False):
    """Test eigenvalue solver using FourierGrid"""
    import numpy as np
    from psecas import Solver, FourierGrid
    from psecas.systems.kh_uniform import KelvinHelmholtzUniform

    grid = FourierGrid(N=64, zmin=0, zmax=2)

    system = KelvinHelmholtzUniform(grid, beta=1e4, nu=1e-2, kx=3.52615254237)
    system.u0 = 1.

    solver = Solver(grid, system)

    Ns = np.hstack((np.arange(2, 16) * 32, np.arange(2, 12) * 64))
    for useOPinv in [True, False]:
        omega, v, err = solver.iterate_solver(Ns, tol=1e-5, useOPinv=useOPinv,
                                              verbose=verbose)
    np.testing.assert_allclose(1.66548246011, omega, atol=1e-5)

    if show:
        from psecas import plot_solution

        plot_solution(system, smooth=True, num=2)


    # Check that information converged is set to False when solver does not converge
    solver.iterate_solver(np.arange(1, 3) * 16, tol=1e-16, useOPinv=useOPinv,
                          verbose=verbose)
    assert solver.system.result['converged'] is False

    return err


if __name__ == '__main__':
    err = test_kh_uniform_solution(show=True, verbose=True)
