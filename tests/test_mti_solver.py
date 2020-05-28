def test_mti_solution(show=False, verbose=False):
    """Test eigenvalue solver using ChebyshevExtremaGrid"""
    import numpy as np
    from psecas import Solver, ChebyshevExtremaGrid
    from psecas.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=64, zmin=0, zmax=1)

    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200, kx=4 * np.pi)
    system.beta = 1e5
    system.Kn0 = 200

    assert system.Kn0 == 200
    assert system.beta == 1e5

    solver = Solver(grid, system)

    Ns = np.hstack(np.arange(1, 10) * 16)
    mode = 0
    omega, vec, err = solver.iterate_solver(
        Ns, mode=mode, tol=1e-8, verbose=verbose
    )

    system.get_bx_and_by()

    if show:
        from psecas import plot_solution

        phi = np.arctan(vec[2].imag / vec[2].real)
        solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)

        plot_solution(system, smooth=True, num=1)

    np.testing.assert_allclose(1.7814514515967603, omega, atol=1e-8)

    return err


if __name__ == '__main__':
    err = test_mti_solution(show=True, verbose=True)
