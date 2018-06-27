
def test_mti_solution(show=False, verbose=False):
    """Test the solver by comparing the eigenvalue with old result"""
    import numpy as np
    from evp import Solver, ChebyshevExtremaGrid
    from evp.systems.mti import MagnetoThermalInstability

    grid = ChebyshevExtremaGrid(N=64, zmin=0, zmax=1)

    system = MagnetoThermalInstability(grid, beta=1e5, Kn0=200)

    mti = Solver(grid, system, kx=4*np.pi)

    Ns = np.hstack(np.arange(1, 10)*16)
    omega, vec, err = mti.iterate_solver(Ns, i=0, tol=1e-8, verbose=verbose)

    if show:
        from evp import plot_solution
        phi = np.arctan(vec[2].imag/vec[2].real)
        mti.keep_result(omega, vec*np.exp(-1j*phi))

        plot_solution(mti, smooth=True)

    np.testing.assert_allclose(1.7814514515967603, omega, atol=1e-8)

    return err

def test_kh_uniform_solution(show=False, verbose=False):
    """Test the solver by comparing the eigenvalue with old result"""
    import numpy as np
    from evp import Solver, FourierGrid
    from evp.systems.kh_uniform import KelvinHelmholtzUniform

    grid = FourierGrid(N=64, zmin=0, zmax=2)

    system = KelvinHelmholtzUniform(grid, beta=1e4, nu=1e-2)

    kh = Solver(grid, system, 3.52615254237)

    Ns = np.hstack((np.arange(1, 4)*32, np.arange(2, 12)*64))
    omega, v, err = kh.iterate_solver(Ns, tol=1e-8, verbose=verbose)

    if show:
        from evp import plot_solution
        plot_solution(kh, smooth=True)

    np.testing.assert_allclose(1.66548246011, omega, atol=1e-8)
    return err

if __name__ == '__main__':
    err = test_mti_solution(show=True, verbose=True)
    err = test_kh_uniform_solution(show=True, verbose=True)
