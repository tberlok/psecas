def test_mri_solution(show=False, verbose=False):
    """Test eigenvalue solver using ChebyshevExtremaGrid"""
    import numpy as np
    from psecas import Solver, ChebyshevExtremaGrid, System

    class HallMRI(System):
        def __init__(self, grid, kz, variables, eigenvalue):
            # Set parameters
            self.q = 1.5
            self.eta = 0.003
            self.lh = 1
            self.h = 0.25
            self.va = 0.002
            self.kz = kz

            super().__init__(grid, variables, eigenvalue)

    # Create a grid
    grid = ChebyshevExtremaGrid(N=128, zmin=1, zmax=2, z='r')

    variables = ['rho', 'vr', 'vphi', 'vz', 'Aphi', 'bphi']

    kz = 2*np.pi

    # Create the system
    system = HallMRI(grid, kz, variables=variables, eigenvalue='sigma')

    # The linearized equations
    system.add_equation("-r*sigma*rho = r*dr(vr) + vr + 1j*kz*r*vz")
    system.add_equation("-r*r*sigma*vr = - 2*r**(2-q)*vphi + h**2*r*r*dr(rho) + va**2*(DrAphi)")
    system.add_equation("-sigma*vphi = + (2-q)*r**(-q)*vr - va**2*1j*kz*bphi")
    system.add_equation("-sigma*vz = h**2*1j*kz*rho")
    system.add_equation("-r*r*sigma*Aphi = + r*r*vr - eta*(DrAphi) + lh*va*1j*kz*r*r*bphi")
    system.add_equation("-r*r*sigma*bphi = - 1j*kz*r*r*vphi - 1j*kz*q*r**(2-q)*Aphi - eta*(Drbphi) - lh*va*1j*kz*(DrAphi)")

    # The boundary conditions
    Aphi_bound = 'r**2*dr(dr(Aphi)) + r*dr(Aphi) - Aphi = 0'
    system.add_boundary('vr', 'Dirichlet', 'Dirichlet')
    system.add_boundary('vphi', 'Dirichlet', 'Dirichlet')
    system.add_boundary('vz', 'Neumann', 'Neumann')
    system.add_boundary('Aphi', Aphi_bound, Aphi_bound)
    system.add_boundary('bphi', 'Dirichlet', 'Dirichlet')

    # Short hands for long expressions for derivatives
    system.add_substitution('DrAphi = r*r*dr(dr(Aphi)) + r*dr(Aphi) - Aphi - kz**2*r*r*Aphi')
    system.add_substitution('Drbphi = r*r*dr(dr(bphi)) + r*dr(bphi) - bphi - kz**2*r*r*bphi')

    solver = Solver(grid, system)


    mode = 0
    Ns = np.hstack(np.arange(1, 10) * 32)
    omega, vec, err = solver.iterate_solver(
        Ns, mode=mode, tol=1e-8, verbose=verbose
    )

    if show:
        from psecas import plot_solution

        phi = np.arctan(vec[2].imag / vec[2].real)
        solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)

        plot_solution(system, smooth=True, num=1)

    np.testing.assert_allclose(0.09892641, omega, atol=1e-8)

    return err


if __name__ == '__main__':
    err = test_mri_solution(show=True, verbose=True)
