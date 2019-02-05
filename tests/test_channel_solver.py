def test_channel(show=False, verbose=False):
    """Test eigenvalue solver using ChebyshevRationalGrid"""
    import numpy as np
    from psecas import Solver, ChebyshevRationalGrid
    from psecas.systems.channel import Channel

    grid = ChebyshevRationalGrid(N=199, z='r')
    system = Channel(grid)
    solver = Solver(grid, system, do_gen_evp=True)

    # Number of modes to test
    modes = 3
    results = np.zeros(modes, dtype=np.complex128)
    checks = np.array([85.08037778, 69.4741069099, 55.4410282999])

    def sorting_strategy(E):
        """Sorting strategy for channel modes"""
        E[E.real > 100.0] = 0
        E[E.real < -10.0] = 0
        index = np.argsort(np.real(E))[::-1]
        return (E, index)

    solver.sorting_strategy = sorting_strategy

    if show:
        import matplotlib.pyplot as plt

        plt.figure(3)
        plt.clf()
        fig, axes = plt.subplots(num=3, ncols=modes, sharey=True)
    for mode in range(modes):
        Ns = np.arange(1, 6) * 32
        omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True)
        results[mode] = omega
        if show:
            phi = np.arctan(vec[2].imag / vec[2].real)
            solver.keep_result(omega, vec * np.exp(-1j * phi), mode)
            axes[mode].set_title(
                r"$\sigma = ${:1.4f}".format(omega.real), fontsize=10
            )
            axes[mode].plot(grid.zg, system.result['f'].real)
            axes[mode].plot(grid.zg, system.result['f'].imag)
            axes[mode].set_xlim(-4, 4)

    if show:
        plt.show()

    np.testing.assert_allclose(results, checks, rtol=1e-6)


if __name__ == '__main__':
    test_channel(show=True, verbose=True)
