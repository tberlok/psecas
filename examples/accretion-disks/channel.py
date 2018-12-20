import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, ChebyshevRationalGrid
from psecas.systems.channel import Channel

grid = ChebyshevRationalGrid(N=199, z='r')
system = Channel(grid)

ch = Solver(grid, system, do_gen_evp=True)


def sorting_strategy(E):
    """Sorting strategy for channel modes"""
    E[E.real > 100.0] = 0
    E[E.real < -10.0] = 0
    index = np.argsort(np.real(E))[::-1]
    return (E, index)


# Overwrite the default sorting strategy in the Solver class
ch.sorting_strategy = sorting_strategy

plt.figure(1)
plt.clf()
modes = 3
fig, axes = plt.subplots(num=1, ncols=modes, sharey=True)
for mode in range(modes):
    Ns = np.hstack((np.arange(1, 6) * 32, np.arange(2, 12) * 64))
    omega, vec, err = ch.iterate_solver(Ns, mode=mode, tol=1e-8, verbose=True)
    phi = np.arctan(vec[2].imag / vec[2].real)
    ch.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)
    axes[mode].set_title(
        r"$\sigma = ${:1.4f}".format(omega.real), fontsize=10
    )
    axes[mode].plot(grid.zg, system.result["f"].real)
    axes[mode].plot(grid.zg, system.result["f"].imag)
    axes[mode].set_xlim(-4, 4)
plt.show()
