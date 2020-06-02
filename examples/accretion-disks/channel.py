import numpy as np
import matplotlib.pyplot as plt
from psecas import Solver, ChebyshevRationalGrid
from psecas.systems.channel import Channel

"""The linearized equations for channel mode, equation 17 in

*MRI channel flows in vertically stratified models of accretion discs*,
https://doi.org/10.1111/j.1365-2966.2010.16759.x,
by Henrik N. Latter, Sebastien Fromang, Oliver Gressel

is 

    F'' + K² h F = 0

and is solved  by employing a Neumann boundary condition on F.

One of the modes obtained here, with KH=3.1979, is not found
when solving 

    -h K² G = z G' + G'' 
    
Visual inspection reveals that the extra mode is numerical garbage.
Automated detection of such modes would be a good feature to implement.
"""


grid = ChebyshevRationalGrid(N=199, z='z')
system = Channel(grid)

ch = Solver(grid, system)


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
modes = 11
fig, axes = plt.subplots(num=1, nrows=2, ncols=modes, sharex=True,
                         sharey='row')
for mode in range(modes):
    Ns = np.hstack((np.arange(3, 6) * 32, np.arange(4, 12) * 64))
    K2, vec, err = ch.iterate_solver(Ns, mode=mode, tol=1e-4, verbose=True, guess_tol=5e-3)
    K = np.sqrt(K2.real)
    phi = np.arctan(vec[2].imag / vec[2].real)
    A = np.max(np.abs(vec))
    ch.keep_result(K2, vec * np.exp(-1j * phi)/A, mode=mode)
    axes[0, mode].set_title(
        r"$\sigma = ${:1.4f}".format(K), fontsize=10
    )
    axes[0, mode].plot(grid.zg, system.result["F"].real)
    axes[0, mode].plot(grid.zg, system.result["F"].imag)
    axes[0, mode].set_xlim(-4, 4)
    axes[1, mode].set_xlabel(r'$z$')

    G = grid.der(system.result['F'])/K
    axes[1, mode].plot(grid.zg, G.real)
    axes[1, mode].plot(grid.zg, G.imag)

axes[0, 0].set_ylabel(r'$F(z)$')
axes[1, 0].set_ylabel(r'$G(z)$')
plt.show()
