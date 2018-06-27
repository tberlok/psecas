import numpy as np
import matplotlib.pyplot as plt
from evp import Solver, ChebyshevRationalGrid
from evp.systems.channel import Channel

grid = ChebyshevRationalGrid(N=199, L=1)
system = Channel(grid)

# kx is weird to have as a parameter here TODO: fix that
ch = Solver(grid, system, kx=0)

def sorting_strategy(E):
    """Sorting strategy for channel modes"""
    E[E.real > 100.] = 0
    E[E.real < -10.] = 0
    return E

ch.sorting_strategy = sorting_strategy

# omega, vec = ch.solver(verbose=True)

plt.figure(1)
plt.clf()
modes = 3
fig, axes = plt.subplots(num=1, ncols=modes, sharey=True)
for mode in range(modes):
    omega, vec = ch.solver(mode=mode, verbose=True)
    axes[mode].set_title(r"$\sigma = ${:1.4f}".format(omega.real), fontsize=10)
    axes[mode].plot(grid.zg, ch.result['f'].real)
    axes[mode].plot(grid.zg, ch.result['f'].imag)
    axes[mode].set_xlim(-4, 4)
plt.show()
