import numpy as np
from freja import Solver, ChebyshevRationalGrid
from freja.systems.kh_hydro import KelvinHelmholtzHydroOnly
import pickle

N = 88

grid = ChebyshevRationalGrid(N, L=0.2)

u0 = 1.
delta = 0.0

system = KelvinHelmholtzHydroOnly(grid, u0, delta, z1=-0.5, z2=0.5)
system.boundaries = [True, True, True, True]

kx = 2.
kh = Solver(grid, system, kx)

omega, v = kh.solver()
result = {var: v[j*grid.NN: (j+1)*grid.NN] for j,
          var in enumerate(system.variables)}
result.update({'omega': omega, 'kx': kx, 'zg': grid.zg,
               'variables': system.variables, 'mode': 0})

# Save with numpy npz
np.savez('test.npz', **result)
sol = np.load('test.npz')

# Save with pickle (better!)
pickle.dump(result, open('test.p', 'wb'))
sol = pickle.load(open('test.p', 'rb'))

pickle.dump(system, open('system.p', 'wb'))
pickle.dump(grid, open('grid.p', 'wb'))


def plot_solution(sol, filename=None, n=1, smooth=True):
    from freja import setup
    pylab = setup('ps')
    import matplotlib.pyplot as plt

    title = r'$\omega = {:1.2f}, k_x = {:1.2f}, m={}$'
    plt.figure(n)
    plt.clf()
    fig, axes = plt.subplots(num=n, nrows=system.dim, sharex=True)
    for j, var in enumerate(sol['variables']):
        if smooth:
            z = np.linspace(grid.zmin, grid.zmax, 2000)
            axes[j].plot(z, grid.interpolate(z, sol[var].real),
                         'C0', label='Real')
            axes[j].plot(z, grid.interpolate(z, sol[var].imag),
                         'C1', label='Imag')
        else:
            axes[j].plot(sol['zg'], sol[var].real, 'C0+', label='Real')
            axes[j].plot(sol['zg'], sol[var].imag, 'C1+', label='Imag')
        axes[j].set_ylabel(system.labels[j])
    axes[system.dim-1].set_xlabel(r"$z$")
    axes[0].set_title(title.format(sol['omega'], sol['kx'], sol['m']))
    axes[0].legend(frameon=False)

    if not pylab and filename is not None:
        fig.savefig('../figures/' + filename + '.eps')
    else:
        plt.show()


# plot_solution(result)
plot_solution(sol, filename=None, n=1, smooth=False)
