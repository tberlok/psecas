def plot_solution(system, filename=None, num=1, smooth=True, limits=None):
    import numpy as np
    from evp import setup
    pylab = setup('ps')
    import matplotlib.pyplot as plt

    sol = system.result
    grid = system.grid

    title = r'$\omega = {:1.2f}, k_x = {:1.2f}, m={}$'
    plt.figure(num)
    plt.clf()
    fig, axes = plt.subplots(num=num, nrows=system.dim, sharex=True)
    for j, var in enumerate(system.variables):
        if smooth:
            if limits is None:
                z = np.linspace(grid.zmin, grid.zmax, 2000)
            else:
                z = np.linspace(limits[0], limits[1], 2000)
            axes[j].plot(z, grid.interpolate(z, sol[var].real),
                         'C0', label='Real')
            axes[j].plot(z, grid.interpolate(z, sol[var].imag),
                         'C1', label='Imag')
        axes[j].plot(grid.zg, sol[var].real, 'C0.', label='Real')
        axes[j].plot(grid.zg, sol[var].imag, 'C1.', label='Imag')
        axes[j].set_ylabel(system.labels[j])
    axes[system.dim-1].set_xlabel(r"$z$")
    axes[0].set_title(title.format(sol['omega'], system.kx, sol['mode']))
    axes[0].legend(frameon=False)

    if not pylab and filename is not None:
        fig.savefig('../figures/' + filename + '.eps')
    else:
        plt.show()


def load_system(filename):
    import pickle
    system = pickle.load(open(filename, 'rb'))
    return system
