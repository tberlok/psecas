def plot_solution(solver, filename=None, n=1, smooth=True, limits=None):
    import numpy as np
    from evp import setup
    pylab = setup('ps')
    import matplotlib.pyplot as plt

    sol = solver.result
    grid = solver.grid
    system = solver.system

    title = r'$\omega = {:1.2f}, k_x = {:1.2f}, m={}$'
    plt.figure(n)
    plt.clf()
    fig, axes = plt.subplots(num=n, nrows=system.dim, sharex=True)
    for j, var in enumerate(sol['variables']):
        if smooth:
            if limits is None:
                z = np.linspace(grid.zmin, grid.zmax, 2000)
            else:
                z = np.linspace(limits[0], limits[1], 2000)
            axes[j].plot(z, grid.interpolate(z, sol[var].real),
                         'C0', label='Real')
            axes[j].plot(z, grid.interpolate(z, sol[var].imag),
                         'C1', label='Imag')
        axes[j].plot(sol['zg'], sol[var].real, 'C0.', label='Real')
        axes[j].plot(sol['zg'], sol[var].imag, 'C1.', label='Imag')
        axes[j].set_ylabel(system.labels[j])
    axes[system.dim-1].set_xlabel(r"$z$")
    axes[0].set_title(title.format(sol['omega'], sol['kx'], sol['mode']))
    axes[0].legend(frameon=False)

    if not pylab and filename is not None:
        fig.savefig('../figures/' + filename + '.eps')
    else:
        plt.show()
