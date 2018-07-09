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


def write_athena(system, Nz, Lz, path=None):
    """
    Interpolate theory onto grid in Athena
    """
    import numpy as np

    # Grid points where Athena is defined (improve this!)
    dz = Lz/Nz
    z = np.arange(dz/2, Nz*dz, dz)
    znodes = np.arange(0., (Nz+1)*dz, dz)

    grid = system.grid
    result = system.result

    if path is None:
        path = './'

    # Calculate and store imaginary part
    perturb = []
    for key in system.variables:
        if key != 'dA':
            y = np.hstack([grid.interpolate(z, result[key].imag), 0.0])
            perturb.append(y)

    if 'dA' in system.variables:
        znodes = np.arange(0., (Nz+1)*dz, dz)
        perturb.append(grid.interpolate(znodes, result['dA'].imag))

    perturb = np.transpose(perturb)
    np.savetxt(path + 'imagPerturbations{}.txt'.format(Nz), perturb,
               delimiter="\t", newline="\n", fmt="%1.16e")

    # Calculate and store real part
    perturb = []
    for key in system.variables:
        if key != 'dA':
            y = np.hstack([grid.interpolate(z, result[key].real), 0.0])
            perturb.append(y)

    if 'dA' in system.variables:
        perturb.append(grid.interpolate(znodes, result['dA'].real))

    perturb = np.transpose(perturb)
    np.savetxt(path + 'realPerturbations{}.txt'.format(Nz), perturb,
               delimiter="\t", newline="\n", fmt="%1.16e")
