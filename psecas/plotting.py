def plot_solution(system, filename=None, num=1, smooth=True, limits=None):
    """Quickly plot the 1D eigenmodes stored in the system object"""
    import numpy as np
    import matplotlib.pyplot as plt

    sol = system.result
    grid = system.grid

    title = r'$\omega = {:1.4f}, k_x = {:1.2f}, m={}$'
    plt.figure(num)
    plt.clf()
    fig, axes = plt.subplots(num=num, nrows=system.dim, sharex=True)
    for j, var in enumerate(system.variables):
        if smooth:
            if limits is None:
                z = np.linspace(grid.zmin, grid.zmax, 2000)
            else:
                z = np.linspace(limits[0], limits[1], 2000)
            axes[j].plot(
                z, grid.interpolate(z, sol[var].real), 'C0', label='Real'
            )
            axes[j].plot(
                z, grid.interpolate(z, sol[var].imag), 'C1', label='Imag'
            )
        axes[j].plot(grid.zg, sol[var].real, 'C0.', label='Real')
        axes[j].plot(grid.zg, sol[var].imag, 'C1.', label='Imag')
        axes[j].set_ylabel(system.labels[j])
    axes[system.dim - 1].set_xlabel(r"$z$")
    axes[0].set_title(
        title.format(sol[system.eigenvalue], system.kx, sol['mode'])
    )
    axes[0].legend(frameon=False)

    if filename is not None:
        fig.savefig(filename)
    else:
        plt.show()


def get_2Dmap(system, var, xmin, xmax, Nx, Nz, zmin=None, zmax=None, time=0):
    """Create a 2D map of the eigenmode var stored in system.result[var].
       This function assumes that the eigenmodes have the form
       f(z)*exp(i kx x).
    """
    import numpy as np

    dx = (xmax - xmin) / Nx
    xg = (0.5 + np.arange(Nx)) * dx

    if zmin is None or zmax is None:
        zmin = system.grid.zmin
        zmax = system.grid.zmax
    dz = (zmax - zmin) / Nz
    zg = (0.5 + np.arange(Nz)) * dz + zmin
    xx, zz = np.meshgrid(xg, zg)

    # Wavenumber
    kx = system.kx

    val = np.zeros((Nz, Nx))

    def return_real_ampl(f, x):
        """Hardcode to the sigma notation..."""
        return (
            2*f*np.exp(1j*kx*x + system.result[system.eigenvalue]*time)
        ).real

    # Interpolate onto z-grid
    if type(var) is str:
        yr = system.grid.interpolate(zg, system.result[var].real)
        yi = system.grid.interpolate(zg, system.result[var].imag)
    else:
        yr = system.grid.interpolate(zg, var.real)
        yi = system.grid.interpolate(zg, var.imag)
    y = yr + 1j * yi
    for i in range(Nx):
        val[:, i] = return_real_ampl(y, xg[i])

    return val
