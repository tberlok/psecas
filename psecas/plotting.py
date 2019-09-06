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
            # var_interp = grid.interpolate(z, sol[var])
            # axes[j].plot(
            #     z, var_interp.real, 'C0', label='Real'
            # )
            # axes[j].plot(
            #     z, var_interp.imag, 'C1', label='Imag'
            # )
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
    try:
        axes[0].set_title(
            title.format(sol[system.eigenvalue], system.kx, sol['mode'])
        )
    except:
        axes[0].set_title(
            r'$\omega$ = {:1.6f}'.format(sol[system.eigenvalue])
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
        """"""
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


def get_2D_cylindrical_map_in_cylindrical_coords(
    system, var, phimin, phimax, Nphi, Nr, rmin=None, rmax=None, time=0, z=0
):
    """Create a 2D map of the eigenmode var stored in system.result[var].
       This function assumes that the eigenmodes have the form
       f(r)*exp(i kz z + i m phi). It returns a map in the r-phi plane at
       a fixed value of z (default 0)
    """
    import numpy as np

    # Create linear grid in phi
    dphi = (phimax - phimin) / (Nphi - 1)
    phig = (np.arange(Nphi)) * dphi

    # Create linear grid in r
    if rmin is None:
        rmin = system.grid.zmin
    dr = (rmax - rmin) / Nr
    rg = (0.5 + np.arange(Nr)) * dr + rmin

    # Contruct meshgrids
    rr, phiphi = np.meshgrid(rg, phig)

    # Azimuthal mode number
    m = system.m

    # Wavenumber
    kz = system.kz

    val = np.zeros((Nphi, Nr))

    def return_real_ampl(f, phi, z):
        """"""
        return (
            2
            * f
            * np.exp(
                1j*kz*z + 1j*m*phi + system.result[system.eigenvalue]*time
            )
        ).real

    # Interpolate onto r-grid
    if type(var) is str:
        yr = system.grid.interpolate(rg, system.result[var].real)
        yi = system.grid.interpolate(rg, system.result[var].imag)
    else:
        yr = system.grid.interpolate(rg, var.real)
        yi = system.grid.interpolate(rg, var.imag)
    y = yr + 1j * yi
    for i in range(Nphi):
        val[i, :] = return_real_ampl(y, phig[i], z)

    # This is how you would plot the map
    # xx = rr * np.cos(phiphi)
    # yy = rr * np.sin(phiphi)

    # plt.pcolormesh(xx, yy, val)
    # plt.axis('equal')
    # plt.show()

    return (rr, phiphi, val)


def get_2D_cylindrical_map(
    system, var, xmin, xmax, ymin, ymax, Nx, Ny, time=0, z=0
):
    import numpy as np
    x = np.linspace(xmin, xmax, Nx)
    y = np.linspace(ymin, ymax, Ny)
    xx, yy = np.meshgrid(x, y)
    rr = np.sqrt(xx**2 + yy**2)
    phiphi = np.arctan2(yy, xx)

    # Azimuthal mode number
    m = system.m

    # Wavenumber
    kz = system.kz

    # Interpolate onto r-grid
    rg = rr.flatten()
    if type(var) is str:
        yr = system.grid.interpolate(rg, system.result[var].real)
        yi = system.grid.interpolate(rg, system.result[var].imag)
    else:
        yr = system.grid.interpolate(rg, var.real)
        yi = system.grid.interpolate(rg, var.imag)
    y = yr + 1j * yi

    val = np.resize(y, (Nx, Ny))
    val = (2*val*np.exp(1j*kz*z + 1j*m*phiphi + system.result[system.eigenvalue]*time)).real
    # for i in range(Nphi):
    #     val[i, :] = return_real_ampl(y, phig[i])

    return (xx, yy, val)
