import numpy as np
from psecas import Solver, ChebyshevExtremaGrid
from psecas.systems.mti import MagnetoThermalInstability
from psecas import plot_solution
import matplotlib.pyplot as plt
from psecas import get_2Dmap

"""
    The linear solution for the magnetothermal instability (MTI)
    in a quasi-global setup, i.e. periodic in x and non-periodic in z.

    Linearized equations for the MTI with anisotropic viscosity and heat
    conduction for a constant magnetic field in the x-direction.

    See the following paper for more details:

    Suppressed heat conductivity in the intracluster medium:
    implications for the magneto-thermal instability,
    Thomas Berlok, Eliot Quataert, Martin E. Pessah, Christoph Pfrommer
    https://arxiv.org/abs/2007.00018
"""

N = 64
zmin = 0
zmax = 1
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e6
Kn0 = 2000
kx = 2 * np.pi * 10
# kxmax for beta = 1e5 and Kn0 = 200
# kx = 28.42043898247281
# kxmax for beta = 1e3 and Kn0 = 2000
# kx = 7.982608527915205

system = MagnetoThermalInstability(grid, beta, Kn0, kx)
system.boundaries = [False, True, False, True, True]

# No viscosity for now
# system.nu0 = 0
# system.make_background()

solver = Solver(grid, system)

plt.rc('image', origin='lower', aspect='equal', interpolation='nearest',
       cmap='hot')

plt.rc ('font', family = 'serif', serif = 'cm')
plt.rc ('text', usetex = True)
plt.rc ('text.latex', preamble = [
  r'\usepackage[T1]{fontenc}',
  r'\usepackage[english]{babel}',
  r'\usepackage[utf8]{inputenc}',
  r'\usepackage{lmodern}',
  r'\usepackage{microtype}',
  r'\usepackage{amsmath}',
  r'\usepackage{bm}'])

dpi = plt.rcParams['figure.dpi']
textwidth = 510.0
width = textwidth/dpi
height = width*1
plt.rc('figure', figsize=(width, height))

plt.rc('image', origin='lower', cmap='RdBu')
plt.figure(20)
plt.clf()
fig, axes = plt.subplots(
    num=20, sharex=True, sharey=True, nrows=4, ncols=4
)

Ns = np.hstack((np.arange(2, 5) * 16, np.arange(3, 12) * 32))
# for nx in range(1, 21):
for jj, nx in enumerate([1, 5, 10, 15]):
    solver.system.kx = nx * 2 * np.pi
    omega, vec = solver.solve(mode=0, verbose=True, saveall=True)
    axes[0, jj].set_title(r'$n={}$'.format(nx))
    for ii, mode in enumerate([0, 1, 2, 8]):
    # for mode in range(0, 10):
        # omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
        vec = solver.v[:, mode]
        phi = np.arctan(vec[2].imag / vec[2].real)
        solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)

        # Normalize eigenmodes
        y = np.vstack(
            [
                system.result['dvx'].real,
                system.result['dvx'].imag,
                system.result['dvz'].real,
                system.result['dvz'].imag,
            ]
        )

        val = np.max(np.abs(y))
        for key in system.variables:
            system.result[key] /= val

        # plot_solution(system, filename='mti_beta1e6_Kni2000_nx_{}_mode_{}.pdf'.format(nx, mode), smooth=True)

        xmin = 0
        xmax = 1 # 2 * np.pi / kx
        Nx = 512
        Nz = 512
        extent = [xmin, xmax, system.grid.zmin, system.grid.zmax]

        dvz = get_2Dmap(system, 'dvz', xmin, xmax, Nx, Nz)

        axes[ii, jj].imshow(dvz, extent=extent)

for jj in range(4):
    for ii in range(4):
        axes[ii, jj].set_xticklabels(("", "","","","",""))
        axes[ii, jj].set_yticklabels(("", "","","","",""))
        axes[ii, jj].tick_params(axis='x', which='both', bottom=False, top=False, labelbottom=False)
        axes[ii, jj].tick_params(axis='y', which='both', bottom=False, top=False, labelbottom=False)
        axes[ii, jj].tick_params(direction='in')
        # axes[ii, jj].yaxis.set_visible(False)
        # axes[ii, jj].set_xlabel(r'$x/H$')

for ii, mode in enumerate([0, 1, 2, 8]):
    axes[ii, 0].set_ylabel(r'$m={}$'.format(mode))

plt.tight_layout(h_pad=0.05, w_pad=0.05)
plt.savefig('vz_modes.pdf')
# plt.show()

if False:
    # Plot 2D maps of the perturbations
    import matplotlib.pyplot as plt
    from psecas import get_2Dmap

    plt.rc('image', origin='lower', cmap='RdBu')
    plt.figure(2)
    plt.clf()
    fig, axes = plt.subplots(
        num=2, sharex=True, sharey=True, nrows=2, ncols=3
    )
    xmin = 0
    xmax = 1 # 2 * np.pi / kx
    Nx = 512
    Nz = 512
    extent = [xmin, xmax, system.grid.zmin, system.grid.zmax]

    dvx = get_2Dmap(system, 'dvx', xmin, xmax, Nx, Nz)
    dvz = get_2Dmap(system, 'dvz', xmin, xmax, Nx, Nz)
    dT = get_2Dmap(system, 'dT', xmin, xmax, Nx, Nz)
    drho = get_2Dmap(system, 'drho', xmin, xmax, Nx, Nz)

    system.get_bx_and_by()

    dbx = get_2Dmap(system, 'dbx', xmin, xmax, Nx, Nz)
    dbz = get_2Dmap(system, 'dbz', xmin, xmax, Nx, Nz)

    axes[0, 0].imshow(dvx, extent=extent)
    axes[0, 0].set_title(r'$\delta v_x$')
    axes[0, 1].imshow(dvz, extent=extent)
    axes[0, 1].set_title(r'$\delta v_z$')
    axes[0, 2].imshow(drho, extent=extent)
    axes[0, 2].set_title(r'$\delta \rho/\rho$')
    axes[1, 0].imshow(dT, extent=extent)
    axes[1, 0].set_title(r'$\delta T/T$')
    axes[1, 1].imshow(dbx, extent=extent)
    axes[1, 1].set_title(r'$\delta b_x$')
    axes[1, 2].imshow(dbz, extent=extent)
    axes[1, 2].set_title(r'$\delta b_z$')
    plt.show()

    plt.figure(3)
    plt.clf()
    fig, axes = plt.subplots(num=3, ncols=3, sharey=True, constrained_layout=True)
    s = system

    bbdv = 1j * s.kx * s.result['dvx']
    divV = 1j * s.kx * s.result['dvx'] + s.grid.der(s.result['dvz'])
    pa = s.rho * s.nu * (3*bbdv - divV)

    pa = get_2Dmap(system, pa, xmin, xmax, Nx, Nz)
    im = axes[0].imshow(pa, extent=extent)
    axes[0].set_title(r'$\Delta p$')
    fig.colorbar(im, ax=axes[0], shrink=0.4)

    bbdv = get_2Dmap(system, bbdv, xmin, xmax, Nx, Nz)
    im = axes[1].imshow(bbdv, extent=extent)
    axes[1].set_title(r'$bb:\nabla v$')
    fig.colorbar(im, ax=axes[1], shrink=0.4)

    divV = get_2Dmap(system, divV, xmin, xmax, Nx, Nz)
    im = axes[2].imshow(divV, extent=extent)
    axes[2].set_title(r'$\nabla \cdot v$')
    fig.colorbar(im, ax=axes[2], shrink=0.4)

    # Construct total vector potential
    dA = get_2Dmap(system, 'dA', xmin, xmax, Nx, Nz)
    dx = (xmax - xmin) / Nx
    xg = (0.5 + np.arange(Nx)) * dx
    dz = (system.grid.zmax - system.grid.zmin) / Nz
    zg = (0.5 + np.arange(Nz)) * dz
    xx, zz = np.meshgrid(xg, zg)
    ampl = 1e-2
    A = ampl * dA - np.tile(zg, (Nx, 1)).T
    axes[0].contour(xx, zz, A, 32, colors='tab:gray', linestyles='solid')
    t = 1
    sigma = s.result['sigma'].real
    B2 = (s.B0 + s.B0 * dbx * np.exp(sigma * t)) ** 2 + (
        s.B0 * dbz * np.exp(sigma * t)
    ) ** 2
    # axes[1].imshow(B2, extent=extent)
    # axes[1].set_title(r'$B^2$')

    # axes[2].imshow(2 * pa * np.exp(sigma * t) / B2, extent=extent)
    # axes[2].set_title(r'$2\Delta p/B^2$')
    plt.show()

if False:
    # Both methods now give similar results. Beware the eigenvalues can swap in
    # phase, e.g. \pm between calls to the solver.
    from numpy.fft import rfftfreq, fftfreq

    sign = -1

    def rfft2(f, sign):
        """Real-to-complex Fourier transform for fields that are periodic in x
        and either symmetric (sign = +1) or anti-symmetric (sign = -1) about
        the y-boundaries"""
        from numpy import vstack
        from numpy.fft import rfft2

        return rfft2(vstack((f, sign * f[::-1, :])))

    def irfft2(f):
        "Inverse transform"
        from numpy.fft import irfft2

        return irfft2(f)[: f.shape[0] // 2, :]

    Bx = 1.0 + ampl * dbx
    By = ampl * dbz

    Bx = Bx
    By = By
    # Wave numbers
    kx, ky = np.meshgrid(
        2 * np.pi * rfftfreq(Nx) / dx, 2 * np.pi * fftfreq(2 * Nz) / dz
    )
    k2 = kx * kx + ky * ky
    k2[0, 0] = 1.0
    k21 = 1.0 / k2
    k21[0, 0] = 0.0

    # Fourier transformed magnetic field
    Bx_hat = rfft2(Bx, +1 * sign)
    By_hat = rfft2(By, -1 * sign)

    # Compute vector potential in Coulom gauge.
    # Note that Jz = ∂By/∂x - ∂Bz/∂y = -∂²Az
    Az = irfft2(1j * k21 * (kx * By_hat - ky * Bx_hat))
    # Add contribution from mean field
    Az -= By.mean() * xx

    axes[0, 1].contour(xx, zz, Az, 32, colors='tab:gray', linestyles='solid')

if False:
    from psecas import write_athena, save_system

    # Write directly to the Athena directory
    write_athena(
        system, Nz=256, Lz=1.0, path='/Users/berlok/codes/athena/bin/',
        name='MTIbeta1e5_n7_'
    )
    save_system(system, '/Users/berlok/codes/athena/bin/mti-evp.p')

if True:
    import matplotlib.pyplot as plt

    steps = 200
    modes = 10
    grid.N = 64
    kx_vec = 2 * np.pi * np.linspace(0.5, 30, steps)
    omega_vec = []
    bbdv_vec = np.zeros((steps, 10))
    divV_vec = np.zeros((steps, 10))
    for ii, kx in enumerate(kx_vec):
        system.kx = kx
        omega, vec = solver.solve(mode=mode, verbose=True, saveall=True)
        omega_vec.append(solver.E[:modes])
        print(kx/(2*np.pi), omega)
        for mode in range(modes):
        # for mode in range(0, modes):
            # omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
            vec = solver.v[:, mode]
            phi = np.arctan(vec[2].imag / vec[2].real)
            solver.keep_result(omega, vec * np.exp(-1j * phi), mode=mode)
            s = system
            bbdv = 1j * s.kx * s.result['dvx']
            divV = 1j * s.kx * s.result['dvx'] + s.grid.der(s.result['dvz'])
            bbdv_vec[ii, mode] = np.max(np.abs(bbdv))
            divV_vec[ii, mode] = np.max(np.abs(divV))
    omega_vec = np.array(omega_vec)

    plt.figure(4)
    plt.clf()
    for m in range(modes):
        plt.plot(kx_vec/(2*np.pi), omega_vec[:, m].real, label=r'$m={}$'.format(m))
    plt.legend(frameon=False, ncol=2)
    plt.xlabel(r'$k_\parallel H/2\pi$')
    plt.ylabel(r'$\sigma/\omega_\mathrm{dyn}$')
    # plt.show()


    plt.figure(5)
    plt.clf()
    for m in range(modes):
        plt.semilogy(kx_vec/(2*np.pi), divV_vec[:, m]/bbdv_vec[:, m], label=r'$m={}$'.format(m))
    # plt.xlim(0.5, 30)
    plt.legend(frameon=False, ncol=2)
    plt.xlabel(r'$k_\parallel H/2\pi$')
    plt.ylabel(r'$\mathrm{max}(\nabla \cdot v)/\mathrm{max}(b b \mathrm{:}\nabla v)$')
    plt.savefig('compressibility.pdf')

if False:
    from psecas import golden_section

    def f(kx, **kwargs):
        # system = MagnetoThermalInstability(grid, beta, Kn0, kx)
        # print(system.kx)
        # system.boundaries = [False, True, False, True, True]
        # omega, vec, err = solver.iterate_solver(
        #     Ns, mode=mode, verbose=False, tol=1e-4, verbose=True
        # )
        solver.system.kx = kx
        omega, vec = solver.solve(mode=0, verbose=True)
        return -omega.real

    (a, b) = golden_section(f, 1*np.pi, 2*np.pi*20, tol=1e-3)
    print(a, b, (a + b) / 2, -f((a + b) / 2))
    kxmax = (a + b) / 2
    Lx = 2 * np.pi / kxmax
    plt.figure(4)
    plt.plot(kxmax/(2*np.pi), -f((a + b) / 2), 'x')
    plt.savefig('quasi-global_growth_rates.pdf')
