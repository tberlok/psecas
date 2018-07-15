import numpy as np
from evp import Solver, ChebyshevExtremaGrid
from evp.systems.mti import MagnetoThermalInstability
from evp import plot_solution

N = 64
zmin = 0
zmax = 1
grid = ChebyshevExtremaGrid(N, zmin, zmax)

beta = 1e6
Kn0 = 2000
kx = 2*np.pi
# kxmax for beta = 1e5 and Kn0 = 200
# kx = 28.42043898247281
# kxmax for beta = 1e3 and Kn0 = 2000
# kx = 7.982608527915205

system = MagnetoThermalInstability(grid, beta, Kn0, kx, only_interior=True)
system.boundaries = [False, True, False, True, True]

# No viscosity for now
# system.nu0 = 0
# system.make_background()

solver = Solver(grid, system)

mode = 0
Ns = np.hstack((np.arange(2, 5)*16, np.arange(3, 12)*32))
omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=True, tol=1e-5)
phi = np.arctan(vec[2].imag/vec[2].real)
solver.keep_result(omega, vec*np.exp(-1j*phi), mode=mode)

# Normalize eigenmodes
y = np.vstack([system.result['dvx'].real,
               system.result['dvx'].imag,
               system.result['dvz'].real,
               system.result['dvz'].imag])

val = np.max(np.abs(y))
for key in system.variables:
    system.result[key] /= val

plot_solution(system, smooth=True)

if True:
    # Plot 2D maps of the perturbations
    import matplotlib.pyplot as plt
    from evp import get_2Dmap
    plt.rc('image', origin='lower', cmap='RdBu')
    plt.figure(2)
    plt.clf()
    fig, axes = plt.subplots(num=2, sharex=True, sharey=True, nrows=2, ncols=3)
    xmin = 0
    xmax = 2*np.pi/kx
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
    fig, axes = plt.subplots(num=3, ncols=3, sharey=True)
    s = system

    pa = s.rho*s.nu*(2*1j*s.kx*s.result['dvx'] - s.grid.dz(s.result['dvz']))
    pa = get_2Dmap(system, pa, xmin, xmax, Nx, Nz)
    axes[0].imshow(pa, extent=extent)
    axes[0].set_title(r'$\Delta p$')

    # Construct total vector potential
    dA = get_2Dmap(system, 'dA', xmin, xmax, Nx, Nz)
    dx = (xmax-xmin)/Nx
    xg = (0.5 + np.arange(Nx))*dx
    dz = (system.grid.zmax-system.grid.zmin)/Nz
    zg = (0.5 + np.arange(Nz))*dz
    xx, zz = np.meshgrid(xg, zg)
    ampl = 1e-2
    A = ampl*dA - np.tile(zg, (Nx, 1)).T
    axes[0].contour(xx, zz, A, 32, colors='tab:gray',
                 linestyles='solid')
    t = 1
    sigma = s.result['omega'].real
    B2 = ((s.B0 + s.B0*dbx*np.exp(sigma*t))**2 + (s.B0*dbz*np.exp(sigma*t))**2)
    axes[1].imshow(B2, extent=extent)
    axes[1].set_title(r'$B^2$')

    axes[2].imshow(2*pa*np.exp(sigma*t)/B2, extent=extent)
    axes[2].set_title(r'$2\Delta p/B^2$')
    plt.show()

if False:
    # Both methods now give similar results. Beware the eigenvalues can swap in
    # phase, e.g. \pm between calls to the solver.
   from numpy.fft import rfftfreq, fftfreq

   sign = -1

   def rfft2 (f, sign):
       """Real-to-complex Fourier transform for fields that are periodic in x and
       either symmetric (sign = +1) or anti-symmetric (sign = -1) about the
       y-boundaries"""
       from numpy import vstack
       from numpy.fft import rfft2
       return rfft2 (vstack ((f, sign*f[::-1,:])))

   def irfft2 (f):
       "Inverse transform"
       from numpy.fft import irfft2
       return irfft2 (f)[:f.shape[0]//2,:]

   Bx = 1.0 + ampl*dbx
   By = ampl*dbz

   Bx = Bx
   By = By
   # Wave numbers
   kx, ky = np.meshgrid (2*np.pi*rfftfreq (Nx)/dx, 2*np.pi*fftfreq (2*Nz)/dz)
   k2 = kx*kx + ky*ky
   k2[0,0] = 1.0
   k21 = 1.0/k2
   k21[0,0] = 0.0

   # Fourier transformed magnetic field
   Bx_hat = rfft2 (Bx, +1*sign)
   By_hat = rfft2 (By, -1*sign)

   # Compute vector potential in Coulom gauge.
   # Note that Jz = ∂By/∂x - ∂Bz/∂y = -∂²Az
   Az = irfft2 (1j*k21*(kx*By_hat - ky*Bx_hat))
   # Add contribution from mean field
   Az -= By.mean ()*xx

   axes[0, 1].contour(xx, zz, Az, 32, colors='tab:gray',
                 linestyles='solid')

if False:
    from evp import write_athena, save_system
    # Write directly to the Athena directory
    write_athena(system, Nz=256, Lz=1.0,
                 path='/Users/berlok/codes/athena/bin/')
    save_system(system, '/Users/berlok/codes/athena/bin/mti-evp.p')

if False:
    import matplotlib.pyplot as plt
    steps = 20
    grid.N = 32
    kx_vec = 2*np.pi*np.linspace(0.01, 20, steps)
    omega_vec = []
    for kx in kx_vec:
        system.kx = kx
        (omega, v) = solver.solve()
        omega_vec.append(omega)
        print(kx, omega)
    omega_vec = np.array(omega_vec)
    plt.figure(4)
    plt.plot(kx_vec, omega_vec)
    plt.show()

if False:
    from evp import golden_section

    def f(kx, **kwargs):

        # system = MagnetoThermalInstability(grid, beta, Kn0, kx, only_interior=True)
        # system.boundaries = [False, True, False, True, True]

        # solver = Solver(grid, system)
        system.kx = kx
        # mode = 0
        # Ns = np.hstack((np.arange(1, 5)*32, np.arange(3, 12)*64))
        omega, vec, err = solver.iterate_solver(Ns, mode=mode, verbose=False, tol=1e-4)

        return -omega.real

    (a, b) = golden_section(f, 18, 30, tol=1e-3)
    print(a, b, (a+b)/2, -f((a+b)/2))
    kxmax = (a+b)/2
    Lx = 2*np.pi/kxmax
