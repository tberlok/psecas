import numpy as np
from evp import Solver, FourierGrid
from evp.systems.kh_uniform import KelvinHelmholtzUniform

N = 64
zmin = 0
zmax = 2
grid = FourierGrid(N, zmin, zmax)

beta = 1e4
nu = 1e-2

system = KelvinHelmholtzUniform(grid, beta, nu)

kx = 3.52615254237
kh = Solver(grid, system, kx)

omega, v = kh.solver()
result = {var:v[j*grid.NN:(j+1)*grid.NN] for j, 
          var in enumerate(system.variables)}
result.update({'omega':omega, 'kx':kx, 'zg':grid.zg, 
               'variables':system.variables, 'm':0})

result.update({'dbx':-np.matmul(grid.d1, result['dA']), 
               'dbz':1j*kx*result['dA']})

# Save with numpy npz
np.savez('test.npz', **result)
sol = np.load('test.npz')

# Save with pickle (better!)
import pickle
pickle.dump(result, open('test.p', 'wb'))
sol = pickle.load(open('test.p', 'rb'))

pickle.dump(system, open('system.p', 'wb'))
pickle.dump(grid, open('grid.p', 'wb'))

def plot_solution(sol, filename=None, n=1, smooth=True):
    from evp import setup
    pylab = setup('ps')
    import matplotlib.pyplot as plt

    title = r'$\omega = {:1.2f}, k_x = {:1.2f}, m={}$'
    z = np.linspace(grid.zmin, grid.zmax, 2000)
    plt.figure(n)
    plt.clf()
    fig, axes = plt.subplots(num=n, nrows=system.dim, sharex=True)
    for j, var in enumerate(sol['variables']):
        if smooth:
            axes[j].plot(z, grid.interpolate(z, sol[var].real), 'C0', label='Real')
            axes[j].plot(z, grid.interpolate(z, sol[var].imag), 'C1', label='Imag')
        else:
            axes[j].plot(sol['zg'], sol[var].real, 'C0+', label='Real')
            axes[j].plot(sol['zg'], sol[var].imag, 'C1+', label='Imag')
        axes[j].set_ylabel(system.labels[j])
    axes[system.dim-1].set_xlabel(r"$z$")
    axes[0].set_title(title.format(sol['omega'], sol['kx'], sol['m']))
    axes[0].legend(frameon=False)

    if not pylab and filename is not None:
        fig.savefig('../figures/' + filename +'.eps')
    else:
        plt.show()


# plot_solution(result)
plot_solution(sol, filename='test')

f = result['dA'].real
kxs = 2*np.pi*np.fft.rfftfreq(grid.N)*grid.N/grid.L
f_hat = np.fft.rfft(f)
dbx = np.fft.irfft(-1j*kxs*f_hat)

import matplotlib.pyplot as plt
plt.figure(2)
plt.clf()
plt.plot(grid.zg, result['dbx'].real, '+')
plt.plot(grid.zg, result['dbx'].real, '--')
z = np.linspace(grid.zmin, grid.zmax, 4000)
plt.plot(z, grid.interpolate(z, result['dbx'].real))
plt.show()

# import time

# omega_vec = []
# kx_vec = np.arange(1, 5)
# for kx in kx_vec:
#     t1 = time.time()
#     kh.kx = kx
#     (omega, v, err) = kh.iterate_solver(tol=1e-4)
#     omega_vec.append(omega)
#     print(kx, omega, err)
#     t2 = time.time()
#     # print("Solver took {} seconds".format(t2-t1))
# omega_vec = np.array(omega_vec)