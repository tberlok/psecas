def load_system(filename):
    """Load object containing solution.
    Input: filename, eg 'system.p'
    Output: system object
    """
    import pickle

    system = pickle.load(open(filename, 'rb'))
    return system


def save_system(system, filename):
    import pickle

    # Delete d0, d1 and d2 for storage effieciency
    del system.grid.d0
    del system.grid.d1
    del system.grid.d2
    pickle.dump(system, open(filename, 'wb'))


def write_athena(system, Nz, Lz, path=None, name=None):
    """
    Interpolate theory onto grid in Athena
    """
    import numpy as np

    # Grid points where Athena is defined (improve this!)
    dz = Lz / Nz
    z = np.arange(dz / 2, Nz * dz, dz)
    znodes = np.arange(0.0, (Nz + 1) * dz, dz)

    grid = system.grid
    result = system.result

    if path is None:
        path = './'

    if name is None:
        name = 'Pertubations'

    # Calculate and store imaginary part
    perturb = []
    for key in system.variables:
        if key != 'dA':
            y = np.hstack([grid.interpolate(z, result[key].imag), 0.0])
            perturb.append(y)

    if 'dA' in system.variables:
        znodes = np.arange(0.0, (Nz + 1) * dz, dz)
        perturb.append(grid.interpolate(znodes, result['dA'].imag))
    else:
        perturb.append(np.zeros_like(znodes))

    perturb = np.transpose(perturb)
    np.savetxt(
        path + 'imag' + name + '{}.txt'.format(Nz),
        perturb,
        delimiter="\t",
        newline="\n",
        fmt="%1.16e",
    )

    # Calculate and store real part
    perturb = []
    for key in system.variables:
        if key != 'dA':
            y = np.hstack([grid.interpolate(z, result[key].real), 0.0])
            perturb.append(y)

    if 'dA' in system.variables:
        perturb.append(grid.interpolate(znodes, result['dA'].real))
    else:
        perturb.append(np.zeros_like(znodes))

    perturb = np.transpose(perturb)
    np.savetxt(
        path + 'real' + name + '{}.txt'.format(Nz),
        perturb,
        delimiter="\t",
        newline="\n",
        fmt="%1.16e",
    )
