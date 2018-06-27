def setup(backend="pdf"):

    from sys import modules

    # settings = {}

    if 'matplotlib.backends' in modules:
        from matplotlib import get_backend, interactive
        if get_backend() != backend:
            interactive(True)
    else:
        from matplotlib import use
        use(backend)
        from matplotlib import rc
        from numpy import sqrt
        columnwidth = 510.0 # Get this from \showthe\columnwidth
        dpi = 72
        aspect = 0.5#(sqrt(5) - 1)/2
        figwidth = columnwidth/dpi
        fontsize = 10

        rc('figure', figsize=(figwidth, aspect*figwidth), dpi=dpi)
        rc('xtick', direction='out')
        rc('ytick', direction='out')
        rc('axes', labelsize=fontsize, titlesize=fontsize)
        rc('font', size=fontsize)
        rc('legend', fontsize=fontsize)
        rc('xtick', labelsize=fontsize)
        rc('ytick', labelsize=fontsize)
        rc('font', family='serif', serif='cm')
        rc('text', usetex=True)
        rc('text.latex', preamble=[
          r'\usepackage[T1]{fontenc}',
          r'\usepackage[english]{babel}',
          r'\usepackage[utf8]{inputenc}',
          r'\usepackage{lmodern}',
          r'\usepackage{microtype}',
          r'\usepackage{amsmath}',
          r'\usepackage{bm}'])

    from matplotlib import is_interactive

    return is_interactive()


def savefig(figure, scriptname):

    from matplotlib import get_backend
    from os import makedirs
    from os.path import dirname, realpath, exists, splitext, basename

    name, py = splitext(basename(scriptname))

    backend = get_backend()

    ext = {'pdf': 'pdf', 'ps': 'eps', 'TkAgg': 'eps'}[backend]

    figdir = dirname(realpath(__file__))

    if not exists(figdir):
        makedirs(figdir)

    figure.savefig(figdir + '/' + name + '.' + ext)

    if backend == 'ps':
        from os import system
        system('epstopdf --hires ' + name + '.eps')
