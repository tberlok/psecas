[![CircleCI](https://circleci.com/gh/tberlok/evp/tree/master.svg?style=svg&circle-token=067ada3c41e0a21e2ef785e3f7a88d481ca1ed43)](https://circleci.com/gh/tberlok/evp/tree/master)

# Introduction

This github repository contains a collection of methods for solving
eigenvalue problems using pseudo-spectral methods. These methods are described
in e.g. the books Spectral Methods in Matlab by Lloyd N. Trefethen,
A Practical Guide to Pseudospectral Methods by Bengt Fornberg and
Chebyshev and Fourier Spectral Methods by John P. Boyd.

The user writes down a linearized set of equations, the eigenvalue problem,
which is then discretized on either an infinite, finite or periodic domain.

# Installation

I assume you have Python 3.6 installed. If so, all requirements are simply
installed by running the following command

```
$ pip install -r requirements.txt
```
at the top-level directory.

# Testing

Before using the code, the tests should be run to make sure that everything is
working. From the top-level directory
```
$ pytest tests/
```

# TODO

1. Set up Kelvin-Helmholtz instability in a slab.
2. Consider improving the automatic setup of the eigenvalue problem.
3. Set up a problem with MPI, writing data to file and reloading it again
   should work in (almost) the same way as running the code interactively.
4. Add method for outputting txt files for use in Athena.
5. Add method for creating hdf5-files for use in Arepo.
6. Set up MTI simulation with input from evp.

