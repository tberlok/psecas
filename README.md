[![CircleCI](https://circleci.com/gh/tberlok/freja.svg?style=svg&circle-token=067ada3c41e0a21e2ef785e3f7a88d481ca1ed43)](https://circleci.com/gh/tberlok/freja)
# Introduction

This github repository contains a collection of methods for solving
eigenvalue problems (EVPs) using pseudo-spectral methods. These methods are described
in e.g. the books
[Spectral Methods in Matlab](https://people.maths.ox.ac.uk/trefethen/spectral.html)
by Lloyd N. Trefethen,
[A Practical Guide to Pseudospectral Methods](https://books.google.de/books/about/A_Practical_Guide_to_Pseudospectral_Meth.html?id=IqJoihDba3gC&redir_esc=y)
by Bengt Fornberg and
[Chebyshev and Fourier Spectral Methods](http://depts.washington.edu/ph506/Boyd.pdf)
by John P. Boyd.

### How it works
The user writes down a linearized set of equations, the eigenvalue problem,
which is automatically discretized on either an infinite, finite or
periodic domain. The resulting EVP can then be solved to a requested precision.

### Overview of the code
The code consist of three main classes

- The Grid class

  which contains the grid points and methods for performing spectral
  differentation on it.
- The System class

  which contains the linearized equations in string format, and other
  parameters of the problem. The parameters are allowed to depend on
  the coordinate.
- The Solver class

  which contains functionality for

    - automatically creating a (generalized) eigenvalue
      problem of dimension _d_ × _N_ where _N_ is the number of grids points and
      _d_ is the number of equations in the system.
    - solving the eigenvalue problem to a specified tolerance, e.g. 1e-6, of
      the returned eigenvalue.

#### Grids
An overview of various grid types is shown on page 11 in the book by
[Boyd](http://depts.washington.edu/ph506/Boyd.pdf).

<img src="images/boyd_overview.png" width="300">

The code currently has all the grids mentioned on this figure. We use ([a fork](https://github.com/tberlok/dmsuite))
of a Python version of

J. A. C. Weidemann and S. C. Reddy, A MATLAB Differentiation Matrix Suite,
ACM Transactions on Mathematical Software, 26, (2000): 465-519.

for creating the Laguerre and Hermite grids. The other grids are created using
the descriptions in the books by
[Boyd](http://depts.washington.edu/ph506/Boyd.pdf) and
[Trefethen.](https://people.maths.ox.ac.uk/trefethen/spectral.html)

# Examples

We provide a range of examples of using the code to solve linear eigenvalue
problems in astrophysical fluid dynamics and also include some of the examples
found in the books mentioned at the top of the page.

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
$ pytest tests/ freja/grids/*
```

