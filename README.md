[![Build Status](https://travis-ci.com/nbia-astro/skeletor.svg?token=SrP7KstmwUSGLQustYFw&branch=master)](https://travis-ci.com/nbia-astro/skeletor)

# Introduction

This github repository contains a collection of methods for solving
eigenvalue problems using pseudo-spectral methods. These methods are described
in e.g. Boyd, Matlab, and the last one.

The user writes down a linearized set of equations, the eigenvalue problem,
which is then discretized on either an infinite domain, a finite domain (a, b)
or a periodic domain.

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
