## Code verification with pseudo-spectral linear solutions

This directory contains information which should enable you to initialize 
simulations with linear solutions obtained with Psecas.

The procedure is described in the paper 

[insert here](link)

Please feel free to send an email to me (Thomas Berlok) if you need 
assistance with creating the initial conditions. 

### Four tests

The four python files in this directory creates data for initializing each of 
the four tests listed in Table 2 in the paper.

### Loading the perturbed initial condition into an MHD code

There are basically two ways to initialize the simulations: 

 -  By generating 
    an initial snapshot for the simulation to start from, e.g., the MHD code Arepo 
    can start from an HDF5 file containing the initial conditions. 
    We provide python scripts which contains functions that can evalute the 
    initial condition at an arbitrary point in the compuational domain. 
    You will have to add the code for evaluating these 
    functions on the grid used in your MHD code and the code needed to save it
    in the specific format you need.

    The sample Python script can also be used for comparing with simulations.

 -  By reading the Fourier coefficients for the 
    perturbations from two .txt files and adding the perturbation inside the 
    problem generator for the MHD code itself. This is what we have done in 
    the Athena simulations presented in the paper.

    We provide sample (admittedly ugly) C-code for loading in the the Fourier
    coefficients.
