# Code verification with pseudo-spectral linear solutions

This directory contains information which should enable you to initialize 
simulations with linear solutions obtained with Psecas. The procedure is 
described [here](http://arxiv.org/abs/1902.01403)
but please feel free to send an email to me (Thomas Berlok) if you need more
practical assistance with creating the initial conditions. 

## Four verification tests

The four python files in this directory creates data for initializing each of 
the four tests listed in Table 2 in the paper. The .txt files that are 
needed are also included for convenience (for users who do not have the 
required Python setup to run Psecas).


There are basically two ways to initialize the MHD simulations.


#### Read a text file into the MHD code
The first option is to read the Fourier coefficients for the 
perturbations from one of the included text files and add the perturbations 
inside the problem generator for the MHD code itself.

We show some sample C-code below which adds the perturbations.

The second piece of code assumes that the Fourier coefficients for e.g. 
density perturbation, have been loaded from the .txt files and are stored in
'drho_vec' and 'idrho_vec' for the density perturbation.

For complenetess, one can read the coefficients into C using the following 
(admittedly ugly) code:
```
  // Check if the text file is there
  FILE *file = fopen(fname, "r");
  if (file == NULL) 
    error("Could not open initial condition file %s", fname);
  
  // Figure out how many coefficients there are.
  int N = 0; 
  char cha;
  for (cha = getc(file); cha != EOF; cha = getc(file)) 
    if (cha == '\n')
      N = N + 1;
  fclose(file);
  
  printf("There are N=%i coefficients\n", N);

  // Reopen initial conditions and read the coefficients
  file = fopen(fname, "r");
  double drho, dvx, dvz, dT, dA;
  double drho_vec[N], dvx_vec[N], dvz_vec[N], dT_vec[N], dA_vec[N];
  double idrho_vec[N], idvx_vec[N], idvz_vec[N], idT_vec[N], idA_vec[N];
  for (int i=0; i<N; i++) {
    #ifdef MHD
      fscanf(file, "%lf%lf%lf%lf%lf%lf%lf%lf%lf%lf", &drho_vec[i], &idrho_vec[i], 
                                                      &dvx_vec[i],  &idvx_vec[i], 
                                                      &dvz_vec[i],  &idvz_vec[i], 
                                                      &dT_vec[i],   &idT_vec[i], 
                                                      &dA_vec[i],   &idA_vec[i]);
    #else
      fscanf(file, "%lf%lf%lf%lf%lf%lf%lf%lf", &drho_vec[i], &idrho_vec[i], 
                                                &dvx_vec[i],  &idvx_vec[i], 
                                                &dvz_vec[i],  &idvz_vec[i], 
                                                &dT_vec[i],   &idT_vec[i]);
    #endif
  }
fclose(file);
```

The equilibrium can then be set using:

```
// Background values

zm1 = z - 0.5;
zm2 = z - 1.5;

// Background density
rho = rho0*(1.0 + delta/2.0*(tanh(zm1/a) - tanh(zm2/a)));

// Background velocity in x and z
vx = vflow*(tanh(zm1/a) - tanh(zm2/a) - 1.0);
vz = 0.0;

// Temperature
T = rho0/rho;

tmp = fourier_series_real(z, drho_vec, idrho_vec, N);
itmp = fourier_series_imag(z, drho_vec, idrho_vec, N);
drho = amp*2.0*(cos(kx*x)*tmp - sin(kx*x)*itmp);
rho += rho*drho;

tmp = fourier_series_real(z, dvx_vec, idvx_vec, N);
itmp = fourier_series_imag(z, dvx_vec, idvx_vec, N);
dvx = amp*2.0*(cos(kx*x)*tmp - sin(kx*x)*itmp);
vx += dvx;


tmp = fourier_series_real(z, dvz_vec, idvz_vec, N);
itmp = fourier_series_imag(z, dvz_vec, idvz_vec, N);
dvz = amp*2.0*(cos(kx*x)*tmp - sin(kx*x)*itmp);
vz += dvz;

tmp = fourier_series_real(z, dT_vec, idT_vec, N);
itmp = fourier_series_imag(z, dT_vec, idT_vec, N);
dT = amp*2.0*(cos(kx*x)*tmp - sin(kx*x)*itmp);
T += T*dT;

// Set the pressure
p = rho*T;
```

where we have defined the functions

```
// Evaluate the real part of a complex function at coordinate z by using its 
// complex Fourier series.
// cr and ci are the real and imaginary parts of the N coefficients.
static double fourier_series_real(const double z, const double cr[], 
                                  const double ci[], const int N)
{
  double yr = 0.0;
  int n = -N/2;
  double dz = Lz/N;

  for (int i=0; i< N; i++){
        yr += cr[i]*cos(2*PI*n*(z - 0.5*dz)/Lz) - 
              ci[i]*sin(2*PI*n*(z - 0.5*dz)/Lz);
        n += 1;
      }

  return yr/sqrt(N);
}

// Evaluate the imaginary part of a complex function at coordinate z by using 
// its complex Fourier series.
// cr and ci are the real and imaginary parts of the N coefficients.
static double fourier_series_imag(const double z, const double cr[], 
                                  const double ci[], const int N)
{
  double yi = 0.0;
  int n = -N/2;
  double dz = Lz/N;

  for (int i=0; i< N; i++){
        yi += ci[i]*cos(2*PI*n*(z - 0.5*dz)/Lz) + 
              cr[i]*sin(2*PI*n*(z - 0.5*dz)/Lz);
        n += 1; 
      }

  return yi/sqrt(N);
}
```

#### Use Python to create initial conditions

The second option is create an initial snapshot for the simulation.

Using the interpolation method of the grid class, the 
four Python scripts can evalute the linear solution at an arbitrary 
point in the computational domain. You will have to add the code for 
evaluating these functions on the grid used in your MHD code and the code 
needed to save it in the specific format you need.