# Fluid Structure Interaction Simulations and monitoring tool during the simulation

## Simulations in `turtleFSI`

`VaSP` uses [turtleFSI](https://github.com/KVSlab/turtleFSI) for performing FSI simulations. In short, `turtleFSI` is a monolithic FSI solver built based on [FEniCS](https://fenicsproject.org). For the detailed usage of `turtleFSI` and tutorials, users are referred to the [turtleFSI documentation](https://turtlefsi2.readthedocs.io/en/latest/). Here, we will introduce the very basic command for using `turtleFSI` and supporting functions that are specifically added for performing vascular FSI. 

After the pre-processing step, mesh file with `*.h5` suffix is required as an input to the FSI simulation. In addition, user may use mesh file `*_boundaries.pvd` to check the ID for boundaries.