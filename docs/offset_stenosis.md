(tutorial:offset_stenosis)=

# Offset stenosis simulation

As a first tutorial, we will illustrate major functionalities of VaSP with a non-axisymmetric stenosis model. This stenosis model is a synthetic geometry that can be described analytically and has been long used within fluid mechanics research {cite}`Varghese2007`. Due to the stenotic region with sudden area reduction, the flow becomes turbulent quickly without complex geometry or high Reynolds number. Additionally, eccentricity ensures that the simulation is deterministic, avoiding uncertainty associated with the symmetric geometry. Those features enable us to quickly develop and test the functionalities with reasonable computing time.

## Mesh generation

There are couple of methods for determining the size of each tetrahedral cell, but `VaSP` uses centerline diameter by default. Details on different meshing strategy can be found [here](https://kvslab.github.io/VaMPy/artery.html#meshing-based-on-the-centerline-diameter).   

With that in mind, one can generate the mesh as follows: 

```console
vasp-generate-mesh -i offset_stenosis.stl -f True -fli 0 -flo 4 -c 3.8 -nbf 1 -nbs 1
```
Here, `-f` is a boolean argument determining whether to add flow extensions or not, as explained in [here](https://kvslab.github.io/VaMPy/preprocess.html#flow-extensions). In this case, `-flo 0` indicates no flow extension at the inlet, but `-flo 4` will add flow extensions at the outlet with length equal to four times the length of radius of the outlet. `-c` is a coarsing factor determining the density of the mesh, where `c > 1` will corasen the mesh. `-nbf` and `-nbs` are the number of sublayers for fluid boundary layer and solid domain, respectively. In this offset stenosis problem, we only use one layer as solid to speed up the computation. 

```{bibliography}
:filter: docname in docnames
```