# Pre-processing

Since pre-processing part of `VaSP` is based on [VaMPy](https://github.com/KVSlab/VaMPy), overlapping functionality of the pre-processing will not be covered here. Instead, users are encouraged to refer to the [VaMPy documentation](https://kvslab.github.io/VaMPy/preprocess.html) for some of the basic functionalities. In this document, we will focus on the newly implemented parts of the pre-processing that are specific for performing fluid-structure interaction (FSI) simulations. 

##  Fluid and Solid mesh generation

To perform FSI simulations, fluid and solid regions need to be generated and marked so that different partial differential equations can be solved on each domain. In `VaSP`, user is, first, required to provide the surface mesh, which represent the interface of the fluid and solid domain. Then, fluid mesh is created inside the surface mesh to fill the provided surface mesh while solid mesh is created outside the surface mesh to cover the fluid mesh. This mechanism of generating two different mesh works very well for blood vessels. We extended VMTK, a framework for generating mesh based for image-based vascular modelling, to enable the solid mesh generation by utilizing the functionality of adding boundary layer meshing. Fig {numref}`{domains}` is an example of FSI mesh with fluid and solid regions marked as 1 and 2, respectively.

```{figure} figures/case9_domain_ids.png
---
name: domains
---
Fluid and Solid domain with unique ID
```

## Boundary conditions
