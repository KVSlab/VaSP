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

Similar to the domains, boundaries also need to be marked separately for applying different boundary conditions. Below is a list of boundaries that are necessary to be identified for performing FSI simulations. 

<ol>
  <li>Fluid inlet</li>
  <li>Fluid outlet</li>
  <li>Fluid and Solid interface</li>
  <li>Solid inlet & outlet</li>
  <li>Solid outer wall</li>
</ol>

The first two boundaries are necessary for specifying boundary conditions for the fluid domain. In case of do-nothing boundary condition at the outlet, there is no need to specify the ID for the outlet while the inlet ID is always required to specify the inflow. The third boundary, fluid and solid interface, is required to distinguish static and dynamic boundary between fluid and solid. The fourth boundary is usually used to fix the solid inlet and outlet while the last boundary is used to specify Robin boundary condition at the outer wall of the vascular system for representing surrounding environment. 

