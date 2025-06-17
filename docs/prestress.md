# Pre-stressing the mesh

While the process of meshing is as described in the previous section, there is one remaining procedure before running a FSI simulation, which is to pre-deform the mesh. This process is essential because the medical images capture the vascular wall in an in vivo stress equilibrium state, where both the blood pressure and wall stress are unknown. Starting FSI simulations without accounting for this initial stressed state would lead to non-physiological wall displacements {cite}`Hsu2011`. In `VaSP`, this pre-deformation process involves the following two steps:

First, users need to run the `src/vasp/simulations/predeform.py` script to perform an intermediate FSI simulation. This simulation utilizes a fully implicit time integration scheme (`theta=1`). While this scheme has first-order accuracy, its high numerical stability allows for the use of a larger time step, significantly speeding up the process. During this step, an initial medical image-driven mesh is inflated under a cardiac-cycle averaged pressure, specified as `P_final`. How to run the FSI simulation is described in the next section. 

Second, run the following command to generate the pre-deformed mesh:

```console
vasp-generate-mesh --folder /predeform_resutls/1/
```

where `--folder` specifies the path containing the results of the pre-deformation simulation. This command will the inverse of the displacements obtained in the first step to the original mesh, thereby obtaining the stress-free reference configuration required for main FSI simulations.

This pre-deformed mesh is then used as the input for the main FSI simulation, where we will gradually ramp up the pressure to restore the stress equilibrium state. The whole process is illustrated in {numref}`prestress_process`.

```{figure} figures/prestress.png
---
width: 600px
align: center
name: prestress_process
---
The process of pre-stressing the mesh before running a FSI simulation. A. The medical image-driven mesh in the in vivo stress equilibrium state. B. The inflated mesh with estimated displacements under a cardiac-cycle averaged pressure. C. The pre-deformed mesh is obtained by applying the inverse of the displacements to the original mesh. During the main FSI simulation, the pressure is gradually ramped up to restore the stress equilibrium state.
```

```{bibliography}
:filter: docname in docnames
```