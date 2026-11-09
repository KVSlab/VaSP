# Pre-stressing the mesh

While the process of meshing is as described in the previous section, there is one remaining procedure before running a FSI simulation, which is to pre-deform the mesh. This process is essential because the medical images capture the vascular wall in an in vivo stress equilibrium state, where both the blood pressure and wall stress are unknown. Starting FSI simulations without accounting for this initial stressed state would lead to non-physiological wall displacements {cite}`Hsu2011`. In `VaSP`, this pre-deformation process involves the following two steps:

First, users need to run the `src/vasp/simulations/predeform.py` problem file to perform an intermediate FSI simulation, giving the problem file of the main FSI simulation with `--main_problem`:

```console
turtleFSI -p predeform --main_problem=aneurysm
```

where `--main_problem` is the name of the main problem file (looked up in the current directory, then in `vasp.simulations` and `turtleFSI.problems`) or the path to it, e.g. `--main_problem=path/to/my_problem.py`. The pre-deformation simulation takes from the main problem file:

- the solid properties: `material_model` (mandatory), `solid_properties` or `rho_s`, `mu_s` and `lambda_s`, and `dx_s_id`;
- the fluid properties: `fluid_properties` or `rho_f` and `mu_f`, `dx_f_id`, and `nonnewtonian` and `cross_params` for a non-Newtonian fluid;
- the Robin boundary condition, if it is used: `robin_bc`, `k_s`, `c_s` and `ds_s_id`;
- the load of the pre-deformation: `P_final` and `v_max_final` (mandatory);
- the boundary IDs `inlet_id` and `fsi_id` (mandatory), and `rigid_id` and `inlet_outlet_s_id` (optional), each as an integer or a list;
- the function `get_mesh_domain_and_boundaries`, if the main problem file defines it, together with the parameters it takes (e.g. `mesh_path`), so that the regions created there are the same in both simulations.

This way, the pre-deformation and the main simulation always use the same material models and parameters. Missing or inconsistent parameters stop the simulation with an explicit error before the mesh is loaded, and the material models used are printed in the log of both simulations. Arguments given on the command line still override the values of the main problem file.

This simulation utilizes a fully implicit time integration scheme (`theta=1`). While this scheme has first-order accuracy, its high numerical stability allows for the use of a larger time step, significantly speeding up the process. During this step, an initial medical image-driven mesh is inflated under a cardiac-cycle averaged pressure, specified as `P_final`, and inlet centerline velocity, specified as `v_max_final`. The results are stored in `<folder>/predeform_results`, where `<folder>` is the results folder of the main problem file. How to run the FSI simulation is described in the next section.

Second, run the following command to generate the pre-deformed mesh:

```console
vasp-predeform-mesh --folder <folder>/predeform_results/
```

where `--folder` specifies the path containing the results of the pre-deformation simulation. This command will apply the inverse of the displacements obtained in the first step to the original mesh, thereby obtaining the stress-free reference configuration required for main FSI simulations.

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