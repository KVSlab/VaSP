# Aneurysm_Workflow_FSI
A collection of postprocessing scripts for use with TurtleFSI. 

The workflow generally goes like this:\
a. Create meshes for postprocessing (solid-only, fluid-only, refined)\
b. Create solid-only visualization for displacement, fluid only displacement for velocity and pressure\
c. Create a dolfin-readable version of the solid-only and fluid-only visualization, then compute wall shear stress and solid stress. \
d. Compute SPI and Spectrograms (requires wall shear stress). \

Some of these scripts work well in parallel (compute_solid_stress.py, compute_wss_fsi.py, compute_flow_rate_fsi.py) while many don't:\
-The "postprocessing_mesh" scripts do not work in parallel due to their use of "SubMesh" and "adapt". \
-The "postprocessing_h5py" scripts run in parallel but io can be an issue if using the intermediate nodes (using save_deg = 2). A better io solution is required for this code to run effectively in parallel. \
-The "compute_readable_h5.py" script does not run in parallel due to the use of "vector().set_local()". A parallel solution may be possible here so this script can be merged with compute_solid_stress.py and compute_wss_fsi.py.\

This folder contains a short simulation of a coarse cylinder. The folder "Scripts_Desktop" contains some .sh files and a config file that can be used to run the workflow more easily on a desktop. By default, it points to the test cylinder and you can modify the config file to point to your own simulation. To run these scripts, execute the commands:\
./Scripts_Desktop/a_create_meshes.sh Scripts_Desktop/config_files/Cyl.config\
./Scripts_Desktop/b_h5py_postprocess_local.sh Scripts_Desktop/config_files/Cyl.config \
./Scripts_Desktop/c_fenics_postprocess_local.sh Scripts_Desktop/config_files/Cyl.config \
./Scripts_Desktop/d_spectral.sh Scripts_Desktop/config_files/Cyl.config \
./Scripts_Desktop/e_test_flow_rate.sh Scripts_Desktop/config_files/Cyl.config \
