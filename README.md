# Aneurysm_Workflow_FSI
A collection of postprocessing scripts for use with TurtleFSI. \

Updates as of November 12th, 2022:\
-Scripts have had various updates, they are now set up assuming ou have run your simulation using save_deg = 2\
-Added pulsatile offset stenosis as another demo. This runs relatively quickly and generates flow instability with similar spectral characteristics to a high-resolution simulation.\
-Chromagrams, spectral bandedness index (SBI) and power spectral density are now computed in the spectrogram script. \
-Added meshing piepline (work in progress)\
-Added norm calculation for cycle-to-cycle convergence (must run multiple cycles for this to work)\

The workflow generally goes like this:\
a. Create meshes for postprocessing (solid-only, fluid-only, refined)\
b/e. Create visualizations for displacement, fluid only displacement for velocity and pressure. Can also compute spectrograms at this stage. \
c/d. Create a dolfin-readable version of the solid-only and fluid-only visualization, then compute wall shear stress and solid stress/strain. \
f/g. Compute SPI and High-Pass Stress (requires wall shear stress and stress/strain). \

Some of these scripts work well in parallel (compute_solid_stress.py, compute_wss_fsi.py, compute_flow_rate_fsi.py) while many don't:\
-The "postprocessing_mesh" scripts do not run in parallel due to their use of "SubMesh" and "adapt". \
-The "postprocessing_h5py" scripts run in parallel but not effectively. io can be an issue if using the intermediate nodes (using save_deg = 2). A better io solution is required for this code to run effectively in parallel. \
-The "compute_readable_h5.py" script does not run in parallel due to the use of "vector().set_local()". A parallel solution may be possible here so this script can be merged with compute_solid_stress.py and compute_wss_fsi.py.\

This folder contains a short simulation of a coarse cylinder. The folder "Scripts" contains some .sh files that can be used to run the workflow more easily on a desktop. In the following example, we are pointing to a coarse offset stenosis demo file. If the stenosis is run with save_deg = 2 (saving intermediate nodes), a good strategy is to use fewer timesteps for postprocessing with save_deg = 2 (for example, stress and wall shear stress), and save just the corner nodes for some other visualizations where we want to look at all timesteps.  \

bash Scripts/a_create_meshes.sh stenosis_test/offset_stenosis.config\
bash Scripts/b_domain_band_sd1.sh stenosis_test/offset_stenosis.config\
bash Scripts/b_domain_band_sd2.sh stenosis_test/offset_stenosis.config\
bash Scripts/c_compute_norms.sh stenosis_test/offset_stenosis.config\
bash Scripts/c_create_readable_h5.sh stenosis_test/offset_stenosis.config\
bash Scripts/d_compute_stress_wss_local.sh stenosis_test/offset_stenosis.config\
bash Scripts/e_compute_spectrograms_local.sh stenosis_test/offset_stenosis.config\
bash Scripts/f_compute_spi_local.sh stenosis_test/offset_stenosis.config\
bash Scripts/g_domain_band_strain.sh stenosis_test/offset_stenosis.config\

The script "compute_flow_metrics.py" is completely untested, and is a copy from the old aneurysm workflow. It seems that script is still under development. 
