#!/bin/bash

#conda activate fenicsproject

# run with source 00_postprocess_all.sh Cyl_Long.config
# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, save frequency: $save_step, end time of simulation: $end_t"

conda activate base
str="Running spectral/other postprocessing scripts!"
echo $str

python postprocessing_h5py/plot_compute_time.py --case=$case_path
python postprocessing_h5py/compute_spi.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --save_deg=$save_deg
python postprocessing_h5py/create_spectrograms_chromagrams.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --save_deg=$save_deg --r_sphere=$r_sphere --x_sphere=$x_sphere --y_sphere=$y_sphere --z_sphere=$z_sphere
python postprocessing_h5py/compute_spi.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --save_deg=$save_deg --dvp=d
python postprocessing_h5py/create_spectrograms_chromagrams.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --save_deg=$save_deg --r_sphere=$r_sphere --x_sphere=$x_sphere --y_sphere=$y_sphere --z_sphere=$z_sphere --dvp=d
python postprocessing_h5py/compute_spi.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --save_deg=$save_deg --dvp=p
python postprocessing_h5py/create_spectrograms_chromagrams.py --case=$case_path --mesh=$mesh_path --end_t=$end_t --save_deg=$save_deg --r_sphere=$r_sphere --x_sphere=$x_sphere --y_sphere=$y_sphere --z_sphere=$z_sphere --dvp=p
