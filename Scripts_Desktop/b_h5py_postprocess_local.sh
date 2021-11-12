#!/bin/bash

conda activate base

# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, save frequency: $save_step,save degree: $save_deg, end time of simulation: $end_t"

# Run postprocessing scripts
str="Running h5py postprocessing scripts!"
echo $str
python postprocessing_h5py/compute_domain_specific_viz.py --case=$case_path --mesh=$mesh_path --dt=$dt --end_t=$end_t --save_deg=$save_deg
python postprocessing_h5py/compute_domain_specific_viz.py --case=$case_path --mesh=$mesh_path --dt=$dt --end_t=$end_t --save_deg=$save_deg --dvp=d
python postprocessing_h5py/compute_domain_specific_viz.py --case=$case_path --mesh=$mesh_path --dt=$dt --end_t=$end_t --save_deg=$save_deg --dvp=p
