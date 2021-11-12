#!/bin/bash

# conda activate fenicsproject # must run this in an environment with fenics installed

# point to config file on the command line, like this: ./c_fenics_postprocess_local.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, save frequency: $save_step, end time of simulation: $end_t"

# Run postprocessing scripts
str="Creating readable h5 files (must be run in serial)"
echo $str
python postprocessing_fenics/compute_readable_h5.py --case=$case_path --mesh=$mesh_path --dt=$dt

str="Running fenics postprocessing scripts!"

echo $str
python postprocessing_fenics/compute_solid_stress.py --case=$case_path --mesh=$mesh_path --dt=$dt --stride=$stride_stress
python postprocessing_fenics/compute_wss_fsi.py --case=$case_path --mesh=$mesh_path --dt=$dt --stride=$stride_stress