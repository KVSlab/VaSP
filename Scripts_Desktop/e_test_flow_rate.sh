#!/bin/bash

# conda activate fenicsproject # must run this in an environment with fenics installed

# point to config file on the command line, like this: sbatch e_flow_rate.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, end time of simulation: $end_t"

str="Running fenics postprocessing scripts!"

echo $str
python postprocessing_fenics/compute_flow_rate_fsi.py --case=$case_path --mesh=$mesh_path --dt=$dt 