#!/bin/bash

#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=00:55:00
#SBATCH --job-name d_stress_wss

cd $SLURM_SUBMIT_DIR
source /home/s/steinman/dbruneau/sourceme.conf

# run with source 00_postprocess_all.sh Cyl_Long.config
# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt,  end time of simulation: $end_t"


str="Running fenics postprocessing scripts!"
echo $str
mpirun -n 30 python $workflow_location/postprocessing_fenics/compute_wss_fsi.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=$stride_sd2 # --start_t=$start_t_ROI --end_t=$end_t
mpirun python $workflow_location/postprocessing_fenics/compute_solid_stress.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=$stride_sd2
mpirun python $workflow_location/postprocessing_fenics/compute_flow_rate_fsi.py --case=$case_path --mesh=$mesh_path --dt=$dt --save_deg=2 --stride=$stride_sd2
