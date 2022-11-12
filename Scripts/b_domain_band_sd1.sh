#!/bin/bash

source script_params.config

if [ $operating_sys != "local" ] ;
then
cd $SLURM_SUBMIT_DIR
source /home/s/steinman/dbruneau/sourceme.conf
echo "Running scripts on Niagara"
else
echo "Running scripts on local os"
fi

# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.configbat
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt, end time of simulation: $end_t"

# Run postprocessing scripts
str="Running h5py postprocessing scripts!"
echo $str
python $workflow_location/postprocessing_h5py/create_visualizations.py --case=$case_path --mesh=$mesh_path --dt=$dt --start_t=$start_t_sd1 --end_t=$end_t --dvp=v --save_deg=1 --stride=$stride_sd1 --bands="25,100000"
python $workflow_location/postprocessing_h5py/create_visualizations.py --case=$case_path --mesh=$mesh_path --dt=$dt --start_t=$start_t_sd1 --end_t=$end_t --dvp=d --save_deg=1 --stride=$stride_sd1 --bands="25,100000"
python $workflow_location/postprocessing_h5py/create_visualizations.py --case=$case_path --mesh=$mesh_path --dt=$dt --start_t=$start_t_sd1 --end_t=$end_t --dvp=p --save_deg=1 --stride=$stride_sd1 --bands="25,100000"
