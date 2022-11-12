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

# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path, timestep: $dt,  end time of simulation: $end_t"

# Run postprocessing scripts
str="Creating meshes for postprocessing!"
echo $str

python $workflow_location/postprocessing_mesh/Create_Refined_Mesh.py --case=$case_path --mesh=$mesh_path
python $workflow_location/postprocessing_mesh/Create_Solid_Only_Mesh.py --case=$case_path --mesh=$mesh_path
python $workflow_location/postprocessing_mesh/Create_Fluid_Only_Mesh.py --case=$case_path --mesh=$mesh_path
python $workflow_location/postprocessing_mesh/Create_Solid_Only_Mesh.py --case=$case_path --mesh=$refined_mesh_path
python $workflow_location/postprocessing_mesh/Create_Fluid_Only_Mesh.py --case=$case_path --mesh=$refined_mesh_path
