#!/bin/bash

export MPLCONFIGDIR=/scratch/s/steinman/dbruneau/.config/matplotlib

module --force purge
module load CCEnv StdEnv/2020 gcc/9.3.0 vtk/9.0.1 python/3.7.7
source $HOME/../macdo708/.virtualenvs/vtk9/bin/activate

# run with source 00_postprocess_all.sh Cyl_Long.config
# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path"
echo "$CameraPosition"

echo "Running pyvista postprocessing scripts!"

python $workflow_location/postprocessing_video/PlotAneuAmplitude.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale auto 2960 $ColorbarY
python $workflow_location/postprocessing_video/PlotAneuAmplitude.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.1 2960 $ColorbarY
python $workflow_location/postprocessing_video/PlotAneuAmplitude.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.25 2960 $ColorbarY
python $workflow_location/postprocessing_video/PlotAneuAmplitude.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.5 2960 $ColorbarY
python $workflow_location/postprocessing_video/PlotAneuContour.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.3 2960
python $workflow_location/postprocessing_video/PlotAneuContour.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.4 2960
python $workflow_location/postprocessing_video/PlotAneuContour.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.5 2960
python $workflow_location/postprocessing_video/PlotAneuContour.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.6 2960
python $workflow_location/postprocessing_video/PlotAneuContour.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.7 2960
python $workflow_location/postprocessing_video/PlotAneuContour.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 0.75 2960
python $workflow_location/postprocessing_video/PlotAneuQ_hi_lo.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 50000 100 2960
python $workflow_location/postprocessing_video/PlotAneuQ_hi_lo.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 50000 200 2960
python $workflow_location/postprocessing_video/PlotAneuQ_hi_lo.py $case_path $mesh_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale 50000 500 2960

# --mesa-swr 