#!/bin/bash

module --force purge
module load NiaEnv/2018a paraview-offscreen/5.6.0

# run with source 00_postprocess_all.sh Cyl_Long.config
# point to config file on the command line, like this: sbatch d_execute.sh path/to/config.config
. $1

echo "Sourcing config file: $1"
echo "case path: $case_path, mesh path: $mesh_path"
echo "$CameraPosition"

echo "Running paraview postprocessing scripts!"

#python $workflow_location/postprocessing_h5py/create_spectrograms_chromagrams.py $case_path --mesh=$mesh_path --end_t=$end_t --start_t=0.0 --save_deg=1 --r_sphere=$r_sphere --x_sphere=$x_sphere --y_sphere=$y_sphere --z_sphere=$z_sphere --stride=$stride_sd1 --dvp=d
#pvbatch --mesa-swr-avx2 --force-offscreen-rendering $workflow_location/postprocessing_paraview/make_iso.py $case_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale

#pvbatch --force-offscreen-rendering $workflow_location/postprocessing_paraview/make_iso.py $case_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale
#pvbatch --force-offscreen-rendering $workflow_location/postprocessing_paraview/make_amplitude_d_lo.py $case_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale
#pvbatch --force-offscreen-rendering $workflow_location/postprocessing_paraview/make_amplitude_d_hi.py $case_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale
#pvbatch --force-offscreen-rendering $workflow_location/postprocessing_paraview/make_amplitude_ep_hi.py $case_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale
#pvbatch --force-offscreen-rendering $workflow_location/postprocessing_paraview/make_amplitude_ep_lo.py $case_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale
pvbatch --force-offscreen-rendering $workflow_location/postprocessing_paraview/make_Q_Crit_Hi_Lo.py $case_path $CameraPosition $CameraFocalPoint $CameraViewUp $CameraParallelScale

# --mesa-swr 