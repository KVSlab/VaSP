#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks=40
#SBATCH --time=03:59:00
#SBATCH --job-name mpi_aneuturtle

cd $SLURM_SUBMIT_DIR
source /home/s/steinman/dbruneau/sourceme.conf

config_file=$(find $SLURM_SUBMIT_DIR -type f -name '*.config')
. $config_file

# Code to run:
num=$NUM
log_file="logfile_r$num.txt"
checkpoint_folder="$mesh_path/1/Checkpoint"
if [ "$num" -eq 0 ]; then
      mpirun turtleFSI -p=pulsatile_vessel >> $log_file # Start Simulation
else
      mpirun turtleFSI -p=pulsatile_vessel --restart-folder $checkpoint_folder >> $log_file # Restart Simulation
fi

# RESUBMIT ## TIMES HERE
num_restarts=10
FILE=finished
if test -f "$FILE"; then
    echo "simulation is $FILE. Not resubmitting the job."
else 
    if [ "$num" -lt "$num_restarts" ]; then
          num=$(($num+1))
          ssh -t nia-login01 "cd $SLURM_SUBMIT_DIR; sbatch --export=NUM=$num run_aneu.sh";
    fi
fi
# NOTE: you must run this script with:  sbatch --export=NUM=0 run_aneu.sh initially!!