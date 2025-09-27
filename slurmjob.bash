#!/bin/bash

# see:
# https://doku.lrz.de/running-serial-jobs-on-the-linux-cluster-11484000.html
# https://doku.lrz.de/job-processing-on-the-linux-cluster-10745970.html

#SBATCH -J samplejob
#SBATCH -D /dss/dsshome1/03/ru48miy3/master-thesis/code

#SBATCH -o /dss/dsshome1/03/ru48miy3/master-thesis/code/jobs/%x_lrz_%j.log
#SBATCH -e /dss/dsshome1/03/ru48miy3/master-thesis/code/jobs/%x_lrz_%j.log

#SBATCH --mail-type=END
#SBATCH --mail-user=felix.edelmann@campus.lmu.de

#SBATCH --get-user-env

#SBATCH --export=NONE

#SBATCH --clusters=serial
#SBATCH --partition=serial_std

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1

#SBATCH --time=00:10:00

module load slurm_setup

set -e

start=`date +%s`
echo "Starting job at `date`"
echo "Git commit: `git rev-parse HEAD`"

echo "System information:"
#lsb_release -a
cat /etc/os-release
uname -srvmpio
#inxi -C

module load python/3.10.12-base

# pip install uv
# uv sync

echo "Run benchmarkjob.py at `date`"
uv run main.py benchmark ${SLURM_JOB_NAME}

end=`date +%s`
echo "Ending job at `date` (runtime: $((end-start))s)"

# commands:
# sbatch slurmjob.bash
# squeue
# sacct -L
