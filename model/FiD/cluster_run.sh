#!/bin/bash

#SBATCH -J 64_multi_test # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p A100 # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:1
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=32
#SBATCH  --mem=200G

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh test_reader.sh

date
