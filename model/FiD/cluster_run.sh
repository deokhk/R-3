#!/bin/bash

#SBATCH -J 64_new_linear_train_and_test # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p 4A100 # queue name or partiton name
#SBATCH -q 4A100
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:4
#SBATCH  --nodes=1
#SBATCH  --ntasks=8
#SBATCH  --cpus-per-task=4
#SBATCH  --mem=300G

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh train_and_test_reader.sh

date
