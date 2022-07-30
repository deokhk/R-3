#!/bin/bash

#SBATCH -J col_only_up_total # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p A100-80GB # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:1
#SBATCH  --nodes=1
#SBATCH  --ntasks=4
#SBATCH  --cpus-per-task=4
#SBATCH  --mem=300G

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date
sh train_rel_column_tot.sh
date
