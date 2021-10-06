#!/bin/bash

#SBATCH -J dpr_with_table # job name
#SBATCH -output_.%j.out # standard output and error log
#SBATCH -p  A100 # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:4
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=4

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge

date

sh train_encoder_with_table.sh

date
