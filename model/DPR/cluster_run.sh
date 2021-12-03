#!/bin/bash

#SBATCH -J new_data_checkpoint_DPR_baseline # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p 4A100 # queue name or partiton name
#SBATCH -q 4A100
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:4
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh train_plain_encoder_with_table_special_token.sh

date
