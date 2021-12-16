#!/bin/bash

#SBATCH -J DPR_scratch # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p A100-pci # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:4
#SBATCH  --nodes=1
#SBATCH  --ntasks=4
#SBATCH  --cpus-per-task=4
#SBATCH  --mem=300G

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh train_dpr_scratch.sh

date
