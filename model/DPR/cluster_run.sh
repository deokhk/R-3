#!/bin/bash

#SBATCH -J Proposed_chkpt_single # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p A100-pci # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:1
#SBATCH  --nodes=1
#SBATCH  --ntasks=4
#SBATCH  --cpus-per-task=4

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh train_rel_dpr_chkpt.sh

date
