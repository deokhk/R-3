#!/bin/bash

#SBATCH -J gen_retrieval_data_without_hn # job name
#SBATCH -o sbatch_output_log/output_%x_%j.out # standard output and error log
#SBATCH -p cpu-max64-1 # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH --nodes=1
#SBATCH --ntasks=16
#SBATCH --cpus-per-task=4

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date
/home/deokhk/elasticsearch-7.15.0/bin/elasticsearch
python gen_hard_negative.py

date