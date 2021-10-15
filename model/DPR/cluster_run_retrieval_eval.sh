#!/bin/bash

#SBATCH -J eval_re_scratch # job name
#SBATCH -o sbatch_output_log/output_%x.out # standard output and error log
#SBATCH -p  A100 # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:2
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=64
#SBATCH  --mem=512G

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh ./training_scripts/eval_retrieval_scratch.sh

date
