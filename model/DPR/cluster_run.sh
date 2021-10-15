#!/bin/bash

#SBATCH -J generate_embeddings_scratch # job name
#SBATCH -o sbatch_output_log/output_%x.out # standard output and error log
#SBATCH -p  2080ti # queue name or partiton name
#SBATCH -t 72:00:00 # Run time (hh:mm:ss)
#SBATCH  --gres=gpu:8
#SBATCH  --nodes=1
#SBATCH  --ntasks=1
#SBATCH  --tasks-per-node=1
#SBATCH  --cpus-per-task=4

srun -l /bin/hostname
srun -l /bin/pwd
srun -l /bin/date

module purge 

date

sh ./training_scripts/gen_embedding_scratch.sh

date
