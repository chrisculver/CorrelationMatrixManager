#!/bin/bash
#SBATCH -p short,defq -N 1
#SBATCH -e trash/slurm-out%j.log
#SBATCH -o trash/slurm-out%j.log
#SBATCH -t 1-00:00:00
#SBATCH -J compile

module load gcc/4.8.5

make compute_diagrams_gpu

wait
