#!/bin/bash
#
#SBATCH --job-name=test
#
#SBATCH --time=10:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=16G
#SBATCH --partition=rondror

srun hostname
srun source /home/groups/rondror/software/jpaggi/schrodinger.ve/bin/activate
srun python glide_to_binding_affinities.py