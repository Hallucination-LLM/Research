#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem-per-cpu=1G
#SBATCH --time=00:30:00
#SBATCH --account=plghallugrant-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
# SBATCH --gres=gpu:1

conda activate hallu

 
srun python ./scripts/get_dataset.py --start 750 --end 1000
