#!/bin/bash -l
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=12G
#SBATCH --time=02:00:00
#SBATCH --account=plghallugrant-gpu-a100
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output=job-%j.out
#SBATCH --error=job-%j.err
#SBATCH --gres=gpu:1

conda activate hallu

 
srun python ./scripts/get_answers.py --start 750 --end 1000 --exp_name=xsum --return_dict_in_gen

