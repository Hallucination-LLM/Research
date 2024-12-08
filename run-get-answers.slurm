#!/bin/bash
#SBATCH --job-name=get-answers
#SBATCH --time=23:59:59
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output logs/get-answers-%J.txt
#SBATCH --mem=40G
#SBATCH --gpus=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgjaneliasz/conda_research
srun python scripts/get_answers.py --exp_name llama2_our_ds_shred_6 --return_dict_in_gen --start 6000 --end 7000
conda deactivate