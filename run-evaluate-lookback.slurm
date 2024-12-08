#!/bin/bash
#SBATCH --job-name=evaluate-lookback
#SBATCH --time=08:00:00
#SBATCH --partition=plgrid-gpu-a100
#SBATCH --output logs/evaluate-lookback-%J.txt
#SBATCH --mem=10G
#SBATCH --gpus=1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate /net/tscratch/people/plgjaneliasz/conda_research
srun python scripts/evaluate_lookback.py --data_file data/xsum_df.parquet --resps_file data/xsum-lookback-decoding_combined.json --exp_name lookback_llama2_xsum
conda deactivate