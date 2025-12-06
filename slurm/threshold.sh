#!/bin/bash
#SBATCH --job-name=eval_gnn
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

module load anaconda3/2024.06 cuda/12.8.0

source activate lightsout

python scripts/05_calculate_threshold.py \
    --model outputs/models/rl_best_model.pt \
    --config configs/evaluate.yaml \