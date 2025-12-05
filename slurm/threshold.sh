#!/bin/bash
#SBATCH --job-name=eval_gnn
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=2:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

module load anaconda3/2024.06

source activate lightsout

python scripts/05_calculate_threshold.py \
    --model outputs/models/best_model.pt \
    --config configs/large.yaml \
    --test_distances 3 5 7 9 11 \
    --test_samples 100 \
    --device cuda