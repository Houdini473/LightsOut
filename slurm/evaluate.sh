#!/bin/bash
#SBATCH --job-name=eval_gnn
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=1:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Load modules
module load anaconda3/2024.06 cuda/12.8.0

# Activate environment
source activate lightsout

# Create directories
mkdir -p outputs/logs outputs/results

# Evaluate on multiple distances (including extrapolation)
python scripts/03_evaluate_model.py \
    --model outputs/models/best_model.pt \
    --config configs/large.yaml \
    --test_distances 3 5 7 9 11 13 21\
    --test_samples 100 \
    --device cuda

echo "Evaluation complete"
date