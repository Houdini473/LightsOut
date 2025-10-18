#!/bin/bash
#SBATCH --job-name=train_gnn
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Load modules
module load anaconda3/2024.06
module load cuda/11.8

# Activate environment
source activate lightsout

# Create directories
mkdir -p outputs/logs outputs/models

# Run training
python scripts/02_train_model.py \\
    --config configs/small.yaml \\
    --data_dir outputs/data \\
    --output_dir outputs/models \\
    --device cuda

echo "Training complete"
date