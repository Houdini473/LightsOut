#!/bin/bash
#SBATCH --job-name=train_gnn
#SBATCH --output=outputs/logs/train_%j.out
#SBATCH --error=outputs/logs/train_%j.err
#SBATCH --time=8:00:00
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --partition=gpu

# Load modules
module load anaconda3/2024.06 cuda/12.8.0

# Activate environment
source activate lightsout

# Create directories
mkdir -p outputs/logs outputs/models

echo "Starting training"
# Run training
python -u scripts/02_train_model.py \
    --config configs/large.yaml \
    --data_dir outputs/data \
    --output_dir outputs/models \
    --device cuda

echo "Training complete"
date