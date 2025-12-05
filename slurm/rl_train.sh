#!/bin/bash
#SBATCH --job-name=rl_train
#SBATCH --output=outputs/logs/rl_train_%j.out
#SBATCH --error=outputs/logs/rl_train_%j.err
#SBATCH --time=8:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

# Load modules
module load anaconda3/2024.06 cuda/12.8.0

# Activate environment
source activate lightsout

echo "Start Reinforcement Training"
# Run RL fine-tuning
python scripts/03_rl_train_model.py \
    --config configs/rl.yaml \
    --pretrained_model outputs/models/rl_best_model.pt

echo "RL fine-tuning complete"
