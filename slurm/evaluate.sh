"""
#!/bin/bash
#SBATCH --job-name=eval_gnn
#SBATCH --output=outputs/logs/eval_%j.out
#SBATCH --error=outputs/logs/eval_%j.err
#SBATCH --time=12:00:00
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --mem=32G
#SBATCH --partition=gpu

# Load modules
module load python/3.10
module load cuda/11.8

# Activate environment
source ~/venvs/lightsout/bin/activate

# Create directories
mkdir -p outputs/logs outputs/results

# Evaluate on multiple distances (including extrapolation)
python scripts/03_evaluate_model.py \\
    --model outputs/models/best_model.pt \\
    --config configs/small.yaml \\
    --test_distances 3 5 7 9 11 13 \\
    --test_samples 1000 \\
    --device cuda

echo "Evaluation complete"
date
"""
