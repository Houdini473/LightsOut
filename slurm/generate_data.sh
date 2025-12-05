#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=outputs/logs/gen_%j.out
#SBATCH --error=outputs/logs/gen_%j.err
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=short

# Load modules 
module load anaconda3/2024.06

# Activate virtual environment
source activate lightsout

# Create log directory
mkdir -p outputs/logs

# Generate data
python -u scripts/01_generate_data.py \
 --config configs/large.yaml \
 --split train \
 --output_dir outputs/data

echo "Data generation complete"
date