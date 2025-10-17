"""
#!/bin/bash
#SBATCH --job-name=gen_data
#SBATCH --output=outputs/logs/gen_%j.out
#SBATCH --error=outputs/logs/gen_%j.err
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --partition=compute

# Load modules (adjust for your cluster)
module load python/3.10

# Activate virtual environment
source ~/venvs/lightsout/bin/activate

# Create log directory
mkdir -p outputs/logs

# Generate data
python scripts/01_generate_data.py \\
    --config configs/small.yaml \\
    --split both \\
    --output_dir outputs/data

echo "Data generation complete"
date
"""
