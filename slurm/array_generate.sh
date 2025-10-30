#!/bin/bash
#SBATCH --job-name=gen_array
#SBATCH --output=outputs/logs/gen_d%a_%j.out
#SBATCH --error=outputs/logs/gen_d%a_%j.err
#SBATCH --array=3,5,7,9,11,15,21%3
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --partition=short

# Parallel data generation - one distance per job
# %3 means max 3 jobs running simultaneously

module load anaconda3/2024.06
source activate lightsout

DISTANCE=$SLURM_ARRAY_TASK_ID

echo \"Generating data for d=${DISTANCE}\"
date

python scripts/01_generate_data.py \
    --config configs/distance_d${DISTANCE}.yaml \
    --split both \
    --output_dir outputs/data_d${DISTANCE}

echo \"Complete for d=${DISTANCE}\"
date