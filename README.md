# LightsOut QECC: GNN Decoder for Quantum Color Codes

A Graph Neural Network (GNN) based decoder for triangular quantum color codes, trained via imitation learning from optimal MaxSAT solutions.

## Overview

This project implements a neural network decoder for quantum error correction using the LightsOut puzzle analogy described in [Berent et al. (2024)](https://arxiv.org/abs/2303.14237). The decoder learns to find minimum-weight corrections for syndrome patterns by imitating optimal MaxSAT solutions, achieving near-optimal accuracy with significantly faster inference.

**Key Features:**
- Multi-distance training (d=3 to d=21)
- Graph Neural Network with message passing
- Supervised learning from MaxSAT optimal solutions
- SLURM cluster optimization for large-scale training
- Comprehensive evaluation and visualization tools

## Background

Quantum error correction requires fast, accurate decoding of syndrome measurements to infer and correct errors. The MaxSAT-based decoder achieves ~10.1% threshold but takes 10-200ms per syndrome. This GNN decoder aims to maintain similar accuracy while reducing latency to ~1-10ms through learned pattern recognition.

## Installation

### On SLURM Cluster

```bash
# 1. SSH into cluster
ssh username@cluster.university.edu

# 2. Load Python module
module load python/3.10  # Adjust for your cluster

# 3. Create virtual environment
python3 -m venv ~/venvs/lightsout
source ~/venvs/lightsout/bin/activate

# 4. Install dependencies
pip install --upgrade pip setuptools wheel
pip install torch torch-geometric numpy matplotlib tqdm pyyaml mqt.qecc z3-solver

# For GPU support, install CUDA-compatible PyTorch
pip install torch --index-url https://download.pytorch.org/whl/cu118
```

### Local Development

```bash
git clone <your-repo>
cd lightsout_qecc
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Project Structure

```
lightsout_qecc/
├── src/                    # Source code
│   ├── datasets.py         # Dataset generation with MaxSAT labels
│   ├── models.py           # GNN architecture
│   ├── train.py            # Training loop
│   ├── evaluate.py         # Evaluation functions
│   └── utils.py            # Utilities (config, I/O)
├── scripts/                # Command-line scripts
│   ├── 01_generate_data.py
│   ├── 02_train_model.py
│   └── 03_evaluate_model.py
├── slurm/                  # SLURM job scripts
│   ├── generate_data.sh
│   ├── train.sh
│   ├── evaluate.sh
│   └── array_generate.sh
├── configs/                # Experiment configurations
│   ├── small.yaml          # d=[3,5,7]
│   ├── medium.yaml         # d=[3,5,7,9,11]
│   └── large.yaml          # d=[3,7,11,15,21]
└── outputs/                # Results and artifacts
    ├── data/               # Generated datasets
    ├── models/             # Trained models
    ├── logs/               # SLURM logs
    └── results/            # Evaluation results
```

## Quick Start

### Full Pipeline (SLURM)

```bash
# 1. Generate training/validation data
sbatch slurm/generate_data.sh

# 2. After data generation completes, train model
sbatch slurm/train.sh

# 3. After training completes, evaluate
sbatch slurm/evaluate.sh

# 4. Download results to local machine
scp -r username@cluster:~/lightsout_qecc/outputs/results ./
```

### Running Scripts Directly

```bash
# Generate data
python scripts/01_generate_data.py \
    --config configs/small.yaml \
    --split both \
    --output_dir outputs/data

# Train model
python scripts/02_train_model.py \
    --config configs/small.yaml \
    --data_dir outputs/data \
    --output_dir outputs/models \
    --device cuda

# Evaluate model
python scripts/03_evaluate_model.py \
    --model outputs/models/best_model.pt \
    --config configs/small.yaml \
    --test_distances 3 5 7 9 11 \
    --test_samples 1000
```

## Configuration

All experiment settings are controlled via YAML config files:

### Small Config (Development/Testing)
```yaml
data:
  distances: [3, 5, 7]
  train_samples_per_distance: 500
  val_samples_per_distance: 125
  error_rate: 0.1
  use_fast_decoder: false  # Use optimal MaxSAT

model:
  input_dim: 4
  hidden_dim: 128
  num_layers: 12
  dropout: 0.2

training:
  epochs: 200
  batch_size: 32
  lr: 0.001
  patience: 20
```

### Large Config (Research/Production)
```yaml
data:
  distances: [3, 7, 11, 15, 21]
  train_samples_per_distance: 200
  val_samples_per_distance: 50
  error_rate: 0.1
  use_fast_decoder: true  # Required for large distances

model:
  input_dim: 4
  hidden_dim: 256
  num_layers: 32
  dropout: 0.3

training:
  epochs: 300
  batch_size: 16
  lr: 0.001
  patience: 30
```

## Understanding the Approach

### The Decoding Problem

1. **Input:** Syndrome pattern (which stabilizer checks failed)
2. **Output:** Correction vector (which qubits to flip)
3. **Goal:** Find minimum-weight correction that satisfies syndrome

### GNN Architecture

The model uses a Graph Convolutional Network:

- **Nodes:** Qubits in the color code lattice
- **Edges:** Qubits that share a face (stabilizer check)
- **Node Features (4D):**
  - Syndrome values of 3 adjacent faces
  - Normalized degree (boundary detection)
- **Message Passing:** 6-32 layers depending on code distance
- **Output:** Binary prediction per qubit (flip/don't flip)

### Training Process

1. **Label Generation:** MaxSAT decoder provides optimal corrections
2. **Supervised Learning:** GNN learns to imitate MaxSAT decisions
3. **Loss:** Binary cross-entropy on per-qubit predictions
4. **Validation:** Early stopping based on validation loss

## Evaluation Metrics

### Valid Syndrome Rate
Percentage of predictions that satisfy $H\varepsilon = s$ (mod 2)
- Target: >85% for good decoder

### Optimal Weight Rate
Among valid predictions, percentage matching MaxSAT's Hamming weight
- Target: >70% for competitive performance

### Threshold (requires separate calculation)
Critical error rate below which error correction helps
- MaxSAT achieves: 10.1%
- Target: >9% for competitive decoder

## Resource Requirements

| Configuration | Data Gen | Training | GPU | Memory |
|--------------|----------|----------|-----|---------|
| Small (d≤7) | 12h | 6h | Optional | 32GB |
| Medium (d≤11) | 24-48h | 12h | Recommended | 64GB |
| Large (d≤21) | 48-72h | 24h | Required | 128GB |

**Notes:**
- Data generation is CPU-intensive (MaxSAT solving)
- Training benefits significantly from GPU
- Use `use_fast_decoder: true` for d>11 (Union-Find instead of MaxSAT)

## SLURM Job Management

### Check Job Status

```bash
# View your jobs
squeue -u $USER

# Detailed job info
scontrol show job JOBID

# Resource usage
sacct -j JOBID --format=JobID,Elapsed,MaxRSS,State
```

### Cancel Jobs

```bash
# Cancel specific job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# Cancel by name
scancel --name=gen_data
```

### Monitor Logs

```bash
# Live monitoring
tail -f outputs/logs/train_*.out

# Check errors
tail -f outputs/logs/train_*.err

# View all logs
ls -lh outputs/logs/
```

## Advanced Usage

### Parallel Data Generation (Faster)

Use SLURM array jobs to generate each distance in parallel:

```bash
# Generate d=[3,5,7,9,11] simultaneously
sbatch slurm/array_generate.sh

# Monitor all jobs
watch -n 5 'squeue -u $USER'
```

### Resume Training from Checkpoint

```bash
python scripts/02_train_model.py \
    --config configs/small.yaml \
    --resume outputs/models/checkpoint_epoch100.pt
```

### Evaluate on Extrapolation Distances

Test how well the model generalizes to unseen distances:

```bash
# Trained on d=[3,5,7], test on d=[9,11,13]
python scripts/03_evaluate_model.py \
    --model outputs/models/best_model.pt \
    --config configs/small.yaml \
    --test_distances 9 11 13
```

## Expected Results

### Valid Syndrome Rate
- **On training distances:** 85-95%
- **Extrapolation (+2 distance):** 60-80%
- **Extrapolation (+4 distance):** 40-60%

### Speed Comparison
- **MaxSAT:** 50-200ms (varies with syndrome weight)
- **GNN (CPU):** 5-10ms (consistent)
- **GNN (GPU):** 0.5-2ms (consistent)

### Threshold (d=5, trained on MaxSAT labels)
- **Target:** 9-10%
- **Expected:** 8.5-9.5%
- **MaxSAT benchmark:** 10.1%

## Troubleshooting

### Data Generation Takes Forever

**Problem:** MaxSAT is slow for large distances

**Solution:**
```yaml
# In config file, set:
use_fast_decoder: true  # Uses Union-Find (fast but suboptimal)
```

### Model Not Learning (High Loss)

**Check:**
1. Verify input_dim matches feature dimensions
2. Ensure datasets loaded correctly
3. Check learning rate (try 0.01 for faster convergence)
4. Increase model capacity (hidden_dim: 256)

### Out of Memory During Training

**Solutions:**
```yaml
# Reduce batch size
batch_size: 16  # or 8

# Or request more memory in SLURM script
#SBATCH --mem=128G
```

### Poor Generalization to Other Distances

**Expected behavior** - GNNs don't generalize well across distances

**Improvements:**
- Train on more distances
- Use larger num_layers (32 for d=21)
- Increase training data
- Consider multi-task learning

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{berent2024decoding,
  title={Decoding quantum color codes with MaxSAT},
  author={Berent, Lucas and Burgholzer, Lukas and Derks, Peter-Jan HS and Eisert, Jens and Wille, Robert},
  journal={Quantum},
  year={2024},
  publisher={Veritas}
}
```

## Repository Structure for Git

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial commit: GNN decoder for color codes"

# .gitignore recommended entries:
outputs/
*.pkl
*.pt
*.pyc
__pycache__/
venv/
*.out
*.err
.ipynb_checkpoints/
```

## Support

For issues related to:
- **MQT QECC package:** https://github.com/cda-tum/mqt-qecc
- **This implementation:** [Your contact/issues page]

## License

[Your chosen license - MIT recommended for academic code]

## Acknowledgments

This implementation is based on the MaxSAT decoder framework developed by the Munich Quantum Toolkit (MQT) team. The GNN architecture adapts Graph Convolutional Networks for the quantum error correction decoding problem.

---

**Status:** Research prototype  
**Tested on:** SLURM clusters with CUDA 11.8, Python 3.10  
**Last updated:** 2025-10-17
