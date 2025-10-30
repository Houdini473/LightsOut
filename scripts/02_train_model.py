#!/usr/bin/env python3
"""
Train GNN model

Usage:
    python scripts/02_train_model.py --config configs/small.yaml
"""

import argparse
import sys
from pathlib import Path
import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Dataset

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import ColorCodeGNN
from train import train_model
from utils import load_config, load_pickle, save_json, load_pickles
from datasets import MultiDistanceDataset

def main():
    parser = argparse.ArgumentParser(description='Train GNN decoder')
    parser.add_argument('--config', type=str, required=True)
    parser.add_argument('--data_dir', type=str, default='outputs/data')
    parser.add_argument('--output_dir', type=str, default='outputs/models')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'])
    parser.add_argument('--resume', type=str, default=None,
                       help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    print(f'cuda avail: {torch.cuda.is_available()}')

    # Setup device
    if args.device == 'cuda' and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = torch.device('cpu')
    else:
        device = torch.device(args.device)

    print(f"Using device: {device}")
    print(f"Config: {args.config}\n")

    # Load datasets
    data_dir = Path(args.data_dir)
    print("Loading datasets...")

    class SimpleDataset(Dataset):
        def __init__(self, data_list):
            self.data_list = data_list
        def __len__(self):
            return len(self.data_list)
        def __getitem__(self, idx):
            return self.data_list[idx]

    train_data_list = load_pickles(data_dir / "train")
    print(f"Train data w length = {len(train_data_list)} loaded")
    train_data = MultiDistanceDataset(distances = [], n_samples_per_distance =1, data_list = train_data_list)

    val_data = load_pickle(data_dir / 'val.pkl')
    print(f"Validation data w length = {len(val_data)} loaded")

    # train_dataset = SimpleDataset(train_data['data_list'] if isinstance(train_data, dict) else train_data.data_list)
    # val_dataset = SimpleDataset(val_data['data_list'] if isinstance(val_data, dict) else val_data.data_list)

    train_dataset = SimpleDataset(train_data)
    val_dataset = SimpleDataset(val_data)

    print(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}\n")

    # Create loaders
    train_config = config['training']
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_config['batch_size'],
        shuffle=True,
        num_workers=0  # Important for SLURM
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=train_config['batch_size'],
        shuffle=False,
        num_workers=0
    )

    # Create model
    model_config = config['model']
    model = ColorCodeGNN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )

    print(f"Model architecture:")
    print(f"  Input dim: {model_config['input_dim']}")
    print(f"  Hidden dim: {model_config['hidden_dim']}")
    print(f"  Num layers: {model_config['num_layers']}")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}\n")

    # Resume from checkpoint if provided
    if args.resume:
        print(f"Resuming from {args.resume}")
        model.load_state_dict(torch.load(args.resume, map_location=device))

    # Setup output
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / 'best_model.pt'

    # Train
    print("Starting training...\n")
    history = train_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=train_config['epochs'],
        lr=train_config['lr'],
        device=device,
        save_path=model_path,
        max_patience=train_config['patience'],
        log_interval=1
    )

    # Save history
    save_json(history, output_dir / 'training_history.json')

    # Save final model state
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'history': history
    }, output_dir / 'final_checkpoint.pt')

    print(f"\n✓ Training complete")
    print(f"✓ Best model saved to {model_path}")
    print(f"✓ History saved to {output_dir / 'training_history.json'}")


if __name__ == '__main__':
    main()
