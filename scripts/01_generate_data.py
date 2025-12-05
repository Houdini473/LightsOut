#!/usr/bin/env python3
"""
Generate training/validation datasets

Usage:
    python scripts/01_generate_data.py --config configs/large.yaml --split train
    python scripts/01_generate_data.py --config configs/large.yaml --split val
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from datasets import MultiDistanceDataset
from utils import load_config, save_pickle


def main():
    parser = argparse.ArgumentParser(description='Generate color code datasets')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config YAML file')
    parser.add_argument('--split', type=str, required=True,
                       choices=['train', 'val', 'both'],
                       help='Which split to generate')
    parser.add_argument('--output_dir', type=str, default='outputs/data',
                       help='Output directory for datasets')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)
    data_config = config['data']

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate training data
    if args.split in ['train', 'both']:
        print("="*70)
        print("GENERATING TRAINING DATA")
        print("="*70)

        train_dataset = MultiDistanceDataset(
            distances=data_config['distances'],
            n_samples_per_distance=data_config['train_samples_per_distance'],
            error_rate=data_config['error_rate'],
            seed=data_config['train_seed'],
            use_fast_decoder=data_config['use_fast_decoder'],
            verbose=True,
            output_path=output_dir / f'train'
        )

        # train_path = output_dir / 'train.pkl'
        # save_pickle(train_dataset, train_path)
        # print(f"\n✓ Saved {len(train_dataset)} training samples to {train_path}")

    # Generate validation data
    if args.split in ['val', 'both']:
        print("\n" + "="*70)
        print("GENERATING VALIDATION DATA")
        print("="*70)

        distances = data_config['distances']

        val_dataset = MultiDistanceDataset(
            distances=data_config['distances'],
            n_samples_per_distance=data_config['val_samples_per_distance'],
            error_rate=data_config['error_rate'],
            seed=data_config['val_seed'],
            use_fast_decoder=data_config['use_fast_decoder'],
            verbose=True,
            output_path=output_dir / f'val'
        )

        # val_path = output_dir / 'val.pkl'
        # save_pickle(val_dataset, val_path)
        # print(f"\n✓ Saved {len(val_dataset)} validation samples to {val_path}")


if __name__ == '__main__':
    main()
