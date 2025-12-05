#!/usr/bin/env python3
"""
Evaluate trained model on multiple distances and physical error rates

Usage:
    python scripts/03_evaluate_model.py --model outputs/models/best_model.pt --config configs/small.yaml
    python scripts/03_evaluate_model.py --model outputs/models/best_model.pt --config configs/small.yaml --test_distances 3 5 7 9 11
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import ColorCodeGNN
from evaluate import evaluate_single_distance
from utils import load_config, save_json
import numpy as np


def main():
    parser = argparse.ArgumentParser(description='Evaluate GNN decoder')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model')
    parser.add_argument('--config', type=str, required=True,
                       help='Path to config file')
    parser.add_argument('--test_distances', type=int, nargs='+', default=None,
                       help='Distances to test on (default: from config)')
    parser.add_argument('--test_samples', type=int, default=1000,
                       help='Number of test samples per distance')
    parser.add_argument('--output_dir', type=str, default='outputs/results')
    parser.add_argument('--device', type=str, default='cuda')
    args = parser.parse_args()

    # Load config
    config = load_config(args.config)

    # Setup device
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")

    # Load model
    model_config = config['model']
    model = ColorCodeGNN(
        input_dim=model_config['input_dim'],
        hidden_dim=model_config['hidden_dim'],
        num_layers=model_config['num_layers'],
        dropout=model_config['dropout']
    )

    model.load_state_dict(torch.load(args.model, map_location=device))
    model = model.to(device)
    print("✓ Model loaded\n")

    # Determine test distances
    if args.test_distances is None:
        test_distances = config['data']['distances']
        train_distances = set(test_distances)
    else:
        test_distances = args.test_distances
        train_distances = set(config['data']['distances'])

    print(f"Training distances: {sorted(train_distances)}")
    print(f"Testing distances: {sorted(test_distances)}\n")

    # Evaluate on each distance
    all_results = {}

    for distance in sorted(test_distances):
        status = "TRAINED" if distance in train_distances else "EXTRAPOLATION"
        print(f"\n{'='*60}")
        print(f"Distance {distance} [{status}]")
        print(f"{'='*60}")

        results = evaluate_single_distance(
            model=model,
            distance=distance,
            error_rate=config['data']['error_rate'],
            test_samples=args.test_samples,
            device=device,
            verbose=True
        )

        # Print summary
        total = results['total_samples']
        valid = results['valid_syndromes']

        print(f"\nResults for d={distance}:")
        print(f"  Total samples: {total}")
        print(f"  Valid syndrome rate: {valid/total:.2%}")

        if valid > 0:
            print(f"  Optimal weight rate: {results['optimal_weight']/valid:.2%}")
            print(f"  Exact match rate: {results['exact_matches']/valid:.2%}")
            print(f"  Avg GNN weight: {np.mean(results['gnn_weights']):.2f}")
            print(f"  Avg MaxSAT weight: {np.mean(results['mqt_weights']):.2f}")
        else:
            print("  ✗ No valid predictions")

        # Clean results for JSON serialization
        all_results[str(distance)] = {
            'distance': distance,
            'trained_on': distance in train_distances,
            'total_samples': total,
            'valid_syndromes': valid,
            'valid_rate': valid/total if total > 0 else 0,
            'optimal_weight': results['optimal_weight'],
            'exact_matches': results['exact_matches'],
            'avg_gnn_weight': float(np.mean(results['gnn_weights'])) if results['gnn_weights'] else None,
            'avg_mqt_weight': float(np.mean(results['mqt_weights'])) if results['mqt_weights'] else None
        }

    # Save results
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / 'evaluation_results.json'
    save_json(all_results, output_file)

    print(f"\n{'='*60}")
    print(f"✓ Results saved to {output_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()
