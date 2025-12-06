#!/usr/bin/env python3
"""
Evaluation script for RL-trained color code decoder.
Tests model performance across distances and error rates.
"""

import argparse
import yaml
import logging
from typing import List, Tuple, Dict 
import sys
from pathlib import Path
import torch
from time import time
import matplotlib.pyplot as plt
from tqdm import tqdm
# from mqt.qecc.codes import HexagonalColorCode

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import ColorCodeGNN, count_parameters
from rl_trainer import RLColorCodeTrainer

starttime = round(time())



def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)
    
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    file_handler = logging.FileHandler(log_dir / f'evaluation{starttime}.log')
    file_handler.setFormatter(formatter)
    
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    
    logging.basicConfig(
        level=logging.INFO,
        handlers=[file_handler, console_handler]
    )

def load_model(checkpoint_path: str, device: str = 'cuda'):
    """Load trained model from checkpoint"""
    logger = logging.getLogger('evaluator')
    
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)
    
    # Get state dict
    if isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        state_dict = checkpoint
    
    # Infer architecture
    encoder_weight_key = 'encoder.0.weight'
    input_channels = state_dict[encoder_weight_key].size(1)
    hidden_channels = state_dict[encoder_weight_key].size(0)
    num_layers = sum(1 for k in state_dict.keys() if 'conv_layers' in k) // 2
    
    logger.info(f"Loading model:")
    logger.info(f"  Input channels: {input_channels}")
    logger.info(f"  Hidden channels: {hidden_channels}")
    logger.info(f"  Num layers: {num_layers}")
    
    # Create model
    model = ColorCodeGNN(
        input_dim=input_channels,
        hidden_dim=hidden_channels,
        num_layers=num_layers,
        dropout=0.2
    )
    
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    logger.info(f"Model loaded successfully ({count_parameters(model):,} parameters)")
    
    return model


def evaluate_model(
    model: torch.nn.Module,
    distances: List[int],
    error_rates: List[float],
    n_samples: int = 1000,
    max_shots: int = 10,
    device: str = 'cuda'
) -> Dict[int, Dict[str, List[float]]]:
    """
    Evaluate model performance across distances and error rates.
    
    Args:
        model: Trained RL model
        distances: List of code distances to test
        error_rates: List of physical error rates to test
        n_samples: Number of samples per (distance, error_rate) pair
        max_shots: Maximum correction attempts
        device: Device to run on
        
    Returns:
        results: Dict[distance -> {'error_rates': [...], 'failure_rates': [...]}]
    """
    logger = logging.getLogger('evaluator')
    
    # Create RL trainer for evaluation (uses its helper functions)
    rl_trainer = RLColorCodeTrainer(
        model=model,
        distances=distances,
        max_shots=max_shots,
        learning_rate=1e-5,  # Not used in eval
        device=device
    )
    
    results = {}
    
    for distance in distances:
        logger.info(f"\nEvaluating distance {distance}...")
        
        error_rate_list = []
        failure_rate_list = []
        
        for error_rate in error_rates:
            logger.info(f"  Error rate: {error_rate:.4f}")
            
            failures = 0
            logical_errors = 0
            successes = 0
            
            pbar = tqdm(
                range(n_samples), 
                desc=f"d={distance}, p={error_rate:.3f}",
                file=sys.stdout
            )
            
            with torch.no_grad():  # No gradients needed for evaluation
                for sample_idx in pbar:
                    # Generate sample
                    data, error, syndrome = rl_trainer.generate_sample(distance, error_rate)
                    
                    # Get H matrix
                    H = rl_trainer.get_H_matrix(distance)
                    n_qubits = data.x.size(0)
                    
                    # Multi-shot decoding (match training exactly)
                    accumulated_flip = torch.zeros(n_qubits, device=device)
                    
                    decoded = False
                    for shot in range(1, max_shots + 1):
                        # Augment features (same as training)
                        augmented_data = rl_trainer.augment_features(
                            data, accumulated_flip, syndrome, H
                        )
                        augmented_data = augmented_data.to(device)
                        
                        # Get action from model
                        logits = model(augmented_data)
                        probs = torch.sigmoid(logits)
                        
                        # EVALUATION: Use greedy decoding (argmax) instead of sampling
                        # This gives deterministic, best-case performance
                        action = (probs > 0.5).float()
                        
                        # Apply action
                        accumulated_flip += action
                        
                        # Check if syndrome matches
                        predicted_syndrome = rl_trainer.compute_syndrome(
                            accumulated_flip % 2, H
                        )
                        syndrome_match = torch.all(predicted_syndrome == syndrome)
                        
                        if syndrome_match:
                            # Check residual
                            residual = (error + accumulated_flip) % 2
                            is_logical = rl_trainer.is_logical_operator(residual, H, distance)
                            
                            if not is_logical:
                                # Perfect correction!
                                successes += 1
                                decoded = True
                            else:
                                # Logical error
                                logical_errors += 1
                                decoded = True
                            
                            break
                    
                    if not decoded:
                        # Ran out of shots
                        failures += 1
                    
                    # Update progress
                    if sample_idx % 10 == 0:
                        total_failures = failures + logical_errors
                        pbar.set_postfix({
                            'success': f"{successes/(sample_idx+1):.3f}",
                            'timeout': failures,
                            'logical': logical_errors
                        })
            
            # Compute failure rate (both failures and logical errors count as failures)
            total_failures = failures + logical_errors
            failure_rate = total_failures / n_samples
            
            logger.info(f"    Failure rate: {failure_rate:.4f} "
                       f"({failures} timeouts, {logical_errors} logical errors)")
            
            error_rate_list.append(error_rate)
            failure_rate_list.append(failure_rate)
        
        results[distance] = {
            'error_rates': error_rate_list,
            'failure_rates': failure_rate_list
        }
    
    return results

def plot_results(
    results: Dict[int, Dict[str, List[float]]],
    save_path: str,
    title: str = "RL Decoder Performance"
):
    """
    Plot failure rate vs error rate for different distances.
    
    Args:
        results: Results from evaluate_model()
        save_path: Path to save plot
        title: Plot title
    """
    plt.figure(figsize=(10, 6))
    
    # Plot each distance
    for distance in sorted(results.keys()):
        error_rates = results[distance]['error_rates']
        failure_rates = results[distance]['failure_rates']
        
        plt.plot(
            error_rates, 
            failure_rates, 
            marker='o',
            label=f'd={distance}',
            linewidth=2,
            markersize=6
        )
    
    plt.xlabel('Physical Error Rate', fontsize=12)
    plt.ylabel('Logical Failure Rate', fontsize=12)
    plt.title(title, fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.yscale('log')  # Log scale for failure rate
    
    # Save
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Plot saved to {save_path}")


def main(args):
    # Load config
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    # Setup
    output_dir = Path(config['output_dir'])
    setup_logging(output_dir / 'logs')
    logger = logging.getLogger('main')
    
    logger.info("="*60)
    logger.info("RL Model Evaluation")
    logger.info("="*60)
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")
    
    # Load model
    logger.info("\nLoading model...")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")
    
    model = load_model(args.model, device=device)
    
    # Get evaluation parameters
    eval_config = config['evaluation']
    distances = eval_config['distances']
    error_rates = eval_config['error_rates']
    n_samples = eval_config['n_samples']
    max_shots = eval_config.get('max_shots', 10)
    
    logger.info("\nEvaluation parameters:")
    logger.info(f"  Distances: {distances}")
    logger.info(f"  Error rates: {error_rates}")
    logger.info(f"  Samples per point: {n_samples}")
    logger.info(f"  Max shots: {max_shots}")
    
    # Evaluate
    logger.info("\n" + "="*60)
    logger.info("Starting evaluation...")
    logger.info("="*60)
    
    results = evaluate_model(
        model=model,
        distances=distances,
        error_rates=error_rates,
        n_samples=n_samples,
        max_shots=max_shots,
        device=device
    )
    
    # Save results
    results_path = output_dir / 'results' / 'evaluation_results.pt'
    results_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(results, results_path)
    logger.info(f"\nResults saved to {results_path}")
    
    # Plot results
    plot_path = output_dir / 'results' / f'failure_rate_vs_error_rate{starttime}.png'
    plot_results(results, str(plot_path), title=eval_config.get('plot_title', 'RL Decoder Performance'))
    logger.info(f"Plot saved to {plot_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Summary")
    logger.info("="*60)
    for distance in sorted(results.keys()):
        logger.info(f"\nDistance {distance}:")
        for p, fail_rate in zip(results[distance]['error_rates'], 
                                results[distance]['failure_rates']):
            logger.info(f"  p={p:.4f}: failure rate = {fail_rate:.4f}")
    
    logger.info("\nEvaluation complete!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RL model')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='configs/evaluate.yaml',
        help='Path to evaluation config'
    )
    
    args = parser.parse_args()
    main(args)