"""
RL fine-tuning script
Fine-tunes a pre-trained supervised model using reinforcement learning
"""

from torch_geometric.loader import DataLoader
import torch
from pathlib import Path
import logging
import yaml
import argparse
from mqt.qecc.codes import HexagonalColorCode
import sys
from pathlib import Path
from time import time

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from models import ColorCodeGNN, count_parameters
from rl_trainer import RLColorCodeTrainer


def setup_logging(log_dir: Path):
    """Setup logging configuration"""
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / f"rl_training_{round(time())}.log"),
            # logging.StreamHandler(sys.stdout),
        ],
    )

def load_pretrained_model(checkpoint_path: str, load_rl_model: bool = False):
    """
    Load pretrained model for RL training.
    
    Args:
        checkpoint_path: Path to checkpoint
        load_rl_model: True = loading existing RL checkpoint (6 channels) - copy ALL weights
                      False = loading supervised checkpoint (4 channels) - expand to 6 channels
    
    Returns:
        model: ColorCodeGNN model with 6 input channels
        checkpoint: Full checkpoint dict
    """
    logger = logging.getLogger('model_loader')
    
    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location='cpu', weights_only=False)

    if load_rl_model:
        # RL checkpoint format: {'state_dict': ..., 'epoch': ..., etc}
        state_dict = checkpoint['state_dict']
        logger.info(f"Loaded RL checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    else:
        # Supervised checkpoint: just the state dict
        state_dict = checkpoint
        logger.info("Loaded supervised state dict")
    
    # Infer architecture from state dict
    # Get encoder input size
    encoder_weight_key = 'encoder.0.weight'  # First linear layer
    # if encoder_weight_key in state_dict:
    #     supervised_in_channels = state_dict[encoder_weight_key].size(1)
    #     hidden_channels = state_dict[encoder_weight_key].size(0)
    # else:
    #     raise ValueError("Cannot infer model architecture from checkpoint")

    input_dim = state_dict[encoder_weight_key].size(1)
    hidden_dim = state_dict[encoder_weight_key].size(0)
    num_layers = sum(1 for k in state_dict.keys() if 'conv_layers' in k) // 2
    
    logger.info(f"Detected architecture:")
    logger.info(f"  Input dimension: {input_dim}")
    logger.info(f"  Hidden dimension: {hidden_dim}")
    logger.info(f"  Num layers: {num_layers}")
    
    if load_rl_model:
        # Create model with expanded input for RL
        # RL needs: base_features (4) + accumulated_flip (1) + syndrome (1) = 6
        if input_dim != 6:
            raise ValueError(
                f"load_rl_model=True but model has {input_dim} channels (expected 6). "
                f"Use load_rl_model=False to convert from supervised model."
            )
        
        logger.info(f"Loading RL model with {input_dim} input channels")
        
        model = ColorCodeGNN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.2
        )
        
        model.load_state_dict(state_dict)
        logger.info("Loaded all weights from RL checkpoint")
        
    else:
        # Loading supervised model - should have 4 channels
        if input_dim != 4:
            raise ValueError(
                f"load_rl_model=False but model has {input_dim} channels (expected 4). "
                f"Use load_rl_model=True to load existing RL model."
            )

        logger.info("Converting supervised model (4 channels) to RL model (6 channels)")

        model = ColorCodeGNN(
            input_dim=input_dim + 2,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=0.2,
        )

        # Get model's state dict
        model_dict = model.state_dict()
        
        # Copy all weights EXCEPT encoder.0.weight and encoder.0.bias
        pretrained_dict = {
            k: v for k, v in state_dict.items()
            if k != encoder_weight_key and k != 'encoder.0.bias'
        }
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict, strict=False)

        # Copy encoder bias (same for both)
        if 'encoder.0.bias' in state_dict:
            with torch.no_grad():
                model.encoder[0].bias.data = state_dict['encoder.0.bias'].data

        # Handle encoder.0.weight: copy first 4 channels, initialize last 2
        with torch.no_grad():
            old_weight = state_dict[encoder_weight_key]  # [hidden, 4]
            new_weight = model.encoder[0].weight  # [hidden, 6]
            
            # Copy supervised weights for channels 0-3
            new_weight[:, :4] = old_weight
            
            # Randomly initialize channels 4-5 (accumulated_flip, syndrome_mismatch)
            torch.nn.init.xavier_uniform_(new_weight[:, 4:])
        
        logger.info("Transferred weights from supervised model")
        logger.info("Copied first 4 input channels")
        logger.info("Randomly initialized channels 5-6 for RL state")
    
    logger.info(f"Model parameters: {count_parameters(model):,}")
    
    return model, checkpoint


def main(args):
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Setup
    output_dir = Path(config["output_dir"])
    setup_logging(output_dir / "logs")
    logger = logging.getLogger("main")

    logger.info("Starting RL fine-tuning")
    logger.info(f"Config: {args.config}")

    # Load pretrained model
    logger.info(f"Loading pretrained model from {args.pretrained_model}")

    # model = ColorCodeGNN(
    #     input_dim=config["model"]["input_dim"],
    #     hidden_dim=config["model"]["hidden_dim"],
    #     num_layers=config["model"]["num_layers"]
    # )

    # checkpoint = torch.load(args.pretrained_model)
    # model.load_state_dict(checkpoint["model_state_dict"])

    logger.info("\nLoading pretrained model...")
    model, checkpoint = load_pretrained_model(
        args.pretrained_model,
        load_rl_model = True
    )

    if 'epoch' in checkpoint:
        logger.info(f"Pretrained for {checkpoint['epoch']} epochs")

    # Create code for RL trainer
    # currently looks like we're only training on one distance...
    code = HexagonalColorCode(config["rl"]["distances"][0])

    # Initialize RL trainer
    logger.info("Initializing RL trainer...")
    logger.info(f"  Training on distances: {config['rl']['distances']}")
    logger.info(f"  Max shots: {config['rl']['max_shots']}")
    logger.info(f"  Learning rate: {config['rl']['learning_rate']}")
    logger.info(f"  Gamma: {config['rl']['gamma']}")
    logger.info(f"  Success threshold: {config['rl']['success_threshold']}")
    logger.info(f"  Curriculum: Adaptive (data-driven progression)")

    rl_trainer = RLColorCodeTrainer(
        model=model,
        distances=config['rl']['distances'],
        max_shots=config["rl"]["max_shots"],
        learning_rate=config["rl"]["learning_rate"],
        gamma=config["rl"]["gamma"],
        success_threshold=config['rl']['success_threshold'],
        initial_error_rate=config['rl']['initial_error_rate'],
        max_error_rate=config['rl']['max_error_rate'],
        error_rate_step=config['rl']['error_rate_step'],
        # distance_window_size=config['rl']['distance_window_size']
    )

    # Train
    logger.info("Starting RL training...")
    save_path = output_dir / "models" / "rl_best_model.pt"
    save_path.parent.mkdir(parents=True, exist_ok=True)

    history = rl_trainer.train(
        error_rate=config['rl']['error_rate'],
        samples_per_epoch=config['rl']['samples_per_epoch'],
        num_epochs=config["rl"]["num_epochs"],
        log_interval=config["rl"]["log_interval"],
        save_path=str(save_path),
    )

    # Save training history
    history_path = output_dir / 'logs' / 'rl_history.pt'
    torch.save(history, history_path)

    logger.info("\n" + "="*60)
    logger.info("RL training complete!")
    logger.info("="*60)
    logger.info(f"Best model saved to {save_path}")
    logger.info(f"Training history saved to: {history_path}")
    logger.info(f"Final success rate: {history['success_rate'][-1]:.3f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RL for color code decoding")

    parser.add_argument(
        "--config",
        type=str,
        default="configs/rl.yaml",
        help="Path to config file"
    )
    parser.add_argument(
        "--pretrained_model",
        type=str,
        required=True,
        help="Path to pretrained model checkpoint",
    )

    args = parser.parse_args()

    main(args)
