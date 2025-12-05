import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ColorCodeGNN(nn.Module):
    """
    Graph Neural Network for color code decoding.
    Flexible architecture that supports:
    - Supervised learning (fixed input features)
    - RL training (variable input features with state augmentation)
    """

    def __init__(
        self,
        input_dim: int = 4,
        hidden_dim: int = 128,
        num_layers: int = 6,
        dropout: float = 0.2,
        # mode: str = "supervised",
    ):
        """
        Args:
            input_dim: Number of input features per node
                - Supervised: 4 (syndrome, degree, boundary, value)
                - RL: Variable (base features + accumulated_flip + syndrome)
            hidden_dim: Hidden dimension size
            num_layers: Number of message passing layers
            dropout: Dropout probability
            # mode: Training mode ('supervised' or 'rl')
        """
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Encoder: Projects input features to hidden dimension
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )

        # Message Passing Layers: Graph convolutions with residual connections
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        # Decoder: Projects hidden features to output (logits)
        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1),
        )

    def forward(self, data):
        """
        Forward pass through the GNN.
        Args:
            data: PyTorch Geometric Data object with:
                - x: Node features [num_nodes, input_dim]
                - edge_index: Edge connectivity [2, num_edges]
        Returns:
            logits: Output logits [num_nodes, 1] or [num_nodes]
            (hidden): Optional hidden features [num_nodes, hidden_dim]
        """
        x, edge_index = data.x, data.edge_index

        x = self.encoder(x)

        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x_new = conv(x, edge_index)  # Apply graph convolution
            x_new = bn(x_new)  # Batch normalization
            x_new = F.relu(x_new)  # Activation
            x_new = self.dropout(x_new)  # Dropout
            x = x + x_new  # Residual connection

        out = self.decoder(x)  # Decode: [num_nodes, hidden_dim] -> [num_nodes, 1]
        return out.squeeze(-1)  # Squeeze to [num_nodes] for easier use

    def predict_probs(self, data):
        """
        Get prediction probabilities (for RL sampling).

        Returns:
            probs: Sigmoid probabilities [num_nodes]
        """
        out = self.forward(data)
        return torch.sigmoid(out)

    def predict_binary(self, data, threshold: float = 0.5):
        """
        Get binary predictions (for evaluation).

        Returns:
            predictions: Binary values [num_nodes]
        """
        probs = self.predict_probs(data)
        return (probs > threshold).float()


# ============================================================================
# Model Utilities
# ============================================================================


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in model"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def initialize_weights(model: nn.Module):
    """Initialize model weights using Xavier initialization"""
    for m in model.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.ones_(m.weight)
            nn.init.zeros_(m.bias)
