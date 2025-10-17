import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


class ColorCodeGNN(nn.Module):
    """Graph Neural Network for color code decoding"""

    def __init__(self, input_dim=4, hidden_dim=128, num_layers=6, dropout=0.2):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout)
        )

        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layers):
            self.conv_layers.append(GCNConv(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))

        self.dropout = nn.Dropout(dropout)

        self.decoder = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.ReLU(),
            nn.Linear(hidden_dim // 4, 1)
        )

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.encoder(x)

        for conv, bn in zip(self.conv_layers, self.batch_norms):
            x_new = conv(x, edge_index)
            x_new = bn(x_new)
            x_new = F.relu(x_new)
            x_new = self.dropout(x_new)
            x = x + x_new

        out = self.decoder(x)
        return out.squeeze(-1)
