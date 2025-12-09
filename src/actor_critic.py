"""
Actor-Critic model for color code decoding.
More stable than REINFORCE for hard problems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data


class ActorCriticColorCode(nn.Module):
    """
    Actor-Critic architecture for color code decoding.
    
    Actor: Policy network (predicts which qubits to flip)
    Critic: Value network (estimates expected return from current state)
    
    The critic helps stabilize training by providing a baseline,
    reducing variance in policy gradient estimates.
    """
    
    def __init__(
        self,
        actor_model: nn.Module,
        hidden_dim: int = 256
    ):
        """
        Args:
            actor_model: Pretrained ColorCodeGNN to use as actor
            hidden_dim: Hidden dimension for critic network
        """
        super().__init__()
        
        # Actor: The policy network (your pretrained GNN)
        self.actor = actor_model
        
        # Critic: Value function network
        # Input: same 6 features as actor
        self.critic_encoder = nn.Sequential(
            nn.Linear(6, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(0.2)
        )
        
        # Critic uses graph structure like actor
        self.critic_conv = nn.ModuleList([
            GCNConv(hidden_dim, hidden_dim) for _ in range(2)
        ])
        
        # Value head: outputs single scalar (expected return)
        self.critic_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, data: Data, return_value: bool = True):
        """
        Forward pass through actor-critic.
        
        Args:
            data: PyG Data object with augmented features
            return_value: If True, also compute value estimate
            
        Returns:
            actor_logits: Action logits [n_qubits]
            value: State value estimate (scalar) if return_value=True
        """
        # Actor forward pass
        actor_logits = self.actor(data)
        
        if not return_value:
            return actor_logits
        
        # Critic forward pass
        x = data.x
        edge_index = data.edge_index
        
        # Encode features
        x = self.critic_encoder(x)  # [n_qubits, hidden_dim]
        
        # Graph convolutions with residual connections
        for conv in self.critic_conv:
            x_new = conv(x, edge_index)
            x_new = F.relu(x_new)
            x = x + x_new
        
        # Global pooling: get state representation
        # Average over all qubits to get single state vector
        state_embedding = x.mean(dim=0)  # [hidden_dim]
        
        # Value head: predict expected return
        value = self.critic_head(state_embedding)  # [1]
        value = value.squeeze()  # scalar
        
        return actor_logits, value
    
    def get_action_and_value(self, data: Data):
        """
        Sample action from policy and get value estimate.
        
        Args:
            data: PyG Data object
            
        Returns:
            action: Sampled binary action [n_qubits]
            log_prob: Log probability of action (scalar)
            value: Value estimate (scalar)
        """
        actor_logits, value = self.forward(data, return_value=True)
        
        # Sample action from Bernoulli distribution
        probs = torch.sigmoid(actor_logits)
        dist = torch.distributions.Bernoulli(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum()
        
        return action, log_prob, value