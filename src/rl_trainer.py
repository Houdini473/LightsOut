import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
import sys
import json
from typing import Any, Tuple, List, Optional, Dict
import logging
from torch_geometric.data import Data

from mqt.qecc.codes import HexagonalColorCode

sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from actor_critic import ActorCriticColorCode

class RLColorCodeTrainer:
    """RL trainer with actor-critic algorithm with multi-shot correction."""

    def __init__(
        self,
        model: nn.Module,
        distances: List[int] = [],
        max_shots: int = 10,
        learning_rate: float = 1e-5, 
        learning_rate_critic: float = 1e-4,
        gamma: float = 0.99,
        success_threshold: float = 0.7,
        # distance_window_size: int = 4,
        initial_error_rate: float = 0.04, 
        max_error_rate: float = 0.12,
        error_rate_step: float = 0.01,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.max_shots = max_shots
        self.gamma = gamma
        self.device = device
        self.success_threshold = success_threshold
        # self.distance_window_size = distance_window_size
        self.distances = sorted(distances)

        self.current_error_rate = initial_error_rate
        self.max_error_rate = max_error_rate 
        self.error_rate_step = error_rate_step 


        self.logger = logging.getLogger('RLTrainer')

        self.models = {}
        self.optimizers = {}
        for d in self.distances: 
            ac = ActorCriticColorCode(model).to(device)
            actor_opt = optim.Adam(ac.actor.parameters(), lr=learning_rate)
            critic_opt = optim.Adam([p for n, p in ac.named_parameters() if 'critic' in n], lr=learning_rate_critic)
            self.models[d] = ac
            self.optimizers[d] = (actor_opt, critic_opt)
        self.logger.info(f'Created {len(distances)} Actor-Critic models')

        self.H_cache = {}
        for d in self.distances:
            code = HexagonalColorCode(d)
            self.H_cache[f'code_{d}'] = code

            H_matrix = code.H.astype(np.float32)
            self.H_cache[d] = torch.from_numpy(H_matrix).to(self.device)

            self.H_cache[f'edge_index_{d}'] = self._build_edge_index(code)
            self.logger.info(f"  d={d}: H shape {self.H_cache[d].shape}, "
                           f"{self.H_cache[f'edge_index_{d}'].shape[1]} edges")

        self.optimizer = optim.Adam(model.parameters(), lr=learning_rate)

        # Tracking
        self.current_distances = []
        self.distance_success_rates = {}
        self.saved_log_probs = []
        self.saved_values = []
        self.rewards = []



    def _build_edge_index(self, code):
        """
        Build edge connectivity from code structure.
        Qubits are connected if they share a face.
        """
        edges = set()
        for face_id, qubits in code.faces_to_qubits.items():
            for i, q1 in enumerate(qubits):
                for q2 in qubits[i+1:]:
                    edges.add((q1, q2))
                    edges.add((q2, q1))

        if not edges:
            return torch.zeros((2, 0), dtype=torch.long, device=self.device)

        edge_index = torch.tensor(list(edges), dtype=torch.long, device=self.device).t()
        return edge_index


    def get_H_matrix(self, distance: int) -> torch.Tensor:
        """Get cached H matrix for given distance."""
        return self.H_cache[distance]

    def get_code(self, distance: int) -> HexagonalColorCode:
        """Get cached code object for given distance."""
        return self.H_cache[f'code_{distance}']

    def get_edge_index(self, distance: int) -> torch.Tensor:
        """Get cached edge index for given distance."""
        return self.H_cache[f'edge_index_{distance}']

    def _create_node_features(self, syndrome, code):
        """
        Create 4D feature vector per qubit.
        [face_1_syndrome, face_2_syndrome, face_3_syndrome, degree_normalized]
        """
        features = []
        n_qubits = code.n
        for qubit_idx in range(n_qubits):
            faces = code.qubits_to_faces.get(qubit_idx, [])
            syn_vals = [syndrome[f] for f in faces]
            syn_vals += [0] * (3 - len(syn_vals))  # Pad to length 3
            degree_norm = len(faces) / 3.0
            feat = syn_vals[:3] + [degree_norm]
            features.append(feat)
        return torch.tensor(features, dtype=torch.float32, device=self.device)

    def generate_sample(self, distance: int, error_rate: float):
        """
        Generate a single training sample on-the-fly.

        Args:
            distance: Code distance
            error_rate: Physical error rate

        Returns:
            data: PyG Data object
            error: Binary error pattern [n_qubits]
            syndrome: Target syndrome [n_checks]
        """
        # Get cached code and matrices
        code = self.get_code(distance)
        H = self.get_H_matrix(distance)
        edge_index = self.get_edge_index(distance)

        n_qubits = code.n

        # Sample error
        error = torch.bernoulli(
            torch.full((n_qubits,), error_rate, device=self.device)
        )

        # Compute syndrome
        syndrome = (H @ error.float()) % 2

        # Create node features
        node_features = self._create_node_features(syndrome.cpu().numpy(), code)

        # Create PyG Data object
        data = Data(x=node_features, edge_index=edge_index)

        return data, error, syndrome


    def augment_features(
        self,
        data: Data,
        accumulated_flip: torch.Tensor,
        syndrome: torch.Tensor,
        H: torch.Tensor
        ) -> Data:
        """
        Augment node features with RL state information.

        Args:
            data: Original PyG Data object with base features
            accumulated_flip_count: Total number of times each qubit flipped [n_qubits]
            accumulated_flip_mod2: Current correction state mod 2 [n_qubits]
            target_syndrome: Syndrome to match [n_checks]
            H: Parity check matrix [n_checks, n_qubits]

        Returns:
            augmented_data: New Data object with augmented features
        """
        # Base features from data
        base_features = data.x  # [n_qubits, 4]

        # Normalize flip count (0 to 1 scale)
        flip_count_normalized = torch.clamp(
            accumulated_flip / self.max_shots,
            max=1.0
        ).unsqueeze(1)  # [n_qubits, 1]

        # Compute current syndrome using mod 2 flips
        current_syndrome = self.compute_syndrome(accumulated_flip % 2, H)  # [n_checks]

        # Map syndrome to qubit features
        # For each qubit, compute how many adjacent checks are violated
        syndrome_feature = self._map_syndrome_to_qubits(
            syndrome, current_syndrome, H
        )  # [n_qubits, 1]

        # Concatenate all features
        augmented_features = torch.cat([
            base_features,              # [n_qubits, 4] - original features
            flip_count_normalized,      # [n_qubits, 1] - how many times flipped
            syndrome_feature            # [n_qubits, 1] - syndrome mismatch
        ], dim=1)  # [n_qubits, 6]

        # Create new data object
        augmented_data = Data(
            x=augmented_features,
            edge_index=data.edge_index,
            batch=data.batch if hasattr(data, 'batch') else None
        )

        return augmented_data

    def _map_syndrome_to_qubits(
        self,
        target_syndrome: torch.Tensor,
        current_syndrome: torch.Tensor,
        H: torch.Tensor
    ) -> torch.Tensor:
        """
        Map syndrome information to per-qubit features.

        For each qubit, compute the XOR of target and current syndromes
        for all checks that involve that qubit.

        Args:
            target_syndrome: Target syndrome [n_checks]
            current_syndrome: Current syndrome [n_checks]
            H: Parity check matrix [n_checks, n_qubits]

        Returns:
            syndrome_features: [n_qubits, 1]
        """
        # Compute syndrome difference (what needs to be fixed)
        syndrome_diff = (target_syndrome - current_syndrome) % 2

        # Map to qubits: for each qubit, sum over checks it participates in
        # H^T maps from checks to qubits
        syndrome_features = (H.T @ syndrome_diff.float()) % 2

        return syndrome_features.unsqueeze(1)  # [n_qubits, 1]

    def get_action(
        self,
        data,
        accumulated_flip: torch.Tensor,
        syndrome: torch.Tensor,
        H: torch.Tensor,
        distance: int
    ) -> Tuple[torch.Tensor, torch.Tensor, Any]:
        """
        Sample action from policy.

        Args:
            data: PyG Data object with graph structure
            accumulated_flip: Current accumulated correction [n_qubits]
            syndrome: Target syndrome [n_checks]

        Returns:
            action: Binary flip pattern [n_qubits]
            log_prob: Log probability of action (scalar)
        """

        # Augment features with RL state
        augmented_data = self.augment_features(
            data, accumulated_flip, syndrome, H
        ).to(self.device)

        # # Move to device
        # augmented_data = augmented_data.to(self.device)

        # # Forward pass
        # out = self.model(augmented_data)  # [n_qubits, 1]
        # out = out.squeeze(-1)  # [n_qubits]

        # # Sample from Bernoulli distribution for each qubit
        # probs = torch.sigmoid(out)
        # dist = torch.distributions.Bernoulli(probs)

        # action = dist.sample()

        # # Compute log probability
        # log_prob = dist.log_prob(action).sum()

        # return action, log_prob

        return self.models[distance].get_action_and_value(augmented_data)

    def compute_syndrome(self, flip_pattern: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Compute syndrome from flip pattern: s = H @ flip (mod 2)"""
        syndrome = (H @ flip_pattern.float()) % 2
        return syndrome

    def is_logical_operator(self, residual: torch.Tensor, H: torch.Tensor, distance: int) -> bool:
        """Check if residual is a logical operator."""

        weight = residual.sum().item()

        if weight == 0:
            return False # trivial - good!

        # Compute syndrome of residual
        syndrome = self.compute_syndrome(residual, H)
        
        if torch.all(syndrome == 0) and weight >= distance:
            return True
        else: 
            return False

    def compute_reward(
        self,
        accumulated_flip: torch.Tensor,
        true_error: torch.Tensor,
        target_syndrome: torch.Tensor,
        shot_number: int,
        H: torch.Tensor,
        distance: int
    ) -> Tuple[float, bool]:
        """
        Compute reward for current state.

        Returns:
            reward: Scalar reward
            done: Whether episode is complete
        """
        # Check syndrome satisfaction
        predicted_syndrome = self.compute_syndrome(accumulated_flip % 2, H)
        syndrome_match = torch.all(predicted_syndrome == target_syndrome)

        # Compute residual error
        residual = (true_error + accumulated_flip) % 2

        # avg_flips_per_qubit = np.mean(accumulated_flip)
        avg_flips_per_qubit = accumulated_flip.float().mean().item()

        # Check if residual is logical
        # is_logical_operator = self.is_logical_operator(residual, H, distance)
        is_logical_operator = False #just for testing purposes set to false

        # Reward structure
        if syndrome_match and not is_logical_operator:
            # Success! Reward inversely proportional to shots and Hamming weight of flip string
            return 10.0 / (shot_number + avg_flips_per_qubit), True
        elif syndrome_match and is_logical_operator:
            # Syndrome correct but logical error
            return -5.0, True
        elif shot_number >= self.max_shots:
            # Out of shots
            return -10.0, True
        else:
            # Continue - small penalty per shot
            return -0.01, False

    def train_episode(
        self,
        data,
        true_error: torch.Tensor,
        syndrome: torch.Tensor,
        distance: int
    ) -> Tuple[float, int, bool]:
        """
        Train on single episode.

        Returns:
            total_reward: Cumulative reward
            num_shots: Number of shots used
            success: Whether decoding succeeded
            distance: Code distance (to get correct H matrix)
        """
        # Reset episode storage
        self.saved_log_probs = []
        self.rewards = []
        self.saved_values = []

        # Get H matrix for this distance
        H = self.get_H_matrix(distance)

        # Get number of qubits from data
        n_qubits = data.x.size(0)

        # Initialize accumulated flip
        accumulated_flip = torch.zeros(n_qubits, device=self.device)

        # Multi-shot loop
        for shot in range(1, self.max_shots + 1):
            # Get action
            action, log_prob, value = self.get_action(data, accumulated_flip, syndrome, H, distance)
            self.saved_log_probs.append(log_prob)
            self.saved_values.append(value)

            # Apply action
            accumulated_flip += action

            # Compute reward
            reward, done = self.compute_reward(accumulated_flip, true_error, syndrome, shot, H, distance)
            self.rewards.append(reward)

            if done:
                break

        # Update policy
        self.update_actor_critic(distance)

        return sum(self.rewards), len(self.rewards), (self.rewards[-1] > 0 if self.rewards else False)
    
    def update_actor_critic(self, distance: int):
        """
        Update actor and critic using advantage estimates.
        
        Args:
            distance: Which distance model to update
        """
        # Get optimizers
        actor_opt, critic_opt = self.optimizers[distance]
        
        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)
        
        returns = torch.tensor(returns, device=self.device)
        
        # Normalize returns
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # Compute advantages
        values_tensor = torch.stack(self.saved_values)
        advantages = returns - values_tensor.detach()
        
        # Actor loss (policy gradient with advantage)
        actor_loss = []
        for log_prob, advantage in zip(self.saved_log_probs, advantages):
            actor_loss.append(-log_prob * advantage)
        
        # Critic loss (MSE between value and returns)
        critic_loss = nn.functional.mse_loss(values_tensor, returns)
        
        # Update critic
        critic_opt.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in critic_opt.param_groups for p in group['params']], 
            max_norm=1.0
        )
        critic_opt.step()
        
        # Update actor
        actor_opt.zero_grad()
        actor_loss_total = torch.stack(actor_loss).sum()
        actor_loss_total.backward()
        torch.nn.utils.clip_grad_norm_(
            [p for group in actor_opt.param_groups for p in group['params']], 
            max_norm=1.0
        )
        actor_opt.step()

    def update_curriculum(self):
        """
        Update which distances to train on based on adaptive curriculum.
        Adds next distance when current distances reach success threshold.

        Args:
            distances: All available distances (sorted)
        """
        if not self.current_distances:
            # Start with smallest distance
            self.current_distances = self.distances[:1]
            self.logger.info(f"  📚 Starting with {self.current_distances} at p={self.current_error_rate:.3f}")
            return
        
        # Check success rate on current max distance
        max_current = max(self.current_distances)
        success_rate = self.distance_success_rates.get(max_current, 0.0)

        # only update if successful enough
        if success_rate < self.success_threshold:
            return
        
        if max_current != self.distances[-1]:
            idx = self.distances.index(max_current)
            next_distances = self.distances[idx + 1: idx + 2]

            self.logger.info(f"  📈 {self.current_distances} → {next_distances}")
            self.current_distances = next_distances
            
        else:
            self.logger.info(f"  📈 At max distance = {max_current}, {self.current_distances} → {[max_current]}"
                                f" d={max_current} success: {success_rate:.3f}")
            self.current_distances = [max_current]
            # Increase error rate
            if self.current_error_rate < self.max_error_rate:
                self.current_error_rate += self.error_rate_step
                self.current_error_rate = min(self.current_error_rate, self.max_error_rate)
                # Reset success rates when changing error rate
                for d in self.current_distances:
                    self.distance_success_rates[d] = 0.0
                self.logger.info(f"  📈 Increasing error rate to {self.current_error_rate:.3f}")
            else:
                self.logger.info(f"  ✓ Reached max error rate {self.max_error_rate:.3f}")
            

    def train(
        self,
        error_rate: float = 0.08,
        samples_per_epoch: int = 1000,
        num_epochs: int = 100,
        log_interval: int = 10,
        save_path: Optional[str] = None,
    ) -> Dict[str, List[float]]:
        """
        Full RL training loop.

        Args:
            # train_loader: DataLoader with (data, error, syndrome) tuples
            num_epochs: Number of epochs
            log_interval: Logging frequency
            save_path: Path to save best model
            distances: All available distances (sorted, for curriculum)

        Returns:
            history: Dictionary with training metrics
        """

        best_success_rate = 0.0

        self.logger.info(f"Training on distances: {self.distances}")
        self.logger.info(f"Error rate: {error_rate}")
        self.logger.info(f"Samples per epoch: {samples_per_epoch}")

        history = {
            'epoch': [],
            'avg_reward': [],
            'avg_shots': [],
            'success_rate': [],
            'active_distances': [],  # Track curriculum progression
            'error_rate': [], 
            'distance_success': {d: [] for d in self.distances}
        }

        for epoch in range(num_epochs):
            # Update curriculum for this epoch (adaptive)
            self.update_curriculum()

            epoch_rewards = []
            epoch_shots = []
            epoch_successes = []
            epoch_stats = {d: {'success': [], 'reward': []} for d in self.distances}

            for _ in range(samples_per_epoch):
                # Uniform sampling from active distances
                distance = np.random.choice(self.current_distances)

                data, error, syndrome = self.generate_sample(distance, self.current_error_rate)

                reward, shots, success = self.train_episode(data, error, syndrome, distance)

                # Track overall stats
                epoch_rewards.append(reward)
                epoch_shots.append(shots)
                epoch_successes.append(float(success))

                # Track per-distance stats
                epoch_stats[distance]['success'].append(float(success))
                epoch_stats[distance]['reward'].append(reward)

            # Epoch statistics
            avg_reward = np.mean(epoch_rewards)
            avg_shots = np.mean(epoch_shots)
            success_rate = np.mean(epoch_successes)

            # Update per-distance success rates
            for d in self.distances:
                if epoch_stats[d]['success']:
                    d_success = np.mean(epoch_stats[d]['success'])
                    self.distance_success_rates[d] = d_success
                    history['distance_success'][d].append(d_success)
                else:
                    history['distance_success'][d].append(None)

            # Store history
            history['epoch'].append(epoch + 1)
            history['avg_reward'].append(avg_reward)
            history['avg_shots'].append(avg_shots)
            history['success_rate'].append(success_rate)
            history['active_distances'].append(list(self.current_distances))
            history['error_rate'].append(self.current_error_rate)

            # Logging
            if (epoch + 1) % log_interval == 0:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
                self.logger.info(f"  Error rate: {self.current_error_rate:.3f}")
                self.logger.info(f"  Avg Reward: {avg_reward:.3f}")
                self.logger.info(f"  Avg Shots: {avg_shots:.2f}")
                self.logger.info(f"  Success Rate: {success_rate:.3f}, {[history['distance_success'][d][-1] for d in self.distances]}")
                self.logger.info(f"  Active Distances: {self.current_distances}")
                for d in self.current_distances:
                    if d in self.distance_success_rates:
                        self.logger.info(f"    d={d}: {self.distance_success_rates[d]:.3f}")

            # Save best model and history
            if save_path: # and success_rate > best_success_rate:
                save_dir = Path(save_path).parent
                with open(save_dir / 'rl_training_history.json', 'w') as f:
                    json.dump(history, f, indent=2)
                # torch.save(history, Path(save_path).parent / 'rl_training_history.pt')

                for d in self.current_distances:
                    model = self.models[d]
                    actor_opt, critic_opt = self.optimizers[d]
                    torch.save({
                        'state_dict': model.state_dict(),
                        'actor_optimizer': actor_opt.state_dict(),
                        'critic_optimizer': critic_opt.state_dict(),
                        'distance': d,
                        'success_rate': self.distance_success_rates.get(d, 0.0),
                        'error_rate': self.current_error_rate,
                        'epoch': epoch
                    }, save_dir / f'actor_critic_d{d}_p{self.current_error_rate:.3f}.pt')
                self.logger.info(f"  ✓ Saved AC models for {self.current_distances} (success: {success_rate:.3f})")


        return history
