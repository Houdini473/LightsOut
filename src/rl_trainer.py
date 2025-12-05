import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict
from tqdm import tqdm
import logging
from torch_geometric.data import Data

from mqt.qecc.codes import HexagonalColorCode

class RLColorCodeTrainer:
    """
    Reinforcement learning trainer for color code decoder.
    Uses REINFORCE algorithm with multi-shot correction.
    """

    def __init__(
        self,
        model: nn.Module,
        distances: List[int] = [],
        max_shots: int = 10,
        learning_rate: float = 1e-4,  # Lower LR for fine-tuning
        gamma: float = 0.99,
        success_threshold: float = 0.7,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    ):
        self.model = model.to(device)

        self.max_shots = max_shots
        self.gamma = gamma
        self.device = device
        self.success_threshold = success_threshold
        self.distances = sorted(distances)

        # Curriculum tracking
        self.current_distances = []
        self.distance_success_rates = {}

        # Logging
        self.logger = logging.getLogger('RLTrainer')

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

        # Episode tracking
        self.saved_log_probs = []
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
        H: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
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
        )

        # Move to device
        augmented_data = augmented_data.to(self.device)

        # Forward pass
        out = self.model(augmented_data)  # [n_qubits, 1]
        out = out.squeeze(-1)  # [n_qubits]

        # Sample from Bernoulli distribution for each qubit
        probs = torch.sigmoid(out)
        dist = torch.distributions.Bernoulli(probs)

        action = dist.sample()

        # Compute log probability
        log_prob = dist.log_prob(action).sum()

        return action, log_prob

    def compute_syndrome(self, flip_pattern: torch.Tensor, H: torch.Tensor) -> torch.Tensor:
        """Compute syndrome from flip pattern: s = H @ flip (mod 2)"""
        syndrome = (H @ flip_pattern.float()) % 2
        return syndrome

    def is_logical_operator(self, residual: torch.Tensor, H: torch.Tensor, distance: int) -> bool:
        """
        Check if residual is a logical operator.
        For color codes, check if it's in the stabilizer group.
        """

        weight = residual.sum().item()

        if weight == 0:
            return False

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
        is_logical_operator = self.is_logical_operator(residual, H, distance)

        # Reward structure
        if syndrome_match and not is_logical_operator:
            # Success! Reward inversely proportional to shots and Hamming weight of flip string
            reward = 10.0 / (shot_number + avg_flips_per_qubit)
            done = True

        elif syndrome_match and is_logical_operator:
            # Syndrome correct but logical error
            reward = -5.0
            done = True

        elif shot_number >= self.max_shots:
            # Out of shots
            reward = -10.0
            done = True

        else:
            # Continue - small penalty per shot
            reward = -0.1
            done = False

        return reward, done

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

        # Get H matrix for this distance
        H = self.get_H_matrix(distance)

        # Get number of qubits from data
        n_qubits = data.x.size(0)

        # Initialize accumulated flip
        accumulated_flip = torch.zeros(n_qubits, device=self.device)

        success = False

        # Multi-shot loop
        for shot in range(1, self.max_shots + 1):
            # Get action
            action, log_prob = self.get_action(data, accumulated_flip, syndrome, H)
            self.saved_log_probs.append(log_prob)

            # Apply action
            accumulated_flip += action

            # Compute reward
            reward, done = self.compute_reward(
                accumulated_flip, true_error, syndrome, shot, H, distance
            )
            self.rewards.append(reward)

            if done:
                success = (reward > 0)  # Positive reward = success
                break

        # Update policy
        self.update_policy()

        return sum(self.rewards), len(self.rewards), success

    def update_curriculum(self, distances: List[int]):
        """
        Update which distances to train on based on adaptive curriculum.
        Adds next distance when current distances reach success threshold.

        Args:
            distances: All available distances (sorted)
        """
        if not self.current_distances:
            # Start with smallest distance
            self.current_distances = [distances[0]]
            self.logger.info(f"  📚 Curriculum: Starting with distance {distances[0]}")
        else:
            # Check success rate on current max distance
            max_current = max(self.current_distances)
            success_rate = self.distance_success_rates.get(max_current, 0.0)

            if success_rate >= self.success_threshold and max_current != distances[-1]:
                # Add next distance
                next_idx = distances.index(max_current) + 1
                if next_idx < len(distances):
                    next_distance = distances[next_idx]
                    if next_distance not in self.current_distances:
                        self.current_distances.append(next_distance)
                        self.logger.info(
                            f"  📈 Curriculum: Adding distance {next_distance} "
                            f"(d={max_current} success: {success_rate:.3f})"
                        )

    def should_use_sample(self, distance: int) -> bool:
        """
        Decide whether to use a sample based on curriculum.

        Args:
            distance: Sample's distance

        Returns:
            True if sample should be used in current curriculum stage
        """
        if not self.current_distances:
            return True  # No curriculum active yet
        return distance in self.current_distances

    def update_policy(self):
        """Update policy using REINFORCE"""
        # Compute returns
        returns = []
        R = 0
        for r in reversed(self.rewards):
            R = r + self.gamma * R
            returns.insert(0, R)

        returns = torch.tensor(returns, device=self.device)

        # Normalize returns (if more than 1 step)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-8)

        # Compute loss
        policy_loss = []
        for log_prob, R in zip(self.saved_log_probs, returns):
            policy_loss.append(-log_prob * R)

        # Backprop
        self.optimizer.zero_grad()
        loss = torch.stack(policy_loss).sum()
        loss.backward()

        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

        self.optimizer.step()

    def train(
        self,
        # train_loader,
        distances: List[int] = [],
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
        self.model.train()
        best_success_rate = 0.0

        distances = sorted(distances) # sort distances from small to large in case they weren't passed this way


        self.logger.info(f"Training on distances: {distances}")
        self.logger.info(f"Error rate: {error_rate}")
        self.logger.info(f"Samples per epoch: {samples_per_epoch}")


        history = {
            'epoch': [],
            'avg_reward': [],
            'avg_shots': [],
            'success_rate': [],
            'active_distances': [],  # Track curriculum progression
            'distance_success': {d: [] for d in distances}
        }


        for epoch in range(num_epochs):
            # Update curriculum for this epoch (adaptive)
            self.update_curriculum(distances)

            epoch_rewards = []
            epoch_shots = []
            epoch_successes = []
            epoch_distance_stats = {d: {'success': [], 'reward': []} for d in distances}

            pbar = tqdm(range(samples_per_epoch), desc=f"Epoch {epoch+1}/{num_epochs}")

            for sample_idx in pbar:
                # Uniform sampling from active distances
                distance = np.random.choice(self.current_distances)

                data, error, syndrome = self.generate_sample(distance, error_rate)

                reward, shots, success = self.train_episode(
                    data, error, syndrome, distance
                )

                # Track overall stats
                epoch_rewards.append(reward)
                epoch_shots.append(shots)
                epoch_successes.append(float(success))

                # Track per-distance stats
                epoch_distance_stats[distance]['success'].append(float(success))
                epoch_distance_stats[distance]['reward'].append(reward)

            # Epoch statistics
            avg_reward = np.mean(epoch_rewards)
            avg_shots = np.mean(epoch_shots)
            success_rate = np.mean(epoch_successes)

            # Update per-distance success rates
            for d in distances:
                if epoch_distance_stats[d]['success']:
                    d_success = np.mean(epoch_distance_stats[d]['success'])
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

            # Logging
            if (epoch + 1) % log_interval == 0:
                self.logger.info(f"Epoch {epoch+1}/{num_epochs}")
                self.logger.info(f"  Avg Reward: {avg_reward:.3f}")
                self.logger.info(f"  Avg Shots: {avg_shots:.2f}")
                self.logger.info(f"  Success Rate: {success_rate:.3f}, {[round(history['distance_success'][d][-1], 3) for d in distances]}")
                self.logger.info(f"  Active Distances: {self.current_distances}")


            # Save history after every epoch (overwrite)
            if save_path:
                history_path = Path(save_path).parent / 'rl_training_history.pt'
                torch.save(history, history_path)

            # Save best model
            if save_path: # and success_rate > best_success_rate:
                # best_success_rate = success_rate
                torch.save({
                    'epoch': epoch,
                    'state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'success_rate': success_rate,
                    'avg_reward': avg_reward,
                    'history': history,
                    'curriculum_state': {
                        'current_distances': self.current_distances,
                        'distance_success_rates': self.distance_success_rates
                    }
                }, save_path)
                self.logger.info(f"  Saved new best model (success: {success_rate:.3f})")

        return history
