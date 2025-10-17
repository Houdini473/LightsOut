import numpy as np
import torch
from torch_geometric.data import Data, Dataset
from tqdm import tqdm
from mqt.qecc.codes import HexagonalColorCode
from mqt.qecc.cc_decoder.decoder import LightsOut
from mqt.qecc import UFHeuristic, Code


class MultiDistanceDataset(Dataset):
    """Dataset with samples from multiple code distances"""

    def __init__(self, distances, n_samples_per_distance, error_rate=0.1,
                 seed=42, use_fast_decoder=False, verbose=True):
        super().__init__()

        np.random.seed(seed)
        self.data_list = []
        self.distances = distances
        self.verbose = verbose

        if self.verbose:
            print(f"Generating data for distances: {distances}")
            print(f"Samples per distance: {n_samples_per_distance}")
            print(f"Error rate: {error_rate}")
            print(f"Fast decoder: {use_fast_decoder}\n")

        for distance in distances:
            self._generate_distance(distance, n_samples_per_distance,
                                   error_rate, use_fast_decoder)

        if self.verbose:
            print(f"\nTotal samples: {len(self.data_list)}")

    def _generate_distance(self, distance, n_samples, error_rate, use_fast_decoder):
        """Generate samples for a single distance"""
        if self.verbose:
            print(f"{'='*60}")
            print(f"Distance {distance}")
            print(f"{'='*60}")

        code = HexagonalColorCode(distance)

        if self.verbose:
            print(f"  Qubits: {code.n}, Faces: {len(code.faces_to_qubits)}")

        # Create decoder
        if use_fast_decoder:
            decoder = UFHeuristic()
            decoder.set_code(Code(code.H.tolist(), code.H.tolist()))
            decode_fn = lambda s: np.array(decoder.result.estimate, dtype=int)
        else:
            decoder = LightsOut(code.faces_to_qubits, code.qubits_to_faces)
            decoder.preconstruct_z3_instance()

            def decode_fn(syndrome):
                syndrome_bool = [bool(x) for x in syndrome]
                switches, _, _ = decoder.solve(syndrome_bool)
                return np.array([int(s) for s in switches], dtype=int)

        edge_index = self._build_edge_index(code)

        samples_generated = 0
        attempts = 0
        max_attempts = n_samples * 3

        pbar = tqdm(total=n_samples, desc=f"d={distance}", disable=not self.verbose)

        while samples_generated < n_samples and attempts < max_attempts:
            attempts += 1

            error = np.random.random(code.n) < error_rate
            syndrome = (code.H @ error.astype(int)) % 2

            if syndrome.sum() == 0:
                continue

            try:
                if use_fast_decoder:
                    decoder.decode(syndrome)
                    correction = decode_fn(syndrome)
                else:
                    correction = decode_fn(syndrome)

                check_syndrome = (code.H @ correction) % 2
                if not np.array_equal(check_syndrome, syndrome):
                    continue

                node_features = self._create_node_features(syndrome, code)

                data = Data(
                    x=torch.tensor(node_features, dtype=torch.float32),
                    edge_index=edge_index,
                    y=torch.tensor(correction, dtype=torch.float32),
                    syndrome=torch.tensor(syndrome, dtype=torch.float32),
                    distance=torch.tensor(distance, dtype=torch.long)
                )

                self.data_list.append(data)
                samples_generated += 1
                pbar.update(1)

            except Exception as e:
                if attempts % 100 == 0 and self.verbose:
                    print(f"    Attempts: {attempts}, Generated: {samples_generated}")
                continue

        pbar.close()

        if self.verbose:
            print(f"  Generated {samples_generated}/{n_samples} samples")

    def _build_edge_index(self, code):
        edges = set()
        for face_id, qubits_in_face in code.faces_to_qubits.items():
            for i, q1 in enumerate(qubits_in_face):
                for q2 in qubits_in_face[i+1:]:
                    edges.add((q1, q2))
                    edges.add((q2, q1))

        if not edges:
            return torch.zeros((2, 0), dtype=torch.long)

        return torch.tensor(list(edges), dtype=torch.long).t().contiguous()

    def _create_node_features(self, syndrome, code):
        """Create 4D feature vector per qubit"""
        features = []
        for qubit_idx in range(code.n):
            faces = code.qubits_to_faces.get(qubit_idx, [])
            syn_vals = [syndrome[f] for f in faces]
            syn_vals += [0] * (3 - len(syn_vals))
            degree_norm = len(faces) / 3.0
            feat = syn_vals[:3] + [degree_norm]
            features.append(feat)
        return np.array(features, dtype=np.float32)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]
