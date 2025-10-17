import numpy as np
import torch
from torch_geometric.data import Data
from tqdm import tqdm
from mqt.qecc.codes import HexagonalColorCode
from mqt.qecc.cc_decoder.decoder import LightsOut


def evaluate_single_distance(model, distance, error_rate, test_samples, device, verbose=True):
    """Evaluate model on a specific distance"""

    model.eval()

    code = HexagonalColorCode(distance)
    decoder = LightsOut(code.faces_to_qubits, code.qubits_to_faces)
    decoder.preconstruct_z3_instance()

    # Build edge index
    edges = set()
    for face_id, qubits in code.faces_to_qubits.items():
        for i, q1 in enumerate(qubits):
            for q2 in qubits[i+1:]:
                edges.add((q1, q2))
                edges.add((q2, q1))

    edge_index = torch.tensor(list(edges), dtype=torch.long).t().contiguous().to(device)
    H = code.H

    results = {
        'distance': distance,
        'valid_syndromes': 0,
        'optimal_weight': 0,
        'gnn_weights': [],
        'mqt_weights': [],
        'exact_matches': 0,
        'total_samples': 0
    }

    with torch.no_grad():
        for _ in tqdm(range(test_samples), desc=f"d={distance}", disable=not verbose):
            error = np.random.random(code.n) < error_rate
            syndrome = (H @ error.astype(int)) % 2

            if syndrome.sum() == 0:
                continue

            results['total_samples'] += 1

            # MaxSAT solution
            try:
                syndrome_bool = [bool(x) for x in syndrome]
                mqt_switches, _, _ = decoder.solve(syndrome_bool)
                mqt_correction = np.array([int(s) for s in mqt_switches], dtype=int)
                mqt_weight = mqt_correction.sum()
            except:
                continue

            # GNN prediction
            node_features = []
            for qubit_idx in range(code.n):
                faces = code.qubits_to_faces.get(qubit_idx, [])
                syn_vals = [syndrome[f] for f in faces]
                syn_vals += [0] * (3 - len(syn_vals))
                degree = len(faces)
                feat = syn_vals[:3] + [degree / 3.0]
                node_features.append(feat)

            x = torch.tensor(node_features, dtype=torch.float32).to(device)
            data = Data(x=x, edge_index=edge_index)

            out = model(data)
            gnn_correction = (torch.sigmoid(out) > 0.5).cpu().numpy().astype(int)
            gnn_weight = gnn_correction.sum()

            # Check validity
            gnn_syndrome = (H @ gnn_correction) % 2
            if np.array_equal(gnn_syndrome, syndrome):
                results['valid_syndromes'] += 1
                results['gnn_weights'].append(int(gnn_weight))
                results['mqt_weights'].append(int(mqt_weight))

                if gnn_weight == mqt_weight:
                    results['optimal_weight'] += 1

                if np.array_equal(gnn_correction, mqt_correction):
                    results['exact_matches'] += 1

    return results
