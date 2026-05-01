import numpy as np


def distances(xyz: np.ndarray, excluded_neighbors: int = 2) -> np.ndarray:
    """Compute pairwise C_{\alpha} distances for a trajectory.

    Args:
        xyz: C_{\alpha} coordinates of shape [T, L, 3], where T is the number
            of frames and L is the number of residues.
        excluded_neighbors: Number of sequential neighbors to exclude on each side.
            With the default of 2, pairs (i, j) with |i - j| <= 2 are excluded.

    Returns:
        Array of shape [T, n_pairs] containing the selected pairwise distances
        per frame.
    """
    # Compute full pairwise distance matrix: [T, L, L]
    distance_matrix_ca = np.linalg.norm(
        xyz[:, None, :, :] - xyz[:, :, None, :], axis=-1
    )

    n_ca = distance_matrix_ca.shape[-1]

    # Upper-triangle indices (k=1 excludes the diagonal, i.e. self-pairs)
    m, n = np.triu_indices(n_ca, k=1)

    # Keep only pairs whose sequence separation exceeds excluded_neighbors
    mask = np.abs(m - n) > excluded_neighbors
    m, n = m[mask], n[mask]

    # Extract the selected distances for all frames: [T, n_pairs]
    distances_ca = distance_matrix_ca[:, m, n]
    return distances_ca
