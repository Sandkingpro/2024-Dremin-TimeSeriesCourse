import numpy as np

from .utils import *


def top_k_motifs(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k motifs based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k : number of motifs

    Returns
    --------
    motifs: top-k motifs (left and right indices and distances)
    """

    motifs_idx = []
    motifs_dist = []

    # Copy matrix profile and its indices
    matrix_profile_array = matrix_profile['mp'].copy()
    motif_indices = np.array(matrix_profile['mpi']).copy().astype(int)
    exclusion_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        if is_nan_inf(matrix_profile_array):
            break

        min_idx = np.argmin(matrix_profile_array)
        min_distance = matrix_profile_array[min_idx]

        motifs_dist.append(min_distance)
        motifs_idx.append((min_idx, motif_indices[min_idx]))

        apply_exclusion_zone(matrix_profile_array, min_idx, exclusion_zone, np.inf)
        apply_exclusion_zone(matrix_profile_array, motif_indices[min_idx], exclusion_zone, np.inf)

    return {
        "indices": motifs_idx,
        "distances": motifs_dist
    }
