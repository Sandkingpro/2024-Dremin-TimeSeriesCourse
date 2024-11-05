import numpy as np

from .utils import *


def top_k_discords(matrix_profile: dict, top_k: int = 3) -> dict:
    """
    Find the top-k discords based on matrix profile

    Parameters
    ---------
    matrix_profile: the matrix profile structure
    top_k: number of discords

    Returns
    --------
    discords: top-k discords (indices, distances to its nearest neighbor and the nearest neighbors indices)
    """

    discords_idx = []
    discords_dist = []
    discords_nn_idx = []

    mp = matrix_profile['mp']
    mpi = np.array(matrix_profile['mpi'], dtype=int)
    excl_zone = matrix_profile['excl_zone']

    for _ in range(top_k):
        if is_nan_inf(mp):
            break

        max_idx = np.argmax(mp)
        max_dist = mp[max_idx]
        nn_idx = mpi[max_idx]

        discords_idx.append(max_idx)
        discords_dist.append(max_dist)
        discords_nn_idx.append(nn_idx)

        mp = apply_exclusion_zone(mp, max_idx, excl_zone, -np.inf)

    return {
        'indices': discords_idx,
        'distances': discords_dist,
        'nn_indices': discords_nn_idx
    }
