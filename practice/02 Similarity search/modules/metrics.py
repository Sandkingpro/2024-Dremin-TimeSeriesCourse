import numpy as np


def ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    ed_dist: euclidean distance between ts1 and ts2
    """
    
    ed_dist = np.sqrt(np.sum((ts1 - ts2) ** 2))

    return ed_dist


def norm_ED_distance(ts1: np.ndarray, ts2: np.ndarray) -> float:
    """
    Calculate the normalized Euclidean distance

    Parameters
    ----------
    ts1: the first time series
    ts2: the second time series

    Returns
    -------
    norm_ed_dist: normalized Euclidean distance between ts1 and ts2s
    """

    norm_ed_dist = 0

    # INSERT YOUR CODE

    return norm_ed_dist


def DTW_distance(ts1: np.ndarray, ts2: np.ndarray, r: float = 1) -> float:
    """
    Calculate DTW distance

    Parameters
    ----------
    ts1: first time series
    ts2: second time series
    r: warping window size
    
    Returns
    -------
    dtw_dist: DTW distance between ts1 and ts2
    """

    n, m = len(ts1), len(ts2)
    r = max(int(r * max(n, m)), abs(n - m))
    dtw_matrix = np.full((n + 1, m + 1), float('inf'))
    dtw_matrix[0, 0] = 0

    for i in range(1, n + 1):
        dtw_matrix[i, 0] = float('inf')
    for j in range(1, m + 1):
        dtw_matrix[0, j] = float('inf')

    for i in range(1, n + 1):
        for j in range(max(1, i - r), min(m + 1, i + r + 1)):
            cost = (ts1[i - 1] - ts2[j - 1]) ** 2
            dtw_matrix[i, j] = cost + min(dtw_matrix[i - 1, j],
                                          dtw_matrix[i, j - 1],
                                          dtw_matrix[i - 1, j - 1])
    return dtw_matrix[n, m]
