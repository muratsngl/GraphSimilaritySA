import numpy as np

def normalize_distance_matrix(distance_matrix):
    """
    Normalizes a geodesic distance matrix to a [0, 1] scale using Max-Scaling.
    This ensures scale-invariance between skeletons of drastically different sizes
    without altering the underlying node indices or structural topology.
    """
    max_dist = np.max(distance_matrix)
    
    if max_dist == 0:
        print("Warning: Max distance is 0. Returning original matrix to avoid division by zero.")
        return distance_matrix
        
    normalized_matrix = distance_matrix / max_dist
    return normalized_matrix