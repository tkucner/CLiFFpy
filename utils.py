import math

import numpy as np


def gaussian_kernel(distance, bandwidth):
    # euclidean_distance = np.sqrt(((distance)**2).sum(axis=1))
    val = (1 / (bandwidth * math.sqrt(2 * math.pi))) * np.exp(-0.5 * (distance / bandwidth) ** 2)
    return val


def gaussian_kernel_mv(distances, bandwidths):
    # Number of dimensions of the multivariate gaussian
    dim = len(bandwidths)

    # Covariance matrix
    cov = np.multiply(np.power(bandwidths, 2), np.eye(dim))

    # Compute Multivariate gaussian (vectorized implementation)
    exponent = -0.5 * np.sum(np.multiply(np.dot(distances, np.linalg.inv(cov)), distances), axis=1)
    val = (1 / np.power((2 * math.pi), (dim / 2)) * np.power(np.linalg.det(cov), 0.5)) * np.exp(exponent)

    return val


def cutoff(distances, treshold):
    closer = (distances < treshold) & (distances > 0)
    closer = closer.astype(int)
    count = np.sum(closer, axis=1)
    return count
