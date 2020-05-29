from sklearn.preprocessing import normalize

import scipy.spatial.distance as ds
import numpy as np
import torch


def norm_matrix(matrix):
    # Check dtype of the input matrix.
    np.testing.assert_equal(type(matrix).__name__, 'ndarray')
    np.testing.assert_equal(matrix.dtype, np.float32)

    # Get sum of each row ((R, )).
    row_sums = matrix.sum(axis = 1)

    # Replace zero denominator.
    row_sums[row_sums == 0] = 1

    # Added dimension is position of the newaxis object in the selection tuple.
    norm_matrix = matrix / row_sums[:, np.newaxis]

    return norm_matrix


def replace_nan(X):
    # Convert NaN to 0.
    X[np.isnan(X)] = 0
    X[np.isinf(X)] = 0

    return X


def get_sim(sc_vecs, uc_vecs, sim_scale):
    # Normalize each class vector.
    s = normalize(sc_vecs)  # (num_intents, D_I)
    u = normalize(uc_vecs)  # (num_intents, D_I)

    # Get Euclidean distances between each class.
    dist = ds.cdist(u, s, 'euclidean')  # (L, K)
    dist = dist.astype(np.float32)

    # Scale it by given scale parameter.
    sim = np.exp(-np.square(dist) * sim_scale)  # (L, K)
    total = np.sum(sim, axis=1)  # (L)
    sim = replace_nan(sim/total[:, None])  # (L, K)

    return sim


def get_label_embedding(label_embeddings, label_lens):
    # Trim pads in label embedding tensors.
    label_embeddings_new = []
    for i, label in enumerate(label_embeddings):
        label_trimmed = label[:label_lens[i]]
        label_sum = np.sum(label_trimmed, axis=0)
        label_embeddings_new.append(label_sum)

    return np.array(label_embeddings_new)


def squash(input_tensor):
    # Execute Squash function.
    norm = torch.norm(input_tensor, dim=2, keepdim=True)
    norm_squared = norm * norm
    return (input_tensor / norm) * (norm_squared / (0.5 + norm_squared))
