"""NumPy calculations fitted only on reference documents."""

import numpy as np


def fit_scaler(reference):
    """Return reference means, population deviations and nonconstant columns."""
    mean = reference.mean(axis=0)
    std = reference.std(axis=0, ddof=0)
    variable = std > 0
    if not variable.any():
        raise ValueError("All reference features are constant.")
    return mean, std, variable


def transform_style(values, mean, std, variable):
    """Apply an already fitted scaler without consulting evaluation statistics."""
    return (values[:, variable] - mean[variable]) / std[variable]


def nearest(reference, queries, labels):
    """Return predicted labels, distances and indices of nearest reference rows.

    Ties choose the first reference row in deterministic corpus order.
    Squared distances are clipped at zero to remove floating-point noise.
    """
    squared = (queries ** 2).sum(axis=1)[:, None] + (reference ** 2).sum(axis=1)[None, :] - 2 * queries @ reference.T
    distances = np.sqrt(np.maximum(squared, 0))
    indices = distances.argmin(axis=1)
    return np.asarray(labels)[indices], distances[np.arange(len(queries)), indices], indices
