"""Implicit positive weighted pair-density Gram and residual pivot selection."""
import numpy as np


class PairDensityGram:
    def __init__(self, values, pairs, grid_weights=None, pair_weights=None):
        self.values = np.asarray(values)
        self.pairs = list(pairs)
        ng = self.values.shape[1]
        self.grid_sqrt = np.sqrt(np.ones(ng) if grid_weights is None else np.asarray(grid_weights))
        self.weights = np.ones(len(self.pairs)) / len(self.pairs) if pair_weights is None else np.asarray(pair_weights)
        if not self.pairs or np.any(self.weights < 0) or not np.isfinite(self.weights).all() or not np.isfinite(self.grid_sqrt).all():
            raise ValueError("selection weights must be positive finite")

    def diagonal(self):
        norms = np.sum(abs(self.values) ** 2, axis=2)
        return self.grid_sqrt ** 2 * sum(w * norms[i] * norms[j] for (i, j), w in zip(self.pairs, self.weights))

    def column(self, pivot):
        value = np.zeros(len(self.grid_sqrt), dtype=complex)
        for (i, j), w in zip(self.pairs, self.weights):
            left = self.values[i] @ self.values[i, pivot].conj()
            right = self.values[j] @ self.values[j, pivot].conj()
            value += w * left.conj() * right
        return value * self.grid_sqrt * self.grid_sqrt[pivot]


def pivoted_cholesky(oracle, count, relative_tolerance=1e-12, negative_tolerance=1e-10):
    residual = np.asarray(oracle.diagonal(), dtype=float).copy()
    if not 0 < count <= len(residual) or not np.isfinite(residual).all():
        raise ValueError("invalid pivot count/Gram diagonal")
    scale = max(float(residual.max()), np.finfo(float).tiny)
    if residual.min() < -negative_tolerance * scale:
        raise ValueError("Gram has substantially negative diagonal")
    factors = np.zeros((len(residual), count), dtype=complex)
    pivots, history = [], [float(residual.max())]
    for n in range(count):
        pivot = int(np.argmax(residual))
        if residual[pivot] <= relative_tolerance * scale:
            break
        column = oracle.column(pivot) - factors[:, :n] @ factors[pivot, :n].conj()
        factors[:, n] = column / np.sqrt(residual[pivot])
        residual -= abs(factors[:, n]) ** 2
        if residual.min() < -negative_tolerance * scale:
            raise ValueError("Gram deflation lost positive semidefiniteness")
        residual = np.maximum(residual, 0)
        pivots.append(pivot)
        residual[pivots] = 0
        history.append(float(residual.max()))
    if not pivots:
        raise ValueError("zero pair-density Gram")
    return np.array(pivots, dtype=np.int64), dict(method="implicit_residual_pivoted_cholesky", requested_count=count,
                                                   accepted_count=len(pivots), max_residual_history=history,
                                                   relative_tolerance=relative_tolerance)
