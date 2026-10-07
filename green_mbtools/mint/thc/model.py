"""Canonical point-major arrays; I is never the original Gaussian Q index."""
from dataclasses import dataclass
import hashlib
import numpy as np

SCHEMA = "green.thc.df_fit"
VERSION = 1
ORIENTATION = "L_Qmn=sum_I conj(X_ki_Im)*X_kj_In*M_q_IQ"


def pair_features(left, right):
    if left.shape != right.shape or left.ndim != 2:
        raise ValueError("collocation pair dimensions differ")
    return np.einsum("Im,In->mnI", left.conj(), right).reshape(-1, left.shape[0])


def fingerprint_file(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fnv64(data):
    """Portable integrity checksum for the C++ host reader (not authentication)."""
    result = 14695981039346656037
    for b in memoryview(data).cast("B"):
        result = ((result ^ b) * 1099511628211) & ((1 << 64) - 1)
    return f"{result:016x}"


def file_fnv64(path):
    result = 14695981039346656037
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            for b in block:
                result = ((result ^ b) * 1099511628211) & ((1 << 64) - 1)
    return f"{result:016x}"


def transfer_map(k_scaled, tolerance=1e-8):
    k = np.asarray(k_scaled, dtype=float)
    if k.ndim != 2 or k.shape[1] != 3 or not np.isfinite(k).all():
        raise ValueError("invalid scaled k mesh")
    transfers, result = [], np.empty((len(k), len(k)), dtype=np.int64)
    for i in range(len(k)):
        for j in range(len(k)):
            q = (k[j] - k[i]) % 1.0
            q[np.isclose(q, 1, atol=tolerance, rtol=0)] = 0
            matches = [n for n, old in enumerate(transfers)
                       if np.max(np.abs(q - old - np.rint(q - old))) < tolerance]
            if not matches:
                transfers.append(q)
            result[i, j] = matches[0] if matches else len(transfers) - 1
    return np.array(transfers), result


@dataclass
class FactorModel:
    X: np.ndarray
    M: dict
    pair_to_q: np.ndarray
    metadata: dict

    def __post_init__(self):
        self.X = np.asarray(self.X, dtype=np.complex128)
        self.pair_to_q = np.asarray(self.pair_to_q, dtype=np.int64)
        nk, r, n = self.X.shape
        if min(nk, r, n) < 1 or self.pair_to_q.shape != (nk, nk):
            raise ValueError("invalid factor dimensions")
        if not np.isfinite(self.X).all():
            raise ValueError("nonfinite X")
        qs = set(self.pair_to_q.flat)
        if qs != set(range(len(qs))) or set(self.M) != qs:
            raise ValueError("missing or invalid transfer core")
        naux = None
        for q, core in self.M.items():
            self.M[q] = core = np.asarray(core, dtype=np.complex128)
            if core.ndim != 2 or core.shape[0] != r or not np.isfinite(core).all():
                raise ValueError("invalid M core")
            naux = core.shape[1] if naux is None else naux
            if core.shape[1] != naux:
                raise ValueError("v1 reconstruction retains one padded original NQ")

    @property
    def naux(self):
        return self.M[0].shape[1]

    def get_pair(self, ki, kj, aux_slice=None):
        if not (0 <= ki < len(self.X) and 0 <= kj < len(self.X)):
            raise IndexError("k pair out of range")
        core = self.M[int(self.pair_to_q[ki, kj])]
        if aux_slice is not None:
            core = core[:, aux_slice]
        n = self.X.shape[2]
        return (pair_features(self.X[ki], self.X[kj]) @ core).T.reshape(-1, n, n)

    def Z(self, q):
        return self.M[q] @ self.M[q].conj().T
