"""Finite-array algebra oracles, with no claims about physical time mapping."""
import numpy as np
from scipy.linalg import solve


def project(X, G):
    return X @ G @ X.conj().T


def bubble(Xl, Xr, Gl, Gr):
    return -project(Xl, Gl) * project(Xr, Gr).T


def screened_correlation(M, chi):
    Z = M @ M.conj().T
    A = np.eye(len(Z)) - Z @ chi
    Wc = solve(A, Z @ chi @ Z)
    return Wc, float(np.linalg.norm(A @ Wc - Z @ chi @ Z))


def selfenergy(Xl, Xr, G, M, Waux):
    return -Xl.conj().T @ (project(Xr, G) * (M @ Waux @ M.conj().T)) @ Xl


def cyclic_convolution(a, b, mesh):
    if np.prod(mesh) != len(a) or a.shape != b.shape:
        raise ValueError("FFT requires a declared full regular mesh")
    axes = tuple(range(len(mesh)))
    shape = tuple(mesh) + a.shape[1:]
    result = np.fft.ifftn(np.fft.fftn(a.reshape(shape), axes=axes) * np.fft.fftn(b.reshape(shape), axes=axes), axes=axes)
    return result.reshape(a.shape) / len(a)
