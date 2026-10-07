import numpy as np
from scipy.linalg import solve
from green_mbtools.mint.thc import reference
from green_mbtools.mint.thc.model import pair_features


def test_complex_bubble_screening_sigma_and_semidefinite_core():
    rng = np.random.default_rng(77)
    def rand(*shape):
        return rng.normal(size=shape) + 1j * rng.normal(size=shape)
    n, r, Q = 3, 7, 4
    Xl, Xr, M = rand(r,n), rand(r,n), rand(r,Q) * .01
    Gl, Gr = rand(n,n), rand(n,n)
    V = (pair_features(Xl, Xr) @ M).T.reshape(Q,n,n)
    chi = reference.bubble(Xl, Xr, Gl, Gr)
    P = np.array([[-np.trace(V[a].conj().T @ Gl @ V[b] @ Gr) for b in range(Q)] for a in range(Q)])
    np.testing.assert_allclose(P, M.conj().T @ chi @ M, atol=1e-12)
    Wc, residual = reference.screened_correlation(M, chi)
    aux = solve(np.eye(Q) - P, P)
    np.testing.assert_allclose(Wc, M @ aux @ M.conj().T, atol=1e-12)
    assert residual < 1e-12
    sigma = -np.einsum("Qim,mn,Rjn,QR->ij", V, Gr, V.conj(), aux)
    np.testing.assert_allclose(sigma, reference.selfenergy(Xl, Xr, Gr, M, aux), atol=1e-12)
    assert np.linalg.matrix_rank(M @ M.conj().T) == Q


def test_nonseparable_pair_column_cannot_be_one_orbital_product():
    column = np.eye(3).reshape(-1)
    assert np.linalg.matrix_rank(column.reshape(3,3)) == 3
    # Every outer product of two single-orbital rows has rank at most one.
    assert np.linalg.matrix_rank(np.outer([1,2,3], [3,2,1])) == 1


def test_fft_matches_direct_complex_anisotropic_mesh():
    rng = np.random.default_rng(2)
    mesh = (2,3,1)
    a = rng.normal(size=(6,2,2)) + 1j * rng.normal(size=(6,2,2))
    b = rng.normal(size=a.shape) + 1j * rng.normal(size=a.shape)
    direct = np.empty(a.shape, complex)
    coords = list(np.ndindex(mesh))
    for k, kc in enumerate(coords):
        direct[k] = sum(a[np.ravel_multi_index(tuple((np.array(kc)-qc) % mesh), mesh)] * b[q]
                        for q, qc in enumerate(coords)) / len(a)
    np.testing.assert_allclose(reference.cyclic_convolution(a,b,mesh), direct, atol=1e-12)
