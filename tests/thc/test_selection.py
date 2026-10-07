import numpy as np
import pytest
from green_mbtools.mint.thc.selection import PairDensityGram, pivoted_cholesky
from green_mbtools.mint.thc.collocation import stored_basis_values
from green_mbtools.mint.thc.reference import project


def test_weighted_implicit_gram_and_residual_pivots():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(3,19,3)) + 1j * rng.normal(size=(3,19,3))
    pairs, weights = [(0,0),(0,1),(1,2)], np.array([.2,.3,.5])
    grid = rng.uniform(.1,1,size=19)
    oracle = PairDensityGram(values,pairs,grid,weights)
    explicit = np.concatenate([np.sqrt(w) * np.einsum("rm,rn->rmn", values[i].conj(),values[j]).reshape(19,-1)
                               for (i,j),w in zip(pairs,weights)],axis=1) * np.sqrt(grid[:,None])
    gram = explicit @ explicit.conj().T
    np.testing.assert_allclose(oracle.diagonal(), gram.diagonal().real)
    for pivot in (0,4,18):
        np.testing.assert_allclose(oracle.column(pivot),gram[:,pivot],atol=1e-12)
    points, diag = pivoted_cholesky(oracle, 12)
    assert len(set(points)) == len(points)
    assert np.all(np.diff(diag["max_residual_history"]) <= 1e-12)
    np.testing.assert_array_equal(points,pivoted_cholesky(oracle,12)[0])


def test_stored_basis_and_green_projection_covariance():
    rng = np.random.default_rng(22)
    ao = rng.normal(size=(1,7,3)) + 1j * rng.normal(size=(1,7,3))
    A = rng.normal(size=(3,3)) + 1j * rng.normal(size=(3,3)) + 3*np.eye(3)
    G = rng.normal(size=(3,3)) + 1j*rng.normal(size=(3,3))
    X = stored_basis_values(ao, A[None])[0]
    inverse = np.linalg.inv(A)
    Gst = inverse.conj().T @ G @ inverse
    np.testing.assert_allclose(project(ao[0],G),project(X,Gst),atol=1e-11)


def test_global_orbital_phases_and_zero_pivots():
    rng=np.random.default_rng(513)
    X=rng.normal(size=(2,17,3))+1j*rng.normal(size=(2,17,3))
    phases=np.exp(1j*rng.uniform(0,6,size=(2,1,3)))
    pairs=[(i,j) for i in range(2) for j in range(2)]
    a=PairDensityGram(X,pairs,np.ones(17)); b=PairDensityGram(X*phases,pairs,np.ones(17))
    np.testing.assert_allclose(a.column(3),b.column(3),atol=1e-11)
    np.testing.assert_array_equal(pivoted_cholesky(a,10)[0],pivoted_cholesky(b,10)[0])
    zero=PairDensityGram(np.zeros_like(X),pairs,np.ones(17))
    with pytest.raises(ValueError,match="zero"):
        pivoted_cholesky(zero,10)
