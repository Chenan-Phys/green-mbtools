import numpy as np
import pytest
from green_mbtools.mint.thc.selection import PairDensityGram, pivoted_cholesky, df_residual_points, df_residual_points_qr
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


def test_df_residual_shared_points_against_explicit_least_squares():
    rng = np.random.default_rng(74)
    values = rng.normal(size=(2,13,3))+1j*rng.normal(size=(2,13,3))
    class Source:
        naux = 2
        pair_to_q = np.array([[0,1],[1,0]])
        gauge_contract = dict(verified_common_q_frame=True)
        def __init__(self):
            self.rhs = {(i,j):rng.normal(size=(2,3,3))+1j*rng.normal(size=(2,3,3))
                        for i in range(2) for j in range(2)}
        def iter_transfers(self): return range(2)
        def iter_pairs(self,q): return ((i,j) for i in range(2) for j in range(2) if self.pair_to_q[i,j]==q)
        def get_pair(self,i,j): return self.rhs[i,j]
    sources = dict(hf=Source(),correlation=Source())
    norms = {kind:sum(np.vdot(rhs,rhs).real for rhs in source.rhs.values()) for kind,source in sources.items()}
    designs = [np.concatenate([(values[i].conj()[:,:,None]*values[j][:,None,:]).reshape(13,-1).T
                              for i,j in sources['hf'].iter_pairs(q)],axis=0) for q in range(2)]
    targets = [np.concatenate([np.concatenate([source.get_pair(i,j).reshape(2,-1).T
                                              for i,j in source.iter_pairs(q)],axis=0)/np.sqrt(norms[kind])
                               for kind,source in sources.items()],axis=1) for q in range(2)]
    def loss(points):
        return sum(np.linalg.norm(Y-A[:,points]@np.linalg.lstsq(A[:,points],Y,rcond=1e-13)[0])**2
                   for A,Y in zip(designs,targets)) if points else 2.
    expected = []
    history = [loss(expected)]
    for _ in range(6):
        candidates = [(loss(expected+[p]),p) for p in range(13) if p not in expected]
        value,pivot = min(candidates)
        expected.append(pivot)
        history.append(value)
    points,diagnostics = df_residual_points(values,sources,6,np.linspace(.2,1.3,13))
    np.testing.assert_array_equal(points,expected)
    np.testing.assert_allclose(diagnostics['normalized_objective_history'],history,atol=2e-12)
    qr_points,qr_diagnostics=df_residual_points_qr(values,sources,6,np.linspace(.2,1.3,13))
    np.testing.assert_array_equal(qr_points,expected)
    np.testing.assert_allclose(qr_diagnostics['normalized_objective_history'],history,atol=2e-12)
    held_points,held_diagnostics=df_residual_points(values,sources,6,holdout_modulus=3)
    for source in sources.values():
        for rhs in source.rhs.values():
            rhs[:,0,1]+=1e5+3e4j  # Row 1 is held out for n=3, modulus=3.
    np.testing.assert_array_equal(held_points,df_residual_points(values,sources,6,holdout_modulus=3)[0])
    assert held_diagnostics['excluded_target_rows_per_pair']>0
    qr_held=df_residual_points_qr(values,sources,6,holdout_modulus=3)[0]
    for source in sources.values():
        for rhs in source.rhs.values():rhs[:,0,1]-=2e5+6e4j
    np.testing.assert_array_equal(qr_held,df_residual_points_qr(values,sources,6,holdout_modulus=3)[0])
    with pytest.raises(ValueError,match="count"):
        df_residual_points(values,sources,14)
    sources['hf'].gauge_contract = None
    with pytest.raises(ValueError,match="audited"):
        df_residual_points(values,sources,4)


def test_qr_residual_selection_stops_at_degenerate_gamma_feature_rank():
    rng=np.random.default_rng(203)
    values=rng.normal(size=(1,40,3))
    class Source:
        naux=2
        pair_to_q=np.zeros((1,1),int)
        gauge_contract=dict(verified_common_q_frame=True)
        def iter_transfers(self):return [0]
        def iter_pairs(self,q):return [(0,0)]
        def get_pair(self,i,j):return rhs
    rhs=rng.normal(size=(2,3,3))
    points,diagnostics=df_residual_points_qr(values,dict(hf=Source()),30,holdout_modulus=11)
    # Real Gamma pair products span at most six independent symmetric rows.
    assert 0<len(points)<=6
    assert np.isfinite(diagnostics['normalized_objective_history']).all()
    assert np.all(np.diff(diagnostics['normalized_objective_history'])<=1e-12)
