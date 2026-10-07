import numpy as np
import h5py
import pytest
from green_mbtools.mint.thc.fit import dense_fit, tsqr_fit, fit_source
from green_mbtools.mint.thc.validate import validate_source, validate_holdout
from green_mbtools.mint.thc import io
from green_mbtools.mint.thc.source import LegacyDFSource


def test_tsqr_matches_complex_svd_with_rank_deficiency():
    rng = np.random.default_rng(812)
    F = rng.normal(size=(53,7)) + 1j*rng.normal(size=(53,7))
    F[:,6] = F[:,2] + F[:,3]
    rhs = rng.normal(size=(53,5)) + 1j*rng.normal(size=(53,5))
    ref,_ = dense_fit(F,rhs)
    for block in (1,4,18,100):
        actual,diag = tsqr_fit((F[s:s+block],rhs[s:s+block]) for s in range(0,len(F),block))
        np.testing.assert_allclose(F @ actual,F @ ref,atol=1e-11)
        assert diag["effective_rank"] == 6


def test_gauge_rotation_preserves_Z():
    rng = np.random.default_rng(48)
    F = rng.normal(size=(20,4)) + 1j*rng.normal(size=(20,4))
    M = rng.normal(size=(4,7)) + 1j*rng.normal(size=(4,7))
    U,_ = np.linalg.qr(rng.normal(size=(7,7)) + 1j*rng.normal(size=(7,7)))
    fitted,_ = dense_fit(F, F @ M @ U)
    np.testing.assert_allclose(fitted @ fitted.conj().T, M @ M.conj().T,atol=1e-11)


def test_exact_fit_atomic_roundtrip_and_nonfirst_expansion(archive):
    fitted, diag = fit_source(archive.source,archive.model.X,row_block=2,aux_block=2)
    validation = validate_source(fitted,archive.source,1e-10,1e-10)
    assert validation["accepted"]
    destination = archive.root / "thc"
    io.write(fitted,destination,archive.input,archive.source,validation,{"test":True})
    assert not (destination / "meta.h5").exists()
    loaded = io.read(destination)
    io.expand(destination,archive.root / "expanded",chunk_size=4)
    expanded = LegacyDFSource(archive.input,archive.root / "expanded")
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(loaded.get_pair(i,j),expanded.get_pair(i,j),atol=1e-10)
    with pytest.raises(FileExistsError):
        io.write(fitted,destination,archive.input,archive.source,validation,{})
    with h5py.File(destination / "M_q_1.h5", "r+") as f:
        f["M"][0,0] += 1
    with pytest.raises(ValueError,match="checksum"):
        io.read(destination)


def test_input_mismatch_incomplete_and_rejected_export(archive):
    dest = archive.root / "rejected"
    with pytest.raises(ValueError,match="acceptance"):
        io.write(archive.model,dest,archive.input,archive.source,{"accepted":False},{})
    assert not dest.exists()
    fitted,_ = fit_source(archive.source,archive.model.X)
    val = validate_source(fitted,archive.source,1e-9,1e-9)
    io.write(fitted,dest,archive.input,archive.source,val,{})
    with h5py.File(dest / "thc_meta.h5", "r+") as f:
        f["complete"][...] = 0
    with pytest.raises(ValueError,match="incomplete"):
        io.read(dest)
    with h5py.File(dest / "thc_meta.h5", "r+") as f:
        f["complete"][...] = 1
    with h5py.File(archive.input, "r+") as f:
        f["params/nao"][...] = 2
    with pytest.raises(ValueError,match="mismatched"):
        io.read(dest,archive.input)


def test_independent_rows_predict_and_corrupted_descriptor_is_rejected(archive):
    fitted,_ = fit_source(archive.source,archive.model.X,holdout_modulus=4,pair_reversal_constraints=True)
    assert validate_holdout(fitted,archive.source,modulus=4,atol=1e-10,rtol=1e-10)["accepted"]
    val=validate_source(fitted,archive.source,1e-10,1e-10)
    dest=archive.root/"maps"
    io.write(fitted,dest,archive.input,archive.source,val,{})
    with h5py.File(dest/"thc_meta.h5","r+") as f:
        f["pair_to_q"][0,1]=0
    with pytest.raises(ValueError,match="transfer"):
        io.read(dest)


def test_signed_gauge_and_correction_corruption_are_rejected(archive):
    archive.source.gauge_contract["signed_metric"]=True
    with pytest.raises(ValueError,match="signed"):
        fit_source(archive.source,archive.model.X)
    archive.source.gauge_contract["signed_metric"]=False
    with h5py.File(archive.directory/"df_ewald.h5","w") as f:
        f["probe"]=[1.,2.]
    fitted,_=fit_source(archive.source,archive.model.X)
    dest=archive.root/"corrected"
    io.write(fitted,dest,archive.input,archive.source,validate_source(fitted,archive.source,1e-10,1e-10),{})
    io.read(dest)
    with h5py.File(dest/"df_ewald.h5","r+") as f:f["probe"][0]=3.
    with pytest.raises(ValueError,match="correction"):
        io.read(dest)


def test_constraints_reject_unverified_original_Q_conjugacy(archive,monkeypatch):
    original=archive.source.get_pair
    def broken(i,j,aux_slice=None):
        value=original(i,j,aux_slice).copy()
        if i==j:value[:,0,0]+=.01j
        return value
    monkeypatch.setattr(archive.source,'get_pair',broken)
    with pytest.raises(ValueError,match='conjugacy'):
        fit_source(archive.source,archive.model.X,pair_reversal_constraints=True)


def test_insufficient_separable_space_is_rejected_by_accuracy_gate(archive):
    model,_=fit_source(archive.source,archive.model.X[:,:1])
    validation=validate_source(model,archive.source,atol=1e-10,rtol=1e-10)
    assert not validation['accepted']
    assert max(row['worst_absolute'] for row in validation['transfers'].values())>1e-3
    with pytest.raises(ValueError,match='acceptance'):
        io.write(model,archive.root/'bad-rank',archive.input,archive.source,validation,{})
