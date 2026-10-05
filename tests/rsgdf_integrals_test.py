"""Interaction-level tests; factors may differ by auxiliary-space rotations."""
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from pyscf.pbc import df, gto
import green_igen.df as legacy

from green_mbtools.mint import common_utils as comm, integral_utils as iu
from green_mbtools.mint.pyscf_init import pyscf_pbc_init


def snapshot_hooks():
    return [(cls, dict(cls.__dict__)) for cls in (df.GDF, legacy.GDF)]


def assert_hooks_restored(saved):
    for cls, original in saved:
        assert cls.__dict__.keys() == original.keys()
        assert all(cls.__dict__[name] is value for name, value in original.items())


@pytest.mark.parametrize("raises", [False, True])
def test_legacy_hooks_restore(raises):
    saved = snapshot_hooks()
    try:
        with iu.legacy_ewald_coulomb():
            assert df.GDF.weighted_coulG is iu.weighted_coulG_ewald
            legacy.GDF.weighted_coulG = df.GDF.weighted_coulG
            with iu.legacy_ewald_coulomb():
                assert df.GDF.weighted_coulG is iu.weighted_coulG_ewald
            if raises:
                raise ValueError("deliberate failure")
    except ValueError:
        assert raises
    assert_hooks_restored(saved)


def test_inherited_hook_ownership(monkeypatch):
    monkeypatch.delattr(legacy.GDF, "weighted_coulG")
    assert "weighted_coulG" not in legacy.GDF.__dict__
    with iu.legacy_ewald_coulomb():
        legacy.GDF.weighted_coulG = df.GDF.weighted_coulG
    assert "weighted_coulG" not in legacy.GDF.__dict__


def test_concurrent_hook_context_rejected():
    from concurrent.futures import ThreadPoolExecutor
    def attempt():
        with iu.legacy_ewald_coulomb():
            pass
    saved = snapshot_hooks()
    with iu.legacy_ewald_coulomb(), ThreadPoolExecutor(max_workers=1) as executor:
        with pytest.raises(RuntimeError, match="Concurrent"):
            executor.submit(attempt).result()
    assert_hooks_restored(saved)


def test_corrected_producer_exception(monkeypatch):
    saved = snapshot_hooks()
    def fail(*args):
        legacy.GDF.weighted_coulG = df.GDF.weighted_coulG
        raise ValueError("producer failed")
    monkeypatch.setattr(legacy, "make_modrho_basis", lambda *args: None)
    monkeypatch.setattr(legacy, "_make_j3c", fail)
    with pytest.raises(ValueError, match="producer failed"):
        iu.build_legacy_ewald(SimpleNamespace(auxbasis=None, exp_to_discard=None),
                              None, np.zeros((1,3)), "unused.h5")
    assert_hooks_restored(saved)


def block(mydf, pair):
    pieces = []
    for real, imag, sign in mydf.sr_loop(pair, compact=False):
        assert sign == 1, "Fixture must have a positive auxiliary metric"
        pieces.append(real + 1j * imag)
    return np.concatenate(pieces)


@pytest.fixture(scope="module")
def interactions(tmp_path_factory):
    root = tmp_path_factory.mktemp("rsgdf-interactions")
    cell = gto.Cell(a=np.eye(3)*4, atom="H 0 0 0; H 0 0 0.74",
                    basis="sto3g", precision=1e-10, verbose=0).build()
    kpts = cell.make_kpts([2,1,1])
    outputs = {}
    for backend in ("ccgdf", "rsgdf"):
        args = SimpleNamespace(df_backend=backend, auxbasis="weigend", beta=None,
                               Nk=0, space_symm=False, tr_symm=True)
        mydf = comm.construct_gdf(args, cell, kpts)
        mydf._cderi_to_save = str(root / (backend + ".h5"))
        mydf.build()
        ordinary = {(i,j): block(mydf, (ki,kj))
                    for i,ki in enumerate(kpts) for j,kj in enumerate(kpts)}
        corrected_file = str(root / (backend + "-ewald.h5"))
        saved = snapshot_hooks()
        iu.build_legacy_ewald(mydf, cell, kpts, corrected_file)
        assert_hooks_restored(saved)
        corrected = df.GDF(cell, kpts)
        corrected._cderi = corrected_file
        diagonal = {i: block(corrected, (ki,ki)) for i,ki in enumerate(kpts)}
        with h5py.File(mydf._cderi, "r") as f:
            metrics = {q: f["j2c/" + q][()] for q in f["j2c"] if q.isdigit()}
        outputs[backend] = ordinary, diagonal, metrics
    # Ordinary construction AFTER the patched producer, in the same interpreter.
    after = comm.construct_gdf(args, cell, kpts)
    after._cderi_to_save = str(root / "after.h5")
    after.build()
    after_blocks = {i: block(after, (ki,ki)) for i,ki in enumerate(kpts)}
    for i in after_blocks:
        for j in after_blocks:
            np.testing.assert_allclose(
                gram(after_blocks[i], after_blocks[j]),
                gram(outputs["rsgdf"][0][i,i], outputs["rsgdf"][0][j,j]),
                atol=1e-11, rtol=1e-11)
    return outputs


def gram(left, right):
    return left.T @ right.conj()


def test_ordinary_cross_pair_eris(interactions):
    cc, rs = interactions["ccgdf"][0], interactions["rsgdf"][0]
    # q=0 cross-pair contraction plus two q=pi pairs. Self-Grams alone are insufficient.
    for left, right in [((0,0),(0,0)), ((0,0),(1,1)), ((1,1),(1,1)),
                        ((0,1),(0,1)), ((0,1),(1,0))]:
        np.testing.assert_allclose(gram(cc[left], cc[right]), gram(rs[left], rs[right]),
                                   atol=1e-8, rtol=1e-7)


def test_ewald_cross_pair_eris(interactions):
    cc, rs = interactions["ccgdf"][1], interactions["rsgdf"][1]
    for i in cc:
        for j in cc:
            np.testing.assert_allclose(gram(cc[i], cc[j]), gram(rs[i], rs[j]),
                                       atol=1e-8, rtol=1e-7)
    ordinary = interactions["rsgdf"][0][0,0]
    assert np.linalg.norm(gram(rs[0], rs[0]) - gram(ordinary, ordinary)) > 1e-5


def test_auxiliary_metric_equivalence(interactions):
    cc, rs = interactions["ccgdf"][2], interactions["rsgdf"][2]
    assert cc.keys() == rs.keys()
    for q in cc:
        np.testing.assert_allclose(cc[q], rs[q], atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("space_symm", [True, False])
@pytest.mark.parametrize("eig", ["true", "false"])
def test_rsgdf_stored_polarization_covariance(tmp_path, monkeypatch, space_symm, eig):
    """Actual CDERI polarization tests spatial, TR and corrected q=0 frames."""
    monkeypatch.chdir(tmp_path)
    args = comm.init_pbc_params([
        "--a", "4 0 0\n0 4 0\n0 0 4", "--atom", "H 0 0 0\nH 0 0 1",
        "--basis", "sto3g", "--auxbasis", "weigend", "--nk", "3", "3", "1",
        "--df_backend", "rsgdf", "--space_symm", str(space_symm).lower(), "--tr_symm", "true",
        "--use_j2c_eig_decomposition", eig,
        "--grid_only", "true", "--keep_cderi", "true",
        "--output_path", str(tmp_path / "input.h5")])
    system = pyscf_pbc_init(args)
    system.cell.precision = 1e-10
    system.cell.mesh = None
    system.cell.rcut = None
    system.cell.build(False, False)
    mydf = system.df_object()
    system.mean_field_input(mydf)
    iu.build_legacy_ewald(mydf, system.cell, system.kmesh, "ewald.h5")
    corrected = df.GDF(system.cell, system.kmesh)
    corrected._cderi = "ewald.h5"
    with h5py.File(args.output_path) as f:
        qs = f["symmetry/q/mesh_scaled"][()]
        reps = f["symmetry/q/bz2ibz"][()]
        u = f["symmetry/q/k_sym_transform_p0"][()]
        tr = f["symmetry/q/tr_conj"][()]
    assert any(reps != np.arange(len(qs)))
    if not space_symm:
        assert any(tr)
    ks = system.cell.get_scaled_kpts(system.kmesh)
    g = np.linalg.inv(system.cell.pbc_intor("int1e_ovlp", kpts=system.kmesh))
    polar = []
    for iq, q in enumerate(qs):
        p = np.zeros_like(u[iq])
        for i, k in enumerate(ks):
            delta = ks - k - q
            delta -= np.rint(delta)
            j = int(np.argmin(np.linalg.norm(delta, axis=1)))
            assert np.linalg.norm(delta[j]) < 1e-8
            reader = corrected if np.linalg.norm(q) < 1e-8 else mydf
            v = block(reader, (system.kmesh[i], system.kmesh[j])).reshape(-1, 2, 2)
            x = np.einsum("ap,Qpm,mn->Qan", g[i], v, g[j], optimize=True)
            p[:len(v), :len(v)] += np.einsum("Qan,Ran->QR", x, v.conj()) / len(ks)
        polar.append(p)
    for iq in range(len(qs)):
        transformed = u[iq] @ polar[reps[iq]] @ u[iq].conj().T
        if tr[iq]:
            transformed = transformed.conj()
        np.testing.assert_allclose(transformed, polar[iq], atol=1e-8, rtol=1e-7)


@pytest.mark.parametrize("mode", ["gf2", "gw", "gw_s", "coarse_grained"])
def test_unsupported_rsgdf_correction(mode):
    cell = gto.Cell(a=np.eye(3)*4, atom="H 0 0 0; H 0 0 0.74", basis="sto3g", verbose=0).build()
    args = SimpleNamespace(df_backend="rsgdf", finite_size_kind=[mode])
    with pytest.raises(NotImplementedError, match="not validated"):
        comm.construct_gdf(args, cell)
