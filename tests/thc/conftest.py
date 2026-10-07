from types import SimpleNamespace
import h5py
import numpy as np
import pytest
from green_mbtools.mint.thc.model import FactorModel, transfer_map
from green_mbtools.mint.thc.source import LegacyDFSource


@pytest.fixture
def archive(tmp_path):
    rng = np.random.default_rng(415)
    nk, r, n, Q = 3, 3, 3, 5
    X = rng.normal(size=(nk, r, n)) + 1j * rng.normal(size=(nk, r, n))
    _, mapping = transfer_map(np.array([[0, 0, 0], [1/3, 0, 0], [2/3, 0, 0]]))
    core = rng.normal(size=(r, Q)) + 1j * rng.normal(size=(r, Q))
    model = FactorModel(X, {0: rng.normal(size=(r, Q)).astype(complex), 1: core, 2: core.conj()}, mapping, {})
    pairs = np.array([(i, j) for i in range(nk) for j in range(i + 1)])
    input_file = tmp_path / "input.h5"
    with h5py.File(input_file, "w") as f:
        for name, val in dict(nk=nk, nao=n, nso=n, NQ=Q, ns=1).items():
            f["params/" + name] = val
        f["symmetry/k/mesh_scaled"] = [[0,0,0], [1/3,0,0], [2/3,0,0]]
        f["symmetry/k/mesh"] = f["symmetry/k/mesh_scaled"][()]
        for name in ("conj_pairs_list", "trans_pairs_list", "kpair_irre_list"):
            f["symmetry/pairs/" + name] = np.arange(len(pairs))
        f["symmetry/pairs/kpair_idx"] = pairs
    directory = tmp_path / "df"
    directory.mkdir()
    for start in (0, 4):
        block = np.zeros((4, Q, n, n), dtype=complex)
        for local, pair in enumerate(pairs[start:start + 4]):
            block[local] = model.get_pair(*map(int, pair))
        with h5py.File(directory / f"VQ_{start}.h5", "w") as f:
            f[str(start)] = block.view(float)
    with h5py.File(directory / "meta.h5", "w") as f:
        f["chunk_size"] = 4
        f["chunk_indices"] = [0, 4]
        f.attrs["__green_version__"] = "1.0.0"
    source = LegacyDFSource(input_file, directory, gauge_contract={"verified_common_q_frame": True, "method": "synthetic"})
    return SimpleNamespace(model=model, source=source, directory=directory, input=input_file, root=tmp_path)
