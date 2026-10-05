"""Compare complete legacy exports through physical interactions and metadata."""
import json
import sys
import argparse
from collections import defaultdict
from pathlib import Path

import h5py
import numpy as np

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--fixtures',nargs='+',default=['h2','si2tight'])
options=p.parse_args()
root = options.root
report = []
for fixture in options.fixtures:
    paths = {b: root/(fixture+"-"+b) for b in ("ccgdf", "rsgdf")}
    with h5py.File(paths["ccgdf"]/"input.h5") as cc, h5py.File(paths["rsgdf"]/"input.h5") as rs:
        for name in ("params/nao", "params/NQ", "params/nk", "symmetry/pairs/kpair_idx", "symmetry/pairs/kpair_irre_list"):
            np.testing.assert_array_equal(cc[name][()], rs[name][()])
        energy_difference = float(abs(cc["HF/Energy"][()] - rs["HF/Energy"][()]))
        assert energy_difference < 1e-8
        nao = int(cc["params/nao"][()])
        nq = int(cc["params/NQ"][()])
        pairs = cc["symmetry/pairs/kpair_idx"][()][cc["symmetry/pairs/kpair_irre_list"][()]]
        kpts = cc["symmetry/k/mesh_scaled"][()]
        for f in (cc, rs):
            qzero = np.flatnonzero(np.linalg.norm(f["symmetry/q/mesh_scaled"][()], axis=1) < 1e-8)
            assert len(qzero) == 1
            np.testing.assert_allclose(f["symmetry/q/k_sym_transform_p0"][qzero[0]], np.eye(nq), atol=1e-12)
    groups = defaultdict(list)
    for index, (i,j) in enumerate(pairs):
        q = kpts[j]-kpts[i]
        q -= np.floor(q+0.5)
        groups[tuple(q.round(8))].append(index)
    cols = np.random.default_rng(72).choice(nao*nao, min(20, nao*nao), replace=False)
    errors = {}
    blocks = {}
    for folder in ("df_hf_int", "df_int"):
        for backend in paths:
            with h5py.File(paths[backend]/folder/"meta.h5") as meta:
                chunk_indices = meta["chunk_indices"][()]
            loaded = []
            for start in chunk_indices:
                with h5py.File(paths[backend]/folder/f"VQ_{start}.h5") as f:
                    packed = f[str(start)][()]
                    assert packed.dtype == np.float64
                    v = packed.view(np.complex128)
                    assert v.shape[1:] == (nq, nao, nao)
                    loaded.extend(v.reshape(v.shape[0], nq, nao*nao)[:, :, cols])
            blocks[backend,folder] = loaded[:len(pairs)]
        maximum = 0.0
        frobenius = 0.0
        comparisons = 0
        for indices in groups.values():
            for i in indices:
                for j in indices:
                    c = blocks["ccgdf",folder][i].T @ blocks["ccgdf",folder][j].conj()
                    r = blocks["rsgdf",folder][i].T @ blocks["rsgdf",folder][j].conj()
                    maximum = max(maximum, float(np.max(np.abs(c-r))))
                    frobenius = max(frobenius, float(np.linalg.norm(c-r)))
                    np.testing.assert_allclose(c, r, atol=1e-8, rtol=1e-7)
                    comparisons += 1
        errors[folder] = {"max_abs":maximum,"max_frobenius":frobenius,"cross_pair_comparisons":comparisons}
    diag = next(i for i,pair in enumerate(pairs) if pair[0] == pair[1])
    ordinary = blocks["rsgdf","df_hf_int"][diag]
    corrected = blocks["rsgdf","df_int"][diag]
    change = float(np.linalg.norm(ordinary.T@ordinary.conj() - corrected.T@corrected.conj()))
    assert change > 1e-5
    entry = {"fixture":fixture,"nao":nao,"NQ":nq,"energy_abs_difference":energy_difference,
             "q0_transform":"identity verified","ewald_interaction_change":change,
             "errors":errors,"status":"PASS"}
    report.append(entry)
    print(json.dumps(entry), flush=True)
(root/"fixture-comparison.json").write_text(json.dumps(report, indent=2)+"\n")
