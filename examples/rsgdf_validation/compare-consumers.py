"""Compare every physical solver output at fixed, declared tolerances."""
import json
import sys
from pathlib import Path

import h5py
import numpy as np

root = Path(sys.argv[1]) / "consumers"
cases = {p.name: p for p in root.iterdir() if p.is_dir() and (p/"run.json").exists()}
checks = []
with h5py.File(root/'ccgdf-frozen-input.h5') as f:
    packed=f['HF/S-k'][()]
    overlap=packed.view(np.complex128).reshape(packed.shape[:-1])


def compare(left, right, mixed=False):
    atol, rtol = (2e-6, 2e-5) if mixed else (1e-8, 1e-7)
    errors = {}
    with h5py.File(cases[left]/"results.h5") as a, h5py.File(cases[right]/"results.h5") as b:
        assert int(a["iter"][()]) == int(b["iter"][()]) == 2
        for iteration in (1,2):
            for quantity in ("Energy_1b","Energy_2b","Energy_HF","mu","Sigma1","Selfenergy/data","G_tau/data"):
                key = f"iter{iteration}/"+quantity
                av, bv = a[key][()], b[key][()]
                delta = av-bv
                errors[key] = {"max_abs":float(np.max(np.abs(delta))),"frobenius":float(np.linalg.norm(delta))}
                np.testing.assert_allclose(av, bv, atol=atol, rtol=rtol, err_msg=left+" vs "+right+" "+key)
            counts=[]
            for f in (a,b):
                density=-f[f'iter{iteration}/G_tau/data'][-1]
                counts.append(np.einsum('skij,skji->',overlap,density).real/overlap.shape[1])
            np.testing.assert_allclose(*counts,atol=atol,rtol=rtol)
            errors[f'iter{iteration}/electron_count']={'max_abs':float(abs(counts[0]-counts[1])),
                                                     'frobenius':float(abs(counts[0]-counts[1])),
                                                     'left':float(counts[0]),'right':float(counts[1])}
    checks.append({"left":left,"right":right,"atol":atol,"rtol":rtol,"errors":errors,"status":"PASS"})


for name, path in cases.items():
    run = json.loads((path/"run.json").read_text())
    assert run["exit_code"] == 0, name
    if name.startswith("ccgdf-"):
        right = "rsgdf-"+name[len("ccgdf-"):]
        compare(name, right, mixed=("-p1" in name or "s1" in name))
for backend in ("ccgdf","rsgdf"):
    compare(backend+"-gf2-cpu", backend+"-gf2-hybrid")
    for h in (0,1):
        for d in (0,1):
            compare(backend+"-hf-cpu", f"{backend}-hf-gpu-h{h}d{d}")
            for p in (0,1):
                for s in (0,1):
                    compare(backend+"-gw-cpu-dp", f"{backend}-gw-gpu-h{h}d{d}-p{p}s{s}", mixed=bool(p or s))
report = {"case_count":len(cases),"comparison_count":len(checks),"checks":checks}
(root/"comparison.json").write_text(json.dumps(report, indent=2)+"\n")
print('PASS',len(cases),'executed cases;',len(checks),'numerical comparisons')
print('Largest max_abs:',max(v['max_abs'] for check in checks for v in check['errors'].values()))
