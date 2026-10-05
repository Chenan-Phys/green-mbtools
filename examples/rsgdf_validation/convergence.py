"""Controlled Si integral convergence, without changing production defaults."""
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
import h5py
import numpy as np
from pyscf.pbc import gto
from green_mbtools.mint import common_utils as comm

root=Path(sys.argv[1])
out=root/'convergence'
out.mkdir(exist_ok=False)
params=json.loads((root/'si2-ccgdf/parameters.json').read_text())
args=comm.init_pbc_params(params)
with h5py.File(root/'si2-ccgdf/input.h5') as f:
    cell=gto.loads(f['Cell'][()].decode())
    kpts=f['symmetry/k/mesh'][()]
    pairs=f['symmetry/pairs/kpair_idx'][()][f['symmetry/pairs/kpair_irre_list'][()]]
    scaled=f['symmetry/k/mesh_scaled'][()]
cols=np.random.default_rng(72).choice(cell.nao_nr()**2,20,replace=False)
groups=defaultdict(list)
for index,(i,j) in enumerate(pairs):
    q=scaled[j]-scaled[i]; q-=np.floor(q+0.5)
    groups[tuple(q.round(8))].append(index)

def eris(blocks):
    return np.array([blocks[i].T@blocks[j].conj() for inds in groups.values() for i in inds for j in inds])

def baseline(backend):
    folder=root/('si2-'+backend)/'df_hf_int'
    with h5py.File(folder/'meta.h5') as f: starts=f['chunk_indices'][()]
    b=[]
    for start in starts:
        with h5py.File(folder/f'VQ_{start}.h5') as f:
            v=f[str(start)][()].view(np.complex128)
            b.extend(v.reshape(v.shape[0],v.shape[1],-1)[:,:,cols])
    return eris(b[:len(pairs)])

old={b:baseline(b) for b in ('ccgdf','rsgdf')}
results=[]
for precision in (1e-10,1e-12):
    cell.precision=precision
    cell.mesh=None; cell.rcut=None
    cell.build(False,False)
    new={}
    for backend in old:
        args.df_backend=backend
        mydf=comm.construct_gdf(args,cell,kpts)
        mydf._cderi_to_save=str(out/f'{backend}-{precision}.h5')
        start=time.monotonic(); mydf.build()
        blocks=[]
        for i,j in pairs:
            chunks=[]
            for re,im,sign in mydf.sr_loop((kpts[i],kpts[j]),compact=False):
                assert sign==1
                chunks.append(re+1j*im)
            blocks.append(np.concatenate(chunks)[:,cols])
        new[backend]=eris(blocks)
        print(backend,precision,'seconds',time.monotonic()-start,flush=True)
    entry={'precision':precision,'mesh':cell.mesh.tolist(),
           'backend_max_abs':float(np.max(np.abs(new['ccgdf']-new['rsgdf']))),
           'default_to_tight':{b:float(np.max(np.abs(old[b]-new[b]))) for b in old}}
    np.testing.assert_allclose(new['ccgdf'],new['rsgdf'],atol=1e-8,rtol=1e-7)
    print(json.dumps(entry),flush=True); results.append(entry)
    np.savez(out/f'eris-{precision}.npz',**new)
(out/'comparison.json').write_text(json.dumps(results,indent=2)+'\n')
