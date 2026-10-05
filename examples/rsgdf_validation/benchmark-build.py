"""Fresh, bounded ordinary producer builds at fixed precision and comparable settings."""
import argparse
import json
import resource
import time
from pathlib import Path
import h5py
import numpy as np
import pyscf
from pyscf.pbc import gto
from green_mbtools.mint import common_utils as comm

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--backend',choices=['ccgdf','rsgdf'],required=True)
p.add_argument('--nk',type=int,required=True)
p.add_argument('--basis',default='gth-dzvp-molopt-sr')
p.add_argument('--label',required=True)
p.add_argument('--precision',type=float,default=1e-10)
o=p.parse_args()
out=o.root/o.label
out.mkdir(exist_ok=False)
params=json.loads((o.root/'si2-ccgdf/parameters.json').read_text())
args=comm.init_pbc_params(params)
args.df_backend=o.backend; args.nk=[o.nk]*3; args.basis=o.basis
cell=gto.Cell(a=[[0,2.7155,2.7155],[2.7155,0,2.7155],[2.7155,2.7155,0]],
              atom='Si 0 0 0; Si 1.35775 1.35775 1.35775',
              basis=o.basis,pseudo='gth-pbe',precision=o.precision,
              space_group_symmetry=True,verbose=5,max_memory=4000).build()
kpts=cell.make_kpts(args.nk)
mydf=comm.construct_gdf(args,cell,kpts)
mydf._cderi_to_save=str(out/'cderi.h5')
start=time.monotonic();mydf.build();elapsed=time.monotonic()-start
with h5py.File(mydf._cderi) as f:
    ranks={q:int(np.count_nonzero(np.linalg.eigvalsh(f['j2c/'+q][()])>mydf.linear_dep_threshold))
           for q in f['j2c'] if q.isdigit()}
def block(i):
    return np.concatenate([re+1j*im for re,im,sign in mydf.sr_loop((kpts[i],kpts[i]),compact=False)])
b0,b1=block(0),block(len(kpts)-1)
cols=np.random.default_rng(72).choice(cell.nao_nr()**2,min(20,cell.nao_nr()**2),replace=False)
np.save(out/'cross-q0.npy',b0[:,cols].T@b1[:,cols].conj())
entry={'pyscf':pyscf.__version__,'backend':o.backend,'builder':mydf._green_df_builder_name,
       'atoms':cell.natm,'nao':cell.nao_nr(),'NQ':mydf.auxcell.nao_nr(),'nk':o.nk,
       'precision':o.precision,'factorization':'eigenvalue','retained_metric_ranks':ranks,
       'wall_seconds':elapsed,'peak_rss_kib':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
       'cderi_bytes':(out/'cderi.h5').stat().st_size,'status':'PASS'}
(out/'generation.json').write_text(json.dumps(entry,indent=2)+'\n')
print(json.dumps(entry),flush=True)
