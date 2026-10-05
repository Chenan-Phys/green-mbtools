"""Independent polarization covariance against actual CDERI factors and stored U(q)."""
import json
import sys
import argparse
from pathlib import Path
import h5py
import numpy as np
from pyscf.pbc import df,gto

p=argparse.ArgumentParser()
p.add_argument('root',type=Path)
p.add_argument('--fixtures',nargs='+',default=['h2','si2tight'])
p.add_argument('--backends',nargs='+',default=['ccgdf','rsgdf'])
options=p.parse_args()
root=options.root
report=[]
for fixture in options.fixtures:
    for backend in options.backends:
        path=root/(fixture+'-'+backend)
        with h5py.File(path/'input.h5') as f:
            cell=gto.loads(f['Cell'][()].decode())
            kpts=f['symmetry/k/mesh'][()]; ks=f['symmetry/k/mesh_scaled'][()]
            qs=f['symmetry/q/mesh_scaled'][()]
            reps=f['symmetry/q/bz2ibz'][()]
            u=f['symmetry/q/k_sym_transform_p0'][()]
            tr=f['symmetry/q/tr_conj'][()]
            packed=f['HF/S-k'][()]
            overlap=packed.view(np.complex128).reshape(packed.shape[:-1])[0]
        # Inverse overlap is an AO covariant Green function with all stored symmetries.
        g=np.linalg.inv(overlap)
        ordinary=df.GDF(cell,kpts); ordinary._cderi=str(path/'cderi.h5')
        corrected=df.GDF(cell,kpts); corrected._cderi=str(path/'cderi_ewald.h5')
        polar=[]; ranks=[]
        for iq,q in enumerate(qs):
            p=np.zeros(u[iq].shape,dtype=complex)
            for i,k in enumerate(ks):
                delta=ks-k-q; delta-=np.rint(delta)
                j=int(np.argmin(np.linalg.norm(delta,axis=1)))
                assert np.linalg.norm(delta[j])<1e-8
                reader=corrected if np.linalg.norm(q)<1e-8 else ordinary
                chunks=[]
                for re,im,sign in reader.sr_loop((kpts[i],kpts[j]),compact=False):
                    assert sign==1
                    chunks.append(re+1j*im)
                v=np.concatenate(chunks).reshape(-1,cell.nao_nr(),cell.nao_nr())
                ranks.append(v.shape[0])
                x=np.einsum('ap,Qpm,mn->Qan',g[i],v,g[j],optimize=True)
                p[:len(v),:len(v)]+=np.einsum('Qan,Ran->QR',x,v.conj(),optimize=True)/len(ks)
            polar.append(p)
        maximum=0.; frob=0.
        for iq in range(len(qs)):
            reconstructed=u[iq]@polar[reps[iq]]@u[iq].conj().T
            if tr[iq]: reconstructed=reconstructed.conj()
            maximum=max(maximum,float(np.max(np.abs(reconstructed-polar[iq]))))
            frob=max(frob,float(np.linalg.norm(reconstructed-polar[iq])))
            np.testing.assert_allclose(reconstructed,polar[iq],atol=1e-8,rtol=1e-7,
                                       err_msg=f'{fixture} {backend} q={iq}')
        entry={'fixture':fixture,'backend':backend,'q_count':len(qs),'factor_ranks':sorted(set(ranks)),
               'q0_corrected':True,'max_abs':maximum,'max_frobenius':frob,'status':'PASS'}
        print(json.dumps(entry),flush=True);report.append(entry)
(root/'frame-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
