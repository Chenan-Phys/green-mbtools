"""Frozen-input reference contractions; consumer executables remain unchanged.

The executable saves aggregate self-energies. This reference separately checks
Hartree/exchange, polarization projected into orbital pairs, and GF2 diagrams.
It does not claim that the executable exports individual GPU diagram arrays.
"""
import json
import sys
from pathlib import Path
import h5py
import numpy as np
from pyscf.pbc import df,gto

root=Path(sys.argv[1])
with h5py.File(root/'h2-ccgdf/input.h5') as f:
    cell=gto.loads(f['Cell'][()].decode())
    kp=f['symmetry/k/mesh'][()];ks=f['symmetry/k/mesh_scaled'][()]
    packed=f['HF/S-k'][()];overlap=packed.view(complex).reshape(packed.shape[:-1])
    madelung=float(f['HF/madelung'][()])
with h5py.File(root/'consumers/ccgdf-gf2-cpu/results.h5') as f:
    all_g=f['iter1/G_tau/data'][()]
times=[0,len(all_g)//4,len(all_g)//2,len(all_g)-2,len(all_g)-1]
gt=all_g[times];gb=all_g[[len(all_g)-1-t for t in times]]
density=-all_g[-1];nk=len(kp);ns=density.shape[0];nao=cell.nao_nr()
def index(scaled):
    delta=ks-scaled;delta-=np.rint(delta)
    j=int(np.argmin(np.linalg.norm(delta,axis=1)))
    assert np.linalg.norm(delta[j])<1e-8
    return j
results={}
for backend in ('ccgdf','rsgdf'):
    folder=root/('h2-'+backend)
    ordinary=df.GDF(cell,kp);ordinary._cderi=str(folder/'cderi.h5')
    corrected=df.GDF(cell,kp);corrected._cderi=str(folder/'cderi_ewald.h5')
    hf={};corr={}
    for i in range(nk):
        for j in range(nk):
            def read(reader):
                chunks=[]
                for re,im,sign in reader.sr_loop((kp[i],kp[j]),compact=False):
                    assert sign==1;chunks.append(re+1j*im)
                return np.concatenate(chunks).reshape(-1,nao,nao)
            hf[i,j]=read(ordinary)
            corr[i,j]=read(corrected) if i==j else hf[i,j]
    upper=sum(np.einsum('Qab,sba->Q',hf[k,k],density[:,k]) for k in range(nk))/nk
    hartree=np.array([np.einsum('Q,Qij->ij',upper,hf[k,k]) for k in range(nk)])
    exchange=np.zeros_like(density)
    ewald=np.zeros_like(density)
    for s in range(ns):
        for k in range(nk):
            for j in range(nk):
                v=hf[k,j]
                exchange[s,k]-=np.einsum('Qib,ba,Qja->ij',v,density[s,j],v.conj(),optimize=True)/nk
            ewald[s,k]=-madelung*overlap[s,k]@density[s,k]@overlap[s,k]
    polarization=[];spectra=[]
    for q in ks:
        p=np.zeros((len(times),len(upper),len(upper)),complex)
        for k in range(nk):
            j=index(ks[k]+q);v=corr[k,j]
            for s in range(ns):
                p-=np.einsum('tap,Qpm,tmn,Ran->tQR',gb[:,s,k],v,gt[:,s,j],v.conj(),optimize=True)/nk
        p=(p+p.swapaxes(-1,-2).conj())/2
        v=corr[0,index(q)].reshape(len(upper),-1)
        projected=np.einsum('Qa,tQR,Rb->tab',v.conj(),p,v,optimize=True)
        # Exercise complex auxiliary gauges as well as real eigenvector signs.
        rng=np.random.default_rng(47)
        rotation=np.linalg.qr(rng.normal(size=(len(v),len(v)))+
                              1j*rng.normal(size=(len(v),len(v))))[0]
        rotated_v=rotation@v
        rotated_p=np.einsum('AQ,tQR,BR->tAB',rotation,p,rotation.conj(),optimize=True)
        np.testing.assert_allclose(projected,
            np.einsum('Qa,tQR,Rb->tab',rotated_v.conj(),rotated_p,rotated_v,optimize=True),
            atol=1e-12,rtol=1e-12)
        polarization.append(projected)
        spectra.append(np.linalg.eigvalsh(p))
    direct=np.zeros((len(times),ns,nk,nao,nao),complex)
    second_exchange=np.zeros_like(direct)
    for k0 in range(nk):
        for k1 in range(nk):
            for k2 in range(nk):
                k3=index(ks[k0]+ks[k2]-ks[k1])
                v=np.einsum('Qip,Qmk->ipmk',corr[k0,k1],corr[k2,k3],optimize=True)
                vx=np.einsum('Qjl,Qnq->jlnq',corr[k0,k3].conj(),corr[k2,k1].conj(),optimize=True)
                for s in range(ns):
                    for sp in range(ns):
                        direct[:,s,k0]+=np.einsum('ipmk,tpq,tnm,tkl,jqnl->tij',v,gt[:,s,k1],gb[:,sp,k2],gt[:,sp,k3],v.conj(),optimize=True)/nk**2
                        if s==sp:
                            second_exchange[:,s,k0]-=np.einsum('ipmk,tpq,tnm,tkl,jlnq->tij',v,gt[:,s,k1],gb[:,sp,k2],gt[:,sp,k3],vx,optimize=True)/nk**2
    results[backend]={'HF_direct':hartree,'HF_exchange':exchange,'HF_ewald':ewald,
                      'GW_projected_polarization':np.array(polarization),'GW_polarization_spectrum':np.array(spectra),
                      'GF2_direct':direct,'GF2_second_order_exchange':second_exchange}
    print(backend,'reference diagrams complete',flush=True)
errors={}
for name,a in results['ccgdf'].items():
    b=results['rsgdf'][name]
    np.testing.assert_allclose(a,b,atol=1e-8,rtol=1e-7,err_msg=name)
    errors[name]={'max_abs':float(np.max(np.abs(a-b))),'frobenius':float(np.linalg.norm(a-b)),
                  'reference_norm':float(np.linalg.norm(a))}
    assert np.linalg.norm(a)>1e-8,name+' must be exercised'
report={'times':times,'input':'identical CCGDF iter1 G and density','errors':errors,'status':'PASS',
        'scope':'independent frozen-input reference; runtime aggregate comparison reported separately'}
(root/'diagram-comparison.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report),flush=True)
