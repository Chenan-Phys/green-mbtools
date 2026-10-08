from types import SimpleNamespace
import h5py
import numpy as np
from green_mbtools.mint import integral_utils


def test_shorter_corrected_metric_clears_bare_auxiliary_tail(tmp_path, monkeypatch):
    """An Ewald metric losing one row must not inherit the bare row."""
    normal=tmp_path/"bare.h5";corrected=tmp_path/"corrected.h5"
    normal.touch();corrected.touch()
    kmesh=np.array([[0.,0.,0.],[.5,0.,0.]])
    ij=np.array([[0,0],[0,1],[1,1]])
    monkeypatch.setattr(integral_utils,"integrals_grid",lambda cell,k:
                        (ij,np.arange(3),np.arange(3),np.arange(3),3,kmesh[ij[:,0]],kmesh[ij[:,1]]))
    monkeypatch.setattr(integral_utils.addons,"make_auxmol",
                        lambda *args:SimpleNamespace(nao_nr=lambda:3))
    monkeypatch.setattr(integral_utils.imd,"version",lambda name:"test")
    class DF:
        auxbasis=None
        _cderi=None
        def sr_loop(self,*args,**kwargs):
            count=2 if self._cderi==str(corrected) else 3
            value=7. if count==2 else 3.
            yield np.full((count,4),value),np.zeros((count,4)),1
    output=tmp_path/"factors"
    integral_utils.compute_integrals(SimpleNamespace(memory=64,orth="none"),
                                    object(),DF(),kmesh,2,basename=str(output),
                                    cderi_name=str(normal),cderi_name2=str(corrected),
                                    keep=True,keep_after=True)
    with h5py.File(output/"VQ_0.h5") as f:
        blocks=f["0"][()].view(np.complex128)
    assert np.all(blocks[0,:2]==7.)
    assert np.all(blocks[0,2]==0.)
    assert np.all(blocks[1]==3.)
    assert np.all(blocks[2,:2]==7.)
    assert np.all(blocks[2,2]==0.)
