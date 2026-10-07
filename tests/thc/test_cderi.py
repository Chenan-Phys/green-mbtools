from types import SimpleNamespace
import numpy as np
import pytest
from green_mbtools.mint.thc.source import CderiSource


def test_retained_cderi_slices_and_restore_without_build(tmp_path):
    path=tmp_path/'retained.h5';path.write_bytes(b'fixture')
    cell=SimpleNamespace(dimension=3,nao_nr=lambda:2,get_scaled_kpts=lambda k:k)
    df=SimpleNamespace(cell=cell,_cderi='original',get_naoaux=lambda:5)
    blocks=np.arange(20.).reshape(5,4)
    df.sr_loop=lambda *a,**kw:iter([(blocks[:2],blocks[:2]*.1,1),(blocks[2:],blocks[2:]*.1,1)])
    src=CderiSource(df,path,np.zeros((1,3)))
    np.testing.assert_allclose(src.get_pair(0,0,slice(1,4)),(blocks[1:4]*(1+.1j)).reshape(3,2,2))
    assert df._cderi=='original'
    with pytest.raises(ValueError,match='stride'):src.get_pair(0,0,slice(None,None,2))
    assert df._cderi=='original'
    df.sr_loop=lambda *a,**kw:iter([(blocks,blocks,-1)])
    with pytest.raises(ValueError,match='negative'):src.get_pair(0,0)
    assert df._cderi=='original'
