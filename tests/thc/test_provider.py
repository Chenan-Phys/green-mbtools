from types import SimpleNamespace
import numpy as np
from green_mbtools.mint.thc.source import CanonicalReconstructionSource
from green_mbtools.mint.thc.fit import fit_source
from green_mbtools.mint.thc.validate import validate_source


def test_canonical_provider_preserves_oriented_slices_and_full_objective(archive):
    source=archive.source
    provider=SimpleNamespace(nk=source.nk,nao=source.nao,naux=source.naux,
        mesh=SimpleNamespace(coordinates=np.rint(source.k_scaled*3).astype(int),denominator=3),
        attributes={'set_kind':'hf'},read=lambda i,j,output_slice:source.get_pair(i,j,slice(*output_slice)))
    adapter=CanonicalReconstructionSource(archive.input,archive.directory,provider,'hf',source.gauge_contract)
    np.testing.assert_allclose(adapter.get_pair(0,2,slice(1,3)),source.get_pair(0,2,slice(1,3)))
    model,_=fit_source(adapter,archive.model.X,aux_block=2)
    assert validate_source(model,adapter,1e-10,1e-10)['accepted']
