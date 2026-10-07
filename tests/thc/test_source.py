import h5py
import numpy as np
import pytest
from green_mbtools.mint.thc.source import LegacyDFSource


def test_all_orientations_chunks_padding_and_aux_slice(archive):
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(archive.source.get_pair(i,j), archive.model.get_pair(i,j), atol=1e-12)
            np.testing.assert_allclose(archive.source.get_pair(i,j,slice(1,4)), archive.model.get_pair(i,j,slice(1,4)), atol=1e-12)
    assert archive.source.bytes_read < 2 * 9 * 5 * 3 * 3 * 16


def test_wrong_schema_and_nan_rejected(archive):
    with h5py.File(archive.directory / "meta.h5", "r+") as f:
        f.attrs["__green_version__"] = "0.9"
    with pytest.raises(ValueError, match="1.0.0"):
        LegacyDFSource(archive.input, archive.directory)
    with h5py.File(archive.directory / "VQ_4.h5", "r+") as f:
        f["4"][1,0,0,0] = np.nan
    with pytest.raises(ValueError, match="nonfinite"):
        archive.source.get_pair(2,2)
