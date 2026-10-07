"""Read-only bounded legacy/CDERI providers, matching GREEN 1.0.0 ordering."""
from pathlib import Path
import hashlib
import json
import h5py
import numpy as np
from .model import transfer_map, fingerprint_file


class LegacyDFSource:
    def __init__(self, input_file, directory, set_kind="hf", gauge_contract=None):
        self.input_file, self.directory = Path(input_file), Path(directory)
        if set_kind not in ("hf", "correlation"):
            raise ValueError("unknown interaction set")
        self.set_kind, self.gauge_contract = set_kind, gauge_contract
        self._load_input_maps()
        self.positions = {int(rep): pos for pos, rep in enumerate(self.representatives)}
        with h5py.File(self.directory / "meta.h5", "r") as f:
            if str(f.attrs.get("__green_version__", "")) != "1.0.0":
                raise ValueError("source must be verified GREEN 1.0.0 legacy DF")
            self.chunk_size = int(f["chunk_size"][()])
            self.chunk_indices = np.asarray(f["chunk_indices"], dtype=int)
        if self.chunk_size < 1 or not np.array_equal(self.chunk_indices, np.arange(0, len(self.representatives), self.chunk_size)):
            raise ValueError("invalid chunk map")
        self.bytes_read = 0

    def _load_input_maps(self):
        with h5py.File(self.input_file, "r") as f:
            self.nk = int(f["params/nk"][()])
            self.nao = int(f["params/nao"][()])
            self.naux = int(f["params/NQ"][()])
            if int(f["params/nso"][()]) != self.nao:
                raise ValueError("spinor THC is unsupported")
            self.k_scaled = f["symmetry/k/mesh_scaled"][()]
            self.k_abs = f["symmetry/k/mesh"][()]
            self.conj = np.asarray(f["symmetry/pairs/conj_pairs_list"], dtype=int)
            self.trans = np.asarray(f["symmetry/pairs/trans_pairs_list"], dtype=int)
            self.representatives = np.asarray(f["symmetry/pairs/kpair_irre_list"], dtype=int)
            self.pairs = np.asarray(f["symmetry/pairs/kpair_idx"], dtype=int)
        if self.k_scaled.shape != (self.nk, 3) or self.pairs.shape != (self.nk * (self.nk + 1) // 2, 2):
            raise ValueError("invalid source pair/k maps")
        if self.conj.shape != (len(self.pairs),) or self.trans.shape != self.conj.shape:
            raise ValueError("invalid source symmetry maps")
        self.q_scaled, self.pair_to_q = transfer_map(self.k_scaled)

    def metadata(self):
        return dict(source_representation="legacy_df", set_kind=self.set_kind,
                    nk=self.nk, nao_stored=self.nao, naux_original=self.naux,
                    input_sha256=fingerprint_file(self.input_file),
                    momentum="q=k_j-k_i modulo reciprocal lattice, scaled [0,1)",
                    source_orientation="PySCF sr_loop Q,bra,ket; GREEN oriented pair",
                    gauge_contract=self.gauge_contract)

    def iter_transfers(self):
        return range(len(self.q_scaled))

    def iter_pairs(self, q):
        return ((int(i), int(j)) for i, j in np.argwhere(self.pair_to_q == q))

    def get_pair(self, ki, kj, aux_slice=None):
        if not (0 <= ki < self.nk and 0 <= kj < self.nk):
            raise IndexError("k pair out of range")
        idx = max(ki, kj) * (max(ki, kj) + 1) // 2 + min(ki, kj)
        rep = self.conj[idx] if self.conj[idx] != idx else self.trans[idx]
        if int(rep) not in self.positions:
            raise ValueError("symmetry map has no representative")
        pos = self.positions[int(rep)]
        start = pos // self.chunk_size * self.chunk_size
        selection = slice(None) if aux_slice is None else aux_slice
        with h5py.File(self.directory / f"VQ_{start}.h5", "r") as f:
            d = f[str(start)]
            expected = (self.chunk_size, self.naux, self.nao, 2 * self.nao)
            if d.shape != expected or d.dtype != np.dtype("float64"):
                raise ValueError("wrong packed complex shape/dtype")
            raw = np.ascontiguousarray(d[pos % self.chunk_size, selection, :, :])
        self.bytes_read += raw.nbytes
        value = raw.view(np.complex128).reshape(-1, self.nao, self.nao)
        if ki < kj:
            value = value.conj().transpose(0, 2, 1)
        if self.conj[idx] != idx:
            value = value.conj()
        elif self.trans[idx] != idx:
            value = value.transpose(0, 2, 1)
        if not np.isfinite(value).all():
            raise ValueError("nonfinite source factors")
        return np.ascontiguousarray(value)

    def correction_descriptor(self):
        sidecar = self.directory / "df_ewald.h5"
        return {"df_ewald.h5": fingerprint_file(sidecar)} if sidecar.exists() else {}

    def logical_fingerprint(self):
        digest = hashlib.sha256()
        digest.update(json.dumps(self.metadata(), sort_keys=True).encode())
        digest.update(np.ascontiguousarray(self.k_scaled, dtype=np.float64).tobytes())
        for rep in self.representatives:
            pair = tuple(map(int, self.pairs[rep]))
            digest.update(np.asarray(pair, dtype=np.int64).tobytes())
            digest.update(self.get_pair(*pair).tobytes())
        return digest.hexdigest()


class CanonicalReconstructionSource(LegacyDFSource):
    """A validated SG reader supplies complete oriented original-Q slices.

    The provider owns AO/auxiliary gauge/symmetry reconstruction. This adapter
    never applies its operations to interpolation indices or weights orbits.
    """
    def __init__(self,input_file,directory,provider,set_kind,gauge_contract):
        self.input_file,self.directory=Path(input_file),Path(directory)
        self.set_kind,self.gauge_contract=set_kind,gauge_contract
        self._load_input_maps()
        if (provider.nk,provider.nao,provider.naux)!=(self.nk,self.nao,self.naux):
            raise ValueError("canonical provider dimensions differ")
        coordinates=provider.mesh.coordinates/provider.mesh.denominator
        if not np.allclose((coordinates-self.k_scaled)-np.rint(coordinates-self.k_scaled),0,atol=1e-10,rtol=0):
            raise ValueError("canonical provider k ordering differs")
        if provider.attributes['set_kind']!=set_kind:
            raise ValueError("canonical provider interaction set differs")
        self.provider=provider;self.bytes_read=0

    def metadata(self):
        return dict(super().metadata(),source_representation="space_group_df",
                    source_descriptor_sha256=fingerprint_file(self.directory/'meta.h5'),
                    provider_attributes={key:value.item() if isinstance(value,np.generic) else value for key,value in self.provider.attributes.items()},
                    fitting_objective="every full oriented pair; provider reconstructs each orbit contribution")

    def get_pair(self,ki,kj,aux_slice=None):
        start,stop,step=(slice(0,self.naux) if aux_slice is None else aux_slice).indices(self.naux)
        if step!=1:raise ValueError("canonical provider slices require unit stride")
        result=np.asarray(self.provider.read(ki,kj,output_slice=(start,stop)),dtype=complex)
        if result.shape!=(stop-start,self.nao,self.nao) or not np.isfinite(result).all():
            raise ValueError("invalid canonical provider slice")
        self.bytes_read+=result.nbytes
        return np.ascontiguousarray(result)


class CderiSource:
    """Retained PySCF CDERI adapter. Caller supplies the audited DF object/frame."""
    def __init__(self, mydf, filename, kpts, set_kind="hf", gauge_contract=None):
        if mydf.cell.dimension != 3:
            raise ValueError("signed/low-dimensional metrics unsupported")
        self.mydf, self.filename = mydf, Path(filename)
        if not self.filename.is_file():
            raise FileNotFoundError(self.filename)
        self.k_abs = np.asarray(kpts)
        self.k_scaled = mydf.cell.get_scaled_kpts(kpts)
        self.nk, self.nao = len(kpts), mydf.cell.nao_nr()
        old = mydf._cderi
        try:
            mydf._cderi = str(self.filename)
            self.naux = mydf.get_naoaux()
        finally:
            mydf._cderi = old
        self.set_kind, self.gauge_contract = set_kind, gauge_contract
        self.q_scaled, self.pair_to_q = transfer_map(self.k_scaled)

    def metadata(self):
        return dict(source_representation="pyscf_cderi", set_kind=self.set_kind,
                    nk=self.nk, nao_stored=self.nao, naux_original=self.naux,
                    retained_cderi_sha256=fingerprint_file(self.filename), gauge_contract=self.gauge_contract)

    def correction_descriptor(self):
        return {}

    def iter_transfers(self):
        return range(len(self.q_scaled))

    def iter_pairs(self, q):
        return ((int(i), int(j)) for i, j in np.argwhere(self.pair_to_q == q))

    def get_pair(self, ki, kj, aux_slice=None):
        if not (0 <= ki < self.nk and 0 <= kj < self.nk):
            raise IndexError("k pair out of range")
        old = self.mydf._cderi
        selection = slice(0,self.naux) if aux_slice is None else aux_slice
        start,stop,step = selection.indices(self.naux)
        if step != 1: raise ValueError("CDERI auxiliary slices require unit stride")
        result = np.zeros((max(0,stop-start), self.nao, self.nao), dtype=complex)
        offset = 0
        try:
            self.mydf._cderi = str(self.filename)
            for real, imag, sign in self.mydf.sr_loop((self.k_abs[ki], self.k_abs[kj]), compact=False, max_memory=128):
                if sign != 1:
                    raise ValueError("negative metric is unsupported")
                block = (real + 1j * imag).reshape(-1, self.nao, self.nao)
                if offset + len(block)>self.naux or not np.isfinite(block).all():
                    raise ValueError("invalid retained CDERI rank/factors")
                left,right=max(start,offset),min(stop,offset+len(block))
                if right>left:result[left-start:right-start]=block[left-offset:right-offset]
                offset += len(block)
        finally:
            self.mydf._cderi = old
        return result
