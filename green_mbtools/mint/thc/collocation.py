"""Full-orbital Bloch values on a documented uniform cell grid."""
import numpy as np
import h5py


def uniform_grid(cell, shape, origin=(0, 0, 0)):
    shape = np.asarray(shape, dtype=int)
    if shape.shape != (3,) or np.any(shape < 1):
        raise ValueError("grid requires three positive counts")
    scaled = np.indices(shape).reshape(3, -1).T / shape + np.asarray(origin)
    return scaled @ cell.lattice_vectors(), np.full(len(scaled), cell.vol / len(scaled))


def stored_basis_values(ao, forward):
    """L_stored=A L_AO A^H => orbital row X_stored=X_AO A^H."""
    if forward is None:
        return ao
    if forward.shape != (ao.shape[0], ao.shape[2], ao.shape[2]):
        raise ValueError("rectangular/spinor transforms unsupported")
    return np.einsum("kra,kba->krb", ao, forward.conj())


def evaluate(cell, kpts, coords, block_size=512, forward=None):
    if cell.dimension != 3 or block_size < 1:
        raise ValueError("only positive-metric 3D full orbital collocation supported")
    from pyscf.pbc.dft import numint
    result = np.empty((len(kpts), len(coords), cell.nao_nr()), dtype=complex)
    for start in range(0, len(coords), block_size):
        result[:, start:start + block_size] = np.asarray(numint.eval_ao_kpts(cell, coords[start:start + block_size], kpts=kpts))
    return stored_basis_values(result, forward)


def cell_from_input(input_file):
    from pyscf.pbc import gto
    with h5py.File(input_file, "r") as f:
        if "Cell" not in f:
            raise ValueError("archived cell specification missing; regenerate explicit fixture")
        blob = f["Cell"][()]
        cell = gto.loads(blob.decode() if isinstance(blob, bytes) else blob)
        if cell.nao_nr() != int(f["params/nao"][()]):
            raise ValueError("cell orbital count differs from source")
        forward = np.asarray(f["orthogonalization/X_k"]) if "orthogonalization/X_k" in f else None
        overlap = f["HF/S-k"][()]
        overlap = np.ascontiguousarray(overlap).view(complex).reshape(overlap.shape[:-1])[0]
        expected = np.asarray(cell.pbc_intor("int1e_ovlp", hermi=1, kpts=f["symmetry/k/mesh"][()]))
        if forward is not None:
            expected = forward @ expected @ forward.conj().transpose(0, 2, 1)
        if not np.allclose(overlap, expected, atol=1e-9, rtol=1e-9):
            raise ValueError("actual cell/basis/order does not reproduce stored overlap")
    return cell, forward
