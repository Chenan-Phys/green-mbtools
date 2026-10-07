"""Atomic, versioned archives. No legacy meta.h5 is emitted for THC factors."""
from pathlib import Path
import json
import os
import shutil
import tempfile
import h5py
import numpy as np
from .model import SCHEMA, VERSION, ORIENTATION, FactorModel, fnv64, fingerprint_file, file_fnv64, transfer_map


def _write_array(path, name, value):
    raw = np.ascontiguousarray(value, dtype=np.complex128).view(np.float64)
    with h5py.File(path, "x") as f:
        f[name] = raw
        f.attrs["fnv64"] = fnv64(raw)
    return fingerprint_file(path)


def write(model, output, input_file, source, validation, construction):
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    if not validation.get("accepted"):
        raise ValueError("fit failed declared factor/covariance acceptance thresholds")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".thc-unpublished-", dir=output.parent))
    try:
        metadata = dict(model.metadata, schema_name=SCHEMA, schema_version=VERSION,
                        representation="thc", construction="df_fit", orientation=ORIENTATION,
                        X_axis_order="k,I,orbital", M_axis_order="I,Q_original",
                        capabilities=["original_Q_reconstruct"], constructor=construction,
                        validation=validation, files={}, correction_sidecars=source.correction_descriptor())
        metadata["files"]["X.h5"] = _write_array(stage / "X.h5", "X", model.X)
        for q, core in model.M.items():
            name = f"M_q_{q}.h5"
            metadata["files"][name] = _write_array(stage / name, "M", core)
        shutil.copyfile(input_file, stage / "input.h5")
        # All original-Q correction sidecars remain immutable, with their identities.
        for name in metadata["correction_sidecars"]:
            shutil.copyfile(source.directory / name, stage / name)
        with h5py.File(stage / "thc_meta.h5", "x") as f:
            for name, value in dict(schema_version=VERSION, complete=0, nk=len(model.X),
                                    nq=len(model.M), nao=model.X.shape[2], n_interp=model.X.shape[1],
                                    naux_original=model.naux).items():
                f[name] = np.int64(value)
            f["schema_name"] = SCHEMA
            f["orientation"] = ORIENTATION
            f["set_kind"] = source.set_kind
            f["construction"] = "df_fit"
            f["fit_accepted"] = np.int64(1)
            f["original_Q_retained"] = np.int64(1)
            f["input_fnv64"] = file_fnv64(input_file)
            f["correction_present"] = np.int64(bool(metadata["correction_sidecars"]))
            if metadata["correction_sidecars"]:
                f["correction_fnv64"] = file_fnv64(stage / "df_ewald.h5")
            f["pair_to_q"] = model.pair_to_q
            f["k_mesh_scaled"] = source.k_scaled
            f["q_mesh_scaled"] = source.q_scaled
            f["source_pairs"] = source.pairs
            f["source_representatives"] = source.representatives
            f["source_conj"] = source.conj
            f["source_trans"] = source.trans
            f["metadata_json"] = json.dumps(metadata, sort_keys=True)
            f["complete"][...] = 1
            f.flush()
        (stage / "validation.json").write_text(json.dumps(validation, indent=2) + "\n")
        # Rename on the same filesystem publishes only complete immutable exports.
        os.rename(stage, output)
    except BaseException:
        # Keep failed staging data for diagnosis. It has no published destination.
        raise


def read(directory, input_file=None):
    directory = Path(directory)
    with h5py.File(directory / "thc_meta.h5", "r") as f:
        if int(f["schema_version"][()]) != VERSION or int(f["complete"][()]) != 1:
            raise ValueError("unknown/incomplete THC schema")
        metadata = json.loads(f["metadata_json"][()].decode())
        if int(f["fit_accepted"][()]) != 1 or int(f["original_Q_retained"][()]) != 1 or f["construction"][()].decode() != "df_fit":
            raise ValueError("unaccepted or unsupported THC construction/capabilities")
        if metadata["schema_name"] != SCHEMA or metadata["orientation"] != ORIENTATION or metadata["construction"] != "df_fit":
            raise ValueError("unsupported THC conventions")
        nk, r, nao, nq, naux = [int(f[n][()]) for n in ("nk", "n_interp", "nao", "nq", "naux_original")]
        pair_to_q = f["pair_to_q"][()]
        k_mesh = f["k_mesh_scaled"][()]
        q_mesh = f["q_mesh_scaled"][()]
        source_maps = {name: f["source_"+name][()] for name in ("pairs", "representatives", "conj", "trans")}
    input_file = directory / "input.h5" if input_file is None else Path(input_file)
    if fingerprint_file(input_file) != metadata["input_sha256"]:
        raise ValueError("mismatched input/orbital basis fingerprint")
    with h5py.File(input_file, "r") as f:
        if (nk, nao, naux) != tuple(int(f["params/"+name][()]) for name in ("nk", "nao", "NQ")):
            raise ValueError("descriptor/input dimensions differ")
        if not np.array_equal(k_mesh, f["symmetry/k/mesh_scaled"][()]):
            raise ValueError("descriptor/input k ordering differs")
        for name, original in dict(pairs="kpair_idx", representatives="kpair_irre_list", conj="conj_pairs_list", trans="trans_pairs_list").items():
            if not np.array_equal(source_maps[name], f["symmetry/pairs/"+original][()]):
                raise ValueError("descriptor/source pair maps differ")
    expected_q, expected_map = transfer_map(k_mesh)
    if not np.array_equal(pair_to_q, expected_map) or not np.allclose(q_mesh, expected_q, atol=1e-12, rtol=0):
        raise ValueError("invalid transfer convention/map")
    if metadata["set_kind"] not in ("hf", "correlation") or not metadata["validation"]["accepted"]:
        raise ValueError("unaccepted or mislabeled interaction")
    for name, expected in metadata["files"].items():
        if Path(name).name != name or fingerprint_file(directory / name) != expected:
            raise ValueError("factor file checksum mismatch")
    for name, expected in metadata["correction_sidecars"].items():
        if fingerprint_file(directory / name) != expected:
            raise ValueError("correction sidecar checksum mismatch")
    def array(name, dataset, shape):
        with h5py.File(directory / name, "r") as f:
            raw = f[dataset][()]
            if raw.dtype != np.dtype("float64") or raw.size != 2 * np.prod(shape) or fnv64(raw) != f.attrs["fnv64"]:
                raise ValueError("factor shape/dtype/checksum mismatch")
        return raw.view(complex).reshape(shape)
    X = array("X.h5", "X", (nk, r, nao))
    M = {q: array(f"M_q_{q}.h5", "M", (r, naux)) for q in range(nq)}
    return FactorModel(X, M, pair_to_q, metadata)


def expand(directory, output, chunk_size=1):
    directory, output = Path(directory), Path(output)
    model = read(directory)
    if output.exists():
        raise FileExistsError(output)
    if chunk_size < 1:
        raise ValueError("invalid expansion chunk size")
    output.parent.mkdir(parents=True,exist_ok=True)
    with h5py.File(directory / "thc_meta.h5", "r") as f:
        pairs, reps = f["source_pairs"][()], f["source_representatives"][()]
    stage = Path(tempfile.mkdtemp(prefix=".expanded-unpublished-", dir=output.parent))
    for start in range(0, len(reps), chunk_size):
        block = np.zeros((chunk_size, model.naux, model.X.shape[2], model.X.shape[2]), dtype=complex)
        for local, rep in enumerate(reps[start:start + chunk_size]):
            block[local] = model.get_pair(*map(int, pairs[rep]))
        with h5py.File(stage / f"VQ_{start}.h5", "x") as f:
            f[str(start)] = block.view(np.float64)
    with h5py.File(stage / "meta.h5", "x") as f:
        f["chunk_size"] = np.int64(chunk_size)
        f["chunk_indices"] = np.arange(0, len(reps), chunk_size, dtype=np.int64)
        f.attrs["__green_version__"] = "1.0.0"
    shutil.copyfile(directory / "input.h5", stage / "input.h5")
    for name in model.metadata["correction_sidecars"]:
        shutil.copyfile(directory / name, stage / name)
    (stage / "thc_expansion.json").write_text(json.dumps(dict(thc_metadata=model.metadata, chunk_size=chunk_size), indent=2))
    os.rename(stage, output)
