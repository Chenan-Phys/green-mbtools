"""Build a small, explicit Si CCGDF fixture with independent adapter/ERI checks.

Run only on GREEN_workstation. Coarse reciprocal cutoffs are validation fixtures,
not a claim of converged silicon observables.
"""
import argparse
import json
import os
from pathlib import Path
import time
import shutil
import numpy as np
import h5py
from pyscf import df as molecular_df
from green_mbtools.mint import common_utils as comm
from green_mbtools.mint.pyscf_init import pyscf_pbc_init
from green_mbtools.mint.thc.source import LegacyDFSource
from green_mbtools.mint.thc.model import fingerprint_file


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--nk", type=int, nargs=3, default=[3,1,1])
    parser.add_argument("--gf2-correction", action="store_true")
    parser.add_argument("--shift", type=float, nargs=3, default=[0.,0.,0.])
    parser.add_argument("--restricted", action="store_true")
    parser.add_argument("--basis", default="gth-szv")
    parser.add_argument("--auxiliary", choices=("compact", "default", "weigend"), default="compact")
    parser.add_argument("--reciprocal-grid", type=int, default=15)
    parser.add_argument("--source-atol", type=float, default=1e-9)
    parser.add_argument("--eri-atol", type=float, default=1e-8)
    parser.add_argument("--resume-validation", action="store_true",
                        help="Validate an existing unfinished fixture without regenerating its integrals")
    args = parser.parse_args()
    output = Path(args.output).resolve()
    if args.resume_validation:
        if not (output/'input.h5').exists() or (output/'gauge_contract.json').exists():
            raise ValueError("resume requires an unfinished generated fixture")
    else:
        output.mkdir(parents=True, exist_ok=False)
    os.chdir(output)
    a = 5.43
    lattice = np.array([[0,a/2,a/2],[a/2,0,a/2],[a/2,a/2,0]])
    params = ["--a", "\n".join(", ".join(map(str,row)) for row in lattice),
              "--atom", f"Si 0 0 0\nSi {a/4} {a/4} {a/4}",
              "--basis", args.basis, "--pseudo", "gth-pade", "--nk", *map(str,args.nk),
              "--Nk", str(args.reciprocal_grid), "--keep_cderi", "true", "--use_j2c_eig_decomposition", "false",
              "--space_symm", "false", "--tr_symm", "false", "--memory", "1"]
    params += ["--shift", *map(str,args.shift)]
    params += ["--restricted", "true" if args.restricted else "false"]
    init_args = comm.init_pbc_params(params=params)
    # Explicit compact auxiliary set makes this a bounded gauge/reader fixture.
    if args.auxiliary == 'compact':
        init_args.auxbasis = {'Si': [[0,[1.,1.]], [0,[.3,1.]], [1,[.5,1.]]]}
    elif args.auxiliary == 'weigend':
        init_args.auxbasis = 'weigend'
    if args.gf2_correction:
        init_args.finite_size_kind = ['gf2']
    started = time.perf_counter()
    producer = pyscf_pbc_init(init_args)
    if not args.resume_validation:
        producer.mean_field_input()
    if args.gf2_correction and not args.resume_validation:
        # GREEN's GF2 consumer reads df_ewald from its correlation path. The
        # release producer emits bare HF factors plus that original-Q sidecar.
        # Publish a distinct fixture correlation directory with those bytes.
        shutil.copytree(output/'df_hf_int',output/'df_int')
    cell, kpts = producer.cell, producer.kmesh
    df = comm.construct_gdf(init_args, cell, kpts)
    df._cderi = str(output / "cderi.h5")
    naux = molecular_df.addons.make_auxmol(cell, df.auxbasis).nao_nr()
    contract, comparisons = {}, {}
    for kind, path in (("hf", "df_hf_int"),("correlation", "df_int")):
        source = LegacyDFSource(output / "input.h5", output / path, kind)
        maximum, signs, active = 0.0, set(), {}
        for i in range(len(kpts)):
            for j in range(len(kpts)):
                provider = df.copy()
                if kind == 'correlation' and i == j and (output / 'cderi_ewald.h5').exists():
                    provider._cderi = str(output / 'cderi_ewald.h5')
                ref = np.zeros((naux,cell.nao_nr(),cell.nao_nr()),complex)
                offset = 0
                for real, imag, sign in provider.sr_loop((kpts[i],kpts[j]),compact=False,max_memory=128):
                    signs.add(int(sign))
                    block = (real + 1j*imag).reshape(-1,cell.nao_nr(),cell.nao_nr())
                    ref[offset:offset+len(block)] = block
                    offset += len(block)
                active.setdefault(str(source.pair_to_q[i,j]), set()).add(offset)
                maximum = max(maximum,float(np.max(abs(ref-source.get_pair(i,j)))))
        if maximum > args.source_atol or signs != {1} or any(len(ranks)!=1 for ranks in active.values()):
            raise ValueError(f"source orientation/frame preflight failed: {kind} {maximum} {active}")
        cross = []
        if kind == 'hf':
            for q in source.iter_transfers():
                pairs = list(source.iter_pairs(q))
                left, right = pairs[0], pairs[-1]
                A = source.get_pair(*left).reshape(naux,-1).T
                B = source.get_pair(*right).reshape(naux,-1).T
                # (ij|lk) maps the conjugate/reversed second pair back to (kl).
                eri = df.get_eri(kpts=[kpts[left[0]],kpts[left[1]],kpts[right[1]],kpts[right[0]]],compact=False)
                mapped = eri.reshape(cell.nao_nr(),cell.nao_nr(),cell.nao_nr(),cell.nao_nr()).transpose(0,1,3,2).reshape(A.shape[0],B.shape[0])
                err = float(np.max(abs(A @ B.conj().T - mapped)))
                cross.append(dict(q=int(q),left=list(left),right=list(right),max_absolute=err))
            if max(row['max_absolute'] for row in cross) > args.eri_atol:
                raise ValueError(f"chemists ERI orientation preflight failed: {cross}")
        contract[kind] = dict(verified_common_q_frame=True, method="CCGDF Cholesky",
                              gauge_policy="original ordered whitened Gaussian Q; no independent pair rotation",
                              active_rank_per_q={q:list(rank)[0] for q,rank in active.items()},
                              naux_padded=naux, signed_metric=False,
                              source_adapter_max_absolute=maximum,
                              declared_source_atol=args.source_atol,declared_eri_atol=args.eri_atol,
                              source_cderi_sha256=fingerprint_file(output / 'cderi.h5'),
                              correction=("GF2 original-Q df_ewald sidecar; bare normal core" if args.gf2_correction else "producer Ewald q0 already included") if kind=='correlation' else "bare plus solver Madelung")
        comparisons[kind] = dict(source_adapter_max_absolute=maximum,chemists_eri=cross)
    (output / 'gauge_contract.json').write_text(json.dumps(contract,indent=2)+'\n')
    (output / 'cell_spec.json').write_text(json.dumps(dict(cell=cell.dumps(),nk=args.nk,shift=args.shift,restricted=args.restricted,
                                                       basis=args.basis,auxiliary=args.auxiliary,auxbasis=df.auxbasis,
                                                       reciprocal_mesh=[args.reciprocal_grid]*3),indent=2)+'\n')
    (output / 'source_validation.json').write_text(json.dumps(dict(comparisons=comparisons,seconds=time.perf_counter()-started),indent=2)+'\n')
    print(json.dumps(comparisons,indent=2))


if __name__ == '__main__':
    main()
