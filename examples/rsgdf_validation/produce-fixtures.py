"""Generate fresh validation inputs on GREEN_workstation, with explicit settings."""
import argparse
import json
import logging
import os
import resource
import time
from pathlib import Path

import pyscf
from green_mbtools.mint import common_utils as comm
from green_mbtools.mint.pyscf_init import pyscf_pbc_init

parser = argparse.ArgumentParser()
parser.add_argument("root", type=Path)
parser.add_argument("--system", choices=["h2", "si"], default="h2")
parser.add_argument("--nk", type=int, default=2)
parser.add_argument("--backend", choices=["ccgdf", "rsgdf"], required=True)
parser.add_argument("--space-symm", choices=["true", "false"], default="false")
parser.add_argument("--eig", choices=["true", "false"], default="true")
parser.add_argument("--precision", type=float)
parser.add_argument("--basis")
options = parser.parse_args()
root = options.root.resolve()
root.mkdir(parents=True, exist_ok=False)
os.chdir(root)
logging.basicConfig(level=logging.INFO)
if options.system == "si":
    lattice = "0 2.7155 2.7155\n2.7155 0 2.7155\n2.7155 2.7155 0"
    atoms = "Si 0 0 0\nSi 1.35775 1.35775 1.35775"
    basis = "gth-dzvp-molopt-sr"
    extra = ["--pseudo", "gth-pbe", "--xc", "PBE"]
else:
    lattice = "4 0 0\n0 4 0\n0 0 4"
    atoms = "H 0 0 0\nH 0 0 0.74"
    basis = "sto3g"
    extra = ["--auxbasis", "weigend"]
params = ["--a", lattice, "--atom", atoms, "--nk", str(options.nk),
          "--basis", options.basis or basis, "--df_backend", options.backend,
          "--space_symm", options.space_symm, "--tr_symm", "true",
          "--use_j2c_eig_decomposition", options.eig, "--keep_cderi", "true",
          "--output_path", str(root/"input.h5"),
          "--int_path", str(root/"df_int"), "--hf_int_path", str(root/"df_hf_int")] + extra
Path("parameters.json").write_text(json.dumps(params, indent=2)+"\n")
start = time.monotonic()
args = comm.init_pbc_params(params)
system = pyscf_pbc_init(args)
if options.precision is not None:
    system.cell.precision = options.precision
    system.cell.mesh = None
    system.cell.rcut = None
    system.cell.build(False, False)
mydf=system.df_object()
mydf._cderi_to_save=str(root/'cderi.h5')
ordinary_start=time.monotonic()
mydf.build()
ordinary_seconds=time.monotonic()-ordinary_start
system.mean_field_input(mydf)
Path("generation.json").write_text(json.dumps({
    "pyscf": pyscf.__version__, "backend": options.backend,
    "system": options.system, "nk": options.nk, "space_symm": options.space_symm,
    "eig": options.eig, "nao": system.cell.nao_nr(), "natoms": system.cell.natm,
    "precision": system.cell.precision, "mesh": system.cell.mesh.tolist(),
    "NQ": mydf.auxcell.nao_nr(), "builder": mydf._green_df_builder_name,
    "ordinary_build_seconds": ordinary_seconds,
    "wall_seconds": time.monotonic()-start,
    "peak_rss_kib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
}, indent=2)+"\n")
