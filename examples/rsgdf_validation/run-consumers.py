"""Bounded solver matrix using one unchanged executable and fresh result dirs."""
import argparse
import itertools
import json
import os
import resource
import shutil
import subprocess
import time
from pathlib import Path

import h5py

p = argparse.ArgumentParser()
p.add_argument("root", type=Path)
p.add_argument("--backend", required=True, choices=["ccgdf", "rsgdf"])
p.add_argument("--only")
p.add_argument("--skip")
args = p.parse_args()
root = args.root.resolve()
input_dir = root / ("h2-" + args.backend)
case_root = root / "consumers"
case_root.mkdir(exist_ok=True)
input_file = case_root / (args.backend + "-frozen-input.h5")
if not input_file.exists():
    shutil.copy2(input_dir/"input.h5", input_file)
    if args.backend == "rsgdf":
        with h5py.File(root/"h2-ccgdf/input.h5", "r") as cc, h5py.File(input_file, "a") as rs:
            del rs["HF"]
            cc.copy("HF", rs)
cases = [("hf-cpu", "HF", "CPU", False, False, True, True),
         ("gw-cpu-dp", "GW", "CPU", False, False, True, True),
         ("gf2-cpu", "GF2", "CPU", False, False, True, True),
         ("gf2-hybrid", "GF2", "GPU", False, False, True, True)]
for host, device in itertools.product((True, False), repeat=2):
    label = f"h{int(host)}d{int(device)}"
    cases.append(("hf-gpu-"+label, "HF", "GPU", False, False, host, device))
    for polarization, sigma in itertools.product((False, True), repeat=2):
        cases.append((f"gw-gpu-{label}-p{int(polarization)}s{int(sigma)}", "GW", "GPU",
                      polarization, sigma, host, device))
ledger = []
for label, solver, kernel, pol, sigma, host, device in cases:
    if args.only and label != args.only:
        continue
    if args.skip and label == args.skip:
        continue
    case = case_root / (args.backend + "-" + label)
    case.mkdir(exist_ok=False)
    boolstr = lambda flag: str(flag).lower()
    executable = Path(os.environ["GREEN_PREFIX"]) / "bin/mbpt.exe"
    command = [str(executable), "--input_file", str(input_file),
               "--dfintegral_hf_file", str(input_dir/"df_hf_int"),
               "--dfintegral_file", str(input_dir/"df_int"),
               "--scf_type", solver, "--kernel", kernel, "--jobs", "SC",
               "--BETA", "100", "--grid_file", str(Path(os.environ["GREEN_PREFIX"])/"share/ir/1e4.h5"),
               "--itermax", "2", "--restart", "false", "--diis_restart", "false",
               "--results_file", str(case/"results.h5"), "--diis_file", str(case/"diis.h5"),
               "--mixing_type", "CDIIS", "--diis_start", "2", "--diis_size", "5",
               "--mixing_weight", "0.3", "--P_sp", boolstr(pol), "--Sigma_sp", boolstr(sigma),
               "--cuda_low_cpu_memory", boolstr(host), "--cuda_low_gpu_memory", boolstr(device)]
    start = time.monotonic()
    with (case/"run.log").open("w") as log:
        try:
            result = subprocess.run(command, cwd=case, stdout=log, stderr=subprocess.STDOUT,
                                    timeout=180, check=False)
            code = result.returncode
        except subprocess.TimeoutExpired:
            code = 124
    entry = {"backend": args.backend, "case": label, "solver": solver, "kernel": kernel,
             "dispatch": "GPU HF + CPU GF2" if solver == "GF2" and kernel == "GPU" else kernel,
             "exit_code": code, "wall_seconds": time.monotonic()-start, "command": command,
             "status": "EXECUTED" if code == 0 else "FAIL"}
    (case/"run.json").write_text(json.dumps(entry, indent=2)+"\n")
    ledger.append(entry)
    print(args.backend, label, entry["status"], flush=True)
(case_root/(args.backend+"-execution-ledger.json")).write_text(json.dumps(ledger, indent=2)+"\n")
if any(entry["exit_code"] for entry in ledger):
    raise SystemExit(1)
