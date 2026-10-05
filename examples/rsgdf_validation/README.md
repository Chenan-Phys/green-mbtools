# RSGDF validation utilities

Run these with the Python environment importing this branch. Calculations belong
on a compute workstation, with scratch and outputs outside the source checkout.
All generators refuse to reuse an existing output directory.

```bash
export PYSCF_TMPDIR=/data/$USER/pyscf_tmp/rsgdf-integration
export TMPDIR=$PYSCF_TMPDIR
export PYSCF_MAX_MEMORY=4000 OMP_NUM_THREADS=8
export OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
export GREEN_PREFIX=/path/to/unchanged/green-install
ROOT=/data/$USER/green_runs/new-rsgdf-validation
mkdir "$ROOT"
UTIL=/path/to/green-mbtools/examples/rsgdf_validation

for backend in ccgdf rsgdf; do
    python "$UTIL/produce-fixtures.py" "$ROOT/h2-$backend" --backend "$backend"
    python "$UTIL/run-consumers.py" "$ROOT" --backend "$backend"
done
python "$UTIL/compare-fixtures.py" "$ROOT" --fixtures h2
python "$UTIL/validate-frames.py" "$ROOT" --fixtures h2
python "$UTIL/compare-diagrams.py" "$ROOT"
python "$UTIL/compare-consumers.py" "$ROOT"
```

The matrix uses 24 cases per producer, beta=100, the installed IR grid `1e4.h5`,
and two iterations. This is an agreement test, not a convergence claim. GF2/GPU
means GPU HF plus CPU GF2 correlation. The separate frozen RSGDF input copies
only the CCGDF HF group; original producer files remain intact.

Double-precision checks use atol=1e-8, rtol=1e-7; mixed precision uses atol=2e-6,
rtol=2e-5, declared before comparison. Energy, chemical potential, Sigma1,
dynamic self-energy, all saved G(tau), and electron count are checked. Independent
frozen-G contractions test HF direct/exchange, GW projected polarization and its
spectrum, and GF2 direct/second-order exchange. Individual diagram arrays are
not exported by the unchanged executable; these reference checks supplement its
aggregate outputs.

Si fixtures use the plan's primitive geometry, PBE/gth-pbe and
gth-dzvp-molopt-sr. Generate both `si2tight-*` directories with `--system si
--nk 2 --space-symm true --precision 1e-10`, then compare them with
`compare-fixtures.py --fixtures si2tight`. Precision is a validation utility
option; production defaults are unchanged. `convergence.py ROOT` compares fresh
1e-10 and 1e-12 ordinary builds against existing default-precision `si2-*` files.

`validate-frames.py ROOT --fixtures si2tight --backends rsgdf` reconstructs
polarization from actual factors and inverse overlaps, checking stored spatial
and time-reversal q transformations, including the legacy corrected q=0 sector.
The baseline CCGDF spatial q metadata fails this independent check on Si; this
branch preserves its existing behavior and fixes the RSGDF path only. Use a
CCGDF input with `--space_symm false` as the independent spatial-frame reference.

For bounded memory growth, run `benchmark-build.py ROOT --backend BACKEND
--nk N --label UNIQUE_LABEL` for N=2,3,4, and add a small-mesh larger basis with
`--nk 1 --basis gth-tzv2p`. It records version, actual builder, atoms, AO/auxiliary
counts, precision, retained metric ranks, wall time, peak RSS and file size.
`produce-fixtures.py` separately records ordinary-build and complete initialization
times. `timed-run.py REPORT.json COMMAND...` measures child CPU/wall time/RSS and
preserves the command exit status on Linux.

Keep generated HDF5, timing reports and logs outside Git. Review actual memory
and free disk before increasing the mesh. Do not infer universal robustness or a
speedup from these bounded cases.
