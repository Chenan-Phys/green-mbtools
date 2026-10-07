"""Standalone inspect/fit/validate/expand; integral initialization is unchanged."""
import argparse
import json
from pathlib import Path
import tempfile
import os
import time
import numpy as np
from .source import LegacyDFSource, CanonicalReconstructionSource
from .collocation import cell_from_input, uniform_grid, evaluate
from .selection import PairDensityGram, pivoted_cholesky
from .fit import fit_source
from .validate import validate_source, validate_holdout
from . import io


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect")
    inspect.add_argument("--input-dir", required=True)
    inspect.add_argument("--input-file")
    fit = sub.add_parser("fit")
    for name in ("input-file", "source-hf", "source-correlation", "gauge-contract", "output"):
        fit.add_argument("--" + name, required=True)
    fit.add_argument("--grid", type=int, nargs=3, default=[12, 12, 12])
    fit.add_argument("--n-interp", type=int, required=True)
    fit.add_argument("--rcond", type=float, default=1e-12)
    fit.add_argument("--pivot-tolerance", type=float, default=1e-13)
    fit.add_argument("--atol", type=float, default=1e-8)
    fit.add_argument("--rtol", type=float, default=1e-6)
    fit.add_argument("--row-block", type=int, default=512)
    fit.add_argument("--aux-block", type=int, default=128)
    fit.add_argument("--source-format", choices=("legacy", "space_group"), default="legacy")
    validate = sub.add_parser("validate")
    validate.add_argument("--input", required=True)
    validate.add_argument("--input-file")
    expand = sub.add_parser("expand")
    expand.add_argument("--input", required=True)
    expand.add_argument("--output", required=True)
    expand.add_argument("--chunk-size", type=int, default=1)
    args = parser.parse_args(argv)
    if args.command == "inspect":
        directory = Path(args.input_dir)
        if (directory / "thc_meta.h5").exists():
            model = io.read(directory, args.input_file)
            print(json.dumps(model.metadata, indent=2))
        else:
            source = LegacyDFSource(args.input_file or directory.parent / "input.h5", directory)
            print(json.dumps(source.metadata(), indent=2))
        return
    if args.command == "validate":
        model = io.read(args.input, args.input_file)
        print(json.dumps(dict(integrity="passed", approximation=model.metadata["validation"]), indent=2))
        return
    if args.command == "expand":
        io.expand(args.input, args.output, args.chunk_size)
        return
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    if not 0 < args.rcond < 1 or args.atol < 0 or args.rtol < 0:
        raise ValueError("invalid fitting/acceptance thresholds")
    contract = json.loads(Path(args.gauge_contract).read_text())
    cell, forward = cell_from_input(args.input_file)
    sources = {}
    for kind,directory in (("hf",args.source_hf),("correlation",args.source_correlation)):
        if args.source_format=="legacy":sources[kind]=LegacyDFSource(args.input_file,directory,kind,contract[kind])
        else:
            try:
                from green_mbtools.mint.integral_symmetry_io import ArchiveReader, input_fingerprint
            except ImportError as exc:
                raise ValueError("space_group source requires the independently validated SG producer modules in a dedicated integration checkout") from exc
            import h5py
            with h5py.File(args.input_file) as f:kpts=f['symmetry/k/mesh'][()]
            provider=ArchiveReader(directory,expected_fingerprint=input_fingerprint(cell,kpts,forward),expected_set_kind=kind)
            sources[kind]=CanonicalReconstructionSource(args.input_file,directory,provider,kind,contract[kind])
    coords, weights = uniform_grid(cell, args.grid)
    started = time.perf_counter()
    values = evaluate(cell, sources["hf"].k_abs, coords, forward=forward)
    gram = PairDensityGram(values, [(i, j) for i in range(len(values)) for j in range(len(values))], weights)
    points, selection = pivoted_cholesky(gram, args.n_interp, args.pivot_tolerance)
    X = values[:, points, :]
    constructor = dict(grid_shape=args.grid, grid_origin=[0, 0, 0], units="bohr",
                       quadrature_weights=weights[points].tolist(), points=points.tolist(),
                       coordinates=coords[points].tolist(), lattice=cell.lattice_vectors().tolist(),
                       selection=selection, orbital_space="full", orthogonalization="stored A; X_AO A^H" if forward is not None else "AO",
                       cell_serialized=cell.dumps(), selection_seconds=time.perf_counter() - started)
    models, reports = {}, {}
    for kind, source in sources.items():
        started = time.perf_counter()
        training, _ = fit_source(source, X, args.rcond, args.row_block, args.aux_block, holdout_modulus=11,pair_reversal_constraints=True,
                                 covariance_atol=args.atol,covariance_rtol=args.rtol)
        holdout = validate_holdout(training, source, atol=args.atol, rtol=args.rtol)
        model, diagnostics = fit_source(source, X, args.rcond, args.row_block, args.aux_block)
        report = validate_source(model, source, args.atol, args.rtol)
        report["held_out"] = holdout
        report["accepted"] &= holdout["accepted"]
        report.update(fit=diagnostics, elapsed_seconds=time.perf_counter() - started, source_bytes_read=source.bytes_read)
        models[kind], reports[kind] = model, report
    print(json.dumps(dict(selection=selection, sets=reports), indent=2))
    if not all(report["accepted"] for report in reports.values()):
        raise ValueError("rank/grid did not meet declared approximation and source-map covariance tolerances; export rejected")
    output.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=".thc-sets-unpublished-", dir=output.parent))
    for kind in sources:
        io.write(models[kind], stage / kind, args.input_file, sources[kind], reports[kind], constructor)
    (stage / "summary.json").write_text(json.dumps(reports, indent=2) + "\n")
    os.rename(stage, output)


if __name__ == "__main__":
    main()
