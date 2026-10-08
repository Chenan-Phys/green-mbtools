# Opt-in DF-fitted tensor hypercontraction

This feature is based on GREEN 1.0.0. It evaluates the complete periodic AO
basis on a uniform cell grid, selects a common set of interpolation points by
residual-updating implicit pivoted Cholesky, and fits each original Gaussian
auxiliary column in that separable space. It does not truncate occupations.

The canonical convention is
`L[Q,mu,nu](ki,kj) = sum_I conj(X[ki,I,mu]) X[kj,I,nu] M[q,I,Q]`,
where `q=kj-ki` modulo the reciprocal lattice. HF and correlation share X and
have different M cores. Z=M M^H may be singular; no inverse of Z is required.
The stored-basis conversion is `X_stored=X_AO A^H` when
`L_stored=A L_AO A^H`. Grid weights affect selection only, never saved X.

The standalone CLI leaves existing integral initialization unchanged:

```bash
python -m green_mbtools.mint.thc fit \
  --input-file /reference/input.h5 \
  --source-hf /reference/df_hf_int --source-correlation /reference/df_int \
  --gauge-contract /reference/gauge_contract.json \
  --grid 10 10 10 --n-interp 160 --rcond 1e-12 \
  --atol 1e-8 --rtol 1e-6 --output /new/thc
python -m green_mbtools.mint.thc validate --input /new/thc/hf
python -m green_mbtools.mint.thc expand --input /new/thc/hf --output /new/expanded-hf
```

Cell, overlap, k order and basis are recovered and checked from the actual
serialized input. The earlier proposed `--cell-spec` is unnecessary.
`--gauge-contract` requires separate hf/correlation objects with
`verified_common_q_frame=true`, audited factorization, original rank/order,
conjugate-q relation and correction policy. A same-size assertion alone is not
an audit. Signed metrics, spinors and rectangular transforms are rejected.

Fitting streams orbital-row and auxiliary-column tiles through TSQR and a
rank-revealing SVD. The acceptance gate checks every oriented factor, source-map
covariance, and a preliminary fit excluding deterministic off-diagonal rows.
Held-out cross-pair Grams are reported independently. This holdout is within a
fixed geometry and mesh; it does not establish transfer to another system.
The preliminary fit verifies original-Q pair reversal at the declared fit
tolerances, constrains self-reversed q cores to the real frame, and jointly fits
conjugate q cores. Reversed/Hermitian partner observations can remain in training;
these are excluded-row covariance checks, not unconstrained out-of-sample physics.
This avoids unidentifiable minimum-norm core directions when rows are withheld.
Low rank failures refuse publication. Outputs are immutable and atomically
published, with incomplete staging data retained for diagnosis.

The `green.thc.df_fit` version-1 descriptor contains scalar schema/completion,
accepted-construction/original-Q capability fields, axis/orientation, dimensions,
k/q and source maps, set identity, input fingerprint, factor integrity hashes,
and JSON construction/source/validation provenance. X/M use complex128 packed
as little-endian float64. No legacy `meta.h5` exists until offline expansion.
Original-Q correction sidecars are copied and fingerprinted. SHA-256 provides
Python integrity; portable FNV64 allows a dependency-free C++ integrity check.
Neither is a signature or authentication mechanism.

`CanonicalReconstructionSource` accepts the independently validated SG
`ArchiveReader` API. `--source-format space_group` requires those modules in a
dedicated integration checkout. It reconstructs every logical full pair and
never substitutes scalar orbit multiplicities or Gaussian transforms on I.
RSGDF legacy exports use the ordinary legacy adapter, after their gauge audit;
raw M cores from different builders must not be compared across gauges.
`CderiSource` reads retained positive-metric PySCF files in bounded auxiliary
slices. The CLI export path currently requires GREEN input/maps, rather than
exporting arbitrary standalone CDERI objects.

Run `python -m pytest tests/thc -q`. The fixture generator in `examples/thc`
supports Gamma, complex/shifted meshes, both spin conventions and GF2 sidecars.
Run generation and numerical validation on a compute workstation.
`--basis gth-dzvp --auxiliary default --reciprocal-grid 21` selects a larger
physical fixture with GREEN's augmented even-tempered auxiliary basis;
the default compact auxiliary basis preserves the original small oracle.
Source-map audit tolerances may be declared with `--source-atol` and
`--eri-atol`, and are recorded in provenance. Fitting tolerances are separately
declared by the fit CLI and its acceptance gate remains enforced.
The integral writer zero-pads a corrected diagonal block before replacing bare
rows when the corrected auxiliary metric retains fewer rows. This prevents
stale bare rows in the corrected archive.

Consumers require coordinated THC feature revisions of green-symmetry,
green-mbpt and green-gpu. Use explicit
`--interaction_representation thc --thc_mode reconstruct|native`.
Default DF behavior is preserved. Reconstruction supports CPU HF/GW/GF2 and
GPU HF/GW or GPU-HF/CPU-GF2. Native HF/GW supports scalar full BZ and double
precision (`--P_sp false --Sigma_sp false`). Native GW also supports optional
`--thc_gw_k_contraction fft` on a full closed regular mesh with a declared
workspace budget. CPU momentum FFTs use optional batched FFTW or Eigen fallback;
native GPU stages stay resident and use batched cuFFT. Native GW also supports
`--thc_gw_screening auto|point|auxiliary`, choosing the smaller exact screening
solve in auto mode. Native HF stays
direct. Native GF2, extrapolation/AqQ, point-space symmetry, direct real-space
construction and all-electron adaptation are separate work.

The optional fit flag `--selection df-residual` chooses shared real-space points
by the reduction in the existing DF least-squares residual. The default remains
`pair-density`. Each transfer has a separate orthogonal projection; HF and
correlation targets are normalized by their training Frobenius norms. Selection
stores implicit Gram columns and point-by-Q cross products, rather than a full
grid Gram. The orbital collocation remains separable and cores still share the
audited original auxiliary frame.

The CLI excludes the deterministic modulus-11 held-out DF rows from selection
as well as from training. Perturbing those observations must not change the
chosen points. Full-data refitting, held-out/full factor gates, covariance gates,
precision and atomic export rules remain unchanged. Predicted selection loss
does not establish physical accuracy; compare HF/GW against the original DF
interaction using the same error requirements. Rejected candidates are retained
as evidence and are not exported. Rank reduction is conditional on those gates.

`examples/thc/prepare_fixture.py --replicas N` creates a separate Si supercell
along the first primitive lattice vector, with N in 1..4. Its default remains
one primitive cell. This supports a physical basis-size series independently
of a k-mesh series; all heavy preparation and fitting belongs on the workstation.

If the implicit DF-residual Gram update loses positive semidefiniteness on
nearly dependent points, selection restarts with two-pass modified Gram-Schmidt
in the training feature space. This QR fallback stores selected feature vectors
per q and uses blocked feature contractions; it does not materialize a grid
Gram. It can require more preprocessing work/storage than the implicit method.
Negative residual norms trigger a positive-norm recomputation rather than a
wider numerical tolerance. Metadata records the backend and fallback reason.
Held-out observations, pivot tolerance and export/physical error gates remain
unchanged. Independent least-squares and degenerate-Gamma tests cover this path.
