# Independent symmetry/RSGDF integration branch

`feature/symmetry-rsgdf-integration` combines the symmetry producer with the
separate RSGDF selector branch. It is a third development checkout; neither
independent baseline nor upstream main is merged into by this work.

Select `--df_backend ccgdf` or `--df_backend rsgdf`, independently of
`--integral_symmetry legacy` / `--integral_symmetry space_group`. Defaults retain
legacy storage. SG production additionally requires fresh input, HF, correlation
and `--integral_symmetry_work` paths. Full SCF CDERI and representative integral
caches are separate. The selected representative builder is the actual PySCF
`_CCGDFBuilder` or `_RSGDFBuilder`; provenance records its class and backend.
RSGDF here names a numerical DF algorithm, not a change of physical interaction.

Both ordinary builders capture their actual full-rank positive Cholesky factors.
Default corrected q=0 representatives use GREEN's legacy
`green_igen.df._make_j3c`, with its actual solve captured; nonzero-q pairs use the
selected ordinary backend. Scoped legacy hooks restore class/module ownership
on setup and build failures. Concurrent legacy builds in that capture context
are unsupported. No global scipy function is patched.

On a small p-orbital diamond fixture, all four CCGDF/RSGDF × legacy/SG producers
passed fixed-q cross-pair ERI comparisons. CPU/GPU HF/GW, CPU GF2 and hybrid
GPU-HF/CPU-GF2 production kernels passed with arbitrary complex spin/k inputs.
The RSGDF vs CCGDF producer differences use a separate cutoff tolerance from
storage reconstruction. Truncated/negative/ED metric frames remain unsupported;
the standard Si auxiliary fixture exhibited retained ranks 180/183 of 200 and
was not presented as supported SG production. Finite-grid stabilizer covariance
is also required; positive rank alone is insufficient.

Schema, minimum consumers and supported profiles are described in
[`integral_symmetry_format.md`](../green_mbtools/mint/integral_symmetry_format.md).
No pull request or main-branch merge has been performed.
