# THC source integration snapshot

This branch combines the committed source-provider snapshot
`dea98851a22e43bcb884e01851310d2d52afd608` with the independently tested THC
feature. It leaves the active RSGDF and space-group development branches and
their environments separate. The primary THC branch starts at GREEN v1.0.0.

The validated `space_group.ArchiveReader.read(i,j,output_slice=(start,stop))`
contract reconstructs canonical original-Q blocks. The THC fit uses every
logical oriented pair, shares its selected physical collocation points between
HF and correlation, and keeps the original auxiliary index. Point indices are
not Gaussian auxiliary indices or an assumed symmetry representation.

Read-only CCGDF and RSGDF legacy/space-group archives were fitted at matching
rank/grid thresholds. Reconstructed CPU HF/GW/GF2 and GPU HF/GW or hybrid
GPU-HF/CPU-GF2 consumers passed. Native HF/GW passed on the archives' full-BZ
one-body input. This does not establish native point-space symmetry reduction.

The RSGDF producer in this snapshot predates the later cache/reuse changes at
`0d122ede9ded9e49a563bd61633dc00b4b73d4fa`. Those producer changes are not
imported or validated here; the integration consumes existing immutable
archives. Compare resulting interactions or solver observables across builders,
never raw M matrices from different auxiliary gauges.

See `docs/thc.md` for the schema, fitting gates and consumer activation.
