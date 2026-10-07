"""Separate approximation, covariance, and cross-pair Gram error gates."""
import numpy as np
from .fit import holdout_mask


def validate_holdout(model, source, modulus=11, atol=1e-8, rtol=1e-6):
    """Deterministic off-diagonal rows were excluded from every training pair.

    This is a deterministic orbital-row holdout, not a new geometry/k-mesh.
    Cross-pair Grams below use only those excluded rows on both sides.
    """
    records = {}; accepted = True
    mask = holdout_mask(source.nao,modulus)
    for q in source.iter_transfers():
        pairs = list(source.iter_pairs(q)); refs = []; fits = []
        for pair in pairs:
            ref = source.get_pair(*pair).reshape(source.naux, -1)[:, mask].T
            val = model.get_pair(*pair).reshape(source.naux, -1)[:, mask].T
            refs.append(ref); fits.append(val)
            accepted &= np.allclose(ref, val, atol=atol, rtol=rtol)
        cross = [error(refs[a] @ refs[(a+1)%len(pairs)].conj().T,
                       fits[a] @ fits[(a+1)%len(pairs)].conj().T) for a in range(len(pairs))]
        records[str(q)] = dict(factor=error(np.concatenate(refs), np.concatenate(fits)), cross_pair_gram=cross)
    return dict(accepted=bool(accepted), type="excluded orbital rows before full-data refit",
                modulus=modulus, excluded_rows_per_pair=int(mask.sum()), transfers=records,
                pair_reversal_constraints=model.metadata.get("pair_reversal_constraints",False),
                independence="excluded observations; verified reversed/Hermitian partners may remain in training")


def error(reference, candidate):
    delta = candidate - reference
    norm = float(np.linalg.norm(reference))
    absolute = abs(delta).ravel()
    return dict(max_absolute=float(absolute.max(initial=0)), rms=float(np.sqrt(np.mean(absolute ** 2))),
                relative_frobenius=float(np.linalg.norm(delta) / norm) if norm else float(np.linalg.norm(delta)),
                absolute_p50=float(np.quantile(absolute, .5)), absolute_p95=float(np.quantile(absolute, .95)))


def validate_source(model, source, atol=1e-8, rtol=1e-6, cross_samples=4):
    records, accepted = {}, True
    for q in source.iter_transfers():
        pairs = list(source.iter_pairs(q))
        numerator = denominator = 0.0
        worst = 0.0
        pair_records = []
        for i, j in pairs:
            ref, fit = source.get_pair(i, j), model.get_pair(i, j)
            diag = error(ref, fit)
            pair_records.append(dict(pair=[i, j], **diag))
            numerator += float(np.linalg.norm(ref - fit) ** 2)
            denominator += float(np.linalg.norm(ref) ** 2)
            worst = max(worst, diag["max_absolute"])
            accepted &= bool(np.allclose(ref, fit, atol=atol, rtol=rtol))
        cross = []
        # Distinct oriented pairs, with original auxiliary frame explicitly shared.
        for a in range(min(len(pairs), cross_samples)):
            b = (a + 1) % len(pairs)
            l = source.get_pair(*pairs[a]).reshape(source.naux, -1).T
            r = source.get_pair(*pairs[b]).reshape(source.naux, -1).T
            lf = model.get_pair(*pairs[a]).reshape(source.naux, -1).T
            rf = model.get_pair(*pairs[b]).reshape(source.naux, -1).T
            cross.append(dict(left=list(pairs[a]), right=list(pairs[b]), **error(l @ r.conj().T, lf @ rf.conj().T)))
        Z = model.Z(q)
        eigen = np.linalg.eigvalsh(Z)
        records[str(q)] = dict(relative_factor_residual=float(np.sqrt(numerator / denominator)) if denominator else 0,
                               worst_absolute=worst, pairs=pair_records, cross_pair_gram=cross,
                               Z_hermiticity=float(np.linalg.norm(Z - Z.conj().T)),
                               Z_min_eigenvalue=float(eigen.min()))
    covariance = validate_covariance(model, source)
    accepted &= covariance["max_absolute"] <= atol + rtol * covariance["reference_max"]
    return dict(accepted=bool(accepted), atol=atol, rtol=rtol, transfers=records,
                source_map_covariance=covariance, held_out="all logical pairs checked after fitting; no independent holdout in this run")


def validate_covariance(model, source):
    maximum, refmax = 0.0, 0.0
    for i in range(source.nk):
        for j in range(source.nk):
            idx = max(i, j) * (max(i, j) + 1) // 2 + min(i, j)
            rep = source.conj[idx] if source.conj[idx] != idx else source.trans[idx]
            ri, rj = source.pairs[rep]
            value = model.get_pair(int(ri), int(rj))
            if i < j:
                value = value.conj().transpose(0, 2, 1)
            if source.conj[idx] != idx:
                value = value.conj()
            elif source.trans[idx] != idx:
                value = value.transpose(0, 2, 1)
            actual = model.get_pair(i, j)
            maximum = max(maximum, float(np.max(abs(value - actual))))
            refmax = max(refmax, float(np.max(abs(actual))))
    return dict(max_absolute=maximum, reference_max=refmax)
