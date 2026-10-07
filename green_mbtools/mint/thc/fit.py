"""Rank-revealing dense reference and bounded row-block TSQR fits."""
import numpy as np
from scipy.linalg import lstsq, qr
from .model import pair_features, FactorModel


def holdout_mask(n, modulus):
    # Keep diagonal and reciprocal rows: a Gamma real-symmetric source has
    # independently unconstrained diagonal entries, even at complete rank.
    rows=np.arange(n*n)
    return (rows % modulus == 1) & (rows//n < rows % n)


def dense_fit(design, rhs, rcond=1e-12):
    core, _, rank, singular = lstsq(design, rhs, cond=rcond, lapack_driver="gelsd")
    return core, dict(effective_rank=int(rank), rcond=float(rcond),
                     singular_values=singular.tolist(),
                     condition_retained=float(singular[0] / singular[rank - 1]) if rank else None)


def tsqr_fit(blocks, rcond=1e-12):
    R = B = None
    discarded = source_norm = 0.0
    for design, rhs in blocks:
        if not np.isfinite(design).all() or not np.isfinite(rhs).all():
            raise ValueError("nonfinite fitting data")
        source_norm += float(np.vdot(rhs, rhs).real)
        A = design if R is None else np.vstack((R, design))
        Y = rhs if B is None else np.vstack((B, rhs))
        Q, R = qr(A, mode="economic")
        B = Q.conj().T @ Y
        # Orthogonal discarded components affect residuals, not the minimizer.
        discarded += max(0.0, float(np.vdot(Y, Y).real - np.vdot(B, B).real))
    if R is None:
        raise ValueError("empty fitting objective")
    core, diagnostics = dense_fit(R, B, rcond)
    residual = discarded + float(np.linalg.norm(R @ core - B) ** 2)
    diagnostics.update(algorithm="blocked_tsqr_svd", discarded_residual_sq=discarded,
                       relative_factor_residual=float(np.sqrt(residual / source_norm)) if source_norm else 0)
    return core, diagnostics


def fit_source(source, X, rcond=1e-12, row_block=512, aux_block=128, holdout_modulus=None, pair_reversal_constraints=False,
               covariance_atol=1e-8,covariance_rtol=1e-6):
    if source.gauge_contract is None or not source.gauge_contract.get("verified_common_q_frame", False):
        raise ValueError("fitting requires an audited common auxiliary frame per q")
    if source.gauge_contract.get("signed_metric", False):
        raise ValueError("signed metrics are unsupported")
    if row_block < 1 or aux_block < 1:
        raise ValueError("block sizes must be positive")
    if pair_reversal_constraints:
        for i in range(source.nk):
            for j in range(source.nk):
                if not np.allclose(source.get_pair(i,j),source.get_pair(j,i).conj().transpose(0,2,1),atol=covariance_atol,rtol=covariance_rtol):
                    raise ValueError("pair-reversal core constraints require verified original-Q conjugacy")
    cores, diagnostics = {}, {}
    for q in source.iter_transfers():
        first=next(source.iter_pairs(q));reverse_q=int(source.pair_to_q[first[1],first[0]])
        core = np.empty((X.shape[1], source.naux), dtype=complex)
        for offset in range(0, source.naux, aux_block):
            stop = min(source.naux, offset + aux_block)
            def blocks():
                pairs=[(i,j,False) for i,j in source.iter_pairs(q)]
                if pair_reversal_constraints and reverse_q!=q:
                    pairs += [(i,j,True) for i,j in source.iter_pairs(reverse_q)]
                for ki, kj, conjugate in pairs:
                    rhs = source.get_pair(ki, kj, slice(offset, stop)).reshape(stop - offset, -1).T
                    n = X.shape[2]
                    for row in range(0, n*n, row_block):
                        indices = np.arange(row, min(n*n, row + row_block))
                        design = (X[ki][:, indices//n].conj() * X[kj][:, indices % n]).T
                        keep = np.ones(len(indices), dtype=bool) if holdout_modulus is None else ~holdout_mask(n,holdout_modulus)[indices]
                        if np.any(keep):
                            A,B=design[keep],rhs[row:row + row_block][keep]
                            if conjugate:A,B=A.conj(),B.conj()
                            if pair_reversal_constraints and reverse_q==q:
                                # q=-q: the verified original-Q Hermitian frame
                                # has real cores. Do not add arbitrary ridge terms.
                                yield A.real,B.real
                                yield A.imag,B.imag
                            else:yield A,B
            fitted, diag = tsqr_fit(blocks(), rcond)
            core[:, offset:stop] = fitted
            diagnostics.setdefault(str(q), []).append(dict(aux_start=offset, aux_stop=stop, **diag))
        cores[q] = core
    metadata = source.metadata()
    if hasattr(source, "logical_fingerprint"):
        metadata["source_integral_sha256"] = source.logical_fingerprint()
    metadata["fit_weights"] = "uniform positive weights on every oriented pair and orbital row per q"
    metadata["pair_reversal_constraints"] = pair_reversal_constraints
    return FactorModel(X, cores, source.pair_to_q, metadata), diagnostics
