"""Implicit positive weighted pair-density Gram and residual pivot selection."""
import numpy as np
from .fit import holdout_mask


class PairDensityGram:
    def __init__(self, values, pairs, grid_weights=None, pair_weights=None):
        self.values = np.asarray(values)
        self.pairs = list(pairs)
        ng = self.values.shape[1]
        self.grid_sqrt = np.sqrt(np.ones(ng) if grid_weights is None else np.asarray(grid_weights))
        self.weights = np.ones(len(self.pairs)) / len(self.pairs) if pair_weights is None else np.asarray(pair_weights)
        if not self.pairs or np.any(self.weights < 0) or not np.isfinite(self.weights).all() or not np.isfinite(self.grid_sqrt).all():
            raise ValueError("selection weights must be positive finite")

    def diagonal(self):
        norms = np.sum(abs(self.values) ** 2, axis=2)
        return self.grid_sqrt ** 2 * sum(w * norms[i] * norms[j] for (i, j), w in zip(self.pairs, self.weights))

    def column(self, pivot):
        value = np.zeros(len(self.grid_sqrt), dtype=complex)
        for (i, j), w in zip(self.pairs, self.weights):
            left = self.values[i] @ self.values[i, pivot].conj()
            right = self.values[j] @ self.values[j, pivot].conj()
            value += w * left.conj() * right
        return value * self.grid_sqrt * self.grid_sqrt[pivot]


def pivoted_cholesky(oracle, count, relative_tolerance=1e-12, negative_tolerance=1e-10):
    residual = np.asarray(oracle.diagonal(), dtype=float).copy()
    if not 0 < count <= len(residual) or not np.isfinite(residual).all():
        raise ValueError("invalid pivot count/Gram diagonal")
    scale = max(float(residual.max()), np.finfo(float).tiny)
    if residual.min() < -negative_tolerance * scale:
        raise ValueError("Gram has substantially negative diagonal")
    factors = np.zeros((len(residual), count), dtype=complex)
    pivots, history = [], [float(residual.max())]
    for n in range(count):
        pivot = int(np.argmax(residual))
        if residual[pivot] <= relative_tolerance * scale:
            break
        column = oracle.column(pivot) - factors[:, :n] @ factors[pivot, :n].conj()
        factors[:, n] = column / np.sqrt(residual[pivot])
        residual -= abs(factors[:, n]) ** 2
        if residual.min() < -negative_tolerance * scale:
            raise ValueError("Gram deflation lost positive semidefiniteness")
        residual = np.maximum(residual, 0)
        pivots.append(pivot)
        residual[pivots] = 0
        history.append(float(residual.max()))
    if not pivots:
        raise ValueError("zero pair-density Gram")
    return np.array(pivots, dtype=np.int64), dict(method="implicit_residual_pivoted_cholesky", requested_count=count,
                                                   accepted_count=len(pivots), max_residual_history=history,
                                                   relative_tolerance=relative_tolerance)


def df_residual_points(values, sources, count, grid_weights=None,
                       relative_tolerance=1e-13, negative_tolerance=1e-10, holdout_modulus=None):
    """Greedy shared points maximizing the reduction of the DF least-squares loss.

    Each q has its own orthogonal projection and fitted core. HF/correlation
    targets are normalized by their training Frobenius norms so neither set wins
    solely because of its scale. Only implicit Gram columns and point-by-Q
    cross products are stored; no point-by-point Gram is materialized.
    This changes point selection, not the full/held-out export error gates.
    """
    values = np.asarray(values)
    if values.ndim != 3 or not np.isfinite(values).all() or not sources:
        raise ValueError("finite full-orbital values and DF sources are required")
    ng = values.shape[1]
    if not 0 < count <= ng or not 0 < relative_tolerance < 1:
        raise ValueError("invalid residual selection count/tolerance")
    reference = next(iter(sources.values()))
    n=values.shape[2]
    excluded=np.flatnonzero(holdout_mask(n,holdout_modulus)) if holdout_modulus is not None else np.array([],dtype=int)
    def target_rows(source,i,j):
        rhs=source.get_pair(i,j).copy()
        rhs.reshape(source.naux,-1)[:,excluded]=0
        return rhs
    class TrainingGram(PairDensityGram):
        def excluded_features(self,i,j):
            return self.values[i][:,excluded//n].conj()*self.values[j][:,excluded%n]
        def diagonal(self):
            removed=sum(w*np.sum(abs(self.excluded_features(i,j))**2,axis=1)
                        for (i,j),w in zip(self.pairs,self.weights))
            return super().diagonal()-self.grid_sqrt**2*removed
        def column(self,pivot):
            removed=np.zeros(len(self.grid_sqrt),dtype=complex)
            for (i,j),w in zip(self.pairs,self.weights):
                features=self.excluded_features(i,j)
                removed+=w*(features@features[pivot].conj())
            return super().column(pivot)-self.grid_sqrt*self.grid_sqrt[pivot]*removed
    source_norms = {}
    for kind, source in sources.items():
        if not np.array_equal(source.pair_to_q, reference.pair_to_q):
            raise ValueError("residual selection requires shared transfer maps")
        if source.gauge_contract is None or not source.gauge_contract.get("verified_common_q_frame", False) or source.gauge_contract.get("signed_metric",False):
            raise ValueError("residual selection requires an audited common q frame")
        total = 0.
        for q in source.iter_transfers():
            for i, j in source.iter_pairs(q):
                rhs = target_rows(source,i,j)
                total += float(np.vdot(rhs, rhs).real)
        if not np.isfinite(total) or total <= 0:
            raise ValueError("residual selection requires nonzero finite DF targets")
        source_norms[kind] = total
    groups = []
    for q in reference.iter_transfers():
        pairs = list(reference.iter_pairs(q))
        oracle = TrainingGram(values, pairs, grid_weights, np.ones(len(pairs)))
        diagonal = np.asarray(oracle.diagonal(), dtype=float)
        cross = []
        for kind, source in sources.items():
            target = np.zeros((ng, source.naux), dtype=complex)
            for i, j in pairs:
                rhs = target_rows(source,i,j)
                for start in range(0, ng, 128):
                    stop = min(start+128, ng)
                    target[start:stop] += np.einsum("pm,Amn,pn->pA", values[i,start:stop], rhs,
                                                   values[j,start:stop].conj(), optimize=True)
            target *= oracle.grid_sqrt[:,None]/np.sqrt(source_norms[kind])
            cross.append(target)
        groups.append(dict(oracle=oracle, diagonal=diagonal,
                           scale=max(float(diagonal.max()), np.finfo(float).tiny),
                           cross=np.concatenate(cross,axis=1),
                           columns=np.zeros((ng,count),dtype=complex)))
    pivots, gains, remaining = [], [], [float(len(sources))]
    for step in range(count):
        scores = np.zeros(ng)
        for group in groups:
            diagonal = group['diagonal']
            valid = diagonal > relative_tolerance*group['scale']
            scores += np.divide(np.sum(abs(group['cross'])**2,axis=1), diagonal,
                                out=np.zeros(ng),where=valid)
        scores[pivots] = 0
        pivot = int(np.argmax(scores))
        gain = float(scores[pivot])
        if not np.isfinite(gain):
            raise ValueError("nonfinite residual selection score")
        if gain <= np.finfo(float).tiny:
            break
        pivots.append(pivot)
        gains.append(gain)
        remaining.append(max(0.,remaining[-1]-gain))
        for group in groups:
            diagonal, columns = group['diagonal'], group['columns']
            if diagonal[pivot] <= relative_tolerance*group['scale']:
                continue
            # PairDensityGram uses point-row features; the fitting design is
            # their plain transpose, so its Hermitian Gram is conjugated.
            column = group['oracle'].column(pivot).conj()
            column -= columns[:,:step] @ columns[pivot,:step].conj()
            column /= np.sqrt(diagonal[pivot])
            coefficient = group['cross'][pivot].copy()/np.sqrt(diagonal[pivot])
            columns[:,step] = column
            group['cross'] -= column[:,None]*coefficient[None,:]
            diagonal -= abs(column)**2
            if diagonal.min() < -negative_tolerance*group['scale']:
                raise ValueError("residual selection lost positive semidefiniteness")
            np.maximum(diagonal,0,out=diagonal)
            diagonal[pivots] = 0
    if not pivots:
        raise ValueError("zero residual selection objective")
    return np.asarray(pivots,dtype=np.int64), dict(
        method="df_residual_gain",requested_count=count,accepted_count=len(pivots),
        relative_tolerance=relative_tolerance,source_norms_sq=source_norms,
        target_holdout_modulus=holdout_modulus,excluded_target_rows_per_pair=len(excluded),
        normalized_objective_history=remaining,pivot_gains=gains,
        note="Selection loss estimates do not replace held-out/full physical error gates")


def df_residual_points_qr(values, sources, count, grid_weights=None,
                          relative_tolerance=1e-13, holdout_modulus=None):
    """Stable design-space fallback for nearly dependent implicit Gram pivots.

    Two-pass modified Gram-Schmidt stores selected training feature vectors,
    never a grid-by-grid Gram. This uses more arithmetic/storage than the
    implicit path, but avoids squaring the feature condition number. All
    target observations used here exclude the same held-out rows as training.
    """
    values=np.asarray(values)
    if values.ndim!=3 or not np.isfinite(values).all() or not sources:
        raise ValueError('finite full-orbital values and DF sources are required')
    nk,ng,n=values.shape
    if not 0<count<=ng or not 0<relative_tolerance<1:
        raise ValueError('invalid residual selection count/tolerance')
    weights=np.ones(ng) if grid_weights is None else np.asarray(grid_weights)
    if weights.shape!=(ng,) or not np.isfinite(weights).all() or np.any(weights<=0):
        raise ValueError('positive finite grid weights are required')
    sqrt_weights=np.sqrt(weights)
    keep=~holdout_mask(n,holdout_modulus) if holdout_modulus is not None else np.ones(n*n,dtype=bool)
    rows=np.flatnonzero(keep)
    reference=next(iter(sources.values()))
    norms={}
    for kind,source in sources.items():
        if not np.array_equal(source.pair_to_q,reference.pair_to_q):
            raise ValueError('residual selection requires shared transfer maps')
        contract=source.gauge_contract
        if contract is None or not contract.get('verified_common_q_frame',False) or contract.get('signed_metric',False):
            raise ValueError('residual selection requires an audited common q frame')
        norm=sum(np.linalg.norm(source.get_pair(i,j).reshape(source.naux,-1)[:,keep])**2
                 for q in source.iter_transfers() for i,j in source.iter_pairs(q))
        if not np.isfinite(norm) or norm<=0:raise ValueError('nonzero finite DF targets are required')
        norms[kind]=float(norm)
    def features(pairs,start,stop):
        block=np.concatenate([(values[i,start:stop][:,rows//n].conj()*values[j,start:stop][:,rows%n]).T
                              for i,j in pairs],axis=0)*sqrt_weights[None,start:stop]
        return block.astype(complex,copy=False)
    groups=[]
    for q in reference.iter_transfers():
        pairs=list(reference.iter_pairs(q))
        target=np.concatenate([np.concatenate([source.get_pair(i,j).reshape(source.naux,-1).T[keep]
                                              for i,j in pairs],axis=0)/np.sqrt(norms[kind])
                               for kind,source in sources.items()],axis=1)
        diagonal=np.empty(ng);cross=np.empty((ng,target.shape[1]),complex)
        for start in range(0,ng,64):
            stop=min(start+64,ng);block=features(pairs,start,stop)
            diagonal[start:stop]=np.sum(abs(block)**2,axis=0)
            cross[start:stop]=block.conj().T@target
        groups.append(dict(pairs=pairs,target=target,diagonal=diagonal,cross=cross,
                           scale=max(float(diagonal.max()),np.finfo(float).tiny),
                           basis=np.zeros((len(rows)*len(pairs),count),complex)))
    pivots=[];gains=[];history=[float(len(sources))];refreshes=0
    while len(pivots)<count:
        scores=np.zeros(ng)
        for group in groups:
            scores+=np.divide(np.sum(abs(group['cross'])**2,axis=1),group['diagonal'],
                out=np.zeros(ng),where=group['diagonal']>relative_tolerance*group['scale'])
        scores[pivots]=0
        pivot=int(np.argmax(scores))
        if not np.isfinite(scores[pivot]):raise ValueError('nonfinite residual selection score')
        if scores[pivot]<=np.finfo(float).tiny:break
        step=len(pivots);updates=[]
        for group in groups:
            basis=group['basis'][:,:step]
            vector=features(group['pairs'],pivot,pivot+1)[:,0]
            for _ in range(2):vector-=basis@(basis.conj().T@vector)
            norm=float(np.vdot(vector,vector).real)
            group['diagonal'][pivot]=norm
            if norm<=relative_tolerance*group['scale']:continue
            unit=vector/np.sqrt(norm)
            coefficient=unit.conj()@group['target']
            updates.append((group,unit,coefficient))
        if not updates:continue  # Exact training-space norm rejects this pivot.
        pivots.append(pivot)
        gain=sum(float(np.vdot(coefficient,coefficient).real) for _,_,coefficient in updates)
        gains.append(gain);history.append(max(0.,history[-1]-gain))
        for group,unit,coefficient in updates:
            group['basis'][:,step]=unit
            for start in range(0,ng,64):
                stop=min(start+64,ng)
                column=features(group['pairs'],start,stop).conj().T@unit
                group['cross'][start:stop]-=column[:,None]*coefficient[None,:]
                group['diagonal'][start:stop]-=abs(column)**2
            if group['diagonal'].min() < -1e-10*group['scale']:
                # Recompute positive residual norms rather than widening the
                # PSD tolerance or trusting cancellation of two large norms.
                basis=group['basis'][:,:step+1]
                for start in range(0,ng,32):
                    stop=min(start+32,ng);block=features(group['pairs'],start,stop)
                    for _ in range(2):block-=basis@(basis.conj().T@block)
                    group['diagonal'][start:stop]=np.sum(abs(block)**2,axis=0)
                refreshes+=1
            np.maximum(group['diagonal'],0,out=group['diagonal'])
            group['diagonal'][pivots]=0
    if not pivots:raise ValueError('zero residual selection objective')
    return np.asarray(pivots,dtype=np.int64),dict(method='df_residual_gain_reorthogonalized_qr',
        requested_count=count,accepted_count=len(pivots),relative_tolerance=relative_tolerance,
        source_norms_sq=norms,target_holdout_modulus=holdout_modulus,
        excluded_target_rows_per_pair=int((~keep).sum()),normalized_objective_history=history,
        pivot_gains=gains,residual_norm_refreshes=refreshes,
        note='Design-space QR fallback; selection loss does not replace export/physical error gates')
