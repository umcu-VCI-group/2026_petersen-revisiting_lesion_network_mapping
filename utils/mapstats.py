"""Map similarity statistics: Dice, Pearson, cross-domain matrices, exact Mantel test, phi."""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from config import KEYS, LNM, OS, PALM_DIR, AGG_DIR, sigmap, tmap, sample_ids
import nibabel as nib


def dice(a, b):
    tot = a.sum() + b.sum()
    return 2 * (a & b).sum() / tot if tot > 0 else 0.0


def pearson(x, y):
    v = np.isfinite(x) & np.isfinite(y)
    return float(np.corrcoef(x[v], y[v])[0, 1])


def matrix(fn, rows, cols=None):
    """DataFrame M[i, j] = fn(rows[i], cols[j]); symmetric when cols is None."""
    sym = cols is None
    cols = rows if sym else cols
    M = pd.DataFrame(np.nan, index=list(rows), columns=list(cols))
    for i, a in enumerate(rows):
        for j, b in enumerate(cols):
            M.iloc[i, j] = M.iloc[j, i] if (sym and j < i) else fn(a, b)
    return M


def crossdomain(family, brain, keys=KEYS):
    """Cross-domain Dice (pFDR < 0.05 maps) and Pearson r (unthresholded T) for one family."""
    S = {k: sigmap(family, k, brain) for k in keys}
    T = {k: tmap(family, k, brain) for k in keys}
    D = matrix(lambda a, b: 1.0 if a == b else dice(S[a], S[b]), keys)
    R = matrix(lambda a, b: 1.0 if a == b else pearson(T[a], T[b]), keys)
    return D, R


def offdiag(M):
    v = np.asarray(M, dtype=float)
    return v[np.triu_indices(len(v), 1)]


def mantel(A, B):
    """Exact two-sided Mantel test (Spearman rho) over all permutations of the domain labels."""
    A, B = np.asarray(A, float), np.asarray(B, float)
    iu = np.triu_indices(len(A), 1)
    obs = stats.spearmanr(A[iu], B[iu])[0]
    null = np.array([stats.spearmanr(A[iu], B[np.ix_(p, p)][iu])[0]
                     for p in itertools.permutations(range(len(A)))])
    return obs, float(np.mean(np.abs(null) >= abs(obs) - 1e-12))


def impairment_labels(family=LNM, keys=KEYS):
    """Patients x domains binary impairment labels, read from the PALM designs."""
    cols = []
    for k in keys:
        g = pd.read_csv(PALM_DIR / family / k / "design.csv", header=None).iloc[:, 0].astype(int)
        cols.append(pd.Series(g.values, index=sample_ids(family, k), name=k))
    return pd.concat(cols, axis=1)


def phi_matrix(lab):
    """Pairwise phi (Pearson r of binary labels) on patients with both labels."""
    return matrix(lambda a, b: 1.0 if a == b else lab[[a, b]].dropna().corr().iloc[0, 1], list(lab.columns))


def mean_map(key, family=OS):
    """Voxelwise mean of the lesion-derived connectivity maps of the impaired patients (the input
    of the one-sample t-test); cached in output/voxel_statistics/aggregate_mean/ and recomputed
    when the PALM input is newer than the cache."""
    out = AGG_DIR / f"{family}__{key}_mean.nii.gz"
    src = PALM_DIR / family / key / "input.nii"
    if not out.exists() or out.stat().st_mtime < src.stat().st_mtime:
        AGG_DIR.mkdir(parents=True, exist_ok=True)
        img = nib.load(str(src))
        nib.save(nib.Nifti1Image(np.asarray(img.dataobj, dtype=np.float32).mean(axis=3), img.affine), str(out))
    return out
