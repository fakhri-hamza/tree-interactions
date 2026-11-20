"""Friedman's H-statistic for pairwise feature interactions, from scratch.

For features j and k, with partial dependence functions centred to mean zero,

    H^2_jk = sum_i [PD_jk(x_ij, x_ik) - PD_j(x_ij) - PD_k(x_ik)]^2 / sum_i PD_jk(x_ij, x_ik)^2

H^2 = 0 means the model is additive in j and k; H^2 = 1 means the joint effect is pure interaction.
Partial dependence is computed by brute force on a sample of n rows: PD_S(z) = mean over the sample of
f(x with the features in S set to z). That costs n^2 predictions per pair, so n stays small (100-300).

Friedman, J. H. & Popescu, B. E. (2008). Predictive learning via rule ensembles. Ann. Appl. Stat.
"""
from __future__ import annotations

import itertools

import numpy as np


def _pd(model, X: np.ndarray, feats: tuple[int, ...]) -> np.ndarray:
    """Partial dependence of `feats`, evaluated at each row of X's own values (centred)."""
    n = len(X)
    big = np.repeat(X[None, :, :], n, axis=0)                 # big[i] = X with feats set to row i
    for f in feats:
        big[:, :, f] = X[:, f][:, None]
    out = model.predict(big.reshape(n * n, -1)).reshape(n, n).mean(axis=1)
    return out - out.mean()


def h_statistic(model, X: np.ndarray, n: int = 150, seed: int = 0,
                pairs: list[tuple[int, int]] | None = None) -> dict[tuple[int, int], float]:
    """H^2 for every pair (or the given pairs), on a random sample of n rows."""
    rng = np.random.default_rng(seed)
    Xs = np.asarray(X, dtype=float)
    if len(Xs) > n:
        Xs = Xs[rng.choice(len(Xs), n, replace=False)]
    p = Xs.shape[1]
    pairs = pairs or list(itertools.combinations(range(p), 2))
    one = {f: _pd(model, Xs, (f,)) for f in sorted({f for pr in pairs for f in pr})}
    out = {}
    for j, k in pairs:
        joint = _pd(model, Xs, (j, k))
        denom = float(np.sum(joint ** 2))
        out[(j, k)] = 0.0 if denom == 0 else float(np.sum((joint - one[j] - one[k]) ** 2) / denom)
    return out


def pd_grid(model, X: np.ndarray, j: int, k: int, size: int = 20, n: int = 300, seed: int = 0) -> dict:
    """2-D partial dependence on a size x size quantile grid (for the interactive heatmap)."""
    rng = np.random.default_rng(seed)
    Xs = np.asarray(X, dtype=float)
    if len(Xs) > n:
        Xs = Xs[rng.choice(len(Xs), n, replace=False)]
    qs = np.linspace(0.02, 0.98, size)
    gj, gk = np.unique(np.quantile(X[:, j], qs)), np.unique(np.quantile(X[:, k], qs))
    z = np.empty((len(gj), len(gk)))
    for a, vj in enumerate(gj):
        batch = np.repeat(Xs[None, :, :], len(gk), axis=0)
        batch[:, :, j] = vj
        batch[:, :, k] = gk[:, None]
        z[a] = model.predict(batch.reshape(-1, Xs.shape[1])).reshape(len(gk), -1).mean(axis=1)
    return {"x": gj.round(4).tolist(), "y": gk.round(4).tolist(), "z": z.round(4).tolist()}
