"""A cheap structural interaction score read straight from the trees.

Idea: if feature k is split directly below a split on feature j, the tree is modelling the effect of k
*inside* a region defined by j, which is what an interaction is. For every parent->child pair of
internal nodes on different features we add the child's weighted impurity decrease, and sum over
all trees. It needs no predictions at all, so it is thousands of times cheaper than the H-statistic,
but it also picks up "fake" interactions: a greedy tree splits on j then k even for an additive
function, because it can only carve one axis at a time. The benchmark measures how much that matters.
"""
from __future__ import annotations

import numpy as np


def _estimators(model):
    est = getattr(model, "estimators_", None)
    if est is None:
        raise TypeError("expected a fitted sklearn forest or gradient boosting model")
    return [e for e in np.ravel(est)]


def path_scores(model, n_features: int) -> np.ndarray:
    """Symmetric (p x p) matrix of parent->child split co-occurrence weighted by impurity decrease."""
    S = np.zeros((n_features, n_features))
    for est in _estimators(model):
        t = est.tree_
        left, right, feat = t.children_left, t.children_right, t.feature
        w = t.weighted_n_node_samples
        imp = t.impurity
        for node in range(t.node_count):
            if left[node] == -1:
                continue
            for child in (left[node], right[node]):
                if left[child] == -1 or feat[child] == feat[node]:
                    continue
                gain = w[child] * imp[child] - w[left[child]] * imp[left[child]] - w[right[child]] * imp[right[child]]
                a, b = feat[node], feat[child]
                S[a, b] += gain
                S[b, a] += gain
    total = S.sum() / 2
    return S / total if total > 0 else S


def path_lift(model, n_features: int) -> np.ndarray:
    """Path score divided by what independent splitting would give: S_jk / (u_j * u_k), where u is
    each feature's share of all path gain. A greedy tree that splits j then k only because both
    matter additively gets a lift near 1; a real interaction should stand above that."""
    S = path_scores(model, n_features)
    u = S.sum(axis=1)
    u = u / u.sum() if u.sum() > 0 else u
    with np.errstate(divide="ignore", invalid="ignore"):
        L = np.where(np.outer(u, u) > 0, S / np.outer(u, u), 0.0)
    np.fill_diagonal(L, 0.0)
    return L
