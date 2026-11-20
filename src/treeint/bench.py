"""Synthetic benchmark with planted interactions, so detectors can be scored against the truth."""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from treeint.hstat import h_statistic
from treeint.paths import path_lift, path_scores

FORMS = {
    "product": lambda a, b: 2.0 * a * b,
    "threshold": lambda a, b: 2.0 * ((a > 0) & (b > 0)).astype(float),
    "sine": lambda a, b: 1.5 * np.sin(np.pi * a * b),
}


@dataclass
class Scenario:
    name: str
    pairs: list[tuple[int, int]]          # planted interacting pairs
    forms: list[str]


SCENARIOS = [
    Scenario("one product pair", [(0, 1)], ["product"]),
    Scenario("one threshold pair", [(2, 5)], ["threshold"]),
    Scenario("two pairs (sine + product)", [(0, 3), (4, 6)], ["sine", "product"]),
    Scenario("additive only (no interaction)", [], []),
]


def make_data(sc: Scenario, n: int = 3000, p: int = 8, noise: float = 0.3, seed: int = 0):
    """Strong additive main effects on every feature + the planted interactions."""
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, (n, p))
    coef = rng.uniform(0.5, 1.5, p) * rng.choice([-1, 1], p)
    y = X @ coef + 0.8 * np.sin(2 * X[:, 1]) + 0.8 * X[:, 2] ** 2 + 0.5 * np.abs(X[:, 7])
    for (a, b), form in zip(sc.pairs, sc.forms):
        y = y + FORMS[form](X[:, a], X[:, b])
    return X, y + rng.normal(0, noise, n)


def models(seed: int):
    return {
        "gradient boosting": GradientBoostingRegressor(n_estimators=300, max_depth=3, learning_rate=0.05,
                                                       random_state=seed),
        "random forest": RandomForestRegressor(n_estimators=100, min_samples_leaf=5, n_jobs=-1, random_state=seed),
    }


def ranking(scores: dict[tuple[int, int], float]) -> list[tuple[int, int]]:
    return [k for k, _ in sorted(scores.items(), key=lambda kv: -kv[1])]


def path_dict(model, p: int, lift: bool = False) -> dict[tuple[int, int], float]:
    S = path_lift(model, p) if lift else path_scores(model, p)
    return {(j, k): float(S[j, k]) for j in range(p) for k in range(j + 1, p)}


def run(seeds=range(5), n_h: int = 120) -> list[dict]:
    rows = []
    for sc in SCENARIOS:
        for seed in seeds:
            X, y = make_data(sc, seed=seed)
            for mname, m in models(seed).items():
                m.fit(X, y)
                t0 = time.perf_counter()
                h = h_statistic(m, X, n=n_h, seed=seed)
                t_h = time.perf_counter() - t0
                t0 = time.perf_counter()
                pth = path_dict(m, X.shape[1])
                t_p = time.perf_counter() - t0
                lift = path_dict(m, X.shape[1], lift=True)
                for method, sc_dict, secs in (("H-statistic", h, t_h), ("tree paths", pth, t_p),
                                              ("tree paths, lift", lift, t_p)):
                    rank = ranking(sc_dict)
                    k = len(sc.pairs)
                    planted = set(sc.pairs)
                    top = sc_dict[rank[0]]
                    rows.append({
                        "scenario": sc.name, "seed": seed, "model": mname, "method": method,
                        "recall_at_k": (len(planted & set(rank[:k])) / k) if k else np.nan,
                        "top_score": top,
                        "max_score_other": max(v for kk, v in sc_dict.items() if kk not in planted),
                        "seconds": secs,
                    })
    return rows
