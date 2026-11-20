"""Run the planted-interaction benchmark and the California housing case study.

    python scripts/run_all.py            # ~5 minutes on a laptop
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from treeint.bench import run  # noqa: E402
from treeint.hstat import h_statistic, pd_grid  # noqa: E402
from treeint.paths import path_scores  # noqa: E402
from treeint.report import build  # noqa: E402

OUT = ROOT / "results"
OUT.mkdir(exist_ok=True)

# 1. planted interactions ------------------------------------------------------------------------
t0 = time.time()
df = pd.DataFrame(run(seeds=range(5)))
df.to_csv(OUT / "benchmark_raw.csv", index=False)
summ = (df.groupby(["scenario", "model", "method"], sort=False)
          .agg(recall_at_k=("recall_at_k", "mean"), top_score=("top_score", "mean"),
               best_wrong_pair=("max_score_other", "mean"), seconds=("seconds", "mean"))
          .round(3).reset_index())
(OUT / "benchmark_summary.md").write_text(summ.to_string(index=False) + "\n", encoding="utf-8")
print(summ.to_string(index=False))
print(f"benchmark: {time.time() - t0:.0f}s")

# 2. California housing ----------------------------------------------------------------------------
d = fetch_california_housing()
X, y, names = d.data, d.target, list(d.feature_names)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.25, random_state=0)
m = GradientBoostingRegressor(n_estimators=400, max_depth=4, learning_rate=0.05, subsample=0.8, random_state=0)
m.fit(Xtr, ytr)
r2 = m.score(Xte, yte)
p = X.shape[1]
h = h_statistic(m, Xte, n=200, seed=0)
H = np.zeros((p, p))
for (j, k), v in h.items():
    H[j, k] = H[k, j] = v
P = path_scores(m, p)
top = sorted(h.items(), key=lambda kv: -kv[1])[:6]
grids = {f"{j}-{k}": pd_grid(m, Xte, j, k) for (j, k), _ in top}
html = build("Feature interactions in a gradient-boosted model (California housing)",
             f"GradientBoostingRegressor, test R² = {r2:.3f}. Left: Friedman H² on 200 test rows. "
             "Middle: split-path score. Right: 2-D partial dependence of the selected pair.",
             names, H.round(4).tolist(), P.round(4).tolist(), grids, [list(k) for k, _ in top])
(OUT / "california_interactions.html").write_text(html, encoding="utf-8")
lines = [f"test R2 = {r2:.3f}", "", "top pairs by H^2 | H^2 | path score"]
for (j, k), v in top:
    lines.append(f"{names[j]} x {names[k]} | {v:.3f} | {P[j, k]:.3f}")
ptop = sorted(((P[j, k], j, k) for j in range(p) for k in range(j + 1, p)), reverse=True)[:6]
lines += ["", "top pairs by path score | path | H^2"] + [f"{names[j]} x {names[k]} | {s:.3f} | {H[j, k]:.3f}"
                                                       for s, j, k in ptop]
(OUT / "california_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
print("\n".join(lines))
