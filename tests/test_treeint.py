import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression

from treeint.bench import SCENARIOS, make_data, ranking
from treeint.hstat import h_statistic, pd_grid
from treeint.paths import path_lift, path_scores


class Exact:
    """A known function as a 'model', so H can be checked analytically."""

    def __init__(self, f):
        self.f = f

    def predict(self, X):
        return self.f(X)


def test_h_is_zero_for_additive_and_one_for_pure_product():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (80, 3))
    add = h_statistic(Exact(lambda X: X[:, 0] + X[:, 1] ** 2 + X[:, 2]), X, n=80)
    assert max(add.values()) < 1e-10
    prod = h_statistic(Exact(lambda X: X[:, 0] * X[:, 1]), X, n=80, pairs=[(0, 1)])
    assert prod[(0, 1)] > 0.85   # mostly interaction; the sample mean of x1 is not exactly 0, so x0 keeps a small main effect


def test_linear_model_has_no_interaction():
    X, y = make_data(SCENARIOS[0], n=500, seed=1)
    h = h_statistic(LinearRegression().fit(X, y), X, n=60)
    assert max(h.values()) < 1e-8


def test_h_finds_planted_pair_in_boosting():
    sc = SCENARIOS[0]
    X, y = make_data(sc, n=1500, seed=2)
    m = GradientBoostingRegressor(n_estimators=150, max_depth=3, random_state=0).fit(X, y)
    assert ranking(h_statistic(m, X, n=80))[0] == sc.pairs[0]


def test_path_scores_shape_and_symmetry():
    X, y = make_data(SCENARIOS[0], n=800, seed=3)
    m = GradientBoostingRegressor(n_estimators=50, max_depth=3, random_state=0).fit(X, y)
    S = path_scores(m, X.shape[1])
    assert S.shape == (8, 8) and np.allclose(S, S.T) and abs(S.sum() / 2 - 1) < 1e-9
    L = path_lift(m, X.shape[1])
    assert np.all(np.diag(L) == 0) and np.all(L >= 0)


def test_pd_grid_shape():
    rng = np.random.default_rng(0)
    X = rng.uniform(-1, 1, (300, 3))
    g = pd_grid(Exact(lambda X: X[:, 0] * X[:, 1]), X, 0, 1, size=10, n=50)
    assert len(g["z"]) == len(g["x"]) and len(g["z"][0]) == len(g["y"])
