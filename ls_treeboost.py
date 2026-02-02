import numpy as np
from sklearn.tree import DecisionTreeRegressor

class LSTreeBoost:
    """
    Least-Squares TreeBoost (gradient boosting with squared error loss).

    f0(x) = mean(y)
    For m=1..M:
        r = y - f_{m-1}(x)
        fit tree h_m on (X, r)
        f_m(x) = f_{m-1}(x) + nu * h_m(x)
    """
    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.05,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        random_state: int = 42,
    ):
        self.n_estimators = int(n_estimators)
        self.learning_rate = float(learning_rate)
        self.max_depth = int(max_depth)
        self.min_samples_leaf = int(min_samples_leaf)
        self.subsample = float(subsample)
        self.random_state = int(random_state)

        self.init_ = None
        self.trees_ = []
        self.feature_names_ = None

    def fit(self, X, y, feature_names=None):
        X = np.asarray(X)
        y = np.asarray(y).astype(float)

        if X.ndim != 2:
            raise ValueError("X must be 2D array-like.")
        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError("y must be 1D with same number of rows as X.")

        rng = np.random.RandomState(self.random_state)

        # f0 = argmin_c sum (y-c)^2 = mean(y)
        self.init_ = float(np.mean(y))
        pred = np.full_like(y, fill_value=self.init_, dtype=float)

        self.trees_ = []
        self.feature_names_ = feature_names

        n = X.shape[0]

        for m in range(self.n_estimators):
            resid = y - pred  # pseudo-residuals under squared loss

            # Optional stochastic gradient boosting via subsampling
            if self.subsample < 1.0:
                idx = rng.choice(n, size=int(np.floor(self.subsample * n)), replace=False)
                X_m = X[idx]
                r_m = resid[idx]
            else:
                X_m = X
                r_m = resid

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                random_state=rng.randint(0, 1_000_000),
            )
            tree.fit(X_m, r_m)

            update = tree.predict(X)
            pred += self.learning_rate * update

            self.trees_.append(tree)

        return self

    def predict(self, X):
        if self.init_ is None or len(self.trees_) == 0:
            raise ValueError("Model is not fitted yet.")
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(1, -1)

        pred = np.full((X.shape[0],), fill_value=self.init_, dtype=float)
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X)
        return pred
