import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor


def step0_load_state_space(
    path_ss: str = "data_transformations_DFM_ready_state_space.csv",
    date_col: str = "date",
    gdp_col: str = "GrossDomesticProduct_1",
):
    """
    Step 0 (LS-TreeBoost)
    - Load monthly, standardized state-space dataset
    - Split into predictors Z and target y (GDP has NaNs outside quarter months)
    """
    df = pd.read_csv(path_ss)
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.sort_values(date_col).set_index(date_col)

    if gdp_col not in df.columns:
        raise ValueError(f"GDP column '{gdp_col}' not found in {path_ss}")

    y_monthly = pd.to_numeric(df[gdp_col], errors="coerce")
    Z = df.drop(columns=[gdp_col]).apply(pd.to_numeric, errors="coerce")

    target_months = y_monthly.dropna().index
    return {"df_ss": df, "Z": Z, "y_monthly": y_monthly, "target_months": target_months}


def asof_month(target_month: pd.Timestamp, rule: str) -> pd.Timestamp:
    """
    Determine information cut-off ("as-of") month for nowcasting.

    rule:
      'early' -> first month of quarter   (t-2)
      'mid'   -> second month of quarter  (t-1)
      'end'   -> third month of quarter   (t)
    """
    if rule == "end":
        return target_month
    if rule == "mid":
        return target_month - pd.DateOffset(months=1)
    if rule == "early":
        return target_month - pd.DateOffset(months=2)
    raise ValueError("rule must be 'early', 'mid' or 'end'")


def make_lagged_row(
    Z: pd.DataFrame,
    asof_t: pd.Timestamp,
    lags: list[int],
    feature_cols: list[str],
    release_lags: dict[str, int] | None = None,
    default_release_lag: int = 0,
) -> dict:
    """
    Build one feature vector with ragged-edge lags + publication delays.

    For a variable with publication delay d months:
    at asof_t you only observe months <= asof_t - d.
    So feature col_lagL is available only if L >= d, else NaN.
    """
    if release_lags is None:
        release_lags = {}

    row = {}
    for L in lags:
        tL = asof_t - pd.DateOffset(months=L)
        vals = Z.loc[tL, feature_cols] if tL in Z.index else pd.Series(index=feature_cols, data=np.nan)

        for col in feature_cols:
            d = int(release_lags.get(col, default_release_lag))
            if d < 0:
                raise ValueError(f"release_lag must be >= 0, got {d} for {col}")

            # if this observation would not yet be published at asof_t -> mask
            if L < d:
                row[f"{col}_lag{L}"] = np.nan
            else:
                row[f"{col}_lag{L}"] = float(vals[col]) if pd.notna(vals[col]) else np.nan

    return row

def step1_build_supervised_set(
    Z: pd.DataFrame,
    y_monthly: pd.Series,
    target_months: pd.DatetimeIndex,
    asof_rule: str,
    max_lag: int,
    feature_cols: list[str] | None = None,
    release_lags: dict[str, int] | None = None,
    default_release_lag: int = 0,
):
    """
    Step 1 (LS-TreeBoost)
    - Convert ragged-edge monthly Z into a supervised learning set for quarterly GDP months.
    - Use "as-of" rule + lags up to max_lag.
    """
    if feature_cols is None:
        feature_cols = list(Z.columns)

    lags = list(range(max_lag + 1))
    X_rows, y_rows, index = [], [], []

    # GDP lag (quarterly): y_{t-1} for each quarterly t
    y_q = y_monthly.loc[target_months]          # quarterly GDP observations
    y_lag1_q = y_q.shift(1)                     # y_{t-1}


    for t in target_months:
        y_t = y_monthly.loc[t]
        if pd.isna(y_t):
            continue

        a_t = asof_month(t, asof_rule)
        x_t = make_lagged_row(
            Z, a_t, lags, feature_cols,
            release_lags=release_lags,
            default_release_lag=default_release_lag,
        )
        # add AR(1) input as feature (available in real time: last observed quarter)
        x_t["GDP_lag1"] = float(y_lag1_q.loc[t]) if pd.notna(y_lag1_q.loc[t]) else np.nan

        X_rows.append(x_t)
        y_rows.append(float(y_t))
        index.append(t)

    X = pd.DataFrame(X_rows, index=index)
    y = pd.Series(y_rows, index=index, name="GDP_growth")

    # drop rows that are fully NaN (partial NaNs are fine; we impute later)
    mask = ~X.isna().all(axis=1)
    return X.loc[mask], y.loc[mask]

# =========================
# EXPANDING WINDOW
# =========================
def expanding_window_treeboost_nowcast(
    X: pd.DataFrame,
    y: pd.Series,
    min_train_obs: int,
    model_params: dict,
    val_frac_inner: float,
    early_stopping_rounds: int,
    ar1_func,
):
    """
    For each target date t (after min_train_obs), fit model on all dates < t, predict y_t.
    Uses inner time-ordered val split (val_frac_inner) for early stopping within each fit.

    Notes:
    - Everything required is passed in as arguments (no hidden dependencies on main.py).
    - ar1_func should be a callable: ar1_func(y_train: pd.Series, y_last: float) -> float
    """
    dates = y.index
    rows = []

    for i in range(min_train_obs, len(dates)):
        t = dates[i]

        # train: everything strictly before t
        train_dates = dates[:i]
        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]

        # test: the single point t
        X_test = X.loc[[t]]
        y_true = float(y.loc[t])

        # AR(1) benchmark
        y_last = float(y_train.iloc[-1])
        y_pred_ar1 = float(ar1_func(y_train, y_last))

        # Fit TreeBoost
        model = LS_treeboost(**model_params)
        model.fit(
            X_train,
            y_train,
            val_frac=val_frac_inner,
            early_stopping_rounds=early_stopping_rounds,
        )

        y_pred = float(model.predict(X_test)[0])

        rows.append(
            {
                "date": t,
                "y_true": y_true,
                "y_pred": y_pred,
                "y_pred_ar1": y_pred_ar1,
                "error": y_true - y_pred,
                "error_ar1": y_true - y_pred_ar1,
                "abs_error": abs(y_true - y_pred),
                "squared_error": (y_true - y_pred) ** 2,
                "n_train": int(len(y_train)),
                "n_trees": int(len(model.trees_)),
            }
        )

    return pd.DataFrame(rows).set_index("date")

class LS_treeboost:
    """
    LS-TreeBoost (LS-Boost / L2 gradient boosting with regression trees)

    Key property for ragged-edge macro data:
    - Handles NaNs in X via mean-imputation learned on TRAIN only.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        learning_rate: float = 0.01,
        max_depth: int = 3,
        min_samples_leaf: int = 5,
        subsample: float = 1.0,
        random_state: int = 42,
        min_samples_split: int = 10,
        max_features: str | int | float | None = "sqrt",
        use_ar1_init: bool = True,
    ):
        # -------------------------
        # Hyperparameters
        # -------------------------
        self.n_estimators = int(n_estimators)              # max # boosting iterations (trees)
        self.learning_rate = float(learning_rate)          # shrinkage
        self.max_depth = int(max_depth)                    # tree complexity
        self.min_samples_leaf = int(min_samples_leaf)      # regularization
        self.subsample = float(subsample)                  # stochastic boosting fraction
        self.random_state = int(random_state)              # RNG seed
        self.min_samples_split = int(min_samples_split)    # tree split constraint
        self.max_features = max_features                   # feature subsampling in each split

        # -------------------------
        # Options / switches
        # -------------------------
        self.use_ar1_init_ = bool(use_ar1_init)            # if True: start from AR(1) baseline when GDP_lag1 exists

        # -------------------------
        # Learned / fitted state
        # -------------------------
        self.keep_feat_idx_ = None                         # indices of columns kept after dropping all-NaN train cols
        self.col_means_ = None                             # train-only column means for NaN imputation

        # AR(1) baseline parameters (estimated on train only)
        self.ar1_alpha_ = 0.0
        self.ar1_phi_ = 0.0
        self.ar1_idx_ = None                               # column index of GDP_lag1 after keep_feat_idx_ applied

        # Constant baseline (fallback when AR(1) not used)
        self.init_ = None                                  # mean(y_train)

        # Boosting objects
        self.trees_ = []                                   # list[DecisionTreeRegressor]
        self.feature_names_ = None                          # aligned feature names after dropping columns

        # Training history (for debugging/plots)
        self.history_ = {"train_mse": [], "val_mse": []}

        # Convenience: number of trees actually fitted (after early stopping)
        self.n_estimators_fitted_ = 0
    @staticmethod
    def _mse(y_true, y_pred):
        e = np.asarray(y_true) - np.asarray(y_pred)
        return float(np.mean(e * e))

    @staticmethod
    def _impute_with_means(X, col_means):
        # Replace missing values with column means learned from training data
        # This avoids data leakage and allows trees to handle ragged-edge inputs

        # Ensure X is a 2D float array
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        X_imp = X.copy()
        nan_mask = np.isnan(X_imp)
        if nan_mask.any():
            X_imp[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
        return X_imp

    @staticmethod
    def _chronological_split(n, val_frac):
        # Create a time-ordered train/validation split
        # Training data precedes validation data in time (no shuffling)

        if val_frac <= 0.0:
            return np.arange(n), np.array([], dtype=int)

        n_val = max(1, int(np.floor(n * val_frac)))
        if n - n_val < 5:
            raise ValueError("Too few training samples after split; reduce val_frac.")
        return np.arange(0, n - n_val), np.arange(n - n_val, n)


    def _prepare_X(self, X_train_raw, X_val_raw, feature_names):
        # 1) drop train-all-NaN columns
        all_nan = np.all(np.isnan(X_train_raw), axis=0)
        self.keep_feat_idx_ = np.where(~all_nan)[0]
        X_train_raw = X_train_raw[:, self.keep_feat_idx_]
        if X_val_raw is not None:
            X_val_raw = X_val_raw[:, self.keep_feat_idx_]
        if feature_names is not None:
            feature_names = [feature_names[j] for j in self.keep_feat_idx_]

        # 2) impute means (train only)
        self.col_means_ = np.nanmean(X_train_raw, axis=0)
        self.col_means_ = np.where(np.isnan(self.col_means_), 0.0, self.col_means_)
        X_train = self._impute_with_means(X_train_raw, self.col_means_)
        X_val = self._impute_with_means(X_val_raw, self.col_means_) if X_val_raw is not None else None
        return X_train, X_val, feature_names

    def _fit_baseline(self, X_train, y_train, X_val, y_val, feature_names):
        self.init_ = float(np.mean(y_train))
        self.ar1_idx_ = None
        self.ar1_alpha_, self.ar1_phi_ = 0.0, 0.0

        if self.use_ar1_init_ and feature_names is not None and "GDP_lag1" in feature_names:
            self.ar1_idx_ = feature_names.index("GDP_lag1")
            x = X_train[:, self.ar1_idx_]
            beta = np.linalg.lstsq(np.column_stack([np.ones(len(x)), x]), y_train, rcond=None)[0]
            self.ar1_alpha_, self.ar1_phi_ = float(beta[0]), float(beta[1])
            pred_train = self.ar1_alpha_ + self.ar1_phi_ * x
            pred_val = None if X_val is None else (self.ar1_alpha_ + self.ar1_phi_ * X_val[:, self.ar1_idx_])
        else:
            pred_train = np.full_like(y_train, self.init_, dtype=float)
            pred_val = None if X_val is None else np.full_like(y_val, self.init_, dtype=float)

        return pred_train, pred_val

    def fit(
        self,
        X,
        y,
        feature_names=None,
        val_frac: float = 0.0,
        early_stopping_rounds: int = 0,
    ):
        """
        Fit LS-TreeBoost with squared-loss gradient boosting.
        Workflow:
        1) Cast + chronological train/val split
        2) Prepare X: drop train-all-NaN columns + impute (train means)
        3) Baseline init: AR(1) if GDP_lag1 exists (optional), else constant mean
        4) Boosting: fit trees to residuals, optionally early-stop on val MSE
        """

        # -------------------------
        # 1) Cast inputs
        # -------------------------
        if hasattr(X, "values"):  # pandas DataFrame
            if feature_names is None and hasattr(X, "columns"):
                feature_names = list(X.columns)
            X = X.values
        X = np.asarray(X, dtype=float)

        if hasattr(y, "values"):  # pandas Series
            y = y.values
        y = np.asarray(y, dtype=float).reshape(-1)

        n, p = X.shape
        if y.shape[0] != n:
            raise ValueError("X and y must have same number of rows")

        if not (0.0 <= val_frac < 1.0):
            raise ValueError("val_frac must be in [0, 1).")

        # -------------------------
        # 2) Chronological split (no shuffling)
        # -------------------------
        train_idx, val_idx = self._chronological_split(n, val_frac)

        # Safety: ensure strict chronological order (prevents accidental shuffle later)
        if train_idx.size > 1 and not np.all(train_idx[1:] > train_idx[:-1]):
            raise ValueError("train_idx is not strictly increasing (chronological split violated).")
        if val_idx.size > 1 and not np.all(val_idx[1:] > val_idx[:-1]):
            raise ValueError("val_idx is not strictly increasing (chronological split violated).")
        if val_idx.size > 0 and train_idx.size > 0 and not (train_idx.max() < val_idx.min()):
            raise ValueError("Validation must be strictly after training (chronological split violated).")

        X_train_raw, y_train = X[train_idx], y[train_idx]
        X_val_raw, y_val = (X[val_idx], y[val_idx]) if val_idx.size > 0 else (None, None)

        # If feature_names not given, create generic
        if feature_names is None:
            feature_names = [f"x{j}" for j in range(p)]

        # -------------------------
        # 3) Prepare X (drop all-NaN train columns + impute)
        # -------------------------
        X_train, X_val, feature_names = self._prepare_X(X_train_raw, X_val_raw, feature_names)
        self.feature_names_ = feature_names

        # -------------------------
        # 4) Reset fitted state
        # -------------------------
        self.trees_ = []
        self.history_ = {"train_mse": [], "val_mse": []}
        self.n_estimators_fitted_ = 0

        # -------------------------
        # 5) Baseline init (AR(1) if possible, else constant mean)
        # -------------------------
        pred_train, pred_val = self._fit_baseline(X_train, y_train, X_val, y_val, feature_names)

        # Baseline loss
        self.history_["train_mse"].append(self._mse(y_train, pred_train))

        if X_val is not None:
            val_mse0 = self._mse(y_val, pred_val)
            self.history_["val_mse"].append(val_mse0)
            best_val = val_mse0
            best_trees_len = 0  # number of trees corresponding to best_val so far
        else:
            best_val = np.inf
            best_trees_len = 0

        # Clip magnitude: make it configurable if the attribute exists, else default
        max_update_abs = float(getattr(self, "max_update_abs", 0.5))

        # -------------------------
        # 6) Boosting loop (fit trees to residuals)
        # -------------------------
        no_improve = 0
        rng = np.random.default_rng(self.random_state)

        for m in range(self.n_estimators):
            resid = y_train - pred_train

            # subsample rows if requested
            if self.subsample < 1.0:
                base_k = int(np.floor(self.subsample * len(train_idx)))
                k = max(self.min_samples_leaf * 2, base_k)
                k = min(k, len(train_idx))
                sub_idx = rng.choice(len(train_idx), size=k, replace=False)
                X_fit, r_fit = X_train[sub_idx], resid[sub_idx]
            else:
                X_fit, r_fit = X_train, resid

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + m,
            )
            tree.fit(X_fit, r_fit)

            # --- shrinkage update (clipped) ---
            update = self.learning_rate * tree.predict(X_train)
            if max_update_abs > 0:
                update = np.clip(update, -max_update_abs, max_update_abs)

            pred_train = pred_train + update
            self.history_["train_mse"].append(self._mse(y_train, pred_train))

            self.trees_.append(tree)
            self.n_estimators_fitted_ += 1

            # update val predictions and early stopping
            if X_val is not None:
                update_val = self.learning_rate * tree.predict(X_val)
                if max_update_abs > 0:
                    update_val = np.clip(update_val, -max_update_abs, max_update_abs)
                pred_val = pred_val + update_val

                val_mse = self._mse(y_val, pred_val)
                self.history_["val_mse"].append(val_mse)

                if early_stopping_rounds > 0:
                    tol = 1e-5
                    if val_mse < best_val - tol:
                        best_val = val_mse
                        best_trees_len = len(self.trees_)
                        no_improve = 0
                    else:
                        no_improve += 1
                        if no_improve >= early_stopping_rounds:
                            # keep best iteration (may be baseline-only if best_trees_len == 0)
                            self.trees_ = self.trees_[:best_trees_len]
                            self.n_estimators_fitted_ = len(self.trees_)
                            break

        return self


    def _transform_X(self, X):
        if hasattr(X, "values"):  # pandas DataFrame
            X = X.values
        X = np.asarray(X, dtype=float)

        if self.keep_feat_idx_ is not None:
            X = X[:, self.keep_feat_idx_]

        return self._impute_with_means(X, self.col_means_)

    def predict(self, X):
        if self.init_ is None or self.col_means_ is None:
            raise RuntimeError("Model not fitted yet.")

        X_imp = self._transform_X(X)

        # baseline
        if self.use_ar1_init_ and self.ar1_idx_ is not None:
            pred = self.ar1_alpha_ + self.ar1_phi_ * X_imp[:, self.ar1_idx_]
        else:
            pred = np.full(X_imp.shape[0], self.init_, dtype=float)

        # boosted corrections
        for tree in self.trees_:
            upd = self.learning_rate * tree.predict(X_imp)
            upd = np.clip(upd, -0.5, 0.5)  # zelfde clip als in fit()
            pred += upd

        return pred




    
