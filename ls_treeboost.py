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
) -> dict:
    """
    Build one feature vector with ragged-edge lags:
    [Z_a, Z_{a-1}, ..., Z_{a-L}] for all columns in feature_cols.
    """
    row = {}
    for L in lags:
        tL = asof_t - pd.DateOffset(months=L)
        vals = Z.loc[tL, feature_cols] if tL in Z.index else pd.Series(index=feature_cols, data=np.nan)

        for col in feature_cols:
            row[f"{col}_lag{L}"] = float(vals[col]) if pd.notna(vals[col]) else np.nan

    return row


def step1_build_supervised_set(
    Z: pd.DataFrame,
    y_monthly: pd.Series,
    target_months: pd.DatetimeIndex,
    asof_rule: str,
    max_lag: int,
    feature_cols: list[str] | None = None,
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

    for t in target_months:
        y_t = y_monthly.loc[t]
        if pd.isna(y_t):
            continue

        a_t = asof_month(t, asof_rule)
        x_t = make_lagged_row(Z, a_t, lags, feature_cols)

        X_rows.append(x_t)
        y_rows.append(float(y_t))
        index.append(t)

    X = pd.DataFrame(X_rows, index=index)
    y = pd.Series(y_rows, index=index, name="GDP_growth")

    # drop rows that are fully NaN (partial NaNs are fine; we impute later)
    mask = ~X.isna().all(axis=1)
    return X.loc[mask], y.loc[mask]


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
        max_features= "sqrt"
    ):
        self.n_estimators = int(n_estimators)          # Max number of boosting iterations (trees)
        self.learning_rate = float(learning_rate)      # Shrinkage: step size for each tree update
        self.max_depth = int(max_depth)                # Tree complexity control (depth)
        self.min_samples_leaf = int(min_samples_leaf)  # Regularization: minimum samples per leaf
        self.subsample = float(subsample)              # Fraction of training data used per tree (stochastic boosting)
        self.random_state = int(random_state)          # Random seed for reproducibility
        self.min_samples_split = int(min_samples_split)
        self.max_features = max_features

        # fitted attributes
        self.init_ = None                              # Initial constant prediction F0 = mean(y_train)
        self.trees_ = []                               # List of fitted regression trees h_m(.)
        self.feature_names_ = None                      # Optional feature names (for interpretability/debugging)
        self.history_ = {"train_mse": [], "val_mse": []}# Store MSE over iterations (train/validation)

        # learned during fit
        self.col_means_ = None                          # Column means (computed on TRAIN only) for NaN imputation
        self.n_estimators_fitted_ = 0                   # Actual number of fitted trees (after early stopping)

    @staticmethod
    def _mse(y_true, y_pred):
        # Mean squared error loss (squared loss used in LS-boosting)
        e = y_true - y_pred
        return float(np.mean(e * e))


    @staticmethod
    def _impute_with_means(X, col_means):
        # Replace missing values with column means learned from training data
        # This avoids data leakage and allows trees to handle ragged-edge inputs

        # Ensure X is a 2D float array
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError("X must be 2D")

        # Work on a copy to avoid modifying the original array
        X_imp = X.copy()

        # Identify missing entries (NaNs)
        nan_mask = np.isnan(X_imp)

        # Replace NaNs column-wise using training-set column means
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

    def fit(
        self,
        X,
        y,
        feature_names: list[str] | None = None,
        val_frac: float = 0.0,
        early_stopping_rounds: int = 0,
    ):
        # Fit LS-TreeBoost model using squared-loss gradient boosting
        # Trees are sequentially fit to residuals of the current model
        # --- cast ---
        if hasattr(X, "values"):                         # If X is a pandas DataFrame
            if feature_names is None and hasattr(X, "columns"):
                feature_names = list(X.columns)          # Keep column names as feature names
            X = X.values                                 # Convert to NumPy array
        X = np.asarray(X, dtype=float)                   # Ensure numeric array

        if hasattr(y, "values"):                         # If y is a pandas Series
            y = y.values                                 # Convert to NumPy
        y = np.asarray(y, dtype=float).reshape(-1)       # Ensure 1D float array

        n, p = X.shape                                   # n samples, p features
        if y.shape[0] != n:
            raise ValueError("X and y must have same number of rows")  # Basic consistency check

        self.feature_names_ = feature_names if feature_names is not None else [f"x{j}" for j in range(p)]  # Names

        if not (0.0 <= val_frac < 1.0):
            raise ValueError("val_frac must be in [0, 1).")  # Validation fraction must be valid

        train_idx, val_idx = self._chronological_split(n, val_frac)  # Chronological split (no shuffling)

        X_train_raw, y_train = X[train_idx], y[train_idx]            # Training data
        X_val_raw, y_val = (X[val_idx], y[val_idx]) if val_idx.size > 0 else (None, None)  # Optional validation

        # --- NaN imputation learned on TRAIN only ---
        col_means = np.nanmean(X_train_raw, axis=0)                  # Column means ignoring NaNs (TRAIN only)
        col_means = np.where(np.isnan(col_means), 0.0, col_means)    # If column is all-NaN, fall back to 0.0
        self.col_means_ = col_means                                  # Store for later imputations (val/test/predict)

        X_train = self._impute_with_means(X_train_raw, self.col_means_)              # Impute train NaNs
        X_val = self._impute_with_means(X_val_raw, self.col_means_) if X_val_raw is not None else None  # Impute val

        # --- init model F0 = mean(y_train) ---
        self.init_ = float(np.mean(y_train))                         # Optimal constant under squared loss
        self.trees_ = []                                             # Reset list of trees
        self.history_ = {"train_mse": [], "val_mse": []}             # Reset history
        self.n_estimators_fitted_ = 0                                # Reset fitted tree count

        pred_train = np.full_like(y_train, self.init_, dtype=float)  # Initial predictions on train: all mean(y_train)
        pred_val = np.full_like(y_val, self.init_, dtype=float) if X_val is not None else None  # Same for val

        self.history_["train_mse"].append(self._mse(y_train, pred_train))  # Baseline train MSE before any trees

        if X_val is not None:
            val_mse0 = self._mse(y_val, pred_val)                    # Baseline validation MSE
            self.history_["val_mse"].append(val_mse0)                # Store baseline val MSE
            best_val = val_mse0                                      # Best val score so far
            best_iter = -1                                           # -1 corresponds to the constant model (no trees)
        else:
            best_val = np.inf                                        # No validation: disable early stopping tracking
            best_iter = -1

        no_improve = 0                                               # Counter for early stopping
        rng = np.random.default_rng(self.random_state)               # RNG for subsampling

        # --- LS-Boost iterations ---
        for m in range(self.n_estimators):                           # Add trees sequentially
            resid = y_train - pred_train                             # Residuals r_m = y - F_{m-1}(x) (negative gradient)

            if self.subsample < 1.0:
                base_k = int(np.floor(self.subsample * len(train_idx)))   # Desired subsample size
                k = max(self.min_samples_leaf * 2, base_k)                # Ensure enough points given min_samples_leaf
                k = min(k, len(train_idx))                                # Cannot sample more than available
                sub_idx = rng.choice(len(train_idx), size=k, replace=False)# Random subsample indices
                X_fit, r_fit = X_train[sub_idx], resid[sub_idx]           # Fit tree on subsample residuals
            else:
                X_fit, r_fit = X_train, resid                             # Fit tree on full training data

            tree = DecisionTreeRegressor(
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                min_samples_split=self.min_samples_split,
                max_features=self.max_features,
                random_state=self.random_state + m,
            )

            tree.fit(X_fit, r_fit)                                     # Fit tree to residuals (weak learner)

            self.trees_.append(tree)                                   # Store fitted tree
            self.n_estimators_fitted_ += 1                             # Increment fitted tree count

            pred_train = pred_train + self.learning_rate * tree.predict(X_train)  # Update: F_m = F_{m-1} + nu*h_m
            self.history_["train_mse"].append(self._mse(y_train, pred_train))     # Track training MSE

    
            if X_val is not None:
                pred_val = pred_val + self.learning_rate * tree.predict(X_val)   # Apply same update on validation set
                val_mse = self._mse(y_val, pred_val)                              # Compute validation MSE
                self.history_["val_mse"].append(val_mse)                          # Track validation MSE

                if early_stopping_rounds > 0:
                    if val_mse < best_val - 1e-12:                                # Improvement threshold
                        best_val = val_mse                                        # Update best val score
                        best_iter = m                                             # Store best tree index
                        no_improve = 0                                            # Reset counter
                    else:
                        no_improve += 1                                           # No improvement this round
                        if no_improve >= early_stopping_rounds:
                            if best_iter == -1:
                                # Best model is the constant baseline (no trees)
                                self.trees_ = []
                                self.n_estimators_fitted_ = 0
                            else:
                                # Keep trees up to best iteration
                                self.trees_ = self.trees_[: best_iter + 1]
                                self.n_estimators_fitted_ = len(self.trees_)
                            break

        return self                                                     # Allow chaining: model.fit(...)

    def predict(self, X):
        if self.init_ is None or self.col_means_ is None:
            raise RuntimeError("Model not fitted yet.")                 # Must call fit before predict

        if hasattr(X, "values"):                                        # If pandas DataFrame
            X = X.values                                                # Convert to NumPy array
        X = np.asarray(X, dtype=float)                                  # Ensure float array

        X_imp = self._impute_with_means(X, self.col_means_)             # Impute NaNs using TRAIN means (no leakage)

        pred = np.full(X_imp.shape[0], self.init_, dtype=float)         # Start from constant model F0
        for tree in self.trees_:
            pred += self.learning_rate * tree.predict(X_imp)            # Add each tree contribution (shrinked)
        return pred                                                     # Final prediction: F0 + nu * sum h_m