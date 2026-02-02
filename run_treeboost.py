import numpy as np
import pandas as pd
from pathlib import Path
from ls_treeboost import LSTreeBoost

# ============================================================
# Settings
# ============================================================
GDP_COL = "GrossDomesticProduct_1"
DATA_PATH = Path("/Users/babsvanderlugt/Seminar-GDP-nowcasting/data_transformations_DFM_ready_state_space.csv")

# ============================================================
# Helpers
# ============================================================
def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    """
    Add ragged-edge missingness indicators.

    For each feature x, create a binary indicator x__isna in {0,1}.
    This allows the ML model to exploit real-time release patterns, which are often informative
    in macro nowcasting settings.
    """
    miss = X.isna().astype(float)
    miss.columns = [f"{c}__isna" for c in miss.columns]
    return pd.concat([X, miss], axis=1)

def impute_train_mean(X_train: pd.DataFrame, X_any: pd.DataFrame):
    """
    Leakage-safe imputation for models that cannot handle NaNs.

    - Compute per-feature means using the TRAINING sample only.
    - Use those means to fill missing values in both training and test sets.

    If a feature is all-missing in training, its mean stays NaN; we set it to 0.0.
    This is safe here because the dataset is z-scored (0 corresponds to the mean).
    """
    means = X_train.mean(axis=0, skipna=True).fillna(0.0)
    X_train_imp = X_train.fillna(means)
    X_any_imp   = X_any.fillna(means)
    return X_train_imp, X_any_imp, means

def metrics(y_true, y_pred):
    """Compute RMSE, MAE, and bias (mean(pred-true))."""
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return rmse, mae, bias

# ============================================================
# 1) Load transformed dataset
# ============================================================
# The input file is assumed to be pre-transformed and standardised:
# - Monthly indicators: levels / diff / log-diff as chosen in preprocessing
# - GDP: quarterly log-diff (QoQ log growth), mapped to monthly grid only at release dates
# - All series: z-score standardised (skipna), leaving ragged-edge missings in place
df = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
      .set_index("date")
      .sort_index()
)

# ============================================================
# 2) Build feature matrix and target series
# ============================================================
X_all_raw = df.drop(columns=[GDP_COL])
X_all = add_missing_indicators(X_all_raw)

# GDP is NaN in non-release months by construction (ragged edge preserved)
y_all = df[GDP_COL]

# Use only GDP release dates for training/evaluation
known_idx = y_all.dropna().index
y_known = y_all.loc[known_idx].astype(float)

# ============================================================
# 3) Walk-forward (expanding window) evaluation
# ============================================================
# Evaluate on the last n_test GDP release dates
n_test = min(8, max(1, len(y_known) // 4))
split_point = len(y_known) - n_test

preds_treeboost, preds_mean = [], []
trues, dates = [], []

for t in range(split_point, len(y_known)):
    # Expanding window: train on all release dates up to t-1, test at t
    train_idx = y_known.index[:t]
    test_idx  = y_known.index[t:t+1]

    X_train = X_all.loc[train_idx]
    y_train = y_all.loc[train_idx].astype(float)

    X_test  = X_all.loc[test_idx]
    y_test  = y_all.loc[test_idx].astype(float)

    # Leakage-safe imputation (train means only)
    X_train_imp, X_test_imp, _ = impute_train_mean(X_train, X_test)

    # Fit LS TreeBoost (least-squares gradient boosting; Friedman, 2001)
    model = LSTreeBoost(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        min_samples_leaf=5,
        subsample=1.0,
        random_state=42
    )
    model.fit(X_train_imp.values, y_train.values)
    yhat = model.predict(X_test_imp.values)[0]

    # Mean benchmark: constant forecast equal to the training mean of GDP
    yhat_mean = float(np.mean(y_train.values))

    preds_treeboost.append(yhat)
    preds_mean.append(yhat_mean)
    trues.append(float(y_test.values[0]))
    dates.append(test_idx[0])

eval_df = pd.DataFrame(
    {"y_true": trues, "treeboost": preds_treeboost, "mean_bench": preds_mean},
    index=pd.to_datetime(dates)
)

rmse_tb, mae_tb, bias_tb = metrics(eval_df["y_true"].values, eval_df["treeboost"].values)
rmse_m,  mae_m,  bias_m  = metrics(eval_df["y_true"].values, eval_df["mean_bench"].values)

print(f"TreeBoost  RMSE: {rmse_tb:.4f} | MAE: {mae_tb:.4f} | bias: {bias_tb:.4f}")
print(f"MeanBench  RMSE: {rmse_m:.4f}  | MAE: {mae_m:.4f}  | bias: {bias_m:.4f}")
print("\nEvaluation points:")
print(eval_df)

# ============================================================
# 4) Final fit + nowcast
# ============================================================
# Fit on all available GDP release dates
X_train_full = X_all.loc[known_idx]
y_train_full = y_known

X_train_full_imp, X_all_imp, _ = impute_train_mean(X_train_full, X_all)

final_model = LSTreeBoost(n_estimators=300, learning_rate=0.05, random_state=42)
final_model.fit(X_train_full_imp.values, y_train_full.values)

# Nowcast the most recent date where GDP is missing (preferred)
unknown_idx = y_all[y_all.isna()].index
nowcast_date = unknown_idx.max() if len(unknown_idx) > 0 else X_all.index.max()

latest_X = X_all_imp.loc[[nowcast_date]].values
gdp_nowcast = final_model.predict(latest_X)[0]

print("\nTreeBoost GDP nowcast date:", nowcast_date.date())
print("TreeBoost GDP nowcast:", gdp_nowcast)
