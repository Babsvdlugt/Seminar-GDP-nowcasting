import numpy as np
import pandas as pd
from pathlib import Path
from ls_treeboost import LSTreeBoost

GDP_COL = "GrossDomesticProduct_1"
DATA_PATH = Path("/Users/babsvanderlugt/Seminar-GDP-nowcasting/data_transformations_DFM_ready_state_space.csv")

def add_missing_indicators(X: pd.DataFrame) -> pd.DataFrame:
    miss = X.isna().astype(float)
    miss.columns = [f"{c}__isna" for c in miss.columns]
    return pd.concat([X, miss], axis=1)

def impute_train_mean(X_train: pd.DataFrame, X_any: pd.DataFrame):
    means = X_train.mean(axis=0, skipna=True)
    X_train_imp = X_train.fillna(means)
    X_any_imp   = X_any.fillna(means)
    return X_train_imp, X_any_imp, means

# 1) Load
df = (
    pd.read_csv(DATA_PATH, parse_dates=["date"])
      .set_index("date")
      .sort_index()
)

# 2) Features/target (keep NaNs in y for nowcast)
X_all_raw = df.drop(columns=[GDP_COL])
X_all = add_missing_indicators(X_all_raw)
y_all = df[GDP_COL]

# 3) Training sample where GDP known
known_idx = y_all.dropna().index
y_known = y_all.loc[known_idx].astype(float)

# 4) Walk-forward evaluation (last 8 known GDP points)
n_test = min(8, max(1, len(y_known) // 4))
split_point = len(y_known) - n_test

preds, preds_mean = [], []
trues, dates = [], []

for t in range(split_point, len(y_known)):
    train_idx = y_known.index[:t]
    test_idx  = y_known.index[t:t+1]

    X_train = X_all.loc[train_idx]
    y_train = y_all.loc[train_idx].astype(float)

    X_test  = X_all.loc[test_idx]
    y_test  = y_all.loc[test_idx].astype(float)

    # imputatie zonder leakage
    X_train_imp, X_test_imp, _ = impute_train_mean(X_train, X_test)

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

    # benchmark: mean forecast
    yhat_mean = float(np.mean(y_train.values))

    preds.append(yhat)
    preds_mean.append(yhat_mean)
    trues.append(float(y_test.values[0]))
    dates.append(test_idx[0])

eval_df = pd.DataFrame(
    {"y_true": trues, "treeboost": preds, "mean_bench": preds_mean},
    index=pd.to_datetime(dates)
)

def metrics(y_true, y_pred):
    err = y_pred - y_true
    rmse = float(np.sqrt(np.mean(err**2)))
    mae  = float(np.mean(np.abs(err)))
    bias = float(np.mean(err))
    return rmse, mae, bias

rmse_tb, mae_tb, bias_tb = metrics(eval_df["y_true"].values, eval_df["treeboost"].values)
rmse_m,  mae_m,  bias_m  = metrics(eval_df["y_true"].values, eval_df["mean_bench"].values)

print(f"TreeBoost  RMSE: {rmse_tb:.4f} | MAE: {mae_tb:.4f} | bias: {bias_tb:.4f}")
print(f"MeanBench  RMSE: {rmse_m:.4f}  | MAE: {mae_m:.4f}  | bias: {bias_m:.4f}")
print("\nLast evaluation points:")
print(eval_df)

# 5) Fit final model on all known GDP, then nowcast last unknown GDP period
X_train_full = X_all.loc[known_idx]
y_train_full = y_known

X_train_full_imp, X_all_imp, means = impute_train_mean(X_train_full, X_all)

final_model = LSTreeBoost(n_estimators=300, learning_rate=0.05, random_state=42)
final_model.fit(X_train_full_imp.values, y_train_full.values)

unknown_idx = y_all[y_all.isna()].index
nowcast_date = unknown_idx.max() if len(unknown_idx) > 0 else X_all.index.max()

latest_X = X_all_imp.loc[[nowcast_date]].values
gdp_nowcast = final_model.predict(latest_X)[0]
print("\nTreeBoost GDP nowcast date:", nowcast_date.date())
print("TreeBoost GDP nowcast:", gdp_nowcast)
