import numpy as np
import pandas as pd

from ls_treeboost import step0_load_state_space, step1_build_supervised_set, LS_treeboost

# =========================
# CONFIG
# =========================
STATE_SPACE_PATH = "data_transformations_DFM_ready_state_space.csv"
GDP_COL = "GrossDomesticProduct_1"

ASOF_RULE = "end"   # choose: "early", "mid", "end"
MAX_LAG = 12

# Early stopping is done INSIDE each expanding-window fit using a small internal validation tail
VAL_FRAC_INNER = 0.1
EARLY_STOPPING_ROUNDS = 50

MODEL_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 2,
    "min_samples_leaf": 10,
    "min_samples_split": 20,
    "max_features": "sqrt", #log2
    "subsample": 0.6,
    "random_state": 42,
}

# How many first observations to use before starting to forecast (must be big enough)
MIN_TRAIN_OBS = 25

# Example: fill these with your actual column names in out0["Z"].columns
RELEASE_LAGS = {
    # financial market series often 0
    "ecb_3M_Yield": 0,
    "ecb_10Y_Yield": 0,
    # surveys often 0 (same month)
    # "PMI": 0,
    # "Confidence": 0,
    # hard macro data often 1-2 (depends on series)
    # "IndustrialProduction": 2,
    # "RetailSales": 2,
}
DEFAULT_RELEASE_LAG = 1   # choose a conservative baseline if you haven't specified a series

# =========================
# METRICS
# =========================
def rmse(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.sqrt(np.mean(e * e)))

def mae(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(np.abs(e)))

def mse(y_true, y_pred):
    e = np.asarray(y_true) - np.asarray(y_pred)
    return float(np.mean(e * e))

# Benchmark model
def ar1_forecast(y_train: pd.Series, y_last: float) -> float:
    """
    Fit AR(1): y_t = alpha + phi y_{t-1} + eps
    and return one-step-ahead forecast.
    """
    y = y_train.values
    y_lag = y[:-1]
    y_now = y[1:]

    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta = np.linalg.lstsq(X, y_now, rcond=None)[0]

    alpha_hat, phi_hat = beta
    return alpha_hat + phi_hat * y_last

# =========================
# EXPANDING WINDOW
# =========================
def expanding_window_nowcast(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """
    For each target date t (after MIN_TRAIN_OBS), fit model on all dates < t, predict y_t.
    Uses inner time-ordered val split (VAL_FRAC_INNER) for early stopping within each fit.
    """
    dates = y.index
    rows = []

    for i in range(MIN_TRAIN_OBS, len(dates)):
        t = dates[i]

        # train: everything strictly before t
        train_dates = dates[:i]
        X_train = X.loc[train_dates]
        y_train = y.loc[train_dates]

        # test: the single point t
        X_test = X.loc[[t]]
        y_true = float(y.loc[t])

        y_last = float(y_train.iloc[-1])
        y_pred_ar1 = ar1_forecast(y_train, y_last)


        model = LS_treeboost(**MODEL_PARAMS)
        model.fit(
            X_train,
            y_train,
            val_frac=VAL_FRAC_INNER,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
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
                "n_train": len(y_train),
                "n_trees": len(model.trees_),
            }
        )
    return pd.DataFrame(rows).set_index("date")


def main():
    # Step 0: load monthly state-space data once
    out0 = step0_load_state_space(
        path_ss=STATE_SPACE_PATH,
        date_col="date",
        gdp_col=GDP_COL,
    )

    # Step 1: build supervised set once (this defines your quarterly target dates/index)
    X, y = step1_build_supervised_set(
        Z=out0["Z"],
        y_monthly=out0["y_monthly"],
        target_months=out0["target_months"],
        asof_rule=ASOF_RULE,
        max_lag=MAX_LAG,
        release_lags=RELEASE_LAGS,
        default_release_lag=DEFAULT_RELEASE_LAG,
    )


    print(f"Dataset: n_obs={len(y)}, n_features={X.shape[1]}")
    print(f"As-of={ASOF_RULE}, max_lag={MAX_LAG}, MIN_TRAIN_OBS={MIN_TRAIN_OBS}")

    # Expanding-window pseudo real-time nowcasts
    df_fc = expanding_window_nowcast(X, y)

    # Benchmarks on the SAME forecasted dates:
    # (1) zero forecast (natural if y is standardized)
    yhat0 = np.zeros(len(df_fc), dtype=float)

    # Summary metrics
    oos_rmse = rmse(df_fc["y_true"], df_fc["y_pred"])
    oos_rmse_ar1 = rmse(df_fc["y_true"], df_fc["y_pred_ar1"])
    oos_mae = mae(df_fc["y_true"], df_fc["y_pred"])
    oos_rmse_zero = rmse(df_fc["y_true"], yhat0)

    # Skill in MSE terms vs zero:
    skill_mse_vs_zero = 1.0 - (mse(df_fc["y_true"], df_fc["y_pred"]) / mse(df_fc["y_true"], yhat0))
    skill_vs_ar1 = 1.0 - (
        mse(df_fc["y_true"], df_fc["y_pred"])
        / mse(df_fc["y_true"], df_fc["y_pred_ar1"])
    )


    print("\n=== EXPANDING-WINDOW (pseudo real-time) results ===")
    print(f"OOS RMSE (model)      : {oos_rmse:.6f}")
    print(f"OOS MAE  (model)      : {oos_mae:.6f}")
    print(f"OOS RMSE (zero bench) : {oos_rmse_zero:.6f}")
    print(f"OOS RMSE (AR(1) bench) : {oos_rmse_ar1:.6f}")
    print(f"Skill (MSE vs zero)   : {skill_mse_vs_zero:.6f}")
    print(f"Skill (MSE vs AR(1))   : {skill_vs_ar1:.6f}")


    # Save per-quarter forecasts
    out_csv = f"ls_treeboost_expanding_{ASOF_RULE}_lag{MAX_LAG}.csv"
    df_fc.to_csv(out_csv)
    print(f"\nSaved: {out_csv}")

    # Optional: also save a compact summary row
    summary = pd.DataFrame(
        [{
            "asof_rule": ASOF_RULE,
            "max_lag": MAX_LAG,
            "min_train_obs": MIN_TRAIN_OBS,
            "val_frac_inner": VAL_FRAC_INNER,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
            "oos_rmse": oos_rmse,
            "oos_mae": oos_mae,
            "oos_rmse_zero": oos_rmse_zero,
            "skill_mse_vs_zero": float(skill_mse_vs_zero),
            "skill_vs_ar1": float(skill_vs_ar1),
            "avg_n_trees": float(df_fc["n_trees"].mean()),
        }]
    )
    summary_out = f"ls_treeboost_expanding_summary_{ASOF_RULE}_lag{MAX_LAG}.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
