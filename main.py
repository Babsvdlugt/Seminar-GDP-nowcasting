import numpy as np
import pandas as pd

from ls_treeboost import (
    step0_load_state_space,
    step1_build_supervised_set,
    expanding_window_treeboost_nowcast,
    LS_treeboost,
)

from DFM_Model import expanding_window_dfm_nowcast  # load_dfm_ready not needed if we use out0

# =========================
# CONFIG
# =========================
USE_DFM = False  # zet op False om DFM uit te schakelen

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
    "max_features": "sqrt",  # "log2"
    "subsample": 0.6,
    "random_state": 42,
}

# How many first observations to use before starting to forecast (must be big enough)
MIN_TRAIN_OBS = 25

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
DEFAULT_RELEASE_LAG = 1


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
    return float(alpha_hat + phi_hat * y_last)


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

    # -------------------------
    # TreeBoost expanding-window nowcasts
    # -------------------------
    df_fc = expanding_window_treeboost_nowcast(
        X,
        y,
        min_train_obs=MIN_TRAIN_OBS,
        model_params=MODEL_PARAMS,
        val_frac_inner=VAL_FRAC_INNER,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        ar1_func=ar1_forecast,
    )

    # -------------------------
    # Optionally add DFM and create evaluation sample
    # -------------------------
    if USE_DFM:
        df_raw = out0["Z"].join(out0["y_monthly"])
        df_dfm = expanding_window_dfm_nowcast(
            df_raw,
            gdp_col=GDP_COL,
            min_train_obs=40,
        )
        df_eval = df_fc.join(df_dfm[["y_pred_dfm"]], how="inner")
        print(
            f"Compare sample: n_obs={len(df_eval)} "
            f"(TreeBoost={len(df_fc)}, DFM={len(df_dfm)})"
        )
    else:
        df_eval = df_fc.copy()

    # =========================
    # SUMMARY METRICS
    # =========================
    y_true = df_eval["y_true"].to_numpy(dtype=float)

    yhat_tree = df_eval["y_pred"].to_numpy(dtype=float)
    yhat_ar1  = df_eval["y_pred_ar1"].to_numpy(dtype=float)
    yhat_zero = np.zeros_like(y_true, dtype=float)

    rmse_tree = rmse(y_true, yhat_tree)
    mae_tree  = mae(y_true, yhat_tree)

    rmse_zero = rmse(y_true, yhat_zero)
    rmse_ar1  = rmse(y_true, yhat_ar1)

    skill_tree_vs_zero = 1.0 - mse(y_true, yhat_tree) / mse(y_true, yhat_zero)
    skill_tree_vs_ar1  = 1.0 - mse(y_true, yhat_tree) / mse(y_true, yhat_ar1)

    if USE_DFM:
        yhat_dfm = df_eval["y_pred_dfm"].to_numpy(dtype=float)

        rmse_dfm = rmse(y_true, yhat_dfm)
        mae_dfm  = mae(y_true, yhat_dfm)

        skill_dfm_vs_zero = 1.0 - mse(y_true, yhat_dfm) / mse(y_true, yhat_zero)
        skill_dfm_vs_ar1  = 1.0 - mse(y_true, yhat_dfm) / mse(y_true, yhat_ar1)

    # Print
    print("\n=== EXPANDING-WINDOW (pseudo real-time) results ===")
    print(f"RMSE TreeBoost        : {rmse_tree:.6f} | MAE: {mae_tree:.6f}")

    if USE_DFM:
        print(f"RMSE DFM              : {rmse_dfm:.6f} | MAE: {mae_dfm:.6f}")

    print(f"RMSE zero benchmark   : {rmse_zero:.6f}")
    print(f"RMSE AR(1) benchmark  : {rmse_ar1:.6f}")

    print(f"Skill TreeBoost vs zero (MSE): {skill_tree_vs_zero:.6f}")
    print(f"Skill TreeBoost vs AR(1) (MSE): {skill_tree_vs_ar1:.6f}")

    if USE_DFM:
        print(f"Skill DFM      vs zero (MSE): {skill_dfm_vs_zero:.6f}")
        print(f"Skill DFM      vs AR(1) (MSE): {skill_dfm_vs_ar1:.6f}")


if __name__ == "__main__":
    main()
