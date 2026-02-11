import numpy as np
import pandas as pd

from ls_treeboost import (
    step0_load_state_space,
    step1_build_supervised_set,
    expanding_window_treeboost_nowcast,
)

from DFM_Model import expanding_window_nowcast  # your DFM expanding window function

from forecast_selection_unstable import build_switched_forecast


# =========================
# CONFIG
# =========================
USE_DFM = True  # True = run DFM + selection; False = only TreeBoost

print("DEBUG: USE_DFM is nu gezet op:", USE_DFM)

STATE_SPACE_PATH = "data_transformations_DFM_ready_state_space.csv"
GDP_COL = "GrossDomesticProduct_1"

ASOF_RULE = "end"   # choose: "early", "mid", "end"
MAX_LAG = 12

VAL_FRAC_INNER = 0.1
EARLY_STOPPING_ROUNDS = 50

MODEL_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 2,
    "min_samples_leaf": 10,
    "min_samples_split": 20,
    "max_features": "sqrt",
    "subsample": 0.6,
    "random_state": 42,
}

MIN_TRAIN_OBS = 25

RELEASE_LAGS = {
    "Construction_proxy": 2,
    "Domestic_consumption_by_households_VolumeChangesShoppingdayAdjusted_3": 2,
    "NL_Consumer_confidence": 0,
    "NL_Industrial_confidence": 0,
    "NL_Economic_sentiment_confidence": 0,
    "BusinessLeadIndicator_NLD": 0,
    "ConsumLeadIndicator_NLD": 0,
    "IndustProd_Europe_ImprtWeighted": 0,
    "Exports_Europe": 2,
    "Imports_Europe": 2,
    "CPI_1": 1,
    "MaandmutatieCPI_3": 1,
    "ecb_3M_Yield": 0,
    "ecb_10Y_Yield": 0,
    "^AEX": 0,
    "Employment_allGenders_15 to 74 years_SeasonallyAdjusted_8_UnemplyRate": 0,
    "Bankruptcies": 1,
    "GrossDomesticProduct_1": 3,
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

    # Step 1: build supervised set once (defines your quarterly target dates/index)
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
        print("Start DFM pipeline")

        df_raw = out0["Z"].join(out0["y_monthly"].rename(GDP_COL))

        df_dfm = expanding_window_nowcast(
            df_raw,
            min_train_months=40,
            k_max=6,
            criterion="bic",
            use_filtered_factors=True
        )

        print("DFM pipeline klaar → aantal rijen:", len(df_dfm))
        if not df_dfm.empty:
            print(df_dfm.head(3))

        print("TreeBoost y_pred head:\n", df_fc["y_pred"].head(5))
        print("DFM y_pred head:\n", df_dfm["y_pred"].head(5))

        df_eval = df_fc.join(
            df_dfm[["y_pred"]].rename(columns={"y_pred": "y_pred_dfm"}),
            how="inner"
        )

        print("\n=== DEBUG JOIN RESULTAAT ===")
        print("Aantal rijen df_eval:", len(df_eval))
        print("Kolommen in df_eval:", df_eval.columns.tolist())

        print("\nVergelijk y_pred TreeBoost vs DFM op eerste 5 rijen:")
        print(df_eval[["y_pred", "y_pred_dfm"]].head(5))

        print("\nCorrelatie tussen TreeBoost en DFM voorspellingen:")
        print(df_eval["y_pred"].corr(df_eval["y_pred_dfm"]))

        print(
            f"\nCompare sample: n_obs={len(df_eval)} "
            f"(TreeBoost={len(df_fc)}, DFM={len(df_dfm)})"
        )
    else:
        df_eval = df_fc.copy()

    # ============================================================
    # Richter & Smetanina (Forecast Selection in Unstable Environments) mean selection
    # Only meaningful if DFM exists
    # ============================================================
    if USE_DFM:
        df_eval, df_sel = build_switched_forecast(
            df_eval=df_eval,
            yhat_A_col="y_pred",        # Model A = TreeBoost
            yhat_B_col="y_pred_dfm",    # Model B = DFM
            y_true_col="y_true",
            loss="squared",
            d=1,
            h=0.20,
            min_train=40
        )

        mask_sel = df_eval["y_pred_selected"].notna()
        rmse_selected = rmse(df_eval.loc[mask_sel, "y_true"], df_eval.loc[mask_sel, "y_pred_selected"])
        print(f"\nRMSE Selected (Richter–Smetanina mean selection): {rmse_selected:.6f}")

        print("\nSelection counts (modelA=TreeBoost, modelB=DFM):")
        print(df_eval.loc[mask_sel, "choice"].value_counts(dropna=False))

    # ============================================================
    # CRISIS SAMPLE EVALUATION
    # ============================================================
    CRISIS_PERIODS = [
        # ("2008-10-01", "2009-07-01"),  # GFC
        ("2011-07-01", "2013-04-01"),  # Eurozone debt crisis
        ("2018-10-01", "2019-10-01"),  # Trade war/ global slowdown
        ("2020-01-01", "2020-07-01"),  # COVID-19 crisis
    ]

    df_crisis = pd.concat([df_eval.loc[start:end] for start, end in CRISIS_PERIODS])
    print(f"\nCrisis sample: n_obs={len(df_crisis)}")

    y_c = df_crisis["y_true"].to_numpy(dtype=float)
    yhat_tree_c = df_crisis["y_pred"].to_numpy(dtype=float)
    yhat_ar1_c = df_crisis["y_pred_ar1"].to_numpy(dtype=float)

    rmse_tree_c = rmse(y_c, yhat_tree_c)
    rmse_ar1_c = rmse(y_c, yhat_ar1_c)
    rmse_zero_c = rmse(y_c, np.zeros_like(y_c))

    skill_tree_vs_ar1_c = 1.0 - mse(y_c, yhat_tree_c) / mse(y_c, yhat_ar1_c)
    skill_tree_vs_zero_c = 1.0 - mse(y_c, yhat_tree_c) / mse(y_c, np.zeros_like(y_c))

    print("\n=== CRISIS PERFORMANCE ===")
    print(f"RMSE TreeBoost (crisis): {rmse_tree_c:.6f}")
    print(f"RMSE AR(1)    (crisis): {rmse_ar1_c:.6f}")
    print(f"RMSE zero     (crisis): {rmse_zero_c:.6f}")
    print(f"Skill TreeBoost vs AR(1) (crisis, MSE): {skill_tree_vs_ar1_c:.6f}")
    print(f"Skill TreeBoost vs zero  (crisis, MSE): {skill_tree_vs_zero_c:.6f}")

    # Per-quarter loss differences: AR1 loss minus TreeBoost loss
    df_crisis = df_crisis.copy()
    df_crisis["loss_diff"] = (
        (df_crisis["y_true"] - df_crisis["y_pred_ar1"]) ** 2
        - (df_crisis["y_true"] - df_crisis["y_pred"]) ** 2
    )
    print("\nPer-crisis-quarter loss differences (AR1 - TreeBoost):")
    print(df_crisis[["loss_diff"]])

    # ============================================================
    # SUMMARY METRICS (FULL SAMPLE)
    # ============================================================
    y_true = df_eval["y_true"].to_numpy(dtype=float)
    yhat_tree = df_eval["y_pred"].to_numpy(dtype=float)
    yhat_ar1 = df_eval["y_pred_ar1"].to_numpy(dtype=float)
    yhat_zero = np.zeros_like(y_true, dtype=float)

    rmse_tree = rmse(y_true, yhat_tree)
    mae_tree = mae(y_true, yhat_tree)

    rmse_zero = rmse(y_true, yhat_zero)
    rmse_ar1 = rmse(y_true, yhat_ar1)

    skill_tree_vs_zero = 1.0 - mse(y_true, yhat_tree) / mse(y_true, yhat_zero)
    skill_tree_vs_ar1 = 1.0 - mse(y_true, yhat_tree) / mse(y_true, yhat_ar1)

    print("\n=== EXPANDING-WINDOW (pseudo real-time) results ===")
    print(f"RMSE TreeBoost        : {rmse_tree:.6f} | MAE: {mae_tree:.6f}")
    if USE_DFM:
        yhat_dfm = df_eval["y_pred_dfm"].to_numpy(dtype=float)
        rmse_dfm = rmse(y_true, yhat_dfm)
        mae_dfm = mae(y_true, yhat_dfm)
        print(f"RMSE DFM              : {rmse_dfm:.6f} | MAE: {mae_dfm:.6f}")

    print(f"RMSE zero benchmark   : {rmse_zero:.6f}")
    print(f"RMSE AR(1) benchmark  : {rmse_ar1:.6f}")

    print(f"Skill TreeBoost vs zero (MSE): {skill_tree_vs_zero:.6f}")
    print(f"Skill TreeBoost vs AR(1) (MSE): {skill_tree_vs_ar1:.6f}")

    if USE_DFM:
        skill_dfm_vs_zero = 1.0 - mse(y_true, yhat_dfm) / mse(y_true, yhat_zero)
        skill_dfm_vs_ar1 = 1.0 - mse(y_true, yhat_dfm) / mse(y_true, yhat_ar1)
        print(f"Skill DFM      vs zero (MSE): {skill_dfm_vs_zero:.6f}")
        print(f"Skill DFM      vs AR(1) (MSE): {skill_dfm_vs_ar1:.6f}")


if __name__ == "__main__":
    main()
