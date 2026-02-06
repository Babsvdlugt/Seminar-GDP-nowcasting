import numpy as np
import pandas as pd

from ls_treeboost import step0_load_state_space, step1_build_supervised_set, LS_treeboost


# =========================
# CONFIG
# =========================
STATE_SPACE_PATH = "data_transformations_DFM_ready_state_space.csv"
GDP_COL = "GrossDomesticProduct_1"

ASOF_RULE = "early"   # choose: "early", "mid", "end"
MAX_LAG = 12

# Early stopping is done INSIDE each expanding-window fit using a small internal validation tail
VAL_FRAC_INNER = 0.2
EARLY_STOPPING_ROUNDS = 25

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

# How many first observations to use before starting to forecast (must be big enough)
MIN_TRAIN_OBS = 25

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
                "error": y_true - y_pred,
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
    oos_mae = mae(df_fc["y_true"], df_fc["y_pred"])
    oos_rmse_zero = rmse(df_fc["y_true"], yhat0)

    # Skill in MSE terms vs zero:
    skill_mse_vs_zero = 1.0 - (mse(df_fc["y_true"], df_fc["y_pred"]) / mse(df_fc["y_true"], yhat0))

    print("\n=== EXPANDING-WINDOW (pseudo real-time) results ===")
    print(f"OOS RMSE (model)      : {oos_rmse:.6f}")
    print(f"OOS MAE  (model)      : {oos_mae:.6f}")
    print(f"OOS RMSE (zero bench) : {oos_rmse_zero:.6f}")
    print(f"Skill (MSE vs zero)   : {skill_mse_vs_zero:.6f}")

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
            "avg_n_trees": float(df_fc["n_trees"].mean()),
        }]
    )
    summary_out = f"ls_treeboost_expanding_summary_{ASOF_RULE}_lag{MAX_LAG}.csv"
    summary.to_csv(summary_out, index=False)
    print(f"Saved: {summary_out}")


if __name__ == "__main__":
    main()
