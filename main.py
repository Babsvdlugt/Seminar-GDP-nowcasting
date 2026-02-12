# main.py
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ls_treeboost import (
    step0_load_state_space,
    step1_build_supervised_set,
    expanding_window_treeboost_nowcast,
)

from forecast_selection_unstable import build_switched_forecast
from dfm_cpb import run_cpb_mixture


# ============================================================
# METRICS
# ============================================================
def fmse(y_true, y_pred) -> float:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() == 0:
        return np.nan
    e = y_true[m] - y_pred[m]
    return float(np.mean(e * e))


def frmse(y_true, y_pred) -> float:
    m = fmse(y_true, y_pred)
    return float(np.sqrt(m)) if np.isfinite(m) else np.nan


def expanding_rmse_series(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    """
    Expanding-window RMSE over time:
      RMSE_t = sqrt( mean_{s<=t} (y_s - yhat_s)^2 )
    """
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if df.empty:
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([], name="date"))
    e2 = (df["y_true"] - df["y_pred"]) ** 2
    rmse_t = np.sqrt(e2.expanding().mean())
    rmse_t.name = "frmse_expanding"
    return rmse_t


def squared_loss_series(y_true: pd.Series, y_pred: pd.Series) -> pd.Series:
    df = pd.concat([y_true.rename("y_true"), y_pred.rename("y_pred")], axis=1).dropna()
    if df.empty:
        return pd.Series([], dtype=float, index=pd.DatetimeIndex([], name="date"))
    loss = (df["y_true"] - df["y_pred"]) ** 2
    loss.name = "loss_sq"
    return loss


# ============================================================
# AR(1) expanding-window forecasts (standalone benchmark)
# ============================================================
def expanding_window_ar1_forecast(y: pd.Series, min_train: int = 10) -> pd.Series:
    """
    Expanding AR(1):
      y_t = alpha + phi y_{t-1} + eps
    For each t >= min_train: fit on y[:t-1], forecast y_t from y_{t-1}.
    """
    y = y.dropna().copy()
    dates = y.index
    yhat = pd.Series(index=dates, dtype=float, name="y_pred_ar1")

    for i in range(len(dates)):
        if i < max(2, min_train):
            continue
        y_train = y.iloc[:i]  # up to t-1
        if len(y_train) < 3:
            continue

        y_lag = y_train.values[:-1]
        y_now = y_train.values[1:]
        X = np.column_stack([np.ones(len(y_lag)), y_lag])
        beta = np.linalg.lstsq(X, y_now, rcond=None)[0]
        alpha_hat, phi_hat = beta

        y_last = float(y_train.values[-1])
        yhat.iloc[i] = float(alpha_hat + phi_hat * y_last)

    return yhat


# Needed by ls_treeboost.expanding_window_treeboost_nowcast (must be callable)
def ar1_forecast(y_train: pd.Series, y_last: float) -> float:
    """
    Fit AR(1): y_t = alpha + phi y_{t-1} + eps
    Return 1-step forecast given last observed y_{t-1}=y_last.
    """
    y = np.asarray(y_train, dtype=float)
    if y.size < 3 or not np.isfinite(y_last):
        return float(y_last)

    y_lag = y[:-1]
    y_now = y[1:]
    X = np.column_stack([np.ones(len(y_lag)), y_lag])
    beta = np.linalg.lstsq(X, y_now, rcond=None)[0]
    alpha_hat, phi_hat = float(beta[0]), float(beta[1])
    return float(alpha_hat + phi_hat * float(y_last))


# ============================================================
# CPB mixture DFM helpers
# ============================================================
def build_monthly_panel_for_cpb(out0: dict, gdp_col: str) -> pd.DataFrame:
    """
    Create monthly panel for CPB runner: indicators Z + GDP monthly series,
    with an explicit 'date' column (dfm_cpb.py expects DATE_COL='date').
    """
    df_raw = out0["Z"].copy()
    y_monthly = out0["y_monthly"].rename(gdp_col)
    df_raw = df_raw.join(y_monthly, how="left").sort_index()

    # dfm_cpb expects a column named 'date'
    df_raw = df_raw.reset_index().rename(columns={"index": "date"})
    return df_raw


def align_cpb_nowcasts_to_target_dates(
    cpb_out: pd.DataFrame,
    target_dates: pd.DatetimeIndex,
    *,
    use_column_mix: str = "mix_nowcast",
    q_col: str = "q_now",
    asof_col: str = "as_of",
) -> pd.Series:
    """
    Robust alignment:
    For each target quarter date t:
      - take cpb rows with q_now == t
      - pick the last available as_of (max as_of)
    """
    cpb = cpb_out.copy()
    cpb[asof_col] = pd.to_datetime(cpb[asof_col])
    cpb[q_col] = pd.to_datetime(cpb[q_col])

    preds = pd.Series(index=target_dates, dtype=float, name="y_pred_dfm_cpb")

    for t in target_dates:
        cand = cpb.loc[cpb[q_col] == pd.Timestamp(t), [asof_col, use_column_mix]].dropna()
        if cand.empty:
            continue
        cand = cand.sort_values(asof_col)
        preds.loc[t] = float(cand[use_column_mix].iloc[-1])

    return preds


# ============================================================
# PLOTS
# ============================================================
def plot_loss_two_models(
    y_true: pd.Series,
    yhat_a: pd.Series,
    yhat_b: pd.Series,
    label_a: str,
    label_b: str,
    title: str,
    outpath: str | None = None,
) -> None:
    loss_a = squared_loss_series(y_true, yhat_a)
    loss_b = squared_loss_series(y_true, yhat_b)

    plt.figure()
    if not loss_a.empty:
        plt.plot(loss_a.index, loss_a.values, label=f"{label_a} loss")
    if not loss_b.empty:
        plt.plot(loss_b.index, loss_b.values, label=f"{label_b} loss")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Squared loss")
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.show()


def plot_frmse_comparison(
    y_true: pd.Series,
    preds: dict[str, pd.Series],
    title: str,
    outpath: str | None = None,
) -> None:
    plt.figure()
    for name, yhat in preds.items():
        s = expanding_rmse_series(y_true, yhat)
        if not s.empty:
            plt.plot(s.index, s.values, label=name)
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Expanding FRMSE")
    plt.legend()
    plt.tight_layout()
    if outpath:
        plt.savefig(outpath, dpi=150)
    plt.show()


# ============================================================
# CONFIG
# ============================================================
STATE_SPACE_PATH = "data_transformations_DFM_ready_state_space.csv"
GDP_COL = "GrossDomesticProduct_1"

ASOF_RULE = "end"
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


# ============================================================
# MAIN
# ============================================================
def main():
    # -------------------------
    # Load + supervised set
    # -------------------------
    out0 = step0_load_state_space(
        path_ss=STATE_SPACE_PATH,
        date_col="date",
        gdp_col=GDP_COL,
    )

    X, y = step1_build_supervised_set(
        Z=out0["Z"],
        y_monthly=out0["y_monthly"],
        target_months=out0["target_months"],
        asof_rule=ASOF_RULE,
        max_lag=MAX_LAG,
        release_lags=RELEASE_LAGS,
        default_release_lag=DEFAULT_RELEASE_LAG,
    )

    print(f"[DATA] n_obs={len(y)}, n_features={X.shape[1]}")
    print(f"[SETUP] ASOF_RULE={ASOF_RULE}, MAX_LAG={MAX_LAG}, MIN_TRAIN_OBS={MIN_TRAIN_OBS}")

    # Quarterly target series
    y_series = y.copy()
    if not isinstance(y_series.index, pd.DatetimeIndex):
        y_series.index = pd.to_datetime(y_series.index)
    y_series = y_series.sort_index()

    # -------------------------
    # 1) AR(1) benchmark (standalone)
    # -------------------------
    yhat_ar1 = expanding_window_ar1_forecast(y_series, min_train=MIN_TRAIN_OBS)
    print("\n=== AR(1) RESULTS (quarterly target dates) ===")
    print(f"FRMSE AR(1): {frmse(y_series, yhat_ar1):.6f}")
    print(f"FMSE  AR(1): {fmse(y_series, yhat_ar1):.6f}")

    # -------------------------
    # 2) LS TreeBoost
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

    # These are the quarterly forecast dates where TreeBoost produced predictions
    target_dates = df_fc.index

    # Ensure y_true exists
    if "y_true" not in df_fc.columns:
        df_fc["y_true"] = y_series.reindex(df_fc.index)

    yhat_tree = df_fc["y_pred"].astype(float)

    print("\n=== LS TreeBoost RESULTS ===")
    print(f"FRMSE TreeBoost: {frmse(df_fc['y_true'], yhat_tree):.6f}")
    print(f"FMSE  TreeBoost: {fmse(df_fc['y_true'], yhat_tree):.6f}")

    # -------------------------
    # 3) DFM CPB mixture
    # -------------------------
    print("\n[DFM CPB] Running CPB mixture DFM (r=2..5, p=1..3) on monthly panel...")
    df_monthly_panel = build_monthly_panel_for_cpb(out0, gdp_col=GDP_COL)

    cpb_out = run_cpb_mixture(df_monthly_panel)

    yhat_dfm_cpb = align_cpb_nowcasts_to_target_dates(
        cpb_out,
        target_dates=target_dates,
        use_column_mix="mix_nowcast",
        q_col="q_now",
        asof_col="as_of",
    )

    print("[DFM ALIGN] non-missing:", int(yhat_dfm_cpb.notna().sum()), "/", len(yhat_dfm_cpb))
    print("[DFM ALIGN] first non-missing date:",
          yhat_dfm_cpb.dropna().index.min() if yhat_dfm_cpb.notna().any() else None)

    print("\n=== DFM CPB (mixture) RESULTS (aligned to target dates) ===")
    print(f"FRMSE DFM-CPB: {frmse(df_fc['y_true'], yhat_dfm_cpb):.6f}")
    print(f"FMSE  DFM-CPB: {fmse(df_fc['y_true'], yhat_dfm_cpb):.6f}")

    # -------------------------
    # Build evaluation table
    # -------------------------
    df_eval = pd.DataFrame(index=target_dates)
    df_eval["y_true"] = df_fc["y_true"].astype(float)
    df_eval["y_pred_tree"] = yhat_tree.astype(float)
    df_eval["y_pred_dfm"] = yhat_dfm_cpb.astype(float)

    # Use AR(1) benchmark produced inside TreeBoost loop (most consistent)
    df_eval["y_pred_ar1"] = df_fc["y_pred_ar1"].astype(float)

    # -------------------------
    # 4) Forecast selection (Richter–Smetanina)
    # -------------------------
    print("\n[SELECTION] Running Richter–Smetanina mean selection on TreeBoost vs DFM-CPB...")
    df_eval2 = df_eval.copy()

    overlap = df_eval2[["y_true", "y_pred_tree", "y_pred_dfm"]].dropna()
    n_avail = len(overlap)
    min_train_sel = min(40, max(5, n_avail // 2))
    print(f"[SELECTION] overlap n={n_avail} -> min_train={min_train_sel}")

    df_eval2, df_sel = build_switched_forecast(
        df_eval=df_eval2,
        yhat_A_col="y_pred_tree",   # model A = TreeBoost
        yhat_B_col="y_pred_dfm",    # model B = DFM CPB
        y_true_col="y_true",
        loss="squared",
        d=1,
        h=0.20,
        min_train=min_train_sel,
    )

    mask = df_eval2["y_pred_selected"].notna()
    df_sel_eval = df_eval2.loc[mask, ["y_true", "y_pred_selected", "choice"]].copy()

    print("\n=== SELECTION COUNTS ===")
    print(df_sel_eval["choice"].value_counts(dropna=False))

    print("\n=== SELECTION RESULTS (on selection-valid dates) ===")
    print(f"FRMSE Selected: {frmse(df_sel_eval['y_true'], df_sel_eval['y_pred_selected']):.6f}")
    print(f"FMSE  Selected: {fmse(df_sel_eval['y_true'], df_sel_eval['y_pred_selected']):.6f}")

    # -------------------------
    # 5) PLOTS
    # -------------------------
    plot_loss_two_models(
        y_true=df_eval["y_true"],
        yhat_a=df_eval["y_pred_tree"],
        yhat_b=df_eval["y_pred_dfm"],
        label_a="LS TreeBoost",
        label_b="DFM CPB",
        title="Squared loss over time: TreeBoost vs DFM CPB",
        outpath="loss_treeboost_vs_dfmcpb.png",
    )

    preds_for_plot = {
        "AR(1)": df_eval["y_pred_ar1"],
        "LS TreeBoost": df_eval["y_pred_tree"],
        "DFM CPB": df_eval["y_pred_dfm"],
        "Selection": df_eval2["y_pred_selected"],
    }
    plot_frmse_comparison(
        y_true=df_eval["y_true"],
        preds=preds_for_plot,
        title="Expanding FRMSE: AR(1) vs LS TreeBoost vs DFM CPB vs Selection",
        outpath="frmse_comparison.png",
    )

    # -------------------------
    # Save evaluation table
    # -------------------------
    out_eval_path = Path("eval_all_models.csv")
    df_out = df_eval.copy()
    df_out["y_pred_selected"] = df_eval2["y_pred_selected"]
    df_out["choice"] = df_eval2.get("choice", np.nan)
    df_out.to_csv(out_eval_path)
    print(f"\nSaved evaluation table -> {out_eval_path.resolve()}")
    print("Saved plots -> loss_treeboost_vs_dfmcpb.png, frmse_comparison.png")


if __name__ == "__main__":
    main()
