import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import statsmodels.api as sm
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)

GDP_COL = "GrossDomesticProduct_1"
USE_QUARTER_FACTOR = True


# -------------------------
# DATA INLADEN
# -------------------------
def load_dfm_ready(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.asfreq("MS")
    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    logger.info(f"Loaded DFM-ready data: {df.shape}")
    return df


# -------------------------
# Cleaning helpers
# -------------------------
def drop_near_constant_cols(X: pd.DataFrame, eps: float = 1e-6) -> pd.DataFrame:
    sd = X.std(skipna=True, ddof=0)
    keep = sd[sd > eps].index
    dropped = list(set(X.columns) - set(keep))
    if dropped:
        logger.info(f"Dropping {len(dropped)} near-constant cols: {sorted(dropped)}")
    X2 = X[keep].copy()
    if X2.shape[1] == 0:
        raise ValueError("After dropping near-constant cols, no columns remain.")
    return X2


def drop_sparse_cols(X: pd.DataFrame, min_non_missing_frac: float = 0.30) -> pd.DataFrame:
    frac = X.notna().mean()
    keep = frac[frac >= min_non_missing_frac].index
    dropped = list(set(X.columns) - set(keep))
    if dropped:
        logger.info(f"Dropping {len(dropped)} sparse cols: {sorted(dropped)}")
    X2 = X[keep].copy()
    if X2.shape[1] == 0:
        raise ValueError("After dropping sparse cols, no columns remain.")
    return X2


def standardize_window(X: pd.DataFrame) -> pd.DataFrame:
    """Standardize using window-only mean/std to avoid future leakage."""
    mu = X.mean(skipna=True)
    sd = X.std(skipna=True, ddof=0)
    sd = sd.replace(0.0, np.nan)
    return (X - mu) / sd


# -------------------------
# DFM fit
# -------------------------
def fit_dfm(endog: pd.DataFrame, k_factors=1, factor_order=1, error_order=0):
    if endog.shape[0] < 50:
        raise ValueError(f"Too few observations for DFM: {endog.shape[0]}")
    if endog.shape[1] < 3:
        raise ValueError(f"Too few series for DFM: {endog.shape[1]}")

    mod = DynamicFactor(
        endog=endog,
        k_factors=k_factors,
        factor_order=factor_order,
        error_order=error_order
    )
    res = mod.fit(method="lbfgs", maxiter=1500, disp=False)
    logger.info(
        f"Converged: {res.mle_retvals.get('converged', None)} | "
        f"llf={res.llf:.2f} | AIC={res.aic:.2f} | BIC={res.bic:.2f}"
    )
    return res


def extract_factors(res, index, use_filtered: bool = True) -> pd.DataFrame:
    """
    For real-time nowcasting, filtered factors are preferred (no future info).
    Smoothed factors are ex-post.
    """
    f = np.asarray(res.factors.filtered if use_filtered else res.factors.smoothed)

    T = len(index)
    if f.shape[0] == T:
        pass
    elif f.shape[1] == T:
        f = f.T
    else:
        raise ValueError(f"Unexpected factor shape {f.shape}, expected one dim == T={T}")

    F = pd.DataFrame(f, index=index, columns=[f"Factor_{i+1}" for i in range(f.shape[1])])
    return F


# -------------------------
# Factor selection
# -------------------------
def select_k_factors(
    X: pd.DataFrame,
    k_max: int = 8,
    factor_order: int = 1,
    error_order: int = 0,
    criterion: str = "bic"
) -> pd.DataFrame:
    rows = []
    for k in range(1, k_max + 1):
        try:
            res = fit_dfm(X, k_factors=k, factor_order=factor_order, error_order=error_order)
            converged = bool(res.mle_retvals.get("converged", False))
            rows.append({"k_factors": k, "aic": res.aic, "bic": res.bic, "llf": res.llf, "converged": converged})
        except Exception as e:
            logger.warning(f"k={k} failed: {e}")
            rows.append({"k_factors": k, "aic": np.nan, "bic": np.nan, "llf": np.nan, "converged": False})

    df_ic = pd.DataFrame(rows)
    crit = criterion.lower().strip()
    if crit not in ["aic", "bic"]:
        raise ValueError("criterion must be 'aic' or 'bic'")
    return df_ic


def pick_best_k(ic_table: pd.DataFrame, criterion: str = "bic") -> int:
    crit = criterion.lower().strip()
    ic_ok = ic_table[ic_table["converged"]].dropna(subset=[crit])
    if ic_ok.empty:
        raise RuntimeError("No converged valid models for factor selection.")
    return int(ic_ok.sort_values(crit).iloc[0]["k_factors"])


# -------------------------
# Bridge regression
# -------------------------
def quarterly_average_factors(F: pd.DataFrame) -> pd.DataFrame:
    Fq = F.copy()
    Fq["__q__"] = Fq.index.to_period("Q")
    Fq = Fq.groupby("__q__").mean(numeric_only=True)
    Fq.index = Fq.index.to_timestamp(how="end")
    return Fq


def fit_bridge_regression(gdp: pd.Series, F: pd.DataFrame):
    gdp = pd.to_numeric(gdp, errors="coerce").dropna()
    if gdp.shape[0] < 10:
        raise ValueError(f"Too few GDP observations for bridge regression: {gdp.shape[0]}")

    if USE_QUARTER_FACTOR:
        F_use = quarterly_average_factors(F)

        gdp_q = gdp.copy()
        gdp_q.index = gdp_q.index.to_period("Q").to_timestamp(how="end")
        gdp_q = gdp_q.groupby(gdp_q.index).last()

        df_reg = pd.concat([gdp_q.rename("gdp"), F_use], axis=1).dropna()
        y = df_reg["gdp"]
        Xreg = sm.add_constant(df_reg.drop(columns=["gdp"]), has_constant="add")
    else:
        F_m = F.reindex(gdp.index)
        df_reg = pd.concat([gdp.rename("gdp"), F_m], axis=1).dropna()
        y = df_reg["gdp"]
        Xreg = sm.add_constant(df_reg.drop(columns=["gdp"]), has_constant="add")

    return sm.OLS(y, Xreg).fit()


# =========================
# EXPANDING WINDOW PIPELINE 
# =========================
def expanding_window_nowcast(
    df: pd.DataFrame,
    k_max: int = 8,
    factor_order: int = 1,
    error_order: int = 0,
    min_train_months: int = 80,
    criterion: str = "bic",
    use_filtered_factors: bool = True,
) -> pd.DataFrame:
    """
    Real-time expanding window evaluation.

    For each quarter month t with GDP observed:
      - use only data up to t
      - standardize within window
      - select k_t by BIC within window
      - fit DFM within window
      - extract FILTERED (real-time) factors
      - fit bridge regression with GDP up to t
      - nowcast GDP at t
    """

    # --- keep columns fixed across windows (important for comparability) ---
    X_full = df.drop(columns=[GDP_COL]).copy()
    X_full = drop_near_constant_cols(X_full, eps=1e-6)
    X_full = drop_sparse_cols(X_full, min_non_missing_frac=0.30)

    gdp_full = pd.to_numeric(df[GDP_COL], errors="coerce")
    target_months = gdp_full.dropna().index  # quarterly GDP months

    rows = []

    for t in target_months:
        # 1) expanding window data
        X_win_raw = X_full.loc[:t].copy()
        if len(X_win_raw) < min_train_months:
            continue

        # 2) window standardization (NO future leakage)
        X_win = standardize_window(X_win_raw)

        # 3) select k_t by criterion (BIC) on window
        ic_t = select_k_factors(
            X_win, k_max=k_max, factor_order=factor_order, error_order=error_order, criterion=criterion
        )
        try:
            k_t = pick_best_k(ic_t, criterion=criterion)
        except RuntimeError:
            continue

        # 4) fit final dfm on window
        res_t = fit_dfm(X_win, k_factors=k_t, factor_order=factor_order, error_order=error_order)

        # 5) extract factors (filtered recommended)
        F_t = extract_factors(res_t, X_win.index, use_filtered=use_filtered_factors)

        # 6) bridge regression using GDP up to t only
        gdp_win = gdp_full.loc[:t].dropna()
        try:
            ols_t = fit_bridge_regression(gdp_win, F_t)
        except Exception:
            continue

        # 7) nowcast GDP for quarter of t
        if USE_QUARTER_FACTOR:
            Fq_t = quarterly_average_factors(F_t)
            q_t = t.to_period("Q").to_timestamp(how="end")
            if q_t not in Fq_t.index:
                continue
            X_now = sm.add_constant(Fq_t.loc[[q_t]], has_constant="add")
            y_pred = float(ols_t.predict(X_now).iloc[0])
        else:
            X_now = sm.add_constant(F_t.loc[[t]], has_constant="add")
            y_pred = float(ols_t.predict(X_now).iloc[0])

        y_true = float(gdp_full.loc[t])

        rows.append(
            {
                "date": t,
                "y_true": y_true,
                "y_pred": y_pred,
                "error": y_true - y_pred,
                "k_selected": k_t,
                "bic_best": float(ic_t[ic_t["converged"]]["bic"].min())
                if not ic_t[ic_t["converged"]].dropna(subset=["bic"]).empty
                else np.nan,
                "n_months_train": int(len(X_win)),
            }
        )

    out = pd.DataFrame(rows).set_index("date")
    return out


def main():
    path = Path("data_transformations_DFM_ready_state_space.csv")
    df = load_dfm_ready(path)

    if GDP_COL not in df.columns:
        raise ValueError(f"GDP_COL '{GDP_COL}' not found in dataset.")

    # =========================
    # run expanding-window evaluation 
    # =========================
    bt = expanding_window_nowcast(
        df,
        k_max=8,
        factor_order=1,
        error_order=0,
        min_train_months=80,
        criterion="bic",
        use_filtered_factors=True,
    )

    if bt.empty:
        print("No backtest results. Try lowering min_train_months or k_max.")
        return

    # --- summary stats  ---
    avg_k = bt["k_selected"].mean()
    rmse_val = float(np.sqrt(np.mean(bt["error"] ** 2)))

    print("\n=== Expanding-window results ===")
    print(f"Backtest points: {len(bt)}")
    print(f"Average selected k (BIC): {avg_k:.2f}")
    print(f"RMSE: {rmse_val:.4f}")
    print("\nSelected k distribution:")
    print(bt["k_selected"].value_counts().sort_index())

    # --- plot errors ---
    plt.figure(figsize=(12, 4))
    plt.plot(bt.index, bt["error"])
    plt.title("Expanding-window nowcast errors (y_true - y_pred)")
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # --- plot y_true vs y_pred ---
    plt.figure(figsize=(12, 4))
    plt.plot(bt.index, bt["y_true"], label="Observed GDP")
    plt.plot(bt.index, bt["y_pred"], label="Nowcast GDP")
    plt.title("Observed vs nowcast GDP (expanding window)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
