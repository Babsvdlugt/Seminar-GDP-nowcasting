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

# Zet dit op True als je later factor->kwartaal aggregatie wil gebruiken
USE_QUARTER_FACTOR = False


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


def drop_sparse_cols(X: pd.DataFrame, min_non_missing_frac: float = 0.70) -> pd.DataFrame:
    frac = X.notna().mean()
    keep = frac[frac >= min_non_missing_frac].index
    dropped = list(set(X.columns) - set(keep))
    if dropped:
        logger.info(f"Dropping {len(dropped)} sparse cols: {sorted(dropped)}")
    X2 = X[keep].copy()
    if X2.shape[1] == 0:
        raise ValueError("After dropping sparse cols, no columns remain.")
    return X2


# -------------------------
# DFM fit
# -------------------------
def fit_dfm(endog: pd.DataFrame, k_factors=1, factor_order=1, error_order=1):
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


def extract_smoothed_factors(res, index) -> pd.DataFrame:
    f = np.asarray(res.factors.smoothed)
    # vaak (k_factors, T) -> transpose naar (T, k_factors)
    if f.shape[0] < f.shape[1]:
        f = f.T
    F = pd.DataFrame(f, index=index, columns=[f"Factor_{i+1}" for i in range(f.shape[1])])
    return F


def plot_factors(F: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    for col in F.columns:
        plt.plot(F.index, F[col], label=col)
    plt.title("Smoothed latent factors (DFM)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()



# -------------------------
# Factor selection: Comparing AIC and BIC
# -------------------------
def select_k_factors(
    X: pd.DataFrame,
    k_max: int = 8,
    factor_order: int = 1,
    error_order: int = 1,
    criterion: str = "bic"
) -> pd.DataFrame:
    """
    Fit DFM for k=1..k_max factors and return AIC/BIC results.
    """
    rows = []

    # Loop over candidate numbers of factors
    for k in range(1, k_max + 1):
        try:
            # Fit the DFM with k latent factors (state-space + Kalman + MLE)
            res = fit_dfm(
                endog=X,
                k_factors=k,
                factor_order=factor_order,
                error_order=error_order
            )

            # Convergence flag from statsmodels optimizer
            converged = res.mle_retvals.get("converged", False)
            
            # Store model comparison stats
            rows.append({
                "k_factors": k,
                "aic": res.aic,
                "bic": res.bic,
                "llf": res.llf,
                "converged": converged
            })

        except Exception as e:
            # If the model fails (non-invertible, convergence problems, etc.),
            # store NaNs so we can still inspect what failed.
            logger.warning(f"k={k} failed: {e}")
            rows.append({
                "k_factors": k,
                "aic": np.nan,
                "bic": np.nan,
                "llf": np.nan,
                "converged": False
            })

    df_ic = pd.DataFrame(rows)

    # OPTIONAL: log "best" k, but be careful: use only valid (non-NaN) rows
    if criterion.lower() == "bic":
        best_k = df_ic.loc[df_ic["bic"].idxmin(), "k_factors"]
    elif criterion.lower() == "aic":
        best_k = df_ic.loc[df_ic["aic"].idxmin(), "k_factors"]
    else:
        raise ValueError("criterion must be 'aic' or 'bic'")

    logger.info(f"Selected k_factors={best_k} by {criterion.upper()}")

    return df_ic


# -------------------------
# Bridge regression: GDP on factors
# -------------------------
def quarterly_average_factors(F: pd.DataFrame) -> pd.DataFrame:
    """Gemiddelde factor per kwartaal (over 3 maanden)."""
    Fq = F.copy()
    Fq["__q__"] = Fq.index.to_period("Q")
    Fq = Fq.groupby("__q__").mean(numeric_only=True)
    Fq.index = Fq.index.to_timestamp(how="end")  # kwartaal-einde timestamp
    return Fq


def fit_bridge_regression(gdp: pd.Series, F: pd.DataFrame):
    """
    OLS: GDP op factoren. We matchen op index van GDP.
    - Als USE_QUARTER_FACTOR=False: gebruik factoren op GDP-datums (jouw huidige aanpak).
    - Als True: gebruik kwartaalgemiddelde factoren en match op kwartaal.
    """
    gdp = pd.to_numeric(gdp, errors="coerce").dropna()

    if gdp.shape[0] < 30:
        raise ValueError(f"Too few GDP observations for bridge regression: {gdp.shape[0]}")

    if USE_QUARTER_FACTOR:
        F_use = quarterly_average_factors(F)
        # match GDP naar kwartaal-einde index (zelfde timestamp convention)
        gdp_q = gdp.copy()
        gdp_q.index = gdp_q.index.to_period("Q").to_timestamp(how="end")
        gdp_q = gdp_q.groupby(gdp_q.index).last()
        df_reg = pd.concat([gdp_q.rename("gdp"), F_use], axis=1).dropna()
        y = df_reg["gdp"]
        Xreg = sm.add_constant(df_reg.drop(columns=["gdp"]), has_constant="add")
    else:
        # match factor rows op GDP-datums
        F_q = F.loc[gdp.index]
        df_reg = pd.concat([gdp.rename("gdp"), F_q], axis=1).dropna()
        y = df_reg["gdp"]
        Xreg = sm.add_constant(df_reg.drop(columns=["gdp"]), has_constant="add")

    ols = sm.OLS(y, Xreg).fit()
    return ols


def main():
    path = Path("data_transformations_DFM_ready_state_space.csv")
    df = load_dfm_ready(path)

    if GDP_COL not in df.columns:
        raise ValueError(f"GDP_COL '{GDP_COL}' not found in dataset.")

    # 1) DFM op indicatoren (zonder GDP) voor stabiliteit
    X = df.drop(columns=[GDP_COL]).copy()

    # extra safety 
    X = drop_near_constant_cols(X, eps=1e-6)  # data kleiner dan 1e-6 wordt weggelaten waarom dat getal?
    X = drop_sparse_cols(X, min_non_missing_frac=0.70)

    # 2) Select k_factors via informatiecriteria
    ic_table = select_k_factors(
        X,
        k_max=8,
        factor_order=1,
        error_order=1,
        criterion="bic"
    )
    print("\nFactor selection table:")
    print(ic_table)

    # kies beste k op basis van criterion, maar negeer niet-converged fits
    ic_ok = ic_table[ic_table["converged"]].copy()
    if ic_ok.empty:
        raise RuntimeError("No converged DFM fits during factor selection. Try smaller k_max or simpler orders.")

    best_k = int(ic_ok.sort_values("bic").iloc[0]["k_factors"])
    logger.info(f"Using k_factors={best_k} for final model")

    # 3) Fit final DFM with selected k
    res = fit_dfm(X, k_factors=best_k, factor_order=1, error_order=1)
    print(res.summary())

    # 3) Factors
    F = extract_smoothed_factors(res, X.index)
    plot_factors(F)

    # 4) GDP bridge
    gdp = df[GDP_COL].dropna()
    ols = fit_bridge_regression(gdp, F)
    print("\nBridge regression: GDP on factors")
    print(ols.summary())

    # 5) Nowcast op laatste maand (op basis van laatste factor-stand)
    latest = sm.add_constant(F.iloc[[-1]], has_constant="add")
    gdp_nowcast = float(ols.predict(latest).iloc[0])  # <- warning fix
    print(f"\nGDP nowcast (based on latest month factors): {gdp_nowcast:.4f}")

    # 6) Plot: observed GDP vs fitted (alleen op GDP-observaties)
    fitted_q = ols.fittedvalues
    plt.figure(figsize=(12, 4))
    plt.scatter(ols.model.endog.index, ols.model.endog, s=18, label="Observed GDP (used in bridge)", alpha=0.7)
    plt.plot(fitted_q.index, fitted_q, label="Fitted GDP (bridge)", linewidth=2)
    plt.title("GDP (observed) vs fitted from factors (bridge regression)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
