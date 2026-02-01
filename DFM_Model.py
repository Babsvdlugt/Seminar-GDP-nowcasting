"""
Dynamic Factor Model (DFM) voor tijdreeksdata.
Schat latente factoren uit verklarende variabelen en plot de resultaten.
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from statsmodels.tsa.api import DynamicFactor
from statsmodels.tools.sm_exceptions import ConvergenceWarning

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=ConvergenceWarning)


def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    """
    Load CSV with a 'date' column as index and keep missing values (ragged edge).
    """
    data = pd.read_csv(file_path, index_col="date", parse_dates=True, dayfirst=True)

    # monthly start frequency
    data = data.asfreq("MS")

    # basic cleanup: ensure numeric
    for c in data.columns:
        data[c] = pd.to_numeric(data[c], errors="coerce")

    logger.info(f"Loaded data: {data.shape[0]} rows, {data.shape[1]} columns.")
    return data


def drop_sparse_columns(df: pd.DataFrame, min_non_missing_frac: float = 0.70) -> pd.DataFrame:
    """
    Drop columns with too many missing values.
    """
    keep = df.columns[df.notna().mean() >= min_non_missing_frac]
    dropped = set(df.columns) - set(keep)
    if dropped:
        logger.info(f"Dropping {len(dropped)} sparse columns (>{1-min_non_missing_frac:.0%} missing).")
    return df[keep].copy()


def standardize(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Z-score standardization column-wise, ignoring missings.
    """
    mu = df.mean(skipna=True)
    sd = df.std(skipna=True, ddof=0).replace(0.0, np.nan)
    z = (df - mu) / sd
    return z, mu, sd


def fit_dynamic_factor_model(X: pd.DataFrame, k_factors: int = 1, factor_order: int = 1):
    model = DynamicFactor(endog=X, k_factors=k_factors, factor_order=factor_order)

    # More robust fit settings
    res = model.fit(method="lbfgs", maxiter=2000, disp=False)
    logger.info(f"Converged: {res.mle_retvals.get('converged', None)} | llf={res.llf:.2f}")
    return res


def plot_factors(factors_df: pd.DataFrame) -> None:
    plt.figure(figsize=(12, 5))
    for col in factors_df.columns:
        plt.plot(factors_df.index, factors_df[col], label=col)
    plt.title("Smoothed latent factors (DFM)")
    plt.xlabel("Time")
    plt.ylabel("Factor")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


def plot_loadings(res, X_columns: list[str]) -> None:
    """
    Plot factor loadings. Statsmodels stores loadings in res.params with naming convention.
    """
    # loadings parameters typically like "loading.f1.<varname>" depending on version
    loadings = {}
    for name, val in res.params.items():
        if "loading" in name:
            loadings[name] = val

    if not loadings:
        logger.warning("No loadings found in params (statsmodels naming may differ).")
        return

    s = pd.Series(loadings).sort_values()
    plt.figure(figsize=(10, max(4, 0.25 * len(s))))
    plt.barh(s.index, s.values)
    plt.title("Estimated factor loadings (raw parameter names)")
    plt.tight_layout()
    plt.show()


def main():
    data_path = Path("Data prep/Data prep/top20_monthly_grid_from2005.csv")
    data = load_and_prepare_data(data_path)

    # If you *really* intend last column = y, keep it explicit:
    # y = data["gdp"]  # <-- better: name-based
    # X = data.drop(columns=["gdp"])

    X = data.iloc[:, :-1].copy()  # keep your assumption, but consider naming explicitly

    # Drop columns with too many missings
    X = drop_sparse_columns(X, min_non_missing_frac=0.70)

    # Standardize
    Xz, mu, sd = standardize(X)

    # Fit DFM (handles missing values via Kalman filter)
    res = fit_dynamic_factor_model(Xz, k_factors=1, factor_order=1)

    # Factors
    f = res.factors.smoothed
    factors_df = pd.DataFrame(f, index=Xz.index, columns=[f"Factor_{i+1}" for i in range(f.shape[1])])
    plot_factors(factors_df)

    # Optional: loadings plot
    plot_loadings(res, Xz.columns.tolist())


if __name__ == "__main__":
    main()
