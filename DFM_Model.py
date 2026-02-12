# DFM_Model.py
from __future__ import annotations

import warnings
from typing import Optional, Literal, Dict, Any, List, Tuple
import time

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor


Criterion = Literal["bic", "aic"]


def _ensure_monthly_ms_index(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure DatetimeIndex and enforce monthly-start frequency ("MS") to avoid
    statsmodels 'no frequency information' warnings.
    """
    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out = out.sort_index()

    # Enforce MS freq (monthly start). If your data are month-end, change to "ME".
    try:
        out = out.asfreq("MS")
    except Exception:
        # If asfreq fails, still keep sorted DatetimeIndex
        pass
    return out


def _drop_bad_columns(train: pd.DataFrame) -> pd.DataFrame:
    """
    Drop columns that cause PCA / factor init issues:
    - all-NA columns
    - (almost) constant columns (std==0 ignoring NA)
    """
    x = train.dropna(axis=1, how="all")

    if x.shape[1] == 0:
        return x

    # drop zero-variance columns (ignoring NA)
    std = x.std(axis=0, skipna=True)
    keep = std > 0
    x = x.loc[:, keep]
    return x


def _effective_TN(train: pd.DataFrame) -> Tuple[int, int]:
    """
    Effective sample size for initialization constraints:
    - T_eff: number of rows that contain at least one non-NA
    - N_eff: number of columns (after dropping all-NA / zero-var already)
    """
    if train.empty:
        return 0, 0
    T_eff = int((~train.isna().all(axis=1)).sum())
    N_eff = int(train.shape[1])
    return T_eff, N_eff


def expanding_window_nowcast(
    df_raw: pd.DataFrame,
    *,
    gdp_col: str = "GrossDomesticProduct_1",
    gdp_series: Optional[str] = None,
    min_train_months: int = 40,
    k_max: int = 6,
    factor_order: int = 1,
    criterion: Criterion = "bic",
    use_filtered_factors: bool = True,
    fit_maxiter: int = 200,
    fit_method: str = "em",
    fit_disp: bool = False,
    verbose: bool = False,
    progress_every: int = 1,
) -> pd.DataFrame:
    """
    Expanding-window nowcasting with a Dynamic Factor Model (statsmodels).

    Input
    -----
    df_raw : monthly dataframe with predictors + GDP column (GDP observed quarterly).
            Index must be dates (monthly).
    gdp_col : GDP column name in df_raw
    min_train_months : minimum months in the training window before producing forecasts
    k_max : maximum number of factors to try (will be capped dynamically each step)
    factor_order : factor VAR order in DynamicFactor
    criterion : "bic" or "aic" for selecting k_factors
    use_filtered_factors : kept for API compatibility; forecasting uses get_prediction/get_forecast
    fit_* : optimizer settings

    Output
    ------
    DataFrame indexed by GDP observation dates (quarterly timestamps from df_raw[gdp_col].dropna().index)
    with columns:
      - y_true
      - y_pred
      - k_best
      - crit_best (bic/aic value)
      - n_months_train
    """
    # Backward compatibility: allow gdp_series as alias for gdp_col
    if gdp_series is not None:
        if gdp_col != "GrossDomesticProduct_1" and gdp_col != gdp_series:
            raise ValueError("Provide only one of gdp_col or gdp_series (or keep them identical).")
        gdp_col = str(gdp_series)

    df = _ensure_monthly_ms_index(df_raw)

    if gdp_col not in df.columns:
        raise ValueError(f"gdp_col='{gdp_col}' not found in df_raw columns.")

    # Target dates = the timestamps where GDP is observed (typically quarterly months)
    target_dates = df[gdp_col].dropna().index
    if len(target_dates) == 0:
        raise ValueError("No non-missing GDP observations found to define target dates.")

    rows: List[Dict[str, Any]] = []

    # Reduce noisy warnings from PCA init when sample is tiny
    warnings.filterwarnings("once", category=UserWarning)
    warnings.filterwarnings("once", category=RuntimeWarning)

    total_targets = len(target_dates)
    t0_all = time.perf_counter()

    for i, t in enumerate(target_dates, start=1):
        t_iter_start = time.perf_counter()
        # We want a real nowcast: use info up to the month BEFORE t
        # (Otherwise you may inadvertently include contemporaneous GDP.)
        as_of = (pd.Timestamp(t) - pd.offsets.MonthBegin(1))  # go to previous month-start
        train = df.loc[:as_of].copy()

        # Require minimum training length (in calendar months)
        if train.shape[0] < int(min_train_months):
            if verbose and progress_every and (i % progress_every == 0):
                elapsed = time.perf_counter() - t_iter_start
                print(
                    f"[DFM] {i}/{total_targets} {t.date()} -> skip (train_months={train.shape[0]}), {elapsed:.2f}s"
                )
            continue

        train = _drop_bad_columns(train)

        # After dropping, we need GDP still present
        if gdp_col not in train.columns:
            # GDP might be dropped if it was all-NA in train; keep it explicitly
            train[gdp_col] = df.loc[:as_of, gdp_col]

        # If still completely unusable, skip
        if train.shape[1] == 0:
            if verbose and progress_every and (i % progress_every == 0):
                elapsed = time.perf_counter() - t_iter_start
                print(f"[DFM] {i}/{total_targets} {t.date()} -> skip (no columns), {elapsed:.2f}s")
            continue

        T_eff, N_eff = _effective_TN(train)
        if T_eff < 5 or N_eff < 2:
            if verbose and progress_every and (i % progress_every == 0):
                elapsed = time.perf_counter() - t_iter_start
                print(
                    f"[DFM] {i}/{total_targets} {t.date()} -> skip (T_eff={T_eff}, N_eff={N_eff}), {elapsed:.2f}s"
                )
            # too little info to fit a factor model sensibly
            continue

        # Dynamic cap on number of factors
        # Need k <= min(T_eff, N_eff) (safe)
        k_cap = min(int(k_max), int(T_eff), int(N_eff))
        if k_cap < 1:
            if verbose and progress_every and (i % progress_every == 0):
                elapsed = time.perf_counter() - t_iter_start
                print(f"[DFM] {i}/{total_targets} {t.date()} -> skip (k_cap<1), {elapsed:.2f}s")
            continue

        best_val = np.inf
        best_res = None
        best_k = None
        best_status = None

        # Try k=1..k_cap
        for k in range(1, k_cap + 1):
            try:
                mod = DynamicFactor(
                    endog=train,
                    k_factors=int(k),
                    factor_order=int(factor_order),
                    error_cov_type="diagonal",
                )
                res = mod.fit(maxiter=int(fit_maxiter), method=str(fit_method), disp=bool(fit_disp))

                aic = float(getattr(res, "aic", np.nan))
                bic = float(getattr(res, "bic", np.nan))
                val = bic if criterion == "bic" else aic

                if np.isfinite(val) and val < best_val:
                    best_val = val
                    best_res = res
                    best_k = k
                    best_status = {"aic": aic, "bic": bic, "llf": float(getattr(res, "llf", np.nan))}

            except Exception:
                # silently skip failed specs
                continue

        if best_res is None or best_k is None:
            if verbose and progress_every and (i % progress_every == 0):
                elapsed = time.perf_counter() - t_iter_start
                print(f"[DFM] {i}/{total_targets} {t.date()} -> skip (no fit), {elapsed:.2f}s")
            continue

        # Forecast GDP at target date t
        target_ts = pd.Timestamp(t)
        last_train_ts = train.index.max()

        try:
            if target_ts <= last_train_ts:
                pred = best_res.get_prediction(start=target_ts, end=target_ts)
                yhat = float(pred.predicted_mean[gdp_col].iloc[0])
            else:
                # steps in months between last_train_ts and target_ts
                steps = (target_ts.year - last_train_ts.year) * 12 + (target_ts.month - last_train_ts.month)
                steps = max(1, int(steps))
                fc = best_res.get_forecast(steps=steps)
                yhat = float(fc.predicted_mean[gdp_col].iloc[-1])
        except Exception:
            if verbose and progress_every and (i % progress_every == 0):
                elapsed = time.perf_counter() - t_iter_start
                print(f"[DFM] {i}/{total_targets} {t.date()} -> skip (forecast error), {elapsed:.2f}s")
            continue

        ytrue = float(df.loc[target_ts, gdp_col]) if pd.notna(df.loc[target_ts, gdp_col]) else np.nan

        rows.append(
            {
                "date": target_ts,
                "y_true": ytrue,
                "y_pred": float(yhat),
                "k_best": int(best_k),
                "crit_best": float(best_val),
                "criterion": str(criterion),
                "n_months_train": int(train.shape[0]),
                "status": best_status,
            }
        )
        if verbose and progress_every and (i % progress_every == 0):
            elapsed = time.perf_counter() - t_iter_start
            total_elapsed = time.perf_counter() - t0_all
            print(
                f"[DFM] {i}/{total_targets} {t.date()} -> ok (k={best_k}, train={train.shape[0]}), "
                f"{elapsed:.2f}s, total {total_elapsed/60:.1f}m"
            )

    out = pd.DataFrame(rows)
    if out.empty:
        # Return empty but well-typed frame
        return pd.DataFrame(columns=["y_true", "y_pred", "k_best", "crit_best", "criterion", "n_months_train"]).set_index(
            pd.DatetimeIndex([], name="date")
        )

    out = out.sort_values("date").set_index("date")
    # status column is optional diagnostics; drop it if you don’t want it
    # out = out.drop(columns=["status"])
    return out
