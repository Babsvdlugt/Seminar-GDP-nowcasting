
# DFM_main.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.dynamic_factor import DynamicFactor

from DFM_helpers import (
    DFMSpec,
    ForecastRecord,
    TargetBounds,
    apply_cpb_fallback_policy,
    build_cpb_mixture_grid,
    compute_target_bounds,
    get_nowcast_target_quarter,
    get_training_sample_expanding,
    load_dfm_ready,
    load_release_lags_csv,
    make_vintage,
    run_dfm_mixture_for_update,
    update_forecast_history,
)

logger = logging.getLogger("DFM_MAIN")

# -----------------------------
# Config 
# -----------------------------
DATA_PATH = Path("/Users/babsvanderlugt/Downloads/seminar vs code/Seminar-GDP-nowcasting-12/data_transformations_DFM_ready_state_space.csv")
LAGS_PATH = Path("/Users/babsvanderlugt/Downloads/seminar vs code/Seminar-GDP-nowcasting-12/release_lags_clean.csv")

# -----------------------------
# Settings
# -----------------------------
GDP_SERIES = "GrossDomesticProduct_1"

# CPB mixture grid: r=2..5, p=1..3
R_VALUES = (2, 3, 4, 5)
P_VALUES = (1, 2, 3)

# Fit settings
FIT_MAXITER = 200
FIT_METHOD = "powell"
# "powell" = more robust but slower (derivative-free optimizer),
# "lbfgs"  = much faster gradient-based optimizer, but may fail to converge more often
#            when the data contain many missing values (ragged-edge vintages)
FIT_DISP = False

# As-of grid: month-end schedule (simple and stable choice)
ASOF_START = "2010-01-31"   # first real-time "observation moment" (data available at that date)
ASOF_END   = "2010-05-31"   # last as-of in this run (short window for faster testing/debugging)
# This range determines for which update moments we construct vintages and produce nowcasts.
# In the final analysis, this typically runs until the end of the sample (e.g. 2019/2020).

# Generate month-end "as-of" dates (information sets) used to build real-time vintages and nowcasts.
def month_end_grid(df_index: pd.DatetimeIndex, start=None, end=None) -> list[pd.Timestamp]:
    # "ME" means Month-End frequency in pandas.
    
    idx_min = df_index.min()
    idx_max = df_index.max()

    if start is None:
        start = idx_min.to_period("M").to_timestamp(how="end")
    else:
        start = pd.to_datetime(start)

    if end is None:
        end = idx_max.to_period("M").to_timestamp(how="end")
    else:
        end = pd.to_datetime(end)

    grid = pd.date_range(start=start, end=end, freq="ME")  # month-end
    return [pd.Timestamp(x) for x in grid]

# Compute the number of whole months between two timestamps (used to determine forecast horizon length).
def _months_diff(a: pd.Timestamp, b: pd.Timestamp) -> int:
    return (b.year - a.year) * 12 + (b.month - a.month)


def fit_and_forecast_single_statsmodels(
    *,
    vintage_df: pd.DataFrame,
    as_of: pd.Timestamp,
    target_quarter: pd.Period,
    gdp_series: str,
    spec: DFMSpec,
) -> Tuple[float, Dict[str, Any]]:
    """
    Fit a DynamicFactor model on vintage_df up to as_of (expanding window)
    and forecast GDP at target_quarter (timestamp = quarter start).

    GDP in jullie dataset lijkt op kwartaal-start (Jan/Apr/Jul/Oct) te staan,
    dus target_ts = quarter start is logisch.
    """
    as_of = pd.to_datetime(as_of)
    train = get_training_sample_expanding(vintage_df, as_of=as_of)

    if gdp_series not in train.columns:
        raise ValueError(f"GDP series '{gdp_series}' not in dataframe columns.")

    endog = train.copy()

    model = DynamicFactor(
        endog=endog,
        k_factors=int(spec.r),
        factor_order=int(spec.p),
        error_cov_type="diagonal",
    )

    res = model.fit(maxiter=FIT_MAXITER, method=FIT_METHOD, disp=FIT_DISP)

    converged = bool(getattr(res, "mle_retvals", {}).get("converged", True))

    target_ts = target_quarter.to_timestamp(how="start")
    last_train_ts = endog.index.max()

    if target_ts <= last_train_ts:
        pred = res.get_prediction(start=target_ts, end=target_ts)
        yhat = float(pred.predicted_mean[gdp_series].iloc[0])
    else:
        steps = _months_diff(last_train_ts, target_ts)
        steps = max(1, steps)
        fc = res.get_forecast(steps=steps)
        yhat = float(fc.predicted_mean[gdp_series].iloc[-1])

    status = {
        "ok": True,
        "converged": converged,
        "llf": float(getattr(res, "llf", np.nan)),
        "aic": float(getattr(res, "aic", np.nan)),
        "bic": float(getattr(res, "bic", np.nan)),
        "spec": {"r": int(spec.r), "p": int(spec.p)},
    }
    return yhat, status


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    logger.info("Loading data + lags...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"DATA_PATH not found: {DATA_PATH}")
    if not LAGS_PATH.exists():
        raise FileNotFoundError(f"LAGS_PATH not found: {LAGS_PATH}")

    df = load_dfm_ready(DATA_PATH, freq="MS")
    lags = load_release_lags_csv(LAGS_PATH)

    if GDP_SERIES not in df.columns:
        raise ValueError(f"GDP_SERIES='{GDP_SERIES}' not found in data columns.")

    # Bounds (CPB trimming)
    bounds: TargetBounds = compute_target_bounds(df[GDP_SERIES], method="minmax")
    logger.info(f"Target bounds: {bounds.y_min:.4f} .. {bounds.y_max:.4f}")

    # Mixture grid
    specs = build_cpb_mixture_grid(r_values=R_VALUES, p_values=P_VALUES)
    logger.info(f"Mixture grid: {len(specs)} DFMs")

    # As-of schedule
    asofs = month_end_grid(df.index, start=ASOF_START, end=ASOF_END)
    logger.info(f"As-of dates: n={len(asofs)} | {asofs[0].date()} .. {asofs[-1].date()}")

    # Fallback history
    history: Dict[Tuple[str, pd.Period], list[ForecastRecord]] = {}

    rows = []

    # GDP meta: als niet in lags.csv, default naar Q met period_end
    gdp_meta = lags.get(
        GDP_SERIES,
        {"freq": "Q", "ref_point": "period_end", "lag_unit": "days", "lag_value": 45},
    )
    # zorg dat freq echt Q is voor get_nowcast_target_quarter
    gdp_meta = dict(gdp_meta)
    gdp_meta["freq"] = "Q"

    for as_of in asofs:
        # 1) vintage
        vintage = make_vintage(df, as_of=as_of, use_real_lags=True, lags=lags, verbose=False)

        # 2) target quarter (nowcast-only)
        target_q, _ = get_nowcast_target_quarter(as_of, df.index, gdp_meta=gdp_meta)

        # 3) per-model fallback wrapper (previous update if fail)
        def fit_single_with_fallback(**kwargs):
            spec_local: DFMSpec = kwargs["spec"]
            model_id = spec_local.model_id()

            try:
                yhat, status = fit_and_forecast_single_statsmodels(**kwargs)
                decision = apply_cpb_fallback_policy(
                    status=status,
                    model_id=model_id,
                    target_quarter=kwargs["target_quarter"],
                    as_of=kwargs["as_of"],
                    history=history,
                    current_value=yhat,
                    prefer_previous_update=True,
                )
                status = dict(status)
                status["used_fallback"] = decision.used_fallback
                status["fallback_source"] = decision.fallback_source
                status["note"] = decision.note
                return float(decision.value), status

            except Exception as e:
                status = {"ok": False, "converged": False, "error": str(e)}
                decision = apply_cpb_fallback_policy(
                    status=status,
                    model_id=model_id,
                    target_quarter=kwargs["target_quarter"],
                    as_of=kwargs["as_of"],
                    history=history,
                    current_value=None,
                    prefer_previous_update=True,
                )
                status["used_fallback"] = decision.used_fallback
                status["fallback_source"] = decision.fallback_source
                status["note"] = decision.note
                return float(decision.value), status

        # 4) mixture run
        gdp_hist = vintage[GDP_SERIES].dropna()
        rw_fallback = float(gdp_hist.iloc[-1]) if len(gdp_hist) > 0 else 0.0

        mix_out = run_dfm_mixture_for_update(
            vintage_df=vintage,
            as_of=as_of,
            target_quarter=target_q,
            gdp_series=GDP_SERIES,
            specs=specs,
            bounds=bounds,
            fit_and_forecast_single=fit_single_with_fallback,
            trim_before_mix=True,
            drop_failed=True,              # drop failures; je fallback zit per model al
            fallback_value=rw_fallback,    # maar als ALLES faalt: RW
            verbose=True,
        )


        # 5) update history
        for mid, val in mix_out.per_model_forecast.items():
            rec = ForecastRecord(
                as_of=as_of,
                target_quarter=target_q,
                value=float(val) if val is not None else np.nan,
                model_id=mid,
                status=mix_out.per_model_status.get(mid, {}),
            )
            update_forecast_history(history, rec)

        rows.append(
            {
                "as_of": as_of,
                "target_quarter": str(target_q),
                "mixture_forecast": float(mix_out.mixture_forecast),
                "n_models_used": len(mix_out.mixture_components_used),
                "n_failed": int(mix_out.n_failed),
                "n_trimmed": int(mix_out.n_trimmed),
            }
        )

    out = pd.DataFrame(rows).sort_values("as_of").reset_index(drop=True)
    out_path = "dfm_mixture_results.csv"
    out.to_csv(out_path, index=False)
    logger.info(f"Saved -> {out_path}")

    print(out.tail(10).to_string(index=False))


if __name__ == "__main__":
    main()
