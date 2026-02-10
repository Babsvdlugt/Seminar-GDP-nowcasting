# DFM_helpers.py
import logging
import numpy as np
import calendar
from pathlib import Path
from typing import Dict, Optional, Union, Tuple, Any, List, Iterable, Callable
from dataclasses import dataclass

import pandas as pd

logger = logging.getLogger(__name__)

# -------------------------------------------------------------------
# RELEASE-LAG SPEC
# -------------------------------------------------------------------
LagSpec = Tuple[str, int]  # (unit, value) where unit in {"days","weeks","months"}

# Per-series metadata needed for CPB-style quasi real-time vintages
# Keys:
#   - freq:       {"M","Q","timestamp"}  (monthly, quarterly, or use index as-is)
#   - ref_point:  {"period_end","period_start","timestamp"}
#   - lag_unit:   {"days","weeks","months"}
#   - lag_value:  non-negative int
ReleaseLagMeta = Dict[str, Dict[str, Any]]


# -------------------------------------------------------------------
# DATA INLADEN
# -------------------------------------------------------------------
def load_dfm_ready(path: Path, freq: str = "MS") -> pd.DataFrame:
    """
    Load a DFM-ready CSV with a 'date' column.
    No data modification beyond:
      - parsing dates
      - sorting
      - setting frequency
      - coercing to numeric
    """
    df = pd.read_csv(path)

    if "date" not in df.columns:
        raise ValueError("CSV must contain a 'date' column.")

    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").set_index("date")
    df = df.asfreq(freq)

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    logger.info(f"Loaded data: shape={df.shape}, freq={freq}")
    return df


# -------------------------------------------------------------------
# PUBLICATION LAGS INLADEN (RICH META)
# -------------------------------------------------------------------
def load_release_lags_csv(path: Union[str, Path]) -> ReleaseLagMeta:
    """
    Load release lag information per series from a CSV file.

    REQUIRED
    --------
    - 'series' : name of the data series (must exactly match df.columns)

    OPTIONAL (recommended for CPB-style quasi real-time)
    ----------------------------------------------------
    - 'freq'      : 'M' or 'Q' (defaults to 'M')
    - 'ref_point' : 'period_end' or 'period_start' (defaults to 'period_end')

    OPTIONAL lags (at least one should be filled per series)
    --------------------------------------------------------
    - 'lag_days'
    - 'lag_weeks'
    - 'lag_months'

    Priority rule for lag columns:
        lag_days -> lag_weeks -> lag_months

    OUTPUT
    ------
    meta : dict[str, dict]
        meta[series] = {
            'freq': 'M'|'Q'|'timestamp',
            'ref_point': 'period_end'|'period_start'|'timestamp',
            'lag_unit': 'days'|'weeks'|'months',
            'lag_value': int
        }
    """
    path = Path(path)
    lags_df = pd.read_csv(path)

    if "series" not in lags_df.columns:
        raise ValueError("Release lags CSV must contain a 'series' column.")

    # Ensure optional meta columns exist
    if "freq" not in lags_df.columns:
        lags_df["freq"] = "M"
    if "ref_point" not in lags_df.columns:
        lags_df["ref_point"] = "period_end"

    # Ensure lag columns exist (if missing, treat as all-NaN)
    for col in ["lag_days", "lag_weeks", "lag_months"]:
        if col not in lags_df.columns:
            lags_df[col] = pd.NA

    out: ReleaseLagMeta = {}

    for _, row in lags_df.iterrows():
        series = str(row["series"]).strip()
        if series == "" or series.lower() == "nan":
            continue

        freq = str(row["freq"]).strip().upper()
        if freq not in {"M", "Q", "TIMESTAMP"}:
            # allow empty/NA to default to monthly
            freq = "M"
        if freq == "TIMESTAMP":
            freq = "timestamp"

        ref_point = str(row["ref_point"]).strip().lower()
        if ref_point not in {"period_end", "period_start", "timestamp"}:
            ref_point = "period_end"
        if freq == "timestamp":
            ref_point = "timestamp"

        # choose lag by priority: days -> weeks -> months
        lag_unit: Optional[str] = None
        lag_value: Optional[int] = None

        if pd.notna(row["lag_days"]):
            lag_unit, lag_value = "days", int(row["lag_days"])
        elif pd.notna(row["lag_weeks"]):
            lag_unit, lag_value = "weeks", int(row["lag_weeks"])
        elif pd.notna(row["lag_months"]):
            lag_unit, lag_value = "months", int(row["lag_months"])
        else:
            continue  # omit series without lag info

        if lag_value < 0:
            raise ValueError(f"Negative lag not allowed: {series} has {lag_unit}={lag_value}")

        out[series] = {
            "freq": freq,
            "ref_point": ref_point,
            "lag_unit": lag_unit,
            "lag_value": lag_value,
        }
        
        # pass-through schedule fields if present in the CSV
        for extra in ["release_day_in_month", "release_week_in_month", "release_weekday", "release_dates", "release_group"]:
            if extra in lags_df.columns and pd.notna(row.get(extra, pd.NA)):
                v_extra = row.get(extra)
                if extra in {"release_day_in_month", "release_week_in_month", "release_weekday"}:
                    v_extra = int(v_extra)
                out[series][extra] = v_extra


    return out


# -------------------------------------------------------------------
# INTERNAL: reference dates & availability dates (vectorized)
# -------------------------------------------------------------------
def _reference_dates_for_index(
    idx: pd.DatetimeIndex, freq: str, ref_point: str
) -> pd.DatetimeIndex:
    """
    Map each index timestamp to the reference date of its observation period.

    - freq='M' and ref_point='period_end'   -> month end of that month
    - freq='M' and ref_point='period_start' -> month start of that month
    - freq='Q' and ref_point='period_end'   -> quarter end
    - freq='Q' and ref_point='period_start' -> quarter start
    - freq='timestamp' or ref_point='timestamp' -> idx itself
    """
    if freq == "timestamp" or ref_point == "timestamp":
        return pd.DatetimeIndex(idx)

    if freq == "M":
        p = idx.to_period("M")
        how = "end" if ref_point == "period_end" else "start"
        return pd.DatetimeIndex(p.to_timestamp(how=how))

    if freq == "Q":
        p = idx.to_period("Q")
        how = "end" if ref_point == "period_end" else "start"
        return pd.DatetimeIndex(p.to_timestamp(how=how))

    # fallback: treat as timestamps
    return pd.DatetimeIndex(idx)


def _add_lag_to_dates(ref_dates: pd.DatetimeIndex, unit: str, value: int) -> pd.DatetimeIndex:
    """availability_date = reference_date + lag"""
    value = int(value)
    if unit == "days":
        return pd.DatetimeIndex(ref_dates + pd.Timedelta(days=value))
    if unit == "weeks":
        return pd.DatetimeIndex(ref_dates + pd.Timedelta(weeks=value))
    if unit == "months":
        return pd.DatetimeIndex(ref_dates + pd.DateOffset(months=value))
    raise ValueError(f"Unknown lag unit '{unit}'")


def _availability_mask(
    idx: pd.DatetimeIndex,
    as_of: pd.Timestamp,
    meta: Dict[str, Any],
) -> pd.Series:
    """
    Returns a boolean mask (indexed like idx) indicating whether an observation
    would be available at 'as_of' given (freq, ref_point, lag_unit, lag_value).
    """
    freq = meta.get("freq", "M")
    ref_point = meta.get("ref_point", "period_end")
    unit = meta.get("lag_unit", "months")
    value = int(meta.get("lag_value", 0))

    ref_dates = _reference_dates_for_index(idx, freq=freq, ref_point=ref_point)
    avail_dates = _add_lag_to_dates(ref_dates, unit=unit, value=value)
    return pd.Series(avail_dates <= as_of, index=idx)


# -------------------------------------------------------------------
# VINTAGE MAKEN (CPB-style: availability(t,i) <= as_of)
# -------------------------------------------------------------------
def make_vintage(
    df: pd.DataFrame,
    as_of: Union[str, pd.Timestamp],
    use_real_lags: bool = False,
    lags: Optional[ReleaseLagMeta] = None,
    default_meta: Optional[Dict[str, Any]] = None,
    verbose: bool = False,
    max_log_cols: int = 5,
) -> pd.DataFrame:
    """
    Construct a vintage (real-time) version of the dataset as of a given date.

    If use_real_lags=False:
      - Simple truncation: set all rows after as_of to NA.

    If use_real_lags=True:
      - CPB-style availability: an observation is available at as_of iff
        (reference_date + release_lag) <= as_of, applied per series.
    """
    if not isinstance(df.index, pd.DatetimeIndex):
        raise ValueError("df.index must be a DatetimeIndex.")

    as_of = pd.to_datetime(as_of)
    v = df.copy()

    if not use_real_lags:
        v.loc[v.index > as_of, :] = pd.NA
        if verbose:
            logger.info(f"make_vintage: as_of={as_of.date()} | use_real_lags=False")
        return v

    if lags is None:
        raise ValueError("use_real_lags=True requires `lags` (from load_release_lags_csv).")

    if default_meta is None:
        default_meta = {
            "freq": "M",
            "ref_point": "period_end",
            "lag_unit": "months",
            "lag_value": 0,
        }

    idx = v.index

    # Group columns by identical lag meta to avoid recomputing masks repeatedly
    groups: Dict[Tuple[str, str, str, int], List[str]] = {}
    for col in v.columns:
        meta_i = lags.get(col, default_meta)
        key = (
            str(meta_i.get("freq", "M")),
            str(meta_i.get("ref_point", "period_end")),
            str(meta_i.get("lag_unit", "months")),
            int(meta_i.get("lag_value", 0)),
        )
        groups.setdefault(key, []).append(col)

    examples = []

    for (freq, ref_point, unit, value), cols in groups.items():
        meta_key = {"freq": freq, "ref_point": ref_point, "lag_unit": unit, "lag_value": value}

        # availability mask: True = available; False = not yet published at as_of
        avail = _availability_mask(idx=idx, as_of=as_of, meta=meta_key)

        # set unavailable observations to NA for all columns in this group
        v.loc[~avail, cols] = pd.NA

        if verbose and len(examples) < max_log_cols:
            examples.append((cols[0], meta_key))  # log one representative series per group

    if verbose:
        logger.info(f"make_vintage: as_of={as_of.date()} | use_real_lags=True | groups={len(groups)}")
        for col, meta in examples:
            logger.info(
                f"  {col}: freq={meta.get('freq')} ref_point={meta.get('ref_point')} "
                f"lag={meta.get('lag_value')} {meta.get('lag_unit')}"
            )
        if len(groups) > max_log_cols:
            logger.info(f"  (logged {max_log_cols} of {len(groups)} meta-groups)")

    return v



# -------------------------------------------------------------------
# NOWCAST-ONLY: map as_of -> target (unknown) GDP quarter
# -------------------------------------------------------------------
def get_nowcast_target_quarter(
    as_of: Union[str, pd.Timestamp],
    df_index: pd.DatetimeIndex,
    gdp_meta: Dict[str, Any],
) -> Tuple[pd.Period, pd.Timestamp]:
    """
    Nowcast-only helper.

    Returns:
      - target_q: the first quarter whose GDP would NOT yet be available at as_of
      - target_q_end: timestamp of that quarter's end date (period_end)

    This uses the same availability-date logic as make_vintage:
      availability(q) = quarter_end(q) + lag_gdp
      GDP for quarter q is known iff availability(q) <= as_of.

    Assumptions:
      - GDP is quarterly: gdp_meta['freq'] should be 'Q' (recommended).
      - We interpret the GDP "observation period" by quarter end by default.
    """
    as_of = pd.to_datetime(as_of)

    freq = str(gdp_meta.get("freq", "Q")).upper()
    if freq != "Q":
        raise ValueError(f"get_nowcast_target_quarter expects GDP freq 'Q', got '{freq}'")

    ref_point = str(gdp_meta.get("ref_point", "period_end")).lower()
    if ref_point not in {"period_end", "period_start"}:
        ref_point = "period_end"

    # Build a candidate set of quarters that covers the sample plus a little forward
    start_q = df_index.min().to_period("Q")
    end_q = df_index.max().to_period("Q") + 8  # look ahead up to 2 years

    quarters = pd.period_range(start=start_q, end=end_q, freq="Q")

    # Reference date per quarter
    if ref_point == "period_end":
        q_ref = quarters.to_timestamp(how="end")
    else:
        q_ref = quarters.to_timestamp(how="start")

    # Availability date per quarter: reference + lag
    unit = gdp_meta.get("lag_unit", "months")
    value = int(gdp_meta.get("lag_value", 0))
    q_avail = _add_lag_to_dates(pd.DatetimeIndex(q_ref), unit=unit, value=value)

    # First quarter not yet available at as_of
    not_available = q_avail > as_of
    if not not_available.any():
        # If everything looks "available", target the next quarter after the last candidate
        target_q = quarters[-1] + 1
    else:
        target_q = quarters[not_available.argmax()]

    target_q_end = target_q.to_timestamp(how="end")
    return target_q, pd.Timestamp(target_q_end)

# ------------------------------------------------------------
# DATACLASSES used by helpers
# ------------------------------------------------------------
@dataclass(frozen=True)
class UpdateEvent:
    """One update moment where a batch of series is released."""
    as_of: pd.Timestamp
    released_series: Tuple[str, ...]
    label: Optional[str] = None


@dataclass(frozen=True)
class DFMSpec:
    """One DFM specification in the CPB mixture grid."""
    r: int
    p: int

    def model_id(self) -> str:
        return f"DFM_r{self.r}_p{self.p}"


@dataclass(frozen=True)
class TargetBounds:
    """Bounds used for trimming forecasts."""
    y_min: float
    y_max: float


@dataclass
class MixtureRunOutput:
    """Output container for one mixture run at one update."""
    as_of: pd.Timestamp
    target_quarter: pd.Period
    per_model_forecast: Dict[str, float]
    per_model_status: Dict[str, Dict[str, Any]]
    mixture_forecast: float
    mixture_components_used: List[str]
    n_failed: int
    n_trimmed: int


@dataclass
class ForecastRecord:
    """Stores one forecast outcome for fallback history."""
    as_of: pd.Timestamp
    target_quarter: pd.Period
    value: float
    model_id: str
    status: Dict[str, Any]


@dataclass
class FallbackDecision:
    """Describes whether fallback was applied and what value was returned."""
    used_fallback: bool
    fallback_source: Optional[str]
    value: float
    note: str

# ------------------------------------------------------------
# UPDATE SCHEDULE (representative release calendar)
# ------------------------------------------------------------
def clamp_day_to_month_end(year: int, month: int, day: int) -> int:
    last = calendar.monthrange(int(year), int(month))[1]
    return int(min(int(day), last))


def get_series_release_dates(
    series: str,
    meta_i: Dict[str, Any],
    start: pd.Timestamp,
    end: pd.Timestamp,
) -> List[pd.Timestamp]:
    start = pd.to_datetime(start).normalize()
    end = pd.to_datetime(end).normalize()
    if end < start:
        return []

    # Option C: explicit dates
    if "release_dates" in meta_i and meta_i["release_dates"] is not None and str(meta_i["release_dates"]).strip():
        raw = str(meta_i["release_dates"])
        parts = [p.strip() for p in raw.split(";") if p.strip()]
        dates = []
        for p in parts:
            d = pd.to_datetime(p, errors="coerce")
            if pd.isna(d):
                continue
            d = d.normalize()
            if start <= d <= end:
                dates.append(d)
        return sorted(set(dates))

    months = pd.period_range(start=start.to_period("M"), end=end.to_period("M"), freq="M")

    # Option A: fixed day in month
    if "release_day_in_month" in meta_i and pd.notna(meta_i["release_day_in_month"]):
        day = int(meta_i["release_day_in_month"])
        out = []
        for m in months:
            y, mo = int(m.year), int(m.month)
            d = clamp_day_to_month_end(y, mo, day)
            ts = pd.Timestamp(year=y, month=mo, day=d)
            if start <= ts <= end:
                out.append(ts)
        return out

    # Option B: nth weekday
    if (
        "release_week_in_month" in meta_i
        and "release_weekday" in meta_i
        and pd.notna(meta_i["release_week_in_month"])
        and pd.notna(meta_i["release_weekday"])
    ):
        week_in_month = int(meta_i["release_week_in_month"])
        weekday = int(meta_i["release_weekday"])  # 0=Mon..6=Sun
        if week_in_month < 1:
            return []

        out = []
        for m in months:
            y, mo = int(m.year), int(m.month)
            first = pd.Timestamp(year=y, month=mo, day=1)
            first_wd = int(first.weekday())
            delta = (weekday - first_wd) % 7
            first_target = first + pd.Timedelta(days=delta)
            candidate = first_target + pd.Timedelta(days=7 * (week_in_month - 1))

            if candidate.month != mo:
                last_day = calendar.monthrange(y, mo)[1]
                last = pd.Timestamp(year=y, month=mo, day=last_day)
                back = (int(last.weekday()) - weekday) % 7
                candidate = last - pd.Timedelta(days=back)

            candidate = candidate.normalize()
            if start <= candidate <= end:
                out.append(candidate)

        return out

    return []


def group_releases_by_day(
    releases: Dict[pd.Timestamp, List[str]],
    *,
    sort_series: bool = True,
) -> List[UpdateEvent]:
    events: List[UpdateEvent] = []
    for d, sers in releases.items():
        if not sers:
            continue
        if sort_series:
            sers = sorted(set(sers))
        events.append(UpdateEvent(as_of=pd.to_datetime(d).normalize(), released_series=tuple(sers)))
    events.sort(key=lambda e: e.as_of)
    return events


def build_representative_update_schedule(
    meta: ReleaseLagMeta,
    start: Union[str, pd.Timestamp],
    end: Union[str, pd.Timestamp],
    *,
    include_empty_events: bool = False,
    bundle_same_day: bool = True,
    sort_series: bool = True,
    verbose: bool = False,
) -> List[UpdateEvent]:
    start_ts = pd.to_datetime(start).normalize()
    end_ts = pd.to_datetime(end).normalize()
    if end_ts < start_ts:
        return []

    releases_by_day: Dict[pd.Timestamp, List[str]] = {}
    for series, meta_i in meta.items():
        dates = get_series_release_dates(series, meta_i, start_ts, end_ts)
        for d in dates:
            d = pd.to_datetime(d).normalize()
            releases_by_day.setdefault(d, []).append(series)

    events = (
        group_releases_by_day(releases_by_day, sort_series=sort_series)
        if bundle_same_day
        else [UpdateEvent(as_of=d, released_series=(s,)) for d, ss in releases_by_day.items() for s in (sorted(ss) if sort_series else ss)]
    )
    events.sort(key=lambda e: e.as_of)

    if include_empty_events:
        all_days = pd.date_range(start=start_ts, end=end_ts, freq="D")
        existing = {e.as_of for e in events}
        for d in all_days:
            d = pd.to_datetime(d).normalize()
            if d not in existing:
                events.append(UpdateEvent(as_of=d, released_series=tuple()))
        events.sort(key=lambda e: e.as_of)

    if verbose:
        logger.info(f"build_representative_update_schedule: start={start_ts} end={end_ts} events={len(events)}")

    return events


# ------------------------------------------------------------
# EXPANDING WINDOW
# ------------------------------------------------------------
def validate_expanding_window_monotonicity(schedule: List[Any]) -> None:
    asofs = [pd.to_datetime(getattr(ev, "as_of", ev["as_of"])) for ev in schedule]
    if any(asofs[i] >= asofs[i + 1] for i in range(len(asofs) - 1)):
        raise ValueError("Schedule as_of dates must be strictly increasing for expanding-window recursion.")


def get_training_sample_expanding(
    df_full: pd.DataFrame,
    as_of: Union[str, pd.Timestamp],
    *,
    start_date: Optional[Union[str, pd.Timestamp]] = None,
) -> pd.DataFrame:
    as_of = pd.to_datetime(as_of)
    if start_date is None:
        return df_full.loc[df_full.index <= as_of].copy()
    start_date = pd.to_datetime(start_date)
    return df_full.loc[(df_full.index >= start_date) & (df_full.index <= as_of)].copy()


def build_prev_update_cache_key(
    target_quarter: pd.Period,
    as_of: pd.Timestamp,
    model_id: str,
) -> str:
    as_of = pd.to_datetime(as_of)
    return f"{model_id}|{str(target_quarter)}|{as_of.strftime('%Y-%m-%d')}"


# ------------------------------------------------------------
# MIXTURE GRID + TRIMMING
# ------------------------------------------------------------
def build_cpb_mixture_grid(
    r_values: Tuple[int, ...] = (2, 3, 4, 5),
    p_values: Tuple[int, ...] = (1, 2, 3),
) -> List[DFMSpec]:
    return [DFMSpec(r=r, p=p) for r in r_values for p in p_values]


def compute_target_bounds(
    y: pd.Series,
    *,
    method: str = "minmax",
    quantiles: Tuple[float, float] = (0.01, 0.99),
) -> TargetBounds:
    y = pd.to_numeric(y, errors="coerce").dropna()
    if y.empty:
        raise ValueError("Cannot compute bounds: target series is empty after dropping NaNs.")

    method = str(method).lower()
    if method == "minmax":
        return TargetBounds(float(y.min()), float(y.max()))
    if method == "quantile":
        ql, qh = quantiles
        return TargetBounds(float(y.quantile(ql)), float(y.quantile(qh)))

    raise ValueError("method must be one of {'minmax','quantile'}")


def trim_forecast_to_bounds(y_hat: float, bounds: TargetBounds) -> float:
    if y_hat is None or (isinstance(y_hat, float) and np.isnan(y_hat)):
        return y_hat
    return float(min(max(float(y_hat), bounds.y_min), bounds.y_max))


def simple_average(values: List[float]) -> float:
    vals = [float(v) for v in values if v is not None and not (isinstance(v, float) and np.isnan(v))]
    if len(vals) == 0:
        raise ValueError("simple_average: no valid values.")
    return float(np.mean(vals))


def run_dfm_mixture_for_update(
    *,
    vintage_df: pd.DataFrame,
    as_of: pd.Timestamp,
    target_quarter: pd.Period,
    gdp_series: str,
    specs: List[DFMSpec],
    bounds: TargetBounds,
    fit_and_forecast_single: Callable[..., Tuple[float, Dict[str, Any]]],
    trim_before_mix: bool = True,
    drop_failed: bool = True,
    fallback_value: Optional[float] = None,
    verbose: bool = False,
) -> MixtureRunOutput:
    """
    Run the CPB-style DFM mixture for a single as-of update:
      - Fit each (r,p) spec (12 models by default)
      - Optionally trim forecasts to target bounds
      - Average across usable models (or fallback if none usable)

    Failure handling is centralized via is_failed_fit(status).
    """
    as_of = pd.to_datetime(as_of)

    per_model_forecast: Dict[str, float] = {}
    per_model_status: Dict[str, Dict[str, Any]] = {}

    used_models: List[str] = []
    used_values: List[float] = []
    n_failed = 0
    n_trimmed = 0

    for spec in specs:
        mid = spec.model_id()

        try:
            y_hat, status = fit_and_forecast_single(
                vintage_df=vintage_df,
                as_of=as_of,
                target_quarter=target_quarter,
                gdp_series=gdp_series,
                spec=spec,
            )
            if status is None:
                status = {}
            status.setdefault("ok", True)
        except Exception as e:
            y_hat = np.nan
            status = {"ok": False, "converged": False, "error": str(e)}

        per_model_status[mid] = status

        failed = is_failed_fit(status)

        # Store raw forecast (even if failed) for diagnostics
        per_model_forecast[mid] = float(y_hat) if y_hat is not None else np.nan

        if failed:
            n_failed += 1

            # Optionally still include failed outputs if they are numeric
            if not drop_failed and y_hat is not None and not (isinstance(y_hat, float) and np.isnan(y_hat)):
                y_use = float(y_hat)
                if trim_before_mix:
                    y_trim = trim_forecast_to_bounds(y_use, bounds)
                    if y_trim != y_use:
                        n_trimmed += 1
                    y_use = y_trim
                used_models.append(mid)
                used_values.append(y_use)

            continue

        # Success: include in mixture
        y_use = float(y_hat)
        if trim_before_mix:
            y_trim = trim_forecast_to_bounds(y_use, bounds)
            if y_trim != y_use:
                n_trimmed += 1
            y_use = y_trim

        used_models.append(mid)
        used_values.append(y_use)

        # Keep the (possibly trimmed) value in per_model_forecast
        per_model_forecast[mid] = y_use

    # Mixture aggregation / fallback
    if len(used_values) == 0:
        if fallback_value is None:
            raise ValueError(
                "All DFM specs failed (or were dropped). Provide fallback_value or set drop_failed=False."
            )
        mix = float(fallback_value)
        used_models = []
    else:
        mix = simple_average(used_values)

    if verbose:
        logger.info(
            f"mixture update as_of={as_of.date()} target={target_quarter} "
            f"used={len(used_models)}/{len(specs)} failed={n_failed} trimmed={n_trimmed} mix={mix:.6f}"
        )

    return MixtureRunOutput(
        as_of=as_of,
        target_quarter=target_quarter,
        per_model_forecast=per_model_forecast,
        per_model_status=per_model_status,
        mixture_forecast=mix,
        mixture_components_used=used_models,
        n_failed=n_failed,
        n_trimmed=n_trimmed,
    )


# ------------------------------------------------------------
# CPB-STYLE FALLBACK (CONVERGENCE FAILURES)
# ------------------------------------------------------------
TREAT_NONCONVERGENCE_AS_FAILURE = True  # CPB-consistent default

def is_failed_fit(status: Dict[str, Any]) -> bool:
    ok = bool(status.get("ok", True))
    converged = status.get("converged", True)
    err = status.get("error", None)

    if err is not None:
        return True
    if ok is False:
        return True
    if TREAT_NONCONVERGENCE_AS_FAILURE and (converged is False):
        return True
    return False



def update_forecast_history(
    history: Dict[Tuple[str, pd.Period], List[ForecastRecord]],
    record: ForecastRecord,
) -> None:
    key = (record.model_id, record.target_quarter)
    history.setdefault(key, []).append(record)
    # keep sorted by as_of to make "previous" retrieval deterministic
    history[key].sort(key=lambda r: r.as_of)


def get_previous_update_forecast(
    history: Dict[Tuple[str, pd.Period], List[ForecastRecord]],
    model_id: str,
    target_quarter: pd.Period,
    as_of: pd.Timestamp,
) -> Optional[ForecastRecord]:
    key = (model_id, target_quarter)
    if key not in history:
        return None
    as_of = pd.to_datetime(as_of)
    prev = [r for r in history[key] if r.as_of < as_of]
    return prev[-1] if prev else None


def get_previous_valid_forecast(
    history: Dict[Tuple[str, pd.Period], List[ForecastRecord]],
    model_id: str,
    target_quarter: pd.Period,
    as_of: pd.Timestamp,
) -> Optional[ForecastRecord]:
    key = (model_id, target_quarter)
    if key not in history:
        return None
    as_of = pd.to_datetime(as_of)
    prev = [r for r in history[key] if r.as_of < as_of and (not is_failed_fit(r.status))]
    return prev[-1] if prev else None


def apply_cpb_fallback_policy(
    *,
    status: Dict[str, Any],
    model_id: str,
    target_quarter: pd.Period,
    as_of: pd.Timestamp,
    history: Dict[Tuple[str, pd.Period], List[ForecastRecord]],
    current_value: Optional[float] = None,
    prefer_previous_update: bool = True,
) -> FallbackDecision:
    as_of = pd.to_datetime(as_of)

    # If success, return current
    if not is_failed_fit(status) and current_value is not None and not (isinstance(current_value, float) and np.isnan(current_value)):
        return FallbackDecision(
            used_fallback=False,
            fallback_source=None,
            value=float(current_value),
            note="ok",
        )

    # Failure -> CPB-style: previous update for SAME target quarter
    rec = None
    if prefer_previous_update:
        rec = get_previous_update_forecast(history, model_id, target_quarter, as_of)
        if rec is not None and not (isinstance(rec.value, float) and np.isnan(rec.value)):
            return FallbackDecision(
                used_fallback=True,
                fallback_source="previous_update",
                value=float(rec.value),
                note="fit failed; used previous update forecast",
            )

    # Otherwise: previous valid
    rec = get_previous_valid_forecast(history, model_id, target_quarter, as_of)
    if rec is not None and not (isinstance(rec.value, float) and np.isnan(rec.value)):
        return FallbackDecision(
            used_fallback=True,
            fallback_source="previous_valid",
            value=float(rec.value),
            note="fit failed; used last valid forecast",
        )

    # No fallback available
    raise ValueError(
        f"CPB fallback failed: no previous forecast available for model_id={model_id}, target_quarter={target_quarter} before as_of={as_of.date()}."
    )
