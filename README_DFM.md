# README — CPB-style DFM Mixture (Dutch GDP nowcasting)

This folder contains a **CPB-style pseudo real-time nowcasting pipeline** for Dutch quarterly GDP using a **Dynamic Factor Model (DFM)** estimated on a **ragged-edge** monthly/quarterly panel. The implementation follows the core DFM-nowcasting idea used by many institutions (Kalman filter on unbalanced panels, updating as new releases arrive). ([ScienceDirect][1])

The code is split into:

* `DFM_Model.py` — main script: loads data, constructs vintages, runs the mixture of DFMs, writes results. 
* `DFM_helpers.py` — utilities for loading data, lags, building vintages, mixture aggregation, and CPB-style fallback logic. 

(Background/context on the “nowcast the present-edge publication delays is also described in the project kickoff material.) 

---

## 1) What the model does

For each **inform-of date”, typically month-end), the pipeline:

1. **Builds a real-time vintage**: for each series, observations are kept only if they would have been **available by that as-of date** given a per-series publication lag specification. 
2. Determines the **nowcast target quarter**: that would **not yet be published** at that as-of date (based on GDP’s own lag).
3. Fits a **mixture (ensemble) of 12 DFMs** and produces a **single mixture nowcast**:

   * Factors (r \in {2,3,4,5})
   * Factor VAR order (p \in {1,2,3})
   * Mixture = simple average across usable model outputs

This “mixture over specs” is the robustness trick: you don’t bet on one (r, p).

---

## 2) Inputs

### A) DFM-ready data panel (CSV)

Configured in `DFM_Model.py` as `DATA_PATH`. 

Expected format:

* A column named `date` (parse index).
* All other columns are numeric series (monthly, quarterly, or “timestamped” series).
* Missing values are allowed (ragged-edge is expected). 

The loader:

* sorts by date,
* sets a monthly fault),
* coerces values to numeric. 

### B) Publication lags file (CSV)

Configured iLAGS_PATH`. 

Required column:

* `series` — must match the daly.

Recommended metadata columns:

* `freq`: `M` or `Q` (default `M`)
* `ref_point`: `period_end` or `period_start` (default `period_end`)
* One of: `lag_days`, `lag_weeks`, `lag_months` (priority: days → weeks → months)

Optional scheduling columns supported (pass-through):

* `release_day_in_month`, `release_week_in_month`, `release_weekday`, `release_dates`, `release_group`

---

## 3) Core methodology details

### Real-time vintages (“ragged edge”)

`make_vintage(df, as_of, use_real_lags=True, lags=...)` applies the CPB-style availability rule:

> observation is available at `as_of` iff `(reference_date + release_lag) <= as_of`

This is computed per series, using `freq` + `ref_point` to map each timestamp to the appropriate reference date (month end/start, quarter end/start, or raw timestamp).

### Target quarter selection (nowcast-only)

`get_nowcast_target_quarter(as_of, df.index, gdp_meta=...)` selects the first GDP quarter whose availability date is **after** `as_of`. GDP is expected to be quarterly (`freq='Q'`).

If GDP isn’t found in the lags file, `DFM_Model.py` defaults GDP to a **45-day lag** from quarter end.

### DFM estimation and forecasting

Each model is a `statsmodels` `DynamicFactor` with:

* `k_factors = r`
* `factor_order = p`
* `error_cov_type="diagonal"` 

Forecast extraction:

* If target timestamp is mple → `get_prediction`
* Else → `get_forecast(steps=months_diff)`

(Conceptually aligned with the standard DFM + Kalman filter treatment of unbalanced panels for nowcasting.) ([ScienceDirect][1])

### Trimming bounds (CPB-style sanity bounds)

Bounds are computed from historical GDP (default `minmax`) and each model forecast can be clipped to ([y_{\min}, y_{\max}]) before averaging.

### Fallback policy (important)

Model fits can fail or not converge (common in ragged-edge ML estimation). Failure is defined as:

* explicit error, or
* `ok=False`, or
* non-convergence (if `TREAT_NONCONVERGENCE_AS_FAILURE=True`). 

Fallback is applied **per model** (CPB-style):

1. If the fit fails at `as_of`, use the **previous update’s forecast for the same talable.
2. Else use the **previous valid forecast**. 
3. If *all* models fail at an update moment, the mixture returns a **random-walk fallback**: last observed GDP in the vintage. file7

---

## 4) How to run

### Dependencies

Install (minimal):

* `pandas`
* `numpy`
* `statsmodels`

### Configure paths

In `DFM_Model.py`, update:

* `DATA_PATH` (DFM-ready panel CSV)
* `LAGS_PATH` (release lags CSV) 

Also set:

* `GDP_SERIES` (default `GrossDomesticProduct_1`)
* `ASOF_START`, `ASOF_END` (debug window by default) Run

```bash
python DFM_Model.py
```

Output:

* Writes `dfm_mixture_results.csv` with columns:

  * `as_of`, `target_quarter`, `mixture_forecast`, `n_models_used`, `n_failed`, `n_trimmed`

---

## 5) Interpreting the logs

A typical line looks like:

> `mixture update as_of=2010-01-31 target=2009Q4 used=8/12 failed=4 trimmed=0 mix=...`

Meaning:

* At `as_of=2010-01-31`, the pipeline targeted quarter `2009Q4`
* 8 out of 12 models were usable after failure handling
* 4 models failed (or were treated as failures due to non-convergence)
* `mix` is the average of the usable (and possibly trimmed) forecasts

---

## 6) Recommended extensions (next steps)

* **Longer as-of schedule**: run from your first feasible real-time start until end of sample (monthly month-end grid is already implemented).
* **Better target bounds**: use quantile bounds instead of min/max if you see extreme outliers drive clipping. 
* **News decomposition** (optional): if you later want to attribute forecast changes to releases, see the “news” framework in Gianno([SSRN][2])

---

## 7) Key references (academic + institutional)

* CPB: *Nowcasting GDP growth* (CPB model description and motivation). ([cpb.nl][3])
* Giannone, Reichlin & Small (2008): real-time data flow and nowcasting with DFMs. ([ScienceDirect][1])
* Bańbura & Modugno (2014): ML/EM estimation for DFMs with arbitrary missing datadged-edge patterns. ([Wiley Online Library][4])
* Mariano & Murasawa (2003): mixed-frequency factor approach linking quarterly GDP and monthly latent activity. ([JSTOR][5])

---

## 8) File map (quick)

* `DFM_Model.py`

  * defines mixture grid, as-of schedule, model fit settings, and writes results.
* `DFM_helpers.py`

  * vintage construction (`make_vintage`), lags parsing (`load_release_lags_csv`), target quarter logic, mixture aggregation, fallback policy.

If you want, I can also generate a **repo-style folder structure** section (data/outputs/scripts), plus a short “How we construct release lags” paragraph that matches your CPB Section 6–6.2 write-up style—but this README already matches what the code currently does.
