# Dynamic Factor Model for GDP Nowcasting

## Overview

This repository implements a **Dynamic Factor Model (DFM)** in state-space form to extract latent economic factors from a large panel of monthly macroeconomic indicators and to use these factors for **GDP nowcasting** via a bridge regression.

The model follows a standard macroeconomic nowcasting framework:

1. Extract a small number of latent common factors from high-frequency indicators using a DFM estimated via maximum likelihood and the Kalman filter.
2. Link quarterly GDP to the latent factors using a bridge regression.
3. Produce a real-time GDP nowcast based on the latest available information.

The implementation is modular, transparent, and designed to be academically and professionally defensible.

---

## Methodological Framework

### Dynamic Factor Model

Let \( X_t \in \mathbb{R}^N \) denote a vector of standardized monthly indicators. The model assumes:

\[
X_t = \Lambda f_t + e_t
\]

\[
f_t = A f_{t-1} + u_t
\]

where:
- \( f_t \in \mathbb{R}^K \) are latent common factors,
- \( \Lambda \) are factor loadings,
- \( e_t \) are idiosyncratic components,
- the factor dynamics follow an AR(\(p\)) process.

The model is estimated in **state-space form** using maximum likelihood and the Kalman filter via  
`statsmodels.tsa.statespace.DynamicFactor`.

---

### Factor Selection

The number of latent factors \(K\) is selected using **information criteria**:

- Akaike Information Criterion (AIC)
- Bayesian Information Criterion (BIC, preferred)

For \(K = 1, \dots, K_{\max}\), the model is estimated and only **converged** solutions are considered.  
The final number of factors is chosen by minimizing the selected criterion.

---

### Bridge Regression

Quarterly GDP is linked to the latent factors via a bridge regression:

\[
GDP_q = \alpha + \beta' \bar{f}_q + \varepsilon_q
\]

where:
- \( \bar{f}_q \) is the quarterly average of monthly factors,
- GDP is aligned to quarter-end timestamps.

This approach avoids imposing mixed-frequency structure inside the state-space model and follows common practice in applied macroeconomic nowcasting.

---

## Code Structure

### 1. Data Handling

#### `load_dfm_ready(path)`
- Loads a CSV file with a `date` column.
- Enforces monthly frequency (`MS`).
- Converts all columns to numeric.
- Sorts and indexes data chronologically.

#### `make_asof_df(df, asof, release_lags)`
- Constructs a **real-time (vintage) dataset**.
- Applies series-specific publication lags.
- Masks observations not yet available at the `asof` date.
- Enables realistic nowcasting with ragged edges.

---

### 2. Data Cleaning

#### `drop_near_constant_cols(X, eps)`
Removes series with near-zero variance to ensure numerical stability.

- Near-constant series do not contribute to factor identification.
- Default tolerance `eps = 1e-6` is appropriate after standardization.

#### `drop_sparse_cols(X, min_non_missing_frac)`
Drops series with insufficient data coverage.

- Balances information retention and estimation stability.
- A relatively permissive threshold is used to exploit the Kalman filter’s ability to handle missing data.

---

### 3. Standardization

All indicator series are standardized prior to estimation:

\[
X_{it}^{std} = \frac{X_{it} - \mu_i}{\sigma_i}
\]

This prevents high-variance series from dominating factor extraction and is essential for meaningful interpretation.

---

### 4. Model Estimation

#### `fit_dfm(endog, k_factors, factor_order, error_order)`
- Estimates the DFM via maximum likelihood.
- Uses AR(1) dynamics for the latent factors.
- Allows either white-noise or AR idiosyncratic errors.
- Logs convergence status and information criteria.

Only converged models are used in downstream analysis.

---

### 5. Factor Extraction

#### `extract_smoothed_factors(res, index)`
- Extracts **smoothed** latent factors from the Kalman smoother.
- Smoothed factors use the full sample and are appropriate for:
  - structural interpretation,
  - bridge regression estimation.

Note: Filtered factors would be required for strict real-time backtesting.

---

### 6. Factor Selection

#### `select_k_factors(X, k_max, factor_order, error_order, criterion)`
- Estimates DFMs for \(K = 1, \dots, K_{\max}\).
- Collects AIC, BIC, log-likelihood, and convergence status.
- Identifies the preferred number of factors based on the chosen criterion.

Only converged models with valid information criteria are considered.

---

### 7. Bridge Regression

#### `quarterly_average_factors(F)`
- Aggregates monthly factors to quarterly averages.
- Aligns factors to quarter-end timestamps.

#### `fit_bridge_regression(gdp, F)`
- Estimates an OLS regression of GDP on the latent factors.
- Supports:
  - quarterly factor aggregation (default and recommended),
  - monthly factor matching (optional).

---

### 8. Nowcasting

Two nowcasting modes are supported:

- **Quarterly nowcast (default)**  
  Uses quarterly-averaged factors for the current quarter.

- **Monthly nowcast (optional)**  
  Uses the most recent monthly factor estimate.

The nowcast is computed as the fitted value from the bridge regression.

---

## Main Execution Flow

The `main()` function implements the full pipeline:

1. Load and inspect the dataset.
2. Separate GDP from the indicator panel.
3. Clean and standardize indicators.
4. Select the number of factors using BIC.
5. Estimate the final DFM.
6. Extract and visualize latent factors.
7. Estimate the GDP bridge regression.
8. Produce a GDP nowcast.
9. Plot observed versus fitted GDP.

---

## Key Design Choices and Justifications

- **State-space DFM**: naturally handles missing data and explicit factor dynamics.
- **BIC-based factor selection**: favors parsimony and mitigates overfitting.
- **Quarterly bridge regression**: transparent handling of mixed frequencies.
- **Standardization prior to estimation**: essential for stable and interpretable factor extraction.
- **Convergence-aware model selection**: avoids reliance on unstable likelihood solutions.

---

## Dependencies

- Python 3.x
- numpy
- pandas
- matplotlib
- statsmodels

---

## Notes and Possible Extensions

- Real-time evaluation should use **filtered** rather than smoothed factors.
- Factor order and idiosyncratic error order can be selected analogously to the number of factors.
- The vintage construction (`make_asof_df`) enables full real-time nowcasting extensions.
- Rolling or expanding-window nowcast evaluation can be added for forecast performance assessment.

---

## Disclaimer

This code is intended for case studies.


