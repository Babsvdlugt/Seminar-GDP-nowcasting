# dfm_cpb.py
# ------------------------------------------------------------
# CPB-style mixture DFM nowcasting (mixture of 12 DFMs):
#   r = 2..5 factors, p = 1..3 VAR lags on factors  => 12 models
#
# Core ideas:
# - Monthly panel (incl. GDP). GDP is treated as a monthly series with missing values except in
#   the “GDP quarter months” detected from your dataset (e.g., [1,4,7,10] or [3,6,9,12]).
# - Ragged edge via RELEASE_LAGS per series (quasi real-time vintage by as_of month).
# - DFM in state-space:
#     y_t = mu + Lambda f_t + eps_t   (eps diag)
#     f_t follows VAR(p) in companion form
# - Estimation via EM:
#     E-step: Kalman filter + RTS smoother (handles missing)
#     M-step: update mu, Lambda, diag(eps var), VAR(p) coefficients and factor shock variances
# - Mixture forecast: equal-weight average across all 12 DFMs
# - “Extreme forecast correction”: clip mixture component forecasts to [min observed GDP, max observed GDP]
#
# INPUT:
#   data_transformations_DFM_ready_state_space.csv
# OUTPUT:
#   cpb_mixture_dfm_outputs.csv
#
# Notes:
# - This is a practical implementation (stable restrictions: diagonal measurement noise; factor shock
#   only in current-factor block; VAR(p) estimated by OLS on smoothed factors).
# - It prints progress as it runs.
# ------------------------------------------------------------

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA

# -------------------------------
# User settings
# -------------------------------
DATA_PATH = "data_transformations_DFM_ready_state_space.csv"
OUT_PATH = "cpb_mixture_dfm_outputs.csv"

DATE_COL = "date"
GDP_COL = "GrossDomesticProduct_1"

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

# EM settings
EM_MAX_ITERS = 40
EM_TOL = 1e-4

# Kalman numeric stability
JITTER = 1e-7

# training settings
MIN_TRAIN_MONTHS = 60
STANDARDIZE_ON_VINTAGE = True  # standardize per vintage on available history

# -------------------------------
# Time helpers
# -------------------------------
def to_month_start(x: pd.Timestamp) -> pd.Timestamp:
    return pd.Timestamp(x.year, x.month, 1)

def shift_months(dt: pd.Timestamp, k: int) -> pd.Timestamp:
    return (dt - pd.DateOffset(months=int(k))).normalize().replace(day=1)

# -------------------------------
# GDP quarter-month detection
# -------------------------------
def detect_gdp_quarter_months(df: pd.DataFrame) -> list[int]:
    """
    Detect which four months of the year GDP is observed (quarterly release months),
    by inferring the most common phase modulo 3 from observed GDP dates.
    Returns e.g. [1,4,7,10] or [3,6,9,12].
    """
    g = df.loc[df[GDP_COL].notna(), DATE_COL]
    if g.empty:
        return []
    months = g.dt.month.to_numpy()
    phase = months % 3
    phase_star = int(pd.Series(phase).value_counts().idxmax())
    quarter_months = [m for m in range(1, 13) if (m % 3) == phase_star]
    return sorted(quarter_months)

def enforce_gdp_quarterly_missing(df: pd.DataFrame, quarter_months: list[int]) -> pd.DataFrame:
    """
    Keep GDP only in the detected quarterly months; set all other months to NaN.
    """
    out = df.copy()
    if not quarter_months:
        return out
    keep = out[DATE_COL].dt.month.isin(quarter_months)
    out.loc[~keep, GDP_COL] = np.nan
    return out

def quarter_month_for_asof(as_of: pd.Timestamp, quarter_months: list[int]) -> pd.Timestamp:
    """
    Map an as_of month to the current 'quarter GDP month' based on the detected quarter_months.
    We use the next quarter month >= as_of.month within same year, else wrap to next year.
    """
    m = as_of.month
    qms = sorted(quarter_months)
    for qm in qms:
        if qm >= m:
            return pd.Timestamp(as_of.year, qm, 1)
    return pd.Timestamp(as_of.year + 1, qms[0], 1)

def prev_quarter_month(as_of: pd.Timestamp, quarter_months: list[int]) -> pd.Timestamp:
    """
    Previous quarter GDP month relative to as_of.
    """
    cur = quarter_month_for_asof(as_of, quarter_months)
    return shift_months(cur, 3)

# -------------------------------
# Vintage builder (ragged edge)
# -------------------------------
def build_vintage(df: pd.DataFrame, as_of: pd.Timestamp, cols: list[str]) -> pd.DataFrame:
    """
    Quasi real-time vintage at 'as_of':
    For each column c, values after (as_of - RELEASE_LAGS[c]) are not available -> NaN.
    """
    out = df[[DATE_COL] + cols].copy()
    out = out[out[DATE_COL] <= as_of].copy()

    for c in cols:
        lag = int(RELEASE_LAGS.get(c, 0))
        cutoff = shift_months(as_of, lag)
        out.loc[out[DATE_COL] > cutoff, c] = np.nan

    return out

# -------------------------------
# Companion form for VAR(p)
# -------------------------------
def build_companion(A_list: list[np.ndarray]) -> np.ndarray:
    """
    A_list: [A1,...,Ap], each (r,r)
    Companion transition T has shape (r*p, r*p)
    """
    p = len(A_list)
    r = A_list[0].shape[0]
    top = np.hstack(A_list)  # (r, r*p)

    if p == 1:
        return top

    lower = np.hstack([np.eye(r * (p - 1)), np.zeros((r * (p - 1), r))])
    T = np.vstack([top, lower])
    return T

# -------------------------------
# Kalman filter + RTS smoother with missing observations
# -------------------------------
def kalman_filter_smoother_missing(
    Y: np.ndarray,         # (T, N) may contain NaN
    Z_full: np.ndarray,    # (N, m)
    Tmat: np.ndarray,      # (m, m)
    Q: np.ndarray,         # (m, m)
    R_diag: np.ndarray,    # (N,)
    a0: np.ndarray,
    P0: np.ndarray,
    jitter: float = JITTER,
):
    Tn, N = Y.shape
    m = Tmat.shape[0]

    a_pred = np.zeros((Tn, m))
    P_pred = np.zeros((Tn, m, m))
    a_filt = np.zeros((Tn, m))
    P_filt = np.zeros((Tn, m, m))

    loglik = 0.0

    a_prev = a0.copy()
    P_prev = P0.copy()

    # Filter
    for t in range(Tn):
        a_t_pred = Tmat @ a_prev
        P_t_pred = Tmat @ P_prev @ Tmat.T + Q

        y_t = Y[t, :]
        obs = ~np.isnan(y_t)
        idx = np.where(obs)[0]

        if idx.size == 0:
            a_t_f = a_t_pred
            P_t_f = P_t_pred
        else:
            Z = Z_full[idx, :]  # (m_obs, m)
            y = y_t[idx]
            R = np.diag(R_diag[idx]) + np.eye(idx.size) * jitter

            v = y - (Z @ a_t_pred)
            F = Z @ P_t_pred @ Z.T + R

            Finv = np.linalg.inv(F)
            K = P_t_pred @ Z.T @ Finv

            a_t_f = a_t_pred + K @ v
            P_t_f = P_t_pred - K @ Z @ P_t_pred

            sign, logdet = np.linalg.slogdet(F)
            if sign <= 0:
                logdet = np.log(np.maximum(np.linalg.det(F), 1e-12))
            loglik += -0.5 * (idx.size * np.log(2 * np.pi) + logdet + v.T @ Finv @ v)

        a_pred[t], P_pred[t] = a_t_pred, P_t_pred
        a_filt[t], P_filt[t] = a_t_f, P_t_f
        a_prev, P_prev = a_t_f, P_t_f

    # RTS smoother
    a_smooth = np.zeros_like(a_filt)
    P_smooth = np.zeros_like(P_filt)

    a_smooth[-1] = a_filt[-1]
    P_smooth[-1] = P_filt[-1]

    for t in range(Tn - 2, -1, -1):
        P_t = P_filt[t]
        P_tp1_pred = P_pred[t + 1]
        J = P_t @ Tmat.T @ np.linalg.inv(P_tp1_pred)

        a_smooth[t] = a_filt[t] + J @ (a_smooth[t + 1] - a_pred[t + 1])
        P_smooth[t] = P_t + J @ (P_smooth[t + 1] - P_tp1_pred) @ J.T

    return {
        "a_filt": a_filt,
        "P_filt": P_filt,
        "a_smooth": a_smooth,
        "P_smooth": P_smooth,
        "loglik": float(loglik),
    }

# -------------------------------
# EM for DFM with VAR(p) factors
# -------------------------------
def em_dfm_varp(
    Y: np.ndarray,     # (T, N), standardized panel, NaN allowed
    r: int,
    p: int,
    max_iters: int = EM_MAX_ITERS,
    tol: float = EM_TOL,
    verbose: bool = True,
    seed: int = 0,
):
    """
    State is companion: a_t = [f_t, f_{t-1},...,f_{t-p+1}]', dim m=r*p
    Measurement: y_t = mu + [Lambda, 0,...,0] a_t + eps_t, eps diag
    """
    Tn, N = Y.shape
    m = r * p

    # Init: PCA on filled Y for factors
    Y0 = Y.copy()
    Y0[np.isnan(Y0)] = 0.0

    if verbose:
        print(f"      [init] PCA for r={r}")

    pca = PCA(n_components=r, random_state=seed)
    f_hat = pca.fit_transform(Y0)      # (T, r)
    Lambda = pca.components_.T         # (N, r)

    # init mu from available mean
    mu = np.nanmean(Y, axis=0)         # (N,)
    Yc = Y - mu.reshape(1, -1)

    # init VAR(p) via OLS on PCA factors
    if verbose:
        print(f"      [init] VAR({p}) on PCA factors")
    A_list: list[np.ndarray] = []
    if Tn > (p + 5):
        Yreg = f_hat[p:, :]
        Xreg = []
        for lag in range(1, p + 1):
            Xreg.append(f_hat[p - lag : Tn - lag, :])
        Xreg = np.hstack(Xreg)  # (T-p, r*p)

        A_stack, *_ = np.linalg.lstsq(Xreg, Yreg, rcond=None)  # (r*p, r)
        A_stack = A_stack.T  # (r, r*p)
        A_list = [A_stack[:, j * r : (j + 1) * r] for j in range(p)]
    else:
        A_list = [0.2 * np.eye(r) for _ in range(p)]

    # Q: only first r dims get shocks; rest are near-deterministic shifts
    Q = np.zeros((m, m))
    Q[:r, :r] = np.eye(r) * 0.5
    Q += np.eye(m) * 1e-9

    # Measurement noise diag from residuals
    resid0 = Yc - (f_hat @ Lambda.T)
    sigma_eps = np.nanvar(resid0, axis=0)
    sigma_eps = np.where(np.isfinite(sigma_eps) & (sigma_eps > 1e-6), sigma_eps, 1.0)

    # Z_full: (N, m) = [Lambda, 0,...,0]
    Z_full = np.zeros((N, m))
    Z_full[:, :r] = Lambda

    a0 = np.zeros(m)
    P0 = np.eye(m) * 10.0

    prev_ll = -np.inf

    for it in range(1, max_iters + 1):
        if verbose:
            print(f"      [EM] iter {it:02d}/{max_iters} (r={r}, p={p})")

        # E-step
        Tmat = build_companion(A_list)
        res = kalman_filter_smoother_missing(Yc, Z_full, Tmat, Q, sigma_eps, a0=a0, P0=P0)
        ll = res["loglik"]
        aS = res["a_smooth"]  # (T, m)
        fS = aS[:, :r]

        # M-step: update mu, Lambda, sigma_eps (row-by-row OLS handling missing)
        if verbose:
            print("        [M] update mu, Lambda, sigma_eps (diag)")

        mu_new = np.zeros(N)
        Lambda_new = np.zeros((N, r))
        sigma_new = np.zeros(N)

        for i in range(N):
            yi = Y[:, i]
            obs = ~np.isnan(yi)
            if obs.sum() < max(10, r + 2):
                mu_new[i] = mu[i]
                Lambda_new[i, :] = Lambda[i, :]
                sigma_new[i] = sigma_eps[i]
                continue

            Fi = fS[obs, :]
            yi_obs = yi[obs]

            X = np.hstack([np.ones((Fi.shape[0], 1)), Fi])
            b, *_ = np.linalg.lstsq(X, yi_obs, rcond=None)
            mu_new[i] = b[0]
            Lambda_new[i, :] = b[1:]

            e = yi_obs - X @ b
            sigma_new[i] = float(np.mean(e**2) + 1e-6)

        mu = mu_new
        Lambda = Lambda_new
        sigma_eps = sigma_new
        Yc = Y - mu.reshape(1, -1)

        # Update VAR(p) on smoothed factors by OLS
        if verbose:
            print("        [M] update VAR(p) on factors")

        if Tn > (p + 5):
            Yreg = fS[p:, :]
            Xreg = []
            for lag in range(1, p + 1):
                Xreg.append(fS[p - lag : Tn - lag, :])
            Xreg = np.hstack(Xreg)  # (T-p, r*p)

            A_stack, *_ = np.linalg.lstsq(Xreg, Yreg, rcond=None)  # (r*p, r)
            A_stack = A_stack.T
            A_list = [A_stack[:, j * r : (j + 1) * r] for j in range(p)]

            U = Yreg - Xreg @ A_stack.T  # (T-p, r)
            Q[:r, :r] = np.diag(np.maximum(np.mean(U**2, axis=0), 1e-6))

        # refresh Z_full with new Lambda
        Z_full[:, :r] = Lambda

        if verbose:
            rel = abs(ll - prev_ll) / (1.0 + abs(prev_ll)) if np.isfinite(prev_ll) else np.inf
            print(f"        [LL] {ll:.2f} | rel_change={rel:.3e}")

        if np.isfinite(prev_ll):
            if abs(ll - prev_ll) / (1.0 + abs(prev_ll)) < tol:
                if verbose:
                    print("        [conv] converged")
                prev_ll = ll
                break

        prev_ll = ll

    # final smoother
    Tmat = build_companion(A_list)
    Z_full[:, :r] = Lambda
    res = kalman_filter_smoother_missing(Yc, Z_full, Tmat, Q, sigma_eps, a0=a0, P0=P0)

    return {
        "mu": mu,
        "Lambda": Lambda,
        "sigma_eps": sigma_eps,
        "A_list": A_list,
        "Q": Q,
        "Tmat": Tmat,
        "Z_full": Z_full,
        "smooth_state": res["a_smooth"],  # (T, r*p)
        "loglik": res["loglik"],
        "r": r,
        "p": p,
    }

# -------------------------------
# Main mixture runner
# -------------------------------
def run_cpb_mixture(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df[DATE_COL] = pd.to_datetime(df[DATE_COL]).map(to_month_start)
    df = df.sort_values(DATE_COL).reset_index(drop=True)

    # Detect GDP quarter months from the data
    quarter_months = detect_gdp_quarter_months(df)
    print(f"[info] Detected GDP quarter months: {quarter_months}")
    if not quarter_months:
        raise ValueError("Could not detect GDP quarter months (no non-missing GDP in the dataset).")

    # Enforce GDP missing outside quarter months
    df = enforce_gdp_quarterly_missing(df, quarter_months=quarter_months)

    # Panel columns include GDP
    panel_cols = [c for c in df.columns if c != DATE_COL]
    if GDP_COL not in panel_cols:
        raise ValueError(f"GDP column '{GDP_COL}' not in dataset columns.")

    gdp_obs = df[GDP_COL].dropna()
    if gdp_obs.empty:
        raise ValueError("GDP column has no observed values after quarterly-missing enforcement.")
    gdp_min_global = float(gdp_obs.min())
    gdp_max_global = float(gdp_obs.max())

    print("============================================================")
    print("[CPB Mixture DFM] Start")
    print(f"Rows={len(df)}, Vars={len(panel_cols)} (incl. GDP)")
    print(f"GDP observed range: [{gdp_min_global:.3f}, {gdp_max_global:.3f}]")
    print("Mixture: r=2..5 and p=1..3 (12 DFMs), equal weights")
    print("============================================================")

    dates_all = df[DATE_COL].tolist()
    results: list[dict] = []

    # Iterate as-of months
    for t_idx, as_of in enumerate(dates_all):
        if (t_idx + 1) < MIN_TRAIN_MONTHS:
            continue

        print(f"\n---------------- AS-OF {as_of.date()} (t_idx={t_idx}) ----------------")

        # Build vintage with ragged edge
        vint = build_vintage(df, as_of=as_of, cols=panel_cols).sort_values(DATE_COL).reset_index(drop=True)

        # Standardize per vintage (columnwise), keeping NaN
        Y = vint[panel_cols].to_numpy(dtype=float)  # (T_v, N)
        if STANDARDIZE_ON_VINTAGE:
            mu_col = np.nanmean(Y, axis=0)
            sd_col = np.nanstd(Y, axis=0, ddof=0)
            sd_col = np.where(sd_col == 0.0, 1.0, sd_col)
            Yz = (Y - mu_col.reshape(1, -1)) / sd_col.reshape(1, -1)
        else:
            mu_col = np.zeros(Y.shape[1], dtype=float)
            sd_col = np.ones(Y.shape[1], dtype=float)
            Yz = Y

        # Identify current/previous quarter GDP months for this as_of
        q_now = quarter_month_for_asof(as_of, quarter_months)
        q_back = prev_quarter_month(as_of, quarter_months)

        dates_v = vint[DATE_COL].tolist()

        def safe_index(d: pd.Timestamp) -> int | None:
            try:
                return dates_v.index(d)
            except ValueError:
                return None

        idx_now = safe_index(q_now)
        idx_back = safe_index(q_back)

        if idx_now is None and idx_back is None:
            print("  [skip] Neither current nor previous quarter month is in the vintage.")
            continue

        gdp_pos = panel_cols.index(GDP_COL)

        # Mixture components
        comps_now: list[float] = []
        comps_back: list[float] = []

        for r in range(2, 6):
            for p in range(1, 4):
                print(f"  [DFM] Fit (r={r}, p={p})")
                params = em_dfm_varp(Yz, r=r, p=p, verbose=False, seed=0)

                Z_full = params["Z_full"]
                state_smooth = params["smooth_state"]  # (T_v, r*p)
                Z_gdp = Z_full[gdp_pos, :]            # (r*p,)

                def pred_at_idx(idx: int) -> float:
                    a_idx = state_smooth[idx, :]
                    y_z_hat = float(Z_gdp @ a_idx)  # standardized
                    y_hat = float(y_z_hat * sd_col[gdp_pos] + mu_col[gdp_pos])
                    return y_hat

                y_now = pred_at_idx(idx_now) if idx_now is not None else None
                y_back = pred_at_idx(idx_back) if idx_back is not None else None

                # Extreme forecast correction: clip component forecasts to global GDP range
                if y_now is not None:
                    y_now = float(np.clip(y_now, gdp_min_global, gdp_max_global))
                    comps_now.append(y_now)
                if y_back is not None:
                    y_back = float(np.clip(y_back, gdp_min_global, gdp_max_global))
                    comps_back.append(y_back)

                print(
                    f"    [done] (r={r}, p={p}) "
                    f"now={y_now if y_now is not None else np.nan:.4f}, "
                    f"back={y_back if y_back is not None else np.nan:.4f}, "
                    f"LL={params['loglik']:.2f}"
                )

        mix_now = float(np.mean(comps_now)) if comps_now else np.nan
        mix_back = float(np.mean(comps_back)) if comps_back else np.nan

        # True GDP at those quarter months (for evaluation only; not conditioning)
        true_now = df.loc[df[DATE_COL] == q_now, GDP_COL]
        true_back = df.loc[df[DATE_COL] == q_back, GDP_COL]
        y_true_now = float(true_now.iloc[0]) if len(true_now) and pd.notna(true_now.iloc[0]) else np.nan
        y_true_back = float(true_back.iloc[0]) if len(true_back) and pd.notna(true_back.iloc[0]) else np.nan

        print("  [Mixture] equal-weight mean over 12 DFMs")
        print(f"    q_now={q_now.date()}   mix_now={mix_now:.4f} | true={y_true_now if np.isfinite(y_true_now) else np.nan}")
        print(f"    q_back={q_back.date()}  mix_back={mix_back:.4f} | true={y_true_back if np.isfinite(y_true_back) else np.nan}")
        print(f"    components used: now={len(comps_now)} back={len(comps_back)}")

        results.append(
            {
                "as_of": as_of,
                "q_now": q_now,
                "q_back": q_back,
                "mix_nowcast": mix_now,
                "mix_backcast": mix_back,
                "true_now": y_true_now,
                "true_back": y_true_back,
                "n_components_now": len(comps_now),
                "n_components_back": len(comps_back),
            }
        )

    out = pd.DataFrame(results).sort_values("as_of").reset_index(drop=True)

    def mse(y_true: np.ndarray, y_hat: np.ndarray) -> float:
        m = np.isfinite(y_true) & np.isfinite(y_hat)
        return float(np.mean((y_true[m] - y_hat[m]) ** 2)) if m.sum() else np.nan

    mse_now = mse(out["true_now"].to_numpy(), out["mix_nowcast"].to_numpy()) if not out.empty else np.nan
    mse_back = mse(out["true_back"].to_numpy(), out["mix_backcast"].to_numpy()) if not out.empty else np.nan

    print("\n============================================================")
    print("[CPB Mixture DFM] Finished")
    print(f"MSE nowcast (mixture):  {mse_now:.6f}")
    print(f"MSE backcast (mixture): {mse_back:.6f}")
    print(f"Saved -> {OUT_PATH}")
    print("============================================================\n")

    return out

# -------------------------------
# Entrypoint
# -------------------------------
if __name__ == "__main__":
    df = pd.read_csv(DATA_PATH)
    out = run_cpb_mixture(df)
    out.to_csv(OUT_PATH, index=False)

