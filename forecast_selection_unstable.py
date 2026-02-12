# forecast_selection_unstable.py
from __future__ import annotations

import numpy as np
import pandas as pd


# ============================================================
# Kernels (paper gebruikt Epanechnikov in simulaties; kernel in Assumption 2)
# One-sided aan boundary: support [-1, 0]
# ============================================================
def epanechnikov(u: np.ndarray) -> np.ndarray:
    """
    Epanechnikov kernel K(u) = 0.75*(1-u^2) for |u|<=1 else 0
    """
    u = np.asarray(u, dtype=float)
    out = np.zeros_like(u)
    m = np.abs(u) <= 1.0
    out[m] = 0.75 * (1.0 - u[m] ** 2)
    return out


# ============================================================
# Helper: build lag matrix for tvAR(d): X_{t-1} = [1, dL_{t-1},...,dL_{t-d}]
# ============================================================
def build_lag_design(dL: np.ndarray, d: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Input:
      dL: array length T with loss differentials (may contain nan -> caller should clean)
      d : lag order
    Output:
      Y: (T-d,) vector of dL_t for t=d..T-1
      X: (T-d, d+1) matrix with columns [1, dL_{t-1},...,dL_{t-d}]
    """
    T = len(dL)
    if T <= d + 5:
        raise ValueError("Too few observations for chosen d")

    Y = dL[d:]
    X = np.ones((T - d, d + 1), dtype=float)
    for j in range(1, d + 1):
        X[:, j] = dL[d - j : T - j]
    return Y, X


# ============================================================
# One-sided local linear estimator at boundary u=1
# Paper eq (3)-(4) but evaluated at u=1 with one-sided kernel window [-h,0]
# We implement local linear in time only, for tv coefficients.
# ============================================================
def local_linear_tvAR_at_boundary(
    dL: np.ndarray,
    d: int = 1,
    h: float = 0.20,
    kernel=epanechnikov,
) -> tuple[np.ndarray, float, dict]:
    """
    Estimate rho(1) and sigma(1) using one-sided local linear estimation at boundary u=1.

    Model: dL_t = rho0(u) + sum_{j=1}^d rho_j(u) dL_{t-j} + xi_t
    u = t/T. We estimate at u=1 using weights K((t/T - 1)/h) with support [-1,0].

    Returns:
      rho_hat: (d+1,) array with [rho0_hat(1), rho1_hat(1),...,rhod_hat(1)]
      sigma_hat: scalar sigma_hat(1) (std dev)
      info: dict diagnostics
    """
    dL = np.asarray(dL, dtype=float).reshape(-1)
    dL = dL[np.isfinite(dL)]
    T = len(dL)
    if T < max(30, d + 10):
        raise ValueError(f"Too few observations for tvAR estimation: T={T}")

    # Build Y_t and X_{t-1}
    Y, X = build_lag_design(dL, d=d)          # length n = T-d
    n = Y.shape[0]

    # Time index for Y corresponds to original t = d..T-1 (0-based)
    # rescaled time u_t = (t+1)/T  (use 1..T convention)
    t_idx = np.arange(d + 1, T + 1)  # t = d+1,...,T in 1-based indexing for Y
    u = t_idx / T                    # in (0,1]
    u0 = 1.0

    # One-sided weights at boundary u0=1: only u <= 1 and within bandwidth
    z = (u - u0) / h                 # should be in [-1,0]
    w = kernel(z) / h
    w[(z < -1.0) | (z > 0.0)] = 0.0

    # If too few effective points, fail gracefully
    eff = np.sum(w > 0)
    if eff < max(10, 3 * (d + 1)):
        raise ValueError(f"Bandwidth too small or sample too short: effective points={eff}")

    # Local linear in time for coefficients:
    # minimize sum w_t (Y_t - X_t * [rho + rho'(u0)*(u-u0)])^2
    # Stack regressors as Z_t = kron([1, (u-u0)], X_t) -> dimension 2*(d+1)
    du = (u - u0).reshape(-1, 1)  # (n,1)
    Z = np.hstack([X, X * du])    # (n, 2*(d+1))

    # Weighted least squares: theta_hat = (Z'WZ)^{-1} Z'W Y
    W = w.reshape(-1, 1)
    ZW = Z * W
    A = Z.T @ ZW
    b = Z.T @ (Y * w)

    # Numerical stability: small ridge if needed
    ridge = 1e-10
    A = A + ridge * np.eye(A.shape[0])

    theta_hat = np.linalg.solve(A, b)

    rho_hat = theta_hat[: (d + 1)].copy()  # rho(u0)=rho(1)

    # Residuals using local-constant part only (paper Step 2 uses xi_hat_t(u) = dL_t - X_{t-1} rho_hat(u))
    resid = Y - (X @ rho_hat)

    # Local linear variance estimate at boundary (paper eq (5)-(6)):
    # regress resid^2 on [1, (u-u0)] with weights w (using its own bandwidth in paper; we reuse h for simplicity)
    F = np.hstack([np.ones((n, 1)), du])   # (n,2)
    FW = F * W
    A2 = F.T @ FW
    b2 = F.T @ (resid**2 * w)
    A2 = A2 + ridge * np.eye(A2.shape[0])
    varsigma_hat = np.linalg.solve(A2, b2)
    sigma2_hat = float(varsigma_hat[0])
    if not np.isfinite(sigma2_hat) or sigma2_hat <= 0:
        sigma2_hat = 1e-6
    sigma_hat = float(np.sqrt(sigma2_hat))

    info = {
        "T": T,
        "d": d,
        "h": h,
        "effective_points": int(eff),
        "rho_hat": rho_hat,
        "sigma_hat": sigma_hat,
    }
    return rho_hat, sigma_hat, info


# ============================================================
# Forecast conditional mean mu_{t} given info up to t-1 (one-sided)
# Paper eq (10)-(11) evaluated recursively in real time:
# mu_t = E[dL_t | A_{t-1}] ≈ X_{t-1} rho_hat(1) using sample up to t-1
# ============================================================
def one_sided_mean_selection(
    loss_diff: pd.Series,
    d: int = 1,
    h: float = 0.20,
    min_train: int = 40,
) -> pd.DataFrame:
    """
    Real-time selection between two models using Richter–Smetanina mean selection:
      select model A at time t if mu_hat_t < 0 else model B,
    where mu_hat_t is predicted conditional mean of dL_t based on dL_{<=t-1}.

    Inputs:
      loss_diff: pd.Series indexed by date, values = dL_t = L_A - L_B at each t
      d        : AR lag order in tvAR(d)
      h        : bandwidth as fraction of sample length (because u=t/T rescaling)
      min_train: minimum #observations of loss_diff needed before selecting

    Returns DataFrame indexed by date t with columns:
      mu_hat, choice  (choice in {"modelA","modelB","tie","na"})
    """
    s = loss_diff.dropna().copy()
    idx = s.index
    vals = s.to_numpy(dtype=float)

    out_mu = np.full(len(s), np.nan, dtype=float)
    out_choice = np.array(["na"] * len(s), dtype=object)

    for i in range(len(s)):
        # We choose for time t_i using info up to t_{i-1} (no look-ahead)
        if i < max(min_train, d + 5):
            continue

        hist = vals[:i]  # up to i-1
        try:
            rho_hat, _, _ = local_linear_tvAR_at_boundary(hist, d=d, h=h)
        except Exception:
            continue

        # Build X_{t-1} = [1, dL_{t-1},...,dL_{t-d}]
        if i - d < 0:
            continue
        x = np.ones(d + 1, dtype=float)
        for j in range(1, d + 1):
            x[j] = hist[-j]

        mu_hat = float(x @ rho_hat)
        out_mu[i] = mu_hat

        if np.isclose(mu_hat, 0.0, atol=1e-12):
            out_choice[i] = "tie"
        elif mu_hat < 0:
            out_choice[i] = "modelA"  # A has lower expected loss diff => A better
        else:
            out_choice[i] = "modelB"

    df = pd.DataFrame({"mu_hat": out_mu, "choice": out_choice}, index=idx)
    return df


def build_switched_forecast(
    df_eval: pd.DataFrame,
    yhat_A_col: str,
    yhat_B_col: str,
    y_true_col: str = "y_true",
    loss: str = "squared",
    d: int = 1,
    h: float = 0.20,
    min_train: int = 40,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Convenience wrapper:
      - computes loss differentials dL_t = L_A - L_B
      - runs one-sided mean selection
      - constructs y_pred_selected

    Returns:
      (df_out, df_sel)
    """
    df = df_eval.copy()

    y = df[y_true_col].astype(float)
    yA = df[yhat_A_col].astype(float)
    yB = df[yhat_B_col].astype(float)

    eA = y - yA
    eB = y - yB

    if loss == "squared":
        dL = (eA**2) - (eB**2)
    elif loss == "absolute":
        dL = eA.abs() - eB.abs()
    else:
        raise ValueError("loss must be 'squared' or 'absolute'")

    dL = dL.rename("loss_diff")

    df_sel = one_sided_mean_selection(dL, d=d, h=h, min_train=min_train)

    # Build selected forecast: modelA -> use yhat_A else yhat_B
    choice = df_sel["choice"]
    y_sel = np.where(choice == "modelA", yA.to_numpy(), np.where(choice == "modelB", yB.to_numpy(), np.nan))
    df["y_pred_selected"] = y_sel
    df["loss_diff"] = dL
    df["mu_hat"] = df_sel["mu_hat"]
    df["choice"] = df_sel["choice"]

    return df, df_sel
