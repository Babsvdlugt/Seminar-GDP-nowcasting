import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller

# ---------- Laad de data ----------
df = pd.read_csv("top20_monthly_grid_from2005.csv")
df["date"] = pd.to_datetime(df["date"])
df = df.sort_values("date").set_index("date")

OUT_PATH = "data_transformations.csv"

GDP_COL = "GrossDomesticProduct_1"
variables = list(df.columns)  # date is index, dus alle kolommen zijn variabelen

# Zorg dat alles numeriek is
for c in variables:
    df[c] = pd.to_numeric(df[c], errors="coerce")

# ---------- Helper functies ----------
def safe_log(series: pd.Series) -> pd.Series:
    """Log transform als strikt positief. Zeros → log1p. Negatieven → all-NaN."""
    s = pd.to_numeric(series, errors="coerce")
    if (s.dropna() < 0).any():
        return pd.Series(index=s.index, dtype=float)
    if (s.dropna() == 0).any():
        return np.log1p(s)
    return np.log(s)

def log_diff(series: pd.Series) -> pd.Series:
    """Δ log(x_t)"""
    return safe_log(series).diff()

def diff(series: pd.Series) -> pd.Series:
    """Δ x_t"""
    return pd.to_numeric(series, errors="coerce").diff()

def zscore_ignore_na(series: pd.Series) -> pd.Series:
    """(x - mean)/std met skipna=True"""
    s = pd.to_numeric(series, errors="coerce")
    mu = s.mean(skipna=True)
    sd = s.std(skipna=True, ddof=0)
    if sd == 0 or np.isnan(sd):
        return s * np.nan
    return (s - mu) / sd

# ---------- ADF helper ----------
def adf_pvalue(series: pd.Series) -> float:
    """Alleen regression='c', retourneert p-value of np.nan bij probleem"""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.shape[0] < 25 or s.nunique() <= 2:
        return np.nan
    try:
        return adfuller(s, regression="c", autolag="AIC")[1]
    except Exception:
        return np.nan

def is_stationary(pval: float, alpha: float = 0.05) -> bool:
    return (not np.isnan(pval)) and (pval <= alpha)

# ---------- Uitzonderingen per type variabele ----------
RATE_LIKE = {
    "ecb_3M_Yield",
    "ecb_10Y_Yield",
    "Employment_allGenders_15 to 74 years_SeasonallyAdjusted_8_UnemplyRate",
}

ALREADY_CHANGE = {
    "Domestic_consumption_by_households_VolumeChangesShoppingdayAdjusted_3",
    "MaandmutatieCPI_3",
}

# ---------- ADF test + beslissing (GDP doen we apart) ----------
alpha = 0.05
decision_rows = []
transform_map = {}

for c in variables:
    if c == GDP_COL:
        continue  # GDP behandelen we apart (quarterly log-diff)
    if c not in df.columns:
        continue

    s = df[c]
    p_c = adf_pvalue(s)
    stat_c = is_stationary(p_c, alpha=alpha)

    # Beslissingsregel: ALLEEN op basis van 'c'
    if c in ALREADY_CHANGE:
        rule = "level (already change series)"
    elif stat_c:
        rule = "level (ADF c stationary)"
    else:
        if c in RATE_LIKE:
            rule = "diff (rate-like, ADF c nonstationary)"
        else:
            rule = "log-diff (ADF c nonstationary)"

    transform_map[c] = {"rule": rule, "p_c": p_c, "stat_c": stat_c}

    decision_rows.append(
        {"variable": c, "pvalue_c": p_c, "stationary_c": stat_c, "chosen_rule": rule}
    )

# Voeg GDP ook toe aan de beslissings-tabel (transformatie vast)
decision_rows.append(
    {"variable": GDP_COL, "pvalue_c": np.nan, "stationary_c": np.nan, "chosen_rule": "quarterly log-diff (special)"}
)

decisions = pd.DataFrame(decision_rows).sort_values("variable")
print("\nADF beslissingen (alpha=0.05, alleen 'c'):")
print(decisions.to_string(index=False))

# Opslaan beslissingen (handig voor verslag)
DECISIONS_PATH = OUT_PATH.replace(".csv", "_ADF_decisions.csv")
decisions.to_csv(DECISIONS_PATH, index=False)
print("\nBeslissingstabel opgeslagen:", DECISIONS_PATH)

# ---------- Pas transformaties toe ----------
df_tr = pd.DataFrame(index=df.index)

# (A) GDP: quarterly log-diff (Option A)
# (A) GDP: quarterly log-diff (Option A) — ROBUUSTE MAPPING VIA PERIODS
if GDP_COL in df.columns:
    # originele kwartaalobservaties (alleen daar waar GDP bestaat)
    gdp_raw = df[GDP_COL].dropna()

    # naar kwartaalreeks (1 waarde per kwartaal)
    gdp_q = gdp_raw.copy()
    gdp_q.index = gdp_q.index.to_period("Q")
    gdp_q = gdp_q.groupby(level=0).last()

    # QoQ log-diff (kwartaalgroei)
    gdp_qoq = np.log(gdp_q).diff()

    # maak maandserie met NaN
    gdp_month = pd.Series(index=df.index, dtype=float)

    # zet alleen waarden op de datums waar GDP in het origineel aanwezig was
    # (en pak per datum het bijbehorende kwartaal)
    for dt in gdp_raw.index:
        q = dt.to_period("Q")
        gdp_month.loc[dt] = gdp_qoq.get(q, np.nan)

    df_tr[GDP_COL] = gdp_month

    # - of alleen op kwartaalmomenten (laat de rest NaN):
    # df_tr[GDP_COL] = gdp_qoq.reindex(df.index)

# overige variabelen
for c in variables:
    if c == GDP_COL:
        continue
    if c not in df.columns:
        continue

    rule = transform_map[c]["rule"]
    x = df[c]

    if rule.startswith("level"):
        df_tr[c] = pd.to_numeric(x, errors="coerce")

    elif rule.startswith("diff"):
        df_tr[c] = diff(x)

    elif rule.startswith("log-diff"):
        tr = log_diff(x)

        # Fallback als te weinig bruikbare waarden
        if tr.dropna().shape[0] < 10 and pd.to_numeric(x, errors="coerce").dropna().shape[0] > 25:
            print(f"[WARN] {c}: log-diff niet mogelijk/te weinig data → fallback naar diff()")
            tr = diff(x)

        df_tr[c] = tr

# Drop eerste rij (diff/log-diff geeft daar NaN)
df_tr = df_tr.iloc[1:].copy()

# Verwijder rijen waar ALLE kolommen NaN zijn na transformatie (optioneel)
df_tr = df_tr.dropna(how="all")

# ---------- State-space output (missing blijft; standaardiseren per kolom) ----------
df_ss = df_tr.apply(zscore_ignore_na, axis=0)

OUT_PATH_SS = OUT_PATH.replace(".csv", "_DFM_ready_state_space.csv")
df_ss.reset_index().to_csv(OUT_PATH_SS, index=False)

print("\nKlaar! State-space bestand opgeslagen:", OUT_PATH_SS)
