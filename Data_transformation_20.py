import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt

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

def adf_table(df_in: pd.DataFrame, alpha: float = 0.05, label: str = "") -> pd.DataFrame:
    rows = []
    for c in df_in.columns:
        s = df_in[c]
        p = adf_pvalue(s)
        rows.append({
            "variable": c,
            f"pvalue_{label}": p,
            f"stationary_{label}": is_stationary(p, alpha=alpha)
        })
    return pd.DataFrame(rows).sort_values("variable")




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

# ---------- ADF test + beslissing ----------
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

    # Beslissingsregel: alleen op basis van 'c'
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

# Opslaan beslissingen
DECISIONS_PATH = OUT_PATH.replace(".csv", "_ADF_decisions.csv")
decisions.to_csv(DECISIONS_PATH, index=False)
print("\nBeslissingstabel opgeslagen:", DECISIONS_PATH)

# ---------- Pas transformaties toe ----------
df_tr = pd.DataFrame(index=df.index)

# GDP: quarterly log-diff — robuuste mapping via periods
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

# Verwijder rijen waar ALLE kolommen NaN zijn na transformatie
df_tr = df_tr.dropna(how="all")

# ---------- Tweede ronde: check of nog non-stationair, dan second diff ----------
print("\nTweede ronde ADF-check na eerste transformatie...")

# ADF op de getransformeerde data
adf_after_first = adf_table(df_tr, alpha=alpha, label="after_first")

# Filter non-stationaire variabelen, maar exclude GDP
still_nonstat_vars = adf_after_first[
    (adf_after_first["stationary_after_first"] == False) &
    (adf_after_first["variable"] != GDP_COL)
    ]["variable"].tolist()

print(f"Variabelen die nog non-stationair zijn na eerste transformatie (excl. GDP): {still_nonstat_vars}")

# Pas second diff toe op de nog non-stationaire variabelen (zonder GDP)
for c in still_nonstat_vars:
    if c not in df_tr.columns:
        continue

    x = df_tr[c]

    # Second diff
    second_tr = x.diff().dropna()

    # Update df_tr
    df_tr[c] = pd.Series(np.nan, index=df_tr.index)
    df_tr.loc[second_tr.index, c] = second_tr

    print(f"[SECOND DIFF] Toegepast op {c} → nu hopelijk stationair")

# Optioneel: verwijder weer één extra rij (door de tweede diff)
# (nu totaal 2 rijen verwijderd: 1 van first diff, 1 van second diff)
df_tr = df_tr.iloc[1:].copy()

# Her-test alleen op de aangepaste variabelen (excl. GDP)
if still_nonstat_vars:
    print("\nADF na second diff (alleen op aangepaste variabelen):")
    adf_after_second = adf_table(df_tr[still_nonstat_vars], alpha=alpha, label="after_second")
    print(adf_after_second.to_string(index=False))
else:
    print("\nGeen variabelen meer die second diff nodig hadden (excl. GDP).")

# ---------- State-space output ----------
df_ss = df_tr.apply(zscore_ignore_na, axis=0)

OUT_PATH_SS = OUT_PATH.replace(".csv", "_DFM_ready_state_space.csv")
df_ss.reset_index().to_csv(OUT_PATH_SS, index=False)

# def plot_raw_series(df, var):
#     plt.figure(figsize=(12,4))
#     plt.plot(df.index, df[var])
#     plt.title(f"Raw series: {var}")
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()
#
# plot_raw_series(df, "^AEX")


def plot_before_after(df_raw, df_trans, var):
    fig, axs = plt.subplots(2, 1, figsize=(12,6), sharex=True)

    axs[0].plot(df_raw.index, df_raw[var])
    axs[0].set_title(f"{var} – raw")

    axs[1].plot(df_trans.index, df_trans[var])
    axs[1].set_title(f"{var} – transformed")

    for ax in axs:
        ax.grid(True)

    plt.tight_layout()
    plt.show()

plot_before_after(df, df_tr, "^AEX")

gdp_tr = df["GrossDomesticProduct_1"]

plt.figure(figsize=(12,4))
plt.plot(gdp_tr.dropna().index, gdp_tr.dropna(), marker="o")
plt.title("GDP (quarterly observations only)")
plt.ylabel("GDP")
plt.grid(True)
plt.tight_layout()
plt.show()


# def plot_seasonality_check(df, var):
#     plt.figure(figsize=(12,4))
#     plt.plot(df.index, df[var])
#     plt.title(f"Seasonality check: {var}")
#     plt.grid(True)
#     plt.show()
#
# plot_seasonality_check(df, "Bankruptcies")


