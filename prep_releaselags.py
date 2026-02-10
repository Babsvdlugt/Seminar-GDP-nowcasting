import pandas as pd

IN_XLSX = "release_lags.xlsx"
OUT_CSV = "release_lags_clean.csv"

df = pd.read_excel(IN_XLSX)

# 1) map kolomnamen naar wat load_release_lags_csv verwacht
df = df.rename(columns={
    "Data_seminar": "series",
    "Lags_days": "lag_days",
    "Lags_weeks": "lag_weeks",
    "Lags_months": "lag_months",
})

# 2) hou alleen relevante kolommen (extra mag, maar dit is clean)
keep = ["series", "lag_days", "lag_weeks", "lag_months"]
df = df[keep].copy()

# 3) strip series names + drop lege
df["series"] = df["series"].astype(str).str.strip()
df = df[df["series"].ne("") & df["series"].ne("nan")]

# 4) OPTIONAL: zet freq/ref_point defaults (handig voor GDP)
df["freq"] = "M"
df["ref_point"] = "period_end"

# 5) GDP is quarterly + CPB gebruikt ~45 dagen (flash). Zet dit hard als je wilt.
# Pas de seriesnaam aan aan jouw kolomnaam in df (bij jou: GrossDomesticProduct_1)
gdp_name = "GrossDomesticProduct_1"
mask = df["series"].eq(gdp_name)
if mask.any():
    df.loc[mask, "freq"] = "Q"
    df.loc[mask, "ref_point"] = "period_end"
    df.loc[mask, "lag_days"] = 45
    df.loc[mask, "lag_weeks"] = pd.NA
    df.loc[mask, "lag_months"] = pd.NA

df.to_csv(OUT_CSV, index=False)
print(f"Saved {OUT_CSV} with {len(df)} rows")
