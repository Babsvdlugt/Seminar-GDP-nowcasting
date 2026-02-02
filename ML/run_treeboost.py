import pandas as pd
from pathlib import Path
from ML.ls_treeboost import LSTreeBoost


GDP_COL = "GrossDomesticProduct_1"

# 1) Load data
df = pd.read_csv(
    Path("/Users/babsvanderlugt/Seminar-GDP-nowcasting/data_transformations_DFM_ready_state_space.csv"),
    parse_dates=["date"]
).set_index("date")

# 2) Build ML dataset (voorbeeld: factors later)
X = df.drop(columns=[GDP_COL]).fillna(0.0)
y = df[GDP_COL].dropna()

X = X.loc[y.index]

# 3) Fit TreeBoost
model = LSTreeBoost(n_estimators=300, learning_rate=0.05)
model.fit(X.values, y.values)

# 4) Nowcast
latest_X = X.iloc[[-1]].values
gdp_nowcast = model.predict(latest_X)[0]
print("TreeBoost GDP nowcast:", gdp_nowcast)
