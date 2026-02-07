# Seminar-GDP-nowcasting

LS-TreeBoost GDP Nowcasting (Code Guide)

Deze repo implementeert een pseudo real-time expanding-window nowcasting pipeline voor kwartaal-GDP
op basis van maandelijkse (state-space) macrodata met ragged-edge missings.

De code is bewust opgesplitst:

- main.py  -> experiment runner / backtest pipeline
             (data -> features -> expanding-window -> metrics -> output)

- ls_treeboost.py -> model + feature engineering utilities
                    (as-of rule, lagged features, LS-TreeBoost)

- DFM_Model.py -> Als je deze code wil aanslaan in de main, moet je use_dfm op true zetten, als je hem niet wil gebruiken op false.


============================================================
QUICK START
============================================================

1) Zet de dataset in de project root:
   data_transformations_DFM_ready_state_space.csv

2) Run:
   python main.py

Outputs:
- ls_treeboost_expanding_{ASOF_RULE}_lag{MAX_LAG}.csv
- ls_treeboost_expanding_summary_{ASOF_RULE}_lag{MAX_LAG}.csv


============================================================
1) MAIN.PY — WAT GEBEURT HIER?
============================================================

main.py is de driver van het experiment.
Hier staan alle timing-, backtest- en configuratiekeuzes.


----------------------------
1.1 CONFIG -> Hier staan alle keuze variabelen die je zelf moet instellen
----------------------------

DATA
- STATE_SPACE_PATH: pad naar state-space CSV
- GDP_COL: target kolom (GDP is alleen aanwezig in release-maanden -> daarbuiten NaN)

AS-OF RULE
- ASOF_RULE = "early"   (keuze: "early", "mid", "end")

Dit bepaalt welke maandinformatie beschikbaar is bij het maken van features voor een kwartaal-GDP target t.

Rule  | As-of maand | Interpretatie
----- | ----------- | ------------------------------
end   | t           | Laat in kwartaal (meeste info)
mid   | t - 1       | Halverwege kwartaal
early | t - 2       | Vroeg in kwartaal (minste info)

Waarom is early gekozen?
- Meest realistische early-warning scenario
- Beste / meest stabiele out-of-sample prestaties in tests
- Minimaliseert risico op look-ahead bias


LAGLENGTE
- MAX_LAG = 12

Betekenis:
- Per macro-feature maak je lag0 t/m lag12.
- 12 is gekozen via gridsearch als beste bias–variance trade-off (te klein -> infoverlies, te groot -> ruis/overfit).


START EXPANDING-WINDOW
- MIN_TRAIN_OBS = 25

Betekenis:
- Eerste voorspelling pas na 25 historische GDP-observaties.
- Reden: boosting met trees wordt instabiel bij te kleine sample.
- Beste getest: 25.


EARLY STOPPING (BINNEN ELKE FIT)
- VAL_FRAC_INNER = 0.2
- EARLY_STOPPING_ROUNDS = 25

Binnen elke expanding-window fit:
- Chronologische train/validation split (laatste 20% = validation)
- Stop als validation MSE 25 iteraties niet verbetert

Dit maakt n_estimators een bovengrens in plaats van een harde keuze.


MODEL PARAMETERS (WORDEN DOORGEEVEN AAN LS_treeboost)
MODEL_PARAMS = {
  n_estimators       : 3000,
  learning_rate      : 0.02,
  max_depth          : 2,
  min_samples_leaf   : 10,
  min_samples_split  : 20,
  max_features       : "sqrt",
  subsample          : 0.6,
  random_state       : 42,
}

Intuïtie:
- max_depth laag + leaf/split constraints -> sterke regularisatie
- learning_rate relatief klein -> stabiel leerproces
- subsample + max_features -> stochastic boosting (minder overfitting)


----------------------------
1.2 EXPANDING-WINDOW NOWCASTING
----------------------------

Functie:
- expanding_window_nowcast(X, y)

Voor elke targetdatum t (vanaf MIN_TRAIN_OBS):
1) Train model op alle data < t
2) Predict y_t met X_t (single-point test)
3) Sla y_true, y_pred, errors en aantal trees op

Waarom zo?
- Dit bootst pseudo real-time forecasting na: je gebruikt alleen verledeninformatie.


----------------------------
1.3 METRICS & BENCHMARKS
----------------------------

Metrics:
- RMSE
- MAE
- MSE

Benchmarks:
- **Zero-forecast benchmark**:  
  \(\hat y_t = 0\) voor alle voorspelde kwartalen

- **AR(1) benchmark**:  
  Een AR(1)-model wordt in elk expanding window opnieuw geschat op de
  trainingsdata en gebruikt voor een one-step-ahead voorspelling.

De zero-benchmark correspondeert, omdat de GDP-target is gestandaardiseerd,
met het voorspellen van het onvoorwaardelijke gemiddelde en fungeert als een
strikte “no-information” baseline.  
De AR(1)-benchmark vertegenwoordigt een eenvoudige lineaire
tijdreeksbenchmark met beperkte dynamiek.

Skill (MSE t.o.v. benchmark):
- \(1 - \text{MSE(model)} / \text{MSE(benchmark)}\)

Interpretatie:
- > 0 : model presteert beter dan de benchmark  
- = 0 : model en benchmark zijn even goed  
- < 0 : model presteert slechter dan de benchmark


============================================================
2) LS_TREEBOOST.PY — WAT ZIT HIERIN? (CODE-CONFORM)
============================================================

`ls_treeboost.py` bevat:
A) Data/feature utilities (pseudo real-time feature engineering)
B) Expanding-window backtest helper
C) Model: `class LS_treeboost`

Doel:
- `main.py` blijft puur “experiment runner” (config, pipeline, outputs)
- `ls_treeboost.py` blijft herbruikbaar voor andere targets / datasets (features + model + helper)

A) Data/feature utilities:
- `step0_load_state_space`
- `asof_month`
- `make_lagged_row`
- `step1_build_supervised_set`

B) Backtest helper:
- `expanding_window_treeboost_nowcast`

C) Model:
- `class LS_treeboost`

----------------------------
2.1 STEP 0: STATE-SPACE DATA LADEN
----------------------------

`step0_load_state_space(path_ss, date_col, gdp_col)`

Doet:
- CSV lezen (maandelijks)
- `date_col` naar datetime + sorteren + datum als index
- Splitst:
  - `y_monthly` = GDP kolom (NaNs buiten release-maanden)
  - `Z` = overige kolommen (predictors; numeriek gecast)
- `target_months` = index van maanden waar GDP niet-NaN is (GDP-release maanden)

Return (dict):
- `df_ss` : volledige DataFrame (incl. GDP)
- `Z` : predictor DataFrame (maandelijks)
- `y_monthly` : maand-indexed GDP
- `target_months` : DatetimeIndex met GDP-release maanden

----------------------------
2.2 AS-OF LOGIC (INFORMATIE-AFKAPPUNT)
----------------------------

`asof_month(target_month, rule)`

Mapping:
- `end`   -> `a_t = t`
- `mid`   -> `a_t = t - 1` maand
- `early` -> `a_t = t - 2` maanden

Dit is de kern om info timing expliciet te maken en look-ahead te voorkomen.

----------------------------
2.3 BUILDING BLOCK: 1 LAGGED FEATURE ROW (RAGGED EDGE + PUBLICATION LAGS)
----------------------------

`make_lagged_row(Z, asof_t, lags, feature_cols, release_lags=None, default_release_lag=0)`

Doel:
- Bouw één featurevector voor één target op basis van `asof_t`
- Construeert lagged macro-features voor lags `L ∈ lags`
- Past publication-delay masking toe (ragged edge)

Mechaniek:
- Voor lag `L` wordt de observatiemaand genomen: `tL = asof_t - L`
- Voor variabele `col` geldt publicatielag `d` (in maanden):
  - `d = release_lags[col]` als aanwezig, anders `default_release_lag`

Masking-regel (belangrijk):
- Als `L < d`, dan is die observatie op `asof_t` nog niet gepubliceerd -> feature = `NaN`
- Als `L >= d`, dan gebruik `Z[tL, col]` (als beschikbaar), anders `NaN`

Output:
- dict met keys `{col}_lag{L}`

NB:
- In dit bestand is `default_release_lag=0` de default argumentwaarde.
  In de pipeline kan dit worden overschreven (bijv. vanuit `main.py`) om strengere
  real-time aannames af te dwingen (zoals 1 maand delay voor “unspecified” series).

----------------------------
2.4 STEP 1: SUPERVISED SET BOUWEN (TARGET MONTHS -> (X, y))
----------------------------

`step1_build_supervised_set(Z, y_monthly, target_months, asof_rule, max_lag, feature_cols=None, release_lags=None, default_release_lag=0)`

Voor elke GDP target maand `t`:
1) Neem target `y_t = y_monthly[t]` (alleen in release-maanden niet-NaN)
2) Bepaal `a_t = asof_month(t, asof_rule)`
3) Bouw lagged macro-features (lags 0..max_lag) via `make_lagged_row(...)`
4) Voeg extra AR(1)-achtige feature toe:
   - `GDP_lag1 = y_{t-1}` op kwartaalfrequentie (laatste geobserveerde GDP-release)
   - Deze is real-time beschikbaar omdat het vorige kwartaal al gepubliceerd is

Resultaat:
- `X` : DataFrame met 1 rij per GDP-release maand
- `y` : Series met dezelfde index (GDP target)

Belangrijk:
- Rijen die volledig NaN zijn worden gedropt
- Partiële NaNs blijven -> worden later geïmputeerd tijdens model fit (train-only)

----------------------------
2.5 EXPANDING WINDOW HELPER (PSEUDO REAL-TIME BACKTEST)
----------------------------

`expanding_window_treeboost_nowcast(X, y, min_train_obs, model_params, val_frac_inner, early_stopping_rounds, ar1_func)`

Voor elke targetdatum `t` (vanaf `min_train_obs`):
- Train: alle data strikt vóór `t`
- Test: enkel datapunt `t`

Wat wordt opgeslagen per `t`:
- `y_pred`       : TreeBoost voorspelling
- `y_pred_ar1`   : AR(1) benchmark voorspelling (via `ar1_func(y_train, y_last)`)
- `error` / `error_ar1`
- `abs_error`, `squared_error`
- `n_train`      : # train observaties
- `n_trees`      : # bomen daadwerkelijk gefit (na early stopping)

Return:
- DataFrame met index `date` en alle voorspellingen/foutmaten

----------------------------
2.6 MODEL: CLASS LS_TREEBOOST
----------------------------

`class LS_treeboost`

LS-TreeBoost (LS-Boost / L2 gradient boosting met regressiebomen).

Kernpunten:

(1) Ragged edge / NaNs: train-only imputatie
- Drop train-all-NaN columns (kolommen die in train volledig NaN zijn)
- Bereken kolomgemiddelden op TRAIN (`nanmean`)
- Impute train/val/test met dezelfde train means (geen leakage)

(2) Chronologische validation split
- `_chronological_split(n, val_frac)`:
  - geen shuffle
  - validation = laatste stuk van de tijdreeks

(3) Baseline initialisatie
- Altijd fallback: constante baseline `init_ = mean(y_train)`
- Optioneel (`use_ar1_init=True`) én als feature `GDP_lag1` bestaat:
  - schat baseline: `y ≈ alpha + phi * GDP_lag1` op TRAIN
  - gebruik deze AR(1)-baseline als startvoorspelling

(4) Boosting + early stopping
- Fit boom op residuen
- Update: `learning_rate * tree.predict(...)` (met clipping voor stabiliteit)
- Early stopping op validation MSE:
  - stop na `early_stopping_rounds` zonder verbetering
  - behoud bomen t/m beste iteratie

(5) Predict
- Start met baseline (AR(1) als beschikbaar, anders constante mean)
- Tel boosted correcties op voor alle bomen



============================================================
3) DFM_MODEL.PY — DYNAMIC FACTOR MODEL (OPTIONEEL)
============================================================

`DFM_Model.py` bevat een Dynamic Factor Model (DFM) dat kan worden gebruikt
als alternatief of aanvulling op LS-TreeBoost.

In `main.py` kan dit model worden geactiveerd via:
- `use_dfm = True`

Gedrag:
- Als `use_dfm = True`:
  - wordt eerst een DFM geschat op de maandelijkse macrodata,
  - worden factor-nowcasts gebruikt als input / benchmark.
- Als `use_dfm = False`:
  - draait de pipeline volledig op LS-TreeBoost.

Dit maakt het mogelijk om:
- klassieke lineaire factor-modellen te vergelijken met ML-methoden,
- of DFM-output te combineren met TreeBoost (hybride setting).
