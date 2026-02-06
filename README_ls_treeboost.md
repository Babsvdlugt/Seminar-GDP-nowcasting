# Seminar-GDP-nowcasting

LS-TreeBoost GDP Nowcasting (Code Guide)

Deze repo implementeert een pseudo real-time expanding-window nowcasting pipeline voor kwartaal-GDP
op basis van maandelijkse (state-space) macrodata met ragged-edge missings.

De code is bewust opgesplitst:

- main.py  -> experiment runner / backtest pipeline
             (data -> features -> expanding-window -> metrics -> output)

- ls_treeboost.py -> model + feature engineering utilities
                    (as-of rule, lagged features, LS-TreeBoost)


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
1.1 CONFIG (WAT JE HIER INSTELT)
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
1.3 METRICS & BENCHMARK
----------------------------

Metrics:
- RMSE, MAE, MSE

Benchmark:
- yhat0 = 0 voor alle voorspelde kwartalen

Skill (MSE vs zero):
- 1 - MSE(model) / MSE(zero)

Interpretatie:
- > 0 : model beter dan benchmark
- = 0 : even goed
- < 0 : slechter

Benchmark:
We use a zero-forecast benchmark, i.e. \hat y_t = 0 for all t.
Since the GDP target is standardized, this corresponds to predicting
the unconditional mean and represents a natural “no-information”
baseline. Forecast skill is reported in MSE terms relative to this benchmark.

============================================================
2) LS_TREEBOOST.PY — WAT ZIT HIERIN?
============================================================

ls_treeboost.py bevat twee soorten onderdelen:

A) Data/feature utilities:
   - step0_load_state_space
   - asof_month
   - make_lagged_row
   - step1_build_supervised_set

B) Model:
   - class LS_treeboost

Doel:
- main.py blijft puur “experiment runner”
- ls_treeboost.py blijft herbruikbaar voor andere targets / datasets


----------------------------
2.1 STEP 0: STATE-SPACE DATA LADEN
----------------------------

step0_load_state_space(path_ss, date_col, gdp_col)

Doet:
- CSV lezen
- date_col naar datetime + sorteren
- y_monthly = GDP kolom (met NaNs buiten release-maanden)
- Z = overige kolommen (predictors)
- target_months = index van maanden waar GDP niet-NaN is

Return:
- {"df_ss": df, "Z": Z, "y_monthly": y_monthly, "target_months": target_months}


----------------------------
2.2 AS-OF LOGIC
----------------------------

asof_month(target_month, rule)

Mapping:
- end   -> t
- mid   -> t - 1 maand
- early -> t - 2 maanden

Dit is de kern om info timing expliciet te maken en look-ahead te voorkomen.


----------------------------
2.3 STEP 1: SUPERVISED SET BOUWEN (RAGGED EDGE + LAGS)
----------------------------

step1_build_supervised_set(Z, y_monthly, target_months, asof_rule, max_lag, feature_cols=None)

Voor elke GDP target maand t:
1) a_t = asof_month(t, asof_rule)
2) Maak feature row:
   voor elke lag L in [0..max_lag]:
     pak Z[a_t - L] voor alle feature_cols
   -> kolomnamen worden: {col}_lag{L}

Resultaat:
- X: DataFrame met 1 rij per GDP-release maand (target_months)
- y: Series met dezelfde index (GDP target)

Belangrijk:
- Rijen die volledig NaN zijn worden gedropt
- Partiële NaNs blijven -> worden later geïmputeerd tijdens model fit


============================================================
3) MODEL: CLASS LS_TREEBOOST (LS BOOSTING MET REGRESSION TREES)
============================================================

LS_treeboost is een L2 gradient boosting model:

- Start met constante voorspelling:
  F0 = mean(y_train)

- Iteratief (m = 1..M):
  resid = y_train - F_{m-1}(X_train)
  fit tree h_m op resid
  update:
  F_m = F_{m-1} + learning_rate * h_m

Dit is “least squares boosting” met regressiebomen.


----------------------------
3.1 NANS / RAGGED EDGE: TRAIN-ONLY IMPUTATION
----------------------------

Waarom?
- Door ragged-edge data zitten er NaNs in X.
- We willen geen leakage.

Wat gebeurt er?
- Tijdens fit:
  col_means = nanmean(X_train_raw) per kolom (TRAIN only)
  train/val/test worden geïmputeerd met diezelfde train means.

Dus:
- Geen informatie uit validation/test wordt gebruikt in imputatie.


----------------------------
3.2 CHRONOLOGISCHE VALIDATION SPLIT
----------------------------

_chronological_split(n, val_frac)

- Geen shuffle
- Validation is het laatste stuk van de tijdreeks (tail)
- Dit sluit aan bij forecasting setting


----------------------------
3.3 EARLY STOPPING
----------------------------

Als val_frac > 0 en early_stopping_rounds > 0:
- track val_mse per iteratie
- stop na early_stopping_rounds zonder verbetering
- bewaar trees t/m best_iter

n_estimators is dus alleen de max-cap.


============================================================
4) “WAT PAS IK WAAR AAN?” (VOOR TEAMMATES)
============================================================

- andere informatie-set (early/mid/end):
  -> main.py: ASOF_RULE

- meer/minder lags:
  -> main.py: MAX_LAG  (default 12 = best via gridsearch)

- expanding-window eerder starten:
  -> main.py: MIN_TRAIN_OBS  (default 25 = best getest)

- model regularisatie / agressiviteit:
  -> main.py: MODEL_PARAMS
     (max_depth, min_samples_leaf/split, subsample, learning_rate)

- model-implementatie wijzigen (imputatie, split, loss):
  -> ls_treeboost.py


