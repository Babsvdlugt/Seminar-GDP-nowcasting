"""
Dynamic Factor Model (DFM) voor tijdreeksdata.
Schat latente factoren uit verklarende variabelen en plot de resultaten.
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from statsmodels.tsa.api import DynamicFactor
import matplotlib.pyplot as plt
from statsmodels.tools.sm_exceptions import ConvergenceWarning

# Configureer logging voor betere foutmeldingen
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Onderdruk waarschuwingen (optioneel)
import warnings
warnings.filterwarnings("ignore", category=ConvergenceWarning)

def load_and_prepare_data(file_path: Path) -> pd.DataFrame:
    """
    Laadt en bereidt de data voor voor het DFM.
    Args:
        file_path: Pad naar het CSV-bestand.
    Returns:
        DataFrame met tijdreeksdata (zonder missende waarden).
    """
    try:
        data = pd.read_csv(
            file_path,
            index_col="date",
            parse_dates=True,
            dayfirst=True
        )
        data = data.asfreq("MS")  # Maandelijkse frequentie
        logger.info("Data succesvol ingeladen en voorbereid.")
        return data.dropna()
    except Exception as e:
        logger.error(f"Fout bij het inladen van data: {e}")
        raise

def fit_dynamic_factor_model(X: pd.DataFrame, k_factors: int = 1, factor_order: int = 1) -> DynamicFactor:
    """
    Past een Dynamic Factor Model op de data.
    Args:
        X: DataFrame met verklarende variabelen.
        k_factors: Aantal factoren.
        factor_order: Orde van de factor (AR-proces).
    Returns:
        Gefit model.
    """
    try:
        model = DynamicFactor(
            endog=X,
            k_factors=k_factors,
            factor_order=factor_order
        )
        result = model.fit(disp=False)
        logger.info("DFM succesvol gefit.")
        return result
    except Exception as e:
        logger.error(f"Fout bij het fitten van het model: {e}")
        raise

def plot_factors(factors: pd.DataFrame) -> None:
    """
    Plot de geschatte factoren.
    Args:
        factors: DataFrame met geschatte factoren.
    """
    plt.figure(figsize=(12, 6))
    for i, col in enumerate(factors.columns):
        plt.plot(factors[col], label=f"Factor {i+1}")
    plt.title("Geschatte Factoren uit Dynamic Factor Model")
    plt.xlabel("Tijd")
    plt.ylabel("Factor Waarde")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def main():
    # 1. Data inladen
    data_path = Path("Data prep/Data prep/top20_monthly_grid_from2005.csv")
    data = load_and_prepare_data(data_path)

    # 2. Scheid X (verklarende variabelen) en y (afhankelijke variabele)
    X = data.iloc[:, :-1]

    # 3. DFM fitten
    dfm_result = fit_dynamic_factor_model(X, k_factors=1, factor_order=1)

    # 4. Factoren ophalen en plotten
    factors = dfm_result.factors.smoothed
    factors_df = pd.DataFrame(factors, index=X.index, columns=[f"Factor_{i+1}" for i in range(factors.shape[1])])
    plot_factors(factors_df)

if __name__ == "__main__":
    main()

