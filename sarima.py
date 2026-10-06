from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import *
from carregar import limpar_natal


# Vizinhas comparadas por AIC/BIC, sempre no treino. A primeira é a que sai direto da ACF/PACF.
CANDIDATAS = [
    ((1, 0, 0), (0, 1, 1)),
    ((1, 0, 1), (0, 1, 1)),
    ((2, 0, 0), (0, 1, 1)),
    ((1, 0, 1), (1, 1, 1)),
    ((0, 1, 1), (0, 1, 1)),
    ((1, 1, 1), (0, 1, 1)),
]


def ajustar(y: pd.Series, ordem: tuple, sazonal: tuple):
    """Ajusta o SARIMA na série de treino, com os zeros de Natal interpolados."""
    return SARIMAX(limpar_natal(y), order=ordem, seasonal_order=sazonal + (M,)).fit(disp=False, maxiter=200)


def comparar_ordens(treino: pd.DataFrame) -> None:
    for s in SERIES:
        print(f"[sarima] candidatas para {s}")
        for ordem, sazonal in CANDIDATAS:
            res = ajustar(treino[s], ordem, sazonal)
            print(f"    {ordem}{sazonal}_{M}  AIC={res.aic:9.1f}  BIC={res.bic:9.1f}")
