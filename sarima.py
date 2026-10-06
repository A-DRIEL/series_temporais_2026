from statsmodels.tsa.statespace.sarimax import SARIMAX
from config import *
from carregar import limpar_natal


# (p, d, q), (P, D, Q) escolhidos pela ACF/PACF de (1 - B^7) y e pelo AIC/BIC no treino (ver README).
ORDENS = {
    "store_total": ((1, 0, 1), (0, 1, 1)),
    "FOODS": ((1, 0, 1), (0, 1, 1)),
    "HOBBIES": ((1, 1, 1), (0, 1, 1)),
}


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

def sarima(treino: pd.DataFrame) -> pd.DataFrame:
    """Ajusta um SARIMA por série só com o treino e devolve as previsões dos H dias seguintes (formato longo)."""
    previsoes = []
    for s in SERIES:
        ordem, sazonal = ORDENS[s]
        res = ajustar(treino[s], ordem, sazonal)
        yhat = res.forecast(H)
        assert yhat.index.min() > TRAIN_END and len(yhat) == H

        coefs = "  ".join(f"{nome}={valor:.3f}" for nome, valor in res.params.items() if nome != "sigma2")
        print(f"[sarima] {s:12s} SARIMA{ordem}{sazonal}_{M}  AIC={res.aic:.1f}  BIC={res.bic:.1f}")
        print(f"         {coefs}")

        previsoes.append(pd.DataFrame({"date": yhat.index, "series": s, "modelo": "sarima", "yhat": yhat.values}))
    return pd.concat(previsoes, ignore_index=True)

