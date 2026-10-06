from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from config import *
from carregar import limpar_natal
from diagnostico_series import marcar_sazonais
import matplotlib.pyplot as plt



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

def ljung_box(res, ordem: tuple, sazonal: tuple) -> pd.Series:
    """p-valores do Ljung-Box nos resíduos (sem os primeiros, que dependem da inicialização difusa)."""
    n_params = ordem[0] + ordem[2] + sazonal[0] + sazonal[2]
    queima = ordem[1] + sazonal[1] * M
    return acorr_ljungbox(res.resid.iloc[queima:], lags=[14, 28], model_df=n_params)["lb_pvalue"]


def plotar_residuos(res, s: str, titulo: str) -> None:
    resid = res.resid.iloc[M + 1:]
    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(resid.index, resid.values, lw=0.6)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_title(f"Resíduos — {s} — {titulo}")
    ax_acf = fig.add_subplot(2, 2, 3)
    plot_acf(resid, lags=35, ax=ax_acf, zero=False, auto_ylims=True, title=f"ACF dos resíduos — {s}")
    marcar_sazonais(ax_acf, 35)
    ax_pacf = fig.add_subplot(2, 2, 4)
    plot_pacf(resid, lags=35, ax=ax_pacf, zero=False, auto_ylims=True, method="ywm", title=f"PACF dos resíduos — {s}")
    marcar_sazonais(ax_pacf, 35)
    fig.tight_layout()
    fig.savefig(FIG / f"sarima_residuos_{s}.png", dpi=120)
    plt.close(fig)


def sarima(treino: pd.DataFrame) -> pd.DataFrame:
    """Ajusta um SARIMA por série só com o treino e devolve as previsões dos H dias seguintes (formato longo)."""
    FIG.mkdir(exist_ok=True)
    previsoes = []
    for s in SERIES:
        ordem, sazonal = ORDENS[s]
        titulo = f"SARIMA{ordem}{sazonal}_{M}"
        res = ajustar(treino[s], ordem, sazonal)
        yhat = res.forecast(H)
        assert yhat.index.min() > TRAIN_END and len(yhat) == H

        lb = ljung_box(res, ordem, sazonal)
        coefs = "  ".join(f"{nome}={valor:.3f}" for nome, valor in res.params.items() if nome != "sigma2")
        print(f"[sarima] {s:12s} {titulo}  AIC={res.aic:.1f}  BIC={res.bic:.1f}  Ljung-Box p(14)={lb[14]:.3f}  p(28)={lb[28]:.3f}")
        print(f"         {coefs}")

        plotar_residuos(res, s, titulo)
        previsoes.append(pd.DataFrame({"date": yhat.index, "series": s, "modelo": "sarima", "yhat": yhat.values}))
    return pd.concat(previsoes, ignore_index=True)


