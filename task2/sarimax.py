import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

from carregar import limpar_natal
from config import H, M, SERIES, TRAIN_END, pasta_figuras
from features import calendario
from residuos import diagnosticar_residuos
from sarima import ORDENS

COVARIAVEIS = ["snap", "evento_Cultural", "evento_National", "evento_Religious", "evento_Sporting", "thanksgiving"]


def plotar_previsao(y: pd.Series, y_val: pd.Series, prev, s: str, titulo: str) -> None:
    ic = prev.conf_int(alpha=0.05)
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index[-8 * M:], y.values[-8 * M:], lw=1, color="C0", label="treino (últimas 8 semanas)")
    ax.plot(y_val.index, y_val.values, lw=1, color="black", label="validação")
    ax.plot(prev.predicted_mean.index, prev.predicted_mean.values, lw=1.5, color="C1", label="SARIMAX")
    ax.fill_between(ic.index, ic.iloc[:, 0], ic.iloc[:, 1], color="C1", alpha=0.2, label="IC 95%")
    ax.axvline(TRAIN_END, color="gray", ls="--", lw=0.8)
    ax.set_title(f"{s} — {titulo}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(pasta_figuras("sarimax") / f"sarimax_previsao_{s}.png", dpi=120)
    plt.close(fig)


def sarimax(treino: pd.DataFrame, val: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    datas_val = pd.date_range(TRAIN_END + pd.Timedelta(days=1), periods=H, freq="D")
    previsoes = []
    for s in SERIES:
        ordem, sazonal = ORDENS[s]
        titulo = f"SARIMAX{ordem}{sazonal}_{M} com SNAP e eventos"
        X = calendario(treino.index.append(datas_val), cov, s, 0.0, 1.0)[COVARIAVEIS]
        res = SARIMAX(limpar_natal(treino[s]), exog=X.loc[treino.index], order=ordem,
                      seasonal_order=sazonal + (M,)).fit(disp=False, maxiter=200)
        prev = res.get_forecast(H, exog=X.loc[datas_val])
        yhat = prev.predicted_mean
        assert yhat.index.min() > TRAIN_END and len(yhat) == H

        coefs = "  ".join(f"{c}={res.params[c]:.1f}" for c in COVARIAVEIS)
        print(f"[sarimax] {s:12s} AIC={res.aic:.1f}  {coefs}")
        diagnosticar_residuos(res.resid.iloc[M + 1:], s, "sarimax")
        plotar_previsao(treino[s], val[s], prev, s, titulo)
        previsoes.append(pd.DataFrame({"date": yhat.index, "series": s, "modelo": "sarimax", "yhat": yhat.values}))
    return pd.concat(previsoes, ignore_index=True)
