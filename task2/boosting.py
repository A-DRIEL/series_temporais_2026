import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor

from config import FIG, H, M, SERIES, TRAIN_END
from features import montar_features
from residuos import diagnosticar_residuos



def plotar_previsao(y: pd.Series, y_val: pd.Series, yhat: pd.Series, s: str) -> None:
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index[-8 * M:], y.values[-8 * M:], lw=1, color="C0", label="treino (últimas 8 semanas)")
    ax.plot(y_val.index, y_val.values, lw=1, color="black", label="validação")
    ax.plot(yhat.index, yhat.values, lw=1.5, color="C3", label="boosting")
    ax.axvline(TRAIN_END, color="gray", ls="--", lw=0.8)
    ax.set_title(f"{s} — gradient boosting de árvores, estratégia direta")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"boosting_previsao_{s}.png", dpi=120)
    plt.close(fig)


def boosting(treino: pd.DataFrame, val: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    FIG.mkdir(exist_ok=True)
    previsoes = []
    for s in SERIES:
        X_treino, y_z, X_val, mu, sd = montar_features(treino[s], cov, s)
        modelo = HistGradientBoostingRegressor(early_stopping=False, random_state=0).fit(X_treino, y_z)
        yhat = pd.Series(modelo.predict(X_val) * sd + mu, index=X_val.index)
        assert yhat.index.min() > TRAIN_END and len(yhat) == H


        print(f"[boosting] {s:12s} n={len(X_treino)}  colunas={X_treino.shape[1]}  R² treino={modelo.score(X_treino, y_z):.3f}")
        #diagnostico dos resíduos
        resid = pd.Series((y_z - modelo.predict(X_treino)) * sd, index=X_treino.index)
        diagnosticar_residuos(resid, s, "boosting")

        plotar_previsao(treino[s], val[s], yhat, s)
        previsoes.append(pd.DataFrame({"date": yhat.index, "series": s, "modelo": "boosting", "yhat": yhat.values}))
    return pd.concat(previsoes, ignore_index=True)
