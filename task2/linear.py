import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import FIG, H, M, SERIES, TRAIN_END
from features import montar_features


def plotar_previsao(y: pd.Series, y_val: pd.Series, yhat: pd.Series, s: str) -> None:
    """A validação entra só aqui, para desenhar o realizado ao lado da previsão."""
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index[-8 * M:], y.values[-8 * M:], lw=1, color="C0", label="treino (últimas 8 semanas)")
    ax.plot(y_val.index, y_val.values, lw=1, color="black", label="validação")
    ax.plot(yhat.index, yhat.values, lw=1.5, color="C2", label="linear (OLS)")
    ax.axvline(TRAIN_END, color="gray", ls="--", lw=0.8)
    ax.set_title(f"{s} — regressão linear (OLS), estratégia direta")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"linear_previsao_{s}.png", dpi=120)
    plt.close(fig)


def linear(treino: pd.DataFrame, val: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """OLS por série, só com o treino, prevendo os H dias seguintes (formato longo)."""
    FIG.mkdir(exist_ok=True)
    previsoes = []
    for s in SERIES:
        X_treino, y_z, X_val, mu, sd = montar_features(treino[s], cov, s)
        modelo = LinearRegression().fit(X_treino, y_z)
        yhat = pd.Series(modelo.predict(X_val) * sd + mu, index=X_val.index)
        assert yhat.index.min() > TRAIN_END and len(yhat) == H

        print(f"[linear] {s:12s} n={len(X_treino)}  colunas={X_treino.shape[1]}  R² treino={modelo.score(X_treino, y_z):.3f}")
        plotar_previsao(treino[s], val[s], yhat, s)
        previsoes.append(pd.DataFrame({"date": yhat.index, "series": s, "modelo": "linear", "yhat": yhat.values}))
    return pd.concat(previsoes, ignore_index=True)
