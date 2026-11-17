import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor

from config import pasta_figuras, H, M, SERIES, TRAIN_END
from features import montar_features
from residuos import diagnosticar_residuos

# Fixados a priori, sem olhar o erro da validação
SEMENTES = [0, 1, 2, 3, 4]
PARAMS = dict(
    hidden_layer_sizes=(32, 16),
    activation="relu",
    alpha=1e-2,
    max_iter=500,
    early_stopping=False,
)


def plotar_previsao(y, y_val, yhat, s):
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(y.index[-8 * M:], y.values[-8 * M:], lw=1, color="C0", label="treino (últimas 8 semanas)")
    ax.plot(y_val.index, y_val.values, lw=1, color="black", label="validação")
    ax.plot(yhat.index, yhat.values, lw=1.5, color="C4", label="rede (MLP)")
    ax.axvline(TRAIN_END, color="gray", ls="--", lw=0.8)
    ax.set_title(f"{s} — rede MLP (média de {len(SEMENTES)} sementes), estratégia direta")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(pasta_figuras("rede") / f"rede_previsao_{s}.png", dpi=120)
    plt.close(fig)


def rede(treino: pd.DataFrame, val: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    previsoes = []
    for s in SERIES:
        X_treino, y_z, X_val, mu, sd = montar_features(treino[s], cov, s)
        modelos = [MLPRegressor(random_state=k, **PARAMS).fit(X_treino, y_z) for k in SEMENTES]
        pred_val = np.mean([m.predict(X_val) for m in modelos], axis=0)
        yhat = pd.Series(pred_val * sd + mu, index=X_val.index)
        assert yhat.index.min() > TRAIN_END and len(yhat) == H

        pred_treino = np.mean([m.predict(X_treino) for m in modelos], axis=0)
        r2 = 1 - ((y_z - pred_treino) ** 2).sum() / ((y_z - y_z.mean()) ** 2).sum()
        print(f"[rede] {s:12s} n={len(X_treino)}  colunas={X_treino.shape[1]}  R² treino={r2:.3f}")

        resid = pd.Series((y_z - pred_treino) * sd, index=X_treino.index)
        diagnosticar_residuos(resid, s, "rede")

        plotar_previsao(treino[s], val[s], yhat, s)
        previsoes.append(pd.DataFrame({"date": yhat.index, "series": s, "modelo": "rede", "yhat": yhat.values}))
    return pd.concat(previsoes, ignore_index=True)