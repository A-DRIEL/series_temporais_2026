import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from config import SERIES, H, M, FIG, TRAIN_END


def plotar_previsoes_baselines(y_train, y_val, preds, s):
    FIG.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_train.index[-8 * M:], y_train.values[-8 * M:], lw=1, color="C0", label="treino")
    ax.plot(y_val.index, y_val.values, lw=1.5, color="black", label="validação (real)")

    estilos = {
        "media": {"color": "#F7A72F", "ls": "--"},
        "naive": {"color": "#CE1818", "ls": "--"},
        "naive_sazonal": {"color": "#38BC1E", "ls": "-"},
        "drift": {"color": "#952BB2", "ls": "-."}
    }
    
    for nome_modelo, y_hat in preds.items():
        ax.plot(y_val.index, y_hat, lw=1.5, color=estilos[nome_modelo]["color"], 
                ls=estilos[nome_modelo]["ls"], label=nome_modelo)
        
    ax.axvline(TRAIN_END, color="gray", ls="--", lw=0.8)
    ax.set_title(f"Previsões Baselines — {s}")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(FIG / f"baselines_previsao_{s}.png", dpi=120)
    plt.close(fig)


def calcular_baselines(treino, val):
    linhas_previsoes = []
    
    for s in SERIES:
        y_train = treino[s]
        
        preds = {
            "media": np.full(H, y_train.mean()),
            "naive": np.full(H, y_train.iloc[-1]),
            "naive_sazonal": np.tile(y_train.iloc[-M:], int(H/M)),
            "drift": y_train.iloc[-1] + ((y_train.iloc[-1] - y_train.iloc[0]) / (len(y_train) - 1)) * np.arange(1, H + 1)
        }
        plotar_previsoes_baselines(y_train, val[s], preds, s)

        datas_validacao = val.index
        for nome_modelo, y_hat in preds.items():
            for data, valor_previsto in zip(datas_validacao, y_hat):
                linhas_previsoes.append({
                    "date": data, "series": s, 
                    "modelo": nome_modelo, "yhat": valor_previsto
                })

    df_previsoes = pd.DataFrame(linhas_previsoes)
    return df_previsoes