import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import Patch

from config import FIG, SERIES

MODELOS = ["media", "naive", "drift", "naive_sazonal", "sarima", "linear", "boosting", "rede"]
MODELOS_TASK2 = {"linear", "boosting", "rede"}
COR_TASK1, COR_TASK2 = "#a9a8a2", "#2a78d6"
TEXTO, TEXTO_2, GRADE = "#0b0b0b", "#52514e", "#e4e3df"
TITULOS = {
    "mae": "MAE na validação (28 dias), em unidades vendidas",
    "rmse": "RMSE na validação (28 dias), em unidades vendidas",
    "mase": "MASE na validação (28 dias). Linha tracejada: erro do naive sazonal no treino",
}


def plotar_metrica(metricas: pd.DataFrame, coluna: str) -> None:
    mesma_escala = coluna == "mase"
    fig, eixos = plt.subplots(1, len(SERIES), figsize=(13, 4.8), sharey=True, sharex=mesma_escala)
    for ax, s in zip(eixos, SERIES):
        sub = metricas[metricas["series"] == s].set_index("modelo").reindex(MODELOS)
        valores = sub[coluna]
        cores = [COR_TASK2 if m in MODELOS_TASK2 else COR_TASK1 for m in sub.index]
        ax.barh(range(len(sub)), valores, color=cores, height=0.7, edgecolor="white", linewidth=2)
        maximo = metricas[coluna].max() if mesma_escala else valores.max()
        formato = "{:.2f}" if mesma_escala else "{:.0f}"
        melhor = valores.idxmin()
        for i, (m, v) in enumerate(valores.items()):
            ax.text(v + 0.015 * maximo, i, formato.format(v), va="center", fontsize=9, color=TEXTO,
                    fontweight="bold" if m == melhor else "normal",
                    bbox=dict(facecolor="white", edgecolor="none", pad=1))
        if mesma_escala:
            ax.axvline(1, color=TEXTO_2, ls="--", lw=1)
        ax.set_title(s, color=TEXTO, fontsize=12, loc="left")
        ax.set_xlim(0, maximo * 1.15)
        ax.grid(axis="x", color=GRADE, lw=0.8)
        ax.set_axisbelow(True)
        for lado in ("top", "right", "left"):
            ax.spines[lado].set_visible(False)
        ax.tick_params(colors=TEXTO_2, length=0)
        ax.set_xlabel(coluna.upper(), color=TEXTO_2)

    eixos[0].set_yticks(range(len(MODELOS)), MODELOS)
    eixos[0].invert_yaxis()
    fig.legend(handles=[Patch(color=COR_TASK1, label="Task 1"), Patch(color=COR_TASK2, label="Task 2")],
               loc="upper right", frameon=False, ncol=2)
    fig.suptitle(TITULOS[coluna], x=0.01, ha="left", color=TEXTO, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / f"metricas_{coluna}.png", dpi=120)
    plt.close(fig)


def plotar_metricas(metricas: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    for coluna in ("mae", "rmse", "mase"):
        plotar_metrica(metricas, coluna)
