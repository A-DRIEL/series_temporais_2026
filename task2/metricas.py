"""Calcula MAE, RMSE e MASE das previsões no horizonte de validação."""
from __future__ import annotations
import numpy as np
import pandas as pd
from config import M, ROOT, SERIES

def calcular_metricas(
    treino: pd.DataFrame,
    val: pd.DataFrame,
    previsoes: pd.DataFrame,
) -> pd.DataFrame:
    """Retorna uma linha por série/modelo, comparando previsões com a validação.

    A escala do MASE é o MAE in-sample do naive sazonal calculado no treino:
    média de ``|y[t] - y[t-M]|`` para todos os pares disponíveis.
    """
    colunas_prev = {"date", "series", "modelo", "yhat"}
    faltantes = colunas_prev - set(previsoes.columns)
    if faltantes:
        raise ValueError(f"Colunas ausentes nas previsões: {sorted(faltantes)}")
    prev = previsoes.copy()
    prev["date"] = pd.to_datetime(prev["date"])
    if prev.duplicated(["date", "series", "modelo"]).any():
        raise ValueError("Há mais de uma previsão para a mesma data/série/modelo.")
    linhas: list[dict[str, str | float]] = []
    for serie in SERIES:
        observado = val[serie].astype(float)
        yhat_serie = prev.loc[prev["series"] == serie]
        if yhat_serie.empty:
            raise ValueError(f"Não há previsões para a série {serie}.")
        escala_mase = float((treino[serie].astype(float) - treino[serie].astype(float).shift(M)).abs().dropna().mean())
        if not np.isfinite(escala_mase) or escala_mase <= 0:
            raise ValueError(f"Escala sazonal inválida para {serie}: {escala_mase}.")
        for modelo, grupo in yhat_serie.groupby("modelo", sort=False):
            pred = grupo.set_index("date")["yhat"].reindex(observado.index)
            if pred.isna().any():
                raise ValueError(f"Previsões incompletas para {serie}/{modelo}.")
            erro = pred.to_numpy(dtype=float) - observado.to_numpy(dtype=float)
            mae = float(np.mean(np.abs(erro)))
            rmse = float(np.sqrt(np.mean(np.square(erro))))
            linhas.append({
                "series": serie,
                "modelo": str(modelo),
                "mae": mae,
                "rmse": rmse,
                "mase": mae / escala_mase,
            })
    return pd.DataFrame(linhas, columns=["series", "modelo", "mae", "rmse", "mase"])


def salvar_metricas(treino: pd.DataFrame,val: pd.DataFrame,previsoes: pd.DataFrame,destino=ROOT / "metricas.csv"):
    """Calcula e grava as métricas em CSV."""
    metricas = calcular_metricas(treino, val, previsoes)
    metricas.to_csv(destino, index=False, float_format="%.10f")
    return metricas
