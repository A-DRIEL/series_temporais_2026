"""Tabela de features da Task 2 (estratégia direta), a mesma para linear, árvore e rede."""
import numpy as np
import pandas as pd

from carregar import limpar_natal
from config import H, JANELA_MEDIA, LAGS, TRAIN_END

TIPOS_EVENTO = ["Cultural", "National", "Religious", "Sporting"]


def calendario(datas: pd.DatetimeIndex, cov: pd.DataFrame, serie: str, media_preco: float, desvio_preco: float) -> pd.DataFrame:
    """Colunas que só dependem da data e de covariaveis.csv (conhecidas na origem)."""
    c = cov.loc[datas]
    X = pd.DataFrame(index=datas)
    # Dia da semana e mês, com domingo e janeiro como referência.
    for d in range(6):
        X[f"dow_{d}"] = (datas.dayofweek == d).astype(float)
    for m in range(2, 13):
        X[f"mes_{m}"] = (datas.month == m).astype(float)
    X["natal"] = ((datas.month == 12) & (datas.day == 25)).astype(float)
    X["vespera_natal"] = ((datas.month == 12) & (datas.day == 24)).astype(float)
    X["thanksgiving"] = ((c["event_name_1"] == "Thanksgiving") | (c["event_name_2"] == "Thanksgiving")).astype(float)
    for tipo in TIPOS_EVENTO:
        X[f"evento_{tipo}"] = ((c["event_type_1"] == tipo) | (c["event_type_2"] == tipo)).astype(float)
    X["snap"] = c["snap_CA"].astype(float)
    X["preco"] = (c[f"preco_{serie}"] - media_preco) / desvio_preco
    return X


def montar_features(y_treino: pd.Series, cov: pd.DataFrame, serie: str):
    """Devolve (X_treino, y_treino padronizado, X_val, média, desvio).

    Os lags saem de limpar_natal(y) estendido com H NaN nas datas da validação: como todo lag é >= H,
    nenhum valor da validação é usado, e o assert no fim garante isso.
    Média e desvio (da série e do preço) saem só do treino.
    """
    assert y_treino.index.max() == TRAIN_END
    datas_val = pd.date_range(TRAIN_END + pd.Timedelta(days=1), periods=H, freq="D")
    datas = y_treino.index.append(datas_val)

    mu, sd = float(y_treino.mean()), float(y_treino.std())
    preco_treino = cov.loc[y_treino.index, f"preco_{serie}"]
    X = calendario(datas, cov, serie, float(preco_treino.mean()), float(preco_treino.std()))

    z = ((limpar_natal(y_treino) - mu) / sd).reindex(datas)  # validação fica NaN
    for k in LAGS:
        X[f"lag_{k}"] = z.shift(k)
    ini, fim = JANELA_MEDIA
    X[f"media_{ini}_{fim}"] = z.shift(ini).rolling(fim - ini + 1).mean()

    X_treino = X.loc[y_treino.index].dropna()
    X_val = X.loc[datas_val]
    assert not X_val.isna().any().any(), "feature da validação dependeria de y fora do treino"
    y_z = ((y_treino - mu) / sd).loc[X_treino.index]
    return X_treino, y_z, X_val, mu, sd
