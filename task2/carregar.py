import pandas as pd
import numpy as np
from config import DADOS, H, MODELOS_TASK1, PREVISOES, TRAIN_END


def carregar() -> tuple[pd.DataFrame, pd.DataFrame]:
    treino = pd.read_csv(DADOS / "treino.csv", parse_dates=["date"], index_col="date").asfreq("D")
    val = pd.read_csv(DADOS / "validacao.csv", parse_dates=["date"], index_col="date")
    # Garantia contra vazamento: nada depois de 2016-03-27 entra no ajuste.
    assert treino.index.max() == TRAIN_END
    assert val.index.min() > TRAIN_END and len(val) == H
    return treino, val


def carregar_task1() -> pd.DataFrame:
    """Linhas da Task 1 (baselines e SARIMA) exatamente como foram entregues em previsoes_validacao.csv.

    Reajustar o SARIMA muda o yhat na 3a casa decimal conforme a versão do statsmodels/BLAS,
    então por padrão a Task 2 reaproveita as linhas entregues em vez de refazê-las.
    """
    # round_trip: o parser padrão do pandas perde o último dígito e reescreveria o CSV.
    prev = pd.read_csv(PREVISOES, parse_dates=["date"], float_precision="round_trip")
    prev = prev[prev["modelo"].isin(MODELOS_TASK1)].reset_index(drop=True)
    faltando = set(MODELOS_TASK1) - set(prev["modelo"])
    assert not faltando, f"linhas da Task 1 ausentes em {PREVISOES.name}: {sorted(faltando)}"
    return prev


def limpar_natal(y: pd.Series) -> pd.Series:
    """Zeros de 25/12 (loja fechada) viram NaN e são interpolados. Usado só no ajuste do SARIMA."""
    y = y.astype(float).copy()
    natal = (y.index.month == 12) & (y.index.day == 25)
    y[natal] = np.nan
    return y.interpolate()