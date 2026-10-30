import pandas as pd
import numpy as np
from config import *


def carregar() -> tuple[pd.DataFrame, pd.DataFrame]:
    treino = pd.read_csv(DADOS / "treino.csv", parse_dates=["date"], index_col="date").asfreq("D")
    val = pd.read_csv(DADOS / "validacao.csv", parse_dates=["date"], index_col="date")
    # Garantia contra vazamento: nada depois de 2016-03-27 entra no ajuste.
    assert treino.index.max() == TRAIN_END
    assert val.index.min() > TRAIN_END and len(val) == H
    return treino, val


def limpar_natal(y: pd.Series) -> pd.Series:
    """Zeros de 25/12 (loja fechada) viram NaN e são interpolados. Usado só no ajuste do SARIMA."""
    y = y.astype(float).copy()
    natal = (y.index.month == 12) & (y.index.day == 25)
    y[natal] = np.nan
    return y.interpolate()