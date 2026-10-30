from pathlib import Path
import pandas as pd


ROOT = Path(__file__).resolve().parent
DADOS = ROOT / "dados"
FIG = ROOT / "figuras"
PREVISOES = ROOT / "previsoes_validacao.csv"
SERIES = ["store_total", "FOODS", "HOBBIES"]
H = 28  # horizonte de validação
M = 7  # sazonalidade semanal
TRAIN_END = pd.Timestamp("2016-03-27")
MODELOS_TASK1 = ["media", "naive", "naive_sazonal", "drift", "sarima"]