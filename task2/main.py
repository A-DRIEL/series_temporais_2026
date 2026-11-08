import argparse
import warnings

import matplotlib

matplotlib.use("Agg") 

import pandas as pd

from baselines import calcular_baselines
from boosting import boosting
from carregar import carregar, carregar_covariaveis, carregar_task1
from config import PREVISOES
from diagnostico_series import diagnostico
from linear import linear
from metricas import salvar_metricas
from sarima import comparar_ordens, sarima



warnings.filterwarnings("ignore")


def main() -> None:
    parser = argparse.ArgumentParser(description="Reproduz previsoes_validacao.csv e metricas.csv.")
    parser.add_argument(
        "--refazer-task1",
        action="store_true",
        help="refaz diagnóstico, baselines e SARIMA a partir de dados/ (sobrescreve as linhas entregues na Task 1)",
    )
    args = parser.parse_args()

    treino, val = carregar()
    cov = carregar_covariaveis()
    if args.refazer_task1:
        diagnostico(treino)
        comparar_ordens(treino)
        task1 = pd.concat([calcular_baselines(treino, val), sarima(treino, val)], ignore_index=True)
    else:
        task1 = carregar_task1()

    previsoes = pd.concat([task1, linear(treino, val, cov), boosting(treino, val, cov)], ignore_index=True)

    previsoes.to_csv(PREVISOES, index=False)
    metricas = salvar_metricas(treino, val, previsoes)
    print("\nMétricas na validação (menor é melhor):")
    for serie, grupo in metricas.groupby("series", sort=False):
        print(f"\n{serie}")
        print("  {:<16} {:>10} {:>10} {:>10}".format("modelo", "MAE", "RMSE", "MASE"))
        for linha in grupo.itertuples(index=False):
            print("  {:<16} {:>10.3f} {:>10.3f} {:>10.3f}".format(
                linha.modelo, linha.mae, linha.rmse, linha.mase
            ))


if __name__ == '__main__':
    main()
