import warnings

from carregar import *
from diagnostico_series import * 
from sarima import *
from baselines import *
from metricas import salvar_metricas

warnings.filterwarnings("ignore")

def main() -> None:
    treino, val = carregar()
    diagnostico(treino)
    comparar_ordens(treino)

    previsoes = pd.concat([calcular_baselines(treino, val), sarima(treino, val)], ignore_index=True)
    previsoes.to_csv(ROOT / "previsoes_validacao.csv", index=False)
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
