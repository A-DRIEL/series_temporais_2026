import warnings

from carregar import *
from diagnostico_series import * 
from sarima import *
from baselines import *

warnings.filterwarnings("ignore")

def main() -> None:
    treino, val = carregar()
    diagnostico(treino)
    comparar_ordens(treino)

    previsoes = pd.concat([calcular_baselines(treino, val), sarima(treino, val)], ignore_index=True)
    previsoes.to_csv(ROOT / "previsoes_validacao.csv", index=False)


if __name__ == '__main__':
    main()
