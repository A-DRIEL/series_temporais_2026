import warnings

from carregar import *
from diagnostico_series import * 
from sarima import *

warnings.filterwarnings("ignore")

def main() -> None:
    treino, val = carregar()
    diagnostico(treino)
    comparar_ordens(treino)
    previsoes = pd.concat([sarima(treino, val)], ignore_index=True)
    previsoes.to_csv(ROOT / "previsoes_validacao.csv", index=False)



if __name__ == '__main__':
    main()
