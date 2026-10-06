import warnings

from carregar import *
from diagnostico_series import * 
from sarima import *

warnings.filterwarnings("ignore")

def main() -> None:
    treino, val = carregar()
    diagnostico(treino)
    treino, val = carregar()
    diagnostico(treino)
    comparar_ordens(treino)


if __name__ == '__main__':
    main()
