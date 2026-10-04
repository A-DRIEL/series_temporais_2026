from carregar import *
from diagnostico_series import * 


def main() -> None:
    treino, val = carregar()
    diagnostico(treino)


if __name__ == '__main__':
    main()
