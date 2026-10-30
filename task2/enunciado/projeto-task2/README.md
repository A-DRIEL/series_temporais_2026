# Task 2 — Features e três classes

Pacote da Task 2 (linear 01/11, árvore 08/11, rede 15/11; 15% da A2). Enunciado, critérios e formato de entrega: [`enunciado.md`](enunciado.md).

| Peça | Arquivo |
|------|---------|
| Enunciado (três domingos, 15% da A2, tabela de 10 pontos) | [`enunciado.md`](enunciado.md) · [`enunciado.pdf`](enunciado.pdf) |
| Treino, validação, datas do holdout, SNAP, eventos e preço | [`dados/`](dados/) |
| Script de checks (crédito parcial) | [`checks/check_task2.py`](checks/check_task2.py) |

O y do holdout não entra nesta entrega. Treino, validação, datas do holdout e covariáveis estão em `dados/`. O default de `--dados` é essa pasta, ao lado de `checks/`. A partir da raiz do repositório do curso:

```bash
python materia/projeto-task2/checks/check_task2.py --submission /caminho/do/repo/do/grupo
```

O comando acima é o check de 15/11. Antes de 01/11 e de 08/11, use `--ate linear` e `--ate arvore`. Detalhe em [`checks/README.md`](checks/README.md).
