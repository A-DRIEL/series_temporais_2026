# Resumo de Uso de IA no Projeto

## Ferramentas usadas:
- GPT
- Claude

## Parte do Diagnóstico das Séries:

### P1:
- **Prompt:** Formas de gerar os gráficos da ACF e PACF para as 3 Séries.
- **Sugestão:** Usar a biblioteca **statsmodels**, mais especificamente **from statsmodels.graphics.tsaplots import plot_acf, plot_pacf**.

### P2:
- **Prompt:** Ajuda para escrever o código para os gráficos.
- **Sugestão:** Nos forneceu uma função chamada **diagnostico()** que plota os gráficos para as análises.

## Parte da comparação do SARIMA:

- **Prompt:** Correção e sugestões de melhoria para o texto.
- **Sugestão:** Nos forneceu o texto corrigido e algumas sugestões para melhorá-lo, como a inclusão da tabela com o percentual de decrescimento do SARIMA em relação às baselines.



### Erros dela:
- Os gráficos de ACF/PACF não marcavam os lags sazonais. Ajustamos a função para que os gráficos contivessem linhas verticais em 7, 14, ...

## Responsabilidade:
O grupo revisou, executou e entendeu todo o código e o texto entregues, e responde integralmente pelo conteúdo deste repositório.

---

# Task 2

## Ferramentas usadas
- Claude (Claude Code, no VS Code)

## Onde ajudou
- **Organização do repositório:** congelar as linhas da Task 1 (`carregar_task1()`), trocar `import *` por imports explícitos e renomear este arquivo para `AI_USAGE.md`, nome que o check procura.
- **Features (`features.py`):** montar a tabela da estratégia direta com calendário, eventos, SNAP, preço padronizado com o treino e lags de 28 dias ou mais. A tabela tem um `assert` que impede usar y da validação.
- **Modelo linear (`linear.py`):** OLS por série, previsão dos 28 dias e figura de previsão.
- **Texto:** primeira versão da seção da Task 2 no README, com a comparação do linear contra o naive sazonal e o SARIMA.

## Prompts representativos
1. "Leia os arquivos, o projeto e os enunciados para se ambientar e entender o que precisamos fazer."
2. "Antes, você sugere que alteremos alguma coisa que herdamos da task1 para melhorar?"
3. "Vamos fazer o modelo linear". A IA perguntou a regressão, a estratégia e as features; escolhemos OLS e a estratégia direta com lags ≥ 28, e pedimos a recomendação de features.
4. "Now I want you to build a function boosting(train, val, cov) which fits a boosting model in this dataset, its a time series data type, so dont shuffle the datapoints." A IA montou a função "boosting()". 



## Erros da IA que o grupo corrigiu
- **Linhas da Task 1 reescritas.** Para testar a reprodutibilidade, a IA rodou o `main.py`, que reajustava o SARIMA e sobrescreveu o `previsoes_validacao.csv`, o `metricas.csv` e as figuras entregues; o refit mudava o `yhat` do SARIMA na terceira casa decimal. Restauramos os arquivos pelo git e mudamos o `main.py` para reaproveitar as linhas entregues da Task 1. O refit ficou atrás de `--refazer-task1`.
- **Perda de precisão ao reler o CSV.** A primeira versão de `carregar_task1()` lia o CSV com o parser de float padrão do pandas, que perde o último dígito. Rodar o `main.py` reescrevia 66 linhas da Task 1. O `git diff` mostrou o problema e a correção foi `float_precision="round_trip"`.

## Responsabilidade
O grupo revisou, executou e entendeu todo o código e o texto da Task 2 e responde integralmente pelo que entregou. As features foram fixadas antes de olharmos o erro da validação, e o OLS não tem hiperparâmetro a escolher.
