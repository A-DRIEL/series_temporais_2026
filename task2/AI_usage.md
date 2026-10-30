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
