# Resumo de Uso de IA no Projeto

## Parte do Diagnóstico das Séries:

### P1:
- **Perguntamos:** Formas de gerar os gráficos da ACF e PACF para as 3 Séries.
- **Sugestão:** Usar a biblioteca **statsmodels**, mais especificamente **from statsmodels.graphics.tsaplots import plot_acf, plot_pacf**.

### P2:
- **Perguntamos:** Ajuda para escrever o código para os gráficos.
- **Sugestão:** Nos forneceu uma função chamada **diagnostico()** que plota os gráficos para as análises.


### Erros dela:
- Os gráficos de ACF/PACF não marcavam os lags sazonais. Ajustamos a função para que os gráficos contivessem linhas verticais em 7, 14, ...


