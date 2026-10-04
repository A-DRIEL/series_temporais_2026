# Resumo de Uso de IA no Projeto

## Parte do Diagnóstico das Séries:

### P1:
- **Perguntamos:** Formas de gerar os gráficos da ACF e PACF para as 3 Séries.
- **Sugestão:** Usar a biblioteca **statsmodels**, mais especificamente **from statsmodels.graphics.tsaplots import plot_acf, plot_pacf**.

### P2:
- **Perguntamos:** Ajuda para escrever o código para os gráficos.
- **Sugestão:** Nos forneceu uma função chamada **diagnostico()** que plota os gráficos para as análises.


### Erros dela:
Ela estava montando gráficos que não marcavam os valores da sazonalidade, então ajustamos para que fosse desenhado uma linha vertical nos pontos (7,14,....)


