# Trabalho de Séries Temporais

Para rodar (a partir desta pasta):

```shell
python -m pip install -r requirements.txt
python main.py
```

O `main.py` reaproveita as linhas da Task 1 (média, naive, naive sazonal, drift e SARIMA) exatamente como foram entregues em `previsoes_validacao.csv`, ajusta os modelos da Task 2 e regrava `previsoes_validacao.csv` e `metricas.csv`. Para refazer também o diagnóstico, os baselines e o SARIMA a partir de `dados/`, use `python main.py --refazer-task1`. O refit do SARIMA pode mudar o `yhat` na terceira casa decimal, conforme a versão do statsmodels.

Check público da Task 2 (padrão `--ate rede`; aceita `--ate linear` ou `--ate arvore`):

```shell
python checks/check_task2.py --submission .
```

# Task 1

## Diagnóstico dos dados

### FOODS
- **Tendência:** sobe até um pico em meados de 2013 ($\sim 3400$ na média móvel) e depois cai devagar até $\sim 2600\text{–}3000$. O padrão anual é parecido com o do store_total, o que faz sentido: FOODS é a maior parte das vendas. O ADF também não rejeita no nível ($p = 0{,}15$).

- **Sazonalidade $m = 7$:** forte, com picos de ACF de $\sim 0{,}75\text{–}0{,}8$ a cada $7$ dias. Depois da diferença, o padrão é igual ao do total: $\text{AR}(1)$ + $\text{MA}(1)$ sazonal. O ciclo mensal (defasagens $28\text{–}32$, $\approx 0{,}33$) é mais forte do que no total, o que combina com o efeito SNAP, que pesa mais em alimentos.

- **Outliers:** os zeros de Natal e quedas pontuais no fim de novembro.

### STORE_TOTAL

- **Tendência:** sobe de $\sim 3000$ (2011) para $\sim 4500$ (meados de 2013) e depois fica estável entre $4200$ e $4700$. Há um padrão anual leve, com pico no meio do ano e queda na virada do ano. O ADF não rejeita raiz unitária no nível ($p = 0{,}25$).

- **Sazonalidade $m = 7$:** muito forte. A ACF tem picos de $\sim 0{,}8$ nas defasagens $7, 14, \dots, 56$ e eles quase não caem, o que pede a diferença sazonal $D = 1$. Depois de $(1 - B^7)$:
    - nas defasagens $1\text{–}4$ a ACF cai aos poucos e a PACF corta na $1$, o que sugere um $\text{AR}(1)$ não sazonal;
    - na defasagem $7$ há um pico isolado ($\approx -0{,}45$) e a PACF cai nas defasagens $7, 14, 21, 28$, o que indica um $\text{MA}(1)$ sazonal ($Q = 1$).
    - Há também uma elevação nas defasagens $26\text{–}33$, que parece um ciclo mensal (pode ser o dia do benefício SNAP).

- **Outliers:** os $5$ zeros de Natal, mais algumas quedas para $\sim 2000\text{–}2500$ no fim de novembro, que provavelmente são o Thanksgiving


### HOBBIES

- **Tendência:** diferente das outras duas. O nível fica em $\sim 400$ em 2011–2012, cai para $\sim 300$ no fim de 2012 e salta para $\sim 450\text{–}500$ no começo de 2013 (parece uma quebra estrutural). Depois cresce devagar até $\sim 550$. O ADF dá $p = 0{,}037$, mas como o teste não lida bem com quebras de nível, esse resultado é pouco confiável.

- **Sazonalidade $m = 7$:** existe, mas é mais fraca e a série tem mais ruído. Os picos de ACF ficam em $\sim 0{,}5$, e a ACF é positiva em todas as defasagens, caindo devagar, por causa da persistência do nível. Depois da diferença:
    - a ACF tem pico de $\approx -0{,}52$ na defasagem $7$ e quase nada nas outras, ou seja, um $\text{MA}(1)$ sazonal com $\Theta$ próximo de $1$. Isso sugere que a sazonalidade semanal é quase determinística (perto de diferenciar demais);
    - na parte não sazonal há só um efeito pequeno na defasagem $1$ ($\approx 0{,}2$);
    - não há ciclo mensal.
- **Outliers:** os zeros de Natal, um pico de $\sim 950$ no fim de 2014 e quedas para $\sim 150$ no fim de novembro de 2015.


## Métricas de validação

Ao executar `python main.py`, o projeto também grava `metricas.csv` com MAE, RMSE e MASE para cada modelo (média, naive, naive sazonal, drift e SARIMA) e cada série. As métricas são calculadas nos 28 dias de validação. A escala do MASE é o MAE in-sample do naive sazonal semanal no treino, isto é, a média de `|y[t] - y[t-7]|` para todos os pares disponíveis no treino.

## Baselines 

As quatro baselines exigidas (média, naive, naive sazonal e drift) foram calculadas para o horizonte de validação (28 dias) e estão consolidadas no arquivo `previsoes_validacao.csv`. 

Analisando o comportamento das previsões geradas (`figuras/baselines_previsao_*.png`), os principais resultados observados foram:

- **Naive Sazonal ($m=7$):** Como as três séries apresentam forte sazonalidade semanal, este foi, de longe, o baseline que apresentou o melhor resultado visual. Ele conseguiu reproduzir fielmente os ciclos de dias da semana (como os picos de fim de semana), acompanhando bem de perto a série de validação real.

- **Média, Naive e Drift:** Por serem métodos que não incorporam o componente sazonal, geraram apenas projeções retilíneas (retas horizontais ou com leve inclinação). Visualmente, eles "cortam" a série no meio, falhando em capturar a variação diária. 

Esses resultados mostram que a sazonalidade é o fator predominante das séries.


## ARIMA/SARIMA

O código está em `sarima.py`. O ajuste usa só o treino (até 2016-03-27), com os $5$ zeros de Natal interpolados, e a previsão cobre os $28$ dias da validação. Nenhuma covariável foi usada.

### Como as ordens foram escolhidas

1. **Diferenciação.** A ACF no nível tem picos de $\sim 0{,}8$ em $7, 14, \dots$ que quase não caem, então usamos $D = 1$ com $m = 7$ nas três séries. Depois de $(1 - B^7)$ o ADF rejeita raiz unitária nas três ($p < 10^{-26}$).
2. **Ponto de partida pela ACF/PACF de $(1 - B^7)y$.** Pico isolado na defasagem $7$ da ACF e PACF caindo em $7, 14, 21, 28$ dão $Q = 1$ e $P = 0$. Nas defasagens baixas, a ACF cai aos poucos e a PACF corta na $1$, o que dá $p = 1$. O ponto de partida é o $\text{SARIMA}(1,0,0)(0,1,1)_7$.
3. **Refino por AIC/BIC no treino**, comparando com modelos vizinhos (a tabela é impressa pelo `main.py`):

| Modelo | store_total (AIC / BIC) | FOODS (AIC / BIC) | HOBBIES (AIC / BIC) |
|---|---|---|---|
| $(1,0,0)(0,1,1)_7$ | $27773{,}9$ / $27790{,}6$ | $26692{,}7$ / $26709{,}3$ | $22062{,}4$ / $22079{,}0$ |
| $(1,0,1)(0,1,1)_7$ | $\mathbf{27761{,}4}$ / $\mathbf{27783{,}5}$ | $\mathbf{26678{,}1}$ / $\mathbf{26700{,}3}$ | $22020{,}8$ / $22043{,}0$ |
| $(2,0,0)(0,1,1)_7$ | $27763{,}3$ / $27785{,}4$ | $26680{,}6$ / $26702{,}7$ | $22041{,}9$ / $22064{,}1$ |
| $(1,0,1)(1,1,1)_7$ | $27763{,}4$ / $27791{,}1$ | $26679{,}9$ / $26707{,}5$ | $22018{,}7$ / $22046{,}4$ |
| $(0,1,1)(0,1,1)_7$ | $27907{,}4$ / $27924{,}0$ | $26891{,}8$ / $26908{,}4$ | $22032{,}9$ / $22049{,}5$ |
| $(1,1,1)(0,1,1)_7$ | $27769{,}0$ / $27791{,}1$ | $26692{,}4$ / $26714{,}5$ | $\mathbf{21996{,}5}$ / $\mathbf{22018{,}7}$ |

### Modelos escolhidos

| Série | Modelo | $\phi_1$ | $\theta_1$ | $\Theta_1$ |
|---|---|---|---|---|
| store_total | $\text{SARIMA}(1,0,1)(0,1,1)_7$ | $0{,}622$ | $-0{,}207$ | $-0{,}850$ |
| FOODS | $\text{SARIMA}(1,0,1)(0,1,1)_7$ | $0{,}678$ | $-0{,}186$ | $-0{,}860$ |
| HOBBIES | $\text{SARIMA}(1,1,1)(0,1,1)_7$ | $0{,}159$ | $-0{,}947$ | $-0{,}999$ |

- **store_total e FOODS:** acrescentar um $\text{MA}(1)$ ao ponto de partida reduz AIC e BIC. O $\text{AR}$ sazonal ($P = 1$) piora os dois critérios. Não usamos $d = 1$ porque o $(1,1,1)(0,1,1)_7$ tem BIC maior e a série, depois da diferença sazonal, já é estacionária.
- **HOBBIES:** é a única série em que os modelos com $d = 1$ ganham dos com $d = 0$, o que combina com a quebra de nível de 2013 vista no diagnóstico. O $(1,1,1)(0,1,1)_7$ tem o menor AIC e BIC. O $\Theta_1 \approx -1$ fica na fronteira de invertibilidade: é o que o diagnóstico já indicava, uma sazonalidade semanal quase determinística.

### Resíduos

As figuras `figuras/sarima_residuos_*.png` mostram os resíduos e suas ACF/PACF. O Ljung-Box rejeita ruído branco nas três séries ($p \le 0{,}002$ em store_total, $p < 0{,}001$ em FOODS e $p = 0{,}022$ / $0{,}005$ nas defasagens $14$ / $28$ em HOBBIES), então os modelos não estão totalmente adequados. Em store_total e FOODS o que sobra é a autocorrelação nas defasagens $29\text{–}32$ (até $\sim 0{,}16$), o ciclo mensal já apontado no diagnóstico. Um SARIMA com $m = 7$ e sem regressores não consegue capturar esse ciclo. Em HOBBIES as autocorrelações que sobram são pequenas ($|\rho| < 0{,}07$) e sem padrão.

### Previsão

As figuras `figuras/sarima_previsao_*.png` mostram a previsão dos $28$ dias com intervalo de $95\%$ ao lado do realizado. As previsões são gravadas em `previsoes_validacao.csv` com `modelo = sarima`.


## Comparação do modelo complexo com os quatro baselines

O modelo mais complexo, o **SARIMA**, foi comparado com os quatro baselines (média, naive, naive sazonal com m = 7 e drift) no mesmo split, no horizonte de validação de 28 dias (2016-03-28 a 2016-04-24). As métricas estão em `metricas.csv`.

### Redução do erro do SARIMA frente a cada baseline

| Série | Métrica | vs. média | vs. naive | vs. naive sazonal | vs. drift |
|---|---|---:|---:|---:|---:|
| store_total | MAE | −61,1% | −63,3% | −31,1% | −63,3% |
| store_total | RMSE | −64,3% | −59,0% | −39,5% | −59,0% |
| FOODS | MAE | −57,7% | −66,4% | −35,1% | −66,4% |
| FOODS | RMSE | −59,9% | −62,7% | −35,9% | −62,7% |
| HOBBIES | MAE | −33,3% | −56,2% | −27,6% | −56,6% |
| HOBBIES | RMSE | −32,5% | −53,4% | −35,0% | −53,7% |

### Resultado geral

O SARIMA obteve o menor MAE, RMSE e MASE nas três séries. Ele não perdeu para nenhum baseline, incluindo a média e o naive sazonal, que são os mais relevantes por serem os mais competitivos neste problema.

### Contra o naive sazonal

O naive sazonal é o melhor baseline em `store_total` e `FOODS`, e o SARIMA o supera com folga: o MAE cai 31,1% em `store_total` e 35,1% em `FOODS`. Em `HOBBIES` a vantagem é menor (−27,6% no MAE).

O MASE do naive sazonal fica em torno de 1 (1,06 / 1,01 / 1,12), ou seja, na validação ele erra aproximadamente o mesmo que errou in-sample, que é o esperado para a escala usada. O SARIMA fica abaixo de 1 nas três séries (0,73 / 0,65 / 0,81), isto é, erra menos do que o naive sazonal errou no treino.

### Contra média, naive e drift

Esses três são claramente piores. O naive e o drift têm desempenho bem semelhante (MASE ≈ 1,85–1,98), porque a inclinação do drift estimada no treino é muito pequena e a previsão praticamente repete o último valor, sem capturar o padrão semanal. A média também ignora a sazonalidade e erra mais que o naive sazonal em `store_total` e `FOODS`.

### Outras observações:

1. **HOBBIES é o caso mais apertado.** Nessa série o ganho do SARIMA é o menor e, no RMSE, a *média* (141,3) supera o naive sazonal (146,7), embora o naive sazonal ganhe da média no MAE (102,3 vs. 111,0). Isso sugere que a série é ruidosa, com erros grandes pontuais que o naive sazonal replica ao copiar a semana anterior, e que o sinal sazonal semanal é mais fraco aqui do que nas outras séries.
2. **Complexidade.** O SARIMA exige identificação de ordem e ajuste, enquanto o naive sazonal não tem parâmetros. Em `HOBBIES` o ganho absoluto é pequeno (≈ 28 unidades de MAE), e o custo de complexidade pode pesar mais.

### Conclusão

Nas três séries, o ganho do SARIMA sobre o naive sazonal é consistente em MAE, RMSE e MASE, o que justifica o uso do modelo, com a cautela acima para `HOBBIES`.


# Task 2

## Estratégia e features

O código está em `features.py`, que monta a tabela usada por todas as classes da Task 2 (`linear.py`, `boosting.py` e `rede.py`), e em `residuos.py`, que faz o diagnóstico dos resíduos no treino. Usamos a estratégia **direta**: um único modelo por série prevê os 28 dias, e toda coluna que depende de y usa defasagem de **28 dias ou mais**. Assim, o lag de qualquer dia da validação cai dentro do treino. No código, a série usada para os lags é o treino seguido de 28 `NaN` nas datas da validação, e um `assert` exige que a matriz da validação não tenha `NaN`. Se algum lag tocasse a validação, a execução pararia. O y da validação só é lido para desenhar o gráfico e calcular as métricas.

Cada série tem o próprio ajuste: `store_total`, `FOODS` e `HOBBIES` nunca estão na mesma tabela.

| Grupo | Colunas | Origem |
|---|---|---|
| Dia da semana | 6 dummies (domingo é a referência) | calendário |
| Mês | 11 dummies (janeiro é a referência) | calendário |
| Natal, véspera, Thanksgiving | 3 dummies | calendário e `event_name_1/2` |
| Tipo de evento | Cultural, National, Religious, Sporting | `event_type_1/2` |
| SNAP | `snap_CA` | `covariaveis.csv` |
| Preço | `preco_<série>`, padronizado com média e desvio **do treino** | `covariaveis.csv` |
| Lags | `y[t-28]`, `y[t-35]`, `y[t-42]`, `y[t-49]` e a média de `y[t-28]` a `y[t-55]` | treino |

- **Lags e alvo.** Os lags são calculados sobre a série com os zeros de Natal interpolados, a mesma `limpar_natal` do SARIMA, para que um zero não contamine os dias seguintes. O alvo continua sendo o y bruto, e o zero do Natal fica a cargo da dummy.
- **Padronização.** A série é padronizada com média e desvio do treino e a previsão volta à escala original no fim.
- **Primeiros 55 dias.** Ficam fora do ajuste, porque ainda não têm todos os lags.
- **Sem tendência linear.** A média dos lags já carrega o nível recente. Uma reta ajustada em cinco anos extrapolaria mal, principalmente em HOBBIES, que tem a quebra de nível de 2013.
- **Escolha a priori.** A lista de colunas foi fixada antes de olharmos o erro da validação, a partir do diagnóstico da Task 1:
  - sazonalidade semanal: dia da semana e lags múltiplos de 7;
  - quedas de Natal e Thanksgiving: dummies;
  - ciclo mensal que sobrou nos resíduos do SARIMA: SNAP.

## `linear`

Usamos regressão linear por mínimos quadrados (OLS, `sklearn.linear_model.LinearRegression`). O OLS **não tem hiperparâmetro**, então não há nada a escolher pela validação. O R² no treino é $0{,}856$ em store_total, $0{,}817$ em FOODS e $0{,}541$ em HOBBIES. As figuras `figuras/linear_previsao_*.png` mostram a previsão ao lado do realizado.

| Série | Métrica | linear | naive sazonal | SARIMA | linear vs. naive sazonal | linear vs. SARIMA |
|---|---|---:|---:|---:|---:|---:|
| store_total | MAE | 200,3 | 459,2 | 316,6 | −56,4% | −36,7% |
| store_total | RMSE | 241,3 | 655,5 | 396,5 | −63,2% | −39,1% |
| store_total | MASE | 0,460 | 1,056 | 0,728 | −56,4% | −36,7% |
| FOODS | MAE | 122,0 | 343,9 | 223,1 | −64,5% | −45,3% |
| FOODS | RMSE | 149,2 | 420,1 | 269,5 | −64,5% | −44,6% |
| FOODS | MASE | 0,357 | 1,005 | 0,652 | −64,5% | −45,3% |
| HOBBIES | MAE | 75,6 | 102,3 | 74,0 | −26,1% | **+2,1%** |
| HOBBIES | RMSE | 89,1 | 146,7 | 95,4 | −39,2% | −6,5% |
| HOBBIES | MASE | 0,827 | 1,120 | 0,810 | −26,1% | **+2,1%** |

- **Contra o naive sazonal.** O linear **não perde** em nenhuma série nem métrica.
- **Contra o SARIMA, em store_total e FOODS.** O linear ganha com folga, reduzindo o MAE em 37% e 45%. O SARIMA com $m = 7$ e sem regressores não enxerga o ciclo mensal; o linear recebe o SNAP, que pesa mais em alimentos, e o nível recente pelos lags.
- **Em HOBBIES.** O linear perde para o SARIMA no MAE e no MASE (+2,1%) e ganha no RMSE (−6,5%), ou seja, erra menos nos dias de erro grande.
- **O que muda em HOBBIES em relação a FOODS.** HOBBIES usa as mesmas colunas, mas com ajuste próprio e o preço da própria categoria. A série tem mais ruído e não tem ciclo mensal (diagnóstico da Task 1), então o SNAP ajuda pouco e o R² no treino é bem menor ($0{,}54$). O ganho do linear aqui vem quase todo do padrão semanal e do nível, que o SARIMA já capturava.
- **Resíduos no treino (`figuras/linear_residuos_*.png`).** Os resíduos têm média zero nas três séries e desvio padrão de $373{,}4$ (`store_total`), $298{,}7$ (`FOODS`) e $83{,}0$ (`HOBBIES`). O teste de Ljung-Box rejeita ruído branco nas defasagens $14$ e $28$ ($p < 10^{-75}$ nas três séries), com ACF no lag $1$ de $\sim 0{,}36$ (`store_total`), $\sim 0{,}45$ (`FOODS`) e $\sim 0{,}24$ (`HOBBIES`). Isso é esperado na **estratégia direta** para $H = 28$: como o modelo só recebe lags $\ge 28$ dias (sem $y_{t-1}, \dots, y_{t-7}$), choques locais de uma semana para a seguinte permanecem correlacionados nos resíduos in-sample. Em compensação, a elevação mensal em torno das defasagens $28\text{–}32$ que aparecia nos resíduos do SARIMA cai bastante, já que o linear incorpora o `snap_CA` e as dummies de mês.

## `boosting`

O código está em `boosting.py`. Usamos *gradient boosting* de árvores baseado em histogramas (`sklearn.ensemble.HistGradientBoostingRegressor`), ajustado separadamente para cada série sobre a mesma matriz de features de `features.py` (estratégia direta, lags $\ge 28$ e padronização com média e desvio exclusivos do treino).

### Decisões e hiperparâmetros

- **Hiperparâmetros fixados a priori:** mantivemos os valores padrão da biblioteca (`learning_rate = 0.1`, `max_iter = 100`, `max_leaf_nodes = 31`, `min_samples_leaf = 20`, perda quadrática), fixando `early_stopping=False` e `random_state=0`. Nenhum hiperparâmetro foi escolhido ou ajustado olhando o erro da validação. Desligar o `early_stopping` evita que o estimador separe internamente uma fração aleatória (embaralhada) do treino para parada antecipada.
- **Ajuste e resíduos no treino (`figuras/boosting_residuos_*.png`):** por capturar interações não lineares entre dia da semana, mês, SNAP, eventos, preço e lags, o ajuste in-sample sobe para $R^2 = 0{,}953$ em `store_total` (desvio dos resíduos de $213{,}7$), $R^2 = 0{,}946$ em `FOODS` (desvio de $162{,}3$) e $R^2 = 0{,}855$ em `HOBBIES` (desvio de $46{,}7$). Na ACF/PACF dos resíduos, a autocorrelação residual cai drasticamente em relação ao modelo linear: o lag $1$ recua para $\sim 0{,}14$ (`store_total`), $\sim 0{,}17$ (`FOODS`) e $\sim 0{,}07$ (`HOBBIES`), e os múltiplos sazonais ($7, 14, 21$) ficam quase todos dentro da banda de confiança (embora o Ljung-Box ainda rejeite ruído branco a $5\%$, com $p(14) = 5{,}4 \times 10^{-11}$ em `store_total`, $1{,}9 \times 10^{-10}$ em `FOODS` e $0{,}0015$ em `HOBBIES`).

### Resultados na validação

As figuras `figuras/boosting_previsao_*.png` mostram a previsão dos $28$ dias ao lado do realizado.

| Série | Métrica | boosting | naive sazonal | SARIMA | linear | vs. naive sazonal | vs. SARIMA | vs. linear |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| store_total | MAE | **195,5** | 459,2 | 316,6 | 200,3 | −57,4% | −38,2% | −2,4% |
| store_total | RMSE | 258,3 | 655,5 | 396,5 | **241,3** | −60,6% | −34,9% | +7,0% |
| store_total | MASE | **0,449** | 1,056 | 0,728 | 0,460 | −57,4% | −38,2% | −2,4% |
| FOODS | MAE | 175,6 | 343,9 | 223,1 | **122,0** | −48,9% | −21,3% | +44,0% |
| FOODS | RMSE | 209,9 | 420,1 | 269,5 | **149,2** | −50,0% | −22,1% | +40,6% |
| FOODS | MASE | 0,513 | 1,005 | 0,652 | **0,357** | −48,9% | −21,3% | +44,0% |
| HOBBIES | MAE | 86,9 | 102,3 | **74,0** | 75,6 | −15,0% | **+17,4%** | +15,0% |
| HOBBIES | RMSE | 105,4 | 146,7 | 95,4 | **89,1** | −28,1% | **+10,6%** | +18,3% |
| HOBBIES | MASE | 0,951 | 1,120 | **0,810** | 0,827 | −15,0% | **+17,4%** | +15,0% |

- **Contra o naive sazonal.** O `boosting` **não perde** em nenhuma série nem métrica (reduz o MAE em $57{,}4\%$ em `store_total`, $48{,}9\%$ em `FOODS` e $15{,}0\%$ em `HOBBIES`).
- **Em `store_total`.** O `boosting` alcança o **menor MAE ($195{,}5$) e MASE ($0{,}449$) entre todos os modelos do trabalho**, superando com folga o SARIMA (−38,2%) e batendo levemente o `linear` (−2,4% em MAE/MASE, mas com RMSE 7,0% maior). Na série agregada da loja, a combinação não linear entre calendário, SNAP e nível recente ajuda a acertar os picos de fim de semana do início do mês.
- **Em `FOODS`.** Supera com folga o naive sazonal (−48,9% no MAE) e o SARIMA (−21,3% no MAE), mas fica atrás do `linear` (+44,0% no MAE). Como a dinâmica de `FOODS` é fortemente aditiva nos efeitos de dia da semana e SNAP, a flexibilidade das $100$ árvores com até $31$ folhas sobreajusta parte das oscilações do treino ($R^2 = 0{,}946$ vs. $0{,}817$ no linear).
- **O que muda em `HOBBIES` em relação a `FOODS`.** Com ajuste próprio e `preco_HOBBIES`, o `boosting` ganha do naive sazonal nas três métricas, mas **perde para o SARIMA** (+17,4% no MAE e +10,6% no RMSE) e para o `linear` (+15,0% no MAE). Em `HOBBIES` a razão sinal-ruído é baixa e não há ciclo mensal de SNAP; ao subir o $R^2$ de treino de $0{,}541$ (linear) para $0{,}855$, a árvore memoriza ruído específico do histórico e generaliza pior nos $28$ dias de validação.

## `rede`

O código está em `rede.py`. Usamos uma rede neural *Perceptron Multicamadas* (`sklearn.neural_network.MLPRegressor`), ajustada separadamente para cada série sobre a mesma matriz de features de `features.py` (estratégia direta, onde tanto o alvo quanto os lags e o preço já estão padronizados com média e desvio exclusivos do treino).

### Decisões e hiperparâmetros

- **Arquitetura e regularização fixadas a priori:** duas camadas ocultas com `hidden_layer_sizes = (32, 16)`, função de ativação `relu`, penalização L2 `alpha = 1e-2`, `max_iter = 500` e `early_stopping = False`. Como a matriz de entrada tem $26$ colunas, a arquitetura em funil $(26 \to 32 \to 16 \to 1)$ tem capacidade moderada e usa regularização L2 para conter pesos grandes sem consultar a validação.
- **Média de $5$ sementes (`SEMENTES = [0, 1, 2, 3, 4]`):** como o treinamento da MLP depende da inicialização aleatória dos pesos e da otimização estocástica (Adam), treinamos $5$ redes independentes com sementes fixadas a priori e tomamos a média das previsões no treino e na validação, reduzindo a variância do preditor.
- **Ajuste e resíduos no treino (`figuras/rede_residuos_*.png`):** o $R^2$ no treino é $0{,}950$ em `store_total` (desvio dos resíduos de $219{,}5$), $0{,}939$ em `FOODS` (desvio de $172{,}4$) e $0{,}858$ em `HOBBIES` (desvio de $46{,}1$). Assim como no linear e no boosting, o Ljung-Box rejeita ruído branco nas defasagens $14$ e $28$ ($p < 10^{-6}$ nas três séries) por conta da autocorrelação de curto prazo no lag $1$ ($\sim 0{,}31$ em `store_total`, $\sim 0{,}37$ em `FOODS` e $\sim 0{,}10$ em `HOBBIES`), inerente ao horizonte direto de $28$ dias sem lags curtos.

### Resultados na validação

As figuras `figuras/rede_previsao_*.png` mostram a previsão da MLP ao lado do realizado.

| Série | Métrica | rede | naive sazonal | SARIMA | linear | boosting | vs. naive sazonal | vs. SARIMA |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| store_total | MAE | 240,7 | 459,2 | 316,6 | 200,3 | **195,5** | −47,6% | −24,0% |
| store_total | RMSE | 297,9 | 655,5 | 396,5 | **241,3** | 258,3 | −54,6% | −24,9% |
| store_total | MASE | 0,553 | 1,056 | 0,728 | 0,460 | **0,449** | −47,6% | −24,0% |
| FOODS | MAE | 172,4 | 343,9 | 223,1 | **122,0** | 175,6 | −49,9% | −22,7% |
| FOODS | RMSE | 216,1 | 420,1 | 269,5 | **149,2** | 209,9 | −48,6% | −19,8% |
| FOODS | MASE | 0,504 | 1,005 | 0,652 | **0,357** | 0,513 | −49,9% | −22,7% |
| HOBBIES | MAE | 88,5 | 102,3 | **74,0** | 75,6 | 86,9 | −13,5% | **+19,6%** |
| HOBBIES | RMSE | 102,7 | 146,7 | 95,4 | **89,1** | 105,4 | −30,0% | **+7,7%** |
| HOBBIES | MASE | 0,969 | 1,120 | **0,810** | 0,827 | 0,951 | −13,5% | **+19,6%** |

- **Contra o naive sazonal.** A `rede` **não perde** em nenhuma série nem métrica (reduz o MAE em $47{,}6\%$ em `store_total`, $49{,}9\%$ em `FOODS` e $13{,}5\%$ em `HOBBIES`).
- **Contra o SARIMA, em `store_total` e `FOODS`.** A `rede` supera o SARIMA com folga nas duas séries (−24,0% e −22,7% no MAE; −24,9% e −19,8% no RMSE), beneficiando-se das covariáveis de calendário e SNAP que o SARIMA univariado não possui. Em `FOODS`, a `rede` também supera ligeiramente o `boosting` no MAE ($172{,}4$ vs. $175{,}6$) e no MASE ($0{,}504$ vs. $0{,}513$), ficando atrás apenas do `linear`.
- **O que muda em `HOBBIES` em relação a `FOODS`.** Em `HOBBIES` (ajustada separadamente com `preco_HOBBIES`), a `rede` supera o naive sazonal nas três métricas, tem RMSE ligeiramente menor que o `boosting` ($102{,}7$ vs. $105{,}4$), mas **perde para o SARIMA** (+19,6% no MAE e +7,7% no RMSE) e para o `linear`. Pelo mesmo motivo observado no `boosting`, a série de `HOBBIES` é mais ruidosa, sofreu quebra estrutural em 2013 e quase não responde a SNAP/eventos; modelos não lineares com maior número de parâmetros ($R^2 = 0{,}858$ no treino) acabam sobreajustando variações idiossincráticas do treino.

## Resumo comparativo da Task 2

| Série | Melhor MAE / MASE | Melhor RMSE | Perdeu para o `naive_sazonal`? | Perdeu para o `SARIMA`? |
|---|---|---|---|---|
| `store_total` | `boosting` ($195{,}5$ / $0{,}449$) | `linear` ($241{,}3$) | Nenhum modelo | Nenhum modelo |
| `FOODS` | `linear` ($122{,}0$ / $0{,}357$) | `linear` ($149{,}2$) | Nenhum modelo | Nenhum modelo |
| `HOBBIES` | `sarima` ($74{,}0$ / $0{,}810$) | `linear` ($89{,}1$) | Nenhum modelo | `linear` (só MAE/MASE, +2,1%), `boosting` e `rede` |

1. **Importância das features exógenas e de calendário:** em `store_total` e `FOODS`, todas as três classes da Task 2 (`linear`, `boosting` e `rede`) superam com grande margem tanto o `naive_sazonal` quanto o `SARIMA`. O ganho vem sobretudo da inclusão de `snap_CA`, dummies de mês/eventos e múltiplos lags sazonais ($\ge 28$), que capturam o ciclo mensal de compras que sobrava nos resíduos do SARIMA.
2. **Parcimônia vs. complexidade (linear vs. não lineares):** com features bem construídas a partir do diagnóstico da Task 1, o modelo `linear` (OLS) teve o melhor desempenho global em `FOODS` e o menor RMSE em todas as três séries. Os modelos não lineares (`boosting` e `rede`) tiveram $R^2$ de treino muito superior ($\sim 0{,}94\text{–}0{,}95$ vs. $0{,}82\text{–}0{,}86$), trouxeram ganho em MAE/MASE no agregado da loja (`store_total` com `boosting`), mas sofreram maior variância fora da amostra nas séries mais ruidosas (`HOBBIES`).
