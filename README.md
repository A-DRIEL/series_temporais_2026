# Trabalho de Séries Temporais

Para rodar:

```shell
python -m pip install -r requirements.txt
python main.py
```


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
