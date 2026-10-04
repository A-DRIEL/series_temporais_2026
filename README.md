# Trabalho de Séries Temporais

Para rodar:

```shell
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
