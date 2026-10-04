from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from config import *
from carregar import limpar_natal
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import adfuller


def marcar_sazonais(ax, lags: int) -> None:
    """Linhas verticais tracejadas nos múltiplos de M (7, 14, ...), com esses valores no eixo x."""
    sazonais = list(range(M, lags + 1, M))
    for k in sazonais:
        ax.axvline(k, color="gray", ls="--", lw=0.8, alpha=0.6, zorder=0)
    ax.set_xticks([0] + sazonais)


def diagnostico(treino: pd.DataFrame) -> None:
    FIG.mkdir(exist_ok=True)
    for s in SERIES:
        y = treino[s]
        fig = plt.figure(figsize=(12, 7))
        ax1 = fig.add_subplot(2, 1, 1)
        ax1.plot(y.index, y.values, lw=0.6)
        ax1.plot(y.index, y.rolling(28, center=True).mean(), lw=1.5, color="C1", label="média móvel 28d")
        zeros = y[y == 0]
        ax1.scatter(zeros.index, zeros.values, color="red", s=15, zorder=3, label="zeros (Natal)")
        ax1.set_title(f"{s} — treino (2011-01-29 a 2016-03-27)")
        ax1.legend(loc="upper left")
        ax_acf = fig.add_subplot(2, 2, 3)
        plot_acf(y, lags=56, ax=ax_acf, title=f"ACF — {s}")
        marcar_sazonais(ax_acf, 56)
        plot_pacf(y, lags=56, ax=fig.add_subplot(2, 2, 4), method="ywm", title=f"PACF — {s}")
        fig.tight_layout()
        fig.savefig(FIG / f"diagnostico_{s}.png", dpi=120)
        plt.close(fig)

        # ACF/PACF depois da diferença sazonal: base da identificação do SARIMA
        d7 = limpar_natal(y).diff(M).dropna()
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        plot_acf(d7, lags=35, ax=axes[0], title=f"ACF de (1-B^7) {s}")
        marcar_sazonais(axes[0], 35)
        plot_pacf(d7, lags=35, ax=axes[1], method="ywm", title=f"PACF de (1-B^7) {s}")
        fig.tight_layout()
        fig.savefig(FIG / f"acf_pacf_dif7_{s}.png", dpi=120)
        plt.close(fig)

        p_nivel = adfuller(limpar_natal(y))[1]
        p_d7 = adfuller(d7)[1]
        print(f"[diag] {s:12s} zeros={int((y == 0).sum())}  ADF p (nível)={p_nivel:.3f}  ADF p (dif. 7)={p_d7:.2e}")
