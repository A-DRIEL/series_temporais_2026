"""Diagnóstico dos resíduos no treino, o mesmo para linear e boosting."""
import matplotlib.pyplot as plt
import pandas as pd
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox

from config import FIG
from diagnostico_series import marcar_sazonais


def diagnosticar_residuos(resid: pd.Series, s: str, modelo: str) -> None:
    """Imprime média, desvio e Ljung-Box e salva a figura dos resíduos in-sample (só treino)."""
    lb = acorr_ljungbox(resid, lags=[14, 28])["lb_pvalue"]
    print(f"[{modelo}] {s:12s} resíduos: média={resid.mean():.1f}  desvio={resid.std():.1f}  "
          f"Ljung-Box p(14)={lb[14]:.2e}  p(28)={lb[28]:.2e}")

    fig = plt.figure(figsize=(12, 7))
    ax1 = fig.add_subplot(2, 1, 1)
    ax1.plot(resid.index, resid.values, lw=0.6)
    ax1.axhline(0, color="gray", lw=0.8)
    ax1.set_title(f"Resíduos no treino — {s} — {modelo}")
    ax_acf = fig.add_subplot(2, 2, 3)
    plot_acf(resid, lags=35, ax=ax_acf, zero=False, auto_ylims=True, title=f"ACF dos resíduos — {s}")
    marcar_sazonais(ax_acf, 35)
    ax_pacf = fig.add_subplot(2, 2, 4)
    plot_pacf(resid, lags=35, ax=ax_pacf, zero=False, auto_ylims=True, method="ywm", title=f"PACF dos resíduos — {s}")
    marcar_sazonais(ax_pacf, 35)
    fig.tight_layout()
    fig.savefig(FIG / f"{modelo}_residuos_{s}.png", dpi=120)
    plt.close(fig)
