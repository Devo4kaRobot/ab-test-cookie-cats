"""Общий стиль графиков: палитра проверена на различимость при дальтонизме, тонкие линии, приглушённые оси,
десятичная запятая в подписях (как в тексте отчёта)."""
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter, ScalarFormatter

CONTROL, TEST, NEUTRAL, TEXT, MUTED = "#2a78d6", "#eb6834", "#8a8984", "#0b0b0b", "#52514e"
COLORS = {"gate_30": CONTROL, "gate_40": TEST}
LABELS = {"gate_30": "ворота на 30-м уровне (контроль)", "gate_40": "ворота на 40-м уровне (тест)"}
METRIC = {"retention_7": "удержание на 7-й день", "retention_1": "удержание на 1-й день", "sum_gamerounds": "раунды за 14 дней"}


def setup():
    plt.rcParams.update({
        "figure.figsize": (8, 4), "figure.dpi": 110, "font.family": "DejaVu Sans", "font.size": 10,
        "axes.edgecolor": "#c9c8c2", "axes.labelcolor": MUTED, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlecolor": TEXT, "axes.titlelocation": "left", "axes.titlepad": 10,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": "#ecebe6", "grid.linewidth": 0.8, "xtick.color": MUTED, "ytick.color": MUTED,
        "lines.linewidth": 2, "legend.frameon": False, "legend.fontsize": 9, "axes.unicode_minus": True,
    })


def num(x: float, digits: int = 2, sign: bool = False) -> str:
    """Число с десятичной запятой и типографским минусом: num(-0.8201) -> '−0,82'."""
    s = f"{x:+.{digits}f}" if sign else f"{x:.{digits}f}"
    return s.replace("-", "−").replace(".", ",")


class _CommaFormatter(ScalarFormatter):
    def __call__(self, x, pos=None):
        return super().__call__(x, pos).replace(".", ",")


def commas(ax, x: bool = True, y: bool = True):
    """Десятичная запятая на линейных осях (логарифмические и процентные оси не трогаем)."""
    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if on and axis.get_scale() == "linear" and type(axis.get_major_formatter()) is ScalarFormatter:
            f = _CommaFormatter(); f.set_useOffset(False)
            axis.set_major_formatter(f)
    return ax


def percent(ax, axis: str = "y", digits: int = 0):
    """Ось в процентах для долей 0..1, с запятой: 0.425 -> '42,5%'."""
    fmt = FuncFormatter(lambda v, _: num(v * 100, digits) + "%")
    (ax.yaxis if axis == "y" else ax.xaxis).set_major_formatter(fmt)
    return ax


def pct(x: float, digits: int = 1) -> str:
    """Доля → проценты: pct(0.19020) -> '19,0%'."""
    return num(x * 100, digits) + "%"


def pp(x: float, digits: int = 2, sign: bool = True) -> str:
    """Разность долей → процентные пункты: pp(-0.0082) -> '−0,82'."""
    return num(x * 100, digits, sign)


def thousands(n: float, sign: bool = False) -> str:
    """Целое с неразрывным пробелом между разрядами: 44700 -> '44 700', со знаком: '−1 329'."""
    s = f"{n:+,.0f}" if sign else f"{n:,.0f}"
    return s.replace(",", " ").replace("-", "−")


def pval(p: float) -> str:
    """p-value для текста: 0,0016 / 0,074 / 0,0502 / < 0,0001."""
    if p < 1e-4:
        return "< 0,0001"
    return num(p, 4 if p < 0.01 or 0.04 <= p < 0.06 else 3)  # 0,0502 не должно выглядеть как 0,050
