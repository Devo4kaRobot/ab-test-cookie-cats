"""Статистика для A/B-тестов на numpy/scipy — без statsmodels, чтобы все формулы были на виду.

Все функции принимают «сырые» числа или массивы и возвращают словари/кортежи, удобные для отчёта.
"""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np
from scipy import stats

# ---------------------------------------------------------------- корректность сплита

def srm_test(counts: Sequence[int], expected_share: Sequence[float] | None = None) -> dict:
    """Sample Ratio Mismatch: хи-квадрат «фактические размеры групп против ожидаемых».
    По умолчанию ожидается равное деление. Для SRM обычно берут порог p < 0.001."""
    counts = np.asarray(counts, dtype=float)
    share = np.full(len(counts), 1 / len(counts)) if expected_share is None else np.asarray(expected_share, float)
    expected = counts.sum() * share / share.sum()
    chi2, p = stats.chisquare(counts, expected)
    return {"chi2": float(chi2), "p_value": float(p), "observed_share": (counts / counts.sum()).tolist()}


# ---------------------------------------------------------------- доли (retention)

def two_proportion_test(x_a: int, n_a: int, x_b: int, n_b: int, alpha: float = 0.05) -> dict:
    """Z-тест разности долей (pooled SE для p-value) и ДИ разности по Вальду (unpooled SE).
    Разность считается как B − A."""
    p_a, p_b = x_a / n_a, x_b / n_b
    diff = p_b - p_a
    p_pool = (x_a + x_b) / (n_a + n_b)
    se_pool = math.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))
    z = diff / se_pool
    p_value = 2 * stats.norm.sf(abs(z))
    se_unpooled = math.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    q = stats.norm.ppf(1 - alpha / 2)
    return {"p_a": p_a, "p_b": p_b, "diff": diff, "rel_lift": diff / p_a if p_a else float("nan"),
            "z": z, "p_value": float(p_value), "ci": (diff - q * se_unpooled, diff + q * se_unpooled)}


def bootstrap_proportion_diff(x_a: int, n_a: int, x_b: int, n_b: int, n_boot: int = 10_000,
                              alpha: float = 0.05, seed: int = 0) -> dict:
    """Bootstrap разности долей. Ресэмплинг бинарной выборки размера n с долей p эквивалентен
    Binomial(n, p)/n — поэтому без перебора миллионов строк."""
    rng = np.random.default_rng(seed)
    boot_a = rng.binomial(n_a, x_a / n_a, n_boot) / n_a
    boot_b = rng.binomial(n_b, x_b / n_b, n_boot) / n_b
    d = boot_b - boot_a
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": x_b / n_b - x_a / n_a, "ci": (float(lo), float(hi)),
            "p_b_less": float((d < 0).mean()), "samples": d}


# ---------------------------------------------------------------- непрерывные метрики (раунды)

def bootstrap_stat_diff(a: np.ndarray, b: np.ndarray, stat: Callable = np.mean, n_boot: int = 2000,
                        alpha: float = 0.05, seed: int = 0, chunk: int = 200) -> dict:
    """Bootstrap разности статистики B − A. Считаем кусками, чтобы не держать в памяти
    матрицу n_boot × n (45 тыс. × 2000 int64 = 720 МБ)."""
    rng = np.random.default_rng(seed)
    a, b = np.asarray(a), np.asarray(b)
    out = []
    for start in range(0, n_boot, chunk):
        k = min(chunk, n_boot - start)
        ia = rng.integers(0, len(a), (k, len(a)))
        ib = rng.integers(0, len(b), (k, len(b)))
        out.append(stat(b[ib], axis=1) - stat(a[ia], axis=1))
    d = np.concatenate(out)
    lo, hi = np.percentile(d, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"diff": float(stat(b) - stat(a)), "ci": (float(lo), float(hi)), "samples": d}


def mann_whitney(a: np.ndarray, b: np.ndarray) -> dict:
    """Манн-Уитни. Проверяет не «равенство медиан», а P(X_B > X_A) + 0.5·P(X_B = X_A) = 0.5.
    prob_superiority — эта вероятность (common language effect size)."""
    res = stats.mannwhitneyu(b, a, alternative="two-sided")
    return {"U": float(res.statistic), "p_value": float(res.pvalue),
            "prob_superiority": float(res.statistic / (len(a) * len(b)))}


def welch_log_ttest(a: np.ndarray, b: np.ndarray) -> dict:
    """t-тест Уэлча на log(1 + x) — для метрик с тяжёлым хвостом.
    ratio_geo = exp(mean log(1+x_B) − mean log(1+x_A)) — отношение геометрических средних величины 1 + x:
    во сколько раз «типичное» значение в B отличается от A. ДИ — тот же t-интервал, переведённый из логарифмов."""
    la, lb = np.log1p(np.asarray(a, float)), np.log1p(np.asarray(b, float))
    res = stats.ttest_ind(lb, la, equal_var=False)
    ci = res.confidence_interval()
    return {"t": float(res.statistic), "p_value": float(res.pvalue), "ratio_geo": float(np.exp(lb.mean() - la.mean())),
            "ratio_ci": (float(np.exp(ci.low)), float(np.exp(ci.high)))}


# ---------------------------------------------------------------- множественные сравнения

def holm(p_values: Sequence[float], alpha: float = 0.05) -> list[dict]:
    """Поправка Холма–Бонферрони: контролирует FWER, мощнее Бонферрони. Возвращает скорректированные p."""
    p = np.asarray(p_values, float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return [{"p": float(p[i]), "p_holm": float(adj[i]), "reject": bool(adj[i] < alpha)} for i in range(m)]


# ---------------------------------------------------------------- мощность и размер выборки

def _z(alpha: float, power: float) -> tuple[float, float]:
    return stats.norm.ppf(1 - alpha / 2), stats.norm.ppf(power)


def mde_proportion(p_base: float, n_a: int, n_b: int | None = None, alpha: float = 0.05, power: float = 0.8) -> float:
    """Минимальный детектируемый эффект (абсолютный, в долях) для разности долей при заданных n."""
    n_b = n_a if n_b is None else n_b
    za, zb = _z(alpha, power)
    return (za + zb) * math.sqrt(p_base * (1 - p_base) * (1 / n_a + 1 / n_b))


def sample_size_proportion(p_base: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8,
                           ratio: float = 1.0) -> tuple[int, int]:
    """Размер групп (A, B) для обнаружения абсолютного эффекта mde_abs. ratio = n_B / n_A.
    Дисперсия считается по альтернативе: p_A(1-p_A) и p_B(1-p_B)."""
    za, zb = _z(alpha, power)
    p_b = p_base + mde_abs
    var = p_base * (1 - p_base) + p_b * (1 - p_b) / ratio
    n_a = math.ceil((za + zb) ** 2 * var / mde_abs ** 2)
    return n_a, math.ceil(n_a * ratio)


def power_proportion(p_base: float, mde_abs: float, n_a: int, n_b: int | None = None, alpha: float = 0.05) -> float:
    n_b = n_a if n_b is None else n_b
    p_b = p_base + mde_abs
    se = math.sqrt(p_base * (1 - p_base) / n_a + p_b * (1 - p_b) / n_b)
    za = stats.norm.ppf(1 - alpha / 2)
    return float(stats.norm.sf(za - abs(mde_abs) / se) + stats.norm.cdf(-za - abs(mde_abs) / se))


def sample_size_mean(sd: float, mde_abs: float, alpha: float = 0.05, power: float = 0.8) -> int:
    za, zb = _z(alpha, power)
    return math.ceil(2 * (za + zb) ** 2 * sd ** 2 / mde_abs ** 2)


# ---------------------------------------------------------------- CUPED

def cuped(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, float]:
    """CUPED: y' = y − θ(x − mean x), θ = cov(x, y) / var(x). x — метрика ДО эксперимента.
    Дисперсия падает примерно в (1 − ρ²) раз, среднее не меняется."""
    theta = float(np.cov(x, y, ddof=1)[0, 1] / np.var(x, ddof=1))
    return y - theta * (x - x.mean()), theta


# ---------------------------------------------------------------- поправка на дисбаланс (пост-стратификация)

def poststratified_diff(y_a: np.ndarray, s_a: np.ndarray, y_b: np.ndarray, s_b: np.ndarray, alpha: float = 0.05) -> dict:
    """Разность средних B − A с пост-стратификацией по ковариате, на которую воздействие не влияет.

    В каждом страте s считается разность средних, затем страты взвешиваются по общей доле
    w_s = (n_As + n_Bs) / (n_A + n_B). Дисперсия: Σ w_s² (var_As / n_As + var_Bs / n_Bs).
    Если в одной группе стратов больше по случайности (или из-за кривого сплита), оценка это убирает;
    при честной рандомизации она несмещена и обычно точнее простой разности.
    Страты, где нет хотя бы двух наблюдений в каждой группе, не допускаются.
    """
    y_a, s_a, y_b, s_b = (np.asarray(v) for v in (y_a, s_a, y_b, s_b))
    strata = np.union1d(np.unique(s_a), np.unique(s_b))
    n = len(y_a) + len(y_b)
    est, var, rows = 0.0, 0.0, []
    for s in strata:
        ya, yb = y_a[s_a == s].astype(float), y_b[s_b == s].astype(float)
        if len(ya) < 2 or len(yb) < 2:
            raise ValueError(f"в страте {s!r} меньше двух наблюдений в одной из групп")
        w = (len(ya) + len(yb)) / n
        d = yb.mean() - ya.mean()
        est += w * d
        var += w ** 2 * (ya.var(ddof=1) / len(ya) + yb.var(ddof=1) / len(yb))
        rows.append({"stratum": s, "weight": w, "n_a": len(ya), "n_b": len(yb), "diff": d})
    se = math.sqrt(var)
    z = est / se
    q = stats.norm.ppf(1 - alpha / 2)
    return {"diff": est, "se": se, "z": z, "p_value": float(2 * stats.norm.sf(abs(z))),
            "ci": (est - q * se, est + q * se), "strata": rows}
