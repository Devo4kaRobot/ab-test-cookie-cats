"""Групповой последовательный дизайн: двусторонний z-тест с K равномерно расположенными проверками.

Границы вида c_k = C·sqrt(K / k) (O'Brien–Fleming) и c_k = C (Pocock). Константа C подбирается так,
чтобы вероятность хотя бы раз пересечь границу при отсутствии эффекта была ровно alpha.

Калибровка — Монте-Карло: при нулевой гипотезе накопленная z-статистика на k-й проверке равна
S_k / sqrt(k), где S_k — сумма k независимых N(0, 1) (стандартное броуновское движение на долях
информации k/K). Граница записывается как c_k = C·w_k, где w_k = sqrt(K / k) для O'Brien–Fleming
и w_k = 1 для Pocock. Тест отвергает H0, если |Z_k| ≥ C·w_k хотя бы на одной проверке, то есть если
max_k |Z_k| / w_k ≥ C. Значит, C — это (1 − alpha)-квантиль величины max_k |Z_k| / w_k, и подбирать
корень не нужно. По умолчанию берётся 2 млн путей (кусками, чтобы не держать в памяти всю матрицу):
стандартная ошибка C около 0,001–0,002, для K = 5 получается 2,040 (O'Brien–Fleming) и 2,413 (Pocock),
как в таблицах Jennison & Turnbull (2000). Расчёт для K = 44 занимает около двух секунд; результат кэшируется.

Это классические границы O'Brien–Fleming (1979) и Pocock (1977) для заранее заданного числа равных
шагов, а не функции расходования alpha Лана — ДеМетса: если проверки неравномерны, границы надо считать иначе.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from scipy import stats

KINDS = ("obf", "pocock")


def _shape(K: int, kind: str) -> np.ndarray:
    """Форма границы w_k: c_k = C · w_k."""
    k = np.arange(1, K + 1)
    if kind == "obf":
        return np.sqrt(K / k)
    if kind == "pocock":
        return np.ones(K)
    raise ValueError(f"kind должен быть одним из {KINDS}, получено {kind!r}")


@lru_cache(maxsize=64)
def _calibrate(K: int, alpha: float, kind: str, n_sim: int, seed: int) -> float:
    w = _shape(K, kind)
    rng = np.random.default_rng(seed)
    sqrt_k = np.sqrt(np.arange(1, K + 1))
    stat = np.empty(n_sim)
    chunk = max(1, 4_000_000 // K)                     # не держим в памяти всю матрицу n_sim × K
    for start in range(0, n_sim, chunk):
        m = min(chunk, n_sim - start)
        z = np.cumsum(rng.standard_normal((m, K)), axis=1) / sqrt_k
        stat[start:start + m] = np.max(np.abs(z) / w, axis=1)
    return float(np.quantile(stat, 1 - alpha))


def boundaries(K: int, alpha: float = 0.05, kind: str = "obf", n_sim: int = 2_000_000, seed: int = 0) -> dict:
    """Границы для |z| на проверках 1..K.

    Возвращает C, массив границ z (c_k), доли информации k/K и номинальные двусторонние пороги p-value
    на каждой проверке (p_k = 2·(1 − Φ(c_k)): «значимо» на k-й проверке, если p < p_k)."""
    if K < 1:
        raise ValueError("K должно быть не меньше 1")
    if not 0 < alpha < 1:
        raise ValueError("alpha должна быть в интервале (0, 1)")
    if K == 1:                                         # одна проверка — обычный z-тест
        C = float(stats.norm.ppf(1 - alpha / 2))
    else:
        C = _calibrate(int(K), float(alpha), kind, int(n_sim), int(seed))
    z = C * _shape(K, kind)
    return {"kind": kind, "K": K, "alpha": alpha, "C": C, "z": z,
            "info": np.arange(1, K + 1) / K, "p_nominal": 2 * stats.norm.sf(z)}


def first_crossing(z: np.ndarray, bounds: dict | np.ndarray) -> np.ndarray:
    """Номер первой проверки (с 1), на которой |z| вышел за границу, для каждой строки матрицы z
    (симуляции × проверки). 0 — граница так и не пересечена."""
    c = np.asarray(bounds["z"] if isinstance(bounds, dict) else bounds, float)
    z = np.atleast_2d(np.asarray(z, float))
    if z.shape[1] != len(c):
        raise ValueError(f"число проверок в z ({z.shape[1]}) не совпадает с числом границ ({len(c)})")
    cross = np.abs(z) >= c
    return np.where(cross.any(axis=1), cross.argmax(axis=1) + 1, 0)


def sequential_test(z: np.ndarray, bounds: dict | np.ndarray) -> dict:
    """Последовательный тест по накопленным z-статистикам (z[k−1] — после k-й проверки).

    Проверки, которых ещё не было, можно не передавать: тогда stopped=False значит «продолжаем».
    Возвращает, остановлен ли тест, номер проверки (с 1), z и границу в момент остановки,
    направление эффекта (+1 / −1)."""
    c = np.asarray(bounds["z"] if isinstance(bounds, dict) else bounds, float)
    z = np.asarray(z, float).ravel()
    if len(z) > len(c):
        raise ValueError(f"проверок в z ({len(z)}) больше, чем границ ({len(c)})")
    look = int(first_crossing(z[None, :], c[:len(z)])[0])
    if look == 0:
        return {"stopped": False, "look": None, "z": float(z[-1]) if len(z) else float("nan"),
                "boundary": float(c[len(z) - 1]) if len(z) else float("nan"), "direction": 0,
                "final": len(z) == len(c)}
    return {"stopped": True, "look": look, "z": float(z[look - 1]), "boundary": float(c[look - 1]),
            "direction": int(np.sign(z[look - 1])), "final": look == len(c)}
